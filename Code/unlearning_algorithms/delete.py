"""Native notebook integration for the DELETE unlearning algorithm."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .common import build_epoch_efficiency_variants, build_shuffled_loader, require_torch, resolve_unlearning_profile, run_unlearning_workflow_bank, tqdm

try:
    import torch
except ImportError:  # pragma: no cover - depends on local environment.
    torch = None


DELETE_UNLEARNING_PROFILES: dict[str, dict[str, Any]] = {
    "cifar10": {
        "epochs": 8,
        "lr": 1e-3,
        "momentum": 0.9,
        "batch_size": 64,
        "soft_label": "inf",
        "disable_bn": False,
    },
    "mufac": {
        "epochs": 6,
        "lr": 5e-4,
        "momentum": 0.9,
        "batch_size": 32,
        "soft_label": "inf",
        "disable_bn": False,
    },
}


def _resolve_delete_profile(dataset: str, profile: str | None) -> dict[str, Any]:
    """Resolve a named DELETE profile, defaulting to the dataset name."""

    _profile_name, profile_config = resolve_unlearning_profile(
        dataset=dataset,
        profile=profile,
        profiles=DELETE_UNLEARNING_PROFILES,
        algorithm_name="DELETE",
    )
    return profile_config


def _normalize_delete_hyperparameters(profile_config: dict[str, Any]) -> dict[str, Any]:
    """Normalize DELETE hyperparameters into a stable JSON-serializable shape."""

    return {
        "epochs": int(profile_config["epochs"]),
        "lr": float(profile_config["lr"]),
        "momentum": float(profile_config["momentum"]),
        "batch_size": int(profile_config["batch_size"]),
        "soft_label": str(profile_config["soft_label"]),
        "disable_bn": bool(profile_config["disable_bn"]),
    }


def _build_delete_efficiency_variants(
    profile_name: str,
    profile_config: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Build a small quality-to-speed variant ladder for DELETE."""

    return build_epoch_efficiency_variants(
        profile_name=profile_name,
        profile_config=_normalize_delete_hyperparameters(profile_config),
        epoch_candidates=[int(profile_config["epochs"]), 6, 4, 3, 2, 1],
        min_epochs=1,
    )


def _delete_build_masked_teacher_probs(
    teacher_logits: Any,
    targets: Any,
    *,
    soft_label: str,
) -> Any:
    """Build DELETE soft targets from the teacher logits."""

    if soft_label != "inf":
        raise ValueError(f"Unsupported DELETE soft label method '{soft_label}'.")
    masked_logits = teacher_logits.detach().clone()
    # DELETE removes the ground-truth class from the teacher target so the
    # student is pushed toward the teacher's remaining mass on forget examples.
    masked_logits[torch.arange(targets.size(0), device=targets.device), targets] = -1e10
    return torch.nn.functional.softmax(masked_logits, dim=1)


def _run_delete_forget_epoch(
    *,
    student_model: Any,
    teacher_model: Any,
    loader: Any,
    optimizer: Any,
    criterion: Any,
    device: Any,
    soft_label: str,
    disable_bn: bool,
) -> float:
    """Run one DELETE forget-only epoch and return mean KL loss."""

    student_model.train()
    teacher_model.eval()
    loss_sum = 0.0
    sample_count = 0
    progress = tqdm(loader, desc="DELETE forget", leave=False)
    for inputs, targets in progress:
        inputs = inputs.to(device)
        targets = targets.to(device)
        if disable_bn:
            for module in student_model.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()
        optimizer.zero_grad()
        with torch.no_grad():
            teacher_logits = teacher_model(inputs)
        target_probs = _delete_build_masked_teacher_probs(
            teacher_logits,
            targets,
            soft_label=soft_label,
        )
        # DELETE trains only on the forget split: match the masked teacher
        # distribution so the deleted label loses its dominance.
        student_log_probs = torch.nn.functional.log_softmax(student_model(inputs), dim=1)
        loss = criterion(student_log_probs, target_probs)
        loss.backward()
        optimizer.step()
        batch_size = targets.size(0)
        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size
    return 0.0 if sample_count == 0 else loss_sum / sample_count


def _run_delete_unlearning_seed(
    *,
    deps: dict[str, Any],
    dataset: str,
    checkpoint_path: str | Path,
    output_family_name: str,
    profile_name: str,
    profile_config: dict[str, Any],
    data_bundle: Any,
    checkpoint_dir: str | Path,
    device_name: str,
    image_size: int,
    use_wandb: bool,
    wandb_project: str | None,
    reuse_existing: bool = True,
) -> dict[str, Any]:
    """Run the native DELETE method for one seed checkpoint."""

    require_torch()
    context = data_bundle.context
    device = deps["choose_device"](device_name)
    source_metadata = json.loads(Path(checkpoint_path).with_suffix(".json").read_text(encoding="utf-8"))
    seed = int(source_metadata["seed"])
    output_dir = Path(checkpoint_dir) / context.dataset / context.task_id / output_family_name
    checkpoint_stem = f"seed_{seed}"
    output_checkpoint_path = output_dir / f"{checkpoint_stem}.pth"
    output_metadata_path = output_dir / f"{checkpoint_stem}.json"
    algorithm_hyperparameters = _normalize_delete_hyperparameters(profile_config)
    if reuse_existing and output_checkpoint_path.exists() and output_metadata_path.exists():
        existing_metadata = json.loads(output_metadata_path.read_text(encoding="utf-8"))
        if (
            existing_metadata.get("source_checkpoint") == str(checkpoint_path)
            and existing_metadata.get("unlearning_algorithm") == "DELETE"
            and existing_metadata.get("algorithm_profile") == profile_name
            and existing_metadata.get("algorithm_hyperparameters") == algorithm_hyperparameters
            and existing_metadata.get("runtime_excludes_validation") is True
        ):
            reused_metadata = dict(existing_metadata)
            reused_metadata["reused_existing"] = True
            return reused_metadata

    epochs = int(algorithm_hyperparameters["epochs"])
    learning_rate = float(algorithm_hyperparameters["lr"])
    momentum = float(algorithm_hyperparameters["momentum"])
    batch_size = int(algorithm_hyperparameters["batch_size"])
    soft_label = str(algorithm_hyperparameters["soft_label"])
    disable_bn = bool(algorithm_hyperparameters["disable_bn"])

    deps["set_random_seed"](seed)
    teacher_model = deps["build_model"](deps["create_resnet18"], num_classes=context.num_classes, dataset=dataset).to(device)
    student_model = deps["build_model"](deps["create_resnet18"], num_classes=context.num_classes, dataset=dataset).to(device)
    deps["load_model_checkpoint"](teacher_model, checkpoint_path, device)
    deps["load_model_checkpoint"](student_model, checkpoint_path, device)
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)

    forget_loader = build_shuffled_loader(data_bundle.loaders["forget"].dataset, batch_size=batch_size)
    val_loader = data_bundle.loaders["val"]
    criterion = torch.nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.SGD(student_model.parameters(), lr=learning_rate, momentum=momentum)
    wandb_run = deps["init_wandb_run"](
        enabled=use_wandb,
        entity="inmdev-university-of-british-columbia",
        project=deps["resolve_wandb_project"](dataset, wandb_project),
        run_name=f"{output_family_name}_seed_{seed}",
        config={
            "dataset": dataset,
            "task_id": context.task_id,
            "train_split": "forget",
            "seed": seed,
            "num_classes": context.num_classes,
            "image_size": image_size,
            "algorithm": "DELETE",
            "algorithm_profile": profile_name,
            "algorithm_hyperparameters": algorithm_hyperparameters,
            "source_checkpoint": str(checkpoint_path),
            "output_family_name": output_family_name,
        },
    )

    epoch_history: list[dict[str, Any]] = []
    best_val_accuracy = 0.0
    final_val_accuracy = 0.0
    wall_clock_start = time.perf_counter()
    training_runtime_seconds = 0.0
    epoch_iterator = tqdm(
        range(1, epochs + 1),
        desc=f"DELETE epochs ({output_family_name}, seed {seed})",
        leave=False,
    )
    for epoch in epoch_iterator:
        epoch_training_start = time.perf_counter()
        forget_loss = _run_delete_forget_epoch(
            student_model=student_model,
            teacher_model=teacher_model,
            loader=forget_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            soft_label=soft_label,
            disable_bn=disable_bn,
        )
        training_runtime_seconds += time.perf_counter() - epoch_training_start
        val_accuracy = deps["compute_accuracy"](student_model, val_loader, device)
        best_val_accuracy = max(best_val_accuracy, float(val_accuracy))
        final_val_accuracy = float(val_accuracy)
        epoch_metrics = {
            "epoch": float(epoch),
            "forget_loss": float(forget_loss),
            "val_accuracy": float(val_accuracy),
            "stage": "forget_only",
            "learning_rate": learning_rate,
        }
        epoch_history.append(epoch_metrics)
        wandb_run.log(epoch_metrics)
        if hasattr(epoch_iterator, "set_postfix"):
            epoch_iterator.set_postfix(
                {
                    "forget": f"{forget_loss:.4f}",
                    "val_acc": f"{float(val_accuracy):.4f}",
                }
            )

    wall_clock_seconds = time.perf_counter() - wall_clock_start
    runtime_seconds = training_runtime_seconds
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(student_model.state_dict(), output_checkpoint_path)
    output_metadata = {
        "dataset": context.dataset,
        "task_id": context.task_id,
        "run_name": output_family_name,
        "train_split": "forget",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "weight_decay": 0.0,
        "num_classes": context.num_classes,
        "label_to_index": context.label_to_index,
        "class_names": context.class_names,
        "runtime_seconds": runtime_seconds,
        "wall_clock_seconds": wall_clock_seconds,
        "runtime_excludes_validation": True,
        "best_val_accuracy": best_val_accuracy,
        "final_val_accuracy": final_val_accuracy,
        "checkpoint_path": str(output_checkpoint_path),
        "epochs_logged": epoch_history,
        "image_size": image_size,
        "source_checkpoint": str(checkpoint_path),
        "unlearning_algorithm": "DELETE",
        "algorithm_profile": profile_name,
        "algorithm_hyperparameters": algorithm_hyperparameters,
        "soft_label": soft_label,
        "disable_bn": disable_bn,
        "reused_existing": False,
    }
    output_metadata_path.write_text(json.dumps(output_metadata, indent=2), encoding="utf-8")
    wandb_run.finish()
    return output_metadata


def run_delete_unlearning_workflow(
    *,
    deps: dict[str, Any],
    dataset: str,
    base_family_dir: str | Path,
    output_family_name: str = "DELETE",
    num_bank_seeds: int = 3,
    profile: str | None = None,
    checkpoint_dir: str | Path = "checkpoints",
    data_root: str | Path | None = None,
    task_manifest: str | Path | None = None,
    samples_csv: str | Path | None = None,
    device_name: str = "auto",
    use_wandb: bool = False,
    wandb_project: str | None = None,
    image_size: int | None = None,
    reuse_existing: bool = True,
    efficiency_aware: bool = False,
    reference_family_dir: str | Path | None = None,
    efficiency_ratio: float = 0.2,
) -> dict[str, Any]:
    """Run the native DELETE method over a checkpoint bank."""

    profile_name = dataset if profile is None else profile
    profile_config = _resolve_delete_profile(dataset, profile_name)
    return run_unlearning_workflow_bank(
        deps=deps,
        algorithm_name="DELETE",
        dataset=dataset,
        base_family_dir=base_family_dir,
        output_family_name=output_family_name,
        num_bank_seeds=num_bank_seeds,
        profile_name=profile_name,
        profile_config=profile_config,
        normalize_hyperparameters=_normalize_delete_hyperparameters,
        build_efficiency_variants=_build_delete_efficiency_variants,
        resolve_bundle_batch_size=lambda candidate_profile_config: int(candidate_profile_config["batch_size"]),
        seed_runner=_run_delete_unlearning_seed,
        checkpoint_dir=checkpoint_dir,
        data_root=data_root,
        task_manifest=task_manifest,
        samples_csv=samples_csv,
        device_name=device_name,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        image_size=image_size,
        reuse_existing=reuse_existing,
        efficiency_aware=efficiency_aware,
        reference_family_dir=reference_family_dir,
        efficiency_ratio=efficiency_ratio,
    )
