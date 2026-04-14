"""Native notebook integration for the CT unlearning algorithm."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .common import build_epoch_efficiency_variants, require_torch, resolve_unlearning_profile, run_unlearning_workflow_bank, tqdm

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover - depends on local environment.
    torch = None
    nn = None
    optim = None
    DataLoader = None


def _no_grad() -> Any:
    if torch is None:  # pragma: no cover - depends on local environment.
        def decorator(func: Any) -> Any:
            return func
        return decorator
    return torch.no_grad()


CT_UNLEARNING_PROFILES: dict[str, dict[str, Any]] = {
    "cifar10": {
        "epochs": 3,
        "lr": 1e-3,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "batch_size": 64,
    },
    "mufac": {
        "epochs": 3,
        "lr": 5e-4,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "batch_size": 32,
    },
}


def _resolve_ct_profile(dataset: str, profile: str | None) -> dict[str, Any]:
    """Resolve a named CT profile, defaulting to the dataset name."""

    _profile_name, profile_config = resolve_unlearning_profile(
        dataset=dataset,
        profile=profile,
        profiles=CT_UNLEARNING_PROFILES,
        algorithm_name="CT",
    )
    return profile_config


def _normalize_ct_hyperparameters(profile_config: dict[str, Any]) -> dict[str, Any]:
    """Normalize CT hyperparameters into a stable JSON-serializable shape."""

    return {
        "epochs": int(profile_config["epochs"]),
        "lr": float(profile_config["lr"]),
        "momentum": float(profile_config["momentum"]),
        "weight_decay": float(profile_config["weight_decay"]),
        "batch_size": int(profile_config["batch_size"]),
    }


def _build_ct_efficiency_variants(
    profile_name: str,
    profile_config: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Build a small quality-to-speed variant ladder for CT."""

    return build_epoch_efficiency_variants(
        profile_name=profile_name,
        profile_config=_normalize_ct_hyperparameters(profile_config),
        epoch_candidates=[int(profile_config["epochs"]), 2, 1],
        min_epochs=1,
    )


@_no_grad()
def _apply_ct_conv_transpose(model: Any) -> None:
    """Transpose convolution kernels exactly as in the source implementation."""

    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            module.weight = nn.Parameter(module.weight.swapaxes(2, 3))


def _run_ct_unlearning_seed(
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
    """Run CT for one seed checkpoint."""

    require_torch()
    context = data_bundle.context
    device = deps["choose_device"](device_name)
    source_metadata = json.loads(Path(checkpoint_path).with_suffix(".json").read_text(encoding="utf-8"))
    seed = int(source_metadata["seed"])
    output_dir = Path(checkpoint_dir) / context.dataset / context.task_id / output_family_name
    checkpoint_stem = f"seed_{seed}"
    output_checkpoint_path = output_dir / f"{checkpoint_stem}.pth"
    output_metadata_path = output_dir / f"{checkpoint_stem}.json"
    algorithm_hyperparameters = _normalize_ct_hyperparameters(profile_config)
    if reuse_existing and output_checkpoint_path.exists() and output_metadata_path.exists():
        existing_metadata = json.loads(output_metadata_path.read_text(encoding="utf-8"))
        if (
            existing_metadata.get("source_checkpoint") == str(checkpoint_path)
            and existing_metadata.get("unlearning_algorithm") == "CT"
            and existing_metadata.get("algorithm_source_alias") == "KGLTop5"
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
    weight_decay = float(algorithm_hyperparameters["weight_decay"])
    batch_size = int(algorithm_hyperparameters["batch_size"])

    deps["set_random_seed"](seed)
    model = deps["build_model"](deps["create_resnet18"], num_classes=context.num_classes, dataset=dataset).to(device)
    deps["load_model_checkpoint"](model, checkpoint_path, device)
    _apply_ct_conv_transpose(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    retain_loader = DataLoader(data_bundle.loaders["retain"].dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = data_bundle.loaders["val"]
    wandb_run = deps["init_wandb_run"](
        enabled=use_wandb,
        entity="inmdev-university-of-british-columbia",
        project=deps["resolve_wandb_project"](dataset, wandb_project),
        run_name=f"{output_family_name}_seed_{seed}",
        config={
            "dataset": dataset,
            "task_id": context.task_id,
            "train_split": "retain",
            "seed": seed,
            "num_classes": context.num_classes,
            "image_size": image_size,
            "algorithm": "CT",
            "algorithm_source_alias": "KGLTop5",
            "algorithm_profile": profile_name,
            "algorithm_hyperparameters": algorithm_hyperparameters,
            "source_checkpoint": str(checkpoint_path),
            "output_family_name": output_family_name,
        },
    )

    model.train()
    epoch_history: list[dict[str, Any]] = []
    best_val_accuracy = 0.0
    final_val_accuracy = 0.0
    wall_clock_start = time.perf_counter()
    training_runtime_seconds = 0.0
    epoch_iterator = tqdm(
        range(1, epochs + 1),
        desc=f"CT epochs ({output_family_name}, seed {seed})",
        leave=False,
    )
    for epoch in epoch_iterator:
        epoch_training_start = time.perf_counter()
        retain_loss_sum = 0.0
        retain_sample_count = 0
        for inputs, targets in retain_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            batch_items = targets.size(0)
            retain_loss_sum += float(loss.item()) * batch_items
            retain_sample_count += batch_items
        scheduler.step()
        training_runtime_seconds += time.perf_counter() - epoch_training_start
        current_lr = float(optimizer.param_groups[0]["lr"])
        val_accuracy = deps["compute_accuracy"](model, val_loader, device)
        best_val_accuracy = max(best_val_accuracy, float(val_accuracy))
        final_val_accuracy = float(val_accuracy)
        epoch_metrics = {
            "epoch": float(epoch),
            "retain_loss": 0.0 if retain_sample_count == 0 else retain_loss_sum / retain_sample_count,
            "val_accuracy": float(val_accuracy),
            "stage": "retain_finetune",
            "learning_rate": current_lr,
        }
        epoch_history.append(epoch_metrics)
        wandb_run.log(epoch_metrics)
        if hasattr(epoch_iterator, "set_postfix"):
            epoch_iterator.set_postfix(
                {
                    "retain": f"{epoch_metrics['retain_loss']:.4f}",
                    "val_acc": f"{epoch_metrics['val_accuracy']:.4f}",
                }
            )

    wall_clock_seconds = time.perf_counter() - wall_clock_start
    runtime_seconds = training_runtime_seconds
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_checkpoint_path)
    output_metadata = {
        "dataset": context.dataset,
        "task_id": context.task_id,
        "run_name": output_family_name,
        "train_split": "retain",
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "weight_decay": weight_decay,
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
        "unlearning_algorithm": "CT",
        "algorithm_source_alias": "KGLTop5",
        "algorithm_profile": profile_name,
        "algorithm_hyperparameters": algorithm_hyperparameters,
        "reused_existing": False,
    }
    output_metadata_path.write_text(json.dumps(output_metadata, indent=2), encoding="utf-8")
    wandb_run.finish()
    return output_metadata


def run_ct_unlearning_workflow(
    *,
    deps: dict[str, Any],
    dataset: str,
    base_family_dir: str | Path,
    output_family_name: str = "CT",
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
    """Run CT over a checkpoint bank."""

    profile_name = dataset if profile is None else profile
    profile_config = _resolve_ct_profile(dataset, profile_name)
    return run_unlearning_workflow_bank(
        deps=deps,
        algorithm_name="CT",
        dataset=dataset,
        base_family_dir=base_family_dir,
        output_family_name=output_family_name,
        num_bank_seeds=num_bank_seeds,
        profile_name=profile_name,
        profile_config=profile_config,
        normalize_hyperparameters=_normalize_ct_hyperparameters,
        build_efficiency_variants=_build_ct_efficiency_variants,
        resolve_bundle_batch_size=lambda candidate_profile_config: int(candidate_profile_config["batch_size"]),
        seed_runner=_run_ct_unlearning_seed,
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
