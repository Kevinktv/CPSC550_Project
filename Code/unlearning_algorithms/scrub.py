"""Native notebook integration for the SCRUB unlearning algorithm."""

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


SCRUB_UNLEARNING_PROFILES: dict[str, dict[str, Any]] = {
    "cifar10": {
        "forget_batch_size": 64,
        "retain_batch_size": 32,
        "epochs": 10,
        "msteps": 3,
        "optimizer": "adam",
        "lr": 5e-4,
        "weight_decay": 0.1,
        "lr_decay_epochs": [5, 8, 9],
        "lr_decay_rate": 0.1,
        "alpha": 0.5,
        "gamma": 1.0,
        "kd_temperature": 2.0,
    },
    "mufac": {
        "forget_batch_size": 16,
        "retain_batch_size": 16,
        "epochs": 10,
        "msteps": 3,
        "optimizer": "adam",
        "lr": 2e-4,
        "weight_decay": 0.05,
        "lr_decay_epochs": [5, 8, 9],
        "lr_decay_rate": 0.1,
        "alpha": 0.5,
        "gamma": 1.0,
        "kd_temperature": 2.0,
    },
}


def _resolve_scrub_profile(dataset: str, profile: str | None) -> dict[str, Any]:
    """Resolve a named SCRUB profile, defaulting to the dataset name."""

    _profile_name, profile_config = resolve_unlearning_profile(
        dataset=dataset,
        profile=profile,
        profiles=SCRUB_UNLEARNING_PROFILES,
        algorithm_name="SCRUB",
    )
    return profile_config


def _normalize_scrub_hyperparameters(profile_config: dict[str, Any]) -> dict[str, Any]:
    """Normalize SCRUB hyperparameters into a stable JSON-serializable shape."""

    return {
        "forget_batch_size": int(profile_config["forget_batch_size"]),
        "retain_batch_size": int(profile_config["retain_batch_size"]),
        "epochs": int(profile_config["epochs"]),
        "msteps": int(profile_config["msteps"]),
        "optimizer": str(profile_config["optimizer"]),
        "lr": float(profile_config["lr"]),
        "weight_decay": float(profile_config["weight_decay"]),
        "lr_decay_epochs": [int(value) for value in profile_config["lr_decay_epochs"]],
        "lr_decay_rate": float(profile_config["lr_decay_rate"]),
        "alpha": float(profile_config["alpha"]),
        "gamma": float(profile_config["gamma"]),
        "kd_temperature": float(profile_config["kd_temperature"]),
    }


def _build_scrub_efficiency_variants(
    profile_name: str,
    profile_config: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Build a small quality-to-speed variant ladder for SCRUB."""

    return build_epoch_efficiency_variants(
        profile_name=profile_name,
        profile_config=_normalize_scrub_hyperparameters(profile_config),
        epoch_candidates=[int(profile_config["epochs"]), 8, 6, 4, 3, 2],
        min_epochs=2,
        postprocess_variant=lambda variant_config: {
            **variant_config,
            "msteps": min(int(variant_config["msteps"]), int(variant_config["epochs"]) - 1),
        },
    )


def _scrub_temperature_scaled_kl(student_logits: Any, teacher_logits: Any, *, temperature: float) -> Any:
    """Compute the temperature-scaled teacher-to-student KL divergence."""

    scaled_student = student_logits / float(temperature)
    scaled_teacher = teacher_logits / float(temperature)
    return torch.nn.functional.kl_div(
        torch.nn.functional.log_softmax(scaled_student, dim=1),
        torch.nn.functional.softmax(scaled_teacher, dim=1),
        reduction="batchmean",
    ) * (float(temperature) ** 2)


def _build_scrub_optimizer(
    model: Any,
    *,
    optimizer_name: str,
    lr: float,
    weight_decay: float,
    momentum: float = 0.9,
) -> Any:
    """Build the optimizer for the native SCRUB workflow."""

    resolved_name = optimizer_name.lower()
    if resolved_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if resolved_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    if resolved_name in {"rmsp", "rmsprop"}:
        return torch.optim.RMSprop(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    raise ValueError(f"Unsupported SCRUB optimizer '{optimizer_name}'.")


def _apply_scrub_milestone_lr_decay(
    optimizer: Any,
    *,
    base_lr: float,
    epoch: int,
    lr_decay_epochs: list[int],
    lr_decay_rate: float,
) -> float:
    """Apply milestone-based learning-rate decay and return the current LR."""

    decay_steps = sum(epoch > int(milestone) for milestone in lr_decay_epochs)
    current_lr = float(base_lr) * (float(lr_decay_rate) ** decay_steps)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr
    return current_lr


def _run_scrub_phase(
    *,
    phase: str,
    student_model: Any,
    teacher_model: Any,
    loader: Any,
    optimizer: Any,
    retain_criterion: Any,
    device: Any,
    temperature: float,
    alpha: float,
    gamma: float,
) -> dict[str, float]:
    """Run one SCRUB forget or retain phase and return average losses."""

    if phase not in {"forget", "retain"}:
        raise ValueError(f"Unsupported SCRUB phase '{phase}'.")
    student_model.train()
    teacher_model.eval()
    kd_loss_sum = 0.0
    ce_loss_sum = 0.0
    objective_loss_sum = 0.0
    sample_count = 0
    progress = tqdm(
        loader,
        desc=f"SCRUB {phase}",
        leave=False,
    )
    for inputs, targets in progress:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        with torch.no_grad():
            teacher_logits = teacher_model(inputs)
        student_logits = student_model(inputs)
        # SCRUB uses KL as the "stay close / move away" control knob:
        # retain batches minimize teacher-student divergence, forget batches maximize it.
        kd_loss = _scrub_temperature_scaled_kl(
            student_logits,
            teacher_logits,
            temperature=temperature,
        )
        if phase == "forget":
            ce_loss_value = 0.0
            objective_loss = -kd_loss
        else:
            ce_loss = retain_criterion(student_logits, targets)
            ce_loss_value = float(ce_loss.item())
            objective_loss = (float(alpha) * kd_loss) + (float(gamma) * ce_loss)
        objective_loss.backward()
        optimizer.step()
        batch_size = targets.size(0)
        kd_loss_sum += float(kd_loss.item()) * batch_size
        ce_loss_sum += ce_loss_value * batch_size
        objective_loss_sum += float(objective_loss.item()) * batch_size
        sample_count += batch_size
    if sample_count == 0:
        return {
            "mean_kd_loss": 0.0,
            "mean_ce_loss": 0.0,
            "mean_objective_loss": 0.0,
        }
    return {
        "mean_kd_loss": kd_loss_sum / sample_count,
        "mean_ce_loss": ce_loss_sum / sample_count,
        "mean_objective_loss": objective_loss_sum / sample_count,
    }


def _run_scrub_unlearning_seed(
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
    class_weighting: str = "auto",
    reuse_existing: bool = True,
) -> dict[str, Any]:
    """Run the native SCRUB method for one seed checkpoint."""

    require_torch()
    context = data_bundle.context
    device = deps["choose_device"](device_name)
    source_metadata = json.loads(Path(checkpoint_path).with_suffix(".json").read_text(encoding="utf-8"))
    seed = int(source_metadata["seed"])
    output_dir = Path(checkpoint_dir) / context.dataset / context.task_id / output_family_name
    checkpoint_stem = f"seed_{seed}"
    output_checkpoint_path = output_dir / f"{checkpoint_stem}.pth"
    output_metadata_path = output_dir / f"{checkpoint_stem}.json"
    algorithm_hyperparameters = _normalize_scrub_hyperparameters(profile_config)
    if reuse_existing and output_checkpoint_path.exists() and output_metadata_path.exists():
        existing_metadata = json.loads(output_metadata_path.read_text(encoding="utf-8"))
        if (
            existing_metadata.get("source_checkpoint") == str(checkpoint_path)
            and existing_metadata.get("unlearning_algorithm") == "SCRUB"
            and existing_metadata.get("algorithm_profile") == profile_name
            and existing_metadata.get("algorithm_hyperparameters") == algorithm_hyperparameters
            and existing_metadata.get("runtime_excludes_validation") is True
        ):
            reused_metadata = dict(existing_metadata)
            reused_metadata["reused_existing"] = True
            return reused_metadata

    forget_batch_size = int(algorithm_hyperparameters["forget_batch_size"])
    retain_batch_size = int(algorithm_hyperparameters["retain_batch_size"])
    epochs = int(algorithm_hyperparameters["epochs"])
    msteps = int(algorithm_hyperparameters["msteps"])
    optimizer_name = str(algorithm_hyperparameters["optimizer"])
    base_lr = float(algorithm_hyperparameters["lr"])
    weight_decay = float(algorithm_hyperparameters["weight_decay"])
    lr_decay_epochs = list(algorithm_hyperparameters["lr_decay_epochs"])
    lr_decay_rate = float(algorithm_hyperparameters["lr_decay_rate"])
    alpha = float(algorithm_hyperparameters["alpha"])
    gamma = float(algorithm_hyperparameters["gamma"])
    kd_temperature = float(algorithm_hyperparameters["kd_temperature"])
    resolved_class_weighting = deps["resolve_class_weighting"](dataset, class_weighting)

    deps["set_random_seed"](seed)
    teacher_model = deps["build_model"](deps["create_resnet18"], num_classes=context.num_classes, dataset=dataset).to(device)
    student_model = deps["build_model"](deps["create_resnet18"], num_classes=context.num_classes, dataset=dataset).to(device)
    deps["load_model_checkpoint"](teacher_model, checkpoint_path, device)
    deps["load_model_checkpoint"](student_model, checkpoint_path, device)
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)

    forget_loader = build_shuffled_loader(data_bundle.loaders["forget"].dataset, batch_size=forget_batch_size)
    retain_loader = build_shuffled_loader(data_bundle.loaders["retain"].dataset, batch_size=retain_batch_size)
    val_loader = data_bundle.loaders["val"]
    class_counts = deps["compute_split_class_counts"](data_bundle, "retrain")
    retain_criterion = deps["build_loss"](
        class_counts,
        context.num_classes,
        device=device,
        class_weighting=resolved_class_weighting,
    )
    optimizer = _build_scrub_optimizer(
        student_model,
        optimizer_name=optimizer_name,
        lr=base_lr,
        weight_decay=weight_decay,
    )
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
            "class_weighting": resolved_class_weighting,
            "image_size": image_size,
            "algorithm": "SCRUB",
            "algorithm_profile": profile_name,
            "algorithm_hyperparameters": algorithm_hyperparameters,
            "source_checkpoint": str(checkpoint_path),
            "output_family_name": output_family_name,
        },
    )

    best_val_accuracy = 0.0
    final_val_accuracy = 0.0
    epoch_history: list[dict[str, Any]] = []
    wall_clock_start = time.perf_counter()
    training_runtime_seconds = 0.0

    epoch_iterator = tqdm(
        range(1, epochs + 1),
        desc=f"SCRUB epochs ({output_family_name}, seed {seed})",
        leave=False,
    )
    for epoch in epoch_iterator:
        current_lr = _apply_scrub_milestone_lr_decay(
            optimizer,
            base_lr=base_lr,
            epoch=epoch,
            lr_decay_epochs=lr_decay_epochs,
            lr_decay_rate=lr_decay_rate,
        )
        epoch_stage = "forget_then_retain" if epoch <= msteps else "retain_only"
        forget_metrics = {
            "mean_kd_loss": 0.0,
            "mean_ce_loss": 0.0,
            "mean_objective_loss": 0.0,
        }
        epoch_training_start = time.perf_counter()
        # The early SCRUB epochs alternate a max-step on forget data with a
        # min-step on retain data; later epochs keep only the retain repair step.
        if epoch <= msteps:
            forget_metrics = _run_scrub_phase(
                phase="forget",
                student_model=student_model,
                teacher_model=teacher_model,
                loader=forget_loader,
                optimizer=optimizer,
                retain_criterion=retain_criterion,
                device=device,
                temperature=kd_temperature,
                alpha=alpha,
                gamma=gamma,
            )
        retain_metrics = _run_scrub_phase(
            phase="retain",
            student_model=student_model,
            teacher_model=teacher_model,
            loader=retain_loader,
            optimizer=optimizer,
            retain_criterion=retain_criterion,
            device=device,
            temperature=kd_temperature,
            alpha=alpha,
            gamma=gamma,
        )
        training_runtime_seconds += time.perf_counter() - epoch_training_start
        val_accuracy = deps["compute_accuracy"](student_model, val_loader, device)
        best_val_accuracy = max(best_val_accuracy, float(val_accuracy))
        final_val_accuracy = float(val_accuracy)
        epoch_metrics = {
            "epoch": float(epoch),
            "forget_loss": float(forget_metrics["mean_kd_loss"]),
            "retain_kd_loss": float(retain_metrics["mean_kd_loss"]),
            "retain_ce_loss": float(retain_metrics["mean_ce_loss"]),
            "val_accuracy": float(val_accuracy),
            "stage": epoch_stage,
            "learning_rate": float(current_lr),
        }
        epoch_history.append(epoch_metrics)
        wandb_run.log(epoch_metrics)
        if hasattr(epoch_iterator, "set_postfix"):
            epoch_iterator.set_postfix(
                {
                    "stage": epoch_stage,
                    "forget": f"{epoch_metrics['forget_loss']:.4f}",
                    "retain_ce": f"{epoch_metrics['retain_ce_loss']:.4f}",
                    "val_acc": f"{epoch_metrics['val_accuracy']:.4f}",
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
        "train_split": "retain",
        "seed": seed,
        "epochs": epochs,
        "batch_size": forget_batch_size,
        "learning_rate": base_lr,
        "momentum": None,
        "weight_decay": weight_decay,
        "optimizer": optimizer_name,
        "class_weighting": resolved_class_weighting,
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
        "unlearning_algorithm": "SCRUB",
        "algorithm_profile": profile_name,
        "algorithm_hyperparameters": algorithm_hyperparameters,
        "forget_batch_size": forget_batch_size,
        "retain_batch_size": retain_batch_size,
        "reused_existing": False,
    }
    output_metadata_path.write_text(json.dumps(output_metadata, indent=2), encoding="utf-8")
    wandb_run.finish()
    return output_metadata


def run_scrub_unlearning_workflow(
    *,
    deps: dict[str, Any],
    dataset: str,
    base_family_dir: str | Path,
    output_family_name: str = "SCRUB",
    num_bank_seeds: int = 3,
    profile: str | None = None,
    checkpoint_dir: str | Path = "checkpoints",
    data_root: str | Path | None = None,
    task_manifest: str | Path | None = None,
    samples_csv: str | Path | None = None,
    device_name: str = "auto",
    use_wandb: bool = False,
    wandb_project: str | None = None,
    class_weighting: str = "auto",
    image_size: int | None = None,
    reuse_existing: bool = True,
    efficiency_aware: bool = False,
    reference_family_dir: str | Path | None = None,
    efficiency_ratio: float = 0.2,
) -> dict[str, Any]:
    """Run the native SCRUB method over a checkpoint bank."""

    profile_name = dataset if profile is None else profile
    profile_config = _resolve_scrub_profile(dataset, profile_name)
    return run_unlearning_workflow_bank(
        deps=deps,
        algorithm_name="SCRUB",
        dataset=dataset,
        base_family_dir=base_family_dir,
        output_family_name=output_family_name,
        num_bank_seeds=num_bank_seeds,
        profile_name=profile_name,
        profile_config=profile_config,
        normalize_hyperparameters=_normalize_scrub_hyperparameters,
        build_efficiency_variants=_build_scrub_efficiency_variants,
        resolve_bundle_batch_size=lambda candidate_profile_config: max(
            int(candidate_profile_config["forget_batch_size"]),
            int(candidate_profile_config["retain_batch_size"]),
        ),
        seed_runner=_run_scrub_unlearning_seed,
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
        seed_runner_kwargs={"class_weighting": class_weighting},
    )
