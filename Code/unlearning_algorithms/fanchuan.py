"""Native notebook integration for the Fanchuan unlearning baseline."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .common import build_epoch_efficiency_variants, build_shuffled_loader, require_torch, resolve_unlearning_profile, run_unlearning_workflow_bank

try:
    import torch
except ImportError:  # pragma: no cover - depends on local environment.
    torch = None


FANCHUAN_UNLEARNING_PROFILES: dict[str, dict[str, float | int]] = {
    "cifar10": {
        "loader_batch_size": 64,
        "retain_batch_size": 256,
        "epochs": 6,
        "uniform_stage_lr": 5e-3,
        "forget_stage_lr": 3e-4,
        "retain_stage_lr": 4e-3,
        "retain_weight_decay": 1e-2,
        "forget_weight_decay": 0.0,
        "uniform_weight_decay": 0.0,
    },
    "mufac": {
        "loader_batch_size": 32,
        "retain_batch_size": 64,
        "epochs": 6,
        "uniform_stage_lr": 1e-3,
        "forget_stage_lr": 1e-4,
        "retain_stage_lr": 1e-3,
        "retain_weight_decay": 1e-2,
        "forget_weight_decay": 0.0,
        "uniform_weight_decay": 0.0,
    },
}


def _resolve_fanchuan_profile(dataset: str, profile: str | None) -> dict[str, float | int]:
    """Resolve a named FanchuanUnlearning profile, defaulting to the dataset name."""

    _profile_name, profile_config = resolve_unlearning_profile(
        dataset=dataset,
        profile=profile,
        profiles=FANCHUAN_UNLEARNING_PROFILES,
        algorithm_name="FanchuanUnlearning",
    )
    return profile_config


def _normalize_fanchuan_hyperparameters(profile_config: dict[str, float | int]) -> dict[str, float | int]:
    """Normalize Fanchuan hyperparameters into a stable JSON-serializable shape."""

    return {
        "loader_batch_size": int(profile_config["loader_batch_size"]),
        "retain_batch_size": int(profile_config["retain_batch_size"]),
        "epochs": int(profile_config["epochs"]),
        "uniform_stage_lr": float(profile_config["uniform_stage_lr"]),
        "forget_stage_lr": float(profile_config["forget_stage_lr"]),
        "retain_stage_lr": float(profile_config["retain_stage_lr"]),
        "retain_weight_decay": float(profile_config["retain_weight_decay"]),
        "forget_weight_decay": float(profile_config["forget_weight_decay"]),
        "uniform_weight_decay": float(profile_config["uniform_weight_decay"]),
    }


def _build_fanchuan_efficiency_variants(
    profile_name: str,
    profile_config: dict[str, float | int],
) -> list[tuple[str, dict[str, float | int]]]:
    """Build a small quality-to-speed variant ladder for Fanchuan."""

    return build_epoch_efficiency_variants(
        profile_name=profile_name,
        profile_config=_normalize_fanchuan_hyperparameters(profile_config),
        epoch_candidates=[int(profile_config["epochs"]), 4, 3, 2],
        min_epochs=1,
    )


def _fanchuan_uniform_kl_loss(logits: Any) -> Any:
    """Match model predictions to a uniform target distribution."""

    uniform_targets = torch.full_like(logits, 1.0 / float(logits.shape[1]))
    return torch.nn.KLDivLoss(reduction="batchmean")(torch.nn.LogSoftmax(dim=-1)(logits), uniform_targets)


def _fanchuan_contrastive_loss(forget_logits: Any, retain_logits: Any, *, temperature: float) -> Any:
    """Contrast forget logits against retain logits using the FanchuanUnlearning objective."""

    pairwise_scores = (forget_logits @ retain_logits.T) / float(temperature)
    return -torch.nn.LogSoftmax(dim=-1)(pairwise_scores).mean()


def _run_fanchuan_unlearning_seed(
    *,
    deps: dict[str, Any],
    dataset: str,
    checkpoint_path: str | Path,
    output_family_name: str,
    profile_name: str,
    profile_config: dict[str, float | int],
    data_bundle: Any,
    checkpoint_dir: str | Path,
    device_name: str,
    image_size: int,
    use_wandb: bool,
    wandb_project: str | None,
    class_weighting: str = "auto",
    reuse_existing: bool = True,
) -> dict[str, Any]:
    """Run the two-stage FanchuanUnlearning method for one seed checkpoint."""

    require_torch()
    context = data_bundle.context
    device = deps["choose_device"](device_name)
    metadata = json.loads(Path(checkpoint_path).with_suffix(".json").read_text(encoding="utf-8"))
    seed = int(metadata["seed"])
    output_dir = Path(checkpoint_dir) / context.dataset / context.task_id / output_family_name
    checkpoint_stem = f"seed_{seed}"
    output_checkpoint_path = output_dir / f"{checkpoint_stem}.pth"
    output_metadata_path = output_dir / f"{checkpoint_stem}.json"
    algorithm_hyperparameters = _normalize_fanchuan_hyperparameters(profile_config)
    if reuse_existing and output_checkpoint_path.exists() and output_metadata_path.exists():
        existing_metadata = json.loads(output_metadata_path.read_text(encoding="utf-8"))
        if (
            existing_metadata.get("source_checkpoint") == str(checkpoint_path)
            and existing_metadata.get("unlearning_algorithm") == "FanchuanUnlearning"
            and existing_metadata.get("algorithm_profile") == profile_name
            and existing_metadata.get("algorithm_hyperparameters") == algorithm_hyperparameters
            and existing_metadata.get("runtime_excludes_validation") is True
        ):
            reused_metadata = dict(existing_metadata)
            reused_metadata["reused_existing"] = True
            return reused_metadata

    loader_batch_size = int(profile_config["loader_batch_size"])
    retain_batch_size = int(profile_config["retain_batch_size"])
    epochs = int(profile_config["epochs"])
    temperature = 1.15
    momentum = 0.9
    resolved_class_weighting = deps["resolve_class_weighting"](dataset, class_weighting)

    deps["set_random_seed"](seed)
    model = deps["build_model"](deps["create_resnet18"], num_classes=context.num_classes, dataset=dataset).to(device)
    deps["load_model_checkpoint"](model, checkpoint_path, device)

    forget_loader = build_shuffled_loader(data_bundle.loaders["forget"].dataset, batch_size=loader_batch_size)
    retain_loader = build_shuffled_loader(data_bundle.loaders["retain"].dataset, batch_size=retain_batch_size)
    retain_pair_loader = build_shuffled_loader(data_bundle.loaders["retain"].dataset, batch_size=retain_batch_size)
    val_loader = data_bundle.loaders["val"]

    class_counts = deps["compute_split_class_counts"](data_bundle, "retrain")
    retain_criterion = deps["build_loss"](
        class_counts,
        context.num_classes,
        device=device,
        class_weighting=resolved_class_weighting,
    )
    uniform_optimizer = deps["build_optimizer"](
        model,
        lr=float(profile_config["uniform_stage_lr"]),
        momentum=momentum,
        weight_decay=float(profile_config["uniform_weight_decay"]),
    )
    forget_optimizer = deps["build_optimizer"](
        model,
        lr=float(profile_config["forget_stage_lr"]),
        momentum=momentum,
        weight_decay=float(profile_config["forget_weight_decay"]),
    )
    retain_optimizer = deps["build_optimizer"](
        model,
        lr=float(profile_config["retain_stage_lr"]),
        momentum=momentum,
        weight_decay=float(profile_config["retain_weight_decay"]),
    )
    forget_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        forget_optimizer,
        T_max=max(1, len(forget_loader) * epochs),
        eta_min=1e-6,
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
            "algorithm": "FanchuanUnlearning",
            "algorithm_profile": profile_name,
            "algorithm_hyperparameters": algorithm_hyperparameters,
            "loader_batch_size": loader_batch_size,
            "retain_batch_size": retain_batch_size,
            "epochs": epochs,
            "uniform_stage_lr": float(profile_config["uniform_stage_lr"]),
            "forget_stage_lr": float(profile_config["forget_stage_lr"]),
            "retain_stage_lr": float(profile_config["retain_stage_lr"]),
            "uniform_weight_decay": float(profile_config["uniform_weight_decay"]),
            "forget_weight_decay": float(profile_config["forget_weight_decay"]),
            "retain_weight_decay": float(profile_config["retain_weight_decay"]),
            "temperature": temperature,
            "source_checkpoint": str(checkpoint_path),
            "output_family_name": output_family_name,
        },
    )

    epoch_history: list[dict[str, Any]] = []
    wall_clock_start = time.perf_counter()
    training_runtime_seconds = 0.0

    model.train()
    uniform_loss_sum = 0.0
    uniform_sample_count = 0
    uniform_stage_start = time.perf_counter()
    # Stage 0 flattens predictions on forget examples before the alternating
    # forget/retain updates begin.
    for forget_inputs, _forget_targets in forget_loader:
        forget_inputs = forget_inputs.to(device)
        uniform_optimizer.zero_grad()
        forget_logits = model(forget_inputs)
        uniform_loss = _fanchuan_uniform_kl_loss(forget_logits)
        uniform_loss.backward()
        uniform_optimizer.step()
        batch_size = forget_inputs.size(0)
        uniform_loss_sum += float(uniform_loss.item()) * batch_size
        uniform_sample_count += batch_size
    training_runtime_seconds += time.perf_counter() - uniform_stage_start

    stage_zero_val_accuracy = deps["compute_accuracy"](model, val_loader, device)
    stage_zero_metrics = {
        "epoch": 0.0,
        "forget_loss": 0.0 if uniform_sample_count == 0 else uniform_loss_sum / uniform_sample_count,
        "retain_loss": 0.0,
        "val_accuracy": float(stage_zero_val_accuracy),
        "temperature": temperature,
        "stage": "uniform_forget",
    }
    epoch_history.append(stage_zero_metrics)
    wandb_run.log(stage_zero_metrics)

    best_val_accuracy = float(stage_zero_val_accuracy)
    final_val_accuracy = float(stage_zero_val_accuracy)

    for epoch in range(1, epochs + 1):
        model.train()
        forget_loss_sum = 0.0
        forget_sample_count = 0
        epoch_training_start = time.perf_counter()
        # Fanchuan's forget step separates forget features from retain features,
        # then a retain CE pass restores ordinary task performance.
        for (forget_inputs, _forget_targets), (retain_inputs, _retain_targets) in zip(forget_loader, retain_pair_loader):
            forget_inputs = forget_inputs.to(device)
            retain_inputs = retain_inputs.to(device)
            forget_optimizer.zero_grad()
            forget_logits = model(forget_inputs)
            retain_logits = model(retain_inputs).detach()
            forget_loss = _fanchuan_contrastive_loss(
                forget_logits,
                retain_logits,
                temperature=temperature,
            )
            forget_loss.backward()
            forget_optimizer.step()
            forget_scheduler.step()
            batch_size = forget_inputs.size(0)
            forget_loss_sum += float(forget_loss.item()) * batch_size
            forget_sample_count += batch_size

        retain_loss_sum = 0.0
        retain_sample_count = 0
        for retain_inputs, retain_targets in retain_loader:
            retain_inputs = retain_inputs.to(device)
            retain_targets = retain_targets.to(device)
            retain_optimizer.zero_grad()
            retain_logits = model(retain_inputs)
            retain_loss = retain_criterion(retain_logits, retain_targets)
            retain_loss.backward()
            retain_optimizer.step()
            batch_size = retain_targets.size(0)
            retain_loss_sum += float(retain_loss.item()) * batch_size
            retain_sample_count += batch_size
        training_runtime_seconds += time.perf_counter() - epoch_training_start

        val_accuracy = deps["compute_accuracy"](model, val_loader, device)
        best_val_accuracy = max(best_val_accuracy, float(val_accuracy))
        final_val_accuracy = float(val_accuracy)
        epoch_metrics = {
            "epoch": float(epoch),
            "forget_loss": 0.0 if forget_sample_count == 0 else forget_loss_sum / forget_sample_count,
            "retain_loss": 0.0 if retain_sample_count == 0 else retain_loss_sum / retain_sample_count,
            "val_accuracy": float(val_accuracy),
            "temperature": temperature,
            "stage": "two_stage",
        }
        epoch_history.append(epoch_metrics)
        wandb_run.log(epoch_metrics)

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
        "batch_size": loader_batch_size,
        "learning_rate": float(profile_config["forget_stage_lr"]),
        "momentum": momentum,
        "weight_decay": float(profile_config["forget_weight_decay"]),
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
        "unlearning_algorithm": "FanchuanUnlearning",
        "algorithm_profile": profile_name,
        "algorithm_hyperparameters": algorithm_hyperparameters,
        "algorithm_temperature": temperature,
        "loader_batch_size": loader_batch_size,
        "retain_batch_size": retain_batch_size,
        "algorithm_learning_rates": {
            "uniform_stage": float(profile_config["uniform_stage_lr"]),
            "forget_stage": float(profile_config["forget_stage_lr"]),
            "retain_stage": float(profile_config["retain_stage_lr"]),
        },
        "algorithm_weight_decays": {
            "uniform_stage": float(profile_config["uniform_weight_decay"]),
            "forget_stage": float(profile_config["forget_weight_decay"]),
            "retain_stage": float(profile_config["retain_weight_decay"]),
        },
        "reused_existing": False,
    }
    output_metadata_path.write_text(json.dumps(output_metadata, indent=2), encoding="utf-8")
    wandb_run.finish()
    return output_metadata


def run_fanchuan_unlearning_workflow(
    *,
    deps: dict[str, Any],
    dataset: str,
    base_family_dir: str | Path,
    output_family_name: str = "FanchuanUnlearning",
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
    """Run the ported FanchuanUnlearning two-stage method over a checkpoint bank."""

    profile_name = dataset if profile is None else profile
    profile_config = _resolve_fanchuan_profile(dataset, profile_name)
    return run_unlearning_workflow_bank(
        deps=deps,
        algorithm_name="FanchuanUnlearning",
        dataset=dataset,
        base_family_dir=base_family_dir,
        output_family_name=output_family_name,
        num_bank_seeds=num_bank_seeds,
        profile_name=profile_name,
        profile_config=profile_config,
        normalize_hyperparameters=_normalize_fanchuan_hyperparameters,
        build_efficiency_variants=_build_fanchuan_efficiency_variants,
        resolve_bundle_batch_size=lambda candidate_profile_config: int(candidate_profile_config["loader_batch_size"]),
        seed_runner=_run_fanchuan_unlearning_seed,
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
