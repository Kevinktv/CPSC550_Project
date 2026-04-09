"""Reusable training helpers shared by CLI scripts and notebooks."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - depends on local environment.
    torch = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - depends on local environment.
    tqdm = None

try:
    import wandb
except ImportError:  # pragma: no cover - depends on local environment.
    wandb = None

try:
    from Code.metrics import compute_accuracy
    from Code.model_utils import build_model, create_resnet18, choose_device, resolve_image_size, set_random_seed
except ImportError:  # pragma: no cover - allows direct module execution.
    from metrics import compute_accuracy
    from model_utils import build_model, create_resnet18, choose_device, resolve_image_size, set_random_seed


class NullWandbRun:
    """No-op W&B replacement when tracking is disabled or unavailable."""

    def log(self, *_args: Any, **_kwargs: Any) -> None:
        return

    def finish(self) -> None:
        return


DEFAULT_WANDB_PROJECT_PREFIX = "machine-unlearning"


def require_torch() -> None:
    if torch is None:  # pragma: no cover - depends on local environment.
        raise ImportError(
            "torch is required for training. Install dependencies from requirements.txt first."
        )


def default_epochs(dataset: str) -> int:
    """Return the default epoch budget for a dataset."""

    return 30 if dataset == "mufac" else 20


def resolve_class_weighting(dataset: str, mode: str) -> str:
    """Resolve `auto` into a concrete class-weighting mode."""

    if mode == "auto":
        return "inverse_freq" if dataset == "mufac" else "none"
    return mode


def resolve_wandb_project(dataset: str, project: str | None = None) -> str:
    """Return the explicit W&B project or a dataset-specific default."""

    if project:
        return project
    return f"{DEFAULT_WANDB_PROJECT_PREFIX}-{dataset}"


def compute_split_class_counts(data_bundle: Any, split_name: str) -> dict[int, int]:
    """Count classes for the given manifest split."""

    context = data_bundle.context
    source_split = "retrain" if split_name == "retain" else split_name
    counts = {index: 0 for index in range(context.num_classes)}
    for record in context.splits[source_split]:
        counts[record.class_index] = counts.get(record.class_index, 0) + 1
    return counts


def build_loss(class_counts: dict[int, int], num_classes: int, device: Any, class_weighting: str) -> Any:
    """Build the configured classification loss."""

    require_torch()
    if class_weighting == "none":
        return torch.nn.CrossEntropyLoss()

    weights = np.ones(num_classes, dtype=np.float32)
    for class_index in range(num_classes):
        count = class_counts.get(class_index, 0)
        weights[class_index] = 1.0 if count == 0 else 1.0 / float(count)
    weights = weights / weights.sum() * num_classes
    return torch.nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))


def build_optimizer(model: Any, lr: float, momentum: float, weight_decay: float) -> Any:
    """Build the default SGD optimizer."""

    require_torch()
    return torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )


def iter_with_progress(loader: Any, epoch: int, epochs: int, enabled: bool = True) -> Any:
    """Wrap a dataloader with `tqdm` when available."""

    if not enabled or tqdm is None:
        return loader
    return tqdm(loader, desc=f"Epoch {epoch}/{epochs}", leave=False)


def init_wandb_run(
    *,
    enabled: bool,
    entity: str | None = None,
    project: str,
    run_name: str,
    config: dict[str, Any],
) -> Any:
    """Initialize W&B or return a no-op stub."""

    if not enabled or wandb is None:
        return NullWandbRun()

    entity = entity or os.environ.get("WANDB_ENTITY")
    mode = os.environ.get("WANDB_MODE")
    init_kwargs: dict[str, Any] = {
        "project": project,
        "name": run_name,
        "config": config,
    }
    if entity:
        init_kwargs["entity"] = entity
    if mode:
        init_kwargs["mode"] = mode

    try:
        return wandb.init(**init_kwargs)
    except Exception as exc:  # pragma: no cover - depends on local W&B runtime.
        print(
            "W&B initialization failed; continuing without W&B tracking. "
            f"Set USE_WANDB=False or WANDB_MODE=offline to suppress this. Error: {exc}"
        )
        return NullWandbRun()


def train_one_epoch(
    *,
    model: Any,
    loader: Any,
    criterion: Any,
    optimizer: Any,
    device: Any,
    epoch: int,
    epochs: int,
    use_progress: bool = True,
) -> float:
    """Train a model for a single epoch and return average loss."""

    require_torch()
    model.train()
    running_loss = 0.0
    sample_count = 0
    progress = iter_with_progress(loader, epoch=epoch, epochs=epochs, enabled=use_progress)
    for inputs, targets in progress:
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        sample_count += batch_size
    return 0.0 if sample_count == 0 else running_loss / sample_count


def fit_model(
    *,
    data_bundle: Any,
    dataset: str,
    train_loader_name: str,
    class_count_split_name: str,
    metadata_train_split: str,
    batch_size: int,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    seed: int,
    device_name: str,
    checkpoint_dir: str | Path,
    run_name: str,
    class_weighting: str = "auto",
    wandb_project: str | None = None,
    use_wandb: bool = True,
    image_size: int | None = None,
    model_factory: Any = None,
    initial_state_dict: dict[str, Any] | None = None,
    checkpoint_stem: str | None = None,
    metadata_extra: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Fit one model and persist checkpoint plus sidecar metadata."""

    require_torch()
    model_factory = model_factory or create_resnet18
    context = data_bundle.context
    class_weighting = resolve_class_weighting(dataset, class_weighting)
    image_size = resolve_image_size(dataset, image_size)
    device = choose_device(device_name)
    set_random_seed(seed)

    train_loader = data_bundle.loaders[train_loader_name]
    val_loader = data_bundle.loaders["val"]
    class_counts = compute_split_class_counts(data_bundle, class_count_split_name)
    model = build_model(model_factory, num_classes=context.num_classes, dataset=dataset).to(device)
    if initial_state_dict is not None:
        model.load_state_dict(initial_state_dict)
    criterion = build_loss(class_counts, context.num_classes, device=device, class_weighting=class_weighting)
    optimizer = build_optimizer(model, lr=lr, momentum=momentum, weight_decay=weight_decay)

    config = {
        "dataset": dataset,
        "task_id": context.task_id,
        "train_split": metadata_train_split,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "seed": seed,
        "class_weighting": class_weighting,
        "num_classes": context.num_classes,
        "image_size": image_size,
    }
    wandb_run = init_wandb_run(
        enabled=use_wandb,
        entity="inmdev-university-of-british-columbia",
        project=resolve_wandb_project(dataset, wandb_project),
        run_name=run_name,
        config=config,
    )

    best_val_accuracy = 0.0
    epoch_history: list[dict[str, float]] = []
    wall_clock_start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            epochs=epochs,
            use_progress=True,
        )
        val_accuracy = compute_accuracy(model, val_loader, device)
        best_val_accuracy = max(best_val_accuracy, val_accuracy)
        epoch_metrics = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
            "val_accuracy": float(val_accuracy),
        }
        epoch_history.append(epoch_metrics)
        wandb_run.log(epoch_metrics)

    runtime_seconds = time.perf_counter() - wall_clock_start
    output_dir = Path(checkpoint_dir) / context.dataset / context.task_id / run_name
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_stem = checkpoint_stem or f"seed_{seed}"
    checkpoint_path = output_dir / f"{checkpoint_stem}.pth"
    metadata_path = output_dir / f"{checkpoint_stem}.json"
    torch.save(model.state_dict(), checkpoint_path)
    final_val_accuracy = epoch_history[-1]["val_accuracy"] if epoch_history else 0.0
    metadata = {
        "dataset": context.dataset,
        "task_id": context.task_id,
        "run_name": run_name,
        "train_split": metadata_train_split,
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "class_weighting": class_weighting,
        "num_classes": context.num_classes,
        "label_to_index": context.label_to_index,
        "class_names": context.class_names,
        "runtime_seconds": runtime_seconds,
        "best_val_accuracy": best_val_accuracy,
        "final_val_accuracy": final_val_accuracy,
        "checkpoint_path": str(checkpoint_path),
        "epochs_logged": epoch_history,
        "image_size": image_size,
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    wandb_run.finish()
    return model, metadata
