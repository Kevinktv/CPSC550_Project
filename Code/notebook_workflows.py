"""Notebook-oriented orchestration for training, inference, and benchmarking."""

from __future__ import annotations

import argparse
import csv
import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover - depends on local environment.
    pd = None

try:
    import wandb
except ImportError:  # pragma: no cover - depends on local environment.
    wandb = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - depends on local environment.
    def tqdm(iterable: Any, *args: Any, **kwargs: Any) -> Any:
        return iterable

try:
    import torch
except ImportError:  # pragma: no cover - depends on local environment.
    torch = None

try:
    from Code.benchmark_utils import benchmark_model_families
    from Code.data_utils import create_dataloaders_from_manifest, resolve_pipeline_paths
    from Code.metrics import compute_accuracy
    from Code.model_utils import build_model, choose_device, create_resnet18, load_model_checkpoint, resolve_image_size, set_random_seed
    from Code.train_baselines import train
    from Code.training_utils import (
        build_loss,
        build_optimizer,
        compute_split_class_counts,
        default_epochs,
        fit_model,
        init_wandb_run,
        resolve_class_weighting,
        resolve_wandb_project,
        train_one_epoch,
    )
except ImportError:  # pragma: no cover - allows direct module execution.
    from benchmark_utils import benchmark_model_families
    from data_utils import create_dataloaders_from_manifest, resolve_pipeline_paths
    from metrics import compute_accuracy
    from model_utils import build_model, choose_device, create_resnet18, load_model_checkpoint, resolve_image_size, set_random_seed
    from train_baselines import train
    from training_utils import (
        build_loss,
        build_optimizer,
        compute_split_class_counts,
        default_epochs,
        fit_model,
        init_wandb_run,
        resolve_class_weighting,
        resolve_wandb_project,
        train_one_epoch,
    )


DEFAULT_SEARCH_SPACES: dict[str, dict[str, list[Any]]] = {
    "mufac": {
        "lr": [0.02, 0.05, 0.08],
        "weight_decay": [1e-4, 5e-4, 1e-3],
        "batch_size": [128],
    },
    "cifar10": {
        "lr": [0.001, 0.01, 0.05, 0.1, 0.5],
        "weight_decay": [1e-5, 5e-3],
        "batch_size": [64,128, 256],
    },
}

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


def require_torch() -> None:
    if torch is None:  # pragma: no cover - depends on local environment.
        raise ImportError(
            "torch is required for notebook training and benchmarking. "
            "Install dependencies from requirements.txt first."
        )


def build_small_search_space(dataset: str) -> dict[str, list[Any]]:
    """Return the fixed small search space for the requested dataset."""

    if dataset not in DEFAULT_SEARCH_SPACES:
        raise ValueError(f"Unsupported dataset: {dataset}")
    return deepcopy(DEFAULT_SEARCH_SPACES[dataset])


def expand_search_space(search_space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand a cartesian-product search space into a list of configs."""

    configs: list[dict[str, Any]] = []
    batch_sizes = search_space.get("batch_size", [])
    learning_rates = search_space.get("lr", [])
    weight_decays = search_space.get("weight_decay", [])
    for batch_size in batch_sizes:
        for learning_rate in learning_rates:
            for weight_decay in weight_decays:
                configs.append(
                    {
                        "batch_size": batch_size,
                        "lr": learning_rate,
                        "weight_decay": weight_decay,
                    }
                )
    return configs


def apply_smoke_mode(
    search_space: dict[str, list[Any]],
    *,
    epochs: int,
    num_bank_seeds: int,
    smoke_mode: bool,
) -> dict[str, Any]:
    """Reduce runtime settings for notebook smoke runs."""

    if not smoke_mode:
        return {
            "search_space": deepcopy(search_space),
            "epochs": epochs,
            "num_bank_seeds": num_bank_seeds,
        }
    reduced_search = {key: values[:1] for key, values in search_space.items()}
    return {
        "search_space": reduced_search,
        "epochs": 1,
        "num_bank_seeds": 1,
    }


def rank_search_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort search results by validation accuracy, then runtime, then config index."""

    return sorted(
        results,
        key=lambda item: (
            -float(item["best_val_accuracy"]),
            float(item["runtime_seconds"]),
            int(item["config_index"]),
        ),
    )


def select_best_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the top-ranked result from a search table."""

    if not results:
        raise ValueError("Cannot select a best result from an empty search table.")
    return rank_search_results(results)[0]


def _namespace_for_training(
    *,
    dataset: str,
    train_split: str,
    batch_size: int,
    epochs: int,
    lr: float,
    momentum: float,
    weight_decay: float,
    seed: int,
    device: str,
    checkpoint_dir: str | Path,
    run_name: str,
    wandb_project: str | None,
    use_wandb: bool,
    class_weighting: str,
    image_size: int | None,
    samples_csv: str | Path | None,
    task_manifest: str | Path | None,
    data_root: str | Path | None,
) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=dataset,
        task_manifest=task_manifest,
        samples_csv=samples_csv,
        data_root=data_root,
        train_split=train_split,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
        num_workers=0,
        seed=seed,
        device=device,
        checkpoint_dir=str(checkpoint_dir),
        run_name=run_name,
        wandb_project=wandb_project,
        use_wandb=use_wandb,
        class_weighting=class_weighting,
        image_size=image_size,
    )


def _ensure_table_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def _write_results_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _results_root_for_split(results_dir: str | Path, dataset: str, split_name: str) -> Path:
    """Return the standard results directory for one dataset split."""

    return Path(results_dir) / dataset / split_name


def _search_space_signature(search_space: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Return a deterministic normalized signature for the expanded search space."""

    return [
        {
            "batch_size": int(config["batch_size"]),
            "learning_rate": float(config["lr"]),
            "weight_decay": float(config["weight_decay"]),
        }
        for config in expand_search_space(search_space)
    ]


def results_to_dataframe(rows: list[dict[str, Any]]) -> Any:
    """Convert rows to a DataFrame when pandas is available."""

    if pd is None:
        return rows
    return pd.DataFrame(rows)


def _load_saved_train_best_result(
    *,
    dataset: str,
    results_dir: str | Path,
    epochs: int,
) -> tuple[dict[str, Any], Path]:
    """Load the persisted best train config for retrain-only notebook runs."""

    best_config_path = _results_root_for_split(results_dir, dataset, "train") / "best_config.json"
    if not best_config_path.exists():
        raise FileNotFoundError(
            "Missing train best-config file for retrain. Run the train grid search first: "
            f"{best_config_path}"
        )
    best_result = json.loads(best_config_path.read_text(encoding="utf-8"))
    best_result["epochs"] = int(best_result.get("epochs", epochs))
    return best_result, best_config_path


def _load_cached_grid_search_for_split(
    *,
    dataset: str,
    train_split: str,
    search_space: dict[str, list[Any]],
    epochs: int,
    seed: int,
    results_dir: str | Path,
    class_weighting: str,
    momentum: float,
    image_size: int,
) -> dict[str, Any] | None:
    """Load cached grid-search results when the saved search definition matches."""

    results_root = _results_root_for_split(results_dir, dataset, train_split)
    results_path = results_root / "grid_search_results.json"
    best_config_path = results_root / "best_config.json"
    metadata_path = results_root / "search_metadata.json"
    if not results_path.exists() or not best_config_path.exists():
        return None

    results = json.loads(results_path.read_text(encoding="utf-8"))
    best_result = json.loads(best_config_path.read_text(encoding="utf-8"))
    expected_signature = _search_space_signature(search_space)

    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        current_metadata = {
            "dataset": dataset,
            "train_split": train_split,
            "epochs": int(epochs),
            "seed": int(seed),
            "class_weighting": class_weighting,
            "momentum": float(momentum),
            "image_size": int(image_size),
            "search_space_signature": expected_signature,
        }
        if metadata != current_metadata:
            return None
        cache_match = "metadata"
    else:
        cached_signature = [
            {
                "batch_size": int(row["batch_size"]),
                "learning_rate": float(row["learning_rate"]),
                "weight_decay": float(row["weight_decay"]),
            }
            for row in sorted(results, key=lambda item: int(item["config_index"]))
        ]
        seeds = {int(row["seed"]) for row in results}
        splits = {row["train_split"] for row in results}
        if cached_signature != expected_signature or seeds != {int(seed)} or splits != {train_split}:
            return None
        cache_match = "legacy_results"

    return {
        "results": results,
        "best_result": best_result,
        "results_dir": str(results_root),
        "loaded_from_cache": True,
        "cache_match": cache_match,
    }


def _build_reused_search_info(
    *,
    dataset: str,
    results_dir: str | Path,
    best_result: dict[str, Any],
    source_best_config_path: str | Path,
) -> dict[str, Any]:
    """Build and persist the retrain config-reuse summary."""

    results_root = _ensure_table_dir(_results_root_for_split(results_dir, dataset, "retrain"))
    reused_best_result = dict(best_result)
    summary = {
        "selection_mode": "reuse_train_best",
        "source_split": "train",
        "selected_on_split": "train",
        "applied_to_split": "retrain",
        "source_best_config_path": str(Path(source_best_config_path)),
        "best_result": reused_best_result,
        "results_dir": str(results_root),
    }
    (results_root / "selection_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    (results_root / "best_config.json").write_text(
        json.dumps(reused_best_result, indent=2),
        encoding="utf-8",
    )
    return summary


def run_grid_search_for_split(
    *,
    dataset: str,
    train_split: str,
    search_space: dict[str, list[Any]],
    epochs: int,
    seed: int,
    checkpoint_dir: str | Path,
    results_dir: str | Path,
    use_wandb: bool,
    load_existing_if_match: bool = False,
    wandb_project: str | None = None,
    class_weighting: str = "auto",
    momentum: float = 0.9,
    image_size: int | None = None,
    device: str = "auto",
    data_root: str | Path | None = None,
    task_manifest: str | Path | None = None,
    samples_csv: str | Path | None = None,
) -> dict[str, Any]:
    """Run a fixed-size grid search for one training split."""

    if load_existing_if_match:
        cached = _load_cached_grid_search_for_split(
            dataset=dataset,
            train_split=train_split,
            search_space=search_space,
            epochs=epochs,
            seed=seed,
            results_dir=results_dir,
            class_weighting=class_weighting,
            momentum=momentum,
            image_size=image_size,
        )
        if cached is not None:
            return cached

    configurations = expand_search_space(search_space)
    wandb_project = resolve_wandb_project(dataset, wandb_project)
    search_rows: list[dict[str, Any]] = []
    for index, config in enumerate(configurations, start=1):
        run_name = f"grid_search_{train_split}_cfg{index:02d}"
        args = _namespace_for_training(
            dataset=dataset,
            train_split=train_split,
            batch_size=int(config["batch_size"]),
            epochs=epochs,
            lr=float(config["lr"]),
            momentum=momentum,
            weight_decay=float(config["weight_decay"]),
            seed=seed,
            device=device,
            checkpoint_dir=checkpoint_dir,
            run_name=run_name,
            wandb_project=wandb_project,
            use_wandb=use_wandb,
            class_weighting=class_weighting,
            image_size=image_size,
            samples_csv=samples_csv,
            task_manifest=task_manifest,
            data_root=data_root,
        )
        metadata = train(args)
        search_rows.append(
            {
                "config_index": index,
                "train_split": train_split,
                "batch_size": int(config["batch_size"]),
                "learning_rate": float(config["lr"]),
                "weight_decay": float(config["weight_decay"]),
                "best_val_accuracy": float(metadata["best_val_accuracy"]),
                "final_val_accuracy": float(metadata["final_val_accuracy"]),
                "runtime_seconds": float(metadata["runtime_seconds"]),
                "checkpoint_path": metadata["checkpoint_path"],
                "seed": seed,
            }
        )

    best_result = select_best_result(search_rows)
    results_root = _ensure_table_dir(_results_root_for_split(results_dir, dataset, train_split))
    _write_results_csv(results_root / "grid_search_results.csv", rank_search_results(search_rows))
    (results_root / "grid_search_results.json").write_text(
        json.dumps(rank_search_results(search_rows), indent=2),
        encoding="utf-8",
    )
    (results_root / "best_config.json").write_text(json.dumps(best_result, indent=2), encoding="utf-8")
    (results_root / "search_metadata.json").write_text(
        json.dumps(
            {
                "dataset": dataset,
                "train_split": train_split,
                "epochs": int(epochs),
                "seed": int(seed),
                "class_weighting": class_weighting,
                "momentum": float(momentum),
                "image_size": int(image_size),
                "search_space_signature": _search_space_signature(search_space),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "results": search_rows,
        "best_result": best_result,
        "results_dir": str(results_root),
        "loaded_from_cache": False,
        "cache_match": None,
    }


def train_best_model_bank(
    *,
    dataset: str,
    train_split: str,
    best_result: dict[str, Any],
    canonical_seed: int,
    num_bank_seeds: int,
    checkpoint_dir: str | Path,
    use_wandb: bool,
    wandb_project: str | None = None,
    class_weighting: str = "auto",
    momentum: float = 0.9,
    image_size: int | None = None,
    device: str = "auto",
    data_root: str | Path | None = None,
    task_manifest: str | Path | None = None,
    samples_csv: str | Path | None = None,
    metadata_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train the canonical checkpoint and optional small seed bank for one split."""

    family_name = "baseline_retrain" if train_split == "retrain" else "baseline_train"
    wandb_project = resolve_wandb_project(dataset, wandb_project)
    seeds = [canonical_seed + offset for offset in range(num_bank_seeds)]
    metadata_by_seed: list[dict[str, Any]] = []
    for current_seed in seeds:
        args = _namespace_for_training(
            dataset=dataset,
            train_split=train_split,
            batch_size=int(best_result["batch_size"]),
            epochs=default_epochs(dataset) if "epochs" not in best_result else int(best_result.get("epochs", default_epochs(dataset))),
            lr=float(best_result["learning_rate"]) if "learning_rate" in best_result else float(best_result["lr"]),
            momentum=momentum,
            weight_decay=float(best_result["weight_decay"]),
            seed=current_seed,
            device=device,
            checkpoint_dir=checkpoint_dir,
            run_name=family_name,
            wandb_project=wandb_project,
            use_wandb=use_wandb,
            class_weighting=class_weighting,
            image_size=image_size,
            samples_csv=samples_csv,
            task_manifest=task_manifest,
            data_root=data_root,
        )
        if metadata_extra:
            args.metadata_extra = dict(metadata_extra)
        metadata_by_seed.append(train(args))

    return {
        "family_name": family_name,
        "seed_bank": metadata_by_seed,
        "canonical_checkpoint": metadata_by_seed[0]["checkpoint_path"] if metadata_by_seed else None,
    }


def run_training_notebook_workflow(
    *,
    dataset: str,
    run_train: bool,
    run_retrain: bool,
    search_space: dict[str, list[Any]] | None = None,
    epochs: int | None = None,
    seed: int = 0,
    num_bank_seeds: int = 3,
    use_wandb: bool = False,
    smoke_mode: bool = False,
    checkpoint_dir: str | Path = "checkpoints",
    results_dir: str | Path = "results/grid_search",
    data_root: str | Path | None = None,
    task_manifest: str | Path | None = None,
    samples_csv: str | Path | None = None,
    momentum: float = 0.9,
    image_size: int | None = None,
    class_weighting: str = "auto",
    device: str = "auto",
    wandb_project: str | None = None,
    reuse_existing_train_grid: bool = False,
) -> dict[str, Any]:
    """Execute the notebook-facing grid-search and final-training workflow."""

    search_space = search_space or build_small_search_space(dataset)
    epochs = default_epochs(dataset) if epochs is None else epochs
    runtime_config = apply_smoke_mode(
        search_space,
        epochs=epochs,
        num_bank_seeds=num_bank_seeds,
        smoke_mode=smoke_mode,
    )
    search_space = runtime_config["search_space"]
    epochs = runtime_config["epochs"]
    num_bank_seeds = runtime_config["num_bank_seeds"]

    outputs: dict[str, Any] = {
        "dataset": dataset,
        "epochs": epochs,
        "num_bank_seeds": num_bank_seeds,
        "search_space": search_space,
        "splits": {},
    }
    train_best_result: dict[str, Any] | None = None
    train_best_config_path: Path | None = None

    if run_train:
        search_info = run_grid_search_for_split(
            dataset=dataset,
            train_split="train",
            search_space=search_space,
            epochs=epochs,
            seed=seed,
            checkpoint_dir=checkpoint_dir,
            results_dir=results_dir,
            use_wandb=use_wandb,
            load_existing_if_match=reuse_existing_train_grid,
            wandb_project=wandb_project,
            class_weighting=class_weighting,
            momentum=momentum,
            image_size=image_size,
            device=device,
            data_root=data_root,
            task_manifest=task_manifest,
            samples_csv=samples_csv,
        )
        train_best_result = dict(search_info["best_result"])
        train_best_result["epochs"] = epochs
        train_best_config_path = Path(search_info["results_dir"]) / "best_config.json"
        bank_info = train_best_model_bank(
            dataset=dataset,
            train_split="train",
            best_result=train_best_result,
            canonical_seed=seed,
            num_bank_seeds=num_bank_seeds,
            checkpoint_dir=checkpoint_dir,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            class_weighting=class_weighting,
            momentum=momentum,
            image_size=image_size,
            device=device,
            data_root=data_root,
            task_manifest=task_manifest,
            samples_csv=samples_csv,
        )
        outputs["splits"]["train"] = {
            "grid_search": search_info,
            "bank_training": bank_info,
        }

    if run_retrain:
        if train_best_result is None or train_best_config_path is None:
            train_best_result, train_best_config_path = _load_saved_train_best_result(
                dataset=dataset,
                results_dir=results_dir,
                epochs=epochs,
            )
        else:
            train_best_result = dict(train_best_result)

        train_best_result["epochs"] = epochs
        search_info = _build_reused_search_info(
            dataset=dataset,
            results_dir=results_dir,
            best_result=train_best_result,
            source_best_config_path=train_best_config_path,
        )
        bank_info = train_best_model_bank(
            dataset=dataset,
            train_split="retrain",
            best_result=train_best_result,
            canonical_seed=seed,
            num_bank_seeds=num_bank_seeds,
            checkpoint_dir=checkpoint_dir,
            use_wandb=use_wandb,
            wandb_project=wandb_project,
            class_weighting=class_weighting,
            momentum=momentum,
            image_size=image_size,
            device=device,
            data_root=data_root,
            task_manifest=task_manifest,
            samples_csv=samples_csv,
            metadata_extra={
                "selection_mode": "reuse_train_best",
                "selected_on_split": "train",
                "applied_to_split": "retrain",
                "source_best_config_path": str(train_best_config_path),
            },
        )
        outputs["splits"]["retrain"] = {
            "grid_search": search_info,
            "bank_training": bank_info,
        }
    return outputs


def preview_checkpoint_predictions(
    *,
    dataset: str,
    checkpoint_path: str | Path,
    loader_name: str = "test",
    sample_count: int = 5,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int | None = None,
    data_root: str | Path | None = None,
    task_manifest: str | Path | None = None,
    samples_csv: str | Path | None = None,
    device_name: str = "auto",
    model_factory: Any = None,
) -> list[dict[str, Any]]:
    """Preview a few predictions from a checkpoint on a selected loader."""

    require_torch()
    model_factory = model_factory or create_resnet18
    image_size = resolve_image_size(dataset, image_size)
    data_bundle = create_dataloaders_from_manifest(
        dataset=dataset,
        task_manifest=task_manifest,
        samples_csv=samples_csv,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        image_size=image_size,
    )
    device = choose_device(device_name)
    context = data_bundle.context
    model = build_model(model_factory, num_classes=context.num_classes, dataset=dataset).to(device)
    load_model_checkpoint(model, checkpoint_path, device)
    model.eval()

    preview_rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for inputs, targets in data_bundle.loaders[loader_name]:
            inputs = inputs.to(device)
            logits = model(inputs)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            predictions = np.argmax(probabilities, axis=1)
            for row_index, (target, prediction, probs) in enumerate(zip(targets.tolist(), predictions.tolist(), probabilities.tolist())):
                preview_rows.append(
                    {
                        "loader": loader_name,
                        "target_index": int(target),
                        "target_label": context.class_names[int(target)],
                        "predicted_index": int(prediction),
                        "predicted_label": context.class_names[int(prediction)],
                        "predicted_probability": float(max(probs)),
                        "probabilities": probs,
                    }
                )
                if len(preview_rows) >= sample_count:
                    return preview_rows
    return preview_rows


def _resolve_fanchuan_profile(dataset: str, profile: str | None) -> dict[str, float | int]:
    """Resolve a named FanchuanUnlearning profile, defaulting to the dataset name."""

    profile_name = dataset if profile is None else profile
    if profile_name not in FANCHUAN_UNLEARNING_PROFILES:
        raise ValueError(
            f"Unsupported FanchuanUnlearning profile '{profile_name}'. "
            f"Available profiles: {sorted(FANCHUAN_UNLEARNING_PROFILES)}"
        )
    return dict(FANCHUAN_UNLEARNING_PROFILES[profile_name])


def _resolve_scrub_profile(dataset: str, profile: str | None) -> dict[str, Any]:
    """Resolve a named SCRUB profile, defaulting to the dataset name."""

    profile_name = dataset if profile is None else profile
    if profile_name not in SCRUB_UNLEARNING_PROFILES:
        raise ValueError(
            f"Unsupported SCRUB profile '{profile_name}'. "
            f"Available profiles: {sorted(SCRUB_UNLEARNING_PROFILES)}"
        )
    return deepcopy(SCRUB_UNLEARNING_PROFILES[profile_name])


def _resolve_delete_profile(dataset: str, profile: str | None) -> dict[str, Any]:
    """Resolve a named DELETE profile, defaulting to the dataset name."""

    profile_name = dataset if profile is None else profile
    if profile_name not in DELETE_UNLEARNING_PROFILES:
        raise ValueError(
            f"Unsupported DELETE profile '{profile_name}'. "
            f"Available profiles: {sorted(DELETE_UNLEARNING_PROFILES)}"
        )
    return deepcopy(DELETE_UNLEARNING_PROFILES[profile_name])


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


def _fanchuan_uniform_kl_loss(logits: Any) -> Any:
    """Match model predictions to a uniform target distribution."""

    uniform_targets = torch.full_like(logits, 1.0 / float(logits.shape[1]))
    return torch.nn.KLDivLoss(reduction="batchmean")(torch.nn.LogSoftmax(dim=-1)(logits), uniform_targets)


def _fanchuan_contrastive_loss(forget_logits: Any, retain_logits: Any, *, temperature: float) -> Any:
    """Contrast forget logits against retain logits using the FanchuanUnlearning objective."""

    pairwise_scores = (forget_logits @ retain_logits.T) / float(temperature)
    return -torch.nn.LogSoftmax(dim=-1)(pairwise_scores).mean()


def _build_shuffled_loader(dataset_obj: Any, *, batch_size: int) -> Any:
    """Create a shuffled loader over an existing dataset object."""

    return torch.utils.data.DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
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


def _load_family_runtime_mean(family_dir: str | Path, *, num_models: int | None = None) -> float:
    """Load the mean runtime from a checkpoint family sidecar bank."""

    metadata_paths = sorted(Path(family_dir).glob("seed_*.json"))
    if not metadata_paths:
        raise FileNotFoundError(f"No checkpoint metadata found in {family_dir}")
    if num_models is not None:
        metadata_paths = metadata_paths[:num_models]
    runtimes = [
        float(json.loads(metadata_path.read_text(encoding="utf-8"))["runtime_seconds"])
        for metadata_path in metadata_paths
    ]
    return float(np.mean(runtimes))


def _build_fanchuan_efficiency_variants(
    profile_name: str,
    profile_config: dict[str, float | int],
) -> list[tuple[str, dict[str, float | int]]]:
    """Build a small quality-to-speed variant ladder for Fanchuan."""

    base = _normalize_fanchuan_hyperparameters(profile_config)
    epoch_candidates = [int(base["epochs"]), 4, 3, 2]
    variants: list[tuple[str, dict[str, float | int]]] = []
    seen_epochs: set[int] = set()
    for epochs in epoch_candidates:
        bounded_epochs = max(1, min(int(base["epochs"]), int(epochs)))
        if bounded_epochs in seen_epochs:
            continue
        seen_epochs.add(bounded_epochs)
        variant_name = profile_name if bounded_epochs == int(base["epochs"]) else f"{profile_name}_epochs_{bounded_epochs}"
        variant_config = dict(base)
        variant_config["epochs"] = bounded_epochs
        variants.append((variant_name, variant_config))
    return variants


def _build_scrub_efficiency_variants(
    profile_name: str,
    profile_config: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Build a small quality-to-speed variant ladder for SCRUB."""

    base = _normalize_scrub_hyperparameters(profile_config)
    epoch_candidates = [int(base["epochs"]), 8, 6, 4, 3, 2]
    variants: list[tuple[str, dict[str, Any]]] = []
    seen_epochs: set[int] = set()
    for epochs in epoch_candidates:
        bounded_epochs = max(2, min(int(base["epochs"]), int(epochs)))
        if bounded_epochs in seen_epochs:
            continue
        seen_epochs.add(bounded_epochs)
        variant_name = profile_name if bounded_epochs == int(base["epochs"]) else f"{profile_name}_epochs_{bounded_epochs}"
        variant_config = dict(base)
        variant_config["epochs"] = bounded_epochs
        variant_config["msteps"] = min(int(base["msteps"]), bounded_epochs - 1)
        variants.append((variant_name, variant_config))
    return variants


def _build_delete_efficiency_variants(
    profile_name: str,
    profile_config: dict[str, Any],
) -> list[tuple[str, dict[str, Any]]]:
    """Build a small quality-to-speed variant ladder for DELETE."""

    base = _normalize_delete_hyperparameters(profile_config)
    epoch_candidates = [int(base["epochs"]), 6, 4, 3, 2, 1]
    variants: list[tuple[str, dict[str, Any]]] = []
    seen_epochs: set[int] = set()
    for epochs in epoch_candidates:
        bounded_epochs = max(1, min(int(base["epochs"]), int(epochs)))
        if bounded_epochs in seen_epochs:
            continue
        seen_epochs.add(bounded_epochs)
        variant_name = profile_name if bounded_epochs == int(base["epochs"]) else f"{profile_name}_epochs_{bounded_epochs}"
        variant_config = dict(base)
        variant_config["epochs"] = bounded_epochs
        variants.append((variant_name, variant_config))
    return variants


def _select_efficiency_variant(
    *,
    algorithm_name: str,
    output_family_name: str,
    candidate_variants: list[tuple[str, dict[str, Any]]],
    reference_family_dir: str | Path,
    efficiency_ratio: float,
    trial_runner: Any,
) -> dict[str, Any]:
    """Pick the first quality-ordered variant that fits the runtime budget."""

    reference_runtime_mean = _load_family_runtime_mean(reference_family_dir)
    runtime_budget_seconds = float(efficiency_ratio) * float(reference_runtime_mean)
    trials: list[dict[str, Any]] = []
    selected_trial: dict[str, Any] | None = None
    fastest_trial: dict[str, Any] | None = None
    progress = tqdm(
        candidate_variants,
        desc=f"{algorithm_name} efficiency search ({output_family_name})",
        leave=False,
    )
    for variant_name, variant_config in progress:
        trial_metadata = trial_runner(variant_name, variant_config)
        trial = {
            "variant_name": variant_name,
            "runtime_seconds": float(trial_metadata["runtime_seconds"]),
            "best_val_accuracy": float(trial_metadata.get("best_val_accuracy", 0.0)),
            "checkpoint_path": trial_metadata.get("checkpoint_path"),
            "profile_config": variant_config,
        }
        trials.append(trial)
        if fastest_trial is None or trial["runtime_seconds"] < fastest_trial["runtime_seconds"]:
            fastest_trial = trial
        if hasattr(progress, "set_postfix"):
            progress.set_postfix(
                {
                    "variant": variant_name,
                    "runtime": f"{trial['runtime_seconds']:.2f}s",
                    "budget": f"{runtime_budget_seconds:.2f}s",
                }
            )
        if trial["runtime_seconds"] <= runtime_budget_seconds:
            selected_trial = trial
            break
    if selected_trial is None:
        if fastest_trial is None:
            raise ValueError(f"No efficiency-search trials produced for {algorithm_name}.")
        selected_trial = fastest_trial
    return {
        "algorithm": algorithm_name,
        "output_family_name": output_family_name,
        "efficiency_ratio": float(efficiency_ratio),
        "reference_runtime_mean": reference_runtime_mean,
        "runtime_budget_seconds": runtime_budget_seconds,
        "selected_variant": selected_trial["variant_name"],
        "selected_runtime_seconds": selected_trial["runtime_seconds"],
        "selected_best_val_accuracy": selected_trial["best_val_accuracy"],
        "selected_profile_config": deepcopy(selected_trial["profile_config"]),
        "passed_budget": selected_trial["runtime_seconds"] <= runtime_budget_seconds,
        "trials": trials,
    }


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
        student_log_probs = torch.nn.functional.log_softmax(student_model(inputs), dim=1)
        loss = criterion(student_log_probs, target_probs)
        loss.backward()
        optimizer.step()
        batch_size = targets.size(0)
        loss_sum += float(loss.item()) * batch_size
        sample_count += batch_size
    return 0.0 if sample_count == 0 else loss_sum / sample_count


def _run_scrub_unlearning_seed(
    *,
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
    device = choose_device(device_name)
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
    resolved_class_weighting = resolve_class_weighting(dataset, class_weighting)

    set_random_seed(seed)
    teacher_model = build_model(create_resnet18, num_classes=context.num_classes, dataset=dataset).to(device)
    student_model = build_model(create_resnet18, num_classes=context.num_classes, dataset=dataset).to(device)
    load_model_checkpoint(teacher_model, checkpoint_path, device)
    load_model_checkpoint(student_model, checkpoint_path, device)
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)

    forget_loader = _build_shuffled_loader(data_bundle.loaders["forget"].dataset, batch_size=forget_batch_size)
    retain_loader = _build_shuffled_loader(data_bundle.loaders["retain"].dataset, batch_size=retain_batch_size)
    val_loader = data_bundle.loaders["val"]
    class_counts = compute_split_class_counts(data_bundle, "retrain")
    retain_criterion = build_loss(
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
    wandb_run = init_wandb_run(
        enabled=use_wandb,
        entity="inmdev-university-of-british-columbia",
        project=resolve_wandb_project(dataset, wandb_project),
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
        val_accuracy = compute_accuracy(student_model, val_loader, device)
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


def _run_delete_unlearning_seed(
    *,
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
    device = choose_device(device_name)
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

    set_random_seed(seed)
    teacher_model = build_model(create_resnet18, num_classes=context.num_classes, dataset=dataset).to(device)
    student_model = build_model(create_resnet18, num_classes=context.num_classes, dataset=dataset).to(device)
    load_model_checkpoint(teacher_model, checkpoint_path, device)
    load_model_checkpoint(student_model, checkpoint_path, device)
    teacher_model.eval()
    for parameter in teacher_model.parameters():
        parameter.requires_grad_(False)

    forget_loader = _build_shuffled_loader(data_bundle.loaders["forget"].dataset, batch_size=batch_size)
    val_loader = data_bundle.loaders["val"]
    criterion = torch.nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.SGD(student_model.parameters(), lr=learning_rate, momentum=momentum)
    wandb_run = init_wandb_run(
        enabled=use_wandb,
        entity="inmdev-university-of-british-columbia",
        project=resolve_wandb_project(dataset, wandb_project),
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
        val_accuracy = compute_accuracy(student_model, val_loader, device)
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


def _run_fanchuan_unlearning_seed(
    *,
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
    device = choose_device(device_name)
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
    resolved_class_weighting = resolve_class_weighting(dataset, class_weighting)

    set_random_seed(seed)
    model = build_model(create_resnet18, num_classes=context.num_classes, dataset=dataset).to(device)
    load_model_checkpoint(model, checkpoint_path, device)

    forget_loader = _build_shuffled_loader(data_bundle.loaders["forget"].dataset, batch_size=loader_batch_size)
    retain_loader = _build_shuffled_loader(data_bundle.loaders["retain"].dataset, batch_size=retain_batch_size)
    retain_pair_loader = _build_shuffled_loader(data_bundle.loaders["retain"].dataset, batch_size=retain_batch_size)
    val_loader = data_bundle.loaders["val"]

    class_counts = compute_split_class_counts(data_bundle, "retrain")
    retain_criterion = build_loss(
        class_counts,
        context.num_classes,
        device=device,
        class_weighting=resolved_class_weighting,
    )
    uniform_optimizer = build_optimizer(
        model,
        lr=float(profile_config["uniform_stage_lr"]),
        momentum=momentum,
        weight_decay=float(profile_config["uniform_weight_decay"]),
    )
    forget_optimizer = build_optimizer(
        model,
        lr=float(profile_config["forget_stage_lr"]),
        momentum=momentum,
        weight_decay=float(profile_config["forget_weight_decay"]),
    )
    retain_optimizer = build_optimizer(
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

    wandb_run = init_wandb_run(
        enabled=use_wandb,
        entity="inmdev-university-of-british-columbia",
        project=resolve_wandb_project(dataset, wandb_project),
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

    stage_zero_val_accuracy = compute_accuracy(model, val_loader, device)
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

        val_accuracy = compute_accuracy(model, val_loader, device)
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

    require_torch()
    profile_name = dataset if profile is None else profile
    profile_config = _resolve_fanchuan_profile(dataset, profile_name)
    image_size = resolve_image_size(dataset, image_size)
    base_checkpoints = sorted(Path(base_family_dir).glob("seed_*.pth"))[:num_bank_seeds]
    if not base_checkpoints:
        raise FileNotFoundError(f"No base checkpoints found in {base_family_dir}")

    selection_summary: dict[str, Any] | None = None
    selected_profile_name = profile_name
    selected_profile_config = _normalize_fanchuan_hyperparameters(profile_config)
    if efficiency_aware:
        if reference_family_dir is None:
            raise ValueError("`reference_family_dir` is required when `efficiency_aware=True`.")
        candidate_variants = _build_fanchuan_efficiency_variants(profile_name, profile_config)
        selected_summary = _select_efficiency_variant(
            algorithm_name="FanchuanUnlearning",
            output_family_name=output_family_name,
            candidate_variants=candidate_variants,
            reference_family_dir=reference_family_dir,
            efficiency_ratio=efficiency_ratio,
            trial_runner=lambda variant_name, variant_config: _run_fanchuan_unlearning_seed(
                dataset=dataset,
                checkpoint_path=base_checkpoints[0],
                output_family_name=f"{output_family_name}__selection__{variant_name}",
                profile_name=variant_name,
                profile_config=variant_config,
                data_bundle=create_dataloaders_from_manifest(
                    dataset=dataset,
                    task_manifest=task_manifest,
                    samples_csv=samples_csv,
                    data_root=data_root,
                    batch_size=int(variant_config["loader_batch_size"]),
                    num_workers=0,
                    image_size=image_size,
                ),
                checkpoint_dir=checkpoint_dir,
                device_name=device_name,
                image_size=image_size,
                use_wandb=False,
                wandb_project=None,
                class_weighting=class_weighting,
                reuse_existing=True,
            ),
        )
        selection_summary = selected_summary
        selected_profile_name = str(selected_summary["selected_variant"])
        selected_profile_config = dict(selected_summary["selected_profile_config"])

    data_bundle = create_dataloaders_from_manifest(
        dataset=dataset,
        task_manifest=task_manifest,
        samples_csv=samples_csv,
        data_root=data_root,
        batch_size=int(selected_profile_config["loader_batch_size"]),
        num_workers=0,
        image_size=image_size,
    )

    outputs: list[dict[str, Any]] = []
    for checkpoint_path in base_checkpoints:
        outputs.append(
            _run_fanchuan_unlearning_seed(
                dataset=dataset,
                checkpoint_path=checkpoint_path,
                output_family_name=output_family_name,
                profile_name=selected_profile_name,
                profile_config=selected_profile_config,
                data_bundle=data_bundle,
                checkpoint_dir=checkpoint_dir,
                device_name=device_name,
                image_size=image_size,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
                class_weighting=class_weighting,
                reuse_existing=reuse_existing,
            )
        )

    return {
        "family_name": output_family_name,
        "seed_bank": outputs,
        "family_dir": str(Path(checkpoint_dir) / dataset / data_bundle.context.task_id / output_family_name),
        "efficiency_selection": selection_summary,
    }


run_second_place_unlearning_workflow = run_fanchuan_unlearning_workflow


def run_scrub_unlearning_workflow(
    *,
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

    require_torch()
    profile_name = dataset if profile is None else profile
    profile_config = _resolve_scrub_profile(dataset, profile_name)
    image_size = resolve_image_size(dataset, image_size)
    base_checkpoints = sorted(Path(base_family_dir).glob("seed_*.pth"))[:num_bank_seeds]
    if not base_checkpoints:
        raise FileNotFoundError(f"No base checkpoints found in {base_family_dir}")

    selection_summary: dict[str, Any] | None = None
    selected_profile_name = profile_name
    selected_profile_config = _normalize_scrub_hyperparameters(profile_config)
    if efficiency_aware:
        if reference_family_dir is None:
            raise ValueError("`reference_family_dir` is required when `efficiency_aware=True`.")
        candidate_variants = _build_scrub_efficiency_variants(profile_name, profile_config)
        selected_summary = _select_efficiency_variant(
            algorithm_name="SCRUB",
            output_family_name=output_family_name,
            candidate_variants=candidate_variants,
            reference_family_dir=reference_family_dir,
            efficiency_ratio=efficiency_ratio,
            trial_runner=lambda variant_name, variant_config: _run_scrub_unlearning_seed(
                dataset=dataset,
                checkpoint_path=base_checkpoints[0],
                output_family_name=f"{output_family_name}__selection__{variant_name}",
                profile_name=variant_name,
                profile_config=variant_config,
                data_bundle=create_dataloaders_from_manifest(
                    dataset=dataset,
                    task_manifest=task_manifest,
                    samples_csv=samples_csv,
                    data_root=data_root,
                    batch_size=max(
                        int(variant_config["forget_batch_size"]),
                        int(variant_config["retain_batch_size"]),
                    ),
                    num_workers=0,
                    image_size=image_size,
                ),
                checkpoint_dir=checkpoint_dir,
                device_name=device_name,
                image_size=image_size,
                use_wandb=False,
                wandb_project=None,
                class_weighting=class_weighting,
                reuse_existing=True,
            ),
        )
        selection_summary = selected_summary
        selected_profile_name = str(selected_summary["selected_variant"])
        selected_profile_config = dict(selected_summary["selected_profile_config"])

    data_bundle = create_dataloaders_from_manifest(
        dataset=dataset,
        task_manifest=task_manifest,
        samples_csv=samples_csv,
        data_root=data_root,
        batch_size=max(
            int(selected_profile_config["forget_batch_size"]),
            int(selected_profile_config["retain_batch_size"]),
        ),
        num_workers=0,
        image_size=image_size,
    )

    outputs: list[dict[str, Any]] = []
    checkpoint_iterator = tqdm(
        base_checkpoints,
        desc=f"SCRUB seed bank ({output_family_name})",
        leave=False,
    )
    for checkpoint_path in checkpoint_iterator:
        outputs.append(
            _run_scrub_unlearning_seed(
                dataset=dataset,
                checkpoint_path=checkpoint_path,
                output_family_name=output_family_name,
                profile_name=selected_profile_name,
                profile_config=selected_profile_config,
                data_bundle=data_bundle,
                checkpoint_dir=checkpoint_dir,
                device_name=device_name,
                image_size=image_size,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
                class_weighting=class_weighting,
                reuse_existing=reuse_existing,
            )
        )

    return {
        "family_name": output_family_name,
        "seed_bank": outputs,
        "family_dir": str(Path(checkpoint_dir) / dataset / data_bundle.context.task_id / output_family_name),
        "efficiency_selection": selection_summary,
    }


def run_delete_unlearning_workflow(
    *,
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

    require_torch()
    profile_name = dataset if profile is None else profile
    profile_config = _resolve_delete_profile(dataset, profile_name)
    image_size = resolve_image_size(dataset, image_size)
    base_checkpoints = sorted(Path(base_family_dir).glob("seed_*.pth"))[:num_bank_seeds]
    if not base_checkpoints:
        raise FileNotFoundError(f"No base checkpoints found in {base_family_dir}")

    selection_summary: dict[str, Any] | None = None
    selected_profile_name = profile_name
    selected_profile_config = _normalize_delete_hyperparameters(profile_config)
    if efficiency_aware:
        if reference_family_dir is None:
            raise ValueError("`reference_family_dir` is required when `efficiency_aware=True`.")
        candidate_variants = _build_delete_efficiency_variants(profile_name, profile_config)
        selected_summary = _select_efficiency_variant(
            algorithm_name="DELETE",
            output_family_name=output_family_name,
            candidate_variants=candidate_variants,
            reference_family_dir=reference_family_dir,
            efficiency_ratio=efficiency_ratio,
            trial_runner=lambda variant_name, variant_config: _run_delete_unlearning_seed(
                dataset=dataset,
                checkpoint_path=base_checkpoints[0],
                output_family_name=f"{output_family_name}__selection__{variant_name}",
                profile_name=variant_name,
                profile_config=variant_config,
                data_bundle=create_dataloaders_from_manifest(
                    dataset=dataset,
                    task_manifest=task_manifest,
                    samples_csv=samples_csv,
                    data_root=data_root,
                    batch_size=int(variant_config["batch_size"]),
                    num_workers=0,
                    image_size=image_size,
                ),
                checkpoint_dir=checkpoint_dir,
                device_name=device_name,
                image_size=image_size,
                use_wandb=False,
                wandb_project=None,
                reuse_existing=True,
            ),
        )
        selection_summary = selected_summary
        selected_profile_name = str(selected_summary["selected_variant"])
        selected_profile_config = dict(selected_summary["selected_profile_config"])

    data_bundle = create_dataloaders_from_manifest(
        dataset=dataset,
        task_manifest=task_manifest,
        samples_csv=samples_csv,
        data_root=data_root,
        batch_size=int(selected_profile_config["batch_size"]),
        num_workers=0,
        image_size=image_size,
    )

    outputs: list[dict[str, Any]] = []
    checkpoint_iterator = tqdm(
        base_checkpoints,
        desc=f"DELETE seed bank ({output_family_name})",
        leave=False,
    )
    for checkpoint_path in checkpoint_iterator:
        outputs.append(
            _run_delete_unlearning_seed(
                dataset=dataset,
                checkpoint_path=checkpoint_path,
                output_family_name=output_family_name,
                profile_name=selected_profile_name,
                profile_config=selected_profile_config,
                data_bundle=data_bundle,
                checkpoint_dir=checkpoint_dir,
                device_name=device_name,
                image_size=image_size,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
                reuse_existing=reuse_existing,
            )
        )

    return {
        "family_name": output_family_name,
        "seed_bank": outputs,
        "family_dir": str(Path(checkpoint_dir) / dataset / data_bundle.context.task_id / output_family_name),
        "efficiency_selection": selection_summary,
    }


def run_retain_finetune_placeholder(
    *,
    dataset: str,
    base_family_dir: str | Path,
    output_family_name: str = "placeholder_unlearn",
    num_bank_seeds: int = 3,
    epochs: int = 1,
    lr: float = 1e-3,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    batch_size: int = 128,
    class_weighting: str = "auto",
    image_size: int | None = None,
    checkpoint_dir: str | Path = "checkpoints",
    data_root: str | Path | None = None,
    task_manifest: str | Path | None = None,
    samples_csv: str | Path | None = None,
    device_name: str = "auto",
    use_wandb: bool = False,
    wandb_project: str | None = None,
) -> dict[str, Any]:
    """Placeholder unlearning by fine-tuning the full model on the retain split."""

    require_torch()
    model_factory = create_resnet18
    wandb_project = resolve_wandb_project(dataset, wandb_project)
    image_size = resolve_image_size(dataset, image_size)
    base_checkpoints = sorted(Path(base_family_dir).glob("seed_*.pth"))[:num_bank_seeds]
    if not base_checkpoints:
        raise FileNotFoundError(f"No base checkpoints found in {base_family_dir}")
    data_bundle = create_dataloaders_from_manifest(
        dataset=dataset,
        task_manifest=task_manifest,
        samples_csv=samples_csv,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=0,
        image_size=image_size,
    )
    outputs: list[dict[str, Any]] = []
    for checkpoint_path in base_checkpoints:
        metadata = json.loads(checkpoint_path.with_suffix(".json").read_text(encoding="utf-8"))
        seed = int(metadata["seed"])
        base_state = torch.load(checkpoint_path, map_location="cpu")
        _, new_metadata = fit_model(
            data_bundle=data_bundle,
            dataset=dataset,
            train_loader_name="retain",
            class_count_split_name="retrain",
            metadata_train_split="retain",
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            seed=seed,
            device_name=device_name,
            checkpoint_dir=checkpoint_dir,
            run_name=output_family_name,
            class_weighting=class_weighting,
            wandb_project=wandb_project,
            use_wandb=use_wandb,
            image_size=image_size,
            model_factory=model_factory,
            initial_state_dict=base_state,
            checkpoint_stem=f"seed_{seed}",
            metadata_extra={
                "source_checkpoint": str(checkpoint_path),
                "placeholder_algorithm": "retain_finetune",
            },
        )
        outputs.append(new_metadata)
    return {
        "family_name": output_family_name,
        "seed_bank": outputs,
        "family_dir": str(Path(checkpoint_dir) / dataset / data_bundle.context.task_id / output_family_name),
    }


def run_benchmark_notebook_workflow(
    *,
    dataset: str,
    checkpoint_dir: str | Path = "checkpoints",
    baseline_train_family: str = "baseline_train",
    baseline_retrain_family: str = "baseline_retrain",
    candidate_family_dirs: dict[str, str | Path] | None = None,
    efficiency_ratio: float = 0.2,
    data_root: str | Path | None = None,
    task_manifest: str | Path | None = None,
    samples_csv: str | Path | None = None,
    device_name: str = "auto",
    batch_size: int = 128,
    image_size: int | None = None,
    use_wandb: bool = False,
    wandb_project: str = "benchmark",
) -> dict[str, Any]:
    """Benchmark baseline families plus any additional candidate unlearning families."""

    samples_path, task_path, data_root_path = resolve_pipeline_paths(dataset, samples_csv, task_manifest, data_root)
    task_data = json.loads(Path(task_path).read_text(encoding="utf-8"))
    task_id = task_data["task_id"]
    family_dirs = {
        baseline_train_family: Path(checkpoint_dir) / dataset / task_id / baseline_train_family,
        baseline_retrain_family: Path(checkpoint_dir) / dataset / task_id / baseline_retrain_family,
    }
    if candidate_family_dirs:
        family_dirs.update({family_name: Path(directory) for family_name, directory in candidate_family_dirs.items()})
    benchmark = benchmark_model_families(
        dataset=dataset,
        family_dirs=family_dirs,
        reference_family=baseline_retrain_family,
        task_manifest=task_manifest,
        samples_csv=samples_csv,
        data_root=data_root,
        efficiency_ratio=efficiency_ratio,
        device_name=device_name,
        num_workers=0,
        batch_size=batch_size,
        image_size=image_size,
    )
    results_root = Path("results") / "benchmark" / dataset / task_id
    results_root.mkdir(parents=True, exist_ok=True)
    output_path = results_root / "benchmark_report.json"
    output_path.write_text(json.dumps(benchmark, indent=2), encoding="utf-8")
    benchmark["report_path"] = str(output_path)
    benchmark_config = {
        "dataset": dataset,
        "task_id": task_id,
        "reference_family": baseline_retrain_family,
        "family_names": list(family_dirs.keys()),
        "family_dirs": {name: str(path) for name, path in family_dirs.items()},
        "efficiency_ratio": float(efficiency_ratio),
        "device_name": device_name,
        "batch_size": int(batch_size),
        "image_size": int(resolve_image_size(dataset, image_size)),
        "report_path": str(output_path),
        "samples_csv": str(samples_path),
        "task_manifest": str(task_path),
        "data_root": str(data_root_path),
    }
    wandb_run = init_wandb_run(
        enabled=use_wandb,
        entity="inmdev-university-of-british-columbia",
        project=wandb_project,
        run_name=f"benchmark_{dataset}_{task_id}",
        config=benchmark_config,
    )
    benchmark_metrics: dict[str, Any] = {}
    for family_name, summary in benchmark["family_summaries"].items():
        for metric_name, metric_value in summary.items():
            benchmark_metrics[f"family/{family_name}/{metric_name}"] = metric_value
    for family_name, comparison in benchmark["comparisons_to_reference"].items():
        benchmark_metrics[f"comparison/{family_name}/forgetting_quality"] = comparison["forgetting_quality"]
        benchmark_metrics[f"comparison/{family_name}/passed_efficiency_cutoff"] = comparison["passed_efficiency_cutoff"]
        if comparison["raw_final_score"] is not None:
            benchmark_metrics[f"comparison/{family_name}/raw_final_score"] = comparison["raw_final_score"]
        if comparison["final_score"] is not None:
            benchmark_metrics[f"comparison/{family_name}/final_score"] = comparison["final_score"]
        benchmark_metrics[f"comparison/{family_name}/candidate_runtime_mean"] = comparison["runtime_seconds"]["candidate_mean"]
        benchmark_metrics[f"comparison/{family_name}/reference_runtime_mean"] = comparison["runtime_seconds"]["reference_mean"]
        benchmark_metrics[f"comparison/{family_name}/candidate_retain_accuracy"] = comparison["retain_accuracy"]["candidate_mean"]
        benchmark_metrics[f"comparison/{family_name}/reference_retain_accuracy"] = comparison["retain_accuracy"]["reference_mean"]
        benchmark_metrics[f"comparison/{family_name}/candidate_test_accuracy"] = comparison["test_accuracy"]["candidate_mean"]
        benchmark_metrics[f"comparison/{family_name}/reference_test_accuracy"] = comparison["test_accuracy"]["reference_mean"]
    wandb_run.log(benchmark_metrics)
    if use_wandb and wandb is not None:
        family_rows = [
            {"family": family_name, **summary}
            for family_name, summary in benchmark["family_summaries"].items()
        ]
        comparison_rows = [
            {
                "family": family_name,
                "forgetting_quality": comparison["forgetting_quality"],
                "passed_efficiency_cutoff": comparison["passed_efficiency_cutoff"],
                "raw_final_score": comparison["raw_final_score"],
                "final_score": comparison["final_score"],
                "candidate_runtime_mean": comparison["runtime_seconds"]["candidate_mean"],
                "reference_runtime_mean": comparison["runtime_seconds"]["reference_mean"],
                "candidate_retain_accuracy": comparison["retain_accuracy"]["candidate_mean"],
                "reference_retain_accuracy": comparison["retain_accuracy"]["reference_mean"],
                "candidate_test_accuracy": comparison["test_accuracy"]["candidate_mean"],
                "reference_test_accuracy": comparison["test_accuracy"]["reference_mean"],
            }
            for family_name, comparison in benchmark["comparisons_to_reference"].items()
        ]
        if family_rows:
            family_columns = list(family_rows[0].keys())
            family_table = wandb.Table(
                columns=family_columns,
                data=[[row[column] for column in family_columns] for row in family_rows],
            )
            wandb_run.log({"family_summaries_table": family_table})
        if comparison_rows:
            comparison_columns = list(comparison_rows[0].keys())
            comparison_table = wandb.Table(
                columns=comparison_columns,
                data=[[row[column] for column in comparison_columns] for row in comparison_rows],
            )
            wandb_run.log({"comparisons_table": comparison_table})
    wandb_run.finish()
    return benchmark
