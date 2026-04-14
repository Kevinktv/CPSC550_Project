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

try:
    from Code.unlearning_algorithms import (
        CT_UNLEARNING_PROFILES as _PACKAGE_CT_UNLEARNING_PROFILES,
        DELETE_UNLEARNING_PROFILES as _PACKAGE_DELETE_UNLEARNING_PROFILES,
        FANCHUAN_UNLEARNING_PROFILES as _PACKAGE_FANCHUAN_UNLEARNING_PROFILES,
        MSG_UNLEARNING_PROFILES as _PACKAGE_MSG_UNLEARNING_PROFILES,
        SCRUB_UNLEARNING_PROFILES as _PACKAGE_SCRUB_UNLEARNING_PROFILES,
        _build_ct_efficiency_variants as _package_build_ct_efficiency_variants,
        _build_delete_efficiency_variants as _package_build_delete_efficiency_variants,
        _build_fanchuan_efficiency_variants as _package_build_fanchuan_efficiency_variants,
        _build_msg_efficiency_variants as _package_build_msg_efficiency_variants,
        _build_scrub_efficiency_variants as _package_build_scrub_efficiency_variants,
        _select_efficiency_variant as _package_select_efficiency_variant,
        run_ct_unlearning_workflow as _package_run_ct_unlearning_workflow,
        run_delete_unlearning_workflow as _package_run_delete_unlearning_workflow,
        run_fanchuan_unlearning_workflow as _package_run_fanchuan_unlearning_workflow,
        run_msg_unlearning_workflow as _package_run_msg_unlearning_workflow,
        run_scrub_unlearning_workflow as _package_run_scrub_unlearning_workflow,
    )
except ImportError:  # pragma: no cover - allows direct module execution.
    from unlearning_algorithms import (
        CT_UNLEARNING_PROFILES as _PACKAGE_CT_UNLEARNING_PROFILES,
        DELETE_UNLEARNING_PROFILES as _PACKAGE_DELETE_UNLEARNING_PROFILES,
        FANCHUAN_UNLEARNING_PROFILES as _PACKAGE_FANCHUAN_UNLEARNING_PROFILES,
        MSG_UNLEARNING_PROFILES as _PACKAGE_MSG_UNLEARNING_PROFILES,
        SCRUB_UNLEARNING_PROFILES as _PACKAGE_SCRUB_UNLEARNING_PROFILES,
        _build_ct_efficiency_variants as _package_build_ct_efficiency_variants,
        _build_delete_efficiency_variants as _package_build_delete_efficiency_variants,
        _build_fanchuan_efficiency_variants as _package_build_fanchuan_efficiency_variants,
        _build_msg_efficiency_variants as _package_build_msg_efficiency_variants,
        _build_scrub_efficiency_variants as _package_build_scrub_efficiency_variants,
        _select_efficiency_variant as _package_select_efficiency_variant,
        run_ct_unlearning_workflow as _package_run_ct_unlearning_workflow,
        run_delete_unlearning_workflow as _package_run_delete_unlearning_workflow,
        run_fanchuan_unlearning_workflow as _package_run_fanchuan_unlearning_workflow,
        run_msg_unlearning_workflow as _package_run_msg_unlearning_workflow,
        run_scrub_unlearning_workflow as _package_run_scrub_unlearning_workflow,
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


def _build_unlearning_runtime_deps() -> dict[str, Any]:
    """Collect notebook_workflows runtime dependencies for modular unlearning code."""

    return {
        "build_loss": build_loss,
        "build_model": build_model,
        "build_optimizer": build_optimizer,
        "choose_device": choose_device,
        "compute_accuracy": compute_accuracy,
        "compute_split_class_counts": compute_split_class_counts,
        "create_dataloaders_from_manifest": create_dataloaders_from_manifest,
        "create_resnet18": create_resnet18,
        "init_wandb_run": init_wandb_run,
        "load_model_checkpoint": load_model_checkpoint,
        "resolve_class_weighting": resolve_class_weighting,
        "resolve_image_size": resolve_image_size,
        "resolve_wandb_project": resolve_wandb_project,
        "set_random_seed": set_random_seed,
    }


# The unlearning source of truth now lives in Code/unlearning_algorithms/.
# These re-exports keep the notebook and tests stable while the implementation
# stays split into per-algorithm modules plus shared workflow helpers.
CT_UNLEARNING_PROFILES = _PACKAGE_CT_UNLEARNING_PROFILES
FANCHUAN_UNLEARNING_PROFILES = _PACKAGE_FANCHUAN_UNLEARNING_PROFILES
SCRUB_UNLEARNING_PROFILES = _PACKAGE_SCRUB_UNLEARNING_PROFILES
DELETE_UNLEARNING_PROFILES = _PACKAGE_DELETE_UNLEARNING_PROFILES
MSG_UNLEARNING_PROFILES = _PACKAGE_MSG_UNLEARNING_PROFILES
_build_ct_efficiency_variants = _package_build_ct_efficiency_variants
_build_fanchuan_efficiency_variants = _package_build_fanchuan_efficiency_variants
_build_scrub_efficiency_variants = _package_build_scrub_efficiency_variants
_build_delete_efficiency_variants = _package_build_delete_efficiency_variants
_build_msg_efficiency_variants = _package_build_msg_efficiency_variants
_select_efficiency_variant = _package_select_efficiency_variant


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
    """Compatibility wrapper around the modular Fanchuan workflow."""

    return _package_run_fanchuan_unlearning_workflow(
        deps=_build_unlearning_runtime_deps(),
        dataset=dataset,
        base_family_dir=base_family_dir,
        output_family_name=output_family_name,
        num_bank_seeds=num_bank_seeds,
        profile=profile,
        checkpoint_dir=checkpoint_dir,
        data_root=data_root,
        task_manifest=task_manifest,
        samples_csv=samples_csv,
        device_name=device_name,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        class_weighting=class_weighting,
        image_size=image_size,
        reuse_existing=reuse_existing,
        efficiency_aware=efficiency_aware,
        reference_family_dir=reference_family_dir,
        efficiency_ratio=efficiency_ratio,
    )


run_second_place_unlearning_workflow = run_fanchuan_unlearning_workflow


def run_msg_unlearning_workflow(
    *,
    dataset: str,
    base_family_dir: str | Path,
    output_family_name: str = "MSG",
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
    """Compatibility wrapper around the modular MSG workflow."""

    return _package_run_msg_unlearning_workflow(
        deps=_build_unlearning_runtime_deps(),
        dataset=dataset,
        base_family_dir=base_family_dir,
        output_family_name=output_family_name,
        num_bank_seeds=num_bank_seeds,
        profile=profile,
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
    """Compatibility wrapper around the modular SCRUB workflow."""

    return _package_run_scrub_unlearning_workflow(
        deps=_build_unlearning_runtime_deps(),
        dataset=dataset,
        base_family_dir=base_family_dir,
        output_family_name=output_family_name,
        num_bank_seeds=num_bank_seeds,
        profile=profile,
        checkpoint_dir=checkpoint_dir,
        data_root=data_root,
        task_manifest=task_manifest,
        samples_csv=samples_csv,
        device_name=device_name,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        class_weighting=class_weighting,
        image_size=image_size,
        reuse_existing=reuse_existing,
        efficiency_aware=efficiency_aware,
        reference_family_dir=reference_family_dir,
        efficiency_ratio=efficiency_ratio,
    )


def run_ct_unlearning_workflow(
    *,
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
    """Compatibility wrapper around the modular CT workflow."""

    return _package_run_ct_unlearning_workflow(
        deps=_build_unlearning_runtime_deps(),
        dataset=dataset,
        base_family_dir=base_family_dir,
        output_family_name=output_family_name,
        num_bank_seeds=num_bank_seeds,
        profile=profile,
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
    """Compatibility wrapper around the modular DELETE workflow."""

    return _package_run_delete_unlearning_workflow(
        deps=_build_unlearning_runtime_deps(),
        dataset=dataset,
        base_family_dir=base_family_dir,
        output_family_name=output_family_name,
        num_bank_seeds=num_bank_seeds,
        profile=profile,
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
