"""Shared helpers for notebook benchmark unlearning algorithms."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - depends on local environment.
    def tqdm(iterable: Any, *args: Any, **kwargs: Any) -> Any:
        return iterable

try:
    import torch
except ImportError:  # pragma: no cover - depends on local environment.
    torch = None


def require_torch() -> None:
    """Fail fast when torch-backed training is requested without torch installed."""

    if torch is None:  # pragma: no cover - depends on local environment.
        raise ImportError(
            "torch is required for notebook training and benchmarking. "
            "Install dependencies from requirements.txt first."
        )


def build_shuffled_loader(dataset_obj: Any, *, batch_size: int) -> Any:
    """Create a small single-process shuffled loader over an existing dataset object."""

    require_torch()
    return torch.utils.data.DataLoader(
        dataset_obj,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )


def resolve_unlearning_profile(
    *,
    dataset: str,
    profile: str | None,
    profiles: dict[str, dict[str, Any]],
    algorithm_name: str,
) -> tuple[str, dict[str, Any]]:
    """Resolve a dataset-default profile and return a detached config copy."""

    profile_name = dataset if profile is None else profile
    if profile_name not in profiles:
        raise ValueError(
            f"Unsupported {algorithm_name} profile '{profile_name}'. "
            f"Available profiles: {sorted(profiles)}"
        )
    return profile_name, deepcopy(profiles[profile_name])


def load_family_runtime_mean(family_dir: str | Path, *, num_models: int | None = None) -> float:
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


def build_epoch_efficiency_variants(
    *,
    profile_name: str,
    profile_config: dict[str, Any],
    epoch_candidates: list[int],
    min_epochs: int,
    postprocess_variant: Any | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Create a small epoch-only speed ladder while keeping other knobs fixed."""

    base = deepcopy(profile_config)
    base_epochs = int(base["epochs"])
    variants: list[tuple[str, dict[str, Any]]] = []
    seen_epochs: set[int] = set()
    for epochs in epoch_candidates:
        bounded_epochs = max(int(min_epochs), min(base_epochs, int(epochs)))
        if bounded_epochs in seen_epochs:
            continue
        seen_epochs.add(bounded_epochs)
        variant_name = profile_name if bounded_epochs == base_epochs else f"{profile_name}_epochs_{bounded_epochs}"
        variant_config = dict(base)
        variant_config["epochs"] = bounded_epochs
        if postprocess_variant is not None:
            variant_config = dict(postprocess_variant(variant_config))
        variants.append((variant_name, variant_config))
    return variants


def select_efficiency_variant(
    *,
    algorithm_name: str,
    output_family_name: str,
    candidate_variants: list[tuple[str, dict[str, Any]]],
    reference_family_dir: str | Path,
    efficiency_ratio: float,
    trial_runner: Any,
) -> dict[str, Any]:
    """Pick the first quality-ordered variant that fits the runtime budget."""

    reference_runtime_mean = load_family_runtime_mean(reference_family_dir)
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


def resolve_checkpoint_bank(base_family_dir: str | Path, *, num_bank_seeds: int) -> list[Path]:
    """Load the baseline checkpoint bank that an unlearning method will transform."""

    checkpoints = sorted(Path(base_family_dir).glob("seed_*.pth"))[:num_bank_seeds]
    if not checkpoints:
        raise FileNotFoundError(f"No base checkpoints found in {base_family_dir}")
    return checkpoints


def create_unlearning_data_bundle(
    *,
    deps: dict[str, Any],
    dataset: str,
    task_manifest: str | Path | None,
    samples_csv: str | Path | None,
    data_root: str | Path | None,
    batch_size: int,
    image_size: int,
) -> Any:
    """Build one manifest-backed data bundle for an unlearning run or trial."""

    return deps["create_dataloaders_from_manifest"](
        dataset=dataset,
        task_manifest=task_manifest,
        samples_csv=samples_csv,
        data_root=data_root,
        batch_size=batch_size,
        num_workers=0,
        image_size=image_size,
    )


def run_unlearning_seed_bank(
    *,
    algorithm_name: str,
    base_checkpoints: list[Path],
    seed_runner: Any,
    dataset: str,
    output_family_name: str,
    profile_name: str,
    profile_config: dict[str, Any],
    data_bundle: Any,
    checkpoint_dir: str | Path,
    device_name: str,
    image_size: int,
    use_wandb: bool,
    wandb_project: str | None,
    reuse_existing: bool,
    deps: dict[str, Any],
    seed_runner_kwargs: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Apply one unlearning algorithm to every checkpoint in the source bank."""

    outputs: list[dict[str, Any]] = []
    checkpoint_iterator = tqdm(
        base_checkpoints,
        desc=f"{algorithm_name} seed bank ({output_family_name})",
        leave=False,
    )
    extra_kwargs = {} if seed_runner_kwargs is None else dict(seed_runner_kwargs)
    for checkpoint_path in checkpoint_iterator:
        outputs.append(
            seed_runner(
                dataset=dataset,
                checkpoint_path=checkpoint_path,
                output_family_name=output_family_name,
                profile_name=profile_name,
                profile_config=profile_config,
                data_bundle=data_bundle,
                checkpoint_dir=checkpoint_dir,
                device_name=device_name,
                image_size=image_size,
                use_wandb=use_wandb,
                wandb_project=wandb_project,
                reuse_existing=reuse_existing,
                deps=deps,
                **extra_kwargs,
            )
        )
    return outputs


def run_unlearning_workflow_bank(
    *,
    deps: dict[str, Any],
    algorithm_name: str,
    dataset: str,
    base_family_dir: str | Path,
    output_family_name: str,
    num_bank_seeds: int,
    profile_name: str,
    profile_config: dict[str, Any],
    normalize_hyperparameters: Any,
    build_efficiency_variants: Any,
    resolve_bundle_batch_size: Any,
    seed_runner: Any,
    checkpoint_dir: str | Path,
    data_root: str | Path | None,
    task_manifest: str | Path | None,
    samples_csv: str | Path | None,
    device_name: str,
    use_wandb: bool,
    wandb_project: str | None,
    image_size: int | None,
    reuse_existing: bool,
    efficiency_aware: bool,
    reference_family_dir: str | Path | None,
    efficiency_ratio: float,
    seed_runner_kwargs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a checkpoint-bank workflow shared by all notebook unlearning algorithms."""

    require_torch()
    resolved_image_size = deps["resolve_image_size"](dataset, image_size)
    base_checkpoints = resolve_checkpoint_bank(base_family_dir, num_bank_seeds=num_bank_seeds)

    selection_summary: dict[str, Any] | None = None
    selected_profile_name = profile_name
    selected_profile_config = dict(normalize_hyperparameters(profile_config))

    def build_data_bundle_for_profile(candidate_profile_config: dict[str, Any]) -> Any:
        return create_unlearning_data_bundle(
            deps=deps,
            dataset=dataset,
            task_manifest=task_manifest,
            samples_csv=samples_csv,
            data_root=data_root,
            batch_size=int(resolve_bundle_batch_size(candidate_profile_config)),
            image_size=resolved_image_size,
        )

    runner_kwargs = {} if seed_runner_kwargs is None else dict(seed_runner_kwargs)
    if efficiency_aware:
        if reference_family_dir is None:
            raise ValueError("`reference_family_dir` is required when `efficiency_aware=True`.")
        candidate_variants = build_efficiency_variants(profile_name, profile_config)
        selection_summary = select_efficiency_variant(
            algorithm_name=algorithm_name,
            output_family_name=output_family_name,
            candidate_variants=candidate_variants,
            reference_family_dir=reference_family_dir,
            efficiency_ratio=efficiency_ratio,
            trial_runner=lambda variant_name, variant_config: seed_runner(
                dataset=dataset,
                checkpoint_path=base_checkpoints[0],
                output_family_name=f"{output_family_name}__selection__{variant_name}",
                profile_name=variant_name,
                profile_config=variant_config,
                data_bundle=build_data_bundle_for_profile(variant_config),
                checkpoint_dir=checkpoint_dir,
                device_name=device_name,
                image_size=resolved_image_size,
                use_wandb=False,
                wandb_project=None,
                reuse_existing=True,
                deps=deps,
                **runner_kwargs,
            ),
        )
        selected_profile_name = str(selection_summary["selected_variant"])
        selected_profile_config = dict(selection_summary["selected_profile_config"])

    data_bundle = build_data_bundle_for_profile(selected_profile_config)
    outputs = run_unlearning_seed_bank(
        algorithm_name=algorithm_name,
        base_checkpoints=base_checkpoints,
        seed_runner=seed_runner,
        dataset=dataset,
        output_family_name=output_family_name,
        profile_name=selected_profile_name,
        profile_config=selected_profile_config,
        data_bundle=data_bundle,
        checkpoint_dir=checkpoint_dir,
        device_name=device_name,
        image_size=resolved_image_size,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        reuse_existing=reuse_existing,
        deps=deps,
        seed_runner_kwargs=runner_kwargs,
    )
    return {
        "family_name": output_family_name,
        "seed_bank": outputs,
        "family_dir": str(Path(checkpoint_dir) / dataset / data_bundle.context.task_id / output_family_name),
        "efficiency_selection": selection_summary,
    }
