"""Reusable benchmark helpers for CLI scripts and notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from tqdm.auto import tqdm

try:
    import torch
except ImportError:  # pragma: no cover - depends on local environment.
    torch = None

try:
    from Code.data_utils import create_dataloaders_from_manifest
    from Code.metrics import (
        _get_epsilons,
        collect_logits_and_targets,
        compute_forget_score_from_confs,
        compute_logit_scaled_confidence,
        compute_utility,
    )
    from Code.model_utils import build_model, choose_device, create_resnet18, load_model_checkpoint, resolve_image_size
except ImportError:  # pragma: no cover - allows direct module execution.
    from data_utils import create_dataloaders_from_manifest
    from metrics import (
        _get_epsilons,
        collect_logits_and_targets,
        compute_forget_score_from_confs,
        compute_logit_scaled_confidence,
        compute_utility,
    )
    from model_utils import build_model, choose_device, create_resnet18, load_model_checkpoint, resolve_image_size


def require_torch() -> None:
    if torch is None:  # pragma: no cover - depends on local environment.
        raise ImportError(
            "torch is required for checkpoint evaluation. "
            "Install dependencies from requirements.txt first."
        )


def list_checkpoints(directory: str | Path) -> list[Path]:
    """List `.pth` checkpoints in a directory."""

    checkpoint_paths = sorted(Path(directory).glob("*.pth"))
    if not checkpoint_paths:
        raise FileNotFoundError(f"No .pth checkpoints found in {directory}")
    return checkpoint_paths


def load_checkpoint_metadata(checkpoint_path: str | Path) -> dict[str, Any]:
    """Load the JSON sidecar next to a checkpoint."""

    metadata_path = Path(checkpoint_path).with_suffix(".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing sidecar metadata for {checkpoint_path}")
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def evaluate_checkpoint_bank(
    checkpoint_paths: list[Path],
    data_bundle: Any,
    device: Any,
    model_factory: Any = None,
) -> dict[str, Any]:
    """Evaluate one checkpoint bank on retain/test/forget splits."""

    require_torch()
    model_factory = model_factory or create_resnet18
    context = data_bundle.context
    retain_loader = data_bundle.loaders["retain"]
    test_loader = data_bundle.loaders["test"]
    forget_loader = data_bundle.loaders["forget"]
    retain_accuracies: list[float] = []
    test_accuracies: list[float] = []
    runtimes: list[float] = []
    confidence_rows: list[np.ndarray] = []
    expected_targets: np.ndarray | None = None

    for checkpoint_path in tqdm(checkpoint_paths, desc="Evaluating Checkpoints", leave=False):
        metadata = load_checkpoint_metadata(checkpoint_path)
        runtimes.append(float(metadata["runtime_seconds"]))
        model = build_model(model_factory, num_classes=context.num_classes, dataset=context.dataset).to(device)
        load_model_checkpoint(model, checkpoint_path, device)
        utility = compute_utility(model, retain_loader=retain_loader, test_loader=test_loader, device=device)
        retain_accuracies.append(float(utility["retain_accuracy"]))
        test_accuracies.append(float(utility["test_accuracy"]))
        logits, targets = collect_logits_and_targets(model, forget_loader, device)
        _, confidences = compute_logit_scaled_confidence(logits, targets)
        if expected_targets is None:
            expected_targets = targets
        elif not np.array_equal(expected_targets, targets):
            raise ValueError("Forget-set target ordering changed across checkpoint evaluations.")
        confidence_rows.append(confidences)

    return {
        "retain_accuracies": retain_accuracies,
        "test_accuracies": test_accuracies,
        "runtime_seconds": runtimes,
        "forget_confidences": np.stack(confidence_rows, axis=0),
    }


def summarize_bank_metrics(bank_metrics: dict[str, Any]) -> dict[str, Any]:
    """Produce per-family mean metrics from a bank evaluation."""

    return {
        "retain_accuracy_mean": float(np.mean(bank_metrics["retain_accuracies"])),
        "test_accuracy_mean": float(np.mean(bank_metrics["test_accuracies"])),
        "runtime_seconds_mean": float(np.mean(bank_metrics["runtime_seconds"])),
        "num_models": len(bank_metrics["retain_accuracies"]),
    }


def compare_candidate_to_reference(
    *,
    candidate_bank: dict[str, Any],
    reference_bank: dict[str, Any],
    efficiency_ratio: float,
) -> dict[str, Any]:
    """Compare one candidate bank against the retrained reference bank."""

    retrain_medians = np.median(reference_bank["forget_confidences"], axis=0)
    candidate_medians = np.median(candidate_bank["forget_confidences"], axis=0)
    retrain_is_positive = retrain_medians > candidate_medians
    pos_confs = np.where(
        retrain_is_positive,
        reference_bank["forget_confidences"],
        candidate_bank["forget_confidences"],
    )
    neg_confs = np.where(
        retrain_is_positive,
        candidate_bank["forget_confidences"],
        reference_bank["forget_confidences"],
    )
    epsilons = _get_epsilons(pos_confs=pos_confs, neg_confs=neg_confs)
    forgetting_quality = compute_forget_score_from_confs(
        unlearned_confs=candidate_bank["forget_confidences"],
        retrained_confs=reference_bank["forget_confidences"],
    )
    rar = float(np.mean(reference_bank["retain_accuracies"]))
    tar = float(np.mean(reference_bank["test_accuracies"]))
    rau = float(np.mean(candidate_bank["retain_accuracies"]))
    tau = float(np.mean(candidate_bank["test_accuracies"]))
    retrain_runtime_mean = float(np.mean(reference_bank["runtime_seconds"]))
    candidate_runtime_mean = float(np.mean(candidate_bank["runtime_seconds"]))
    passed_efficiency_cutoff = candidate_runtime_mean <= (efficiency_ratio * retrain_runtime_mean)
    final_score = None
    if passed_efficiency_cutoff and rar > 0.0 and tar > 0.0:
        # score = F * (rau / rar) * (tau / tar)
        final_score = forgetting_quality * (rau / rar) * (tau / tar)
    return {
        "forgetting_quality": forgetting_quality,
        "per_example_epsilons": epsilons,
        "retain_accuracy": {
            "candidate_mean": rau,
            "reference_mean": rar,
            "candidate_per_model": candidate_bank["retain_accuracies"],
            "reference_per_model": reference_bank["retain_accuracies"],
        },
        "test_accuracy": {
            "candidate_mean": tau,
            "reference_mean": tar,
            "candidate_per_model": candidate_bank["test_accuracies"],
            "reference_per_model": reference_bank["test_accuracies"],
        },
        "runtime_seconds": {
            "candidate_mean": candidate_runtime_mean,
            "reference_mean": retrain_runtime_mean,
            "candidate_per_model": candidate_bank["runtime_seconds"],
            "reference_per_model": reference_bank["runtime_seconds"],
        },
        "efficiency_ratio_threshold": efficiency_ratio,
        "passed_efficiency_cutoff": passed_efficiency_cutoff,
        "final_score": final_score,
    }


def benchmark_model_families(
    *,
    dataset: str,
    family_dirs: dict[str, str | Path],
    reference_family: str,
    task_manifest: str | Path | None = None,
    samples_csv: str | Path | None = None,
    data_root: str | Path | None = None,
    efficiency_ratio: float = 0.2,
    device_name: str = "auto",
    num_workers: int = 0,
    batch_size: int = 128,
    image_size: int | None = None,
    model_factory: Any = None,
) -> dict[str, Any]:
    """Benchmark multiple checkpoint families against one retrained reference."""

    require_torch()
    model_factory = model_factory or create_resnet18
    device = choose_device(device_name)
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
    bank_by_family: dict[str, dict[str, Any]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for family_name, directory in tqdm(list(family_dirs.items()), desc="Benchmarking Families"):
        checkpoint_paths = list_checkpoints(directory)
        bank = evaluate_checkpoint_bank(checkpoint_paths, data_bundle=data_bundle, device=device, model_factory=model_factory)
        bank_by_family[family_name] = bank
        summaries[family_name] = summarize_bank_metrics(bank)

    comparisons: dict[str, dict[str, Any]] = {}
    reference_bank = bank_by_family[reference_family]
    for family_name, bank in bank_by_family.items():
        if family_name == reference_family:
            continue
        comparisons[family_name] = compare_candidate_to_reference(
            candidate_bank=bank,
            reference_bank=reference_bank,
            efficiency_ratio=efficiency_ratio,
        )

    return {
        "dataset": dataset,
        "reference_family": reference_family,
        "family_summaries": summaries,
        "comparisons_to_reference": comparisons,
    }
