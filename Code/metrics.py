"""Utility and privacy metrics for the machine unlearning pipeline."""

from __future__ import annotations

import math
import time
from typing import Any, Callable

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - depends on local environment.
    torch = None


DELTA = 1e-5
EPSILON_CAP = 50.0
NUM_THRESHOLDS_PER_UNIT = 100
DOUBLE_THRESHOLD_BUFFER = 2.0
BUCKET_SIZE = 0.5


def _require_torch() -> None:
    if torch is None:  # pragma: no cover - depends on local environment.
        raise ImportError(
            "torch is required for model evaluation functions. "
            "Install dependencies from requirements.txt first."
        )


def measure_runtime(function: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, float]:
    """Execute `function` and return `(result, elapsed_seconds)`."""

    start = time.perf_counter()
    result = function(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return result, elapsed


def compute_accuracy(model: Any, loader: Any, device: Any) -> float:
    """Compute classification accuracy for a single model and loader."""

    _require_torch()
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            logits = model(inputs)
            predictions = torch.argmax(logits, dim=1)
            total += targets.numel()
            correct += (predictions == targets).sum().item()
    return 0.0 if total == 0 else float(correct / total)


def compute_utility(model: Any, retain_loader: Any, test_loader: Any, device: Any) -> dict[str, float]:
    """Compute retain and test accuracy for a single model."""

    retain_accuracy = compute_accuracy(model, retain_loader, device)
    test_accuracy = compute_accuracy(model, test_loader, device)
    return {
        "retain_accuracy": retain_accuracy,
        "test_accuracy": test_accuracy,
    }


def collect_logits_and_targets(model: Any, loader: Any, device: Any) -> tuple[np.ndarray, np.ndarray]:
    """Collect logits and targets for an entire loader in deterministic order."""

    _require_torch()
    model.eval()
    logits_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            logits = model(inputs).detach().cpu().numpy()
            logits_chunks.append(logits)
            target_chunks.append(targets.detach().cpu().numpy())

    if not logits_chunks:
        return np.empty((0, 0), dtype=np.float64), np.empty((0,), dtype=np.int64)
    return np.concatenate(logits_chunks, axis=0), np.concatenate(target_chunks, axis=0)


def compute_logit_scaled_confidence(logits: np.ndarray | Any, targets: np.ndarray | Any) -> tuple[np.ndarray, np.ndarray]:
    """Transform logits into the released Kaggle logit-scaled confidence values."""

    logits_np = np.asarray(logits, dtype=np.float64).copy()
    targets_np = np.asarray(targets, dtype=np.int64)
    count = logits_np.shape[0]
    if count == 0:
        return np.empty((0,), dtype=np.float64), np.empty((0,), dtype=np.float64)

    logits_np -= np.max(logits_np, axis=1, keepdims=True)
    probs = np.exp(logits_np, dtype=np.float64)
    probs /= np.sum(probs, axis=1, keepdims=True)
    prob_correct = probs[np.arange(count), targets_np[:count]]
    probs[np.arange(count), targets_np[:count]] = 0.0
    prob_wrong = np.sum(probs, axis=1)
    conf = np.log(prob_correct + 1e-45) - np.log(prob_wrong + 1e-45)
    return prob_correct, conf


def _get_single_threshold_rates(
    pos_confs: np.ndarray,
    neg_confs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float]]:
    all_confs = np.concatenate([pos_confs, neg_confs], axis=0)
    min_val = float(np.min(all_confs))
    max_val = float(np.max(all_confs))
    num_thresholds = max(1, int(np.ceil((max_val - min_val) * NUM_THRESHOLDS_PER_UNIT)))
    thresholds = np.linspace(min_val, max_val, num=num_thresholds)
    pos_matrix = pos_confs[:, None]
    neg_matrix = neg_confs[:, None]
    threshold_matrix = thresholds[None, :]
    tpr = np.mean(pos_matrix >= threshold_matrix, axis=0)
    fpr = np.mean(neg_matrix >= threshold_matrix, axis=0)
    tnr = np.mean(neg_matrix < threshold_matrix, axis=0)
    fnr = np.mean(pos_matrix < threshold_matrix, axis=0)
    return tpr, fnr, fpr, tnr, thresholds.tolist()


def _get_double_threshold_rates(
    pos_confs: np.ndarray,
    neg_confs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[float, float]]]:
    pos_diff = float(np.max(pos_confs) - np.min(pos_confs))
    neg_diff = float(np.max(neg_confs) - np.min(neg_confs))
    smallest = neg_diff if pos_diff > neg_diff else pos_diff
    pos_arr = np.asarray(pos_confs, dtype=np.float64)
    neg_arr = np.asarray(neg_confs, dtype=np.float64)

    if pos_diff >= neg_diff:
        pos_arr, neg_arr = neg_arr, pos_arr

    width = float(smallest)
    min_value = float(np.min(pos_arr) + width - DOUBLE_THRESHOLD_BUFFER)
    max_value = float(np.max(pos_arr) + DOUBLE_THRESHOLD_BUFFER)
    num_right_thresholds = max(1, int(np.ceil((max_value - min_value) * NUM_THRESHOLDS_PER_UNIT)))
    right_thresholds = np.linspace(min_value, max_value, num=num_right_thresholds)
    num_left_thresholds = max(1, int(np.ceil(2 * DOUBLE_THRESHOLD_BUFFER * NUM_THRESHOLDS_PER_UNIT)))
    left_thresholds = np.concatenate(
        [
            np.linspace(
                threshold - width - DOUBLE_THRESHOLD_BUFFER,
                threshold - width + DOUBLE_THRESHOLD_BUFFER,
                num=num_left_thresholds,
            )
            for threshold in right_thresholds
        ],
        axis=0,
    )
    right_thresholds_tiled = np.repeat(right_thresholds, num_left_thresholds)
    thresholds = list(zip(left_thresholds.tolist(), right_thresholds_tiled.tolist()))

    pos_matrix = pos_arr[:, None]
    neg_matrix = neg_arr[:, None]
    left_matrix = left_thresholds[None, :]
    right_matrix = right_thresholds_tiled[None, :]
    pos_positive = np.logical_and(left_matrix <= pos_matrix, pos_matrix <= right_matrix).astype(np.float32)
    neg_positive = np.logical_and(left_matrix <= neg_matrix, neg_matrix <= right_matrix).astype(np.float32)
    tpr = np.mean(pos_positive, axis=0)
    fnr = np.mean(1.0 - pos_positive, axis=0)
    fpr = np.mean(neg_positive, axis=0)
    tnr = np.mean(1.0 - neg_positive, axis=0)
    return tpr, fnr, fpr, tnr, thresholds


def _compute_example_epsilon(fprs: np.ndarray, fnrs: np.ndarray, delta: float = DELTA) -> float:
    """Compute one example epsilon from FPR/FNR lists following Algorithm 1."""

    per_attack_epsilons: list[float] = []
    for fpr, fnr in zip(fprs, fnrs):
        if fpr == 0.0 and fnr == 0.0:
            per_attack_epsilons.append(float("inf"))
            continue
        if fpr == 0.0 or fnr == 0.0:
            continue
        first = math.log(1 - delta - fpr) - math.log(fnr)
        second = math.log(1 - delta - fnr) - math.log(fpr)
        per_attack_epsilons.append(float(np.nanmax([first, second])))

    if not per_attack_epsilons:
        return float("nan")
    return float(np.nanmax(per_attack_epsilons))


def _get_epsilons(pos_confs: np.ndarray, neg_confs: np.ndarray, delta: float = DELTA) -> list[float]:
    """Compute per-example epsilons using the released single and double attacks."""

    epsilons: list[float] = []
    num_examples = pos_confs.shape[1]
    for index in range(num_examples):
        pos_example = np.asarray(pos_confs[:, index], dtype=np.float64).reshape(-1)
        neg_example = np.asarray(neg_confs[:, index], dtype=np.float64).reshape(-1)
        pos_diff = float(np.max(pos_example) - np.min(pos_example))
        neg_diff = float(np.max(neg_example) - np.min(neg_example))
        largest = max(pos_diff, neg_diff)
        smallest = min(pos_diff, neg_diff)
        if largest > 0.0 and (smallest / largest) < 0.01:
            epsilons.append(EPSILON_CAP)
            continue

        tpr_d, fnr_d, fpr_d, _tnr_d, _ = _get_double_threshold_rates(pos_example, neg_example)
        tpr_s, fnr_s, fpr_s, _tnr_s, _ = _get_single_threshold_rates(pos_example, neg_example)
        del tpr_d, tpr_s
        fprs = np.concatenate([fpr_d, fpr_s], axis=0)
        fnrs = np.concatenate([fnr_d, fnr_s], axis=0)
        epsilon = _compute_example_epsilon(fprs=fprs, fnrs=fnrs, delta=delta)
        if math.isnan(epsilon):
            epsilon = 0.0
        epsilons.append(float(np.clip(epsilon, 0.0, EPSILON_CAP)))
    return epsilons


def _score_epsilons(epsilons: list[float], num_models: int) -> float:
    if num_models < 2:
        raise ValueError("At least two models per bank are required to score forgetting quality.")

    max_epsilon = float(np.ceil(np.log(num_models - 1)))
    bucket_start = 0.0
    points = 1.0
    bucket_points: dict[float, float] = {}
    while bucket_start + BUCKET_SIZE <= max_epsilon:
        bucket_points[bucket_start] = points
        points /= 2.0
        bucket_start += BUCKET_SIZE

    total_score = 0.0
    for epsilon in epsilons:
        for start, bucket_score in bucket_points.items():
            if epsilon < start + BUCKET_SIZE:
                total_score += bucket_score
                break
    return total_score / len(epsilons) if epsilons else 0.0


def compute_forget_score_from_confs(unlearned_confs: np.ndarray, retrained_confs: np.ndarray) -> float:
    """Return the released Kaggle forgetting-quality score from confidence banks."""

    if unlearned_confs.shape != retrained_confs.shape:
        raise ValueError(
            "Unlearned and retrained confidence arrays must have identical shapes, "
            f"got {unlearned_confs.shape} and {retrained_confs.shape}."
        )

    retrain_medians = np.median(retrained_confs, axis=0)
    unlearned_medians = np.median(unlearned_confs, axis=0)
    retrain_is_positive = retrain_medians > unlearned_medians
    pos_confs = np.where(retrain_is_positive, retrained_confs, unlearned_confs)
    neg_confs = np.where(retrain_is_positive, unlearned_confs, retrained_confs)
    epsilons = _get_epsilons(pos_confs, neg_confs, delta=DELTA)
    return _score_epsilons(epsilons, num_models=unlearned_confs.shape[0])
