import contextlib
import io
import math
import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = REPO_ROOT / "Code"
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import benchmark_utils  # noqa: E402
import metrics  # noqa: E402


def _ref_get_single_threshold_rates(
    pos_confs: np.ndarray,
    neg_confs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    all_confs = np.concatenate([pos_confs, neg_confs], axis=0)
    min_val = float(np.min(all_confs))
    max_val = float(np.max(all_confs))
    num_thresholds = max(1, int(np.ceil((max_val - min_val) * metrics.NUM_THRESHOLDS_PER_UNIT)))
    thresholds = np.linspace(min_val, max_val, num=num_thresholds)
    pos_matrix = pos_confs[:, None]
    neg_matrix = neg_confs[:, None]
    threshold_matrix = thresholds[None, :]
    tpr = np.mean(pos_matrix >= threshold_matrix, axis=0)
    fpr = np.mean(neg_matrix >= threshold_matrix, axis=0)
    tnr = np.mean(neg_matrix < threshold_matrix, axis=0)
    fnr = np.mean(pos_matrix < threshold_matrix, axis=0)
    return tpr, fnr, fpr, tnr


def _ref_get_double_threshold_rates(
    pos_confs: np.ndarray,
    neg_confs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pos_diff = float(np.max(pos_confs) - np.min(pos_confs))
    neg_diff = float(np.max(neg_confs) - np.min(neg_confs))
    smallest = neg_diff if pos_diff > neg_diff else pos_diff
    pos_arr = np.asarray(pos_confs, dtype=np.float64)
    neg_arr = np.asarray(neg_confs, dtype=np.float64)

    if pos_diff >= neg_diff:
        pos_arr, neg_arr = neg_arr, pos_arr

    width = float(smallest)
    min_value = float(np.min(pos_arr) + width - metrics.DOUBLE_THRESHOLD_BUFFER)
    max_value = float(np.max(pos_arr) + metrics.DOUBLE_THRESHOLD_BUFFER)
    num_right_thresholds = max(1, int(np.ceil((max_value - min_value) * metrics.NUM_THRESHOLDS_PER_UNIT)))
    right_thresholds = np.linspace(min_value, max_value, num=num_right_thresholds)
    num_left_thresholds = max(1, int(np.ceil(2 * metrics.DOUBLE_THRESHOLD_BUFFER * metrics.NUM_THRESHOLDS_PER_UNIT)))
    left_thresholds = np.concatenate(
        [
            np.linspace(
                threshold - width - metrics.DOUBLE_THRESHOLD_BUFFER,
                threshold - width + metrics.DOUBLE_THRESHOLD_BUFFER,
                num=num_left_thresholds,
            )
            for threshold in right_thresholds
        ],
        axis=0,
    )
    right_thresholds_tiled = np.repeat(right_thresholds, num_left_thresholds)

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
    return tpr, fnr, fpr, tnr


def _ref_compute_example_epsilon(fprs: np.ndarray, fnrs: np.ndarray, delta: float = metrics.DELTA) -> float:
    per_attack_epsilons: list[float] = []
    for fpr, fnr in zip(fprs, fnrs):
        if fpr <= 0.0 and fnr <= 0.0:
            per_attack_epsilons.append(float("inf"))
            continue
        if fpr <= 0.0 or fnr <= 0.0 or (1 - delta - fpr) <= 0 or (1 - delta - fnr) <= 0:
            continue
        first = math.log(1 - delta - fpr) - math.log(fnr)
        second = math.log(1 - delta - fnr) - math.log(fpr)
        per_attack_epsilons.append(float(np.nanmax([first, second])))
    if not per_attack_epsilons:
        return float("nan")
    return float(np.nanmax(per_attack_epsilons))


def _ref_get_epsilons(
    pos_confs: np.ndarray,
    neg_confs: np.ndarray,
    delta: float = metrics.DELTA,
) -> list[float]:
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
            epsilons.append(metrics.EPSILON_CAP)
            continue

        tpr_d, fnr_d, fpr_d, _tnr_d = _ref_get_double_threshold_rates(pos_example, neg_example)
        tpr_s, fnr_s, fpr_s, _tnr_s = _ref_get_single_threshold_rates(pos_example, neg_example)
        del tpr_d, tpr_s
        fprs = np.concatenate([fpr_d, fpr_s], axis=0)
        fnrs = np.concatenate([fnr_d, fnr_s], axis=0)
        epsilon = _ref_compute_example_epsilon(fprs=fprs, fnrs=fnrs, delta=delta)
        if math.isnan(epsilon):
            epsilon = 0.0
        epsilons.append(float(np.clip(epsilon, 0.0, metrics.EPSILON_CAP)))
    return epsilons


def _ref_score_epsilons(epsilons: list[float], num_models: int) -> float:
    if num_models < 2:
        raise ValueError("At least two models per bank are required to score forgetting quality.")

    max_epsilon = float(np.ceil(np.log(num_models - 1)))
    bucket_start = 0.0
    bucket_index = 1
    bucket_points: dict[float, float] = {}
    while bucket_start + metrics.BUCKET_SIZE <= max_epsilon:
        bucket_points[bucket_start] = 2.0 / (2.0 ** bucket_index)
        bucket_index += 1
        bucket_start += metrics.BUCKET_SIZE

    total_score = 0.0
    for epsilon in epsilons:
        for start, bucket_score in bucket_points.items():
            if epsilon < start + metrics.BUCKET_SIZE:
                total_score += bucket_score
                break
    return total_score / len(epsilons) if epsilons else 0.0


def _ref_compute_forget_score_from_confs(
    unlearned_confs: np.ndarray,
    retrained_confs: np.ndarray,
) -> float:
    retrain_medians = np.median(retrained_confs, axis=0)
    unlearned_medians = np.median(unlearned_confs, axis=0)
    retrain_is_positive = retrain_medians > unlearned_medians
    pos_confs = np.where(retrain_is_positive, retrained_confs, unlearned_confs)
    neg_confs = np.where(retrain_is_positive, unlearned_confs, retrained_confs)
    epsilons = _ref_get_epsilons(pos_confs, neg_confs, delta=metrics.DELTA)
    return _ref_score_epsilons(epsilons, num_models=unlearned_confs.shape[0])


def _ref_compare_candidate_to_reference(
    candidate_bank: dict[str, object],
    reference_bank: dict[str, object],
    efficiency_ratio: float,
) -> dict[str, object]:
    candidate_confidences = np.asarray(candidate_bank["forget_confidences"], dtype=np.float64)
    reference_confidences = np.asarray(reference_bank["forget_confidences"], dtype=np.float64)
    retrain_medians = np.median(reference_confidences, axis=0)
    candidate_medians = np.median(candidate_confidences, axis=0)
    retrain_is_positive = retrain_medians > candidate_medians
    pos_confs = np.where(retrain_is_positive, reference_confidences, candidate_confidences)
    neg_confs = np.where(retrain_is_positive, candidate_confidences, reference_confidences)
    epsilons = _ref_get_epsilons(pos_confs=pos_confs, neg_confs=neg_confs)
    forgetting_quality = _ref_compute_forget_score_from_confs(
        unlearned_confs=candidate_confidences,
        retrained_confs=reference_confidences,
    )
    rar = float(np.mean(reference_bank["retain_accuracies"]))
    tar = float(np.mean(reference_bank["test_accuracies"]))
    rau = float(np.mean(candidate_bank["retain_accuracies"]))
    tau = float(np.mean(candidate_bank["test_accuracies"]))
    retrain_runtime_mean = float(np.mean(reference_bank["runtime_seconds"]))
    candidate_runtime_mean = float(np.mean(candidate_bank["runtime_seconds"]))
    passed_efficiency_cutoff = candidate_runtime_mean <= (efficiency_ratio * retrain_runtime_mean)
    raw_final_score = None
    if rar > 0.0 and tar > 0.0:
        raw_final_score = forgetting_quality * (rau / rar) * (tau / tar)
    final_score = raw_final_score if passed_efficiency_cutoff else None
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
        "raw_final_score": raw_final_score,
        "final_score": final_score,
    }


class MetricsExactnessTests(unittest.TestCase):
    def test_compute_example_epsilon_special_values(self) -> None:
        cases = [
            ("inf", np.array([0.0], dtype=np.float64), np.array([0.0], dtype=np.float64)),
            ("nan", np.array([1.0], dtype=np.float64), np.array([1.0], dtype=np.float64)),
            ("clipped", np.array([1e-30], dtype=np.float64), np.array([1e-30], dtype=np.float64)),
        ]
        for name, fprs, fnrs in cases:
            with self.subTest(case=name):
                expected = _ref_compute_example_epsilon(fprs, fnrs)
                actual = metrics._compute_example_epsilon(fprs, fnrs)
                if math.isnan(expected):
                    self.assertTrue(math.isnan(actual))
                else:
                    self.assertEqual(actual, expected)

    def test_get_epsilons_exact_across_cases(self) -> None:
        rng = np.random.default_rng(7)
        cases = {
            "random_small": (
                rng.normal(loc=0.0, scale=1.0, size=(3, 7)),
                rng.normal(loc=0.1, scale=1.1, size=(3, 7)),
            ),
            "all_equal": (
                np.full((3, 5), 0.25, dtype=np.float64),
                np.full((3, 5), 0.25, dtype=np.float64),
            ),
            "ratio_cap": (
                np.array(
                    [
                        [0.0, 0.30, 0.20, 0.15],
                        [100.0, 0.31, 0.22, 0.18],
                        [0.0, 0.29, 0.21, 0.17],
                    ],
                    dtype=np.float64,
                ),
                np.array(
                    [
                        [0.0, 0.33, 0.24, 0.11],
                        [0.5, 0.35, 0.19, 0.13],
                        [0.0, 0.32, 0.23, 0.12],
                    ],
                    dtype=np.float64,
                ),
            ),
        }
        for name, (pos_confs, neg_confs) in cases.items():
            with self.subTest(case=name):
                expected = _ref_get_epsilons(pos_confs, neg_confs)
                actual = metrics._get_epsilons(pos_confs, neg_confs, show_progress=False)
                self.assertEqual(actual, expected)

    def test_forgetting_quality_exact_across_cases(self) -> None:
        rng = np.random.default_rng(23)
        cases = {
            "random_small": (
                rng.normal(loc=0.05, scale=1.0, size=(3, 6)),
                rng.normal(loc=-0.10, scale=0.9, size=(3, 6)),
            ),
            "all_equal": (
                np.full((3, 4), 0.12, dtype=np.float64),
                np.full((3, 4), 0.12, dtype=np.float64),
            ),
            "ratio_cap": (
                np.array(
                    [
                        [0.0, 0.40, 0.42, 0.18],
                        [90.0, 0.41, 0.43, 0.19],
                        [0.0, 0.39, 0.44, 0.20],
                    ],
                    dtype=np.float64,
                ),
                np.array(
                    [
                        [0.0, 0.50, 0.52, 0.18],
                        [0.9, 0.49, 0.54, 0.17],
                        [0.0, 0.48, 0.51, 0.16],
                    ],
                    dtype=np.float64,
                ),
            ),
        }
        for name, (unlearned, retrained) in cases.items():
            with self.subTest(case=name):
                expected = _ref_compute_forget_score_from_confs(unlearned, retrained)
                actual = metrics.compute_forget_score_from_confs(unlearned, retrained)
                self.assertEqual(actual, expected)

    def test_compare_candidate_to_reference_payload_exact(self) -> None:
        candidate_bank = {
            "retain_accuracies": [0.82, 0.84, 0.83],
            "test_accuracies": [0.73, 0.74, 0.72],
            "runtime_seconds": [101.0, 103.0, 99.0],
            "forget_confidences": np.array(
                [
                    [0.20, 0.30, -0.20, 0.10, 0.40],
                    [0.18, 0.28, -0.10, 0.12, 0.41],
                    [0.19, 0.29, -0.15, 0.11, 0.39],
                ],
                dtype=np.float64,
            ),
        }
        reference_bank = {
            "retain_accuracies": [0.88, 0.87, 0.89],
            "test_accuracies": [0.79, 0.78, 0.80],
            "runtime_seconds": [400.0, 405.0, 395.0],
            "forget_confidences": np.array(
                [
                    [0.31, 0.15, -0.05, 0.25, 0.38],
                    [0.32, 0.17, -0.07, 0.24, 0.37],
                    [0.30, 0.16, -0.06, 0.23, 0.36],
                ],
                dtype=np.float64,
            ),
        }
        expected = _ref_compare_candidate_to_reference(
            candidate_bank=candidate_bank,
            reference_bank=reference_bank,
            efficiency_ratio=0.3,
        )
        with contextlib.redirect_stderr(io.StringIO()):
            actual = benchmark_utils.compare_candidate_to_reference(
                candidate_bank=candidate_bank,
                reference_bank=reference_bank,
                efficiency_ratio=0.3,
            )
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
