"""Evaluate checkpoint banks with the Kaggle-aligned unlearning scorer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from Code.benchmark_utils import compare_candidate_to_reference, evaluate_checkpoint_bank, list_checkpoints
    from Code.data_utils import create_dataloaders_from_manifest
    from Code.model_utils import choose_device, create_resnet18, resolve_image_size
except ImportError:  # pragma: no cover - allows `python Code/evaluate.py`.
    from benchmark_utils import compare_candidate_to_reference, evaluate_checkpoint_bank, list_checkpoints
    from data_utils import create_dataloaders_from_manifest
    from model_utils import choose_device, create_resnet18, resolve_image_size


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cifar10", "mufac"), required=True)
    parser.add_argument("--task-manifest", default=None)
    parser.add_argument("--samples-csv", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--retrained-dir", required=True)
    parser.add_argument("--unlearned-dir", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--efficiency-ratio", type=float, default=0.2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=None)
    return parser


def evaluate(
    args: argparse.Namespace,
    data_bundle: Any | None = None,
    model_factory: Any = None,
) -> dict[str, Any]:
    """Evaluate unlearned checkpoints against exact retraining baselines."""

    model_factory = model_factory or create_resnet18
    device = choose_device(args.device)
    image_size = resolve_image_size(args.dataset, args.image_size)
    data_bundle = data_bundle or create_dataloaders_from_manifest(
        dataset=args.dataset,
        task_manifest=args.task_manifest,
        samples_csv=args.samples_csv,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=image_size,
    )
    context = data_bundle.context
    retrained_paths = list_checkpoints(args.retrained_dir)
    unlearned_paths = list_checkpoints(args.unlearned_dir)
    if len(retrained_paths) != len(unlearned_paths):
        raise ValueError(
            "Checkpoint banks must have the same size, "
            f"got {len(retrained_paths)} retrained and {len(unlearned_paths)} unlearned."
        )

    retrained_bank = evaluate_checkpoint_bank(retrained_paths, data_bundle=data_bundle, device=device, model_factory=model_factory)
    unlearned_bank = evaluate_checkpoint_bank(unlearned_paths, data_bundle=data_bundle, device=device, model_factory=model_factory)
    comparison = compare_candidate_to_reference(
        candidate_bank=unlearned_bank,
        reference_bank=retrained_bank,
        efficiency_ratio=args.efficiency_ratio,
    )
    report = {
        "dataset": context.dataset,
        "task_id": context.task_id,
        "candidate_name": Path(args.unlearned_dir).name,
        "num_models": len(unlearned_paths),
        "forgetting_quality": comparison["forgetting_quality"],
        "per_example_epsilons": comparison["per_example_epsilons"],
        "retain_accuracy": {
            "unlearned_mean": comparison["retain_accuracy"]["candidate_mean"],
            "retrained_mean": comparison["retain_accuracy"]["reference_mean"],
            "unlearned_per_model": comparison["retain_accuracy"]["candidate_per_model"],
            "retrained_per_model": comparison["retain_accuracy"]["reference_per_model"],
        },
        "test_accuracy": {
            "unlearned_mean": comparison["test_accuracy"]["candidate_mean"],
            "retrained_mean": comparison["test_accuracy"]["reference_mean"],
            "unlearned_per_model": comparison["test_accuracy"]["candidate_per_model"],
            "retrained_per_model": comparison["test_accuracy"]["reference_per_model"],
        },
        "runtime_seconds": {
            "unlearned_mean": comparison["runtime_seconds"]["candidate_mean"],
            "retrained_mean": comparison["runtime_seconds"]["reference_mean"],
            "unlearned_per_model": comparison["runtime_seconds"]["candidate_per_model"],
            "retrained_per_model": comparison["runtime_seconds"]["reference_per_model"],
        },
        "efficiency_ratio_threshold": comparison["efficiency_ratio_threshold"],
        "passed_efficiency_cutoff": comparison["passed_efficiency_cutoff"],
        "final_score": comparison["final_score"],
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return evaluate(args)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint.
    main()
