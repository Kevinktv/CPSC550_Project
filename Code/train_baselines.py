"""Train exact baseline models on the full train split or the retrain split."""

from __future__ import annotations

import argparse
from typing import Any

try:
    from Code.data_utils import create_dataloaders_from_manifest
    from Code.training_utils import default_epochs, fit_model, resolve_wandb_project
except ImportError:  # pragma: no cover - allows `python Code/train_baselines.py`.
    from data_utils import create_dataloaders_from_manifest
    from training_utils import default_epochs, fit_model, resolve_wandb_project


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("cifar10", "mufac"), required=True)
    parser.add_argument("--task-manifest", default=None)
    parser.add_argument("--samples-csv", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--train-split", choices=("train", "retrain"), required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--wandb-project", default=None)
    parser.add_argument("--class-weighting", choices=("auto", "none", "inverse_freq"), default="auto")
    parser.add_argument("--image-size", type=int, default=None)
    return parser


def _resolve_run_name(args: argparse.Namespace) -> str:
    if args.run_name:
        return args.run_name
    return "baseline_retrain" if args.train_split == "retrain" else "baseline_train"


def train(args: argparse.Namespace, data_bundle: Any | None = None, model_factory: Any = None) -> dict[str, Any]:
    """Train one baseline model and return its checkpoint metadata."""

    epochs = args.epochs if args.epochs is not None else default_epochs(args.dataset)
    run_name = _resolve_run_name(args)
    data_bundle = data_bundle or create_dataloaders_from_manifest(
        dataset=args.dataset,
        task_manifest=args.task_manifest,
        samples_csv=args.samples_csv,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    _, metadata = fit_model(
        data_bundle=data_bundle,
        dataset=args.dataset,
        train_loader_name=args.train_split,
        class_count_split_name=args.train_split,
        metadata_train_split=args.train_split,
        batch_size=args.batch_size,
        epochs=epochs,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device_name=args.device,
        checkpoint_dir=args.checkpoint_dir,
        run_name=run_name,
        class_weighting=args.class_weighting,
        wandb_project=resolve_wandb_project(args.dataset, args.wandb_project),
        use_wandb=getattr(args, "use_wandb", True),
        image_size=args.image_size,
        model_factory=model_factory,
        metadata_extra=getattr(args, "metadata_extra", None),
    )
    return metadata


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    return train(args)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint.
    main()
