"""Generate canonical deterministic split manifests for CIFAR-10 and MUFAC.

This generator builds one mixed-label unlearning task per dataset with the
following partitions:

- `train`: canonical training split used to define the unlearning task.
- `forget`: deterministic subset of `train` that can contain multiple labels.
- `retrain`: `train \\ forget`, used for retraining from scratch.
- `val` / `test`: held-out evaluation sets.

For CIFAR-10 the canonical training split is the official training set after
removing a deterministic per-class validation slice. For MUFAC the canonical
training split is `custom_train_dataset.csv`.

The forget subset can either draw from all labels or be restricted to the top-k
most frequent labels in the training split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import pickle
import shutil
import tarfile
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]
SAMPLE_FIELDS = [
    "sample_id",
    "dataset",
    "source_partition",
    "label_field",
    "label_id",
    "label_name",
    "raw_index",
    "relative_path",
]
SUPPORTED_DATASETS = ("cifar10", "mufac")


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for split generation and reproducibility checks."""
    parser = argparse.ArgumentParser(
        description="Generate canonical deterministic split manifests for CIFAR-10 and MUFAC."
    )

    script_dir = Path(__file__).resolve().parent
    
    parser.add_argument(
        "--data-root",
        type=Path,
        default=script_dir.parent / "data",
        help="Root directory that contains cifar-10 and MUFAC.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=script_dir,
        help="Output directory for generated split artifacts.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=SUPPORTED_DATASETS,
        default=list(SUPPORTED_DATASETS),
        help="Datasets to generate or verify.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Regenerate into a temporary directory and compare against --out-root.",
    )
    parser.add_argument(
        "--forget-percentage",
        type=float,
        default=None,
        help=(
            "Percentage of the canonical train split to place in the mixed forget set. "
            "If omitted, the script uses 100 / number_of_labels for each dataset."
        ),
    )
    parser.add_argument(
        "--forget-top-k-classes",
        type=int,
        default=None,
        help=(
            "Restrict forgetting to the top-k most frequent labels in the train split. "
            "If omitted, the forget set can draw from all labels."
        ),
    )
    return parser.parse_args()


def stable_sha256(value: Any) -> str:
    """Return a deterministic SHA256 hash for nested JSON-like data.

    The helper normalizes dict key order, path formatting, and set ordering so
    fingerprints stay stable across runs and machines.
    """

    def normalize(obj: Any) -> Any:
        if isinstance(obj, Path):
            return obj.as_posix()
        if isinstance(obj, dict):
            return {str(key): normalize(obj[key]) for key in sorted(obj)}
        if isinstance(obj, (list, tuple)):
            return [normalize(item) for item in obj]
        if isinstance(obj, set):
            return [normalize(item) for item in sorted(obj)]
        return obj

    payload = json.dumps(
        normalize(value),
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ensure_unique_ids(rows: list[dict[str, Any]], dataset: str) -> None:
    """Fail fast if the generated sample IDs are not unique."""
    sample_ids = [row["sample_id"] for row in rows]
    counts = Counter(sample_ids)
    duplicates = [sample_id for sample_id, count in counts.items() if count > 1]
    if duplicates:
        preview = ", ".join(duplicates[:5])
        raise ValueError(f"{dataset}: duplicate sample IDs detected: {preview}")


def write_samples_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write the dataset-wide sample catalog in canonical column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SAMPLE_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SAMPLE_FIELDS})


def write_task_manifest(task: dict[str, Any], path: Path) -> None:
    """Write one task manifest as pretty-printed stable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(task, handle, indent=2, sort_keys=True)
        handle.write("\n")


def validate_forget_percentage(forget_percentage: float | None) -> float | None:
    """Validate an optional forget percentage supplied by the user."""
    if forget_percentage is None:
        return None
    if not 0 < forget_percentage < 100:
        raise ValueError(f"--forget-percentage must be between 0 and 100, got {forget_percentage}")
    return forget_percentage


def validate_forget_top_k(forget_top_k_classes: int | None) -> int | None:
    """Validate an optional top-k restriction for the forget label scope."""
    if forget_top_k_classes is None:
        return None
    if forget_top_k_classes <= 0:
        raise ValueError(
            f"--forget-top-k-classes must be a positive integer, got {forget_top_k_classes}"
        )
    return forget_top_k_classes


def count_labels(sample_ids: list[str], sample_lookup: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Count labels for a split and return them in sorted label order."""
    counts = Counter(sample_lookup[sample_id]["label_name"] for sample_id in sample_ids)
    return dict(sorted(counts.items()))


def resolve_mixed_forget_count(
    label_to_ids: dict[str, list[str]], forget_percentage: float | None
) -> tuple[int, float]:
    """Choose a mixed-label forget size from an explicit or dataset-derived percentage."""
    total_count = sum(len(ids) for ids in label_to_ids.values())
    label_count = len(label_to_ids)
    if total_count == 0 or label_count == 0:
        raise ValueError("Cannot define a mixed forget split from an empty training partition")

    effective_percentage = (
        forget_percentage if forget_percentage is not None else (100.0 / label_count)
    )
    forget_count = max(1, round(total_count * (effective_percentage / 100.0)))
    if total_count > 1:
        forget_count = min(forget_count, total_count - 1)
    return forget_count, effective_percentage


def allocate_proportional_counts(
    label_to_ids: dict[str, list[str]], target_count: int
) -> dict[str, int]:
    """Allocate a global sample count across labels proportionally and deterministically."""
    total_count = sum(len(ids) for ids in label_to_ids.values())
    if target_count < 0 or target_count > total_count:
        raise ValueError(f"Requested {target_count} forget samples from {total_count} training items")

    allocations: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []
    assigned = 0
    for label_name in sorted(label_to_ids):
        capacity = len(label_to_ids[label_name])
        raw = (target_count * capacity) / total_count
        base = min(capacity, int(raw))
        allocations[label_name] = base
        assigned += base
        remainders.append((raw - base, label_name))

    remaining = target_count - assigned
    for _, label_name in sorted(remainders, key=lambda item: (-item[0], item[1])):
        if remaining == 0:
            break
        if allocations[label_name] < len(label_to_ids[label_name]):
            allocations[label_name] += 1
            remaining -= 1

    if remaining:
        for label_name in sorted(label_to_ids):
            while remaining and allocations[label_name] < len(label_to_ids[label_name]):
                allocations[label_name] += 1
                remaining -= 1

    if remaining:
        raise ValueError("Unable to allocate the requested mixed forget count")

    return allocations


def _svg_text(x: float, y: float, text: str, extra: str = "") -> str:
    """Return a single SVG text node."""
    return f'<text x="{x:.2f}" y="{y:.2f}" {extra}>{html.escape(text)}</text>'


def display_dataset_name(dataset: str) -> str:
    """Return a reader-friendly dataset name for plot titles."""
    names = {
        "cifar10": "CIFAR-10",
        "mufac": "MUFAC",
    }
    return names.get(dataset, dataset)


def render_grouped_bar_chart(
    *,
    title: str,
    labels: list[str],
    series: dict[str, dict[str, int]],
    colors: dict[str, str],
    width: int = 700,
    height: int = 420,
) -> str:
    """Render one grouped bar chart as an SVG fragment."""
    margin_left = 60
    margin_right = 20
    margin_top = 55
    margin_bottom = 60
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_value = max(
        1,
        max(value for distribution in series.values() for value in distribution.values()),
    )

    ticks = 5
    category_width = plot_width / max(1, len(labels))
    group_width = category_width * 0.72
    bar_width = group_width / max(1, len(series))
    group_left_offset = (category_width - group_width) / 2
    label_font_size = 9 if max((len(label) for label in labels), default=0) > 6 else 11

    elements = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="white" />',
        _svg_text(
            width / 2,
            28,
            title,
            'text-anchor="middle" font-size="18" font-family="Arial, sans-serif" font-weight="700"',
        ),
    ]

    for tick in range(ticks + 1):
        value = round(max_value * tick / ticks)
        y = margin_top + plot_height - (plot_height * tick / ticks)
        elements.append(
            f'<line x1="{margin_left}" y1="{y:.2f}" x2="{width - margin_right}" y2="{y:.2f}" '
            'stroke="#d9d9d9" stroke-width="1" />'
        )
        elements.append(
            _svg_text(
                margin_left - 8,
                y + 4,
                str(value),
                'text-anchor="end" font-size="11" font-family="Arial, sans-serif" fill="#555"',
            )
        )

    axis_bottom = margin_top + plot_height
    elements.append(
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{axis_bottom}" '
        'stroke="#222" stroke-width="1.5" />'
    )
    elements.append(
        f'<line x1="{margin_left}" y1="{axis_bottom}" x2="{width - margin_right}" y2="{axis_bottom}" '
        'stroke="#222" stroke-width="1.5" />'
    )
    elements.append(
        _svg_text(
            margin_left - 45,
            margin_top + plot_height / 2,
            "Count",
            'text-anchor="middle" font-size="12" font-family="Arial, sans-serif" '
            'transform="rotate(-90, 15, 200)" fill="#333"',
        )
    )
    elements.append(
        _svg_text(
            margin_left + plot_width / 2,
            height - 18,
            "Class",
            'text-anchor="middle" font-size="12" font-family="Arial, sans-serif" fill="#333"',
        )
    )

    series_names = list(series)
    for label_index, label in enumerate(labels):
        category_x = margin_left + (label_index * category_width)
        for series_index, series_name in enumerate(series_names):
            value = series[series_name].get(label, 0)
            bar_height = 0 if max_value == 0 else (value / max_value) * plot_height
            x = category_x + group_left_offset + (series_index * bar_width)
            y = axis_bottom - bar_height
            elements.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width - 2:.2f}" height="{bar_height:.2f}" '
                f'fill="{colors[series_name]}" />'
            )

        label_x = category_x + (category_width / 2)
        elements.append(
            f'<text x="{label_x:.2f}" y="{axis_bottom + 18:.2f}" '
            f'font-size="{label_font_size}" font-family="Arial, sans-serif" fill="#333" '
            f'text-anchor="middle">{html.escape(label)}</text>'
        )

    legend_x = width - margin_right - 140
    legend_y = 18
    for legend_index, series_name in enumerate(series_names):
        y = legend_y + (legend_index * 20)
        elements.append(
            f'<rect x="{legend_x}" y="{y}" width="12" height="12" fill="{colors[series_name]}" />'
        )
        elements.append(
            _svg_text(
                legend_x + 18,
                y + 11,
                series_name,
                'font-size="11" font-family="Arial, sans-serif" fill="#333"',
            )
        )

    return "\n".join(elements)


def write_task_histogram_svg(task: dict[str, Any], project_root: Path) -> str:
    """Write a side-by-side SVG histogram summary for one task."""
    figures_root = project_root / "Report" / "figures"
    figures_root.mkdir(parents=True, exist_ok=True)
    filename = f"{task['dataset']}_{task['task_id']}_histograms.svg"
    relative_path = f"Report/figures/{filename}"
    output_path = figures_root / filename

    labels = list(task["label_distributions"]["train"])
    dataset_name = display_dataset_name(task["dataset"])
    left_svg = render_grouped_bar_chart(
        title=f"{dataset_name}: Train / Val / Test",
        labels=labels,
        series={
            "train": task["label_distributions"]["train"],
            "val": task["label_distributions"]["val"],
            "test": task["label_distributions"]["test"],
        },
        colors={
            "train": "#1f77b4",
            "val": "#ff7f0e",
            "test": "#2ca02c",
        },
    )
    right_svg = render_grouped_bar_chart(
        title=f"{dataset_name}: Retrain / Forget",
        labels=labels,
        series={
            "retrain": task["label_distributions"]["retrain"],
            "forget": task["label_distributions"]["forget"],
        },
        colors={
            "retrain": "#17becf",
            "forget": "#d62728",
        },
    )

    svg = "\n".join(
        [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<svg xmlns="http://www.w3.org/2000/svg" width="1440" height="460" viewBox="0 0 1440 460">',
            f'<text x="720" y="24" text-anchor="middle" font-size="20" font-family="Arial, sans-serif" font-weight="700">Dataset: {html.escape(dataset_name)} | Task: {html.escape(task["task_id"])}</text>',
            '<g transform="translate(10, 30)">',
            left_svg,
            "</g>",
            '<g transform="translate(730, 30)">',
            right_svg,
            "</g>",
            "</svg>",
        ]
    )

    output_path.write_text(svg, encoding="utf-8", newline="\n")
    return relative_path


def rank_labels_by_popularity(
    label_to_ids: dict[str, list[str]], first_positions: dict[str, int]
) -> list[str]:
    """Rank labels by descending frequency with deterministic tie-breaking."""
    return sorted(
        label_to_ids,
        key=lambda label_name: (-len(label_to_ids[label_name]), first_positions[label_name], label_name),
    )


def build_mixed_forget_split(
    dataset: str,
    train_ids: list[str],
    sample_lookup: dict[str, dict[str, Any]],
    forget_percentage: float | None,
    forget_top_k_classes: int | None,
) -> tuple[list[str], list[str], dict[str, Any]]:
    """Select a deterministic mixed-label forget subset from the train split."""
    label_to_ids: dict[str, list[str]] = defaultdict(list)
    first_positions: dict[str, int] = {}
    for position, sample_id in enumerate(train_ids):
        label_name = sample_lookup[sample_id]["label_name"]
        if label_name not in first_positions:
            first_positions[label_name] = position
        label_to_ids[label_name].append(sample_id)

    forget_count, effective_percentage = resolve_mixed_forget_count(
        label_to_ids, forget_percentage
    )
    ranked_labels = rank_labels_by_popularity(label_to_ids, first_positions)
    if forget_top_k_classes is None:
        selected_labels = ranked_labels
        selection_scope = "all_labels"
    else:
        if forget_top_k_classes > len(ranked_labels):
            raise ValueError(
                f"{dataset}: requested top-{forget_top_k_classes} labels but only "
                f"{len(ranked_labels)} labels exist in train"
            )
        selected_labels = ranked_labels[:forget_top_k_classes]
        selection_scope = "top_k_labels"

    selected_label_to_ids = {label_name: label_to_ids[label_name] for label_name in selected_labels}
    selected_capacity = sum(len(ids) for ids in selected_label_to_ids.values())
    if forget_count > selected_capacity:
        raise ValueError(
            f"{dataset}: forget count {forget_count} exceeds capacity {selected_capacity} "
            f"for selected labels {selected_labels}. Lower --forget-percentage or increase "
            "--forget-top-k-classes."
        )

    label_allocations = allocate_proportional_counts(selected_label_to_ids, forget_count)
    selection_salt = f"{dataset}:mixed_forget_v1"

    forget_id_set: set[str] = set()
    for label_name in selected_labels:
        ranked_ids = sorted(
            label_to_ids[label_name],
            key=lambda sample_id: (
                stable_sha256(
                    {
                        "selection_salt": selection_salt,
                        "label_name": label_name,
                        "sample_id": sample_id,
                    }
                ),
                sample_id,
            ),
        )
        forget_id_set.update(ranked_ids[: label_allocations[label_name]])

    forget_ids = [sample_id for sample_id in train_ids if sample_id in forget_id_set]
    retrain_ids = [sample_id for sample_id in train_ids if sample_id not in forget_id_set]

    if len(forget_ids) != forget_count:
        raise ValueError(f"{dataset}: expected {forget_count} forget IDs, found {len(forget_ids)}")
    if len(retrain_ids) != len(train_ids) - len(forget_ids):
        raise ValueError(f"{dataset}: invalid retrain split cardinality")

    return forget_ids, retrain_ids, {
        "mode": "mixed_label_subset",
        "selection_scope": selection_scope,
        "selection_salt": selection_salt,
        "target_count": forget_count,
        "target_percentage_of_train": effective_percentage,
        "target_fraction_of_train": forget_count / len(train_ids),
        "selected_labels": selected_labels,
        "selected_label_count": len(selected_labels),
        "per_label_counts": dict(sorted(label_allocations.items())),
    }


def validate_task(task: dict[str, Any], known_ids: set[str]) -> None:
    """Validate split membership, disjointness, and recorded counts for a task."""
    train_ids = task["train_ids"]
    forget_ids = task["forget_ids"]
    retrain_ids = task["retrain_ids"]
    val_ids = task["val_ids"]
    test_ids = task["test_ids"]

    for field_name in ("train_ids", "forget_ids", "retrain_ids", "val_ids", "test_ids"):
        unknown = sorted(set(task[field_name]) - known_ids)
        if unknown:
            raise ValueError(
                f"{task['dataset']}::{task['task_id']} has unknown IDs in {field_name}: "
                f"{', '.join(unknown[:5])}"
            )

    train_set = set(train_ids)
    forget_set = set(forget_ids)
    retrain_set = set(retrain_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    if not forget_set.issubset(train_set):
        raise ValueError(f"{task['dataset']}::{task['task_id']} has forget not contained in train")
    if retrain_set != (train_set - forget_set):
        raise ValueError(f"{task['dataset']}::{task['task_id']} has retrain != train \\ forget")
    if forget_set & retrain_set:
        raise ValueError(f"{task['dataset']}::{task['task_id']} has forget intersect retrain")
    if train_set & val_set:
        raise ValueError(f"{task['dataset']}::{task['task_id']} has train intersect val")
    if train_set & test_set:
        raise ValueError(f"{task['dataset']}::{task['task_id']} has train intersect test")
    if val_set & test_set:
        raise ValueError(f"{task['dataset']}::{task['task_id']} has val intersect test")

    expected_counts = {
        "train": len(train_ids),
        "forget": len(forget_ids),
        "retrain": len(retrain_ids),
        "val": len(val_ids),
        "test": len(test_ids),
    }
    for key, value in expected_counts.items():
        if task["counts"][key] != value:
            raise ValueError(
                f"{task['dataset']}::{task['task_id']} count mismatch for {key}: "
                f"{task['counts'][key]} != {value}"
            )

    required_distribution_fields = ("train", "forget", "retrain", "val", "test")
    for field_name in required_distribution_fields:
        if field_name not in task.get("label_distributions", {}):
            raise ValueError(f"{task['dataset']}::{task['task_id']} missing label distribution for {field_name}")

    forget_strategy = task.get("forget_strategy", {})
    if "target_percentage_of_train" not in forget_strategy:
        raise ValueError(f"{task['dataset']}::{task['task_id']} missing target_percentage_of_train")
    if "selected_labels" not in forget_strategy:
        raise ValueError(f"{task['dataset']}::{task['task_id']} missing selected_labels")


def dataset_path_rows(
    dataset_root: Path, relative_dir: str, expected_nonempty: bool = True
) -> list[str]:
    """Return sorted repo-relative file paths for a held-out audit directory."""
    target_dir = dataset_root / relative_dir
    if not target_dir.exists():
        raise FileNotFoundError(f"Missing directory: {target_dir}")
    rows = sorted(
        path.relative_to(dataset_root).as_posix()
        for path in target_dir.rglob("*")
        if path.is_file()
    )
    if expected_nonempty and not rows:
        raise ValueError(f"No files found under {target_dir}")
    return rows


def load_cifar10_from_torchvision(data_root: Path) -> tuple[list[int], list[int], list[str]]:
    """Load CIFAR-10 labels from torchvision when the local install supports it."""
    try:
        from torchvision.datasets import CIFAR10  # type: ignore
    except ImportError as exc:
        raise RuntimeError("torchvision is not available") from exc

    cifar_root = data_root / "cifar-10"
    train_dataset = CIFAR10(root=str(cifar_root), train=True, download=False)
    test_dataset = CIFAR10(root=str(cifar_root), train=False, download=False)

    train_labels = list(getattr(train_dataset, "targets", []))
    test_labels = list(getattr(test_dataset, "targets", []))
    class_names = list(getattr(train_dataset, "classes", []))
    if len(train_labels) != 50000 or len(test_labels) != 10000 or len(class_names) != 10:
        raise ValueError("Unexpected CIFAR-10 structure from torchvision")
    return train_labels, test_labels, class_names


def load_cifar10_from_tarball(data_root: Path) -> tuple[list[int], list[int], list[str]]:
    """Load CIFAR-10 labels directly from the downloaded Python tarball.

    This fallback avoids depending on a working extracted directory or a local
    torchvision installation, which makes the generator usable in more
    constrained environments.
    """

    tarball_path = data_root / "cifar-10" / "cifar-10-python.tar.gz"
    if not tarball_path.exists():
        raise FileNotFoundError(f"Missing CIFAR-10 tarball: {tarball_path}")

    train_labels: list[int] = []
    test_labels: list[int] = []
    with tarfile.open(tarball_path, "r:gz") as archive:
        meta_member = archive.getmember("cifar-10-batches-py/batches.meta")
        with archive.extractfile(meta_member) as handle:
            if handle is None:
                raise ValueError("Unable to extract CIFAR-10 metadata from tarball")
            meta = pickle.load(handle, encoding="bytes")
        class_names = [name.decode("utf-8") for name in meta[b"label_names"]]

        for batch_id in range(1, 6):
            member = archive.getmember(f"cifar-10-batches-py/data_batch_{batch_id}")
            with archive.extractfile(member) as handle:
                if handle is None:
                    raise ValueError(f"Unable to extract data_batch_{batch_id}")
                batch = pickle.load(handle, encoding="bytes")
            train_labels.extend(int(label) for label in batch[b"labels"])

        member = archive.getmember("cifar-10-batches-py/test_batch")
        with archive.extractfile(member) as handle:
            if handle is None:
                raise ValueError("Unable to extract test_batch")
            batch = pickle.load(handle, encoding="bytes")
        test_labels.extend(int(label) for label in batch[b"labels"])

    if len(train_labels) != 50000 or len(test_labels) != 10000 or len(class_names) != 10:
        raise ValueError("Unexpected CIFAR-10 structure from tarball")
    return train_labels, test_labels, class_names


def load_cifar10_labels(data_root: Path) -> tuple[list[int], list[int], list[str], str]:
    """Load CIFAR-10 labels and report which backend was used."""
    try:
        train_labels, test_labels, class_names = load_cifar10_from_torchvision(data_root)
        source = "torchvision"
    except Exception:
        train_labels, test_labels, class_names = load_cifar10_from_tarball(data_root)
        source = "tarball"
    return train_labels, test_labels, class_names, source


def generate_cifar10(
    data_root: Path,
    out_root: Path,
    project_root: Path,
    forget_percentage: float | None,
    forget_top_k_classes: int | None,
) -> None:
    """Generate the CIFAR-10 sample catalog and one mixed-class forget task."""
    dataset_root = out_root / "cifar10"
    task_root = dataset_root / "tasks"
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    train_labels, test_labels, class_names, source_name = load_cifar10_labels(data_root)

    if class_names != CIFAR10_CLASSES:
        raise ValueError(f"Unexpected CIFAR-10 class order: {class_names}")

    samples: list[dict[str, Any]] = []
    test_sample_ids: list[str] = []
    class_to_train_ids: dict[int, list[str]] = defaultdict(list)

    for raw_index, label_id in enumerate(train_labels):
        sample_id = f"cifar10-train-{raw_index:05d}"
        class_to_train_ids[label_id].append(sample_id)
        samples.append(
            {
                "sample_id": sample_id,
                "dataset": "cifar10",
                "source_partition": "official_train",
                "label_field": "label",
                "label_id": label_id,
                "label_name": class_names[label_id],
                "raw_index": raw_index,
                "relative_path": "",
            }
        )

    for raw_index, label_id in enumerate(test_labels):
        sample_id = f"cifar10-test-{raw_index:05d}"
        test_sample_ids.append(sample_id)
        samples.append(
            {
                "sample_id": sample_id,
                "dataset": "cifar10",
                "source_partition": "official_test",
                "label_field": "label",
                "label_id": label_id,
                "label_name": class_names[label_id],
                "raw_index": raw_index,
                "relative_path": "",
            }
        )

    ensure_unique_ids(samples, "cifar10")

    val_ids: list[str] = []
    train_ids: list[str] = []
    for label_id in range(10):
        class_ids = class_to_train_ids[label_id]
        if len(class_ids) != 5000:
            raise ValueError(f"CIFAR-10 class {label_id} expected 5000 items, found {len(class_ids)}")
        # Deterministic validation: reserve the first 500 samples from each
        # class in canonical training order, then define train as the remainder.
        val_ids.extend(class_ids[:500])
        train_ids.extend(class_ids[500:])

    known_ids = {row["sample_id"] for row in samples}
    sample_lookup = {row["sample_id"]: row for row in samples}
    samples_fingerprint = stable_sha256(
        {
            "source": source_name,
            "samples": samples,
        }
    )

    write_samples_csv(samples, dataset_root / "samples.csv")

    forget_ids, retrain_ids, forget_strategy = build_mixed_forget_split(
        dataset="cifar10",
        train_ids=train_ids,
        sample_lookup=sample_lookup,
        forget_percentage=forget_percentage,
        forget_top_k_classes=forget_top_k_classes,
    )
    task = {
        "dataset": "cifar10",
        "task_id": "forget_mixed",
        "label_field": "label",
        "split_manifest_version": 3,
        "train_ids": train_ids,
        "forget_ids": forget_ids,
        "retrain_ids": retrain_ids,
        "val_ids": val_ids,
        "test_ids": test_sample_ids,
        "extra_eval_sets": {},
        "counts": {
            "train": len(train_ids),
            "forget": len(forget_ids),
            "retrain": len(retrain_ids),
            "val": len(val_ids),
            "test": len(test_sample_ids),
            "extra_eval_sets": {},
        },
        "label_distributions": {
            "train": count_labels(train_ids, sample_lookup),
            "forget": count_labels(forget_ids, sample_lookup),
            "retrain": count_labels(retrain_ids, sample_lookup),
            "val": count_labels(val_ids, sample_lookup),
            "test": count_labels(test_sample_ids, sample_lookup),
        },
        "artifacts": {},
        "forget_strategy": forget_strategy,
        "source_fingerprint": samples_fingerprint,
    }
    task["artifacts"]["histogram_svg"] = write_task_histogram_svg(task, project_root)
    validate_task(task, known_ids)
    write_task_manifest(task, task_root / f"{task['task_id']}.json")

    verify_cifar10_outputs(dataset_root, project_root)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of string-keyed dictionaries."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def generate_mufac(
    data_root: Path,
    out_root: Path,
    project_root: Path,
    forget_percentage: float | None,
    forget_top_k_classes: int | None,
) -> None:
    """Generate the MUFAC sample catalog and one mixed-age forget task."""
    dataset_source_root = data_root / "MUFAC"
    dataset_root = out_root / "mufac"
    task_root = dataset_root / "tasks"
    if dataset_root.exists():
        shutil.rmtree(dataset_root)

    train_rows = read_csv_rows(dataset_source_root / "custom_train_dataset.csv")
    val_rows = read_csv_rows(dataset_source_root / "custom_val_dataset.csv")
    test_rows = read_csv_rows(dataset_source_root / "custom_test_dataset.csv")

    samples: list[dict[str, Any]] = []
    train_ids: list[str] = []
    val_ids: list[str] = []
    test_ids: list[str] = []
    age_to_train_ids: dict[str, list[str]] = defaultdict(list)

    for partition_name, source_partition, rows, collector in (
        ("train", "custom_train", train_rows, train_ids),
        ("val", "custom_val", val_rows, val_ids),
        ("test", "custom_test", test_rows, test_ids),
    ):
        for raw_index, row in enumerate(rows):
            image_path = Path(row["image_path"]).as_posix()
            age_class = row["age_class"]
            sample_id = f"mufac-{partition_name}-{image_path}"
            collector.append(sample_id)
            if partition_name == "train":
                age_to_train_ids[age_class].append(sample_id)
            samples.append(
                {
                    "sample_id": sample_id,
                    "dataset": "mufac",
                    "source_partition": source_partition,
                    "label_field": "age_class",
                    "label_id": age_class,
                    "label_name": age_class,
                    "raw_index": raw_index,
                    "relative_path": image_path,
                }
            )

    ensure_unique_ids(samples, "mufac")
    known_ids = {row["sample_id"] for row in samples}
    write_samples_csv(samples, dataset_root / "samples.csv")

    extra_eval_sets = {
        "fixed_val_dataset_positive": dataset_path_rows(
            dataset_source_root, "fixed_val_dataset_positive"
        ),
        "fixed_val_dataset_negative": dataset_path_rows(
            dataset_source_root, "fixed_val_dataset_negative"
        ),
        "fixed_test_dataset_positive": dataset_path_rows(
            dataset_source_root, "fixed_test_dataset_positive"
        ),
        "fixed_test_dataset_negative": dataset_path_rows(
            dataset_source_root, "fixed_test_dataset_negative"
        ),
    }

    age_classes = sorted(age_to_train_ids)
    if age_classes != list("abcdefgh"):
        raise ValueError(f"Unexpected MUFAC age classes: {age_classes}")

    source_fingerprint = stable_sha256(
        {
            "train_rows": train_rows,
            "val_rows": val_rows,
            "test_rows": test_rows,
            "extra_eval_sets": extra_eval_sets,
        }
    )
    sample_lookup = {row["sample_id"]: row for row in samples}
    forget_ids, retrain_ids, forget_strategy = build_mixed_forget_split(
        dataset="mufac",
        train_ids=train_ids,
        sample_lookup=sample_lookup,
        forget_percentage=forget_percentage,
        forget_top_k_classes=forget_top_k_classes,
    )
    task = {
        "dataset": "mufac",
        "task_id": "forget_mixed",
        "label_field": "age_class",
        "split_manifest_version": 3,
        "train_ids": train_ids,
        "forget_ids": forget_ids,
        "retrain_ids": retrain_ids,
        "val_ids": val_ids,
        "test_ids": test_ids,
        "extra_eval_sets": extra_eval_sets,
        "counts": {
            "train": len(train_ids),
            "forget": len(forget_ids),
            "retrain": len(retrain_ids),
            "val": len(val_ids),
            "test": len(test_ids),
            "extra_eval_sets": {key: len(value) for key, value in extra_eval_sets.items()},
        },
        "label_distributions": {
            "train": count_labels(train_ids, sample_lookup),
            "forget": count_labels(forget_ids, sample_lookup),
            "retrain": count_labels(retrain_ids, sample_lookup),
            "val": count_labels(val_ids, sample_lookup),
            "test": count_labels(test_ids, sample_lookup),
        },
        "artifacts": {},
        "forget_strategy": forget_strategy,
        "source_fingerprint": source_fingerprint,
    }
    task["artifacts"]["histogram_svg"] = write_task_histogram_svg(task, project_root)
    validate_task(task, known_ids)
    write_task_manifest(task, task_root / f"{task['task_id']}.json")

    verify_mufac_outputs(dataset_root, project_root)


def verify_cifar10_outputs(dataset_root: Path, project_root: Path) -> None:
    """Run dataset-specific invariants over generated CIFAR-10 artifacts."""
    samples = read_samples_csv(dataset_root / "samples.csv")
    known_ids = {row["sample_id"] for row in samples}
    sample_lookup = {row["sample_id"]: row for row in samples}
    tasks = read_task_manifests(dataset_root / "tasks")

    if len(tasks) != 1:
        raise ValueError(f"Expected 1 CIFAR-10 task, found {len(tasks)}")

    task = tasks[0]
    validate_task(task, known_ids)
    histogram_path = project_root / task["artifacts"]["histogram_svg"]
    if not histogram_path.exists():
        raise ValueError(f"CIFAR-10 histogram is missing: {histogram_path}")

    val_counter = Counter(sample_lookup[sample_id]["label_name"] for sample_id in task["val_ids"])
    for class_name in CIFAR10_CLASSES:
        if val_counter[class_name] != 500:
            raise ValueError(f"CIFAR-10 val split for {class_name} expected 500, found {val_counter[class_name]}")

    train_counter = count_labels(task["train_ids"], sample_lookup)
    forget_counter = count_labels(task["forget_ids"], sample_lookup)
    retrain_counter = count_labels(task["retrain_ids"], sample_lookup)
    val_counter_recorded = count_labels(task["val_ids"], sample_lookup)
    test_counter = count_labels(task["test_ids"], sample_lookup)

    for class_name in CIFAR10_CLASSES:
        if train_counter.get(class_name) != 4500:
            raise ValueError(f"CIFAR-10 train split for {class_name} expected 4500, found {train_counter.get(class_name)}")
        if forget_counter.get(class_name, 0) + retrain_counter.get(class_name, 0) != 4500:
            raise ValueError(
                f"CIFAR-10 split mismatch for {class_name}: "
                f"forget + retrain != train ({forget_counter.get(class_name, 0)} + "
                f"{retrain_counter.get(class_name, 0)} != 4500)"
            )

    if task["label_distributions"]["train"] != train_counter:
        raise ValueError("CIFAR-10 recorded train label distribution is incorrect")
    if task["label_distributions"]["forget"] != forget_counter:
        raise ValueError("CIFAR-10 recorded forget label distribution is incorrect")
    if task["label_distributions"]["retrain"] != retrain_counter:
        raise ValueError("CIFAR-10 recorded retrain label distribution is incorrect")
    if task["label_distributions"]["val"] != val_counter_recorded:
        raise ValueError("CIFAR-10 recorded val label distribution is incorrect")
    if task["label_distributions"]["test"] != test_counter:
        raise ValueError("CIFAR-10 recorded test label distribution is incorrect")
    if task["forget_strategy"]["per_label_counts"] != forget_counter:
        raise ValueError("CIFAR-10 forget strategy counts do not match the saved forget split")
    if sorted(task["forget_strategy"]["selected_labels"]) != sorted(forget_counter):
        raise ValueError("CIFAR-10 selected forget labels do not match the saved forget split")

    recorded_percentage = task["forget_strategy"]["target_percentage_of_train"]
    expected_forget_count = max(1, round(len(task["train_ids"]) * (recorded_percentage / 100.0)))
    expected_forget_count = min(expected_forget_count, len(task["train_ids"]) - 1)
    if expected_forget_count != len(task["forget_ids"]):
        raise ValueError("CIFAR-10 forget percentage does not match the saved forget count")


def verify_mufac_outputs(dataset_root: Path, project_root: Path) -> None:
    """Run dataset-specific invariants over generated MUFAC artifacts."""
    samples = read_samples_csv(dataset_root / "samples.csv")
    known_ids = {row["sample_id"] for row in samples}
    sample_lookup = {row["sample_id"]: row for row in samples}
    tasks = read_task_manifests(dataset_root / "tasks")

    if len(tasks) != 1:
        raise ValueError(f"Expected 1 MUFAC task, found {len(tasks)}")

    task = tasks[0]
    validate_task(task, known_ids)
    histogram_path = project_root / task["artifacts"]["histogram_svg"]
    if not histogram_path.exists():
        raise ValueError(f"MUFAC histogram is missing: {histogram_path}")

    train_counter = count_labels(task["train_ids"], sample_lookup)
    forget_counter = count_labels(task["forget_ids"], sample_lookup)
    retrain_counter = count_labels(task["retrain_ids"], sample_lookup)
    val_counter = count_labels(task["val_ids"], sample_lookup)
    test_counter = count_labels(task["test_ids"], sample_lookup)

    if task["label_distributions"]["train"] != train_counter:
        raise ValueError("MUFAC recorded train label distribution is incorrect")
    if task["label_distributions"]["forget"] != forget_counter:
        raise ValueError("MUFAC recorded forget label distribution is incorrect")
    if task["label_distributions"]["retrain"] != retrain_counter:
        raise ValueError("MUFAC recorded retrain label distribution is incorrect")
    if task["label_distributions"]["val"] != val_counter:
        raise ValueError("MUFAC recorded val label distribution is incorrect")
    if task["label_distributions"]["test"] != test_counter:
        raise ValueError("MUFAC recorded test label distribution is incorrect")
    if task["forget_strategy"]["per_label_counts"] != forget_counter:
        raise ValueError("MUFAC forget strategy counts do not match the saved forget split")
    if sorted(task["forget_strategy"]["selected_labels"]) != sorted(forget_counter):
        raise ValueError("MUFAC selected forget labels do not match the saved forget split")

    recorded_percentage = task["forget_strategy"]["target_percentage_of_train"]
    expected_forget_count = max(1, round(len(task["train_ids"]) * (recorded_percentage / 100.0)))
    expected_forget_count = min(expected_forget_count, len(task["train_ids"]) - 1)
    if expected_forget_count != len(task["forget_ids"]):
        raise ValueError("MUFAC forget percentage does not match the saved forget count")

    for extra_name, paths in task["extra_eval_sets"].items():
        if paths != sorted(paths):
            raise ValueError(f"{task['task_id']} extra eval set {extra_name} is not sorted")
        if task["counts"]["extra_eval_sets"].get(extra_name) != len(paths):
            raise ValueError(
                f"{task['task_id']} extra eval count mismatch for {extra_name}: "
                f"{task['counts']['extra_eval_sets'].get(extra_name)} != {len(paths)}"
            )


def read_samples_csv(path: Path) -> list[dict[str, Any]]:
    """Read `samples.csv` and restore the typed `raw_index` field."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    parsed_rows: list[dict[str, Any]] = []
    for row in rows:
        parsed_rows.append(
            {
                "sample_id": row["sample_id"],
                "dataset": row["dataset"],
                "source_partition": row["source_partition"],
                "label_field": row["label_field"],
                "label_id": row["label_id"],
                "label_name": row["label_name"],
                "raw_index": int(row["raw_index"]),
                "relative_path": row["relative_path"],
            }
        )
    return parsed_rows


def read_task_manifests(task_root: Path) -> list[dict[str, Any]]:
    """Load all task manifests from a dataset task directory."""
    manifests = []
    for path in sorted(task_root.glob("*.json")):
        with path.open("r", encoding="utf-8") as handle:
            manifests.append(json.load(handle))
    return manifests


def generate_selected(
    data_root: Path,
    out_root: Path,
    project_root: Path,
    datasets: list[str],
    forget_percentage: float | None,
    forget_top_k_classes: int | None,
) -> None:
    """Generate manifests for the selected datasets."""
    out_root.mkdir(parents=True, exist_ok=True)
    generators = {
        "cifar10": generate_cifar10,
        "mufac": generate_mufac,
    }
    for dataset in datasets:
        generators[dataset](
            data_root,
            out_root,
            project_root,
            forget_percentage,
            forget_top_k_classes,
        )


def collect_relative_files(root: Path) -> dict[str, bytes]:
    """Collect a stable map of relative file paths to file contents."""
    files: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    return files


def verify_outputs(
    data_root: Path,
    out_root: Path,
    project_root: Path,
    datasets: list[str],
    forget_percentage: float | None,
    forget_top_k_classes: int | None,
) -> None:
    """Regenerate into a temp directory and compare artifacts byte-for-byte."""
    if not out_root.exists():
        raise FileNotFoundError(f"Output root does not exist: {out_root}")

    with tempfile.TemporaryDirectory(dir=out_root.parent) as temp_dir:
        temp_root = Path(temp_dir) / "generated"
        temp_project_root = Path(temp_dir) / "project"
        generate_selected(
            data_root,
            temp_root,
            temp_project_root,
            datasets,
            forget_percentage,
            forget_top_k_classes,
        )
        for dataset in datasets:
            expected_root = out_root / dataset
            actual_root = temp_root / dataset
            if not expected_root.exists():
                raise FileNotFoundError(f"Missing generated dataset directory: {expected_root}")
            expected_files = collect_relative_files(expected_root)
            actual_files = collect_relative_files(actual_root)
            if expected_files != actual_files:
                missing = sorted(set(actual_files) - set(expected_files))
                extra = sorted(set(expected_files) - set(actual_files))
                mismatched = sorted(
                    rel_path
                    for rel_path in sorted(set(actual_files) & set(expected_files))
                    if actual_files[rel_path] != expected_files[rel_path]
                )
                raise ValueError(
                    f"Verification failed for {dataset}: "
                    f"missing={missing[:5]}, extra={extra[:5]}, mismatched={mismatched[:5]}"
                )

            expected_task_manifests = read_task_manifests(expected_root / "tasks")
            actual_task_manifests = read_task_manifests(actual_root / "tasks")
            if len(expected_task_manifests) != len(actual_task_manifests):
                raise ValueError(
                    f"Verification failed for {dataset}: manifest count mismatch "
                    f"{len(expected_task_manifests)} != {len(actual_task_manifests)}"
                )

            for expected_task, actual_task in zip(expected_task_manifests, actual_task_manifests):
                expected_artifact_rel = expected_task["artifacts"]["histogram_svg"]
                actual_artifact_rel = actual_task["artifacts"]["histogram_svg"]
                if expected_artifact_rel != actual_artifact_rel:
                    raise ValueError(
                        f"Verification failed for {dataset}: artifact path mismatch "
                        f"{expected_artifact_rel} != {actual_artifact_rel}"
                    )

                expected_artifact_path = project_root / expected_artifact_rel
                actual_artifact_path = temp_project_root / actual_artifact_rel
                if not expected_artifact_path.exists():
                    raise FileNotFoundError(f"Missing expected artifact: {expected_artifact_path}")
                if not actual_artifact_path.exists():
                    raise FileNotFoundError(f"Missing actual artifact: {actual_artifact_path}")
                if expected_artifact_path.read_bytes() != actual_artifact_path.read_bytes():
                    raise ValueError(
                        f"Verification failed for {dataset}: artifact mismatch "
                        f"for {expected_artifact_rel}"
                    )


def main() -> None:
    """CLI entrypoint for generation and reproducibility verification."""
    args = parse_args()
    project_root = Path(__file__).resolve().parents[2]
    data_root = args.data_root.resolve()
    out_root = args.out_root.resolve()
    datasets = list(dict.fromkeys(args.datasets))
    forget_percentage = validate_forget_percentage(args.forget_percentage)
    forget_top_k_classes = validate_forget_top_k(args.forget_top_k_classes)

    if args.verify_only:
        verify_outputs(
            data_root,
            out_root,
            project_root,
            datasets,
            forget_percentage,
            forget_top_k_classes,
        )
        print(f"Verified canonical splits for: {', '.join(datasets)}")
        return

    generate_selected(
        data_root,
        out_root,
        project_root,
        datasets,
        forget_percentage,
        forget_top_k_classes,
    )
    print(f"Generated canonical splits for: {', '.join(datasets)}")


if __name__ == "__main__":
    main()
