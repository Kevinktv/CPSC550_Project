"""Generate canonical deterministic split manifests for CIFAR-10 and MUFAC.

This script materializes the benchmark split definitions described in the
project plan:

- CIFAR-10
  - `D`: official training split after removing a deterministic per-class
    validation slice.
  - `S`: class-conditioned forget set for one target class at a time.
  - `D'`: retain set defined as `D \\ S`.
  - `val` / `test`: unseen held-out evaluation sets.

- MUFAC
  - `D`: rows from `custom_train_dataset.csv`.
  - `S`: age-conditioned forget set for one age bucket at a time.
  - `D'`: retain set defined as `D \\ S`.
  - `val` / `test`: rows from the corresponding held-out CSVs.
  - `extra_eval_sets`: fixed audit directories stored separately from the core
    train/validation/test partition.


"""

from __future__ import annotations

import argparse
import csv
import hashlib
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


def validate_task(task: dict[str, Any], known_ids: set[str]) -> None:
    """Validate split membership, disjointness, and recorded counts for a task."""
    d_ids = task["D_ids"]
    s_ids = task["S_ids"]
    d_prime_ids = task["D_prime_ids"]
    val_ids = task["val_ids"]
    test_ids = task["test_ids"]

    for field_name in ("D_ids", "S_ids", "D_prime_ids", "val_ids", "test_ids"):
        unknown = sorted(set(task[field_name]) - known_ids)
        if unknown:
            raise ValueError(
                f"{task['dataset']}::{task['task_id']} has unknown IDs in {field_name}: "
                f"{', '.join(unknown[:5])}"
            )

    d_set = set(d_ids)
    s_set = set(s_ids)
    d_prime_set = set(d_prime_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)

    if not s_set.issubset(d_set):
        raise ValueError(f"{task['dataset']}::{task['task_id']} has S not contained in D")
    if d_prime_set != (d_set - s_set):
        raise ValueError(f"{task['dataset']}::{task['task_id']} has D' != D \\ S")
    if s_set & d_prime_set:
        raise ValueError(f"{task['dataset']}::{task['task_id']} has S intersect D'")
    if d_set & val_set:
        raise ValueError(f"{task['dataset']}::{task['task_id']} has D intersect val")
    if d_set & test_set:
        raise ValueError(f"{task['dataset']}::{task['task_id']} has D intersect test")
    if val_set & test_set:
        raise ValueError(f"{task['dataset']}::{task['task_id']} has val intersect test")

    expected_counts = {
        "D": len(d_ids),
        "S": len(s_ids),
        "D_prime": len(d_prime_ids),
        "val": len(val_ids),
        "test": len(test_ids),
    }
    for key, value in expected_counts.items():
        if task["counts"][key] != value:
            raise ValueError(
                f"{task['dataset']}::{task['task_id']} count mismatch for {key}: "
                f"{task['counts'][key]} != {value}"
            )


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


def generate_cifar10(data_root: Path, out_root: Path) -> None:
    """Generate the CIFAR-10 sample catalog and 10 class-forgetting tasks."""
    dataset_root = out_root / "cifar10"
    task_root = dataset_root / "tasks"
    if dataset_root.exists():
        shutil.rmtree(dataset_root)
    train_labels, test_labels, class_names, source_name = load_cifar10_labels(data_root)

    if class_names != CIFAR10_CLASSES:
        raise ValueError(f"Unexpected CIFAR-10 class order: {class_names}")

    samples: list[dict[str, Any]] = []
    train_sample_ids: list[str] = []
    test_sample_ids: list[str] = []
    class_to_train_ids: dict[int, list[str]] = defaultdict(list)

    for raw_index, label_id in enumerate(train_labels):
        sample_id = f"cifar10-train-{raw_index:05d}"
        train_sample_ids.append(sample_id)
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
    d_ids: list[str] = []
    for label_id in range(10):
        class_ids = class_to_train_ids[label_id]
        if len(class_ids) != 5000:
            raise ValueError(f"CIFAR-10 class {label_id} expected 5000 items, found {len(class_ids)}")
        # Deterministic validation: reserve the first 500 samples from each
        # class in canonical training order, then define D as the remainder.
        val_ids.extend(class_ids[:500])
        d_ids.extend(class_ids[500:])

    known_ids = {row["sample_id"] for row in samples}
    samples_fingerprint = stable_sha256(
        {
            "source": source_name,
            "samples": samples,
        }
    )

    write_samples_csv(samples, dataset_root / "samples.csv")

    d_id_set = set(d_ids)
    for label_id, label_name in enumerate(class_names):
        s_ids = [sample_id for sample_id in class_to_train_ids[label_id] if sample_id in d_id_set]
        # D' is always defined from the saved D membership, never recomputed
        # from labels downstream.
        d_prime_ids = [sample_id for sample_id in d_ids if sample_id not in set(s_ids)]
        task = {
            "dataset": "cifar10",
            "task_id": f"forget_{label_name}",
            "target_field": "label",
            "target_label": {"id": label_id, "name": label_name},
            "D_ids": d_ids,
            "S_ids": s_ids,
            "D_prime_ids": d_prime_ids,
            "val_ids": val_ids,
            "test_ids": test_sample_ids,
            "extra_eval_sets": {},
            "counts": {
                "D": len(d_ids),
                "S": len(s_ids),
                "D_prime": len(d_prime_ids),
                "val": len(val_ids),
                "test": len(test_sample_ids),
                "extra_eval_sets": {},
            },
            "source_fingerprint": samples_fingerprint,
        }
        validate_task(task, known_ids)
        write_task_manifest(task, task_root / f"{task['task_id']}.json")

    verify_cifar10_outputs(dataset_root)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of string-keyed dictionaries."""
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def generate_mufac(data_root: Path, out_root: Path) -> None:
    """Generate the MUFAC sample catalog and age-conditioned forget tasks."""
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

    for age_class in age_classes:
        s_ids = list(age_to_train_ids[age_class])
        s_id_set = set(s_ids)
        # MUFAC follows the same formal definition as CIFAR-10: D' = D \ S,
        # with D taken from the canonical saved train CSV rows.
        d_prime_ids = [sample_id for sample_id in train_ids if sample_id not in s_id_set]
        task = {
            "dataset": "mufac",
            "task_id": f"forget_age_{age_class}",
            "target_field": "age_class",
            "target_label": {"id": age_class, "name": age_class},
            "D_ids": train_ids,
            "S_ids": s_ids,
            "D_prime_ids": d_prime_ids,
            "val_ids": val_ids,
            "test_ids": test_ids,
            "extra_eval_sets": extra_eval_sets,
            "counts": {
                "D": len(train_ids),
                "S": len(s_ids),
                "D_prime": len(d_prime_ids),
                "val": len(val_ids),
                "test": len(test_ids),
                "extra_eval_sets": {key: len(value) for key, value in extra_eval_sets.items()},
            },
            "source_fingerprint": source_fingerprint,
        }
        validate_task(task, known_ids)
        write_task_manifest(task, task_root / f"{task['task_id']}.json")

    verify_mufac_outputs(dataset_root)


def verify_cifar10_outputs(dataset_root: Path) -> None:
    """Run dataset-specific invariants over generated CIFAR-10 artifacts."""
    samples = read_samples_csv(dataset_root / "samples.csv")
    known_ids = {row["sample_id"] for row in samples}
    sample_lookup = {row["sample_id"]: row for row in samples}
    tasks = read_task_manifests(dataset_root / "tasks")

    if len(tasks) != 10:
        raise ValueError(f"Expected 10 CIFAR-10 tasks, found {len(tasks)}")

    val_counter = Counter(sample_lookup[sample_id]["label_name"] for sample_id in tasks[0]["val_ids"])
    for class_name in CIFAR10_CLASSES:
        if val_counter[class_name] != 500:
            raise ValueError(f"CIFAR-10 val split for {class_name} expected 500, found {val_counter[class_name]}")

    for task in tasks:
        validate_task(task, known_ids)
        target_name = task["target_label"]["name"]
        if any(sample_lookup[sample_id]["label_name"] != target_name for sample_id in task["S_ids"]):
            raise ValueError(f"{task['task_id']} has non-target labels in S")
        target_d_count = sum(
            1 for sample_id in task["D_ids"] if sample_lookup[sample_id]["label_name"] == target_name
        )
        if target_d_count != len(task["S_ids"]):
            raise ValueError(
                f"{task['task_id']} expected S to contain every target label in D; "
                f"found {len(task['S_ids'])} of {target_d_count}"
            )


def verify_mufac_outputs(dataset_root: Path) -> None:
    """Run dataset-specific invariants over generated MUFAC artifacts."""
    samples = read_samples_csv(dataset_root / "samples.csv")
    known_ids = {row["sample_id"] for row in samples}
    sample_lookup = {row["sample_id"]: row for row in samples}
    tasks = read_task_manifests(dataset_root / "tasks")

    if len(tasks) != 8:
        raise ValueError(f"Expected 8 MUFAC tasks, found {len(tasks)}")

    for task in tasks:
        validate_task(task, known_ids)
        target_age = task["target_label"]["name"]
        if any(sample_lookup[sample_id]["label_name"] != target_age for sample_id in task["S_ids"]):
            raise ValueError(f"{task['task_id']} has non-target age classes in S")
        target_d_count = sum(
            1 for sample_id in task["D_ids"] if sample_lookup[sample_id]["label_name"] == target_age
        )
        if target_d_count != len(task["S_ids"]):
            raise ValueError(
                f"{task['task_id']} expected S to contain every target age in D; "
                f"found {len(task['S_ids'])} of {target_d_count}"
            )
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


def generate_selected(data_root: Path, out_root: Path, datasets: list[str]) -> None:
    """Generate manifests for the selected datasets."""
    out_root.mkdir(parents=True, exist_ok=True)
    generators = {
        "cifar10": generate_cifar10,
        "mufac": generate_mufac,
    }
    for dataset in datasets:
        generators[dataset](data_root, out_root)


def collect_relative_files(root: Path) -> dict[str, bytes]:
    """Collect a stable map of relative file paths to file contents."""
    files: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file():
            files[path.relative_to(root).as_posix()] = path.read_bytes()
    return files


def verify_outputs(data_root: Path, out_root: Path, datasets: list[str]) -> None:
    """Regenerate into a temp directory and compare artifacts byte-for-byte."""
    if not out_root.exists():
        raise FileNotFoundError(f"Output root does not exist: {out_root}")

    with tempfile.TemporaryDirectory(dir=out_root.parent) as temp_dir:
        temp_root = Path(temp_dir) / "generated"
        generate_selected(data_root, temp_root, datasets)
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


def main() -> None:
    """CLI entrypoint for generation and reproducibility verification."""
    args = parse_args()
    data_root = args.data_root.resolve()
    out_root = args.out_root.resolve()
    datasets = list(dict.fromkeys(args.datasets))

    if args.verify_only:
        verify_outputs(data_root, out_root, datasets)
        print(f"Verified canonical splits for: {', '.join(datasets)}")
        return

    generate_selected(data_root, out_root, datasets)
    print(f"Generated canonical splits for: {', '.join(datasets)}")


if __name__ == "__main__":
    main()
