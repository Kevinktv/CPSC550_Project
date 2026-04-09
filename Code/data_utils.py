"""Shared data-loading helpers for the machine unlearning pipeline."""

from __future__ import annotations

import csv
import functools
import json
import pickle
import tarfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
try:
    from PIL import Image
except ImportError:  # pragma: no cover - depends on local environment.
    Image = None  # type: ignore[assignment]

try:
    import torch
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset as TorchDataset
except ImportError:  # pragma: no cover - exercised implicitly in environments without torch.
    torch = None
    DataLoader = Any  # type: ignore[assignment]
    TorchDataset = object  # type: ignore[misc,assignment]

try:
    from Code.model_utils import resolve_image_size
except ImportError:  # pragma: no cover - allows direct module execution.
    from model_utils import resolve_image_size


CIFAR10_CLASSES = (
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
)


@dataclass(frozen=True)
class SampleRecord:
    """A typed row from `samples.csv` enriched with a numeric class index."""

    sample_id: str
    dataset: str
    source_partition: str
    label_field: str
    label_id: str
    label_name: str
    raw_index: int
    relative_path: str
    class_index: int


@dataclass(frozen=True)
class ManifestContext:
    """All manifest-derived metadata needed by training and evaluation."""

    dataset: str
    task_id: str
    label_field: str
    data_root: Path
    task_manifest_path: Path
    samples_csv_path: Path
    label_to_index: dict[str, int]
    index_to_label: dict[int, str]
    class_names: list[str]
    sample_lookup: dict[str, SampleRecord]
    splits: dict[str, list[SampleRecord]]
    extra_eval_sets: dict[str, list[Path]]

    @property
    def num_classes(self) -> int:
        return len(self.label_to_index)

    def split_counts(self) -> dict[str, int]:
        return {name: len(records) for name, records in self.splits.items()}


@dataclass
class DataBundle:
    """Resolved loaders plus the manifest context used to build them."""

    context: ManifestContext
    loaders: dict[str, Any]
    class_counts: dict[int, int]


def _require_torch() -> None:
    if torch is None:  # pragma: no cover - depends on local environment.
        raise ImportError(
            "torch is required for dataset construction and training. "
            "Install dependencies from requirements.txt first."
        )


def _require_pillow() -> None:
    if Image is None:  # pragma: no cover - depends on local environment.
        raise ImportError(
            "Pillow is required for image loading. Install dependencies from requirements.txt first."
        )


def _read_samples_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_task_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_label_order(rows: list[dict[str, str]]) -> list[str]:
    label_ids = {row["label_id"] for row in rows}
    if all(label_id.lstrip("-").isdigit() for label_id in label_ids):
        return [str(value) for value in sorted(int(label_id) for label_id in label_ids)]
    return sorted(label_ids)


def _resolve_default_samples_csv(dataset: str, code_root: Path) -> Path:
    return code_root / "splits" / dataset / "samples.csv"


def _resolve_default_task_manifest(dataset: str, code_root: Path) -> Path:
    return code_root / "splits" / dataset / "tasks" / "forget_mixed.json"


def _resolve_default_data_root(code_root: Path) -> Path:
    return code_root / "data"


def resolve_pipeline_paths(
    dataset: str,
    samples_csv: str | Path | None,
    task_manifest: str | Path | None,
    data_root: str | Path | None,
) -> tuple[Path, Path, Path]:
    """Resolve CLI paths, defaulting to the current repo layout."""

    code_root = Path(__file__).resolve().parent
    samples_path = Path(samples_csv) if samples_csv is not None else _resolve_default_samples_csv(dataset, code_root)
    task_path = (
        Path(task_manifest)
        if task_manifest is not None
        else _resolve_default_task_manifest(dataset, code_root)
    )
    data_root_path = Path(data_root) if data_root is not None else _resolve_default_data_root(code_root)
    return samples_path, task_path, data_root_path


def _normalize_data_root(dataset: str, data_root: Path) -> Path:
    data_root = data_root.resolve()
    if dataset == "cifar10":
        if (data_root / "cifar-10").exists():
            return data_root / "cifar-10"
        return data_root
    if dataset == "mufac":
        if (data_root / "MUFAC").exists():
            return data_root / "MUFAC"
        return data_root
    raise ValueError(f"Unsupported dataset: {dataset}")


@functools.lru_cache(maxsize=None)
def _build_dataset_file_index(data_root: str) -> dict[str, list[Path]]:
    """Index dataset files by basename for fallback MUFAC path resolution."""
    root = Path(data_root)
    basename_to_paths: dict[str, list[Path]] = {}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        basename_to_paths.setdefault(path.name, []).append(path)
    return basename_to_paths


def _mufac_candidate_priority(path: Path, source_partition: str) -> tuple[int, str]:
    """Prefer the partition-matching MUFAC subdirectory when duplicates exist."""
    relative = path.as_posix()
    if source_partition == "custom_train":
        preferred_prefixes = (
            "train_images_part1/",
            "train_images_part2/",
            "retain_images/",
            "forget_images/",
        )
    elif source_partition == "custom_val":
        preferred_prefixes = ("val_images/",)
    elif source_partition == "custom_test":
        preferred_prefixes = ("test_images/",)
    else:
        preferred_prefixes = (f"{source_partition}/",)

    for rank, prefix in enumerate(preferred_prefixes):
        if relative.startswith(prefix):
            return rank, relative
    return len(preferred_prefixes), relative


def _resolve_mufac_image_path(data_root: Path, relative_path: str, source_partition: str) -> Path:
    """Resolve MUFAC image paths, including filename-only manifest entries."""
    candidate = data_root / relative_path
    if candidate.exists():
        return candidate

    basename_index = _build_dataset_file_index(str(data_root.resolve()))
    matches = basename_index.get(Path(relative_path).name, [])
    if not matches:
        raise FileNotFoundError(
            f"Unable to resolve MUFAC image '{relative_path}' under '{data_root}'."
        )

    ranked = sorted(
        (path.relative_to(data_root) for path in matches),
        key=lambda path: _mufac_candidate_priority(path, source_partition),
    )
    return data_root / ranked[0]


def build_manifest_context(
    dataset: str,
    samples_csv: str | Path | None = None,
    task_manifest: str | Path | None = None,
    data_root: str | Path | None = None,
) -> ManifestContext:
    """Load `samples.csv` and the task manifest into a typed context object."""

    samples_path, task_path, data_root_path = resolve_pipeline_paths(dataset, samples_csv, task_manifest, data_root)
    rows = _read_samples_csv(samples_path)
    task = _read_task_manifest(task_path)
    label_order = _parse_label_order(rows)
    label_to_index = {label_id: index for index, label_id in enumerate(label_order)}
    label_id_to_name: dict[str, str] = {}

    sample_lookup: dict[str, SampleRecord] = {}
    for row in rows:
        label_id = row["label_id"]
        label_id_to_name.setdefault(label_id, row["label_name"])
        sample = SampleRecord(
            sample_id=row["sample_id"],
            dataset=row["dataset"],
            source_partition=row["source_partition"],
            label_field=row["label_field"],
            label_id=label_id,
            label_name=row["label_name"],
            raw_index=int(row["raw_index"]),
            relative_path=row["relative_path"],
            class_index=label_to_index[label_id],
        )
        sample_lookup[sample.sample_id] = sample

    splits: dict[str, list[SampleRecord]] = {}
    for split_name in ("train", "forget", "retrain", "val", "test"):
        split_ids = task[f"{split_name}_ids"]
        splits[split_name] = [sample_lookup[sample_id] for sample_id in split_ids]

    extra_eval_sets = {
        name: [_normalize_data_root(dataset, data_root_path) / relative_path for relative_path in relative_paths]
        for name, relative_paths in task.get("extra_eval_sets", {}).items()
    }
    index_to_label = {index: label_id for label_id, index in label_to_index.items()}
    class_names = [label_id_to_name[index_to_label[index]] for index in range(len(index_to_label))]
    return ManifestContext(
        dataset=task["dataset"],
        task_id=task["task_id"],
        label_field=task["label_field"],
        data_root=_normalize_data_root(dataset, data_root_path),
        task_manifest_path=task_path,
        samples_csv_path=samples_path,
        label_to_index=label_to_index,
        index_to_label=index_to_label,
        class_names=class_names,
        sample_lookup=sample_lookup,
        splits=splits,
        extra_eval_sets=extra_eval_sets,
    )


def compute_class_counts(records: list[SampleRecord]) -> dict[int, int]:
    """Count class occurrences for a manifest split."""

    return dict(Counter(record.class_index for record in records))


class _SimpleImageTransform:
    """Resize and convert PIL images to float tensors without torchvision transforms."""

    def __init__(self, image_size: int) -> None:
        self.image_size = image_size

    def __call__(self, image: Any) -> Any:
        _require_torch()
        resized = image.convert("RGB").resize((self.image_size, self.image_size))
        array = np.asarray(resized, dtype=np.float32) / 255.0
        array = np.transpose(array, (2, 0, 1))
        return torch.from_numpy(array)


class CIFAR10ImageStore:
    """Load CIFAR-10 arrays once and serve them by manifest record."""

    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root
        self.train_images, self.train_targets, self.test_images, self.test_targets = self._load()

    def _load(self) -> tuple[np.ndarray, list[int], np.ndarray, list[int]]:
        try:
            from torchvision.datasets import CIFAR10

            train_dataset = CIFAR10(root=str(self.data_root), train=True, download=False)
            test_dataset = CIFAR10(root=str(self.data_root), train=False, download=False)
            train_images = np.asarray(getattr(train_dataset, "data"))
            test_images = np.asarray(getattr(test_dataset, "data"))
            train_targets = [int(value) for value in getattr(train_dataset, "targets")]
            test_targets = [int(value) for value in getattr(test_dataset, "targets")]
            return train_images, train_targets, test_images, test_targets
        except Exception:
            return self._load_from_tarball()

    def _load_from_tarball(self) -> tuple[np.ndarray, list[int], np.ndarray, list[int]]:
        tarball_path = self.data_root / "cifar-10-python.tar.gz"
        if not tarball_path.exists():
            raise FileNotFoundError(f"Missing CIFAR-10 tarball: {tarball_path}")

        train_batches: list[np.ndarray] = []
        train_targets: list[int] = []
        test_images = np.empty((0, 32, 32, 3), dtype=np.uint8)
        test_targets: list[int] = []
        with tarfile.open(tarball_path, "r:gz") as archive:
            for batch_id in range(1, 6):
                member = archive.getmember(f"cifar-10-batches-py/data_batch_{batch_id}")
                with archive.extractfile(member) as handle:
                    if handle is None:
                        raise ValueError(f"Unable to extract data_batch_{batch_id}")
                    batch = pickle.load(handle, encoding="bytes")
                batch_images = np.asarray(batch[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
                train_batches.append(np.transpose(batch_images, (0, 2, 3, 1)))
                train_targets.extend(int(label) for label in batch[b"labels"])

            member = archive.getmember("cifar-10-batches-py/test_batch")
            with archive.extractfile(member) as handle:
                if handle is None:
                    raise ValueError("Unable to extract test_batch")
                batch = pickle.load(handle, encoding="bytes")
            batch_images = np.asarray(batch[b"data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
            test_images = np.transpose(batch_images, (0, 2, 3, 1))
            test_targets.extend(int(label) for label in batch[b"labels"])

        train_images = np.concatenate(train_batches, axis=0)
        return train_images, train_targets, test_images, test_targets

    def get_image(self, record: SampleRecord) -> Any:
        _require_pillow()
        if record.source_partition == "official_test":
            image = self.test_images[record.raw_index]
        else:
            image = self.train_images[record.raw_index]
        return Image.fromarray(image)


class ManifestImageDataset(TorchDataset):
    """Dataset backed by manifest records for either CIFAR-10 or MUFAC."""

    def __init__(
        self,
        records: list[SampleRecord],
        dataset_name: str,
        data_root: Path,
        transform: _SimpleImageTransform | None = None,
        cifar_store: CIFAR10ImageStore | None = None,
    ) -> None:
        _require_torch()
        self.records = records
        self.dataset_name = dataset_name
        self.data_root = data_root
        self.transform = transform or _SimpleImageTransform()
        self.cifar_store = cifar_store
        if self.dataset_name == "cifar10" and self.cifar_store is None:
            self.cifar_store = CIFAR10ImageStore(self.data_root)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        _require_torch()
        record = self.records[index]
        image = self._load_image(record)
        tensor = self.transform(image)
        target = torch.tensor(record.class_index, dtype=torch.long)
        return tensor, target

    def _load_image(self, record: SampleRecord) -> Any:
        _require_pillow()
        if self.dataset_name == "cifar10":
            assert self.cifar_store is not None
            return self.cifar_store.get_image(record)

        image_path = _resolve_mufac_image_path(
            self.data_root,
            record.relative_path,
            record.source_partition,
        )
        with Image.open(image_path) as image:
            return image.convert("RGB")


def create_dataloaders_from_manifest(
    dataset: str,
    task_manifest: str | Path | None,
    samples_csv: str | Path | None,
    data_root: str | Path | None,
    batch_size: int,
    num_workers: int,
    include_extra_eval_sets: bool = False,
    image_size: int | None = None,
) -> DataBundle:
    """Resolve loaders for all standard manifest splits."""

    _require_torch()
    context = build_manifest_context(dataset, samples_csv=samples_csv, task_manifest=task_manifest, data_root=data_root)
    resolved_image_size = resolve_image_size(context.dataset, image_size)
    transform = _SimpleImageTransform(image_size=resolved_image_size)
    cifar_store = CIFAR10ImageStore(context.data_root) if context.dataset == "cifar10" else None
    loaders: dict[str, Any] = {}
    dataset_by_split: dict[str, Any] = {}
    for split_name, records in context.splits.items():
        dataset_obj = ManifestImageDataset(
            records=records,
            dataset_name=context.dataset,
            data_root=context.data_root,
            transform=transform,
            cifar_store=cifar_store,
        )
        dataset_by_split[split_name] = dataset_obj
        loaders[split_name] = DataLoader(
            dataset_obj,
            batch_size=batch_size,
            shuffle=split_name in {"train", "retrain"},
            num_workers=num_workers,
        )

    loaders["retain"] = DataLoader(
        dataset_by_split["retrain"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    if include_extra_eval_sets:
        for name, image_paths in context.extra_eval_sets.items():
            records = [
                SampleRecord(
                    sample_id=str(path.relative_to(context.data_root)),
                    dataset=context.dataset,
                    source_partition=name,
                    label_field=context.label_field,
                    label_id="",
                    label_name="",
                    raw_index=index,
                    relative_path=str(path.relative_to(context.data_root)),
                    class_index=-1,
                )
                for index, path in enumerate(image_paths)
            ]
            dataset_obj = ManifestImageDataset(
                records=records,
                dataset_name=context.dataset,
                data_root=context.data_root,
                transform=transform,
                cifar_store=cifar_store,
            )
            loaders[name] = DataLoader(
                dataset_obj,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

    class_counts = compute_class_counts(context.splits["retrain"])
    return DataBundle(context=context, loaders=loaders, class_counts=class_counts)
