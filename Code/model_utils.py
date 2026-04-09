"""Model helpers for the machine unlearning pipeline."""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - depends on local environment.
    torch = None


def _require_torch() -> None:
    if torch is None:  # pragma: no cover - depends on local environment.
        raise ImportError(
            "torch and torchvision are required for model construction. "
            "Install dependencies from requirements.txt first."
        )


def choose_device(device_name: str) -> Any:
    """Resolve the requested device, falling back to CPU when needed."""

    _require_torch()
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def set_random_seed(seed: int) -> None:
    """Seed Python-independent RNGs used by torch and NumPy."""

    _require_torch()
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_image_size(dataset: str) -> int:
    """Return the standard input size for the dataset."""

    if dataset == "cifar10":
        return 32
    return 224


def resolve_image_size(dataset: str, image_size: int | None = None) -> int:
    """Return an explicit image size or the dataset-specific default."""

    return default_image_size(dataset) if image_size is None else int(image_size)


def create_resnet18(num_classes: int, dataset: str | None = None) -> Any:
    """Instantiate a dataset-appropriate ResNet-18 classifier."""

    _require_torch()
    try:
        from torchvision.models import resnet18
    except ImportError as exc:  # pragma: no cover - depends on local environment.
        raise ImportError(
            "torchvision is required for ResNet-18 model construction. "
            "Install dependencies from requirements.txt first."
        ) from exc

    model = resnet18(weights=None)
    if dataset == "cifar10":
        # CIFAR-10 images are 32x32, so keep the stem shallow and avoid the
        # ImageNet max-pool downsampling that would collapse spatial detail.
        model.conv1 = torch.nn.Conv2d(
            3,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        model.maxpool = torch.nn.Identity()
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, num_classes)
    return model


def build_model(model_factory: Any, *, num_classes: int, dataset: str) -> Any:
    """Instantiate a model factory, passing `dataset` when supported."""

    try:
        parameters = inspect.signature(model_factory).parameters
    except (TypeError, ValueError):
        parameters = {}

    if "dataset" in parameters:
        if "num_classes" in parameters:
            return model_factory(num_classes=num_classes, dataset=dataset)
        return model_factory(num_classes, dataset=dataset)

    if "num_classes" in parameters:
        return model_factory(num_classes=num_classes)
    return model_factory(num_classes)


def load_model_checkpoint(model: Any, checkpoint_path: str | Path, device: Any) -> Any:
    """Load a checkpoint into an already-instantiated model."""

    _require_torch()
    state = torch.load(Path(checkpoint_path), map_location=device)
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    model.load_state_dict(state_dict)
    return model
