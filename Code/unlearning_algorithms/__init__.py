"""Modular unlearning algorithm package used by notebook_workflows."""

from .common import select_efficiency_variant as _select_efficiency_variant
from .delete import (
    DELETE_UNLEARNING_PROFILES,
    _build_delete_efficiency_variants,
    run_delete_unlearning_workflow,
)
from .fanchuan import (
    FANCHUAN_UNLEARNING_PROFILES,
    _build_fanchuan_efficiency_variants,
    run_fanchuan_unlearning_workflow,
)
from .scrub import (
    SCRUB_UNLEARNING_PROFILES,
    _build_scrub_efficiency_variants,
    run_scrub_unlearning_workflow,
)

run_second_place_unlearning_workflow = run_fanchuan_unlearning_workflow

__all__ = [
    "DELETE_UNLEARNING_PROFILES",
    "FANCHUAN_UNLEARNING_PROFILES",
    "SCRUB_UNLEARNING_PROFILES",
    "_build_delete_efficiency_variants",
    "_build_fanchuan_efficiency_variants",
    "_build_scrub_efficiency_variants",
    "_select_efficiency_variant",
    "run_delete_unlearning_workflow",
    "run_fanchuan_unlearning_workflow",
    "run_scrub_unlearning_workflow",
    "run_second_place_unlearning_workflow",
]
