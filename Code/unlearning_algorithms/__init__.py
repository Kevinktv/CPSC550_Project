"""Modular unlearning algorithm package used by notebook_workflows."""

from .common import select_efficiency_variant as _select_efficiency_variant
from .ct import (
    CT_UNLEARNING_PROFILES,
    _build_ct_efficiency_variants,
    run_ct_unlearning_workflow,
)
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
from .msg import (
    MSG_UNLEARNING_PROFILES,
    _build_msg_efficiency_variants,
    run_msg_unlearning_workflow,
)
from .scrub import (
    SCRUB_UNLEARNING_PROFILES,
    _build_scrub_efficiency_variants,
    run_scrub_unlearning_workflow,
)

run_second_place_unlearning_workflow = run_fanchuan_unlearning_workflow

__all__ = [
    "CT_UNLEARNING_PROFILES",
    "DELETE_UNLEARNING_PROFILES",
    "FANCHUAN_UNLEARNING_PROFILES",
    "MSG_UNLEARNING_PROFILES",
    "SCRUB_UNLEARNING_PROFILES",
    "_build_ct_efficiency_variants",
    "_build_delete_efficiency_variants",
    "_build_fanchuan_efficiency_variants",
    "_build_msg_efficiency_variants",
    "_build_scrub_efficiency_variants",
    "_select_efficiency_variant",
    "run_ct_unlearning_workflow",
    "run_delete_unlearning_workflow",
    "run_fanchuan_unlearning_workflow",
    "run_msg_unlearning_workflow",
    "run_scrub_unlearning_workflow",
    "run_second_place_unlearning_workflow",
]
