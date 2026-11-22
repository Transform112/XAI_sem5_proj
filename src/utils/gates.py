"""Gate-grade preconditions for run manifests (T17/T18).

Every artifact that a gate decision can cite must carry a real execution lane
(``--lane`` or ``XDRIFT_LANE`` via ``run_lane``) and a reconstructible code pin.
Rather than trusting each CLI to remember the check, ``RunManifest`` calls
:func:`enforce_gate_lane` for gate artifact kinds, so a missing lane fails before
a gate manifest can be written.

Local development can opt out explicitly with ``XDRIFT_ALLOW_DEFAULT_LANE=1``
(recorded in the manifest as ``lane_source=default`` + a warning), but normal
gate runs must not use it.
"""

from __future__ import annotations

import os
from typing import Any

from src.utils.paths import Paths

#: Manifest kinds that a gate decision or the paper may cite.
GATE_KINDS: frozenset[str] = frozenset(
    {
        "data_audit",
        "dataset_identity",
        "graph_build",
        "graph_lift_pilot",
        "train_S3",
        "cl_evaluation",
        "anchor_build",
        "explanation_S7",
        "drift_analysis",
        "noise_floor",
        "matrix_run",
        "anchor_explain_batch",
        "rq2_structural",
        "rq3_assembly",
    }
)

ALLOW_DEFAULT_LANE_ENV = "XDRIFT_ALLOW_DEFAULT_LANE"


class GateRequirementError(RuntimeError):
    """A gate artifact would be written without a real lane / clean-enough pin."""


def enforce_gate_lane(kind: str, paths: Paths | None) -> None:
    """Raise when a gate-kind manifest would record lane_source='default'."""
    if paths is None or kind not in GATE_KINDS:
        return
    lane_source = getattr(paths, "lane_source", "default")
    if lane_source != "default":
        return
    if os.environ.get(ALLOW_DEFAULT_LANE_ENV) == "1":
        # Explicit local-development escape hatch; stays visible in the manifest.
        return
    raise GateRequirementError(
        f"gate artifact kind {kind!r} requires an explicit lane (got lane_source='default'). "
        "Launch via src/scripts/run_lane.ps1 -Lane <legacy|rtx5060|kaggle|cpu> / pass --lane / set "
        f"XDRIFT_LANE, or set {ALLOW_DEFAULT_LANE_ENV}=1 for local development only."
    )


def gate_record(kind: str, paths: Paths | None) -> dict[str, Any]:
    """Small record embedded in manifests describing gate preconditions."""
    if paths is None:
        return {"gate_kind": kind, "lane": None, "lane_source": None}
    return {
        "gate_kind": kind,
        "lane": paths.lane,
        "lane_source": paths.lane_source,
        "allow_default_lane_env": os.environ.get(ALLOW_DEFAULT_LANE_ENV) == "1",
        "real_lane": getattr(paths, "lane_source", "default") != "default",
    }


__all__ = [
    "ALLOW_DEFAULT_LANE_ENV",
    "GATE_KINDS",
    "GateRequirementError",
    "enforce_gate_lane",
    "gate_record",
]
