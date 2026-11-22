"""Feature manifest: classify every physical column of a dataset.

Why this module exists
----------------------
The S-D1 stage must be able to state, for every raw column, whether it is kept
as a model feature, dropped (with the reason), or used as the target/sidecar.
That statement is an auditable artifact (``feature_manifest.csv``) and it is
generated from the registry spec, not from ad-hoc lists in code.

Terminology
-----------
role        ``feature`` | ``dropped`` | ``target``
source_group``nf_v2_feature`` | ``v3_temporal`` | ``label``
transform   what the preprocessor does: ``label_encoding``,
            ``log1p+standardize``, ``standardize``, ``none``
graph_role  how the graph stage uses the column: ``node_id`` (hosts mapped to
            global node ids), ``sidecar`` (raw timestamps kept for period
            boundaries only), ``edge_feature``, ``target``, ``excluded``
reason      human-readable justification (from the dataset YAML where possible)

The 10 temporal columns of the NF-v3 family are classified explicitly:
2 raw timestamps are sidecar-only (leakage if used as features), while the 8
inter-arrival-time statistics are legitimate flow statistics and remain
features. This is the decision the T4 task requires to be documented.
"""

from __future__ import annotations

import pandas as pd

from src.data.registry import DatasetSpec, DatasetSpecError

ROLE_FEATURE = "feature"
ROLE_DROPPED = "dropped"
ROLE_TARGET = "target"

GRAPH_ROLE_NODE = "node_id"
GRAPH_ROLE_SIDECAR = "sidecar"
GRAPH_ROLE_EDGE_FEATURE = "edge_feature"
GRAPH_ROLE_TARGET = "target"
GRAPH_ROLE_EXCLUDED = "excluded"

COLUMNS = [
    "name",
    "role",
    "source_group",
    "transform",
    "graph_role",
    "reason",
    "declared_in",
]


def _source_group(spec: DatasetSpec, column: str) -> str:
    if column in spec.nf_v2_features:
        return "nf_v2_feature"
    if column in spec.temporal_raw or column in spec.temporal_iat:
        return "v3_temporal"
    if column in (spec.optional_column("label_binary"), spec.optional_column("label_multiclass")):
        return "label"
    return "unknown"


def _role_for_column(spec: DatasetSpec, column: str) -> tuple[str, str, str]:
    """Return (role, transform, graph_role) for one physical column."""
    if column == spec.optional_column("label_multiclass"):
        return ROLE_TARGET, "none", GRAPH_ROLE_TARGET
    if column in spec.drop_features:
        if column in spec.temporal_raw:
            graph_role = GRAPH_ROLE_SIDECAR
        elif column in (spec.optional_column("host_src"), spec.optional_column("host_dst")):
            graph_role = GRAPH_ROLE_NODE
        else:
            graph_role = GRAPH_ROLE_EXCLUDED
        return ROLE_DROPPED, "none", graph_role
    if column in spec.categorical:
        return ROLE_FEATURE, "label_encoding", GRAPH_ROLE_EDGE_FEATURE
    if column in spec.log1p:
        return ROLE_FEATURE, "log1p+standardize", GRAPH_ROLE_EDGE_FEATURE
    if column in spec.numeric:
        return ROLE_FEATURE, "standardize", GRAPH_ROLE_EDGE_FEATURE
    raise DatasetSpecError(
        f"dataset {spec.name!r}: column {column!r} is not classified as target, dropped or feature"
    )


def _reason_for_column(spec: DatasetSpec, column: str, role: str, transform: str) -> str:
    if role == ROLE_TARGET:
        return "multiclass target column (class ids defined by class_order)"
    if role == ROLE_DROPPED:
        reasons = spec.drop_reasons
        if column in reasons:
            return reasons[column]
        if column in spec.temporal_raw:
            return "leakage/sidecar: raw timestamp kept for period boundaries only, never a feature"
        return "identity/leakage: excluded from the feature matrix"
    if column in spec.categorical:
        return (
            "kept feature: categorical NetFlow attribute, label-encoded with a "
            "period-1-train vocabulary (unseen categories -> reserved unknown index)"
        )
    if column in spec.temporal_iat:
        return (
            "kept feature: inter-arrival-time statistic added by the v3 temporal "
            "group; a flow statistic, not a clock value, so it does not leak the split"
        )
    if column in spec.log1p:
        return "kept feature: non-negative heavy-tailed NetFlow statistic, log1p then standardised"
    return "kept feature: bounded NetFlow statistic, standardised"


def build_feature_manifest(spec: DatasetSpec) -> pd.DataFrame:
    """Build the per-column classification table (one row per physical column)."""
    rows = []
    for column in spec.all_columns():
        role, transform, graph_role = _role_for_column(spec, column)
        declared_in = []
        if column in spec.drop_features:
            declared_in.append("drop_features")
        if column in spec.categorical:
            declared_in.append("categorical")
        if column in spec.log1p:
            declared_in.append("log1p")
        if column in spec.numeric:
            declared_in.append("numeric")
        if column in spec.temporal_raw:
            declared_in.append("temporal_raw")
        if column in spec.temporal_iat:
            declared_in.append("temporal_iat")
        if column == spec.optional_column("label_multiclass"):
            declared_in.append("label_multiclass")
        if column == spec.optional_column("label_binary"):
            declared_in.append("label_binary")
        rows.append(
            {
                "name": column,
                "role": role,
                "source_group": _source_group(spec, column),
                "transform": transform,
                "graph_role": graph_role,
                "reason": _reason_for_column(spec, column, role, transform),
                "declared_in": ";".join(declared_in) if declared_in else "role",
            }
        )
    manifest = pd.DataFrame(rows, columns=COLUMNS)
    feature_names = manifest.loc[manifest["role"] == ROLE_FEATURE, "name"].tolist()
    if not feature_names:
        raise DatasetSpecError(f"dataset {spec.name!r}: feature manifest has no feature columns")
    return manifest


def manifest_feature_columns(manifest: pd.DataFrame) -> list[str]:
    """Ordered feature column names from a manifest (rows in spec order)."""
    return manifest.loc[manifest["role"] == ROLE_FEATURE, "name"].tolist()


__all__ = [
    "COLUMNS",
    "GRAPH_ROLE_EDGE_FEATURE",
    "GRAPH_ROLE_EXCLUDED",
    "GRAPH_ROLE_NODE",
    "GRAPH_ROLE_SIDECAR",
    "GRAPH_ROLE_TARGET",
    "ROLE_DROPPED",
    "ROLE_FEATURE",
    "ROLE_TARGET",
    "build_feature_manifest",
    "manifest_feature_columns",
]
