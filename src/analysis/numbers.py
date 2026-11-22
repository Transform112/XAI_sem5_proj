"""``paper_writing/manuscript/numbers.json`` exporter (Doc 2b §S13; skeleton "Outputs to generate at S13").

Authority
---------
* ``research/lit/claims_evidence_skeleton.md`` proposes the schema: top-level
  ``git_sha``, ``frozen_at``, ``protocol_version``, ``dataset``, ``claims`` where
  each claim has ``stat``, ``value``, ``ci95``, ``artifact``, ``status`` (C5 adds
  ``f1_delta``; C1 adds ``n_anchors``). The skeleton marks the schema as
  "to confirm at G5"; this module validates against that proposal and refuses
  to emit a malformed numbers file.
* ``docs/source_documents/Doc 2b`` §S13: "a single JSON generated from the frozen tables;
  the manuscript imports numbers from it (checked by script)".

Scope note
----------
``paper_writing/manuscript/`` is outside the analyst write scope (``src/analysis/**``,
``results/text/tables/**``, ``results/images/**``). This module builds and validates
the payload and writes it wherever the caller points; the synthetic demo writes
``src/analysis/examples/paper_numbers.json`` (schema-identical), and the S13 writer
(Lead/verifier) writes the real ``paper_writing/manuscript/numbers.json``.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.analysis import artifacts

#: Status vocabulary from the claim→evidence skeleton.
STATUS_VOCABULARY: tuple[str, ...] = ("pending", "produced", "recomputed", "frozen", "approved")

#: Claims that carry a numeric headline in the proposed schema (C7 is a scope
#: statement and is deliberately not numeric).
NUMERIC_CLAIMS: tuple[str, ...] = ("C1", "C2", "C3", "C4", "C5", "C6")

#: Required keys per claim entry in the proposed schema.
REQUIRED_CLAIM_KEYS: tuple[str, ...] = ("stat", "value", "ci95", "artifact", "status")


class NumbersError(ValueError):
    """Raised when the numbers payload is malformed or unsupported by the tables."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _finite(value: Any, *, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise NumbersError(f"{name} must be finite, got {value!r}")
    return number


def build_claim_entry(
    *,
    stat: str,
    value: Any,
    ci95: Sequence[Any],
    artifact: str,
    status: str = "produced",
    extras: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate one claim entry of ``paper_writing/manuscript/numbers.json``."""
    if status not in STATUS_VOCABULARY:
        raise NumbersError(f"status {status!r} not in {STATUS_VOCABULARY}")
    if not artifact:
        raise NumbersError("artifact path is required for every claim number")
    if len(list(ci95)) != 2:
        raise NumbersError(f"ci95 must have exactly two endpoints, got {ci95!r}")
    endpoints = [_finite(endpoint, name=f"{stat}.ci95") for endpoint in ci95]
    if endpoints[0] > endpoints[1]:
        raise NumbersError(f"ci95 lower bound {endpoints[0]} exceeds upper bound {endpoints[1]}")
    entry: dict[str, Any] = {
        "stat": str(stat),
        "value": _finite(value, name=f"{stat}.value"),
        "ci95": endpoints,
        "artifact": str(artifact),
        "status": status,
    }
    for key, extra in (extras or {}).items():
        if key in entry:
            raise NumbersError(f"extra key {key!r} collides with a required claim key")
        entry[key] = extra
    return entry


def build_paper_numbers(
    claims: Mapping[str, Mapping[str, Any]],
    *,
    git_sha: str,
    frozen_at: str | None = None,
    protocol_version: str = "1",
    dataset: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble the complete ``paper_writing/manuscript/numbers.json`` payload (schema-validated)."""
    if not git_sha:
        raise NumbersError("git_sha is required (no number may be quoted from a dirty/unknown tree)")
    unknown = set(claims) - set(NUMERIC_CLAIMS)
    if unknown:
        raise NumbersError(f"unknown claim ids {sorted(unknown)}; expected subset of {NUMERIC_CLAIMS}")
    payload: dict[str, Any] = {
        "git_sha": str(git_sha),
        "frozen_at": frozen_at or _utc_now(),
        "protocol_version": str(protocol_version),
        "dataset": dict(dataset or {"primary": "NF-UNSW-NB15-v3", "periods": 5}),
        "claims": {claim: dict(entry) for claim, entry in claims.items()},
    }
    issues = validate_numbers_json(payload)
    if issues:
        raise NumbersError("numbers payload failed validation: " + "; ".join(issues))
    return payload


def validate_numbers_json(payload: Mapping[str, Any]) -> list[str]:
    """Structural validation of a ``paper_writing/manuscript/numbers.json`` payload (empty list = valid)."""
    issues: list[str] = []
    for key in ("git_sha", "frozen_at", "protocol_version", "dataset", "claims"):
        if key not in payload:
            issues.append(f"missing top-level key {key!r}")
    claims = payload.get("claims")
    if not isinstance(claims, Mapping):
        return issues + ["'claims' must be a mapping"]
    for claim_id, entry in claims.items():
        if claim_id not in NUMERIC_CLAIMS:
            issues.append(f"unknown claim id {claim_id!r}")
            continue
        if not isinstance(entry, Mapping):
            issues.append(f"claim {claim_id} is not a mapping")
            continue
        for required in REQUIRED_CLAIM_KEYS:
            if required not in entry:
                issues.append(f"claim {claim_id} missing {required!r}")
        if "status" in entry and entry["status"] not in STATUS_VOCABULARY:
            issues.append(f"claim {claim_id} status {entry['status']!r} not in {STATUS_VOCABULARY}")
        if "ci95" in entry:
            try:
                if len(list(entry["ci95"])) != 2:
                    issues.append(f"claim {claim_id} ci95 must have two endpoints")
                else:
                    low, high = (float(value) for value in entry["ci95"])
                    if not np.isfinite(low) or not np.isfinite(high):
                        issues.append(f"claim {claim_id} ci95 endpoints must be finite")
                    elif low > high:
                        issues.append(f"claim {claim_id} ci95 lower bound exceeds upper bound")
            except (TypeError, ValueError):
                issues.append(f"claim {claim_id} ci95 is not numeric")
        if "value" in entry:
            try:
                float(entry["value"])
            except (TypeError, ValueError):
                issues.append(f"claim {claim_id} value is not numeric")
        if "artifact" in entry and not str(entry["artifact"]):
            issues.append(f"claim {claim_id} artifact path is empty")
    return issues


def write_numbers_json(payload: Mapping[str, Any], path: str | Path) -> Path:
    """Validate and write the numbers payload (UTF-8, sorted keys, deterministic)."""
    issues = validate_numbers_json(payload)
    if issues:
        raise NumbersError("refusing to write invalid numbers payload: " + "; ".join(issues))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


# ---------------------------------------------------------------------------
# Extraction from the generated tables (S13 close-out helper)
# ---------------------------------------------------------------------------
def _first_row(frame: pd.DataFrame, mask: pd.Series, description: str) -> pd.Series:
    selected = frame[mask]
    if selected.empty:
        raise NumbersError(f"cannot extract {description}: no matching row")
    return selected.iloc[0]


def _ci(row: pd.Series, low: str, high: str) -> list[float]:
    return [_finite(row[low], name=low), _finite(row[high], name=high)]


def numbers_from_tables(
    tables: Mapping[str, pd.DataFrame],
    *,
    git_sha: str,
    status: str = "produced",
    frozen_at: str | None = None,
    graph_lift: Mapping[str, Any] | None = None,
    dataset: Mapping[str, Any] | None = None,
    primary_strategy: str = "ft",
    primary_explainer: str = "IG",
    primary_model: str = "egraphsage",
) -> dict[str, Any]:
    """Extract the C1–C6 headline numbers from the generated frozen tables.

    ``tables`` must contain ``rq1_h1_tests``, ``rq1_strategy_comparison``,
    ``rq2_correlations``, ``rq3_auc``, ``rq4_comparison`` (:mod:`pandas` frames
    as written by :mod:`analysis.tables`). C6 comes from the audited graph-lift
    artifact (``graph_lift={"delta_rand": ..., "ci95": [...], "artifact": ...}``)
    because it is produced by the data-audit stage, not by these tables.
    """
    required = ("rq1_h1_tests", "rq1_strategy_comparison", "rq2_correlations", "rq3_auc", "rq4_comparison")
    missing = [name for name in required if name not in tables]
    if missing:
        raise NumbersError(f"numbers_from_tables missing table(s) {missing}")
    claims: dict[str, dict[str, Any]] = {}

    h1 = tables["rq1_h1_tests"]
    mask = (h1["strategy"] == primary_strategy) & (h1["explainer"] == primary_explainer)
    if "model" in h1.columns:
        mask &= h1["model"] == primary_model
    row = _first_row(h1, mask, f"C1 H1 row {primary_strategy}/{primary_model}/{primary_explainer}")
    claims["C1"] = build_claim_entry(
        stat="fraction_above_floor_ft",
        value=row["fraction_above_floor"],
        ci95=_ci(row, "fraction_ci_lo", "fraction_ci_hi"),
        artifact="results/text/tables/rq1_h1_tests.csv",
        status=status,
        extras={"n_anchors": int(row["n_anchors"])},
    )

    comparison = tables["rq1_strategy_comparison"]
    row = _first_row(
        comparison,
        (comparison["comparison"] == "replay") & (comparison["metric"] == "drift_primary"),
        "C2 replay−FT drift difference",
    )
    claims["C2"] = build_claim_entry(
        stat="drift_diff_replay_minus_ft",
        value=row["mean_diff"],
        ci95=_ci(row, "ci_lo", "ci_hi"),
        artifact="results/text/tables/rq1_strategy_comparison.csv",
        status=status,
    )

    correlations = tables["rq2_correlations"]
    mask = pd.Series(True, index=correlations.index)
    if "explainer" in correlations.columns:
        mask &= correlations["explainer"] == primary_explainer
    if "strategy" in correlations.columns:
        mask &= correlations["strategy"] == primary_strategy
    row = _first_row(correlations, mask, "C3 RQ2 Spearman")
    claims["C3"] = build_claim_entry(
        stat="spearman_neighbourhood_change_vs_excess",
        value=row["spearman"],
        ci95=_ci(row, "spearman_ci_lo", "spearman_ci_hi"),
        artifact="results/text/tables/rq2_correlations.csv",
        status=status,
    )

    rq3 = tables["rq3_auc"]
    delta = rq3[rq3.get("role", "") == "delta"]
    if delta.empty:
        raise NumbersError("C4 requires a delta-AUC row in rq3_auc")
    row = delta.iloc[0]
    claims["C4"] = build_claim_entry(
        stat="auc_drift_predicts_forgetting",
        value=row["delta_auc"] if "delta_auc" in row and pd.notna(row["delta_auc"]) else row["auc"],
        ci95=_ci(row, "delta_ci_lo" if "delta_ci_lo" in row and pd.notna(row["delta_ci_lo"]) else "ci_lo",
                 "delta_ci_hi" if "delta_ci_hi" in row and pd.notna(row["delta_ci_hi"]) else "ci_hi"),
        artifact="results/text/tables/rq3_auc.csv",
        status=status,
        extras={"h3_supported": bool(row.get("h3_supported", False))},
    )

    rq4 = tables["rq4_comparison"]
    summary = rq4[rq4.get("row_kind", "run") != "run"]
    if summary.empty:
        raise NumbersError("C5 requires H4 summary rows in rq4_comparison")
    row = _first_row(
        summary,
        (summary["model"] == primary_model) & (summary["variant"] == "ear_a"),
        "C5 EAR-A vs replay summary",
    )
    claims["C5"] = build_claim_entry(
        stat="ear_drift_reduction",
        value=row["drift_diff_vs_replay"],
        ci95=_ci(row, "drift_ci_lo", "drift_ci_hi"),
        artifact="results/text/tables/rq4_comparison.csv",
        status=status,
        extras={
            "f1_delta": _finite(row["f1_diff_vs_replay"], name="C5.f1_delta"),
            "h4_supported": bool(row.get("h4_supported", False)),
        },
    )

    if graph_lift is not None:
        claims["C6"] = build_claim_entry(
            stat="graph_lift_delta_rand",
            value=graph_lift["delta_rand"],
            ci95=graph_lift["ci95"],
            artifact=str(graph_lift.get("artifact", "results/text/data_audit/graph_lift_pilot.json")),
            status=status,
        )
    return build_paper_numbers(
        claims,
        git_sha=git_sha,
        frozen_at=frozen_at,
        dataset=dataset,
    )


__all__ = [
    "NUMERIC_CLAIMS",
    "NumbersError",
    "REQUIRED_CLAIM_KEYS",
    "STATUS_VOCABULARY",
    "build_claim_entry",
    "build_paper_numbers",
    "numbers_from_tables",
    "validate_numbers_json",
    "write_numbers_json",
]
