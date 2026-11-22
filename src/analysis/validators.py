"""Real-artifact validators for the S5/S8–S12 pipeline outputs (T19).

Why this module exists
----------------------
The first real S8/S9/S10 runs must be checkable *immediately*, before anyone
builds a paper claim on them. The verifier should not have to re-derive the
artifact contract by hand: this module checks the frozen artifact schemas,
required columns, arithmetic invariants and cross-file consistency, and returns a
machine-readable report with per-check `pass | warn | fail` statuses.

Authority
---------
* ``research/lit/claims_evidence_skeleton.md`` — artifact paths and per-claim
  required content.
* ``docs/DECISIONS.md`` D-043 (writer contract + frozen filenames) and D-044
  (``src/analysis/`` modules are the authoritative metric definitions).
* ``docs/05_EXPERIMENT_PROTOCOL.md`` §8 (drift metrics), §9 (noise floor/excess/H1),
  §10 (RQ2), §11 (RQ3), §12 (RQ4) and §13 (statistics).
* ``src/analysis/artifacts.py`` — path registry; ``src/analysis/tables.py`` — writer
  contract that the synthetic demo exercises.

What is validated
-----------------
* ``dependencies``: scipy/pandas/numpy only — the validators must run on the CPU
  analysis lane without torch.
* ``cl_matrix.json`` / ``cl_metrics.csv`` / ``cl_per_class_f1.json`` (S5 writer),
  ``per_class_f1.{npz,csv}`` (analysis-materialised);
* ``drift_metrics_{ig,gnnexplainer}.csv`` (S8 primary/secondary metrics,
  ``drift = 1 - similarity``, no future/illogical checkpoint pairs);
* ``noise_floor.csv`` (quantile, q95, pooling provenance, channel vocabulary);
* ``rq1_*.csv`` (excess arithmetic, above-floor rule, CI ordering, Holm family);
* ``rq2_*.csv`` (correlation ranges, CI ordering, permutation p family);
* ``rq3_*.csv`` (δ = 0.05 label rule, no future-F1 columns in the feature table,
  AUC ranges, H3 flag consistency);
* ``rq4_*.csv`` (H4 decision consistency, variants present, budget columns).

A partial S5 artifact (``M_1`` only) is a legitimate intermediate state: S5
checks emit `warn` (not `fail`) for the not-yet-computable full-matrix metrics
unless ``allow_partial_s5=False``.

Usage
-----
    .\\.venv\\Scripts\\python.exe -m src.analysis.validators --stage s8
    .\\.venv\\Scripts\\python.exe -m src.analysis.validators --stage all --json
    python -c "from src.analysis.validators import validate_artifact_set; ..."
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

PASS = "pass"
WARN = "warn"
FAIL = "fail"

_STATUS_ORDER = {WARN: 1, PASS: 0, FAIL: 2}

#: Tolerance for arithmetic invariants (columns are written with %.10g precision).
ATOL = 1e-8

#: Frozen protocol constants that artifacts may not silently change.
PROTOCOL_QUANTILE = 0.95
PROTOCOL_FORGETTING_DELTA = 0.05
PROTOCOL_EXPECTED_ABOVE_FRACTION = 0.05
PROTOCOL_H4_F1_LOSS_POINTS = 1.0

CHANNELS = {"model_null", "explainer_repeat", "combined"}
SUPPORTED_LABEL = "supported_set"
ALL_CLASSES_LABEL = "all_classes"


class ValidatorError(ValueError):
    """Raised for programmatic misuse (not for an invalid artifact — that is a Finding)."""


@dataclass
class Finding:
    """One validation check with status and evidence."""

    check: str
    status: str
    detail: str
    path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"check": self.check, "status": self.status, "detail": self.detail, "path": self.path}


@dataclass
class ValidationReport:
    """Collection of findings for one results table directory."""

    results_tables: str
    stage: str = "all"
    findings: list[Finding] = field(default_factory=list)

    def add(self, check: str, status: str, detail: str, path: str | Path | None = None) -> None:
        if status not in _STATUS_ORDER:
            raise ValidatorError(f"unknown status {status!r}")
        self.findings.append(Finding(check, status, detail, None if path is None else str(path)))

    @property
    def counts(self) -> dict[str, int]:
        counts = {PASS: 0, WARN: 0, FAIL: 0}
        for finding in self.findings:
            counts[finding.status] += 1
        return counts

    @property
    def overall(self) -> str:
        counts = self.counts
        if counts[FAIL]:
            return FAIL
        if counts[WARN]:
            return WARN
        return PASS

    @property
    def failures(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.status == FAIL]

    def as_dict(self) -> dict[str, Any]:
        return {
            "results_tables": self.results_tables,
            "stage": self.stage,
            "overall": self.overall,
            "counts": self.counts,
            "findings": [finding.as_dict() for finding in self.findings],
        }


# ---------------------------------------------------------------------------
# Stage requirements (artifact contract)
# ---------------------------------------------------------------------------
#: Cumulative required artifacts per stage (docs/05 §8–§12, skeleton paths).
#:
#: Entries are resolved against ``results/text/tables/`` unless they contain a ``/``,
#: in which case they are globbed from the results root (``results/``). A ``|``
#: separates accepted alternative paths (skeleton vs engineer convention); the
#: canonical (first) path is preferred and a non-canonical choice is reported as a
#: ``warn`` path-convention divergence.
_S5_REQUIRED: tuple[str, ...] = (
    "cl_matrix.json",
    "cl_metrics.csv",
    "cl_per_class_f1.json",
)
_S6_REQUIRED: tuple[str, ...] = _S5_REQUIRED + (
    "anchors/period_*.pt",
    "tables/anchor_retention.csv|anchors/retained_fraction.csv",  # D-048: skeleton path canonical
)
_S7_REQUIRED: tuple[str, ...] = _S6_REQUIRED + (
    "tables/explain_index.csv|explanations/index.csv",  # D-048: committed evidence index canonical
)
_S8_REQUIRED: tuple[str, ...] = _S7_REQUIRED + (
    "explanations_noise/model_null/index.csv",
    "explanations_noise/explainer_repeat/index.csv",
    "explanations_noise/combined_null.csv",
    "drift_metrics_ig.csv",
    "drift_metrics_gnnexplainer.csv",
    "noise_floor.csv",
)
_S9_REQUIRED: tuple[str, ...] = _S8_REQUIRED + (
    "rq1_anchor_level.csv",
    "rq1_aggregated.csv",
    "rq1_h1_tests.csv",
    "rq1_strategy_comparison.csv",
)
_S10_REQUIRED: tuple[str, ...] = _S9_REQUIRED + (
    "rq2_anchor_level.csv",
    "rq2_correlations.csv",
)
_S11_REQUIRED: tuple[str, ...] = _S10_REQUIRED + (
    "rq3_features.csv",
    "rq3_forgetting_labels.csv",
    "rq3_auc.csv",
)
_S12_REQUIRED: tuple[str, ...] = _S11_REQUIRED + (
    "rq4_comparison.csv",
    "rq4_ablations.csv",
)
STAGE_REQUIREMENTS: Mapping[str, tuple[str, ...]] = {
    "s5": _S5_REQUIRED,
    "s6": _S6_REQUIRED,
    "s7": _S7_REQUIRED,
    "s8": _S8_REQUIRED,
    "s9": _S9_REQUIRED,
    "s10": _S10_REQUIRED,
    "s11": _S11_REQUIRED,
    "s12": _S12_REQUIRED,
    "all": _S12_REQUIRED,
}

#: Frozen stage order (docs/02); ``s6``/``s7`` inserted per T20.
STAGE_ORDER = ("s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12")

#: Supplementary artifacts (registered in src/analysis/artifacts.py EXTRA_TABLE_ARTIFACTS).
OPTIONAL_BY_STAGE: Mapping[str, tuple[str, ...]] = {
    "s6": ("anchors/anchors_manifest.json",),
    "s7": ("tables/explain_index.csv",),
    "s11": ("rq3_coefficients.csv",),
    "s12": ("rq3_coefficients.csv",),
    "all": ("rq3_coefficients.csv", "per_class_f1.csv", "per_class_f1.npz"),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _require_columns(report: ValidationReport, frame: pd.DataFrame, required: Sequence[str], path: Path, *, artifact: str) -> bool:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        report.add(f"{artifact}.columns", FAIL, f"missing required column(s) {missing}; found {list(frame.columns)}", path)
        return False
    report.add(f"{artifact}.columns", PASS, f"required columns present ({len(required)} checked)", path)
    return True


def _numeric(report: ValidationReport, frame: pd.DataFrame, columns: Sequence[str], path: Path, *, artifact: str) -> bool:
    bad = [column for column in columns if column in frame.columns and not np.issubdtype(frame[column].dtype, np.number)]
    if bad:
        report.add(f"{artifact}.numeric", FAIL, f"non-numeric column(s) {bad}", path)
        return False
    return True


def _read_csv(report: ValidationReport, path: Path, *, artifact: str) -> pd.DataFrame | None:
    if not path.is_file():
        report.add(f"{artifact}.present", FAIL, "artifact missing", path)
        return None
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - corrupt file path
        report.add(f"{artifact}.readable", FAIL, f"unreadable CSV: {exc}", path)
        return None
    if frame.empty:
        report.add(f"{artifact}.nonempty", FAIL, "artifact has zero rows", path)
        return None
    report.add(f"{artifact}.readable", PASS, f"{len(frame)} row(s), {len(frame.columns)} column(s)", path)
    return frame


def _read_json(report: ValidationReport, path: Path, *, artifact: str) -> Mapping[str, Any] | None:
    if not path.is_file():
        report.add(f"{artifact}.present", FAIL, "artifact missing", path)
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        report.add(f"{artifact}.readable", FAIL, f"unreadable JSON: {exc}", path)
        return None
    if not isinstance(payload, Mapping):
        report.add(f"{artifact}.readable", FAIL, "JSON root is not an object", path)
        return None
    report.add(f"{artifact}.readable", PASS, f"{len(payload)} top-level keys", path)
    return payload


# ---------------------------------------------------------------------------
# S5 validators
# ---------------------------------------------------------------------------
def validate_cl_matrix_json(report: ValidationReport, path: Path, *, allow_partial: bool = True) -> None:
    payload = _read_json(report, path, artifact="cl_matrix")
    if payload is None:
        return
    required = {"dataset", "strategy", "model", "seed", "class_order", "matrix", "dual_macro_f1_per_cell"}
    missing = sorted(required - set(payload))
    if missing:
        report.add("cl_matrix.schema", FAIL, f"missing key(s) {missing}", path)
        return
    class_order = list(payload["class_order"])
    report.add(
        "cl_matrix.class_order",
        PASS if len(class_order) == 10 else FAIL,
        f"{len(class_order)} classes (frozen label space is 10)",
        path,
    )
    matrix_payload = payload.get("matrix") or {}
    if "all_classes" not in matrix_payload:
        report.add("cl_matrix.matrix", FAIL, "matrix.all_classes missing", path)
        return
    matrix = np.asarray(matrix_payload["all_classes"], dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        report.add("cl_matrix.matrix_shape", FAIL, f"matrix.all_classes must be square, got {matrix.shape}", path)
        return
    report.add("cl_matrix.matrix_shape", PASS, f"{matrix.shape[0]}x{matrix.shape[1]}", path)
    if np.any(np.isinf(matrix)):
        report.add("cl_matrix.matrix_values", FAIL, "matrix contains inf", path)
    finite = np.isfinite(matrix)
    populated_rows = sorted(int(key) - 1 for key in (payload.get("dual_macro_f1_per_cell") or {}))
    n_periods = matrix.shape[0]
    complete = len(populated_rows) == n_periods and np.all(finite[n_periods - 1, :])
    if complete:
        report.add("cl_matrix.complete", PASS, "last checkpoint row complete; full CL metrics computable", path)
    elif allow_partial:
        report.add(
            "cl_matrix.complete",
            WARN,
            f"partial checkpoint matrix ({len(populated_rows)}/{n_periods} rows); full forgetting/BWT not computable yet",
            path,
        )
        if payload.get("cl_metrics_all_classes") is not None:
            report.add("cl_matrix.partial_metrics", WARN, "CL metrics present despite incomplete row", path)
    else:
        report.add("cl_matrix.complete", FAIL, f"partial checkpoint matrix ({len(populated_rows)}/{n_periods} rows)", path)
    per_cell = payload.get("dual_macro_f1_per_cell") or {}
    problems: list[str] = []
    for row_key, columns in per_cell.items():
        for column_key, cell in (columns or {}).items():
            dual = (cell or {}).get("dual_macro_f1") or {}
            per_class = (cell or {}).get("per_class_f1") or {}
            if not dual or not per_class:
                problems.append(f"M_{row_key}/P_{column_key} incomplete cell")
                continue
            values = [float(value) for value in per_class.values()]
            if any(not 0.0 <= value <= 1.0 for value in values):
                problems.append(f"M_{row_key}/P_{column_key} per-class F1 outside [0,1]")
            if abs(float(np.mean(values)) - float(dual["macro_f1_all_classes"])) > ATOL:
                problems.append(f"M_{row_key}/P_{column_key} macro_all != mean(per-class)")
            class_counts = dual.get("class_counts") or {}
            if int(sum(int(value) for value in class_counts.values())) != int(dual.get("n_edges", -1)):
                problems.append(f"M_{row_key}/P_{column_key} class_counts sum != n_edges")
    report.add(
        "cl_matrix.dual_macro_consistency",
        PASS if not problems else FAIL,
        f"{sum(len(columns or {}) for columns in per_cell.values())} cell(s) checked; problems={problems[:3]}",
        path,
    )


def validate_cl_metrics_csv(report: ValidationReport, path: Path) -> None:
    frame = _read_csv(report, path, artifact="cl_metrics")
    if frame is None:
        return
    required = ["strategy", "model", "seed", "metric", "final_average_macro_f1", "forgetting", "bwt", "note"]
    if not _require_columns(report, frame, required, path, artifact="cl_metrics"):
        return
    metrics = set(frame["metric"].astype(str))
    report.add(
        "cl_metrics.metric_rows",
        PASS if metrics == {ALL_CLASSES_LABEL, SUPPORTED_LABEL} else FAIL,
        f"metric rows: {sorted(metrics)}",
        path,
    )
    numeric = frame[["final_average_macro_f1", "forgetting", "bwt"]]
    populated = numeric.notna().all(axis=1)
    if populated.any():
        ok = ((numeric["final_average_macro_f1"][populated] >= 0.0) & (numeric["final_average_macro_f1"][populated] <= 1.0)).all()
        report.add("cl_metrics.score_range", PASS if ok else FAIL, "final macro-F1 in [0,1] where populated", path)
    if (~populated).any():
        notes = set(frame["note"][~populated].astype(str))
        report.add(
            "cl_metrics.incomplete_semantics",
            PASS if notes <= {"incomplete_matrix", "nan"} else WARN,
            f"unpopulated rows note(s)={sorted(notes)}",
            path,
        )


def validate_per_class_json(report: ValidationReport, path: Path, *, matrix_path: Path | None = None) -> None:
    payload = _read_json(report, path, artifact="per_class_f1")
    if payload is None:
        return
    if "per_class_f1" not in payload or "class_order" not in payload:
        report.add("per_class_f1.schema", FAIL, "expected keys per_class_f1 + class_order", path)
        return
    class_order = list(payload["class_order"])
    report.add(
        "per_class_f1.class_order",
        PASS if len(class_order) == 10 else FAIL,
        f"{len(class_order)} classes",
        path,
    )
    cells = sum(len(columns or {}) for columns in (payload["per_class_f1"] or {}).values())
    report.add("per_class_f1.cells", PASS if cells > 0 else FAIL, f"{cells} populated cell(s)", path)
    if matrix_path is not None and matrix_path.is_file():
        try:
            matrix_payload = json.loads(matrix_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            matrix_payload = None
        if isinstance(matrix_payload, Mapping) and list(matrix_payload.get("class_order") or []) == class_order:
            report.add("per_class_f1.class_order_matches_matrix", PASS, "identical class_order to cl_matrix.json", path)
        else:
            report.add("per_class_f1.class_order_matches_matrix", FAIL, "class_order differs from cl_matrix.json", path)


def validate_per_class_f1_csv(report: ValidationReport, path: Path) -> None:
    frame = _read_csv(report, path, artifact="per_class_f1_csv")
    if frame is None:
        return
    required = ["strategy", "model", "seed", "checkpoint", "eval_period", "class", "f1"]
    if not _require_columns(report, frame, required, path, artifact="per_class_f1_csv"):
        return
    ok = ((frame["f1"] >= 0.0) & (frame["f1"] <= 1.0)).all()
    report.add("per_class_f1_csv.range", PASS if ok else FAIL, "per-class F1 values in [0,1]", path)
    report.add(
        "per_class_f1_csv.future_cells",
        PASS,
        "future-evaluation cells allowed (Doc 2 S5 'for all i and j') and excluded only from forgetting",
        path,
    )


def validate_per_class_f1_npz(report: ValidationReport, path: Path) -> None:
    """Validate the analysis-materialised ``per_class_f1.npz`` tensor (T20)."""
    report.add("per_class_f1_npz.present", PASS, "artifact present", path)
    if not Path(path).is_file():
        report.add("per_class_f1_npz.readable", FAIL, "npz missing", path)
        return
    try:
        with np.load(path) as payload:
            keys = set(payload.files)
            required = {"R_class", "R_macro", "checkpoints", "eval_periods", "classes", "meta"}
            missing = sorted(required - keys)
            if missing:
                report.add("per_class_f1_npz.keys", FAIL, f"missing npz key(s) {missing}", path)
                return
            report.add("per_class_f1_npz.keys", PASS, f"frozen npz keys present ({sorted(keys)})", path)
            r_class = payload["R_class"]
            r_macro = payload["R_macro"]
            classes = payload["classes"]
            if r_class.ndim != 3 or r_macro.ndim != 2:
                report.add("per_class_f1_npz.shape", FAIL, f"R_class {r_class.shape}, R_macro {r_macro.shape}", path)
                return
            shape_ok = r_class.shape[0] == r_class.shape[1] == r_macro.shape[0] == r_macro.shape[1] and r_class.shape[2] == len(classes)
            report.add("per_class_f1_npz.shape", PASS if shape_ok else FAIL, f"R_class={r_class.shape}, R_macro={r_macro.shape}, classes={len(classes)}", path)
            finite = r_class[np.isfinite(r_class)]
            report.add(
                "per_class_f1_npz.range",
                PASS if finite.size == 0 or bool(np.all((finite >= -1e-9) & (finite <= 1.0 + 1e-9))) else FAIL,
                "per-class F1 values in [0,1]",
                path,
            )
    except Exception as exc:  # pragma: no cover - corrupt artifact
        report.add("per_class_f1_npz.readable", FAIL, f"cannot load npz: {exc}", path)


# ---------------------------------------------------------------------------
# S8/S9 validators
# ---------------------------------------------------------------------------
DRIFT_FEATURE_METRICS = ("cosine", "jaccard_top3", "jaccard_top5", "jaccard_top10", "spearman")
DRIFT_EDGE_METRICS = ("edge_jaccard_top5", "edge_cosine")


def validate_drift_metrics_csv(report: ValidationReport, path: Path, *, explainer: str) -> None:
    frame = _read_csv(report, path, artifact=f"drift_metrics_{explainer}")
    if frame is None:
        return
    required = ["strategy", "model", "explainer", "anchor_id", "class", "checkpoint_j", "checkpoint_t", "drift_primary", "sim_primary"]
    if not _require_columns(report, frame, required, path, artifact=f"drift_metrics_{explainer}"):
        return
    # T49: each explainer has one applicable metric family.  IG carries feature
    # attributions (no edge mask); GNNExplainer carries edge masks (no feature
    # attribution).  The old check required the non-applicable family's columns
    # to be finite too, so a correct real table always tripped the strict check.
    expected_metrics = DRIFT_EDGE_METRICS if explainer == "gnnexplainer" else DRIFT_FEATURE_METRICS
    available = [metric for metric in expected_metrics if f"drift_{metric}" in frame.columns]
    required_count = len(DRIFT_EDGE_METRICS) if explainer == "gnnexplainer" else 4
    report.add(
        "drift_metrics.metric_family",
        PASS if len(available) >= required_count else WARN,
        f"drift columns present: {available}",
        path,
    )
    drift_columns = [f"drift_{metric}" for metric in available]
    numeric = frame[drift_columns].to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    report.add(
        "drift_metrics.finite_rates",
        PASS if finite.size / max(numeric.size, 1) > 0.95 else WARN,
        f"{finite.size}/{numeric.size} finite drift values",
        path,
    )
    primary = frame["drift_primary"].to_numpy(dtype=float)
    if "sim_primary" in frame.columns and np.isfinite(frame["sim_primary"]).any():
        recomputed = 1.0 - frame["sim_primary"].to_numpy(dtype=float)
        bad = int(np.sum(np.abs(primary - recomputed) > ATOL))
        report.add(
            "drift_metrics.primary_definition",
            PASS if bad == 0 else FAIL,
            f"drift_primary == 1 - sim_primary (violations: {bad})",
            path,
        )
    jaccard_primary = "edge_jaccard_top5" if explainer == "gnnexplainer" else "jaccard_top5"
    column = f"drift_{jaccard_primary}"
    if column in frame.columns:
        values = frame[column].to_numpy(dtype=float)
        inside = (values >= -ATOL) & (values <= 1.0 + ATOL)
        report.add(
            "drift_metrics.primary_range",
            PASS if bool(np.all(inside)) else FAIL,
            f"{jaccard_primary} drift in [0,1] (violations: {int((~inside).sum())})",
            path,
        )
    ordering = (frame["checkpoint_j"] <= frame["checkpoint_t"]).all()
    report.add("drift_metrics.checkpoint_order", PASS if ordering else FAIL, "checkpoint_j <= checkpoint_t for every row", path)
    explainers_in_file = set(frame["explainer"].astype(str))
    report.add(
        "drift_metrics.explainer_identity",
        PASS if len(explainers_in_file) == 1 else FAIL,
        f"explainer values: {sorted(explainers_in_file)} (one per file)",
        path,
    )


def validate_noise_floor_csv(report: ValidationReport, path: Path) -> None:
    frame = _read_csv(report, path, artifact="noise_floor")
    if frame is None:
        return
    required = ["channel", "class", "n_null_pairs", "quantile", "q95", "threshold_source"]
    if not _require_columns(report, frame, required, path, artifact="noise_floor"):
        return
    channels = set(frame["channel"].astype(str))
    report.add(
        "noise_floor.channels",
        PASS if channels and channels <= CHANNELS else FAIL,
        f"channels={sorted(channels)} (allowed {sorted(CHANNELS)})",
        path,
    )
    report.add(
        "noise_floor.quantile",
        PASS if bool(np.allclose(frame["quantile"].to_numpy(dtype=float), PROTOCOL_QUANTILE)) else FAIL,
        f"quantile must equal frozen {PROTOCOL_QUANTILE}",
        path,
    )
    sources = set(frame["threshold_source"].astype(str))
    report.add(
        "noise_floor.threshold_source",
        PASS if sources <= {"class", "global", "nan"} else FAIL,
        f"threshold_source values={sorted(sources)}",
        path,
    )
    positive = (frame["n_null_pairs"].to_numpy(dtype=float) > 0).all()
    report.add("noise_floor.n_pairs", PASS if positive else FAIL, "n_null_pairs > 0", path)
    q95 = frame["q95"].to_numpy(dtype=float)
    report.add(
        "noise_floor.q95_range",
        PASS if bool(np.all((q95 >= 0.0) & (q95 <= 1.0))) else WARN,
        "q95 in [0,1] (jaccard-based primary drift is bounded by 1)",
        path,
    )
    # R = 5 null-update reseeds per explainer/class/anchor is the protocol minimum;
    # the per-class pooled count must be an exact multiple of 5 (or 45 repeat pairs).
    if "model_null" in channels:
        subset = frame[frame["channel"] == "model_null"]["n_null_pairs"].to_numpy(dtype=float)
        multiples = np.isclose(np.mod(subset, 5), 0)
        report.add(
            "noise_floor.model_null_multiples",
            PASS if bool(np.all(multiples)) else WARN,
            "model_null pair counts are multiples of R=5",
            path,
        )


def validate_rq1_anchor_level_csv(report: ValidationReport, path: Path) -> None:
    frame = _read_csv(report, path, artifact="rq1_anchor_level")
    if frame is None:
        return
    required = ["strategy", "model", "explainer", "anchor_id", "class", "checkpoint_j", "checkpoint_t",
                "drift_primary", "null_mean", "floor_q95", "excess", "above_floor", "channel"]
    if not _require_columns(report, frame, required, path, artifact="rq1_anchor_level"):
        return
    excess = frame["drift_primary"].to_numpy(dtype=float) - frame["null_mean"].to_numpy(dtype=float)
    reporting = frame["excess"].to_numpy(dtype=float)
    bad_excess = int(np.sum(np.abs(excess - reporting) > ATOL))
    report.add(
        "rq1_anchor_level.excess_definition",
        PASS if bad_excess == 0 else FAIL,
        f"excess == drift_primary - null_mean (violations: {bad_excess})",
        path,
    )
    recomputed_above = frame["drift_primary"].to_numpy(dtype=float) > frame["floor_q95"].to_numpy(dtype=float)
    stored_above = frame["above_floor"].astype(bool).to_numpy()
    mismatched = int(np.sum(recomputed_above != stored_above))
    report.add(
        "rq1_anchor_level.above_floor_rule",
        PASS if mismatched == 0 else FAIL,
        f"above_floor == (drift > q95) strict (violations: {mismatched})",
        path,
    )
    channels = set(frame["channel"].astype(str))
    report.add("rq1_anchor_level.channel", PASS if channels == {"model_null"} else WARN, f"channel={sorted(channels)} (A6 primary floor)", path)


def validate_rq1_aggregated_csv(report: ValidationReport, path: Path) -> None:
    frame = _read_csv(report, path, artifact="rq1_aggregated")
    if frame is None:
        return
    required = ["strategy", "model", "explainer", "class", "checkpoint_j", "checkpoint_t", "n_anchors",
                "drift_real_mean", "drift_null_mean", "excess_mean", "above_floor_fraction",
                "excess_ci_lo", "excess_ci_hi"]
    if not _require_columns(report, frame, required, path, artifact="rq1_aggregated"):
        return
    fraction = frame["above_floor_fraction"].to_numpy(dtype=float)
    report.add(
        "rq1_aggregated.fraction_range",
        PASS if bool(np.all((fraction >= 0.0) & (fraction <= 1.0))) else FAIL,
        "above_floor_fraction in [0,1]",
        path,
    )
    ci_bad = int(np.sum(frame["excess_ci_lo"].to_numpy(dtype=float) > frame["excess_ci_hi"].to_numpy(dtype=float)))
    report.add("rq1_aggregated.ci_ordering", PASS if ci_bad == 0 else FAIL, f"CI lo <= hi (violations: {ci_bad})", path)
    n_positive = bool(np.all(frame["n_anchors"].to_numpy(dtype=float) > 0))
    report.add("rq1_aggregated.n_anchors", PASS if n_positive else FAIL, "n_anchors > 0", path)
    if "expected_fraction" in frame.columns:
        ok = bool(np.allclose(frame["expected_fraction"].to_numpy(dtype=float), PROTOCOL_EXPECTED_ABOVE_FRACTION))
        report.add("rq1_aggregated.expected_fraction", PASS if ok else FAIL, "expected_fraction == 0.05", path)


def validate_rq1_h1_tests_csv(report: ValidationReport, path: Path) -> None:
    frame = _read_csv(report, path, artifact="rq1_h1_tests")
    if frame is None:
        return
    required = ["strategy", "model", "explainer", "n_anchors", "n_above_floor", "fraction_above_floor",
                "fraction_ci_lo", "fraction_ci_hi", "p_binomial", "p_wilcoxon", "p_binomial_holm"]
    if not _require_columns(report, frame, required, path, artifact="rq1_h1_tests"):
        return
    recomputed = frame["n_above_floor"].to_numpy(dtype=float) / frame["n_anchors"].to_numpy(dtype=float)
    bad = int(np.sum(np.abs(recomputed - frame["fraction_above_floor"].to_numpy(dtype=float)) > ATOL))
    report.add("rq1_h1_tests.fraction_consistency", PASS if bad == 0 else FAIL, f"fraction == n_above/n (violations: {bad})", path)
    for column in ("p_binomial", "p_wilcoxon", "p_binomial_holm", "p_wilcoxon_holm"):
        if column not in frame.columns:
            continue
        values = frame[column].to_numpy(dtype=float)
        report.add(
            f"rq1_h1_tests.{column}_range",
            PASS if bool(np.all((values >= 0.0) & (values <= 1.0))) else FAIL,
            "p-values in [0,1]",
            path,
        )
    if {"p_binomial", "p_binomial_holm"} <= set(frame.columns):
        raw = frame["p_binomial"].to_numpy(dtype=float)
        adjusted = frame["p_binomial_holm"].to_numpy(dtype=float)
        bad_holm = int(np.sum(adjusted < raw - ATOL))
        report.add("rq1_h1_tests.holm_monotonicity", PASS if bad_holm == 0 else FAIL, f"Holm-adjusted >= raw (violations: {bad_holm})", path)


def validate_rq1_strategy_comparison_csv(report: ValidationReport, path: Path) -> None:
    frame = _read_csv(report, path, artifact="rq1_strategy_comparison")
    if frame is None:
        return
    required = ["comparison", "baseline", "metric", "n_pairs", "mean_diff", "ci_lo", "ci_hi", "p_value"]
    if not _require_columns(report, frame, required, path, artifact="rq1_strategy_comparison"):
        return
    bad_ci = int(np.sum((frame["ci_lo"].to_numpy(dtype=float) > frame["mean_diff"].to_numpy(dtype=float)) |
                        (frame["ci_hi"].to_numpy(dtype=float) < frame["mean_diff"].to_numpy(dtype=float))))
    report.add("rq1_strategy_comparison.ci_contains_estimate", PASS if bad_ci == 0 else FAIL, f"{bad_ci} interval(s) exclude the point estimate", path)
    p = frame["p_value"].to_numpy(dtype=float)
    report.add("rq1_strategy_comparison.p_range", PASS if bool(np.all((p >= 0.0) & (p <= 1.0))) else FAIL, "p in [0,1]", path)
    report.add("rq1_strategy_comparison.pairs", PASS if bool(np.all(frame["n_pairs"].to_numpy(dtype=float) > 0)) else FAIL, "n_pairs > 0", path)


# ---------------------------------------------------------------------------
# S10 validators
# ---------------------------------------------------------------------------
def validate_rq2_anchor_level_csv(report: ValidationReport, path: Path) -> None:
    frame = _read_csv(report, path, artifact="rq2_anchor_level")
    if frame is None:
        return
    required = ["strategy", "model", "explainer", "anchor_id", "class", "checkpoint_j", "checkpoint_t",
                "excess", "neighbourhood_change"]
    if not _require_columns(report, frame, required, path, artifact="rq2_anchor_level"):
        return
    change = frame["neighbourhood_change"].to_numpy(dtype=float)
    report.add(
        "rq2_anchor_level.change_range",
        PASS if bool(np.all((change >= -ATOL) & (change <= 1.0 + ATOL))) else FAIL,
        "neighbourhood_change = 1 - Jaccard in [0,1]",
        path,
    )


def validate_rq2_correlations_csv(report: ValidationReport, path: Path) -> None:
    frame = _read_csv(report, path, artifact="rq2_correlations")
    if frame is None:
        return
    required = ["strategy", "model", "explainer", "checkpoint_j", "checkpoint_t", "n_anchors",
                "spearman", "spearman_ci_lo", "spearman_ci_hi", "permutation_p_greater"]
    if not _require_columns(report, frame, required, path, artifact="rq2_correlations"):
        return
    for column in ("spearman", "partial_spearman"):
        if column not in frame.columns:
            continue
        values = frame[column].to_numpy(dtype=float)
        report.add(
            f"rq2_correlations.{column}_range",
            PASS if bool(np.all((values >= -1.0 - ATOL) & (values <= 1.0 + ATOL))) else FAIL,
            f"{column} in [-1,1]",
            path,
        )
    p = frame["permutation_p_greater"].to_numpy(dtype=float)
    report.add("rq2_correlations.p_range", PASS if bool(np.all((p >= 0.0) & (p <= 1.0))) else FAIL, "permutation p in [0,1]", path)
    report.add("rq2_correlations.n_anchors", PASS if bool(np.all(frame["n_anchors"].to_numpy(dtype=float) >= 3)) else WARN, "n_anchors >= 3 for reported correlations", path)


# ---------------------------------------------------------------------------
# S11 validators
# ---------------------------------------------------------------------------
def validate_rq3_features_csv(report: ValidationReport, path: Path) -> None:
    frame = _read_csv(report, path, artifact="rq3_features")
    if frame is None:
        return
    required = ["strategy", "cl_model", "checkpoint_t", "class", "label", "mean_excess_drift", "p90_excess_drift"]
    if not _require_columns(report, frame, required, path, artifact="rq3_features"):
        return
    forbidden = [column for column in ("future_f1", "future_checkpoint", "drop") if column in frame.columns]
    report.add(
        "rq3_features.no_future_leakage",
        PASS if not forbidden else FAIL,
        f"future-F1 columns must not be features (found {forbidden}) (docs/05 §11 check)",
        path,
    )
    labels = set(frame["label"].dropna().astype(int))
    report.add("rq3_features.label_binary", PASS if labels <= {0, 1} and labels else FAIL, f"label values={sorted(labels)}", path)
    drift = frame["mean_excess_drift"].to_numpy(dtype=float)
    report.add("rq3_features.drift_finite", PASS if bool(np.all(np.isfinite(drift))) else FAIL, "mean_excess_drift finite", path)


def validate_rq3_forgetting_labels_csv(report: ValidationReport, path: Path) -> None:
    frame = _read_csv(report, path, artifact="rq3_forgetting_labels")
    if frame is None:
        return
    required = ["strategy", "cl_model", "checkpoint_t", "eval_period", "class", "f1", "label", "delta", "horizon"]
    if not _require_columns(report, frame, required, path, artifact="rq3_forgetting_labels"):
        return
    delta_ok = bool(np.allclose(frame["delta"].to_numpy(dtype=float), PROTOCOL_FORGETTING_DELTA))
    report.add("rq3_forgetting_labels.delta", PASS if delta_ok else FAIL, f"delta must equal the frozen {PROTOCOL_FORGETTING_DELTA}", path)
    horizons = set(frame["horizon"].astype(str))
    report.add(
        "rq3_forgetting_labels.horizon",
        PASS if horizons <= {"next_or_final", "any_later"} else FAIL,
        f"horizon values={sorted(horizons)} (pre-registered variants only)",
        path,
    )
    labels = set(frame["label"].dropna().astype(int))
    report.add("rq3_forgetting_labels.label_binary", PASS if labels <= {0, 1} and labels else FAIL, f"label values={sorted(labels)}", path)


def validate_rq3_auc_csv(report: ValidationReport, path: Path) -> None:
    frame = _read_csv(report, path, artifact="rq3_auc")
    if frame is None:
        return
    required = ["strategy", "cl_model", "method", "role", "auc", "ci_lo", "ci_hi"]
    if not _require_columns(report, frame, required, path, artifact="rq3_auc"):
        return
    auc = frame["auc"].to_numpy(dtype=float)
    reference_mask = frame["role"].astype(str) == "reference"
    if reference_mask.any():
        reference_ok = bool(np.allclose(frame.loc[reference_mask, "auc"].to_numpy(dtype=float), 0.5, atol=ATOL))
        report.add("rq3_auc.reference_semantics", PASS if reference_ok else FAIL, "majority reference row(s) have AUC = 0.5", path)
    else:
        report.add("rq3_auc.reference_semantics", WARN, "no majority reference row present", path)
    roles = set(frame["role"].astype(str))
    report.add("rq3_auc.roles", PASS if roles <= {"reference", "competing_baseline", "drift_only", "combined", "delta"} and roles else FAIL, f"roles={sorted(roles)}", path)
    delta = frame[frame["role"].astype(str) == "delta"]
    if not delta.empty:
        delta_values = delta["delta_auc"].to_numpy(dtype=float) if "delta_auc" in delta.columns else delta["auc"].to_numpy(dtype=float)
        report.add("rq3_auc.delta_values", PASS if bool(np.all(np.isfinite(delta_values))) else FAIL, "delta-AUC rows have finite delta", path)
        if "h3_supported" in delta.columns:
            supported = delta["h3_supported"].astype(bool).to_numpy()
            ci_lower = delta["delta_ci_lo"].to_numpy(dtype=float) if "delta_ci_lo" in delta.columns else frame.loc[delta.index, "ci_lo"].to_numpy(dtype=float)
            expected = ci_lower > 0.0
            mismatched = int(np.sum(supported != expected))
            report.add(
                "rq3_auc.h3_flag_rule",
                PASS if mismatched == 0 else FAIL,
                f"h3_supported == (delta CI lower bound > 0) (violations: {mismatched})",
                path,
            )


def validate_rq3_coefficients_csv(report: ValidationReport, path: Path) -> None:
    frame = _read_csv(report, path, artifact="rq3_coefficients")
    if frame is None:
        return
    required = ["model", "feature", "coef_standardized", "ci_lo", "ci_hi"]
    _require_columns(report, frame, required, path, artifact="rq3_coefficients")
    if {"ci_lo", "ci_hi", "coef_standardized"} <= set(frame.columns):
        bad = int(np.sum(frame["ci_lo"].to_numpy(dtype=float) > frame["ci_hi"].to_numpy(dtype=float)))
        report.add("rq3_coefficients.ci_ordering", PASS if bad == 0 else FAIL, f"CI lo <= hi (violations: {bad})", path)


# ---------------------------------------------------------------------------
# S12 validators
# ---------------------------------------------------------------------------
def validate_rq4_comparison_csv(report: ValidationReport, path: Path) -> None:
    frame = _read_csv(report, path, artifact="rq4_comparison")
    if frame is None:
        return
    required = ["model", "variant", "seed", "drift_mean", "macro_f1", "row_kind", "h4_supported"]
    if not _require_columns(report, frame, required, path, artifact="rq4_comparison"):
        return
    summary = frame[frame["row_kind"].astype(str) != "run"]
    report.add(
        "rq4_comparison.variants",
        PASS if {"ear_a", "ear_b"} <= set(frame["variant"].astype(str)) else WARN,
        f"variants={sorted(set(frame['variant'].astype(str)))}",
        path,
    )
    if not summary.empty:
        drift = summary["drift_diff_vs_replay"].to_numpy(dtype=float)
        f1 = summary["f1_diff_vs_replay"].to_numpy(dtype=float)
        expected = (drift < 0.0) & (f1 >= -PROTOCOL_H4_F1_LOSS_POINTS / 100.0)
        stored = summary["h4_supported"].astype(bool).to_numpy()
        mismatched = int(np.sum(expected != stored))
        report.add(
            "rq4_comparison.h4_rule",
            PASS if mismatched == 0 else FAIL,
            f"h4_supported == (drift difference < 0 AND F1 loss <= 1 point) (violations: {mismatched})",
            path,
        )
    runtime_bad = int(np.sum(~(frame["runtime_s"].to_numpy(dtype=float) > 0))) if "runtime_s" in frame.columns else 0
    report.add("rq4_comparison.runtime", PASS if runtime_bad == 0 else WARN, "runtime_s > 0 where present", path)


def validate_rq4_ablations_csv(report: ValidationReport, path: Path) -> None:
    frame = _read_csv(report, path, artifact="rq4_ablations")
    if frame is None:
        return
    required = ["model", "variant", "buffer_size", "beta", "attribution_source", "drift_mean", "macro_f1"]
    if not _require_columns(report, frame, required, path, artifact="rq4_ablations"):
        return
    report.add(
        "rq4_ablations.buffer_grid",
        PASS if set(frame["buffer_size"].astype(int)) <= {500, 2000, 5000} else WARN,
        f"buffer sizes={sorted(set(frame['buffer_size'].astype(int)))}",
        path,
    )
    report.add(
        "rq4_ablations.beta_grid",
        PASS if set(np.round(frame["beta"].astype(float), 6)) <= {0.01, 0.1, 0.5, 1.0} else WARN,
        f"beta values={sorted(set(np.round(frame['beta'].astype(float), 6)))}",
        path,
    )


# ---------------------------------------------------------------------------
# S6/S7 validators — anchor sets, retention table, explanation cache index (T20)
# ---------------------------------------------------------------------------
#: Required anchor-set dict keys (mirrors ``src/anchors/build.py::select_anchors``).
ANCHOR_SET_KEYS: tuple[str, ...] = (
    "period",
    "anchors",
    "n_eligible",
    "n_selected",
    "strata",
    "rare_classes",
    "rare_stratum",
    "class_counts",
    "config",
)

#: Required per-anchor keys (mirrors ``src/anchors/build.py``).
ANCHOR_KEYS: tuple[str, ...] = (
    "anchor_id",
    "period_local_edge_index",
    "period",
    "label",
    "predicted_class",
    "confidence",
    "stratum",
    "host_src_global",
    "host_dst_global",
    "edge_ts",
    "sub_edge_index",
    "sub_edge_attr",
    "sub_edge_ids",
    "sub_node_global_ids",
    "target_local_index",
    "target_edge",
    "n_sub_nodes",
    "n_sub_edges",
)

#: Explanation index columns (mirrors ``src/explain/cache.py::INDEX_COLUMNS``).
EXPLANATION_INDEX_COLUMNS: tuple[str, ...] = (
    "key",
    "checkpoint_sha",
    "anchor_id",
    "explainer",
    "seed",
    "k",
    "config_sha",
    "created_at_utc",
    "npz_path",
    "json_path",
)

#: Per-entry evidence fields required in the canonical S7 index by D-048(c) /
#: docs/05 §6 (the writer emits its own ``src/explain/cache.py`` schema which
#: includes these four columns for both cache copies).
INDEX_PROVENANCE_COLUMNS: tuple[str, ...] = (
    "fidelity_sufficiency",
    "fidelity_comprehensiveness",
    "fidelity_over_random",
    "runtime_s",
)

#: Canonical S6 retention table columns (D-048(b): ``results/text/tables/anchor_retention.csv``).
ANCHOR_RETENTION_COLUMNS: tuple[str, ...] = (
    "strategy",
    "model",
    "class",
    "checkpoint_t",
    "retained_fraction",
)

#: S8 noise channels emitted by ``src/scripts/build_noise_floor.py`` (D-048(d)).
NOISE_CACHE_CHANNELS: tuple[str, ...] = ("model_null", "explainer_repeat")
NOISE_CACHE_EXPECTED_REPEATS: Mapping[str, int] = {"model_null": 5, "explainer_repeat": 10}

#: Engineer explainer slugs (S7 cache) mapped to the metric set they feed.
EXPLAINER_SLUGS: Mapping[str, str] = {"ig": "feature", "gnn_explainer": "edge"}

#: Protocol anchor caps (docs/05 §7).
ANCHOR_MAX_PER_CLASS = 100
ANCHOR_MAX_PER_PERIOD = 1000
ANCHOR_CONFIDENCE_MIN = 0.9


def validate_anchor_artifacts(
    report: ValidationReport,
    anchors_dir: str | Path,
    *,
    expected_periods: Sequence[int] = (1, 2, 3, 4, 5),
) -> None:
    """Structural validation of ``results/text/anchors/period_*.pt`` (S6/T20).

    Checks the stored anchor-set schema, per-anchor fields, global anchor-id
    uniqueness, confidence threshold (docs/05 §7: correctly classified with
    confidence ≥ 0.9), the ≤100/class and ≤1000/period caps, subgraph integrity
    (target edge inside the stored subgraph), and that every frozen period file
    exists. Tensor contents are checked structurally only; provenance hashes stay
    a run-manifest concern.
    """
    directory = Path(anchors_dir)
    files = sorted(directory.glob("period_*.pt")) if directory.is_dir() else []
    if not files:
        report.add("anchors.period_files", FAIL, f"no period_*.pt anchor sets under {directory}", directory)
        return
    try:
        import torch
    except Exception as exc:  # pragma: no cover - torch is present on every lane
        report.add("anchors.torch", WARN, f"torch unavailable, anchor tensors not inspected: {exc}", directory)
        return
    periods_found: list[int] = []
    anchor_ids: list[int] = []
    problems: list[str] = []
    anchor_counts: list[int] = []
    for path in files:
        try:
            payload = torch.load(path, weights_only=False)
        except Exception as exc:
            report.add("anchors.readable", FAIL, f"cannot load {path.name}: {exc}", path)
            continue
        if not isinstance(payload, Mapping):
            problems.append(f"{path.name}: not a mapping")
            continue
        missing = [key for key in ANCHOR_SET_KEYS if key not in payload]
        if missing:
            problems.append(f"{path.name}: missing keys {missing}")
            continue
        period = int(payload["period"])
        periods_found.append(period)
        anchors = list(payload.get("anchors") or [])
        anchor_counts.append(len(anchors))
        if not anchors:
            problems.append(f"{path.name}: empty anchor list")
            continue
        if int(payload.get("n_selected", -1)) != len(anchors):
            problems.append(f"{path.name}: n_selected={payload.get('n_selected')} != len(anchors)={len(anchors)}")
        strata_block = payload.get("strata")
        if isinstance(strata_block, dict) and strata_block:
            try:
                strata_sum = sum(int(value) for value in strata_block.values())
            except (TypeError, ValueError):
                problems.append(f"{path.name}: strata counts are not integers")
            else:
                if strata_sum != len(anchors):
                    problems.append(
                        f"{path.name}: sum(strata)={strata_sum} != len(anchors)={len(anchors)} (D-033 post-cap provenance)"
                    )
        if len(anchors) > ANCHOR_MAX_PER_PERIOD:
            problems.append(f"{path.name}: {len(anchors)} anchors > max_per_period={ANCHOR_MAX_PER_PERIOD}")
        per_class: dict[int, int] = {}
        for anchor in anchors:
            missing_anchor = [key for key in ANCHOR_KEYS if key not in anchor]
            if missing_anchor:
                problems.append(f"{path.name}: anchor missing keys {missing_anchor}")
                continue
            anchor_id = int(anchor["anchor_id"])
            anchor_ids.append(anchor_id)
            if int(anchor["period"]) != period:
                problems.append(f"{path.name}: anchor {anchor_id} period mismatch")
            label = int(anchor["label"])
            predicted = int(anchor["predicted_class"])
            if not 0 <= label <= 9 or not 0 <= predicted <= 9:
                problems.append(f"{path.name}: anchor {anchor_id} label/predicted outside 10-class space")
            confidence = float(anchor["confidence"])
            if confidence < ANCHOR_CONFIDENCE_MIN - 1e-9:
                problems.append(f"{path.name}: anchor {anchor_id} confidence {confidence:.4f} < {ANCHOR_CONFIDENCE_MIN}")
            per_class[label] = per_class.get(label, 0) + 1
            sub_edges = anchor["sub_edge_ids"].tolist() if hasattr(anchor["sub_edge_ids"], "tolist") else list(anchor["sub_edge_ids"])
            target = anchor["target_edge"].tolist() if hasattr(anchor["target_edge"], "tolist") else list(anchor["target_edge"])
            if not set(target).issubset(set(sub_edges)):
                problems.append(f"{path.name}: anchor {anchor_id} target edge not in stored subgraph")
            if int(anchor["n_sub_edges"]) != len(sub_edges):
                problems.append(f"{path.name}: anchor {anchor_id} n_sub_edges != len(sub_edge_ids)")
            if int(anchor["n_sub_nodes"]) < 2:
                problems.append(f"{path.name}: anchor {anchor_id} subgraph has < 2 nodes")
        for class_id, count in per_class.items():
            if count > ANCHOR_MAX_PER_CLASS:
                problems.append(f"{path.name}: class {class_id} has {count} anchors > {ANCHOR_MAX_PER_CLASS}")
    duplicates = {value for value in anchor_ids if anchor_ids.count(value) > 1}
    if duplicates:
        problems.append(f"duplicate anchor ids across periods: {sorted(duplicates)[:5]}")
    missing_periods = sorted(set(expected_periods) - set(periods_found))
    if missing_periods:
        report.add("anchors.periods", WARN, f"missing period anchor set(s): {missing_periods}", directory)
    else:
        report.add("anchors.periods", PASS, f"anchor sets present for periods {sorted(periods_found)}", directory)
    report.add(
        "anchors.structure",
        PASS if not problems else FAIL,
        f"{len(files)} period file(s), {len(anchor_ids)} anchor(s) [per-period counts {anchor_counts}]; problems={problems[:4]}",
        directory,
    )


def validate_retained_fraction_csv(report: ValidationReport, path: Path) -> None:
    """Validate the S6 retained-fraction table (docs/05 §7 step 4; T20)."""
    frame = _read_csv(report, path, artifact="retained_fraction")
    if frame is None:
        return
    required = ["anchor_period", "later_checkpoint", "retained", "total", "retained_fraction"]
    if not _require_columns(report, frame, required, path, artifact="retained_fraction"):
        return
    problems: list[str] = []
    retained = frame["retained"].to_numpy(dtype=float)
    total = frame["total"].to_numpy(dtype=float)
    fraction = frame["retained_fraction"].to_numpy(dtype=float)
    if np.any(total <= 0):
        problems.append("total <= 0 rows")
    if np.any(retained < 0) or np.any(retained > total):
        problems.append("retained outside [0, total]")
    expected = np.divide(retained, total, out=np.zeros_like(retained), where=total > 0)
    if np.max(np.abs(expected - fraction)) > 1e-6:
        problems.append("retained_fraction != retained/total")
    if np.any((frame["anchor_period"].to_numpy(dtype=float) >= frame["later_checkpoint"].to_numpy(dtype=float))):
        problems.append("anchor_period >= later_checkpoint row (retention must be a later checkpoint)")
    report.add(
        "retained_fraction.values",
        PASS if not problems else FAIL,
        f"{len(frame)} row(s); problems={problems[:3]}",
        path,
    )


def _resolve_cache_path(value: str, index_path: Path) -> Path:
    """Resolve an explanation cache path from an index row (shared T49 helper)."""
    from src.utils.artifact_paths import resolve_artifact_path

    return resolve_artifact_path(value, index_path)


def validate_explain_index_csv(report: ValidationReport, path: Path, *, require_entries: bool = True) -> None:
    """Validate the S7 explanation-cache index (docs/05 §6; Doc 2b S7; T20).

    Checks the frozen cache-key tuple fields, explainer slugs, k values
    (IG: 5/10; GNNExplainer: edge mask k=0), entry-file existence and loadability
    (``attributions`` for IG, ``edge_mask`` for GNNExplainer), per-entry metadata
    provenance (checkpoint/anchor period, class, strategy/model) and the
    docs/05 §6 requirement that fidelity + runtime are recorded per explanation
    (currently a WARN if the writer only keeps fidelity in the S7 summary).
    """
    if not path.is_file():
        report.add("explain_index.present", FAIL, "index file missing", path)
        return
    try:
        frame = pd.read_csv(path)
    except Exception as exc:  # pragma: no cover - corrupt index
        report.add("explain_index.readable", FAIL, f"unreadable CSV: {exc}", path)
        return
    report.add("explain_index.present", PASS, f"{len(frame)} row(s)", path)
    missing = [column for column in EXPLANATION_INDEX_COLUMNS if column not in frame.columns]
    if missing:
        report.add("explain_index.columns", FAIL, f"missing index column(s) {missing}", path)
        return
    report.add("explain_index.columns", PASS, f"all {len(EXPLANATION_INDEX_COLUMNS)} cache-key columns present", path)
    # D-048(c) / docs/05 §6: per-entry fidelity + runtime are REQUIRED evidence fields.
    missing_provenance_columns = [column for column in INDEX_PROVENANCE_COLUMNS if column not in frame.columns]
    if missing_provenance_columns:
        report.add(
            "explain_index.provenance_columns",
            FAIL,
            f"missing required per-entry evidence column(s) {missing_provenance_columns} (D-048(c): fidelity_* + runtime_s)",
            path,
        )
    else:
        fidelity_ok = True
        problems: list[str] = []
        for column in INDEX_PROVENANCE_COLUMNS:
            if column == "runtime_s":
                values = pd.to_numeric(frame[column], errors="coerce")
                if values.isna().any() or (values <= 0).any():
                    problems.append(f"{column} must be > 0 and numeric")
            elif column == "fidelity_over_random":
                if pd.to_numeric(frame[column], errors="coerce").isna().any():
                    problems.append(f"{column} must be numeric")
            else:
                values = pd.to_numeric(frame[column], errors="coerce")
                if values.isna().any() or ((values < -1e-9) | (values > 1.0 + 1e-9)).any():
                    problems.append(f"{column} must be in [0,1]")
        if problems:
            fidelity_ok = False
        report.add(
            "explain_index.provenance_columns",
            PASS if fidelity_ok else FAIL,
            f"per-entry fidelity/runtime evidence present; problems={problems[:3]}",
            path,
        )
    if frame.empty:
        report.add(
            "explain_index.entries",
            FAIL if require_entries else WARN,
            "index has no entries (S7 explanation run has not produced cached explanations yet)",
            path,
        )
        return
    report.add("explain_index.entries", PASS, f"{len(frame)} cached explanation(s)", path)
    duplicates = frame["key"][frame["key"].duplicated()].tolist()
    report.add(
        "explain_index.key_uniqueness",
        PASS if not duplicates else FAIL,
        f"duplicate key(s): {duplicates[:3]}",
        path,
    )
    explainers = set(frame["explainer"].astype(str))
    unknown = explainers - set(EXPLAINER_SLUGS)
    report.add(
        "explain_index.explainer_names",
        PASS if not unknown else FAIL,
        f"explainer values={sorted(explainers)} (expected {sorted(EXPLAINER_SLUGS)})",
        path,
    )
    k_problems: list[str] = []
    for _, row in frame.iterrows():
        explainer = str(row["explainer"])
        k = int(row["k"])
        if explainer == "ig" and k not in (5, 10):
            k_problems.append(f"ig k={k}")
        if explainer == "gnn_explainer" and k != 0:
            k_problems.append(f"gnn_explainer k={k}")
        if int(row["seed"]) < 0:
            k_problems.append(f"negative seed {row['seed']}")
    report.add("explain_index.k_values", PASS if not k_problems else FAIL, f"k/seed issues: {k_problems[:4]}", path)
    missing_files: list[str] = []
    bad_arrays: list[str] = []
    metadata_problems: list[str] = []
    missing_provenance: list[str] = []
    inspected = 0
    for _, row in frame.head(500).iterrows():  # cap for large caches; sampled structural check
        npz = _resolve_cache_path(row["npz_path"], path)
        meta = _resolve_cache_path(row["json_path"], path)
        if not npz.is_file() or not meta.is_file():
            missing_files.append(row["key"])
            continue
        inspected += 1
        try:
            with np.load(npz) as archive:
                expected_array = "attributions" if str(row["explainer"]) == "ig" else "edge_mask"
                if expected_array not in archive.files:
                    bad_arrays.append(f"{row['key']}: missing array {expected_array}")
                elif not np.all(np.isfinite(archive[expected_array])):
                    bad_arrays.append(f"{row['key']}: non-finite {expected_array}")
        except Exception as exc:
            bad_arrays.append(f"{row['key']}: unreadable npz ({exc})")
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except Exception as exc:
            metadata_problems.append(f"{row['key']}: unreadable json ({exc})")
            continue
        for key in ("explainer", "anchor_id", "checkpoint_period", "anchor_period"):
            if key not in payload:
                metadata_problems.append(f"{row['key']}: metadata missing {key}")
        if "fidelity" not in payload or "runtime_s" not in payload:
            missing_provenance.append(row["key"])
    report.add(
        "explain_index.entry_files",
        PASS if not missing_files else FAIL,
        f"{inspected} entry pair(s) inspected; missing={missing_files[:3]}",
        path,
    )
    report.add("explain_index.entry_arrays", PASS if not bad_arrays else FAIL, f"array issues={bad_arrays[:3]}", path)
    report.add(
        "explain_index.entry_metadata",
        PASS if not metadata_problems else FAIL,
        f"metadata issues={metadata_problems[:3]}",
        path,
    )
    report.add(
        "explain_index.metadata_fidelity_runtime",
        WARN if missing_provenance else PASS,
        "per-entry metadata duplicates fidelity/runtime (informational; the canonical index columns are authoritative per D-048(c)): "
        f"{len(missing_provenance)}/{inspected} entries omit them"
        if missing_provenance
        else "per-entry metadata also carries fidelity + runtime",
        path,
    )


def validate_dual_index_consistency(report: ValidationReport, results_root: str | Path) -> None:
    """D-048(a): canonical ``tables/explain_index.csv`` and operational
    ``explanations/index.csv`` must carry the same cache-key rows from the same run."""
    results_root = Path(results_root)
    canonical = results_root / "tables" / "explain_index.csv"
    operational = results_root / "explanations" / "index.csv"
    if not canonical.is_file() or not operational.is_file():
        report.add(
            "explain_index.dual_copy",
            WARN,
            f"dual index copies not both present yet (canonical={canonical.is_file()}, operational={operational.is_file()}) — D-048 requires both from the same run",
            canonical,
        )
        return
    try:
        canonical_frame = pd.read_csv(canonical)
        operational_frame = pd.read_csv(operational)
    except Exception as exc:  # pragma: no cover - corrupt files
        report.add("explain_index.dual_copy", FAIL, f"cannot read index copies: {exc}", canonical)
        return
    canonical_keys = set(canonical_frame["key"].astype(str)) if "key" in canonical_frame.columns else set()
    operational_keys = set(operational_frame["key"].astype(str)) if "key" in operational_frame.columns else set()
    same = canonical_keys == operational_keys and len(canonical_keys) > 0
    report.add(
        "explain_index.dual_copy",
        PASS if same else FAIL,
        f"canonical rows={len(canonical_keys)}, operational rows={len(operational_keys)}, "
        f"key sets equal={canonical_keys == operational_keys} (D-048 requires same cache-key rows)",
        canonical,
    )


def validate_anchor_retention_csv(report: ValidationReport, path: Path) -> None:
    """Validate the canonical S6 retention table ``results/text/tables/anchor_retention.csv``.

    Schema (D-048(b)): ``strategy,model,class,checkpoint_t,retained_fraction`` —
    one row per (run, class, later checkpoint); fractions in [0,1], unique keys,
    and non-increasing with later checkpoints for each class (retention cannot
    improve when more updates have happened).
    """
    frame = _read_csv(report, path, artifact="anchor_retention")
    if frame is None:
        return
    if not _require_columns(report, frame, ANCHOR_RETENTION_COLUMNS, path, artifact="anchor_retention"):
        return
    problems: list[str] = []
    fraction = frame["retained_fraction"].to_numpy(dtype=float)
    if np.isnan(fraction).any() or ((fraction < -1e-9) | (fraction > 1.0 + 1e-9)).any():
        problems.append("retained_fraction outside [0,1] or NaN")
    if frame.duplicated(subset=["strategy", "model", "class", "checkpoint_t"]).any():
        problems.append("duplicate (strategy, model, class, checkpoint_t) rows")
    for key, group in frame.groupby(["strategy", "model", "class"], dropna=False):
        ordered = group.sort_values("checkpoint_t")
        values = ordered["retained_fraction"].to_numpy(dtype=float)
        if np.any(np.diff(values) > 1e-9):
            problems.append(f"retained_fraction increases for {key}")
    report.add(
        "anchor_retention.values",
        PASS if not problems else FAIL,
        f"{len(frame)} row(s); problems={problems[:3]}",
        path,
    )


def _relocate_results_artifact(value: Any, results_root: Path) -> Path:
    """Resolve an artifact path recorded by a remote lane (e.g. Kaggle /kaggle/...).

    The authoritative absolute path is used when it exists; otherwise the suffix after
    ``results/`` is relocated under the local results root (T37 downloaded artifacts).
    """
    raw = str(value)
    path = Path(raw)
    if path.is_file():
        return path
    posix_absolute = raw.startswith("/") and not raw.startswith("//")
    if not (path.is_absolute() or posix_absolute):
        return path
    parts = path.parts
    for index in range(len(parts) - 1, -1, -1):
        if parts[index] == "results":
            candidate = results_root.joinpath(*parts[index + 1 :])
            if candidate.is_file():
                return candidate
            break
    return path


def validate_anchors_manifest_json(report: ValidationReport, path: Path) -> None:
    """Validate the S6 anchors manifest (dataset/strategy/model/seed/period files + canonical retention)."""
    payload = _read_json(report, path, artifact='anchors_manifest')
    if payload is None:
        return
    required = ('dataset', 'strategy', 'model', 'seed', 'periods')
    missing = [key for key in required if key not in payload]
    if missing:
        report.add('anchors_manifest.schema', FAIL, f'missing keys {missing}', path)
        return
    results_root = path.parent.parent if path.parent.name in {"anchors", "tables"} else path.parent
    periods = payload.get('periods') or {}
    missing_files = [
        str(entry.get('path'))
        for entry in periods.values()
        if isinstance(entry, Mapping)
        and entry.get('path')
        and not _relocate_results_artifact(entry.get('path'), results_root).is_file()
    ]
    report.add('anchors_manifest.schema', PASS, f"periods={sorted(periods)}, dataset={payload.get('dataset')}", path)
    report.add(
        'anchors_manifest.period_files',
        PASS if not missing_files else FAIL,
        f'period file(s) missing: {missing_files[:3]}' if missing_files else 'all referenced period files exist',
        path,
    )
    canonical = payload.get('canonical_anchor_retention')
    if canonical:
        report.add(
            'anchors_manifest.canonical_retention',
            PASS if _relocate_results_artifact(canonical, results_root).is_file() else FAIL,
            f'canonical anchor_retention={canonical}',
            path,
        )


def validate_noise_cache_artifacts(report: ValidationReport, results_root: str | Path) -> None:
    """Validate the S8 noise-cache artifacts required before the first S8/S9 run.

    docs/05 §9 + Doc 2b S8 + D-048(d): ``results/text/explanations_noise/model_null/``
    (R = 5 same-period reseeds per anchor × explainer), ``.../explainer_repeat/``
    (R = 10 seeded repeats), each with a cache index carrying the S7 evidence
    fields, plus ``combined_null.csv`` (union of the two channel indexes with a
    ``channel`` column). IG-repeat canary (exactly 0 drift) is checked from the
    noise-floor manifest when present.
    """
    results_root = Path(results_root)
    noise_root = results_root / "explanations_noise"
    for channel in NOISE_CACHE_CHANNELS:
        index = noise_root / channel / "index.csv"
        if not index.is_file():
            report.add(f"noise_cache.{channel}.present", FAIL, f"missing {index}", index)
            continue
        report.add(f"noise_cache.{channel}.present", PASS, f"{index}", index)
        try:
            frame = pd.read_csv(index)
        except Exception as exc:  # pragma: no cover
            report.add(f"noise_cache.{channel}.readable", FAIL, f"unreadable: {exc}", index)
            continue
        required = list(EXPLANATION_INDEX_COLUMNS) + list(INDEX_PROVENANCE_COLUMNS)
        missing_columns = [column for column in required if column not in frame.columns]
        report.add(
            f"noise_cache.{channel}.columns",
            PASS if not missing_columns else FAIL,
            f"missing={missing_columns}" if missing_columns else f"all {len(required)} index+evidence columns present",
            index,
        )
        if frame.empty:
            report.add(f"noise_cache.{channel}.entries", FAIL, "index has no entries", index)
            continue
        expected_repeats = NOISE_CACHE_EXPECTED_REPEATS[channel]
        bad_counts: list[str] = []
        for key, group in frame.groupby(["anchor_id", "explainer"], dropna=False):
            if len(group) != expected_repeats:
                bad_counts.append(f"{key}: {len(group)} entries != R={expected_repeats}")
        report.add(
            f"noise_cache.{channel}.repeat_counts",
            PASS if not bad_counts else FAIL,
            f"per (anchor, explainer) counts must equal R={expected_repeats}; violations={bad_counts[:3]}",
            index,
        )
        missing_entries: list[str] = []
        bad_arrays: list[str] = []
        for _, row in frame.head(500).iterrows():
            npz = _resolve_cache_path(row["npz_path"], index)
            meta = _resolve_cache_path(row["json_path"], index)
            if not npz.is_file() or not meta.is_file():
                missing_entries.append(str(row["key"]))
                continue
            try:
                with np.load(npz) as archive:
                    expected_array = "attributions" if str(row["explainer"]) == "ig" else "edge_mask"
                    if expected_array not in archive.files:
                        bad_arrays.append(f"{row['key']}: missing {expected_array}")
            except Exception as exc:  # pragma: no cover
                bad_arrays.append(f"{row['key']}: unreadable npz ({exc})")
        report.add(
            f"noise_cache.{channel}.entry_files",
            PASS if not missing_entries and not bad_arrays else FAIL,
            f"missing files={missing_entries[:3]}, array issues={bad_arrays[:3]}",
            index,
        )
    combined = noise_root / "combined_null.csv"
    if not combined.is_file():
        report.add("noise_cache.combined.present", FAIL, f"missing {combined}", combined)
    else:
        try:
            frame = pd.read_csv(combined)
        except Exception as exc:  # pragma: no cover
            report.add("noise_cache.combined.readable", FAIL, f"unreadable: {exc}", combined)
            frame = None
        if frame is not None:
            required = list(EXPLANATION_INDEX_COLUMNS) + ["channel"]
            missing = [column for column in required if column not in frame.columns]
            channels = set(frame["channel"].astype(str)) if "channel" in frame.columns else set()
            report.add(
                "noise_cache.combined.schema",
                PASS if not missing and channels <= set(NOISE_CACHE_CHANNELS) and not frame.empty else FAIL,
                f"rows={len(frame)}, columns missing={missing}, channels={sorted(channels)}",
                combined,
            )
    manifest = results_root / "data_audit" / "noise_floor_manifest.json"
    if manifest.is_file():
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:  # pragma: no cover
            payload = {}
        channels = payload.get("channels") or payload.get("results") or {}
        canaries = [entry.get("ig_repeat_canary_drift_max") for entry in (channels.values() if isinstance(channels, Mapping) else []) if isinstance(entry, Mapping)]
        canaries = [value for value in canaries if value is not None]
        if canaries:
            report.add(
                "noise_cache.ig_canary",
                PASS if all(abs(float(value)) <= 1e-12 for value in canaries) else FAIL,
                f"IG repeated-explanation drift must be exactly 0; observed={canaries[:3]}",
                manifest,
            )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------
#: filename -> validator callable (special-cased for explainer-named drift files).
VALIDATORS: Mapping[str, Callable[..., None]] = {
    "cl_matrix.json": validate_cl_matrix_json,
    "cl_metrics.csv": validate_cl_metrics_csv,
    "cl_per_class_f1.json": validate_per_class_json,
    "per_class_f1.csv": validate_per_class_f1_csv,
    "per_class_f1.npz": validate_per_class_f1_npz,
    "noise_floor.csv": validate_noise_floor_csv,
    "rq1_anchor_level.csv": validate_rq1_anchor_level_csv,
    "rq1_aggregated.csv": validate_rq1_aggregated_csv,
    "rq1_h1_tests.csv": validate_rq1_h1_tests_csv,
    "rq1_strategy_comparison.csv": validate_rq1_strategy_comparison_csv,
    "rq2_anchor_level.csv": validate_rq2_anchor_level_csv,
    "rq2_correlations.csv": validate_rq2_correlations_csv,
    "rq3_features.csv": validate_rq3_features_csv,
    "rq3_forgetting_labels.csv": validate_rq3_forgetting_labels_csv,
    "rq3_auc.csv": validate_rq3_auc_csv,
    "rq3_coefficients.csv": validate_rq3_coefficients_csv,
    "rq4_comparison.csv": validate_rq4_comparison_csv,
    "rq4_ablations.csv": validate_rq4_ablations_csv,
}


def _resolve_required_spec(tables_root: Path, spec: str) -> tuple[Path | None, bool]:
    """Resolve a required-artifact spec; returns (path, is_canonical_first_choice)."""
    candidates = [candidate.strip() for candidate in spec.split("|") if candidate.strip()]
    for index, candidate in enumerate(candidates):
        if "/" in candidate:
            matches = sorted(tables_root.parent.glob(candidate))
            if matches:
                return matches[0], index == 0
        else:
            path = tables_root / candidate
            if path.is_file():
                return path, index == 0
    return None, True


def _validate_required_path(report: ValidationReport, path: Path, spec: str, *, allow_partial_s5: bool) -> None:
    """Dispatch a resolved required artifact to its validator."""
    canonical = spec.split("|")[0].strip()
    name = path.name
    if canonical.startswith("anchors/") and canonical.endswith("*.pt"):
        validate_anchor_artifacts(report, path.parent)
    elif canonical == "tables/anchor_retention.csv":
        validate_anchor_retention_csv(report, path)
    elif "retained_fraction" in name:
        validate_retained_fraction_csv(report, path)
    elif name == "index.csv" or canonical == "tables/explain_index.csv":
        validate_explain_index_csv(report, path)
    elif name == "anchors_manifest.json":
        validate_anchors_manifest_json(report, path)
    elif name.startswith("drift_metrics_") and name.endswith(".csv"):
        validate_drift_metrics_csv(report, path, explainer=name[len("drift_metrics_") : -len(".csv")])
    elif name == "cl_matrix.json":
        validate_cl_matrix_json(report, path, allow_partial=allow_partial_s5)
    elif name == "cl_per_class_f1.json":
        validate_per_class_json(report, path, matrix_path=path.parent / "cl_matrix.json")
    elif name in VALIDATORS:
        VALIDATORS[name](report, path)
    else:  # pragma: no cover - every registered requirement has a validator
        report.add(f"required.unroutable.{name}", WARN, f"no specific validator for {spec!r}", path)


def validate_artifact(
    results_tables: str | Path,
    filename: str,
    *,
    allow_partial_s5: bool = True,
) -> ValidationReport:
    """Validate one artifact file (or S6/S7 spec) by name (convenience entry point)."""
    root = Path(results_tables)
    report = ValidationReport(results_tables=str(root), stage="single")
    if filename.startswith("anchors/"):
        anchors_dir = root.parent / "anchors" if root.name == "tables" else root / "anchors"
        if filename.rstrip("/").endswith("anchors"):
            validate_anchor_artifacts(report, anchors_dir)
        else:
            path = (root.parent / filename) if root.name == "tables" else (root / filename)
            _validate_required_path(report, path, filename, allow_partial_s5=allow_partial_s5)
    elif filename.startswith("explanations/"):
        path = (root.parent / filename) if root.name == "tables" else (root / filename)
        _validate_required_path(report, path, filename, allow_partial_s5=allow_partial_s5)
    elif filename in {"retained_fraction.csv", "anchor_retention.csv"}:
        candidates = [root / filename, root.parent / "anchors" / "retained_fraction.csv", root / "anchor_retention.csv"]
        existing = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        validate_retained_fraction_csv(report, existing)
    elif filename.startswith("explanations_noise") or filename == "noise_cache":
        validate_noise_cache_artifacts(report, root.parent)
    elif filename in {"index.csv", "explain_index.csv"}:
        candidates = [root / filename, root.parent / "explanations" / "index.csv", root / "explain_index.csv"]
        existing = next((candidate for candidate in candidates if candidate.is_file()), candidates[0])
        validate_explain_index_csv(report, existing)
    elif filename.startswith("drift_metrics_") and filename.endswith(".csv"):
        stem = filename[len("drift_metrics_") : -len(".csv")]
        validate_drift_metrics_csv(report, root / filename, explainer=stem)
    elif filename in VALIDATORS:
        if filename == "cl_matrix.json":
            validate_cl_matrix_json(report, root / filename, allow_partial=allow_partial_s5)
        elif filename == "cl_per_class_f1.json":
            validate_per_class_json(report, root / filename, matrix_path=root / "cl_matrix.json")
        else:
            VALIDATORS[filename](report, root / filename)
    else:
        raise ValidatorError(f"no validator registered for {filename!r}")
    return report


def validate_artifact_set(
    results_tables: str | Path,
    *,
    stage: str = "all",
    allow_partial_s5: bool = True,
    strict: bool = False,
) -> ValidationReport:
    """Validate every artifact required up to and including *stage*.

    ``stage='s5'`` … ``'s12'`` are cumulative (docs/02 stage order, including the
    S6 anchor sets and S7 explanation cache added in T20); ``'all'`` validates the
    full S13 contract. Missing required artifacts are `fail`; missing
    optional/supplementary artifacts are `warn`; an intermediate partial S5
    matrix is `warn` unless ``allow_partial_s5=False``.
    """
    if stage not in STAGE_REQUIREMENTS:
        raise ValidatorError(f"unknown stage {stage!r}; expected one of {sorted(STAGE_REQUIREMENTS)}")
    root = Path(results_tables)
    report = ValidationReport(results_tables=str(root), stage=stage)
    if stage == "all":
        required: tuple[str, ...] = STAGE_REQUIREMENTS["all"]
    else:
        index = STAGE_ORDER.index(stage)
        required = tuple(dict.fromkeys(name for current in STAGE_ORDER[: index + 1] for name in STAGE_REQUIREMENTS[current]))
    optional = OPTIONAL_BY_STAGE.get(stage, ())
    _done_builtin: set[str] = set()
    for spec in required:
        path, canonical = _resolve_required_spec(root, spec)
        if path is None:
            report.add(f"required.present.{spec}", FAIL, f"required artifact missing for stage {stage}", root / spec.split('|')[0])
            continue
        report.add(f"required.present.{spec}", PASS, f"present at {path}", path)
        if not canonical:
            report.add(
                f"required.path_convention.{spec}",
                WARN,
                "canonical path not found; alternative path accepted — confirm/record the path convention (D-048/T20)",
                path,
            )
        canonical_spec = spec.split('|')[0].strip()
        if canonical_spec.startswith("explanations_noise/"):
            if "noise_cache" not in _done_builtin:
                validate_noise_cache_artifacts(report, root.parent)
                _done_builtin.add("noise_cache")
        else:
            _validate_required_path(report, path, spec, allow_partial_s5=allow_partial_s5)
    for spec in optional:
        path, _ = _resolve_required_spec(root, spec)
        if path is not None:
            report.add(f"optional.present.{spec}", PASS, "supplementary artifact present", path)
            _validate_required_path(report, path, spec, allow_partial_s5=allow_partial_s5)
        else:
            report.add(f"optional.present.{spec}", WARN, "supplementary artifact not present", root / spec.split('|')[0])
    if stage in {"s7", "s8", "s9", "s10", "s11", "s12", "all"}:
        validate_dual_index_consistency(report, root.parent)
    if strict:
        report.findings = [
            Finding(finding.check, FAIL, "strict mode: " + finding.detail, finding.path) if finding.status == WARN else finding
            for finding in report.findings
        ]
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def render_summary(report: ValidationReport) -> str:
    lines = [
        f"validators: {report.overall.upper()} — stage={report.stage} root={report.results_tables}",
        f"checks: {report.counts}",
    ]
    for finding in report.findings:
        if finding.status != PASS:
            lines.append(f"  [{finding.status}] {finding.check}: {finding.detail}")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate analysis artifact schemas and invariants (T19).")
    parser.add_argument("--results-tables", default=str(Path("results") / "tables"))
    parser.add_argument("--stage", default="all", choices=sorted(STAGE_REQUIREMENTS))
    parser.add_argument("--strict", action="store_true", help="treat warnings as failures")
    parser.add_argument("--allow-partial-s5", action="store_true", default=True)
    parser.add_argument("--json", action="store_true", help="machine-readable report")
    args = parser.parse_args(argv)
    report = validate_artifact_set(
        args.results_tables,
        stage=args.stage,
        allow_partial_s5=args.allow_partial_s5,
        strict=args.strict,
    )
    if args.json:
        print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    else:
        print(render_summary(report))
    return 0 if report.overall != FAIL else 1


__all__ = [
    "Finding",
    "STAGE_ORDER",
    "STAGE_REQUIREMENTS",
    "VALIDATORS",
    "ValidationReport",
    "ValidatorError",
    "render_summary",
    "validate_anchor_retention_csv",
    "validate_anchors_manifest_json",
    "validate_artifact",
    "validate_artifact_set",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
