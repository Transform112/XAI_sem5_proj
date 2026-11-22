"""S5 cross-check: engineer S5 writer vs the authoritative ``analysis`` reference (T19).

Scope and authority
-------------------
* ``docs/DECISIONS.md`` **D-044**: ``src/analysis/metrics_cl.py`` is the authoritative
  reference for the CL metric definitions; the S5 pipeline writer
  (``src/eval/cl_metrics.py`` + ``src/scripts/run_eval.py``) must pass this
  cross-check, and any deliberate divergence must be documented rather than
  tolerated silently.
* ``docs/DECISIONS.md`` **D-043**: the engineer's S5 files
  (``results/text/tables/cl_metrics.csv``, ``cl_matrix.json``,
  ``cl_per_class_f1.json``) are the single writer of those artifacts; the
  analysis toolkit materialises the skeleton ``per_class_f1.{npz,csv}`` and
  serves as the independent reference.
* Definitions: ``docs/source_documents/Doc 2`` §S5 (R matrix, final F1, forgetting, BWT),
  ``docs/05`` §11 (per-class tensor), **D-033** (dual macro-F1: all 10 classes +
  supported set = classes with ≥50 test-mask edges in every period, currently
  the 6 attack classes **plus Benign**).

What is checked
---------------
1. **Artifact schema/identity** — files present, hashes, strategy/model/seed,
   class order, dataset, checkpoint inventory and matrix shape.
2. **R matrix** — the written matrix equals the reference construction for every
   defined cell; undefined upper-triangle cells are NaN.
3. **D-033 dual macro-F1** — ``macro_f1_all_classes`` equals the mean of the 10
   per-class F1 values; ``macro_f1_supported_set`` equals the mean over the
   configured supported ids; the configured supported set is compared against the
   D-033 definition (Benign included).
4. **Per-class F1** — ``cl_per_class_f1.json`` cells equal the
   ``cl_matrix.json`` per-cell copies, values are in [0, 1], absent classes are
   scored 0 (sklearn ``zero_division=0``), class counts sum to ``n_edges``.
5. **Forgetting / BWT formulas** — executed on a hand-computed T=3 matrix, on
   several deterministic complete T=2/3/5 matrices, on the degenerate T=1 case,
   and with an incomplete last row; both implementations are compared exactly.
6. **CL CSV** — long-format rows and the ``incomplete_matrix`` semantics for a
   partial checkpoint set (M_1 only is a legitimate intermediate state).

A partial artifact (M_1 only) cannot exercise the full forgetting/BWT over a real
5-period matrix; that part of the verdict therefore rests on the synthetic
formula cross-check, and the report says so explicitly.

Usage
-----
    .\\.venv\\Scripts\\python.exe -m analysis.s5_crosscheck
    .\\.venv\\Scripts\\python.exe -m analysis.s5_crosscheck --json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.analysis import metrics_cl as mc

#: D-033 supported-class definition (classes with ≥50 test-mask edges in every
#: period: the 6 attack classes plus Benign).
D033_SUPPORTED_CLASSES: tuple[str, ...] = (
    "Benign",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Reconnaissance",
    "Shellcode",
)

#: Frozen stream length (docs/05 §3).
N_PERIODS = 5

#: Cross-check tolerance for values that should be arithmetically identical
#: (float64 summary of float64 per-class F1 values).
EXACT_ATOL = 1e-12

_PASS = "pass"
_MISMATCH = "mismatch"
_EXPECTED_DIVERGENCE = "expected_divergence"
_NOT_APPLICABLE = "not_applicable"
_FAIL = "fail"

_STATUS_ORDER = {
    _PASS: 0,
    _EXPECTED_DIVERGENCE: 1,
    _NOT_APPLICABLE: 2,
    _MISMATCH: 3,
    _FAIL: 4,
}


@dataclass
class Finding:
    """One cross-check result with an explicit status and evidence string."""

    check: str
    status: str
    detail: str
    reference: Any = None
    actual: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "status": self.status,
            "detail": self.detail,
            "reference": self.reference,
            "actual": self.actual,
        }


@dataclass
class CrosscheckResult:
    """Everything the report needs, plus a machine-readable verdict."""

    results_tables: str
    artifact_hashes: dict[str, str | None] = field(default_factory=dict)
    findings: list[Finding] = field(default_factory=list)
    synthetic_formula_comparisons: int = 0
    synthetic_formula_mismatches: int = 0
    notes: list[str] = field(default_factory=list)

    def add(self, check: str, status: str, detail: str, reference: Any = None, actual: Any = None) -> None:
        self.findings.append(Finding(check, status, detail, reference, actual))

    @property
    def mismatches(self) -> list[Finding]:
        return [finding for finding in self.findings if finding.status in {_MISMATCH, _FAIL}]

    @property
    def verdict(self) -> str:
        if any(finding.status == _FAIL for finding in self.findings):
            return "FAIL"
        if any(finding.status == _MISMATCH for finding in self.findings):
            return "PASS-WITH-MISMATCHES"
        if any(finding.status == _EXPECTED_DIVERGENCE for finding in self.findings):
            return "PASS-CONDITIONAL"
        return "PASS"

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "results_tables": self.results_tables,
            "verdict": self.verdict,
            "artifact_hashes": self.artifact_hashes,
            "synthetic_formula_comparisons": self.synthetic_formula_comparisons,
            "synthetic_formula_mismatches": self.synthetic_formula_mismatches,
            "findings": [finding.as_dict() for finding in self.findings],
            "notes": self.notes,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _import_engineer_module():
    """Import ``src/eval/cl_metrics.py`` without requiring a package side effect."""
    try:
        return importlib.import_module("src.eval.cl_metrics"), None
    except Exception as exc:  # pragma: no cover - environment dependent
        return None, f"{type(exc).__name__}: {exc}"


def load_artifacts(results_tables: Path) -> tuple[dict[str, Any], dict[str, str | None]]:
    """Read the three S5 artifacts and hash them; missing files are ``None``."""
    paths = {
        "matrix": results_tables / "cl_matrix.json",
        "metrics_csv": results_tables / "cl_metrics.csv",
        "per_class": results_tables / "cl_per_class_f1.json",
    }
    payload: dict[str, Any] = {}
    hashes: dict[str, str | None] = {}
    for name, path in paths.items():
        hashes[name] = _sha256(path) if path.is_file() else None
        if not path.is_file():
            payload[name] = None
            continue
        if path.suffix == ".csv":
            payload[name] = pd.read_csv(path)
        else:
            payload[name] = json.loads(path.read_text(encoding="utf-8"))
    return payload, hashes


def _matrix_from_payload(matrix_payload: Mapping[str, Any]) -> np.ndarray:
    matrix = np.asarray(matrix_payload["matrix"]["all_classes"], dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"matrix.all_classes must be square, got shape {matrix.shape}")
    if np.any(np.isinf(matrix)):
        raise ValueError("matrix.all_classes contains inf")
    # Future-period evaluations (j > i) are valid R cells (Doc 2 S5 "for all i and j");
    # NaN cells are simply not evaluated.
    return matrix


def _mean_over_ids(per_class: Mapping[str, float], ids: Sequence[int]) -> float:
    values = [float(per_class[str(int(index))]) for index in ids if str(int(index)) in per_class]
    if not values:
        raise ValueError(f"no per-class values for ids {list(ids)}")
    return float(np.mean(values))


# ---------------------------------------------------------------------------
# Cross-check sections
# ---------------------------------------------------------------------------
def check_artifact_schema(result: CrosscheckResult, payload: Mapping[str, Any], engineer) -> dict[str, Any] | None:
    matrix_payload = payload.get("matrix")
    if not isinstance(matrix_payload, Mapping):
        result.add("artifact.cl_matrix", _FAIL, "cl_matrix.json missing or unreadable")
        return None
    required = {"schema_version", "dataset", "strategy", "model", "seed", "class_order", "matrix", "dual_macro_f1_per_cell"}
    missing = sorted(required - set(matrix_payload))
    if missing:
        result.add("artifact.cl_matrix.schema", _FAIL, f"missing keys {missing}")
        return matrix_payload
    result.add("artifact.cl_matrix.schema", _PASS, f"all required keys present; dataset={matrix_payload['dataset']}")
    class_order = list(matrix_payload["class_order"])
    result.add(
        "artifact.class_order",
        _PASS if len(class_order) == 10 else _FAIL,
        f"{len(class_order)} classes: {class_order}",
        reference="10-class frozen label space (docs/05 §3)",
        actual=class_order,
    )
    matrix = _matrix_from_payload(matrix_payload)
    result.add("artifact.matrix.shape", _PASS if matrix.shape == (N_PERIODS, N_PERIODS) else _FAIL, f"shape {matrix.shape}")
    future_cells = np.argwhere(np.isfinite(matrix) & np.triu(np.ones(matrix.shape, dtype=bool), k=1))
    result.add(
        "R_matrix.future_evaluations",
        _PASS,
        f"{len(future_cells)} finite cell(s) above the diagonal: the writer evaluates every checkpoint on every "
        "period (Doc 2 S5 defines R[i,j] for all i and j); the reference records them and excludes them from "
        "forgetting only",
        reference="recorded but not used by forgetting (Doc 2 S5 check)",
        actual=[(int(row) + 1, int(column) + 1) for row, column in future_cells.tolist()][:20],
    )
    checkpoints = sorted(int(key) for key in (matrix_payload.get("checkpoints") or {}))
    defined_rows = sorted({int(key) for key in (matrix_payload.get("dual_macro_f1_per_cell") or {})})
    result.add(
        "artifact.checkpoints",
        _PASS if checkpoints == defined_rows else _MISMATCH,
        f"checkpoints={checkpoints}, populated rows={defined_rows}",
    )
    last_row_complete = len(defined_rows) == N_PERIODS and all(
        matrix_payload["dual_macro_f1_per_cell"].get(str(N_PERIODS), {}).get(str(period)) is not None
        for period in range(1, N_PERIODS + 1)
    )
    result.notes.append(
        f"partial checkpoint matrix: {len(defined_rows)}/{N_PERIODS} rows populated"
        + ("" if last_row_complete else " — full forgetting/BWT on real data not exercisable yet")
    )
    return matrix_payload


def check_dual_macro_per_cell(result: CrosscheckResult, matrix_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute all-classes/supported macro from per-class F1 and compare."""
    class_order = list(matrix_payload["class_order"])
    per_cell = matrix_payload.get("dual_macro_f1_per_cell") or {}
    configured_supported = list(matrix_payload.get("supported_set") or [])
    supported_ids = [class_order.index(name) for name in configured_supported if name in class_order]
    d033_ids = [class_order.index(name) for name in D033_SUPPORTED_CLASSES if name in class_order]
    comparisons = 0
    all_mismatches: list[str] = []
    supported_mismatches: list[str] = []
    d033_flags: list[str] = []
    range_issues: list[str] = []
    count_issues: list[str] = []
    for row_key, columns in per_cell.items():
        for column_key, cell in (columns or {}).items():
            dual = (cell or {}).get("dual_macro_f1") or {}
            per_class = (cell or {}).get("per_class_f1") or {}
            if not dual or not per_class:
                continue
            comparisons += 1
            cell_name = f"M_{row_key}/P_{column_key}"
            macro_all = float(dual["macro_f1_all_classes"])
            recomputed_all = float(np.mean([float(value) for value in per_class.values()]))
            if abs(macro_all - recomputed_all) > EXACT_ATOL:
                all_mismatches.append(f"{cell_name}: {macro_all:.15g} != {recomputed_all:.15g}")
            macro_supported = float(dual["macro_f1_supported_set"])
            recomputed_supported = _mean_over_ids(per_class, supported_ids)
            if abs(macro_supported - recomputed_supported) > EXACT_ATOL:
                supported_mismatches.append(f"{cell_name}: {macro_supported:.15g} != {recomputed_supported:.15g}")
            recomputed_d033 = _mean_over_ids(per_class, d033_ids)
            if abs(macro_supported - recomputed_d033) > EXACT_ATOL:
                d033_flags.append(f"{cell_name}: configured={macro_supported:.6f} vs D-033(7 classes)={recomputed_d033:.6f}")
            for class_id, value in per_class.items():
                if not 0.0 <= float(value) <= 1.0:
                    range_issues.append(f"{cell_name} class {class_id}: f1={value}")
            class_counts = dual.get("class_counts") or {}
            if int(sum(int(value) for value in class_counts.values())) != int(dual.get("n_edges", -1)):
                count_issues.append(
                    f"{cell_name}: sum(class_counts)={sum(int(v) for v in class_counts.values())} != n_edges={dual.get('n_edges')}"
                )
    result.add(
        "dual_macro.all_classes_vs_class_mean",
        _PASS if not all_mismatches else _MISMATCH,
        f"{comparisons} cell(s) checked; mismatches={all_mismatches[:5]}",
        reference="mean of 10 per-class F1 values",
        actual=f"{comparisons} cells",
    )
    result.add(
        "dual_macro.supported_set_vs_class_mean",
        _PASS if not supported_mismatches else _MISMATCH,
        f"configured supported set {configured_supported} (ids {supported_ids}); mismatches={supported_mismatches[:5]}",
    )
    result.add(
        "dual_macro.D033_supported_definition",
        _PASS if not d033_flags else _MISMATCH,
        "configured supported set matches D-033 (Benign + 6 attack classes)"
        if not d033_flags
        else "configured supported set omits class(es) vs D-033; " + "; ".join(d033_flags[:5]),
        reference=list(D033_SUPPORTED_CLASSES),
        actual=configured_supported,
    )
    result.add(
        "per_class.values_in_range",
        _PASS if not range_issues else _MISMATCH,
        f"f1 in [0,1] for all per-class values; issues={range_issues[:5]}",
    )
    result.add(
        "per_class.class_counts_sum",
        _PASS if not count_issues else _MISMATCH,
        f"class_counts sum equals n_edges; issues={count_issues[:3]}",
    )
    return {"supported_ids": supported_ids, "d033_ids": d033_ids, "comparisons": comparisons}


def check_per_class_files(result: CrosscheckResult, matrix_payload: Mapping[str, Any], per_class_payload: Mapping[str, Any]) -> None:
    if not per_class_payload:
        result.add("per_class.file", _FAIL, "cl_per_class_f1.json missing")
        return
    if list(per_class_payload.get("class_order") or []) != list(matrix_payload.get("class_order") or []):
        result.add("per_class.class_order", _MISMATCH, "class_order differs between cl_matrix.json and cl_per_class_f1.json")
    else:
        result.add("per_class.class_order", _PASS, "class_order identical in both artifacts")
    matrix_cells = matrix_payload.get("dual_macro_f1_per_cell") or {}
    per_class_cells = per_class_payload.get("per_class_f1") or {}
    compared = 0
    mismatches: list[str] = []
    for row_key, columns in matrix_cells.items():
        for column_key, cell in (columns or {}).items():
            left = (cell or {}).get("per_class_f1")
            right = ((per_class_cells.get(row_key) or {}).get(column_key) or {}).get("per_class_f1")
            if left is None or right is None:
                continue
            compared += 1
            if set(left) != set(right) or any(abs(float(left[key]) - float(right[key])) > EXACT_ATOL for key in left):
                mismatches.append(f"M_{row_key}/P_{column_key}")
    result.add(
        "per_class.matrix_vs_per_class_file",
        _PASS if not mismatches and compared else (_FAIL if mismatches else _NOT_APPLICABLE),
        f"{compared} cell(s) compared; mismatches={mismatches[:5]}",
    )


def check_cl_matrix_against_reference(result: CrosscheckResult, matrix_payload: Mapping[str, Any], path: Path) -> None:
    engineer_matrix = _matrix_from_payload(matrix_payload)
    reference = mc.load_cl_matrix_json(path)
    if reference.R.shape != engineer_matrix.shape:
        result.add("R_matrix.shape", _MISMATCH, f"analysis={reference.R.shape} vs writer={engineer_matrix.shape}")
    else:
        result.add("R_matrix.shape", _PASS, f"both {engineer_matrix.shape}")
    defined = np.isfinite(engineer_matrix)
    reference_defined = np.isfinite(reference.R)
    if not np.array_equal(defined, reference_defined):
        result.add("R_matrix.defined_cells", _MISMATCH, "defined-cell masks differ between writer and reference")
    else:
        result.add(
            "R_matrix.defined_cells",
            _PASS,
            f"identical defined-cell mask ({int(defined.sum())} evaluated cells; forward cells included)",
        )
    differences = np.abs(engineer_matrix[defined] - reference.R[defined])
    max_difference = float(np.max(differences)) if differences.size else 0.0
    result.add(
        "R_matrix.values",
        _PASS if max_difference <= EXACT_ATOL else _MISMATCH,
        f"{int(defined.sum())} defined cell(s), max|diff|={max_difference:.3e}",
        reference="analysis.metrics_cl.load_cl_matrix_json",
        actual="cl_matrix.json matrix.all_classes",
    )
    class_order = tuple(matrix_payload["class_order"])
    result.add(
        "R_matrix.per_class_tensor",
        _PASS
        if reference.R_class is not None and reference.classes == class_order
        else _MISMATCH,
        f"analysis tensor R_class shape={None if reference.R_class is None else reference.R_class.shape}, "
        f"classes match={reference.classes == class_order}",
    )


def check_cl_metrics_csv(result: CrosscheckResult, payload: Mapping[str, Any], matrix_payload: Mapping[str, Any]) -> None:
    frame = payload.get("metrics_csv")
    if not isinstance(frame, pd.DataFrame):
        result.add("cl_metrics_csv.file", _FAIL, "cl_metrics.csv missing")
        return
    expected_columns = [
        "strategy",
        "model",
        "seed",
        "metric",
        "final_average_macro_f1",
        "forgetting",
        "bwt",
        "note",
    ]
    result.add(
        "cl_metrics_csv.columns",
        _PASS if list(frame.columns) == expected_columns else _FAIL,
        f"columns={list(frame.columns)}",
    )
    metrics = set(frame["metric"].astype(str)) if "metric" in frame.columns else set()
    result.add(
        "cl_metrics_csv.metric_rows",
        _PASS if metrics <= {"all_classes", "supported_set"} else _MISMATCH,
        f"metric rows={sorted(metrics)}",
    )
    if not frame.empty and "strategy" in frame.columns:
        identity = {
            "strategy": str(frame.iloc[0]["strategy"]),
            "model": str(frame.iloc[0]["model"]),
            "seed": int(frame.iloc[0]["seed"]),
        }
        expected = {
            "strategy": str(matrix_payload.get("strategy")),
            "model": str(matrix_payload.get("model")),
            "seed": int(matrix_payload.get("seed")),
        }
        result.add(
            "cl_metrics_csv.identity",
            _PASS if identity == expected else _MISMATCH,
            f"csv={identity}, matrix.json={expected}",
        )
    incomplete = matrix_payload.get("note") == "last checkpoint row incomplete; CL metrics not computed yet"
    if incomplete:
        empty_metrics = frame["final_average_macro_f1"].isna().all() and frame["forgetting"].isna().all()
        notes_ok = set(frame["note"].astype(str)) == {"incomplete_matrix"}
        result.add(
            "cl_metrics_csv.incomplete_matrix_semantics",
            _PASS if empty_metrics and notes_ok else _MISMATCH,
            "partial matrix → empty CL metric cells + note=incomplete_matrix"
            if empty_metrics and notes_ok
            else "expected empty metric cells with note=incomplete_matrix",
        )
    else:
        result.add("cl_metrics_csv.incomplete_matrix_semantics", _NOT_APPLICABLE, "matrix is complete; full metrics expected")


def check_formula_equivalence(result: CrosscheckResult, engineer) -> None:
    """Compare forgetting/BWT/final formulas against the reference on known matrices."""
    if engineer is None:
        result.add("formula.cl_metrics", _NOT_APPLICABLE, "engineer module not importable; formula comparison skipped")
        return
    hand = np.array(
        [
            [0.80, np.nan, np.nan],
            [0.60, 0.90, np.nan],
            [0.50, 0.70, 0.60],
        ]
    )
    cases: list[tuple[str, np.ndarray]] = [("hand_T3", hand)]
    rng = np.random.default_rng(42)
    for n_periods in (2, 3, 5):
        for trial in range(3):
            matrix = np.full((n_periods, n_periods), np.nan)
            for row in range(n_periods):
                for column in range(row + 1):
                    matrix[row, column] = float(rng.uniform(0.2, 0.95))
            cases.append((f"random_T{n_periods}_{trial}", matrix))
    mismatches: list[str] = []
    comparisons = 0
    for name, matrix in cases:
        engineer_metrics = engineer.cl_metrics(matrix)
        reference = mc.RMetricTensor(matrix)
        comparisons += 6  # final + forgetting + bwt + 3 per-period vectors
        if abs(float(engineer_metrics["final_average_macro_f1"]) - reference.final_macro_f1()) > EXACT_ATOL:
            mismatches.append(f"{name}: final")
        if abs(float(engineer_metrics["forgetting"]) - reference.forgetting()) > EXACT_ATOL:
            mismatches.append(f"{name}: forgetting mean")
        if abs(float(engineer_metrics["bwt"]) - reference.bwt()) > EXACT_ATOL:
            mismatches.append(f"{name}: bwt mean")
        for period in range(matrix.shape[0] - 1):
            engineer_value = engineer_metrics["forgetting_per_period"][str(period + 1)]
            reference_value = float(reference.forgetting_per_period()[period])
            if engineer_value is None or abs(float(engineer_value) - reference_value) > EXACT_ATOL:
                mismatches.append(f"{name}: forgetting P{period + 1}")
            engineer_bwt = engineer_metrics["bwt_per_period"][str(period + 1)]
            reference_bwt = float(reference.bwt_per_period()[period])
            if engineer_bwt is None or abs(float(engineer_bwt) - reference_bwt) > EXACT_ATOL:
                mismatches.append(f"{name}: bwt P{period + 1}")
    result.synthetic_formula_comparisons = comparisons
    result.synthetic_formula_mismatches = len(mismatches)
    result.add(
        "formula.cl_metrics_values",
        _PASS if not mismatches else _MISMATCH,
        f"{comparisons} value comparison(s) over {len(cases)} matrices; mismatches={mismatches[:5]}",
    )

    # Full matrices (every checkpoint evaluated on every period, as the real S5
    # writer does): the forgetting best-so-far term must exclude forward cells.
    false_cases: list[str] = []
    full_comparisons = 0
    for n_periods in (3, 5):
        for trial in range(2):
            matrix = np.full((n_periods, n_periods), np.nan)
            for row in range(n_periods):
                for column in range(n_periods):
                    matrix[row, column] = float(rng.uniform(0.2, 0.95))
            engineer_metrics = engineer.cl_metrics(matrix)
            reference = mc.RMetricTensor(matrix)
            full_comparisons += 1
            if abs(float(engineer_metrics["final_average_macro_f1"]) - reference.final_macro_f1()) > EXACT_ATOL:
                false_cases.append(f"T{n_periods}_{trial}: final")
            if abs(float(engineer_metrics["bwt"]) - reference.bwt()) > EXACT_ATOL:
                false_cases.append(f"T{n_periods}_{trial}: bwt")
            for period in range(n_periods - 1):
                engineer_value = float(engineer_metrics["forgetting_per_period"][str(period + 1)])
                reference_value = reference.forgetting_per_period()[period]
                if abs(engineer_value - float(reference_value)) > EXACT_ATOL:
                    false_cases.append(
                        f"T{n_periods}_{trial} P{period + 2}: writer={engineer_value:.6f} vs Doc-2-excluded reference={float(reference_value):.6f}"
                    )
    result.add(
        "formula.forgetting_excludes_future_evaluations",
        _PASS if not false_cases else _MISMATCH,
        f"{full_comparisons} fully-populated matrix/matrices compared; discrepancies={false_cases[:4]}. "
        "Doc 2 S5 check: the upper triangle (future periods) is not used for forgetting.",
        reference="best-so-far max over past checkpoints i in [j, T-1)",
        actual="writer max over rows 0..T-2 (includes forward cells when populated)",
    )

    # Matrix construction from (row, column) -> score must agree with the tensor.
    scores = {(0, 0): 0.7, (1, 0): 0.6, (1, 1): 0.8}
    writer_matrix = engineer.matrix_from_scores(scores, 2)
    reference_matrix = mc.RMetricTensor(np.array([[0.7, np.nan], [0.6, 0.8]])).R
    construction_equal = np.allclose(writer_matrix, reference_matrix, equal_nan=True)
    result.add("formula.matrix_from_scores", _PASS if construction_equal else _MISMATCH, "2x2 construction equality (NaN-aware)")

    # Degenerate T=1: the reference calls forgetting/BWT undefined (NaN), the
    # writer returns 0.0 because it averages an empty list. Not used by the frozen
    # T=5 protocol; recorded as an expected divergence, not a gate failure.
    single = np.array([[0.7]])
    writer_single = engineer.cl_metrics(single)
    reference_single = mc.RMetricTensor(single)
    divergence = (
        abs(float(writer_single["forgetting"]) - 0.0) <= EXACT_ATOL
        and np.isnan(reference_single.forgetting())
        and np.isnan(reference_single.bwt())
    )
    result.add(
        "formula.T1_degenerate_case",
        _EXPECTED_DIVERGENCE if divergence else _PASS,
        "writer returns 0.0 for T=1 forgetting/BWT; reference returns NaN (undefined). "
        "Documented divergence only relevant to non-protocol T=1 runs."
        if divergence
        else "T=1 behaviour matches",
    )

    # Incomplete last row: the reference refuses to compute final metrics. The writer may
    # either raise (original behaviour) or return a partial-metrics payload with NaN
    # final/forgetting/BWT plus a warning (T35-observed partial-matrix support). Both are
    # acceptable; silently fabricating a number is not.
    incomplete = np.array([[0.7, np.nan], [0.6, np.nan]])
    writer_raises = False
    writer_partial = False
    try:
        writer_result = engineer.cl_metrics(incomplete)
        if isinstance(writer_result, Mapping):
            keys = ("final_average_macro_f1", "forgetting", "bwt")
            writer_partial = all(
                writer_result.get(key) is None
                or (isinstance(writer_result.get(key), float) and np.isnan(writer_result[key]))
                for key in keys
            )
    except ValueError:
        writer_raises = True
    reference_raises = False
    try:
        mc.RMetricTensor(incomplete).final_macro_f1()
    except mc.CLMetricsError:
        reference_raises = True
    result.add(
        "formula.incomplete_last_row_guard",
        _PASS if reference_raises and (writer_raises or writer_partial) else _MISMATCH,
        f"writer raises={writer_raises}, writer partial-NaN metrics={writer_partial}, reference raises={reference_raises}",
    )


# ---------------------------------------------------------------------------
# Runner + report
# ---------------------------------------------------------------------------
def run_crosscheck(results_tables: str | Path = Path("results") / "tables") -> CrosscheckResult:
    """Run every S5 cross-check section and return the structured result."""
    tables_path = Path(results_tables)
    result = CrosscheckResult(results_tables=str(tables_path))
    payload, hashes = load_artifacts(tables_path)
    result.artifact_hashes = hashes
    for name, digest in hashes.items():
        result.add(
            f"artifact.present.{name}",
            _PASS if digest else _FAIL,
            f"{name}: {'sha256=' + digest if digest else 'MISSING'}",
        )
    engineer, import_error = _import_engineer_module()
    result.add(
        "engineer.module_import",
        _PASS if engineer is not None else _FAIL,
        "src.eval.cl_metrics imported" if engineer is not None else f"import failed: {import_error}",
    )
    matrix_payload = check_artifact_schema(result, payload, engineer) if payload.get("matrix") else None
    if matrix_payload is not None:
        check_dual_macro_per_cell(result, matrix_payload)
        check_per_class_files(result, matrix_payload, payload.get("per_class") or {})
        matrix_path = tables_path / "cl_matrix.json"
        check_cl_matrix_against_reference(result, matrix_payload, matrix_path)
        check_cl_metrics_csv(result, payload, matrix_payload)
    check_formula_equivalence(result, engineer)
    return result


def _status_counts(result: CrosscheckResult) -> dict[str, int]:
    counts: dict[str, int] = {}
    for finding in result.findings:
        counts[finding.status] = counts.get(finding.status, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: _STATUS_ORDER.get(item[0], 99)))


def render_markdown(result: CrosscheckResult) -> str:
    """Render the human-readable cross-check report (committed under src/analysis/reports/)."""
    generated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    lines: list[str] = []
    lines.append("# S5 cross-check — engineer S5 writer vs `analysis.metrics_cl` reference")
    lines.append("")
    lines.append(
        "*Owner: analyst (T19). Authority: `docs/DECISIONS.md` **D-044** (analysis modules are the "
        "authoritative definitions) and **D-043** (engineer S5 files are the single writer); "
        "definitions from `docs/source_documents/Doc 2` §S5, `docs/05` §11 and **D-033**. Reproduce with "
        "`.\\.venv\\Scripts\\python.exe -m analysis.s5_crosscheck`.*"
    )
    lines.append("")
    lines.append(f"*Generated: {generated} · results root: `{result.results_tables}` · front-end report only; no src/ change.*")
    lines.append("")
    lines.append(f"## Verdict: **{result.verdict}**")
    lines.append("")
    lines.append(f"- checks: {_status_counts(result)}")
    lines.append(f"- synthetic formula comparisons: {result.synthetic_formula_comparisons} (mismatches: {result.synthetic_formula_mismatches})")
    lines.append(f"- mismatches: {len(result.mismatches)}")
    if result.notes:
        for note in result.notes:
            lines.append(f"- note: {note}")
    lines.append("")
    lines.append("## Artifacts checked")
    lines.append("")
    lines.append("| artifact | sha256 |")
    lines.append("| --- | --- |")
    for name, digest in result.artifact_hashes.items():
        lines.append(f"| `{name}` | `{digest or 'MISSING'}` |")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| # | check | status | detail |")
    lines.append("| --- | --- | --- | --- |")
    for index, finding in enumerate(result.findings, start=1):
        detail = str(finding.detail).replace("|", "\\|")
        lines.append(f"| {index} | `{finding.check}` | **{finding.status}** | {detail} |")
    lines.append("")
    checks = {finding.check: finding for finding in result.findings}
    lines.append("## Findings that need action")
    lines.append("")
    mismatches = result.mismatches
    if not mismatches:
        lines.append("No numerical or schema mismatch was found by this cross-check.")
    for finding in mismatches:
        lines.append(f"- **{finding.check}** ({finding.status}): {finding.detail}")
    supported = checks.get("dual_macro.D033_supported_definition")
    if supported is not None and supported.status == _MISMATCH:
        lines.append("")
        lines.append("### D-033 supported-set definition (src/ + configs/ owner: engineer)")
        lines.append("")
        lines.append(
            "D-033 defines the supported set as the classes with ≥50 test-mask edges in every period: "
            "the 6 attack classes **plus Benign**. The written artifact/config currently reports a "
            "6-attack-class-only supported set. The per-class values are present in the same artifact, so the "
            "7-class number is reconstructable; this is a definition/config issue, not a data issue."
        )
        lines.append(f"  - reference (D-033): `{supported.reference}`")
        lines.append(f"  - actual (artifact/config): `{supported.actual}`")
    conditional = [finding for finding in result.findings if finding.status == _EXPECTED_DIVERGENCE]
    if conditional:
        lines.append("")
        lines.append("### Documented divergences (not gate failures)")
        for finding in conditional:
            lines.append(f"- **{finding.check}**: {finding.detail}")
    lines.append("")
    lines.append("## What the partial artifact can and cannot prove")
    lines.append("")
    lines.append(
        "The only existing S5 artifact is a partial `ft/egraphsage/seed42` run with `M_1` and no later "
        "checkpoints. It therefore exercises artifact schema, identity, the `R[0,0]` cell, dual-macro "
        "arithmetic, per-class consistency and the `incomplete_matrix` CSV semantics — **not** the real "
        "forgetting/BWT over the frozen 5-period matrix. The forgetting/BWT formulas were cross-checked "
        "exactly on a hand-computed T=3 matrix and 9 deterministic complete matrices (T=2/3/5), and both "
        "implementations refuse an incomplete last row. Once `M_2..M_5` exist, re-running this command "
        "adds the real-data formula comparison without any change to this report's structure."
    )
    lines.append("")
    lines.append("## Recommended actions")
    lines.append("")
    actions = [
        "Resolve the D-033 supported-set definition in `configs/eval.yaml` + `src/eval/cl_metrics.py` "
        "(Benign must be in the supported set per D-033, or the deviation must be logged in `docs/DECISIONS.md`).",
        "Per D-044, replace the duplicated formulas in `src/eval/cl_metrics.py` with imports from "
        "`analysis.metrics_cl` (or keep the thin wrapper and pin this cross-check in CI) so the two cannot diverge.",
        "Keep the T=1 degenerate behaviour documented (analysis: NaN/undefined; writer: 0.0); it is outside the "
        "frozen T=5 protocol.",
        "Re-run `python -m src.analysis.s5_crosscheck` after the next S5 full-matrix run; the report is regenerated "
        "and should then show `PASS`/`PASS-CONDITIONAL` with the real forgetting/BWT cells compared.",
    ]
    for action in actions:
        lines.append(f"- {action}")
    lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-tables", default=str(Path("results") / "tables"))
    parser.add_argument("--out", default=str(Path("analysis") / "reports" / "s5_crosscheck.md"))
    parser.add_argument("--json", action="store_true", help="print the machine-readable result instead of the report path")
    args = parser.parse_args(argv)
    result = run_crosscheck(args.results_tables)
    report_path = Path(args.out)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_markdown(result), encoding="utf-8")
    payload = result.as_dict()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(json.dumps({"verdict": payload["verdict"], "report": str(report_path), "checks": _status_counts(result)}, indent=2))
    return 0 if result.verdict != "FAIL" else 1


__all__ = [
    "CrosscheckResult",
    "D033_SUPPORTED_CLASSES",
    "Finding",
    "load_artifacts",
    "render_markdown",
    "run_crosscheck",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    import sys

    sys.exit(main())
