"""T19 S5 cross-check — reference-vs-writer equality and mismatch detection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.analysis import artifacts, metrics_cl as mc, s5_crosscheck, synthetic, tables

REPO_ROOT = Path(__file__).resolve().parents[3]


def _tensor() -> mc.RMetricTensor:
    return synthetic.cl_tensors(models=("egraphsage",), strategies=("ft",))[("ft", "egraphsage", synthetic.SEED)]


def _write_s5_artifacts(root: Path, *, supported: tuple[str, ...] | None = None) -> Path:
    layout = artifacts.ArtifactLayout(root / "results")
    tables.write_s5_writer_artifacts(
        _tensor(),
        layout,
        supported_classes=s5_crosscheck.D033_SUPPORTED_CLASSES if supported is None else supported,
    )
    return layout.tables_root


def test_reference_and_writer_artifacts_compare_on_synthetic(tmp_path: Path) -> None:
    tables_root = _write_s5_artifacts(tmp_path)
    result = s5_crosscheck.run_crosscheck(tables_root)
    checks = {finding.check: finding for finding in result.findings}
    assert checks["R_matrix.values"].status == "pass"
    assert checks["R_matrix.defined_cells"].status == "pass"
    assert checks["dual_macro.all_classes_vs_class_mean"].status == "pass"
    assert checks["dual_macro.supported_set_vs_class_mean"].status == "pass"
    # The synthetic artifact is written with the D-033 7-class supported set.
    assert checks["dual_macro.D033_supported_definition"].status == "pass"
    assert checks["per_class.matrix_vs_per_class_file"].status == "pass"
    assert checks["formula.cl_metrics_values"].status == "pass"
    assert result.verdict in {"PASS", "PASS-CONDITIONAL", "PASS-WITH-MISMATCHES"}


def test_crosscheck_detects_supported_set_definition_mismatch(tmp_path: Path) -> None:
    tables_root = _write_s5_artifacts(tmp_path, supported=("DoS", "Exploits", "Fuzzers", "Generic", "Reconnaissance", "Shellcode"))
    result = s5_crosscheck.run_crosscheck(tables_root)
    checks = {finding.check: finding for finding in result.findings}
    assert checks["dual_macro.D033_supported_definition"].status == "mismatch"
    assert len(result.mismatches) >= 1


class _UpperTriangleEngineer:
    """Bad implementation that wrongly includes forward cells in forgetting (detection test)."""

    @staticmethod
    def cl_metrics(matrix: np.ndarray) -> dict:
        matrix = np.asarray(matrix, dtype=float)
        n = matrix.shape[0]
        last = matrix[n - 1]
        if np.isnan(last).any():
            raise ValueError("last row incomplete")
        final = float(np.mean(last))
        forgetting = {}
        bwt = {}
        for period in range(n):
            if period == n - 1:
                forgetting[str(period + 1)] = None
                bwt[str(period + 1)] = None
                continue
            column = matrix[: n - 1, period]  # includes forward cells -> wrong per Doc 2 S5
            forgetting[str(period + 1)] = float(np.nanmax(column) - last[period])
            bwt[str(period + 1)] = float(last[period] - matrix[period, period])
        forgetting_values = [value for value in forgetting.values() if value is not None]
        bwt_values = [value for value in bwt.values() if value is not None]
        return {
            "final_average_macro_f1": final,
            "forgetting": float(np.mean(forgetting_values)) if forgetting_values else 0.0,
            "forgetting_per_period": forgetting,
            "bwt": float(np.mean(bwt_values)) if bwt_values else 0.0,
            "bwt_per_period": bwt,
        }

    @staticmethod
    def matrix_from_scores(scores, n_periods: int) -> np.ndarray:
        matrix = np.full((n_periods, n_periods), np.nan)
        for (row, column), value in scores.items():
            matrix[row, column] = float(value)
        return matrix


def test_crosscheck_detects_forgetting_upper_triangle_use() -> None:
    result = s5_crosscheck.CrosscheckResult(results_tables="synthetic")
    s5_crosscheck.check_formula_equivalence(result, _UpperTriangleEngineer)
    checks = {finding.check: finding for finding in result.findings}
    assert checks["formula.forgetting_excludes_future_evaluations"].status == "mismatch"
    assert checks["formula.cl_metrics_values"].status == "pass"  # triangular matrices agree
    assert checks["formula.incomplete_last_row_guard"].status == "pass"


def test_real_partial_artifact_crosscheck_does_not_fail() -> None:
    real_tables = REPO_ROOT / "results" / "tables"
    if not (real_tables / "cl_matrix.json").is_file():
        pytest.skip("partial S5 artifacts not present in this checkout")
    result = s5_crosscheck.run_crosscheck(real_tables)
    assert result.verdict in {"PASS", "PASS-CONDITIONAL", "PASS-WITH-MISMATCHES"}
    checks = {finding.check: finding for finding in result.findings}
    assert checks["engineer.module_import"].status == "pass"
    assert checks["R_matrix.values"].status == "pass"
    assert checks["cl_metrics_csv.incomplete_matrix_semantics"].status == "pass"
    # The known partial run has M_1 only; full forgetting/BWT are deferred by design.
    assert "partial checkpoint matrix" in " ".join(result.notes)


def test_render_markdown_contains_verdict_and_hashes(tmp_path: Path) -> None:
    tables_root = _write_s5_artifacts(tmp_path)
    result = s5_crosscheck.run_crosscheck(tables_root)
    markdown = s5_crosscheck.render_markdown(result)
    assert "# S5 cross-check" in markdown
    assert result.verdict in markdown
    assert "sha256" in markdown
    assert "Recommended actions" in markdown
    payload = result.as_dict()
    json.dumps(payload)  # serialisable for the report/manifest
