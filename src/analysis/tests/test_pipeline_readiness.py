"""T20 pipeline-readiness report tests (real repo state + markdown contract)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.analysis import pipeline_readiness

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_TABLES = REPO_ROOT / "results" / "tables"


def test_readiness_report_covers_full_chain_and_renders(tmp_path: Path) -> None:
    if not RESULTS_TABLES.is_dir():
        pytest.skip("no results/text/tables in this checkout")
    readiness = pipeline_readiness.build_readiness(RESULTS_TABLES)
    assert len(readiness.rows) >= 20
    names = {row.row.name for row in readiness.rows}
    for required in ("cl_matrix.json", "anchor sets", "explanation cache index", "noise floor table", "rq1 H1 tests", "rq4 comparison"):
        assert required in names
    statuses = {row.status for row in readiness.rows}
    assert statuses <= {"ready", "present-warn", "present-FAIL", "waiting"}
    assert set(readiness.counts) == {"ready", "present-warn", "present-FAIL", "waiting", "not-applicable"}
    # blocking inputs mirror the engineer wiring guard
    assert any(item["input"].startswith("results/text/explanations_noise") for item in readiness.blocking)
    # config reconciliation checks include the D-043/D-044/D-047 rows
    check_names = {check["check"] for check in readiness.config_checks}
    assert any("D-047" in name for name in check_names)
    assert any("D-044" in name for name in check_names)
    assert any("primary metric" in name for name in check_names)
    markdown = pipeline_readiness.render_markdown(readiness)
    assert "# Pipeline readiness" in markdown
    assert "Artifact chain" in markdown
    assert "Blocking inputs" in markdown
    assert "Config / protocol reconciliation" in markdown
    path = pipeline_readiness.write_report(readiness, tmp_path / "readiness.md")
    assert path.is_file() and "status" in path.read_text(encoding="utf-8")


def test_readiness_marks_s5_ready_when_artifacts_present() -> None:
    if not (RESULTS_TABLES / "cl_matrix.json").is_file():
        pytest.skip("S5 artifacts not present")
    readiness = pipeline_readiness.build_readiness(RESULTS_TABLES)
    s5_rows = [row for row in readiness.rows if row.row.stage == "S5"]
    assert s5_rows and all(row.status in {"ready", "present-warn"} for row in s5_rows)
    index_rows = [row for row in readiness.rows if row.row.name == "explanation cache index"]
    assert index_rows and index_rows[0].status == "waiting"  # current checkout: index header only


def test_first_real_run_checklist_is_complete() -> None:
    checklist = REPO_ROOT / "analysis" / "reports" / "first_real_run_checklist.md"
    assert checklist.is_file(), "first real run checklist missing"
    text = checklist.read_text(encoding="utf-8")
    for stage in ("s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12"):
        assert f"--stage {stage} --strict" in text, stage
    for expected in ("27 pass", "48 pass", "76 pass", "104 pass", "159"):
        assert expected in text, expected
    assert "s5_crosscheck" in text and "s8_validation" in text
    assert "stop on any fail" in text.lower()
