"""T20 independent S8/S9 validation tests (H1 arithmetic cross-check on synthetic demo)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd
import pytest

from src.analysis import artifacts, pipeline_fixtures, s8_validation, synthetic, tables


@pytest.fixture(scope="module")
def demo_tables(tmp_path_factory) -> Path:
    layout = artifacts.ArtifactLayout(tmp_path_factory.mktemp("results"))
    bundle = synthetic.make_bundle(models=("egraphsage",))
    pipeline_fixtures.write_synthetic_s6_s7(layout)
    tables.generate_all(
        cl_tensors=bundle.cl_tensors,
        real_drift=bundle.real_drift,
        model_null_noise=bundle.model_null_noise,
        explainer_repeat_noise=bundle.explainer_repeat_noise,
        retention=bundle.retention,
        neighbourhood=bundle.neighbourhood,
        ear_comparison=bundle.ear_comparison,
        ear_ablations=bundle.ear_ablations,
        layout=layout,
        supported_classes=synthetic.SUPPORTED_CLASSES,
    )
    return layout.tables_root


def test_h1_recomputation_matches_generated_tables(demo_tables: Path, tmp_path: Path) -> None:
    result = s8_validation.run_s8_validation(demo_tables)
    assert result.status == "pass", [finding.as_dict() for finding in result.findings if finding.status == "fail"]
    assert not result.missing
    assert result.configurations  # at least one (strategy, model, explainer) configuration
    assert result.stage_report and result.stage_report["overall"] in {"pass", "warn"}
    checks = {finding.check for finding in result.findings}
    assert any(check.startswith("h1.fraction_above_floor") for check in checks)
    assert any(check.startswith("h1.p_binomial") for check in checks)
    assert any(check.startswith("h1.p_binomial_holm") for check in checks)
    assert any(check.startswith("noise_floor.floor_q95_matches_table") for check in checks)
    report = s8_validation.write_report(result, tmp_path / "s8_validation.md")
    text = report.read_text(encoding="utf-8")
    assert "Status: **PASS**" in text
    assert "Recomputed H1 per configuration" in text


def test_h1_recomputation_detects_tampered_table(demo_tables: Path, tmp_path: Path) -> None:
    tampered = tmp_path / "tables"
    shutil.copytree(demo_tables, tampered)
    path = tampered / "rq1_h1_tests.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "fraction_above_floor"] = float(frame.loc[0, "fraction_above_floor"]) + 0.25
    frame.to_csv(path, index=False)
    result = s8_validation.run_s8_validation(tampered)
    assert result.status == "fail"
    failures = [finding for finding in result.findings if finding.status == "fail"]
    assert any("fraction_above_floor" in finding.check for finding in failures)


def test_waiting_report_names_missing_artifacts(tmp_path: Path) -> None:
    result = s8_validation.run_s8_validation(tmp_path)
    assert result.status == "waiting"
    assert "rq1_anchor_level.csv" in result.missing
    report = s8_validation.write_report(result, tmp_path / "s8_validation.md").read_text(encoding="utf-8")
    assert "WAITING" in report
    assert "run_analysis.py --mode real" in report
