"""`paper_writing/manuscript/numbers.json` exporter — schema build, validation, extraction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import numbers as nm


def _tables() -> dict[str, pd.DataFrame]:
    h1 = pd.DataFrame(
        [
            {
                "strategy": "ft",
                "model": "egraphsage",
                "explainer": "IG",
                "n_anchors": 100,
                "fraction_above_floor": 0.45,
                "fraction_ci_lo": 0.36,
                "fraction_ci_hi": 0.54,
            }
        ]
    )
    comparison = pd.DataFrame(
        [
            {"comparison": "replay", "metric": "drift_primary", "mean_diff": -0.12, "ci_lo": -0.2, "ci_hi": -0.04},
        ]
    )
    correlations = pd.DataFrame(
        [
            {"strategy": "ft", "explainer": "IG", "spearman": 0.42, "spearman_ci_lo": 0.2, "spearman_ci_hi": 0.6},
        ]
    )
    rq3 = pd.DataFrame(
        [
            {"role": "delta", "delta_auc": 0.15, "delta_ci_lo": 0.02, "delta_ci_hi": 0.28, "h3_supported": True},
        ]
    )
    rq4 = pd.DataFrame(
        [
            {
                "model": "egraphsage",
                "variant": "ear_a",
                "row_kind": "h4_delta_vs_replay",
                "drift_diff_vs_replay": -0.08,
                "drift_ci_lo": -0.1,
                "drift_ci_hi": -0.05,
                "f1_diff_vs_replay": -0.004,
                "h4_supported": True,
            }
        ]
    )
    return {
        "rq1_h1_tests": h1,
        "rq1_strategy_comparison": comparison,
        "rq2_correlations": correlations,
        "rq3_auc": rq3,
        "rq4_comparison": rq4,
    }


def test_numbers_from_tables_extracts_c1_to_c5() -> None:
    payload = nm.numbers_from_tables(_tables(), git_sha="abc123", status="pending")
    claims = payload["claims"]
    assert set(claims) == {"C1", "C2", "C3", "C4", "C5"}
    assert claims["C1"]["value"] == pytest.approx(0.45)
    assert claims["C1"]["n_anchors"] == 100
    assert claims["C1"]["artifact"] == "results/text/tables/rq1_h1_tests.csv"
    assert claims["C2"]["value"] == pytest.approx(-0.12)
    assert claims["C3"]["value"] == pytest.approx(0.42)
    assert claims["C4"]["h3_supported"] is True
    assert claims["C5"]["f1_delta"] == pytest.approx(-0.004)
    assert claims["C5"]["h4_supported"] is True
    assert all(entry["status"] == "pending" for entry in claims.values())
    assert nm.validate_numbers_json(payload) == []


def test_build_paper_numbers_requires_git_sha() -> None:
    with pytest.raises(nm.NumbersError):
        nm.build_paper_numbers({}, git_sha="")


def test_invalid_ci_and_status_are_rejected() -> None:
    with pytest.raises(nm.NumbersError):
        nm.build_claim_entry(stat="s", value=1.0, ci95=[0.5], artifact="a.csv")
    with pytest.raises(nm.NumbersError):
        nm.build_claim_entry(stat="s", value=1.0, ci95=[0.5, 0.2], artifact="a.csv")  # lower > upper
    with pytest.raises(nm.NumbersError):
        nm.build_claim_entry(stat="s", value=1.0, ci95=[0.0, 1.0], artifact="a.csv", status="significant")
    with pytest.raises(nm.NumbersError):
        nm.build_claim_entry(stat="s", value=float("nan"), ci95=[0.0, 1.0], artifact="a.csv")


def test_validate_numbers_json_reports_structural_issues() -> None:
    bad = {"claims": {"C9": {"stat": "x"}}}
    issues = nm.validate_numbers_json(bad)
    assert any("missing top-level" in issue for issue in issues)
    assert any("unknown claim id" in issue for issue in issues)


def test_write_numbers_json_roundtrip(tmp_path: Path) -> None:
    payload = nm.numbers_from_tables(_tables(), git_sha="abc123", status="produced")
    path = nm.write_numbers_json(payload, tmp_path / "paper" / "numbers.json")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["git_sha"] == "abc123"
    assert loaded["protocol_version"] == "1"
    assert loaded["dataset"]["periods"] == 5
    assert nm.validate_numbers_json(loaded) == []


def test_missing_required_table_raises() -> None:
    with pytest.raises(nm.NumbersError):
        nm.numbers_from_tables({"rq1_h1_tests": pd.DataFrame()}, git_sha="abc")
