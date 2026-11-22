"""Artifact-contract tests: the toolkit must write the skeleton's exact paths."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.analysis import artifacts


def test_table_artifact_paths_match_claim_skeleton() -> None:
    # research/lit/claims_evidence_skeleton.md (C1-C5 + S5/S8/S12 outputs)
    assert artifacts.TABLE_ARTIFACTS["cl_metrics"] == "results/text/tables/cl_metrics.csv"
    assert artifacts.TABLE_ARTIFACTS["per_class_f1_csv"] == "results/text/tables/per_class_f1.csv"
    assert artifacts.TABLE_ARTIFACTS["per_class_f1_npz"] == "results/text/tables/per_class_f1.npz"
    assert artifacts.TABLE_ARTIFACTS["noise_floor"] == "results/text/tables/noise_floor.csv"
    assert artifacts.TABLE_ARTIFACTS["rq1_anchor_level"] == "results/text/tables/rq1_anchor_level.csv"
    assert artifacts.TABLE_ARTIFACTS["rq1_aggregated"] == "results/text/tables/rq1_aggregated.csv"
    assert artifacts.TABLE_ARTIFACTS["rq1_h1_tests"] == "results/text/tables/rq1_h1_tests.csv"
    assert artifacts.TABLE_ARTIFACTS["rq1_strategy_comparison"] == "results/text/tables/rq1_strategy_comparison.csv"
    assert artifacts.TABLE_ARTIFACTS["rq2_anchor_level"] == "results/text/tables/rq2_anchor_level.csv"
    assert artifacts.TABLE_ARTIFACTS["rq2_correlations"] == "results/text/tables/rq2_correlations.csv"
    assert artifacts.TABLE_ARTIFACTS["rq3_features"] == "results/text/tables/rq3_features.csv"
    assert artifacts.TABLE_ARTIFACTS["rq3_auc"] == "results/text/tables/rq3_auc.csv"
    assert artifacts.TABLE_ARTIFACTS["rq3_forgetting_labels"] == "results/text/tables/rq3_forgetting_labels.csv"
    assert artifacts.TABLE_ARTIFACTS["rq4_comparison"] == "results/text/tables/rq4_comparison.csv"
    assert artifacts.TABLE_ARTIFACTS["rq4_ablations"] == "results/text/tables/rq4_ablations.csv"
    assert artifacts.TABLE_ARTIFACTS["paper_numbers"] == "paper_writing/manuscript/numbers.json"
    assert {name for name, path in artifacts.TABLE_ARTIFACTS.items() if path.startswith("results/text/tables/")}


def test_figure_paths_use_skeleton_prefixes() -> None:
    for name, path in artifacts.FIGURE_ARTIFACTS.items():
        assert path.startswith("results/images/"), name
        assert Path(path).suffix == ".png"
    prefixes = {Path(path).name.split("_")[0] for path in artifacts.FIGURE_ARTIFACTS.values()}
    assert {"s8", "s9", "s10", "s11", "s12"} <= prefixes


def test_explainer_normalisation_and_slug() -> None:
    assert artifacts.canonical_explainer("IG") == "IG"
    assert artifacts.canonical_explainer("integrated gradients") == "IG"
    assert artifacts.canonical_explainer("gnnexplainer") == "GNNExplainer"
    assert artifacts.explainer_slug("GNNExplainer") == "gnnexplainer"
    assert artifacts.explainer_slug("ig") == "ig"
    with pytest.raises(ValueError):
        artifacts.canonical_explainer("SHAP")


def test_artifact_layout_resolves_contract_paths(tmp_path: Path) -> None:
    layout = artifacts.ArtifactLayout(tmp_path / "results")
    assert layout.table("rq1_h1_tests") == tmp_path / "results" / "text" / "tables" / "rq1_h1_tests.csv"
    assert layout.drift_metrics("IG") == tmp_path / "results" / "text" / "tables" / "drift_metrics_ig.csv"
    assert layout.drift_metrics("GNNExplainer") == tmp_path / "results" / "text" / "tables" / "drift_metrics_gnnexplainer.csv"
    assert layout.figure("s8_drift_real_vs_null") == tmp_path / "results" / "images" / "s8_drift_real_vs_null.png"
    assert layout.extra_table("rq3_coefficients") == tmp_path / "results" / "text" / "tables" / "rq3_coefficients.csv"
    assert layout.paper_numbers(tmp_path / "paper" / "numbers.json") == tmp_path / "paper" / "numbers.json"
    with pytest.raises(KeyError):
        layout.table("not_a_table")


def test_proposed_filenames_are_marked() -> None:
    for name in ("rq1_anchor_level", "rq1_aggregated", "rq1_h1_tests", "rq3_auc", "rq4_comparison"):
        assert name in artifacts.PROPOSED
    assert "cl_metrics" not in artifacts.PROPOSED  # fixed by Doc 2 S5 output
