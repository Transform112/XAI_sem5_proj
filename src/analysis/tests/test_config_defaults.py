"""T19 config default proposals — YAML validity and frozen-value coverage."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config_defaults"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str) -> dict:
    path = CONFIG_DIR / name
    assert path.is_file(), f"missing config default proposal {path}"
    with open(path, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    assert isinstance(payload, dict)
    return payload


def test_drift_defaults_cover_protocol_values() -> None:
    config = _load("drift.yaml")
    metrics = config["metrics"]
    assert metrics["feature"]["primary"] == "jaccard_top5"
    assert set(metrics["feature"]["jaccard_topk"]["k"]) == {3, 5, 10}
    assert metrics["edge"]["jaccard_topm"]["m"] == 5
    assert metrics["report_all_metrics"] is True
    floor = config["noise_floor"]
    assert floor["channel_primary"] == "model_null"
    assert floor["components"]["explainer_repeat"]["repeats"] == 10
    assert floor["components"]["model_null"]["reseeds"] == 5
    assert floor["quantile"] == 0.95
    assert floor["pooling"]["by"] == ["explainer", "class"]
    assert floor["pooling"]["min_pairs_per_class"] == 20
    assert floor["excess"]["definition"].startswith("drift_real - drift_null_mean")
    assert config["aggregation"]["bootstrap"]["resamples"] == 1000
    assert config["h1"]["expected_above_fraction"] == 0.05
    assert config["outputs"]["noise_floor_csv"] == "results/text/tables/noise_floor.csv"


def test_rq3_defaults_cover_pre_registered_plan() -> None:
    config = _load("rq3.yaml")
    assert config["exploratory"] is True
    assert config["labels"]["delta"] == 0.05
    assert config["labels"]["horizon"] == "next_or_final"
    assert config["labels"]["no_post_hoc_variants"] is True
    assert set(config["features"]["drift"]) == {"mean_excess_drift", "p90_excess_drift"}
    assert "future_f1" in config["features"]["forbidden"]
    assert "retained_fraction" in config["competing_baselines"]["columns"]
    assert config["competing_baselines"]["reference_model"] == "majority_reference"
    assert config["model"]["C"] == 1.0
    assert config["cross_validation"]["group_primary"] == "checkpoint_t"
    assert config["statistics"]["n_resamples"] == 1000
    assert config["statistics"]["h3_supported_rule"] == "delta_auc_ci_lower_bound > 0"


def test_ear_defaults_cover_h4_and_ablations() -> None:
    config = _load("ear.yaml")
    assert config["replay"]["buffer_edges"] == 2000
    assert set(config["variants"]) == {"ear_a", "ear_b"}
    assert config["training"]["beta_grid"] == [0.1, 0.5, 1.0]
    assert config["ablations"]["buffer_edges"] == [500, 2000, 5000]
    assert set(config["ablations"]["attribution_source"]) == {"IG", "GNNExplainer"}
    assert config["h4"]["macro_f1_loss_points_max"] == 1.0
    assert config["h4"]["comparison_baseline"] == "replay"
    assert config["attribution_targets"]["report_anchor_fidelity_alongside_drift"] is True


def test_defaults_cite_authority_and_target_paths() -> None:
    for name in ("drift.yaml", "rq3.yaml", "ear.yaml"):
        config = _load(name)
        assert config["schema_version"] == 1
        assert "docs/05" in str(config["source"])
        assert str(config["source"]).startswith("docs/05")
    # The markdown explains the proposal contract and file mapping.
    markdown = (CONFIG_DIR / "analysis_config_defaults.md").read_text(encoding="utf-8")
    assert "PROPOSAL" in markdown
    assert "configs/drift.yaml" in markdown
    assert "configs/rq3.yaml" in markdown
    assert "configs/ear.yaml" in markdown
