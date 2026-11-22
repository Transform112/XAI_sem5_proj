"""S13 table generators — end-to-end artifact contract on synthetic fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import artifacts
from src.analysis import metrics_cl as mc
from src.analysis import numbers as numbers_module
from src.analysis import synthetic
from src.analysis import tables


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    layout = artifacts.ArtifactLayout(tmp_path_factory.mktemp("results"))
    bundle = synthetic.make_bundle(models=("egraphsage",))
    outputs = tables.generate_all(
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
    return layout, bundle, outputs


def test_all_skeleton_artifacts_are_written(generated) -> None:
    layout, _, outputs = generated
    expected = {
        "cl_metrics",
        "per_class_f1_csv",
        "per_class_f1_npz",
        "noise_floor",
        "drift_metrics_ig",
        "drift_metrics_gnnexplainer",
        "rq1_anchor_level",
        "rq1_aggregated",
        "rq1_h1_tests",
        "rq1_strategy_comparison",
        "rq2_anchor_level",
        "rq2_correlations",
        "rq3_features",
        "rq3_forgetting_labels",
        "rq3_auc",
        "rq3_coefficients",
        "rq4_comparison",
        "rq4_ablations",
    }
    assert expected <= set(outputs["paths"])
    for name in expected:
        path = Path(outputs["paths"][name])
        assert path.is_file(), name
        assert path.stat().st_size > 0, name
    # drift_metrics filenames use the explainer slug from the contract
    assert Path(outputs["paths"]["drift_metrics_ig"]).name == "drift_metrics_ig.csv"
    assert Path(outputs["paths"]["drift_metrics_gnnexplainer"]).name == "drift_metrics_gnnexplainer.csv"


def test_cl_metrics_and_tensor_roundtrip(generated) -> None:
    layout, bundle, outputs = generated
    frame = pd.read_csv(outputs["paths"]["cl_metrics"])
    # Engineer S5 writer schema (D-043): one run per file, D-033 dual metric rows.
    assert {"strategy", "model", "seed", "metric", "final_average_macro_f1", "forgetting", "bwt", "note"} <= set(frame.columns)
    assert set(frame["metric"]) == {"all_classes", "supported_set"}
    assert frame["final_average_macro_f1"].between(0.0, 1.0).all()
    assert frame["final_average_macro_f1"].notna().all()
    primary_frame = frame[frame["metric"] == "all_classes"].iloc[0]
    tensor = mc.load_per_class_f1(outputs["paths"]["per_class_f1_npz"])
    assert tensor.n_periods == 5
    assert tensor.final_macro_f1() == pytest.approx(float(primary_frame["final_average_macro_f1"]))
    assert bool(np.isfinite(tensor.forgetting())) and bool(np.isfinite(tensor.bwt()))
    # S5 writer-format artifacts are present for the validators/cross-check
    matrix_payload = __import__("json").loads(Path(outputs["paths"]["cl_matrix_json"]).read_text(encoding="utf-8"))
    assert matrix_payload["class_order"] == list(tensor.classes)
    assert "dual_macro_f1_per_cell" in matrix_payload
    per_class_csv = pd.read_csv(outputs["paths"]["per_class_f1_csv"])
    assert set(mc.PER_CLASS_F1_CSV_COLUMNS) <= set(per_class_csv.columns)


def test_noise_floor_has_protocol_channels_and_thresholds(generated) -> None:
    _, _, outputs = generated
    frame = pd.read_csv(outputs["paths"]["noise_floor"])
    assert {"model_null", "explainer_repeat"} <= set(frame["channel"])
    assert set(frame["q95"]) == set(frame["q95"])  # no NaN q95
    for channel in ("model_null", "explainer_repeat"):
        subset = frame[frame["channel"] == channel]
        assert (subset["n_null_pairs"] > 0).all()
        assert subset["threshold_source"].isin(["class", "global"]).all()
    thresholds = outputs["thresholds"]
    assert set(thresholds) == {"model_null", "explainer_repeat"}
    assert thresholds["model_null"].group_columns == ("explainer", "class")


def test_rq1_h1_table_columns_and_values(generated) -> None:
    _, _, outputs = generated
    frame = pd.read_csv(outputs["paths"]["rq1_h1_tests"])
    required = {
        "strategy",
        "model",
        "explainer",
        "n_anchors",
        "n_above_floor",
        "fraction_above_floor",
        "fraction_ci_lo",
        "fraction_ci_hi",
        "expected_fraction",
        "p_binomial",
        "p_binomial_holm",
        "p_wilcoxon",
        "excess_mean",
        "rank_biserial",
    }
    assert required <= set(frame.columns)
    assert (frame["fraction_above_floor"] >= 0).all() and (frame["fraction_above_floor"] <= 1).all()
    assert frame["expected_fraction"].eq(0.05).all()
    # the fine-tuning IG configuration is the H1 headline and must show elevated going-above-floor
    ft_ig = frame[(frame["strategy"] == "ft") & (frame["explainer"] == "IG")].iloc[0]
    assert ft_ig["fraction_above_floor"] > 0.05
    comparison = pd.read_csv(outputs["paths"]["rq1_strategy_comparison"])
    assert {"comparison", "metric", "mean_diff", "ci_lo", "ci_hi", "p_value", "p_holm"} <= set(comparison.columns)


def test_rq2_correlations(generated) -> None:
    _, _, outputs = generated
    frame = pd.read_csv(outputs["paths"]["rq2_correlations"])
    assert {"strategy", "explainer", "checkpoint_j", "checkpoint_t", "n_anchors", "spearman", "partial_spearman",
            "permutation_p_greater", "p_holm"} <= set(frame.columns)
    assert (frame["n_anchors"] >= 3).all()
    # synthetic neighbourhood change is coupled to drift for real updates
    assert frame["spearman"].mean() > 0.0


def test_rq3_auc_rows_and_h3_flag(generated) -> None:
    _, _, outputs = generated
    frame = pd.read_csv(outputs["paths"]["rq3_auc"])
    assert {"strategy", "cl_model", "explainer", "method", "role", "auc", "ci_lo", "ci_hi", "cv_group"} <= set(frame.columns)
    assert frame["method"].notna().all()
    delta = frame[frame["role"] == "delta"]
    assert not delta.empty
    assert delta["delta_auc"].notna().all()
    assert set(delta["h3_supported"].dropna().unique()) <= {True, False}
    labels = pd.read_csv(outputs["paths"]["rq3_forgetting_labels"])
    assert {"strategy", "cl_model", "checkpoint_t", "eval_period", "class", "label", "delta", "horizon"} <= set(labels.columns)
    features = pd.read_csv(outputs["paths"]["rq3_features"])
    assert "future_f1" not in features.columns
    coefficients = pd.read_csv(outputs["paths"]["rq3_coefficients"])
    assert {"model", "feature", "coef_standardized", "ci_lo", "ci_hi"} <= set(coefficients.columns)


def test_rq4_comparison_has_h4_decision(generated) -> None:
    _, _, outputs = generated
    frame = pd.read_csv(outputs["paths"]["rq4_comparison"])
    assert {"model", "variant", "seed", "drift_mean", "macro_f1", "row_kind", "h4_supported"} <= set(frame.columns)
    summary = frame[frame["row_kind"] != "run"]
    assert not summary.empty
    assert set(summary["variant"]) == {"ear_a", "ear_b"}
    assert summary["drift_diff_vs_replay"].notna().all()
    assert summary["h4_supported"].isin([True, False]).all()


def test_numbers_payload_from_generated_tables(generated) -> None:
    _, _, outputs = generated
    payload = numbers_module.numbers_from_tables(
        {
            "rq1_h1_tests": outputs["rq1"]["h1_tests"],
            "rq1_strategy_comparison": outputs["rq1"]["strategy_comparison"],
            "rq2_correlations": outputs["rq2"]["correlations"],
            "rq3_auc": outputs["rq3"]["auc"],
            "rq4_comparison": outputs["rq4"]["comparison"],
        },
        git_sha="deadbeef",
        status="pending",
        graph_lift={"delta_rand": 1.0, "ci95": [0.5, 1.5], "artifact": "results/text/data_audit/graph_lift_pilot.json"},
    )
    assert set(payload["claims"]) == {"C1", "C2", "C3", "C4", "C5", "C6"}
    assert numbers_module.validate_numbers_json(payload) == []


def test_write_table_is_deterministic(tmp_path: Path) -> None:
    frame = pd.DataFrame({"a": [1, 2], "b": [0.1, 0.2]})
    first = tables.write_table(frame, tmp_path / "t.csv")
    bytes_first = first.read_bytes()
    second = tables.write_table(frame, tmp_path / "t.csv")
    assert second.read_bytes() == bytes_first
