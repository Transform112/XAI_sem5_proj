"""S11 RQ3 — competing-baseline ΔAUC helper on small hand-built frames."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import rq3


def _features_frame(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for group in (1, 2, 3, 4):
        for index in range(10):
            drift = index / 10.0  # 0.0 .. 0.9
            label = 1 if drift > 0.5 else 0
            rows.append(
                {
                    "checkpoint_t": group,
                    "eval_period": 1,
                    "class": f"class_{index}",
                    "mean_excess_drift": drift,
                    "p90_excess_drift": drift + 0.02,
                    "confidence_mean": float(rng.normal(0.5, 0.3)),  # uninformative baseline
                    "margin_mean": float(rng.normal(0.0, 0.1)),
                    "retained_fraction": float(rng.uniform(0.8, 1.0)),
                    "f1": float(rng.uniform(0.5, 0.9)),  # uninformative baseline
                    "anchor_count": 10,
                    "label": label,
                }
            )
    return pd.DataFrame(rows)


def test_competing_baseline_drift_adds_auc() -> None:
    cfg = rq3.RQ3Config(group_column="checkpoint_t", n_resamples=300, n_bootstrap_coef=40, seed=42)
    result = rq3.competing_baseline_analysis(_features_frame(), cfg=cfg)
    assert result.n_rows == 40
    assert len(result.auc_table) == 4  # majority + 3 models
    assert set(result.auc_table["model"]) == {"majority_reference", "baselines_only", "drift_only", "drift_plus_baselines"}
    delta_rows = result.delta_auc_table
    assert set(delta_rows["comparison"]) == {
        "drift_plus_baselines_vs_baselines_only",
        "drift_only_vs_baselines_only",
    }
    headline = delta_rows[delta_rows["comparison"] == "drift_plus_baselines_vs_baselines_only"].iloc[0]
    assert headline["delta_auc"] > 0.2  # drift is by construction the signal
    assert headline["ci_lo"] > 0.0
    assert bool(headline["h3_supported"]) is True
    assert "exploratory" in headline["caveat"]
    coefficients = result.coefficient_table
    assert set(coefficients["model"]) == {"baselines_only", "drift_plus_baselines"}
    assert "mean_excess_drift" in set(coefficients["feature"])
    assert np.all(np.isfinite(coefficients["coef_standardized"]))


def test_single_class_label_raises_instead_of_reporting_auc_05() -> None:
    frame = _features_frame()
    frame["label"] = 0
    with pytest.raises(rq3.RQ3Error):
        rq3.competing_baseline_analysis(frame, cfg=rq3.RQ3Config(group_column="checkpoint_t", n_resamples=50, n_bootstrap_coef=10))


def test_missing_cv_group_column_raises() -> None:
    frame = _features_frame()
    with pytest.raises(rq3.RQ3Error):
        rq3.competing_baseline_analysis(frame, cfg=rq3.RQ3Config(group_column="eval_period"))


def test_build_forgetting_labels_and_features_contract() -> None:
    from src.analysis import metrics_cl as mc

    R_class = np.full((3, 3, 2), np.nan)
    R_class[0, 0] = [0.8, 0.7]
    R_class[1, 0] = [0.7, 0.6]
    R_class[1, 1] = [0.9, 0.85]
    R_class[2, 0] = [0.72, 0.62]
    R_class[2, 1] = [0.8, 0.75]
    R_class[2, 2] = [0.6, 0.55]
    tensor = mc.RMetricTensor(
        np.full((3, 3), np.nan),
        R_class=R_class,
        classes=("A", "B"),
        strategy="ft",
        model="egraphsage",
        seed=42,
    )
    labels = rq3.build_forgetting_labels(tensor, delta=0.05)
    assert {"strategy", "model", "checkpoint", "eval_period", "class", "f1", "label"}.issubset(labels.columns)
    drift = pd.DataFrame(
        [
            {
                "strategy": "ft",
                "model": "egraphsage",
                "explainer": "IG",
                "seed": 42,
                "anchor_id": f"a{index}",
                "class": "A" if index % 2 == 0 else "B",
                "checkpoint_t": 2,
                "excess": 0.1 + 0.01 * index,
                "drift_primary": 0.4,
            }
            for index in range(20)
        ]
    )
    labels_renamed = labels.rename(columns={"checkpoint": "checkpoint_t"})
    features = rq3.build_rq3_features(labels_renamed, drift, checkpoint_column="checkpoint_t")
    assert {"mean_excess_drift", "p90_excess_drift", "f1", "label"}.issubset(features.columns)
    model_features = set(rq3.DRIFT_FEATURE_COLUMNS) | set(rq3.BASELINE_FEATURE_COLUMNS) | set(rq3.CONTROL_COLUMNS)
    assert "future_f1" not in model_features  # no future F1 leakage into the model features
    assert features["label"].isin([0, 1]).all()


def test_build_rq3_features_rejects_no_shared_rows() -> None:
    labels = pd.DataFrame(
        [
            {
                "strategy": "ft",
                "model": "egraphsage",
                "checkpoint_t": 2,
                "eval_period": 1,
                "class": "A",
                "f1": 0.7,
                "label": 1,
            }
        ]
    )
    drift = pd.DataFrame(
        [{"strategy": "ft", "model": "gcn", "class": "A", "checkpoint_t": 2, "excess": 0.2}]
    )
    with pytest.raises(rq3.RQ3Error):
        rq3.build_rq3_features(labels, drift, checkpoint_column="checkpoint_t")
