"""S8/S9 noise floor — hand-computed q95/excess/H1 with the frozen protocol defaults."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.analysis import metrics_drift  # noqa: F401  (imported for API completeness in fixtures)
from src.analysis import noise_floor as nf


def _noise_rows(values: list[float], *, class_name: str, explainer: str = "IG", model: str = "m", channel: str = "model_null") -> list[dict]:
    return [
        {
            "channel": channel,
            "model": model,
            "explainer": explainer,
            "anchor_id": f"a{index}",
            "class": class_name,
            "drift_primary": value,
        }
        for index, value in enumerate(values)
    ]


def test_per_class_q95_hand_computed_with_pooling_fallback() -> None:
    frame = pd.DataFrame(
        _noise_rows([0.1, 0.2, 0.3, 0.4, 0.5], class_name="A")
        + _noise_rows([0.6, 0.7], class_name="B")
    )
    cfg = nf.NoiseFloorConfig(min_anchors_per_class=3, n_resamples=100)
    table = nf.per_class_thresholds(frame, cfg=cfg, channel=nf.MODEL_NULL)
    assert table.group_columns == ("explainer", "class")
    # class A n=5 >= 3 -> own linear-interpolated q95: index 4*0.95 = 3.8 -> 0.4 + 0.8*0.1 = 0.48
    assert table.lookup(("IG", "A")) == pytest.approx(0.48)
    # class B n=2 < 3 -> documented global fallback; global q95 over sorted 7 values:
    # index 6*0.95 = 5.7 -> 0.6 + 0.7*0.1 = 0.67
    assert table.global_q95 == pytest.approx(0.67)
    assert table.lookup(("IG", "B")) == pytest.approx(0.67)
    assert table.source_for(("IG", "B")) == "global"
    assert table.source_for(("IG", "A")) == "class"
    assert table.n[("IG", "A")] == 5


def test_explainer_repeat_pairwise_distribution() -> None:
    base = np.array([1.0, 0.0, 0.0, 0.0])
    second = base + np.array([0.0, 0.2, 0.1, 0.05])  # top-3 = {0,1,2}
    third = base + np.array([0.0, 0.05, 0.2, 0.1])  # top-3 = {0,2,3}
    explanations = {"a": [base, base.copy(), base.copy()], "b": [base, second, third]}
    frame = nf.explainer_repeat_drift(explanations)
    assert len(frame) == 3 + 3  # C(3,2) pairs per anchor
    assert (frame[frame["anchor_id"] == "a"]["drift_primary"] == 0.0).all()
    assert frame[frame["anchor_id"] == "b"]["drift_jaccard_top3"].max() > 0.0
    with pytest.raises(nf.NoiseFloorError):
        nf.explainer_repeat_drift({"a": [base]})


def test_ig_repeat_canary_is_exactly_zero() -> None:
    base = np.array([0.3, -0.2, 0.5])
    canary = nf.ig_repeat_canary({"a": [base, base.copy(), base.copy()], "b": [base.copy(), base.copy(), base.copy()]})
    assert canary["max_abs_drift"] == 0.0
    assert canary["is_zero"] == 1.0
    jittered = nf.ig_repeat_canary({"a": [base, base + 1e-6, base]})
    assert jittered["max_abs_drift"] > 0.0


def test_null_update_requires_protocol_R5() -> None:
    base = np.array([1.0, 0.0, 0.0])
    with pytest.raises(nf.NoiseFloorError):
        nf.null_update_drift({"a": base}, {"a": {10_000 + i: base + 0.01 * i for i in range(4)}})
    frame = nf.null_update_drift({"a": base}, {"a": {10_000 + i: base + 0.01 * i for i in range(5)}})
    assert len(frame) == 5
    assert set(frame["channel"]) == {nf.MODEL_NULL}


def test_combined_null_requires_R10() -> None:
    base = np.array([1.0, 0.0, 0.0])
    with pytest.raises(nf.NoiseFloorError):
        nf.combined_null_drift({"a": base}, {"a": {10_000: [base] * 9}})
    frame = nf.combined_null_drift({"a": base}, {"a": {10_000: [base] * 10}})
    assert len(frame) == 10


def test_compute_excess_hand_computed() -> None:
    noise = pd.DataFrame(
        _noise_rows([0.1, 0.2, 0.3, 0.4, 0.5], class_name="A")
    )
    noise = pd.concat([noise.assign(anchor_id="a1"), noise.assign(anchor_id="a2")], ignore_index=True)
    real = pd.DataFrame(
        [
            {"model": "m", "explainer": "IG", "anchor_id": "a1", "class": "A", "strategy": "ft", "drift_primary": 0.6},
            {"model": "m", "explainer": "IG", "anchor_id": "a2", "class": "A", "strategy": "ft", "drift_primary": 0.2},
        ]
    )
    cfg = nf.NoiseFloorConfig(min_anchors_per_class=3, n_resamples=100)
    thresholds = nf.per_class_thresholds(noise, cfg=cfg, channel=nf.MODEL_NULL)
    excess = nf.compute_excess(
        real, noise, thresholds, join_keys=("model", "explainer", "anchor_id", "class"), cfg=cfg
    )
    first = excess[excess["anchor_id"] == "a1"].iloc[0]
    second = excess[excess["anchor_id"] == "a2"].iloc[0]
    assert first["null_mean"] == pytest.approx(0.3)
    # 10 pooled null pairs (two anchors) -> linear q95 of [0.1..0.5, 0.1..0.5] = 0.5
    assert first["floor_q95"] == pytest.approx(0.5)
    assert first["excess"] == pytest.approx(0.3)
    assert bool(first["above_floor"]) is True
    assert second["excess"] == pytest.approx(-0.1)
    assert bool(second["above_floor"]) is False


def test_compute_excess_rejects_unpaired_real_anchor() -> None:
    noise = pd.DataFrame(_noise_rows([0.1, 0.2, 0.3], class_name="A"))
    noise["anchor_id"] = "a1"
    real = pd.DataFrame(
        [{"model": "m", "explainer": "IG", "anchor_id": "missing", "class": "A", "drift_primary": 0.5}]
    )
    cfg = nf.NoiseFloorConfig(min_anchors_per_class=1, n_resamples=50)
    thresholds = nf.per_class_thresholds(noise, cfg=cfg, channel=nf.MODEL_NULL)
    with pytest.raises(nf.NoiseFloorError):
        nf.compute_excess(real, noise, thresholds, join_keys=("model", "explainer", "anchor_id", "class"))


def _excess_frame(n_above: int = 8, n_total: int = 40) -> pd.DataFrame:
    rows = []
    for index in range(n_total):
        class_name = "A" if index % 2 == 0 else "B"
        rows.append(
            {
                "strategy": "ft",
                "model": "m",
                "explainer": "IG",
                "class": class_name,
                "anchor_id": f"a{index}",
                "checkpoint_j": 1,
                "checkpoint_t": 2,
                "drift_primary": 0.5,
                "null_mean": 0.3,
                "excess": 0.2,
                "above_floor": index < n_above,
            }
        )
    return pd.DataFrame(rows)


def test_aggregate_drift_and_h1() -> None:
    frame = _excess_frame()
    cfg = nf.NoiseFloorConfig(n_resamples=200)
    aggregated = nf.aggregate_drift(frame, cfg=cfg)
    assert len(aggregated) == 2  # two classes
    assert aggregated["n_anchors"].sum() == 40
    assert aggregated["excess_mean"].tolist() == pytest.approx([0.2, 0.2])
    assert aggregated["above_floor_fraction"].sum() == pytest.approx(8 / 20 + 0 / 20)
    h1 = nf.h1_test(frame, cfg=cfg)
    assert len(h1) == 1
    row = h1.iloc[0]
    assert row["n_anchors"] == 40
    assert row["n_above_floor"] == 8
    assert row["fraction_above_floor"] == pytest.approx(0.2)
    assert row["fraction_ci_lo"] <= row["fraction_above_floor"] <= row["fraction_ci_hi"]
    assert row["p_binomial"] < 0.05
    assert "p_binomial_holm" in h1.columns and "p_wilcoxon_holm" in h1.columns


def test_strategy_comparison_paired_difference() -> None:
    rows = []
    for index in range(10):
        rows.append(
            {
                "strategy": "ft",
                "model": "m",
                "explainer": "IG",
                "seed": 42,
                "class": "A" if index % 2 == 0 else "B",
                "anchor_id": f"a{index}",
                "checkpoint_j": 1,
                "checkpoint_t": 2,
                "drift_primary": 0.50 + 0.01 * index,
                "excess": 0.20,
            }
        )
        rows.append(
            {
                "strategy": "replay",
                "model": "m",
                "explainer": "IG",
                "seed": 42,
                "class": "A" if index % 2 == 0 else "B",
                "anchor_id": f"a{index}",
                "checkpoint_j": 1,
                "checkpoint_t": 2,
                "drift_primary": 0.35 + 0.01 * index,
                "excess": 0.05,
            }
        )
    frame = pd.DataFrame(rows)
    cfg = nf.NoiseFloorConfig(n_resamples=200)
    comparison = nf.strategy_comparison(frame, comparisons=("replay",), cfg=cfg)
    drift_row = comparison[comparison["metric"] == "drift_primary"].iloc[0]
    assert drift_row["mean_diff"] == pytest.approx(-0.15)
    assert drift_row["ci_hi"] < 0.0
    assert drift_row["p_value"] < 0.05
    assert "p_holm" in comparison.columns


def test_noise_floor_rows_reports_threshold_provenance() -> None:
    frame = pd.DataFrame(_noise_rows([0.1, 0.2, 0.3, 0.4, 0.5], class_name="A"))
    cfg = nf.NoiseFloorConfig(min_anchors_per_class=3)
    table = nf.per_class_thresholds(frame, cfg=cfg, channel=nf.MODEL_NULL)
    rows = nf.noise_floor_rows(frame, thresholds={nf.MODEL_NULL: table}, config=cfg)
    assert len(rows) == 1
    assert rows.iloc[0]["q95"] == pytest.approx(table.lookup(("IG", "A")))
    assert rows.iloc[0]["threshold_source"] in {"class", "global"}
    assert rows.iloc[0]["n_null_pairs"] == 5
