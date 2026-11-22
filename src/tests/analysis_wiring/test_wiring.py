"""Tests for the S8-S12 wiring: frames must equal the authoritative analysis metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tests.helpers_torch import requires_torch


# ---------------------------------------------------------------------------
# S8 frame builders (metric definitions imported from src/analysis/, never re-implemented)
# ---------------------------------------------------------------------------
def _vectors(seed: int = 0, size: int = 12):
    rng = np.random.default_rng(seed)
    return rng.normal(size=size), rng.normal(size=size)


def test_real_drift_frame_matches_analysis_metrics():
    from src.analysis import metrics_drift

    from src.analysis_wiring.frames import build_real_drift_frame

    a, b = _vectors(0)
    explanations = {(0, "anchor-1"): a, (1, "anchor-1"): b}
    meta = {"anchor-1": {"anchor_period": 0, "class": "DoS"}}
    frame = build_real_drift_frame(
        explanations, meta, strategy="ft", model="egraphsage", explainer="ig", seed=42
    )
    assert len(frame) == 2  # diagonal t=0 plus t=1; checkpoint_j is the anchor period
    assert sorted(frame["checkpoint_t"].tolist()) == [0, 1]
    assert set(frame["checkpoint_j"].tolist()) == {0}
    row = frame.loc[frame["checkpoint_t"] == 1].iloc[0]
    expected_sim = metrics_drift.attribution_similarity(a, b, ks=(3, 5, 10))
    expected_drift = metrics_drift.attribution_drift(a, b, ks=(3, 5, 10))
    for key, value in expected_sim.items():
        assert row[f"sim_{key}"] == pytest.approx(float(value))
    for key, value in expected_drift.items():
        assert row[f"drift_{key}"] == pytest.approx(float(value))
    assert row["drift_primary"] == pytest.approx(float(expected_drift["jaccard_top5"]))
    assert int(row["checkpoint_j"]) == 0 and int(row["checkpoint_t"]) == 1


def test_real_drift_frame_skips_non_later_checkpoints():
    from src.analysis_wiring.frames import build_real_drift_frame

    a, b = _vectors(1)
    explanations = {(1, "a"): a, (0, "a"): b, (2, "a"): b}
    meta = {"a": {"anchor_period": 0, "class": "Benign"}}
    frame = build_real_drift_frame(
        explanations, meta, strategy="ft", model="m", explainer="ig", seed=1
    )
    # anchor period 0 with checkpoints 0, 1, 2 -> diagonal plus both later updates
    assert sorted(frame["checkpoint_t"].tolist()) == [0, 1, 2]
    assert set(frame["anchor_period"].tolist()) == {0}
    assert all(frame["checkpoint_t"] >= frame["checkpoint_j"])
    assert set(frame["checkpoint_j"].tolist()) == {0}


def test_model_null_noise_frame_matches_analysis_noise_floor():
    from src.analysis import noise_floor

    from src.analysis_wiring.frames import build_model_null_noise_frame

    a, b = _vectors(2)
    reference = {"a1": a}
    reseeded = {"a1": {0: a, 1: b}}
    meta = {"a1": {"anchor_period": 0, "class": "Fuzzers"}}
    cfg = noise_floor.NoiseFloorConfig(n_null_reseeds=2)
    frame = build_model_null_noise_frame(
        reference, reseeded, meta, strategy="ft", model="m", explainer="ig", seed=42, cfg=cfg
    )
    expected = noise_floor.null_update_drift(
        reference, reseeded, class_by_anchor={"a1": "Fuzzers"}, metric_set="feature", cfg=cfg
    )
    assert len(frame) == len(expected) >= 1
    for column in ("anchor_id", "class", "null_seed", "drift_primary"):
        assert column in frame.columns and column in expected.columns
    merged = frame.merge(
        expected[["anchor_id", "null_seed", "drift_primary"]],
        on=["anchor_id", "null_seed"],
        suffixes=("_ours", "_analysis"),
    )
    np.testing.assert_allclose(
        merged["drift_primary_ours"].to_numpy(), merged["drift_primary_analysis"].to_numpy()
    )
    assert frame["channel"].iloc[0] == noise_floor.MODEL_NULL
    assert {"model", "explainer", "strategy", "seed"} <= set(frame.columns)


def test_explainer_repeat_noise_frame_matches_analysis():
    from src.analysis import noise_floor

    from src.analysis_wiring.frames import build_explainer_repeat_noise_frame

    rng = np.random.default_rng(3)
    masks = [np.abs(rng.normal(size=20)) for _ in range(3)]
    repeats = {"a1": masks}
    meta = {"a1": {"anchor_period": 0, "class": "Worms"}}
    frame = build_explainer_repeat_noise_frame(
        repeats, meta, strategy="ft", model="m", explainer="gnn_explainer", metric_set="edge"
    )
    expected = noise_floor.explainer_repeat_drift(
        repeats, class_by_anchor={"a1": "Worms"}, metric_set="edge"
    )
    assert len(frame) == len(expected)
    assert frame["drift_primary"].tolist() == pytest.approx(expected["drift_primary"].tolist())
    assert frame["channel"].iloc[0] == noise_floor.EXPLAINER_REPEAT


def test_frame_columns_match_analysis_synthetic_contract():
    from src.analysis import synthetic

    from src.analysis_wiring.frames import (
        build_ear_frames,
        build_explainer_repeat_noise_frame,
        build_neighbourhood_frame,
        build_real_drift_frame,
        build_retention_frame,
    )

    bundle = synthetic.make_bundle(seed=42)
    a, b = _vectors(4)
    real = build_real_drift_frame(
        {(0, "x"): a, (1, "x"): b},
        {"x": {"anchor_period": 0, "class": "DoS"}},
        strategy="ft",
        model="m",
        explainer="ig",
        seed=1,
    )
    assert list(real.columns) == list(bundle.real_drift.columns)
    neighbourhood = build_neighbourhood_frame({}, {}, strategy="ft", model="m", explainer="ig", seed=1)
    # Required analysis columns must be present; edge_set_change is an engineering
    # diagnostic appended after the contract columns (ignored by analysis writers).
    assert list(bundle.neighbourhood.columns) == list(neighbourhood.columns)[: len(bundle.neighbourhood.columns)]
    assert "edge_set_change" in neighbourhood.columns
    retention = build_retention_frame({}, {}, strategy="ft", model="m")
    assert list(retention.columns) == list(bundle.retention.columns)
    comparison, ablations = build_ear_frames([])
    assert list(comparison.columns) == list(bundle.ear_comparison.columns)
    assert list(ablations.columns) == list(bundle.ear_ablations.columns)


# ---------------------------------------------------------------------------
# S10 structural change
# ---------------------------------------------------------------------------
@requires_torch
def test_neighbourhood_frame_hand_computed_host_jaccard():
    import torch
    from torch_geometric.data import Data

    from src.analysis_wiring.frames import build_neighbourhood_frame

    def graph(edges, row_ids):
        src = [edge[0] for edge in edges]
        dst = [edge[1] for edge in edges]
        data = Data(
            x=torch.ones((5, 2)),
            edge_index=torch.as_tensor([src, dst], dtype=torch.long),
            edge_attr=torch.zeros((len(edges), 3)),
            y=torch.zeros(len(edges), dtype=torch.long),
        )
        data.edge_row_ids = torch.as_tensor(row_ids, dtype=torch.long)
        data.node_global_ids = torch.arange(5, dtype=torch.long)
        return data

    # checkpoint 0 graph: 0-1, 1-2, 2-3, 3-4; anchor stores hosts {0,1,2}.
    graph0 = graph([(0, 1), (1, 2), (2, 3), (3, 4)], [10, 11, 12, 13])
    graph1 = graph([(0, 1), (1, 2)], [10, 11])
    anchor_set = {
        0: {
            "anchors": [
                {
                    "anchor_id": 10,
                    "label": 0,
                    "host_src_global": 0,
                    "host_dst_global": 1,
                    "sub_node_global_ids": [0, 1, 2],
                    "sub_edge_ids": [10, 11],
                }
            ]
        }
    }
    frame = build_neighbourhood_frame(
        {0: graph0, 1: graph1}, anchor_set, strategy="ft", model="m", explainer="ig", seed=1
    )
    assert len(frame) == 2  # diagonal t=0 plus later checkpoint t=1
    diagonal = frame.loc[frame["checkpoint_t"] == 0].iloc[0]
    row = frame.loc[frame["checkpoint_t"] == 1].iloc[0]
    assert int(diagonal["checkpoint_j"]) == 0
    assert diagonal["neighbourhood_change"] == pytest.approx(0.25)  # {0,1,2} vs {0,1,2,3}
    assert diagonal["edge_set_change"] == pytest.approx(1.0 / 3.0)  # {10,11} vs {10,11,12}
    # BFS at checkpoint 1 from hosts {0,1}: 1 -> 2, so hosts {0,1,2} (J=1.0).
    assert row["neighbourhood_change"] == pytest.approx(0.0)
    assert int(row["checkpoint_j"]) == 0 and int(row["checkpoint_t"]) == 1
    assert float(row["degree"]) > 0.0


# ---------------------------------------------------------------------------
# End-to-end wiring proof (analysis toolkit fixtures; no real results touched)
# ---------------------------------------------------------------------------
def test_synthetic_wiring_writes_all_tables_and_validates(tmp_path):
    from src.analysis_wiring.pipeline import run_synthetic_wiring

    summary = run_synthetic_wiring(out_root=tmp_path, seed=42)
    assert summary["mode"] == "synthetic"
    assert "NOT real results" in summary["banner"]
    tables = Path(summary["tables_root"])
    expected = [
        "cl_metrics.csv",
        "drift_metrics_ig.csv",
        "drift_metrics_gnnexplainer.csv",
        "noise_floor.csv",
        "rq1_aggregated.csv",
        "rq1_anchor_level.csv",
        "rq1_h1_tests.csv",
        "rq2_anchor_level.csv",
        "rq2_correlations.csv",
        "rq3_auc.csv",
        "rq4_comparison.csv",
    ]
    for name in expected:
        assert (tables / name).is_file(), f"missing wired table {name}"
    # The wiring proof covers the S8-S13 table contract; the engine-owned S6/S7
    # artifacts (anchor sets, explanation cache) are not fabricated here.
    validation = summary["validation"]
    assert validation["scope"].startswith("table contract")
    assert validation["table_contract_passed"] is True, validation["table_failures"]
    assert validation["engine_artifacts_absent"], "synthetic proof should not fabricate S6/S7 artifacts"


def test_real_wiring_reports_missing_artifacts(tmp_path):
    from src.analysis_wiring.frames import AnalysisWiringError
    from src.analysis_wiring.pipeline import required_real_inputs, run_real_wiring

    missing = required_real_inputs(out_root=tmp_path)
    assert missing, "fresh temp root must report missing artifacts"
    assert any("cl_matrix" in item for item in missing)
    with pytest.raises(AnalysisWiringError) as excinfo:
        run_real_wiring(out_root=tmp_path, strategy="ft", model="egraphsage", seed=42)
    assert "waiting for frozen artifacts" in str(excinfo.value)


def test_real_drift_frame_counts_own_period_pairs_including_diagonal():
    """T49 follow-up: one row per (anchor, checkpoint_j=anchor_period, checkpoint_t>=j)."""
    import numpy as np
    from src.analysis_wiring.frames import build_real_drift_frame

    explanations = {}
    meta = {}
    for anchor_id, anchor_period, checkpoints in (
        ("a1", 1, (1, 2, 3, 4, 5)),
        ("a2", 3, (3, 4, 5)),
    ):
        for checkpoint in checkpoints:
            explanations[(checkpoint, anchor_id)] = np.arange(5, dtype=float) + checkpoint
        meta[anchor_id] = {"anchor_period": anchor_period, "class": "DoS"}
    frame = build_real_drift_frame(
        explanations, meta, strategy="ft", model="m", explainer="ig", seed=42
    )
    assert len(frame) == 5 + 3
    assert set(frame["checkpoint_j"].tolist()) == {1, 3}
    assert len(frame[["anchor_id", "checkpoint_j", "checkpoint_t"]].drop_duplicates()) == len(frame)
    counts = frame.groupby("anchor_id").agg(
        n_pairs=("checkpoint_t", "size"),
        min_t=("checkpoint_t", "min"),
        max_t=("checkpoint_t", "max"),
    )
    assert counts.loc["a1"].to_dict() == {"n_pairs": 5, "min_t": 1, "max_t": 5}
    assert counts.loc["a2"].to_dict() == {"n_pairs": 3, "min_t": 3, "max_t": 5}
    diagonal = frame[frame["checkpoint_t"] == frame["checkpoint_j"]]
    assert len(diagonal) == 2


def test_discover_group_cl_matrices():
    import tempfile
    from pathlib import Path
    from scripts.run_analysis import discover_group_cl_matrices

    groups = [("ft", "egraphsage", 42), ("ewc", "gcn", 42)]
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp) / "results" / "tables"
        matrix_root = base / "matrix"
        for strategy, model, seed in groups:
            target = matrix_root / f"{strategy}_{model}_seed{seed}"
            target.mkdir(parents=True)
            (target / "cl_matrix.json").write_text("{}", encoding="utf-8")
        paths = discover_group_cl_matrices(str(base / "cl_matrix.json"), groups)
        assert paths is not None and set(paths) == set(groups)
        assert all(path.is_file() for path in paths.values())
        missing = discover_group_cl_matrices(str(base / "cl_matrix.json"), groups[:1] + [("ft", "gcn", 42)])
        assert missing is None
