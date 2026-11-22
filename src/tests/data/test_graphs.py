"""Tests for S1 graph artefact construction (frozen C1 state contract)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.clean import clean_frame, load_raw_frame
from src.data.graphs import build_graphs, build_period_graph, subsample_data
from src.data.periods import assign_periods
from src.data.preprocess import Preprocessor
from src.data.registry import load_dataset_spec
from src.data.splits import build_splits, freeze_split_masks
from src.tests.data.helpers import make_nf_row, write_nf_csv
from src.tests.helpers_torch import requires_torch

pytestmark = requires_torch


@pytest.fixture(scope="module")
def spec():
    return load_dataset_spec("NF-UNSW-NB15-v3")


def _rows(n: int = 240) -> list[dict[str, str]]:
    base = 1424242193000
    labels = ["Benign", "Benign", "Benign", "Exploits", "Fuzzers", "Worms"]
    rows = []
    for index in range(n):
        label = labels[index % len(labels)]
        rows.append(
            make_nf_row(
                FLOW_START_MILLISECONDS=str(base + index * 1000),
                IPV4_SRC_ADDR=f"10.2.0.{1 + index % 5}",
                IPV4_DST_ADDR=f"10.2.1.{1 + index % 4}",
                L4_SRC_PORT=str(40000 + index % 6),
                L4_DST_PORT=str(80 + index % 3),
                PROTOCOL="6" if index % 2 == 0 else "17",
                IN_BYTES=str(100 + index % 40),
                Attack=label,
                Label="0" if label == "Benign" else "1",
            )
        )
    return rows


@pytest.fixture()
def prepared(spec, tmp_path: Path):
    csv_path = write_nf_csv(tmp_path / spec.name / f"{spec.name}.csv", _rows())
    clean = clean_frame(load_raw_frame(spec, csv_path), spec)
    periods = assign_periods(clean.frame, n_periods=2)
    splits = build_splits(
        periods.frame,
        fractions={"train": 0.7, "val": 0.1, "test": 0.2},
        seed=42,
        min_groups_per_split=1,
        min_share=0.0,
    )
    train_positions = splits.masks[min(splits.masks)]["train"]
    preprocessor = Preprocessor(spec, config={}).fit(periods.frame.iloc[train_positions])
    frozen = freeze_split_masks(splits, periods.frame, dataset=spec.name)
    return {"frame": periods.frame, "splits": splits, "preprocessor": preprocessor, "frozen": frozen, "csv": csv_path}


def test_build_period_graph_contract(prepared, spec):
    data, stats = build_period_graph(
        prepared["frame"], prepared["splits"], prepared["preprocessor"], spec, period=0, node_feature_dim=8
    )
    assert data.x.shape[1] == 8
    assert bool((data.x == 1).all())
    assert data.edge_index.shape[0] == 2
    assert data.edge_index.min() >= 0 and data.edge_index.max() < data.num_nodes
    assert data.edge_attr.shape == (data.num_edges, len(prepared["preprocessor"].feature_names))
    assert data.y.shape[0] == data.num_edges
    for name in ("train", "val", "test"):
        assert str(getattr(data, f"{name}_mask").dtype) == "torch.bool"
    assert sum(stats["split_counts"].values()) == data.num_edges
    assert data.node_global_ids.numel() == data.num_nodes
    assert data.edge_row_ids.numel() == data.num_edges
    # Feature names must not contain IP/timestamp leakage columns.
    leaked = {spec.column("host_src"), spec.column("host_dst"), spec.column("timestamp_start")}
    assert not (set(prepared["preprocessor"].feature_names) & leaked)


def test_edge_attr_matches_preprocessor_output(prepared, spec):
    data, _ = build_period_graph(
        prepared["frame"], prepared["splits"], prepared["preprocessor"], spec, period=0
    )
    positions = np.flatnonzero(prepared["frame"]["_period"].to_numpy() == 0)
    expected = prepared["preprocessor"].transform(prepared["frame"].iloc[positions])
    np.testing.assert_allclose(data.edge_attr.numpy(), expected.astype("float32"), rtol=1e-6, atol=1e-6)


def test_build_graphs_is_deterministic_and_hashed(prepared, spec, tmp_path):
    kwargs = dict(
        frame=prepared["frame"],
        splits=prepared["splits"],
        preprocessor=prepared["preprocessor"],
        spec=spec,
        node_feature_dim=8,
        frozen_hash=prepared["frozen"]["frozen_hash"],
        expected_frozen_hash=prepared["frozen"]["frozen_hash"],
        dataset=spec.name,
    )
    first = build_graphs(out_dir=tmp_path / "a", **kwargs)
    second = build_graphs(out_dir=tmp_path / "b", **kwargs)
    assert first["artifact_fingerprint_sha256"] == second["artifact_fingerprint_sha256"]
    assert (tmp_path / "a" / "graphs_manifest.json").is_file()
    assert set(first["periods"]) == {"1", "2"}
    assert first["split_hash_matches_frozen"] is True


def test_build_graphs_refuses_frozen_hash_mismatch(prepared, spec, tmp_path):
    with pytest.raises(RuntimeError):
        build_graphs(
            frame=prepared["frame"],
            splits=prepared["splits"],
            preprocessor=prepared["preprocessor"],
            spec=spec,
            out_dir=tmp_path / "bad",
            node_feature_dim=8,
            frozen_hash=prepared["frozen"]["frozen_hash"],
            expected_frozen_hash="0" * 64,
            dataset=spec.name,
        )


def test_subsample_preserves_masks_and_metadata(prepared, spec):
    data, _ = build_period_graph(
        prepared["frame"], prepared["splits"], prepared["preprocessor"], spec, period=0
    )
    small = subsample_data(data, max_edges=50, seed=42)
    assert small.num_edges == 50
    total = sum(int(getattr(small, f"{name}_mask").sum()) for name in ("train", "val", "test"))
    assert total == 50
    assert small.edge_row_ids.numel() == 50
    assert small.node_global_ids.numel() == small.num_nodes
    assert int(small.edge_index.max()) < small.num_nodes
