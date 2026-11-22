"""Tests for S6 anchor selection, sampled 2-hop subgraphs and retention table."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.tests.helpers_torch import requires_torch

pytestmark = requires_torch

ANCHOR_CONFIG = {
    "confidence_min": 0.9,
    "max_per_class": 100,
    "max_per_period": 1000,
    "rare_min_eligible": 10,
    "rare_stratum": "rare",
    "sampling_seed": 42,
    "num_hops": 2,
    "fanout": 10,
}


class OracleModel(nn.Module):
    """Predicts the true label with high confidence (test double for anchors)."""

    def __init__(self, y: torch.Tensor, n_classes: int = 3) -> None:
        super().__init__()
        self.register_buffer("y", y.clone())
        self.n_classes = n_classes

    def forward(self, x, edge_index, edge_attr):
        n_edges = edge_index.shape[1]
        logits = torch.zeros((n_edges, self.n_classes), dtype=torch.float32)
        logits[torch.arange(n_edges), self.y[:n_edges]] = 10.0
        return logits


def _graph_with_rarest_class():
    from torch_geometric.data import Data

    n_nodes = 24
    rng = np.random.default_rng(7)
    # 40 train, 20 val, then 60 test edges: 45 class0, 5 class1 (rare), 10 class2.
    test_labels = [0] * 45 + [1] * 5 + [2] * 10
    labels = [0] * 40 + [0] * 20 + test_labels
    n_edges = len(labels)
    src = rng.integers(0, n_nodes, n_edges)
    dst = rng.integers(0, n_nodes, n_edges)
    data = Data(
        x=torch.ones((n_nodes, 8)),
        edge_index=torch.as_tensor(np.stack([src, dst])),
        edge_attr=torch.as_tensor(rng.normal(size=(n_edges, 6)).astype("float32")),
        y=torch.as_tensor(labels, dtype=torch.long),
    )
    train = torch.zeros(n_edges, dtype=torch.bool)
    val = torch.zeros(n_edges, dtype=torch.bool)
    test = torch.zeros(n_edges, dtype=torch.bool)
    train[:40] = True
    val[40:60] = True
    test[60:] = True
    data.train_mask = train
    data.val_mask = val
    data.test_mask = test
    data.node_global_ids = torch.arange(n_nodes)
    data.edge_row_ids = torch.arange(n_edges, dtype=torch.long)
    data.edge_ts = torch.arange(n_edges, dtype=torch.long) * 1000
    data.edge_host_src_global = torch.as_tensor(src)
    data.edge_host_dst_global = torch.as_tensor(dst)
    data.period = 0
    return data


def test_select_anchors_stratification_caps_and_rare_stratum():
    from src.anchors.build import select_anchors

    data = _graph_with_rarest_class()
    model = OracleModel(data.y)
    anchor_set = select_anchors(model, data, config=ANCHOR_CONFIG, device="cpu")
    assert anchor_set["n_eligible"] == 60
    assert anchor_set["rare_classes"] == [1]  # only 5 eligible < rare_min_eligible=10
    assert anchor_set["strata"]["rare"] == 5
    assert anchor_set["strata"]["0"] == 45
    assert anchor_set["strata"]["2"] == 10
    assert anchor_set["n_selected"] == 60
    anchor = anchor_set["anchors"][0]
    for key in (
        "anchor_id",
        "period_local_edge_index",
        "label",
        "predicted_class",
        "confidence",
        "sub_edge_index",
        "sub_edge_attr",
        "sub_node_global_ids",
        "target_local_index",
        "n_sub_nodes",
        "n_sub_edges",
    ):
        assert key in anchor
    assert int(anchor["target_local_index"].max()) < anchor["n_sub_edges"]
    assert anchor["confidence"] >= 0.9


def test_select_anchors_respects_per_class_cap():
    from src.anchors.build import select_anchors

    data = _graph_with_rarest_class()
    config = {**ANCHOR_CONFIG, "max_per_class": 10}
    anchor_set = select_anchors(OracleModel(data.y), data, config=config, device="cpu")
    assert anchor_set["strata"]["0"] == 10
    assert anchor_set["strata"]["2"] == 10


def test_reported_strata_are_post_cap():
    """T40/F1: D-033 strata provenance must describe the stored (post-cap) anchors."""
    from src.anchors.build import select_anchors

    data = _graph_with_rarest_class()
    config = {**ANCHOR_CONFIG, "max_per_class": 10, "max_per_period": 3}
    anchor_set = select_anchors(OracleModel(data.y), data, config=config, device="cpu")
    assert anchor_set["n_selected"] <= 3
    assert len(anchor_set["anchors"]) == anchor_set["n_selected"]
    assert sum(anchor_set["strata"].values()) == len(anchor_set["anchors"])


def test_select_anchors_is_deterministic_for_same_seed():
    from src.anchors.build import select_anchors

    data = _graph_with_rarest_class()
    first = select_anchors(OracleModel(data.y), data, config=ANCHOR_CONFIG, device="cpu", seed=42)
    second = select_anchors(OracleModel(data.y), data, config=ANCHOR_CONFIG, device="cpu", seed=42)
    assert [a["anchor_id"] for a in first["anchors"]] == [a["anchor_id"] for a in second["anchors"]]


def test_low_confidence_model_yields_no_anchors():
    from src.anchors.build import select_anchors

    data = _graph_with_rarest_class()

    class ConfusedModel(nn.Module):
        def forward(self, x, edge_index, edge_attr):
            return torch.zeros((edge_index.shape[1], 3), dtype=torch.float32)

    anchor_set = select_anchors(ConfusedModel(), data, config=ANCHOR_CONFIG, device="cpu")
    assert anchor_set["n_eligible"] == 0 and anchor_set["n_selected"] == 0


def test_retained_fraction_table():
    from src.anchors.build import retained_fraction_table, select_anchors

    data = _graph_with_rarest_class()
    anchor_set = select_anchors(OracleModel(data.y), data, config=ANCHOR_CONFIG, device="cpu")
    good = OracleModel(data.y)
    rows = retained_fraction_table({0: anchor_set}, {1: (good, data)}, device="cpu")
    assert rows and rows[0]["retained"] == rows[0]["total"]
    assert rows[0]["retained_fraction"] == 1.0
    flipped = data.y.clone()
    flipped = torch.where(flipped == 1, torch.tensor(0), flipped)
    bad = OracleModel(flipped)
    rows_bad = retained_fraction_table({0: anchor_set}, {1: (bad, data)}, device="cpu")
    assert rows_bad[0]["retained_fraction"] < 1.0


# ---------------------------------------------------------------------------
# T31: vectorised subgraph sampling equivalence + memory/perf regressions
# ---------------------------------------------------------------------------
def _reference_adjacency(edge_index):
    src = [int(v) for v in edge_index[0].tolist()]
    dst = [int(v) for v in edge_index[1].tolist()]
    adjacency = {}
    for source, target in zip(src, dst):
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
    return {node: sorted(values) for node, values in adjacency.items()}, src, dst


def _reference_sample_subgraph(data, target_edge, *, adjacency, src_list, dst_list, num_hops, fanout, rng):
    """Pre-T31 implementation (kept verbatim in tests as the equivalence oracle)."""
    import torch

    src = int(data.edge_index[0, target_edge])
    dst = int(data.edge_index[1, target_edge])
    node_set = {src, dst}
    frontier = [src, dst]
    for _hop in range(num_hops):
        next_frontier = []
        for node in frontier:
            candidates = [value for value in adjacency.get(node, []) if value not in node_set]
            if len(candidates) > fanout:
                picked = rng.choice(len(candidates), size=fanout, replace=False)
                candidates = [candidates[int(position)] for position in picked]
            for neighbour in candidates:
                if neighbour not in node_set:
                    node_set.add(neighbour)
                    next_frontier.append(neighbour)
        frontier = next_frontier
        if not frontier:
            break
    node_list = sorted(node_set)
    sub_edges = [
        edge_index_value
        for edge_index_value, (source, target) in enumerate(zip(src_list, dst_list))
        if source in node_set and target in node_set
    ]
    sub_edges_tensor = torch.as_tensor(sub_edges, dtype=torch.long)
    node_tensor = torch.as_tensor(node_list, dtype=torch.long)
    remap = {node: position for position, node in enumerate(node_list)}
    edge_index = data.edge_index[:, sub_edges_tensor]
    local_edge_index = torch.as_tensor(
        [[remap[int(value)] for value in edge_index[0]], [remap[int(value)] for value in edge_index[1]]],
        dtype=torch.long,
    )
    target_local = int((sub_edges_tensor == target_edge).nonzero(as_tuple=False)[0])
    return {
        "sub_edge_index": local_edge_index,
        "sub_edge_attr": data.edge_attr[sub_edges_tensor],
        "sub_edge_ids": sub_edges_tensor,
        "sub_node_global_ids": data.node_global_ids[node_tensor],
        "target_local_index": torch.tensor([target_local], dtype=torch.long),
        "target_edge": torch.tensor([target_edge], dtype=torch.long),
        "n_sub_nodes": int(node_tensor.numel()),
        "n_sub_edges": int(sub_edges_tensor.numel()),
    }


def _random_graph(n_nodes=40, n_edges=200_000, n_classes=5, seed=3):
    from torch_geometric.data import Data

    rng = np.random.default_rng(seed)
    src = rng.integers(0, n_nodes, n_edges)
    dst = rng.integers(0, n_nodes, n_edges)
    data = Data(
        x=torch.ones((n_nodes, 8)),
        edge_index=torch.as_tensor(np.stack([src, dst])),
        edge_attr=torch.as_tensor(rng.normal(size=(n_edges, 6)).astype("float32")),
        y=torch.as_tensor(rng.integers(0, n_classes, n_edges), dtype=torch.long),
    )
    data.node_global_ids = torch.arange(n_nodes, dtype=torch.long)
    data.edge_row_ids = torch.arange(n_edges, dtype=torch.long)
    data.period = 0
    return data


def test_build_adjacency_matches_reference():
    from src.anchors.build import _build_adjacency

    data = _random_graph(n_nodes=25, n_edges=500)
    reference, _src, _dst = _reference_adjacency(data.edge_index)
    adjacency, src, dst = _build_adjacency(data)
    assert adjacency == reference
    assert np.array_equal(src, data.edge_index[0].numpy())
    assert np.array_equal(dst, data.edge_index[1].numpy())


def test_sample_subgraph_matches_pre_t31_reference():
    from src.anchors.build import _build_adjacency, _sample_subgraph

    data = _random_graph(n_nodes=40, n_edges=2_000, seed=11)
    adjacency, src, dst = _build_adjacency(data)
    ref_adjacency, ref_src, ref_dst = _reference_adjacency(data.edge_index)
    for target in (0, 7, 13, 500, 1999):
        for seed in (42, 43, 99):
            new = _sample_subgraph(
                data, target, adjacency=adjacency, src_list=src, dst_list=dst,
                num_hops=2, fanout=10, rng=np.random.default_rng(seed + target),
            )
            old = _reference_sample_subgraph(
                data, target, adjacency=ref_adjacency, src_list=ref_src, dst_list=ref_dst,
                num_hops=2, fanout=10, rng=np.random.default_rng(seed + target),
            )
            for key in ("sub_edge_index", "sub_edge_attr", "sub_edge_ids", "sub_node_global_ids",
                        "target_local_index", "target_edge"):
                torch.testing.assert_close(new[key], old[key], rtol=0.0, atol=0.0)
            for key in ("n_sub_nodes", "n_sub_edges"):
                assert new[key] == old[key]


def test_large_graph_subgraph_sampling_is_vectorised_and_exact():
    """Regression: 100 anchors on 200k edges must be fast (no Python per-edge loop)."""
    import time

    from src.anchors.build import _build_adjacency, _sample_subgraph

    data = _random_graph(n_nodes=40, n_edges=200_000, seed=5)
    adjacency, src, dst = _build_adjacency(data)
    targets = list(range(0, 200_000, 2_000))
    ref_adjacency, ref_src, ref_dst = _reference_adjacency(data.edge_index)
    for target in targets[:2]:
        old = _reference_sample_subgraph(
            data, target, adjacency=ref_adjacency, src_list=ref_src, dst_list=ref_dst,
            num_hops=2, fanout=10, rng=np.random.default_rng(42 + target),
        )
        new = _sample_subgraph(
            data, target, adjacency=adjacency, src_list=src, dst_list=dst,
            num_hops=2, fanout=10, rng=np.random.default_rng(42 + target),
        )
        torch.testing.assert_close(new["sub_edge_ids"], old["sub_edge_ids"])
    started = time.monotonic()
    for target in targets:
        _sample_subgraph(
            data, target, adjacency=adjacency, src_list=src, dst_list=dst,
            num_hops=2, fanout=10, rng=np.random.default_rng(42 + target),
        )
    elapsed = time.monotonic() - started
    assert elapsed < 3.0, f"vectorised sampling too slow: {elapsed:.2f}s for 100 anchors"


class EdgeBatchOracleModel(nn.Module):
    """Oracle that raises if called without edge_batch (regression: no full-edge head)."""

    def __init__(self, y, n_classes=3):
        super().__init__()
        self.register_buffer("y", y.clone())
        self.n_classes = n_classes

    def forward(self, x, edge_index, edge_attr, edge_batch=None):
        if edge_batch is None:
            raise AssertionError("select_anchors must use the edge_batch head path")
        n = int(edge_batch.numel())
        logits = torch.zeros((n, self.n_classes), dtype=torch.float32)
        labels = self.y[edge_batch]
        logits[torch.arange(n), labels] = 10.0
        return logits


def test_select_anchors_uses_head_only_prediction_path():
    from src.anchors.build import select_anchors

    data = _graph_with_rarest_class()
    anchor_set = select_anchors(EdgeBatchOracleModel(data.y), data, config=ANCHOR_CONFIG, device="cpu")
    assert anchor_set["n_eligible"] == 60
    assert anchor_set["strata"]["0"] == 45


def test_retained_fraction_uses_anchor_period_graph():
    from src.anchors.build import retained_fraction_table, select_anchors

    data = _graph_with_rarest_class()
    anchor_set = select_anchors(EdgeBatchOracleModel(data.y), data, config=ANCHOR_CONFIG, device="cpu")
    other = _graph_with_rarest_class()
    other.y = torch.where(other.y == 0, torch.tensor(2), other.y)  # wrong predictions on other graph
    rows = retained_fraction_table(
        {0: anchor_set}, {1: (EdgeBatchOracleModel(data.y), other)}, graphs={0: data}, device="cpu"
    )
    assert rows and rows[0]["retained_fraction"] == 1.0


def test_dense_subgraph_capped_and_deterministic():
    from src.anchors.build import _build_adjacency, _sample_subgraph

    data = _random_graph(n_nodes=20, n_edges=5_000, seed=9)
    adjacency, src, dst = _build_adjacency(data)
    first = _sample_subgraph(
        data, 0, adjacency=adjacency, src_list=src, dst_list=dst,
        num_hops=2, fanout=10, rng=np.random.default_rng(42), max_subgraph_edges=100,
    )
    second = _sample_subgraph(
        data, 0, adjacency=adjacency, src_list=src, dst_list=dst,
        num_hops=2, fanout=10, rng=np.random.default_rng(42), max_subgraph_edges=100,
    )
    assert first["n_sub_edges"] <= 100
    assert first["sub_edge_ids"][int(first["target_local_index"][0])] == 0
    for key in ("sub_edge_index", "sub_edge_attr", "sub_edge_ids", "target_local_index"):
        torch.testing.assert_close(first[key], second[key])
