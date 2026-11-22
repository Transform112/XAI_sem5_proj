"""T24/D-052: sampled batching is canonical; reference-loss equivalence is diagnostic only."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.tests.helpers_torch import build_toy_graph, requires_torch

pytestmark = requires_torch


def _model_and_data(n_nodes: int = 30, n_edges: int = 180, n_classes: int = 3):
    from src.models.edge_sage import build_model

    data = build_toy_graph(n_nodes=n_nodes, n_edges=n_edges, n_classes=n_classes, seed=0)
    model = build_model(
        "egraphsage",
        node_dim=data.x.shape[1],
        edge_dim=data.edge_attr.shape[1],
        config={"model": {"hidden_dim": 16, "num_classes": n_classes}},
    )
    return model, data


def _weights(data):
    from src.train.trainer import balanced_class_weights

    return balanced_class_weights(data.y[data.train_mask].numpy(), int(data.y.max()) + 1)


def test_full_batch_loss_equals_mini_batch_reference():
    """Chosen transductive loss == exact edge-mean over mini-batch full-graph passes."""
    from src.train.trainer import full_batch_mean_loss, mini_batch_mean_loss

    model, data = _model_and_data(n_nodes=24, n_edges=120)
    model.eval()
    weights = _weights(data)
    train_idx = torch.nonzero(data.train_mask, as_tuple=False).flatten()
    full = full_batch_mean_loss(model, data, train_idx, weights, chunk_edges=7)
    mini = mini_batch_mean_loss(model, data, train_idx, weights, batch_size=10)
    assert float(full) == pytest.approx(float(mini), rel=1e-6, abs=1e-8)


def test_full_batch_loss_chunking_invariant():
    from src.train.trainer import full_batch_mean_loss

    model, data = _model_and_data()
    model.eval()
    weights = _weights(data)
    train_idx = torch.nonzero(data.train_mask, as_tuple=False).flatten()
    small_chunk = float(full_batch_mean_loss(model, data, train_idx, weights, chunk_edges=5))
    big_chunk = float(full_batch_mean_loss(model, data, train_idx, weights, chunk_edges=100_000))
    assert small_chunk == pytest.approx(big_chunk, rel=1e-6, abs=1e-8)


def test_trainer_full_batch_one_epoch_and_determinism():
    from src.train.trainer import TrainSettings, train_edge_classifier

    from src.utils.seed import seed_everything

    seed_everything(42)
    model, data = _model_and_data()
    settings = TrainSettings(
        max_epochs=1, patience=5, batch_size=32, batching="full_batch", chunk_edges=17
    )
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    result = train_edge_classifier(model, data, settings=settings, seed=42, device="cpu")
    assert result["timing"]["batching"] == "full_batch"
    assert result["timing"]["chunk_edges"] == 17
    assert result["epochs_run"] == 1
    assert 0.0 <= result["val"]["macro_f1"] <= 1.0
    changed = any(
        not torch.equal(before[name], parameter.detach())
        for name, parameter in model.named_parameters()
    )
    assert changed

    seed_everything(42)
    model2, data2 = _model_and_data()
    result2 = train_edge_classifier(model2, data2, settings=settings, seed=42, device="cpu")
    assert result2["test"]["macro_f1"] == pytest.approx(result["test"]["macro_f1"])
    assert set(result2["best_state"]) == set(before)


def test_trainer_mini_batch_reference_path_still_runs():
    from src.train.trainer import TrainSettings, train_edge_classifier

    model, data = _model_and_data(n_edges=90)
    settings = TrainSettings(max_epochs=1, patience=5, batch_size=16, batching="mini_batch")
    result = train_edge_classifier(model, data, settings=settings, seed=7, device="cpu")
    assert result["timing"]["batching"] == "mini_batch"
    assert result["epochs_run"] == 1


def test_train_settings_from_config_requires_known_batching_mode():
    from src.train.trainer import TrainSettings

    settings = TrainSettings.from_config(
        {"train": {"lr": 0.002, "batching": "full_batch", "chunk_edges": 1234}}
    )
    assert settings.batching == "full_batch" and settings.chunk_edges == 1234
    with pytest.raises(ValueError):
        TrainSettings.from_config({"train": {"batching": "nope"}})


# ---------------------------------------------------------------------------
# T24/D-052 canonical sampled batching
# ---------------------------------------------------------------------------
def test_default_batching_is_sampled_with_protocol_fanout():
    from src.train.trainer import TrainSettings

    settings = TrainSettings()
    assert settings.batching == "sampled"
    assert settings.neighbor_fanout == (10, 10)
    assert settings.batch_size == 1024
    from_config = TrainSettings.from_config({})
    assert from_config.batching == "sampled"
    assert from_config.neighbor_fanout == (10, 10)


def test_sampler_bounded_and_target_mapping():
    from src.train.sampler import EdgeNeighborSampler

    data = build_toy_graph(n_nodes=30, n_edges=180, n_classes=3, seed=0)
    sampler = EdgeNeighborSampler(data, fanout=(3, 3), max_edges_per_step=50)
    targets = np.arange(20)
    subgraph = sampler.sample(targets, np.random.default_rng(7))
    assert subgraph.n_sub_edges <= 50
    assert subgraph.n_sub_nodes <= data.num_nodes
    assert subgraph.n_targets == len(targets)
    # Target edges are kept first and their feature rows match the frozen graph.
    np.testing.assert_array_equal(subgraph.target_edge_ids.numpy(), targets)
    np.testing.assert_array_equal(subgraph.target_local_index.numpy(), np.arange(len(targets)))
    assert torch.equal(subgraph.edge_attr[subgraph.target_local_index], data.edge_attr[targets])
    # Same seed -> identical subgraph (deterministic batching).
    again = sampler.sample(targets, np.random.default_rng(7))
    assert torch.equal(subgraph.edge_attr, again.edge_attr)


def test_sampled_trainer_records_updates_and_measured_memory():
    from src.train.trainer import TrainSettings, train_edge_classifier

    model, data = _model_and_data(n_nodes=24, n_edges=120)
    settings = TrainSettings(
        max_epochs=2,
        patience=5,
        batch_size=32,
        batching="sampled",
        neighbor_fanout=(3, 3),
        max_edges_per_step=40,
    )
    result = train_edge_classifier(model, data, settings=settings, seed=42, device="cpu")
    timing = result["timing"]
    n_train = int(data.train_mask.sum())
    expected_updates = (n_train + 31) // 32
    assert timing["batching"] == "sampled"
    assert timing["updates_per_epoch"] == expected_updates
    assert timing["n_updates"] == expected_updates * 2
    assert timing["max_edges_per_step"] == 40
    assert timing["memory_device"] == "cpu"
    assert timing["max_memory_allocated_bytes"] is not None
    assert timing["max_memory_allocated_bytes"] > 0
    assert 0.0 <= result["val"]["macro_f1"] <= 1.0


# ---------------------------------------------------------------------------
# T26: phase timings, per-step subgraph sizes, A+B fast sampler (default OFF)
# ---------------------------------------------------------------------------
def test_sampled_trainer_phase_timings_and_subgraph_sizes():
    from src.train.trainer import TrainSettings, train_edge_classifier

    model, data = _model_and_data(n_nodes=24, n_edges=120)
    settings = TrainSettings(
        max_epochs=2, patience=5, batch_size=32, batching="sampled",
        neighbor_fanout=(3, 3), max_edges_per_step=40,
    )
    result = train_edge_classifier(model, data, settings=settings, seed=42, device="cpu")
    timing = result["timing"]
    for key in ("sample_seconds", "forward_backward_seconds", "eval_seconds", "final_eval_seconds"):
        assert key in timing and timing[key] >= 0.0
    assert len(timing["phase_seconds_per_epoch"]) == timing["epochs"]
    for phase in timing["phase_seconds_per_epoch"]:
        assert {
            "epoch", "sample_seconds", "forward_backward_seconds", "eval_seconds"
        } <= set(phase)
        assert phase["sample_seconds"] >= 0.0 and phase["forward_backward_seconds"] >= 0.0
    # Phase sums must be <= the measured train loop duration.
    assert timing["sample_seconds"] + timing["forward_backward_seconds"] <= timing["train_seconds"] + 1.0
    assert 0 < timing["subgraph_edges_mean"] <= 40
    assert 1 <= timing["subgraph_edges_min"] <= timing["subgraph_edges_max"] <= 40
    assert len(timing["subgraph_edges_per_epoch"]) == timing["epochs"]


def test_sampler_fast_path_default_on_with_reference_override():
    from src.train.trainer import TrainSettings

    # T27: vectorised dense-host sampler is canonical after validation.
    assert TrainSettings().sampler_fast_path is True
    assert TrainSettings.from_config({}).sampler_fast_path is True
    assert TrainSettings.from_config({"train": {"sampler_fast_path": False}}).sampler_fast_path is False


def _synthetic_graph(n_nodes: int, edges: list[tuple[int, int]], n_classes: int = 3):
    import torch
    from torch_geometric.data import Data

    src = [source for source, _ in edges]
    dst = [target for _, target in edges]
    n_edges = len(edges)
    data = Data(
        x=torch.ones((n_nodes, 4)),
        edge_index=torch.as_tensor([src, dst], dtype=torch.long),
        edge_attr=torch.arange(n_edges * 6, dtype=torch.float32).reshape(n_edges, 6),
        y=torch.as_tensor([index % n_classes for index in range(n_edges)], dtype=torch.long),
    )
    data.train_mask = torch.ones(n_edges, dtype=torch.bool)
    data.val_mask = torch.ones(n_edges, dtype=torch.bool)
    data.test_mask = torch.ones(n_edges, dtype=torch.bool)
    return data


def test_sampler_fast_path_dense_host_targets_first_and_deterministic():
    from src.train.sampler import EdgeNeighborSampler

    # Complete directed graph on 12 hosts: any 2-hop walk covers every host.
    edges = [(source, target) for source in range(12) for target in range(12) if source != target]
    data = _synthetic_graph(12, edges)
    targets = np.arange(20)
    sampler = EdgeNeighborSampler(data, fanout=(3, 3), max_edges_per_step=80, fast_path=True)
    first = sampler.sample(targets, np.random.default_rng(5))
    second = sampler.sample(targets, np.random.default_rng(5))
    assert first.sampler_mode.startswith("fast_dense_host")
    assert first.n_sub_edges <= 80
    np.testing.assert_array_equal(first.target_edge_ids.numpy(), targets)
    np.testing.assert_array_equal(first.target_local_index.numpy(), np.arange(len(targets)))
    assert torch.equal(first.edge_attr[first.target_local_index], data.edge_attr[targets])
    assert torch.equal(first.edge_attr, second.edge_attr)


def test_sampler_fast_path_sparse_graph_falls_back_without_rng_change():
    from src.train.sampler import EdgeNeighborSampler

    # 60 hosts but edges only among the first 10: 2-hop coverage is incomplete.
    edges = [(source, target) for source in range(10) for target in range(10) if source != target]
    data = _synthetic_graph(60, edges)
    targets = np.arange(20)
    standard = EdgeNeighborSampler(data, fanout=(3, 3), max_edges_per_step=80, fast_path=False)
    fast = EdgeNeighborSampler(data, fanout=(3, 3), max_edges_per_step=80, fast_path=True)
    assert fast._fast_context(targets, np.random.default_rng(0), set(range(10))) is None
    first = standard.sample(targets, np.random.default_rng(9))
    second = fast.sample(targets, np.random.default_rng(9))
    assert first.sampler_mode == "standard" and second.sampler_mode == "standard"
    assert torch.equal(first.edge_attr, second.edge_attr)
    assert torch.equal(first.edge_index, second.edge_index)


def test_sampler_fast_path_tiny_graph_context_pool_exhausted():
    """T26 fix: E < n_targets + context_budget must not raise (tiny toy graph)."""
    from src.train.sampler import EdgeNeighborSampler

    # Complete directed graph on 6 hosts = 30 edges; 20 targets, cap 60.
    edges = [(source, target) for source in range(6) for target in range(6) if source != target]
    data = _synthetic_graph(6, edges)
    targets = np.arange(20)
    sampler = EdgeNeighborSampler(data, fanout=(3, 3), max_edges_per_step=60, fast_path=True)
    subgraph = sampler.sample(targets, np.random.default_rng(3))
    assert subgraph.sampler_mode.startswith("fast_dense_host")
    assert subgraph.n_sub_edges <= min(int(data.num_edges), 60)
    np.testing.assert_array_equal(subgraph.target_edge_ids.numpy(), targets)
    np.testing.assert_array_equal(subgraph.target_local_index.numpy(), np.arange(len(targets)))
    # Degenerate case: every edge is a target and the context budget exceeds E.
    all_targets = np.arange(int(data.num_edges))
    exhausted = sampler.sample(all_targets, np.random.default_rng(4))
    assert exhausted.n_sub_edges == int(data.num_edges)
    np.testing.assert_array_equal(exhausted.target_edge_ids.numpy(), all_targets)
