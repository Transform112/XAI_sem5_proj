"""T27 training-performance tests: optimized path equivalence, both models, defaults."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.tests.helpers_torch import build_toy_graph, requires_torch

pytestmark = requires_torch


def _model_and_data(model_name: str = "egraphsage", n_nodes: int = 30, n_edges: int = 180):
    from src.models.edge_sage import build_model

    data = build_toy_graph(n_nodes=n_nodes, n_edges=n_edges, n_classes=3, seed=0)
    model = build_model(
        model_name,
        node_dim=data.x.shape[1],
        edge_dim=data.edge_attr.shape[1],
        config={"model": {"hidden_dim": 16, "num_classes": 3}},
    )
    return model, data


@pytest.mark.parametrize("model_name", ["egraphsage", "gcn"])
def test_optimized_global_id_batch_equals_relabelled_reference(model_name):
    """T27 equivalence: captured batch logits/loss match the reference path (<=1e-6)."""
    from src.train.fast import DeviceGraph
    from src.train.sampler import EdgeNeighborSampler

    model, data = _model_and_data(model_name)
    sampler = EdgeNeighborSampler(data, fanout=(5, 5), max_edges_per_step=100, fast_path=False)
    targets = np.arange(20)
    selected, _ = sampler.select_edge_ids(targets, np.random.default_rng(7))
    reference = sampler.sample(targets, np.random.default_rng(7))
    np.testing.assert_array_equal(np.asarray(selected)[: len(targets)], reference.target_edge_ids.numpy())

    graph = DeviceGraph(data, "cpu")
    selected_t = torch.as_tensor(selected, dtype=torch.long)
    edge_index_opt = graph.edge_index.index_select(1, selected_t)
    edge_attr_opt = graph.edge_attr.index_select(0, selected_t)

    model.eval()
    with torch.inference_mode():
        ref_logits = model(reference.x, reference.edge_index, reference.edge_attr)[: len(targets)]
        opt_logits = model(graph.x, edge_index_opt, edge_attr_opt)[: len(targets)]
    torch.testing.assert_close(opt_logits, ref_logits, rtol=0.0, atol=1e-6)

    weights = torch.ones(int(data.y.max()) + 1)
    criterion = torch.nn.CrossEntropyLoss(weight=weights)
    target_y = data.y[torch.as_tensor(targets, dtype=torch.long)]
    torch.testing.assert_close(
        criterion(opt_logits, target_y), criterion(ref_logits, target_y), rtol=0.0, atol=1e-6
    )


def test_optimized_defaults_and_reference_override():
    from src.train.trainer import TrainSettings

    default = TrainSettings()
    assert default.optimized_dispatch is True
    assert default.sampler_fast_path is True
    reference = TrainSettings(optimized_dispatch=False, sampler_fast_path=False)
    assert reference.optimized_dispatch is False and reference.sampler_fast_path is False
    from_config = TrainSettings.from_config({"train": {"optimized_dispatch": False, "sampler_fast_path": False}})
    assert from_config.optimized_dispatch is False and from_config.sampler_fast_path is False


@pytest.mark.parametrize("model_name", ["egraphsage", "gcn"])
def test_optimized_trainer_both_models_records_perf_fields(model_name):
    from src.train.trainer import TrainSettings, train_edge_classifier

    from src.utils.seed import seed_everything

    seed_everything(42)
    model, data = _model_and_data(model_name)
    settings = TrainSettings(max_epochs=1, patience=5, batch_size=32)
    result = train_edge_classifier(model, data, settings=settings, seed=42, device="cpu")
    timing = result["timing"]
    assert timing["optimized_dispatch"] is True
    assert timing["sampler_impl"] == "optimized_global_ids"
    assert timing["device_graph_cached"] is True
    assert timing["eval_mode_final"] == "full_graph"
    assert timing["updates_per_epoch"] > 0
    assert timing["phase_seconds_per_epoch"]


def test_reference_relabelled_path_still_callable():
    from src.train.trainer import TrainSettings, train_edge_classifier

    model, data = _model_and_data("egraphsage")
    settings = TrainSettings(
        max_epochs=1, patience=5, batch_size=32, optimized_dispatch=False, sampler_fast_path=False
    )
    result = train_edge_classifier(model, data, settings=settings, seed=7, device="cpu")
    timing = result["timing"]
    assert timing["optimized_dispatch"] is False
    assert timing["sampler_impl"] == "reference_relabelled"
    assert timing["device_graph_cached"] is False


def test_optimized_determinism_same_seed():
    from src.train.trainer import TrainSettings, train_edge_classifier
    from src.utils.seed import seed_everything

    settings = TrainSettings(max_epochs=2, patience=5, batch_size=32)
    results = []
    states = []
    for _ in range(2):
        seed_everything(42)
        model, data = _model_and_data("egraphsage")
        results.append(train_edge_classifier(model, data, settings=settings, seed=42, device="cpu"))
        states.append({key: value.detach().clone() for key, value in model.named_parameters()})
    assert results[0]["test"]["macro_f1"] == pytest.approx(results[1]["test"]["macro_f1"])
    for left, right in zip(states[0].values(), states[1].values()):
        assert torch.equal(left, right)


def test_optimized_tiny_graph_batches_do_not_crash():
    """Regression for the tiny E < targets+budget edge case through the fast trainer."""
    from src.train.trainer import TrainSettings, train_edge_classifier

    from torch_geometric.data import Data

    n_nodes = 6
    edges = [(source, target) for source in range(n_nodes) for target in range(n_nodes) if source != target]
    data = Data(
        x=torch.ones((n_nodes, 4)),
        edge_index=torch.as_tensor([[s for s, _ in edges], [t for _, t in edges]], dtype=torch.long),
        edge_attr=torch.randn((len(edges), 6)),
        y=torch.as_tensor([index % 3 for index in range(len(edges))], dtype=torch.long),
    )
    data.train_mask = torch.ones(len(edges), dtype=torch.bool)
    data.val_mask = torch.ones(len(edges), dtype=torch.bool)
    data.test_mask = torch.ones(len(edges), dtype=torch.bool)
    from src.models.edge_sage import build_model

    model = build_model("egraphsage", node_dim=4, edge_dim=6, config={"model": {"hidden_dim": 8, "num_classes": 3}})
    settings = TrainSettings(max_epochs=1, patience=5, batch_size=20, max_edges_per_step=60)
    result = train_edge_classifier(model, data, settings=settings, seed=1, device="cpu")
    assert result["timing"]["optimized_dispatch"] is True
    assert result["epochs_run"] == 1

def test_graph_batch_timing_has_no_misleading_sampler_flags():
    """T28-2 cleanup: graph_batch metrics report only mode-relevant fields."""
    from src.train.trainer import TrainSettings, train_edge_classifier

    model, data = _model_and_data("egraphsage")
    settings = TrainSettings(batching="graph_batch", max_epochs=1, patience=5, batch_size=32)
    timing = train_edge_classifier(model, data, settings=settings, seed=1, device="cpu")["timing"]
    assert timing["training_mode"] == "graph_batch"
    assert timing["batching"] == "graph_batch"
    # No sampler-specific / optimized-sampled flags in the canonical graph_batch metrics.
    for key in (
        "sampler_fast_path", "sampler_impl", "optimized_dispatch",
        "dense_host_fast_sampler", "device_graph_cached", "max_edges_per_step", "neighbor_fanout",
    ):
        assert key not in timing, f"graph_batch timing must not expose {key}"
    # Mode-relevant optimized flags are present.
    assert timing["graph_batch_optimized"] is True
    assert timing["edge_batch_head"] is True
    assert timing["static_aggregation_cache"] is True
    assert timing["training_mode_semantics"] == "legacy_transductive_full_graph_propagation"


def test_sampled_timing_keeps_sampler_flags_only_in_sampled_mode():
    from src.train.trainer import TrainSettings, train_edge_classifier

    model, data = _model_and_data("egraphsage")
    settings = TrainSettings(batching="sampled", max_epochs=1, patience=5, batch_size=32)
    timing = train_edge_classifier(model, data, settings=settings, seed=1, device="cpu")["timing"]
    assert timing["training_mode"] == "sampled"
    assert timing["sampler_fast_path"] is True
    assert timing["sampler_impl"] == "optimized_global_ids"
    assert timing["optimized_dispatch"] is True
    assert "graph_batch_optimized" not in timing
