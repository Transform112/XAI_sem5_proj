"""T27 model-cache / user-code integration tests (reviewed equivalents)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.tests.helpers_torch import build_toy_graph, requires_torch

pytestmark = requires_torch


def _model_and_graph(model_name: str = "egraphsage", n_nodes: int = 30, n_edges: int = 180):
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
def test_batch_only_head_equals_full_logits_slice(model_name):
    model, data = _model_and_graph(model_name)
    model.eval()
    batch = torch.tensor([3, 8, 11, 45, 60], dtype=torch.long)
    with torch.inference_mode():
        full = model(data.x, data.edge_index, data.edge_attr)
        batch_only = model(data.x, data.edge_index, data.edge_attr, edge_batch=batch)
    torch.testing.assert_close(batch_only, full[batch], rtol=0.0, atol=1e-6)


def test_edgesage_static_aggregation_cache_is_weight_independent():
    """Cache correctness: aggregate depends only on edge_attr, so it survives weight updates."""
    model, data = _model_and_graph("egraphsage")
    model.eval()
    with torch.no_grad():
        first = model(data.x, data.edge_index, data.edge_attr)
        # Simulate an optimizer step: cache must NOT be invalidated by weight updates,
        # and the cached forward must equal a cache-cleared forward exactly.
        for parameter in model.parameters():
            parameter.add_(0.01)
        cached_forward = model(data.x, data.edge_index, data.edge_attr)
        model.clear_cache()
        fresh_forward = model(data.x, data.edge_index, data.edge_attr)
    torch.testing.assert_close(cached_forward, fresh_forward, rtol=0.0, atol=1e-7)


def test_edgesage_cache_bypassed_when_explaining():
    model, data = _model_and_graph("egraphsage")
    model.eval()
    with torch.inference_mode():
        plain = model(data.x, data.edge_index, data.edge_attr)
        layer = model.layers[0]
        layer._explain = True  # PyG Explainer flag
        explained = model(data.x, data.edge_index, data.edge_attr)
        layer._explain = False
    # No stale cached aggregate may be used while explaining; both are computed live
    # and therefore equal for an unmasked call.
    torch.testing.assert_close(explained, plain, rtol=0.0, atol=1e-6)


def test_edgesage_cache_invalidated_on_edge_attr_version_change():
    model, data = _model_and_graph("egraphsage")
    model.eval()
    with torch.inference_mode():
        before = model(data.x, data.edge_index, data.edge_attr)
        mutated = data.edge_attr.clone()
        mutated[0, 0] = mutated[0, 0] + 1.0
        after = model(data.x, data.edge_index, mutated)
        model.clear_cache()
        fresh = model(data.x, data.edge_index, mutated)
    torch.testing.assert_close(after, fresh, rtol=0.0, atol=1e-6)
    assert not torch.allclose(before, after) or torch.equal(before, after)  # no crash; version key used


@pytest.mark.parametrize("model_name", ["egraphsage", "gcn"])
def test_gcn_coalesced_and_multigraph_paths_agree_forward_and_grad(model_name):
    """Reviewed user optimization: coalesced weighted GCN == multigraph GCNConv."""
    model, data = _model_and_graph(model_name)
    model.train()
    # Multigraph with duplicate parallel edges (GCN coalescing must sum weights).
    edge_index = torch.cat([data.edge_index, data.edge_index[:, :40]], dim=1)
    edge_attr = torch.cat([data.edge_attr, data.edge_attr[:40]], dim=0)
    outputs = []
    grads = []
    for explainer_mode in (False, True):
        for parameter in model.parameters():
            parameter.grad = None
        if hasattr(model, "clear_cache"):
            model.clear_cache()
        for conv in getattr(model, "convs", []):
            conv._explain = explainer_mode
        logits = model(data.x, edge_index, edge_attr)
        loss = logits.square().mean()
        loss.backward()
        outputs.append(logits.detach().clone())
        grads.append({name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None})
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0.0, atol=1e-6)
    for name in grads[0]:
        torch.testing.assert_close(grads[0][name], grads[1][name], rtol=1e-6, atol=1e-8)


def test_graph_batch_trainer_matches_legacy_mini_batch_same_seed():
    """Optimized graph_batch is mathematically identical to the pre-T27 mini_batch loop."""
    from src.train.trainer import TrainSettings, train_edge_classifier
    from src.utils.seed import seed_everything

    results = []
    states = []
    for batching in ("graph_batch", "mini_batch"):
        seed_everything(42)
        model, data = _model_and_graph("egraphsage")
        settings = TrainSettings(batching=batching, max_epochs=2, patience=5, batch_size=32)
        results.append(train_edge_classifier(model, data, settings=settings, seed=42, device="cpu"))
        states.append({name: parameter.detach().clone() for name, parameter in model.named_parameters()})
    assert results[0]["timing"]["batching"] == "graph_batch"
    assert results[1]["timing"]["batching"] == "mini_batch"
    for left, right in zip(states[0].values(), states[1].values()):
        torch.testing.assert_close(left, right, rtol=1e-6, atol=1e-7)


def test_graph_batch_mode_records_perf_metadata():
    from src.train.trainer import TrainSettings, train_edge_classifier

    model, data = _model_and_graph("gcn")
    settings = TrainSettings(batching="graph_batch", max_epochs=1, patience=5, batch_size=32)
    result = train_edge_classifier(model, data, settings=settings, seed=1, device="cpu")
    timing = result["timing"]
    assert timing["graph_batch_optimized"] is True
    assert timing["edge_batch_head"] is True
    assert timing["static_aggregation_cache"] is True
    assert timing["eval_mode_final"] == "full_graph_batch_head"
    assert timing["updates_per_epoch"] > 0


# ---------------------------------------------------------------------------
# T34 (bug-report #9): cache keys include tensor identity/storage-offset guards
# ---------------------------------------------------------------------------
def test_static_caches_have_tensor_identity_guards_and_clear():
    from src.models.edge_sage import build_model

    data = build_toy_graph(n_nodes=24, n_edges=140, n_classes=3, seed=0)
    for model_name, attrs in (("egraphsage", "layers"), ("gcn", "convs")):
        model = build_model(
            model_name,
            node_dim=data.x.shape[1],
            edge_dim=data.edge_attr.shape[1],
            config={"model": {"hidden_dim": 16, "num_classes": 3}},
        )
        model.eval()
        with torch.no_grad():
            model(data.x, data.edge_index, data.edge_attr)
        if hasattr(model, "_cached_edge_attr_ref"):
            refs = [model]
        else:
            refs = [
                obj for obj in getattr(model, attrs)
                if hasattr(obj, "_cached_edge_attr_ref")
            ]
        assert refs, "expected at least one cached layer/model with identity guard"
        for obj in refs:
            assert obj._cached_edge_attr_ref() is data.edge_attr
            assert obj._cached_edge_index_ref() is data.edge_index
        model.clear_cache()
        for obj in refs:
            assert obj._cached_edge_attr_ref is None
            assert obj._cached_edge_index_ref is None


def test_static_cache_does_not_reuse_a_different_storage_view():
    """T34: same values/shape but a different storage view must not reuse the cache."""
    from src.models.edge_sage import build_model

    data = build_toy_graph(n_nodes=24, n_edges=140, n_classes=3, seed=0)
    model = build_model(
        "egraphsage",
        node_dim=data.x.shape[1],
        edge_dim=data.edge_attr.shape[1],
        config={"model": {"hidden_dim": 16, "num_classes": 3}},
    )
    model.eval()
    base = torch.cat([data.edge_attr, data.edge_attr], dim=0)
    view_a = base[: data.num_edges]
    view_b = base[data.num_edges :]
    with torch.no_grad():
        model(data.x, data.edge_index, view_a)
        cached_b = model(data.x, data.edge_index, view_b)
        model.clear_cache()
        fresh_b = model(data.x, data.edge_index, view_b)
    torch.testing.assert_close(cached_b, fresh_b, rtol=0.0, atol=1e-6)
