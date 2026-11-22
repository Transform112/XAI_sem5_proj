"""Tests for S12 EAR-A/EAR-B hooks (hand-computed losses + training hook smoke)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.tests.helpers_torch import build_toy_graph, requires_torch

pytestmark = requires_torch


def test_ear_a_cosine_loss_hand_computed():
    from src.cl.ear import cosine_consistency_loss

    stored = torch.tensor([1.0, 0.0, 0.0])
    assert float(cosine_consistency_loss(stored.clone(), stored)) == pytest.approx(0.0)
    orthogonal = torch.tensor([0.0, 1.0, 0.0])
    assert float(cosine_consistency_loss(orthogonal, stored)) == pytest.approx(1.0)
    opposite = -stored
    assert float(cosine_consistency_loss(opposite, stored)) == pytest.approx(2.0)
    assert float(cosine_consistency_loss(torch.zeros(0), torch.zeros(0))) == pytest.approx(0.0)


def test_ear_b_soft_rank_loss_hand_computed():
    from src.cl.ear import ear_b_rank_loss, soft_rank_vector

    stored = torch.tensor([1.0, 0.0, 0.0, 0.0])
    rank = soft_rank_vector(stored, temperature=0.1)
    assert float(rank.sum()) == pytest.approx(1.0, abs=1e-6)
    assert float(ear_b_rank_loss(stored.clone(), stored, temperature=0.1)) == pytest.approx(0.0, abs=1e-5)
    disjoint = torch.tensor([0.0, 0.0, 1.0, 0.0])
    assert float(ear_b_rank_loss(disjoint, stored, temperature=0.1)) > 0.9
    assert float(ear_b_rank_loss(torch.zeros(4), torch.zeros(4), temperature=0.1)) == pytest.approx(0.0, abs=1e-6)


def test_grad_x_input_attributions_are_differentiable():
    from src.cl.ear import grad_x_input_attributions
    from src.models.edge_sage import build_model

    data = build_toy_graph()
    model = build_model(
        "egraphsage",
        node_dim=data.x.shape[1],
        edge_dim=data.edge_attr.shape[1],
        config={"model": {"hidden_dim": 16, "num_classes": int(data.y.max()) + 1}},
    )
    attributions = grad_x_input_attributions(model, data, 0)
    assert attributions.shape == (data.edge_attr.shape[1],)
    attributions.sum().backward()
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert grads and all(torch.isfinite(grad).all() for grad in grads)


def _model_and_data():
    from src.models.edge_sage import build_model

    data = build_toy_graph()
    model = build_model(
        "egraphsage",
        node_dim=data.x.shape[1],
        edge_dim=data.edge_attr.shape[1],
        config={"model": {"hidden_dim": 16, "num_classes": int(data.y.max()) + 1}},
    )
    return model, data


def test_ear_strategy_training_hook_smoke():
    from src.cl.ear import EARStrategy
    from src.cl.strategies import ReplayBuffer
    from src.train.trainer import TrainSettings

    model, data = _model_and_data()
    strategy = EARStrategy(
        variant="ear_a",
        beta=0.5,
        buffer=ReplayBuffer(capacity=20),
        max_anchors_per_step=2,
    )
    settings = TrainSettings(max_epochs=1, patience=5, batch_size=64)
    first = strategy.update(model, data, settings=settings, seed=42, device="cpu")
    assert first["strategy"] == "ear_a"
    assert first["stored_attributions"] > 0
    assert "test" in first
    # Second update exercises the penalty path (stored attributions now exist).
    second = strategy.update(model, data, settings=settings, seed=43, device="cpu")
    assert second["anchors_with_stored_attributions"] >= 0
    assert np.isfinite(second["consistency"]["mean_consistency_loss"]) or second["consistency"]["n_measured"] == 0
    assert second["ear_runtime_s"] >= 0.0


def test_ear_b_training_hook_runs():
    from src.cl.ear import EARStrategy
    from src.train.trainer import TrainSettings

    model, data = _model_and_data()
    strategy = EARStrategy(variant="ear_b", beta=1.0, max_anchors_per_step=1)
    result = strategy.update(model, data, settings=TrainSettings(max_epochs=1, batch_size=64), seed=1, device="cpu")
    assert result["strategy"] == "ear_b"
    assert result["variant"] == "ear_b"


def test_ear_beta_selection_grid():
    from src.cl.ear import EARStrategy

    strategy = EARStrategy(beta_grid=(0.1, 0.5, 1.0))
    result = strategy.select_beta({0.1: 0.40, 0.5: 0.61, 1.0: 0.58})
    assert result["selected_beta"] == 0.5
    assert strategy.beta == 0.5
    with pytest.raises(ValueError):
        strategy.select_beta({0.01: 0.9})


def test_build_ear_strategy_from_config(repo_root):
    from src.cl.strategies import build_strategy
    from src.utils.config import load_config

    config = load_config("configs/ear.yaml", project_root=repo_root)
    strategy = build_strategy("ear_a", config=config.data)
    assert strategy.variant == "ear_a"
    assert tuple(strategy.beta_grid) == (0.1, 0.5, 1.0)
    assert strategy.buffer.capacity == 2000
    strategy_b = build_strategy("ear_b", config=config.data)
    assert strategy_b.variant == "ear_b"


# T32: EAR grad-x-input attribution is head-only but numerically unchanged
def test_grad_x_input_attributions_head_only_matches_full_logits():
    from src.cl.ear import grad_x_input_attributions
    from src.models.edge_sage import build_model

    data = build_toy_graph(n_nodes=20, n_edges=220, n_classes=3, seed=4)
    model = build_model(
        "egraphsage",
        node_dim=data.x.shape[1],
        edge_dim=data.edge_attr.shape[1],
        config={"model": {"hidden_dim": 16, "num_classes": 3}},
    )
    head_only = grad_x_input_attributions(model, data, 7)

    def _full(data_obj):
        edge_attr = data_obj.edge_attr.detach().clone().requires_grad_(True)
        logits = model(data_obj.x, data_obj.edge_index, edge_attr)
        target_class = int(logits[7].argmax().detach())
        grads = torch.autograd.grad(
            logits[7, target_class], edge_attr, create_graph=True, retain_graph=True
        )[0]
        return (edge_attr[7] * grads[7]).reshape(-1)

    reference = _full(data)
    torch.testing.assert_close(head_only, reference, rtol=0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# T34 (bug-report #12): attribution storage is insertion-time / incremental
# ---------------------------------------------------------------------------
def test_ear_store_attributions_skips_already_attributed_entries(monkeypatch):
    import src.explain.ig as ig_module
    from src.cl.ear import EARStrategy
    from src.cl.strategies import ReplayBuffer

    data = build_toy_graph(n_nodes=20, n_edges=180, n_classes=3, seed=6)
    strategy = EARStrategy(
        variant="ear_a",
        buffer=ReplayBuffer(capacity=20, fanout=(5, 5)),
        attribution_source="ig",
        max_attributions_per_period=1000,
    )
    strategy.buffer.add_from_graph(data, seed=42)
    assert strategy.buffer.entries
    calls = {"n": 0}

    def _fake_ig(model, entry, target_edge, *, device="cpu"):
        calls["n"] += 1
        n_features = int(entry.edge_attr.shape[1])
        return {"attributions": np.zeros(n_features, dtype="float64")}

    monkeypatch.setattr(ig_module, "integrated_gradients", _fake_ig)
    first = strategy._store_attributions(object(), "cpu")
    assert first == len(strategy.buffer.entries)
    second = strategy._store_attributions(object(), "cpu")
    assert second == 0
    assert calls["n"] == first


def test_ear_store_attributions_refresh_recomputes(monkeypatch):
    import src.explain.ig as ig_module
    from src.cl.ear import EARStrategy
    from src.cl.strategies import ReplayBuffer

    data = build_toy_graph(n_nodes=20, n_edges=180, n_classes=3, seed=6)
    strategy = EARStrategy(
        buffer=ReplayBuffer(capacity=10, fanout=(5, 5)),
        max_attributions_per_period=1000,
    )
    strategy.buffer.add_from_graph(data, seed=42)
    calls = {"n": 0}

    def _fake_ig(model, entry, target_edge, *, device="cpu"):
        calls["n"] += 1
        return {"attributions": np.zeros(int(entry.edge_attr.shape[1]), dtype="float64")}

    monkeypatch.setattr(ig_module, "integrated_gradients", _fake_ig)
    strategy._store_attributions(object(), "cpu")
    strategy._store_attributions(object(), "cpu", refresh=True)
    assert calls["n"] == 2 * len(strategy.buffer.entries)


def test_ear_attribution_storage_is_bounded_per_period(monkeypatch):
    """T57: storage cost is capped per period instead of scanning all 2,000 entries."""
    import src.explain.ig as ig_module
    from src.cl.ear import EARStrategy
    from src.cl.strategies import ReplayBuffer

    data = build_toy_graph(n_nodes=20, n_edges=180, n_classes=3, seed=8)
    strategy = EARStrategy(
        buffer=ReplayBuffer(capacity=20, fanout=(5, 5)),
        max_attributions_per_period=2,
    )
    strategy.buffer.add_from_graph(data, seed=42)
    assert len(strategy.buffer.entries) > 2
    calls = {"n": 0}

    def _fake_ig(model, entry, target_edge, *, device="cpu"):
        calls["n"] += 1
        return {"attributions": np.zeros(int(entry.edge_attr.shape[1]), dtype="float64")}

    monkeypatch.setattr(ig_module, "integrated_gradients", _fake_ig)
    assert strategy._store_attributions(object(), "cpu") == 2
    assert calls["n"] == 2
    assert strategy._store_attributions(object(), "cpu") == 2
    assert calls["n"] == 4


def test_ear_factory_reads_t57_budget_config(repo_root):
    from src.cl.strategies import build_strategy
    from src.utils.config import load_config

    config = load_config("configs/ear.yaml", project_root=repo_root)
    strategy = build_strategy("ear_a", config=config.data)
    assert strategy.max_attributions_per_period == 64
    assert strategy.beta == 0.5
