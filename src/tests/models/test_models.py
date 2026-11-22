"""Tests for S3 models: shapes, mask-through-propagate, determinism, training."""

from __future__ import annotations

import numpy as np
import pytest

from src.tests.helpers_torch import build_toy_graph, requires_torch

pytestmark = requires_torch


def _build(name: str, data):
    from src.models.edge_sage import build_model

    return build_model(
        name,
        node_dim=int(data.x.shape[1]),
        edge_dim=int(data.edge_attr.shape[1]),
        config={"model": {"hidden_dim": 16, "num_layers": 2, "num_classes": int(data.y.max()) + 1}},
    )


def test_forward_shapes_for_both_models():
    data = build_toy_graph()
    for name in ("egraphsage", "gcn"):
        model = _build(name, data)
        logits = model(data.x, data.edge_index, data.edge_attr)
        assert logits.shape == (data.num_edges, int(data.y.max()) + 1)


def test_mask_through_propagate_contract():
    """Zeroing edge attributes must change the output for both architectures."""
    data = build_toy_graph()
    for name in ("egraphsage", "gcn"):
        model = _build(name, data)
        base = model(data.x, data.edge_index, data.edge_attr)
        masked = model(data.x, data.edge_index, __import__("torch").zeros_like(data.edge_attr))
        assert not __import__("torch").allclose(base, masked), f"{name}: edge mask bypassed propagate"


def test_training_is_deterministic_for_same_seed():
    from src.train.trainer import TrainSettings, train_edge_classifier
    from src.utils.seed import seed_everything

    data = build_toy_graph()
    settings = TrainSettings(max_epochs=2, patience=5, batch_size=64)
    seed_everything(7)
    first_model = _build("egraphsage", data)
    first = train_edge_classifier(first_model, data, settings=settings, seed=7, device="cpu")
    seed_everything(7)  # re-arm global RNGs before re-initialising the second model
    second_model = _build("egraphsage", data)
    second = train_edge_classifier(second_model, data, settings=settings, seed=7, device="cpu")
    assert first["test"]["macro_f1"] == second["test"]["macro_f1"]
    assert first["timing"]["epochs"] == second["timing"]["epochs"]


def test_trainer_returns_metrics_and_best_state():
    from src.train.trainer import TrainSettings, train_edge_classifier

    data = build_toy_graph()
    settings = TrainSettings(max_epochs=3, patience=5, batch_size=64)
    result = train_edge_classifier(_build("gcn", data), data, settings=settings, seed=3, device="cpu")
    assert 0.0 <= result["test"]["macro_f1"] <= 1.0
    assert 0.0 <= result["val"]["macro_f1"] <= 1.0
    assert result["best_state"]
    assert result["timing"]["epochs"] >= 1


def test_balanced_weights_ignore_absent_classes():
    from src.train.trainer import balanced_class_weights

    y = np.array([0, 0, 0, 1], dtype="int64")
    weights = balanced_class_weights(y, n_classes=3)
    assert float(weights[2]) == 0.0
    assert float(weights[0]) < float(weights[1])
