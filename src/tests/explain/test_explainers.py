"""Tests for S7 explainer adapters, fidelity checks and explanation cache."""

from __future__ import annotations

import numpy as np
import pytest

from src.tests.helpers_torch import build_toy_graph, requires_torch

pytestmark = requires_torch


def _model_and_data():
    from src.models.edge_sage import build_model
    from src.data.graphs import subsample_data

    data = subsample_data(build_toy_graph(), max_edges=120, seed=0)
    model = build_model(
        "egraphsage",
        node_dim=int(data.x.shape[1]),
        edge_dim=int(data.edge_attr.shape[1]),
        config={"model": {"hidden_dim": 16, "num_classes": int(data.y.max()) + 1}},
    )
    return model, data


def test_integrated_gradients_shape_determinism_and_completeness():
    from src.explain.ig import integrated_gradients, top_k_features

    model, data = _model_and_data()
    first = integrated_gradients(model, data, 0, steps=32, device="cpu")
    second = integrated_gradients(model, data, 0, steps=32, device="cpu")
    assert first["attributions"].shape == (data.edge_attr.shape[1],)
    assert first["target_class"] == second["target_class"]
    np.testing.assert_allclose(first["attributions"], second["attributions"], rtol=1e-6, atol=1e-8)
    # Completeness: sum(IG) ~= f(x) - f(baseline) for a smooth path.
    assert abs(first["completeness_delta"]) < 0.5
    top = top_k_features(first["attributions"], 3)
    assert len(top) == 3
    abs_values = [abs(item["attribution"]) for item in top]
    assert abs_values == sorted(abs_values, reverse=True)


def test_feature_fidelity_contract():
    from src.explain.fidelity import feature_fidelity
    from src.explain.ig import integrated_gradients

    model, data = _model_and_data()
    ig = integrated_gradients(model, data, 0, steps=16, device="cpu")
    fidelity = feature_fidelity(
        model, data, 0, ig["attributions"], k_values=(3, 5), random_trials=2, seed=42, device="cpu"
    )
    assert len(fidelity) == 2
    for block in fidelity:
        # keeping the top-k features can preserve or even slightly increase the
        # probability, so the sufficiency ratio is only bounded below by 0.
        assert block["sufficiency_ratio"] >= 0.0
        assert 0.0 <= block["base_probability"] <= 1.0
        assert 0.0 <= block["sufficiency_probability"] <= 1.0
        assert isinstance(block["comprehensiveness"], float)
        assert block["random_trials"] == 2


def test_gnn_explainer_returns_edge_mask():
    from src.explain.gnn_explainer import gnn_explainer

    model, data = _model_and_data()
    explanation = gnn_explainer(model, data, 0, epochs=5, seed=42, device="cpu")
    assert len(explanation["edge_mask"]) == int(data.num_edges)
    assert explanation["seed"] == 42
    assert explanation["top_edges"]
    assert explanation["target_edge"] == 0


def test_explanation_cache_roundtrip_and_key_semantics(tmp_path):
    from src.explain.cache import ExplanationCache, cache_key

    cache = ExplanationCache(tmp_path / "cache")
    key_a = cache_key(
        checkpoint_sha="abc", anchor_id=7, explainer="ig", seed=42, k=5, config_sha="cfg"
    )
    key_b = cache_key(
        checkpoint_sha="abc", anchor_id=7, explainer="ig", seed=43, k=5, config_sha="cfg"
    )
    assert key_a != key_b
    assert not cache.has(key_a)
    with pytest.raises(ValueError):
        # D-048/docs/05 §6: per-entry fidelity + runtime are required in the writer schema.
        cache.put(
            key_b,
            arrays={"attributions": np.arange(5, dtype="float64")},
            metadata={"explainer": "ig", "anchor_id": 7},
            index_row={
                "checkpoint_sha": "abc",
                "anchor_id": 7,
                "explainer": "ig",
                "seed": 43,
                "k": 5,
                "config_sha": "cfg",
            },
        )
    cache.put(
        key_a,
        arrays={"attributions": np.arange(5, dtype="float64")},
        metadata={"explainer": "ig", "anchor_id": 7},
        index_row={
            "checkpoint_sha": "abc",
            "anchor_id": 7,
            "explainer": "ig",
            "seed": 42,
            "k": 5,
            "config_sha": "cfg",
            "fidelity_sufficiency": 0.5,
            "fidelity_comprehensiveness": 0.1,
            "fidelity_over_random": 1.2,
            "runtime_s": 0.01,
        },
    )
    assert cache.has(key_a)
    loaded = cache.get(key_a)
    np.testing.assert_allclose(loaded["arrays"]["attributions"], np.arange(5, dtype="float64"))
    rows = cache.index_rows()
    assert len(rows) == 1 and rows[0]["explainer"] == "ig" and rows[0]["k"] == "5"
    assert float(rows[0]["runtime_s"]) == pytest.approx(0.01)
    assert float(rows[0]["fidelity_sufficiency"]) == pytest.approx(0.5)
