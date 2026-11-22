"""Tests for the D-049 pre-flight training memory guard."""

from __future__ import annotations

import pytest

from src.utils.memory import check_memory_budget, device_memory_info, estimate_training_peak_bytes


def test_estimate_scales_with_edges_and_hidden_dim():
    small = estimate_training_peak_bytes(
        n_edges=100_000, node_dim=8, edge_dim=50, hidden_dim=128, n_classes=10
    )
    large = estimate_training_peak_bytes(
        n_edges=1_131_909, node_dim=8, edge_dim=50, hidden_dim=128, n_classes=10
    )
    wider = estimate_training_peak_bytes(
        n_edges=1_131_909, node_dim=8, edge_dim=50, hidden_dim=256, n_classes=10
    )
    assert small < large < wider
    # Calibration anchor: full-size hidden-128 must estimate in the 2–4 GiB range
    # (measured 2.4–3 GB peak on the canonical lane).
    assert 2.0 * 1024**3 <= large <= 4.5 * 1024**3


def test_full_size_estimate_does_not_fit_legacy_3gb_card(monkeypatch):
    import src.utils.memory as memory_module

    monkeypatch.setattr(
        memory_module,
        "device_memory_info",
        lambda device: {"device": "cuda", "total_bytes": 3 * 1024**3, "free_bytes": 28 * 1024**2},
    )
    result = check_memory_budget(
        n_edges=1_131_909,
        model_config={"hidden_dim": 128, "num_classes": 10, "num_layers": 2},
        node_dim=8,
        edge_dim=50,
        device="cuda",
        guard="warn",
    )
    assert result["fits"] is False
    assert "OOM" in result["warning"]
    assert "rtx5060" in result["warning"]


def test_full_size_estimate_fits_8gb_canonical_card(monkeypatch):
    import src.utils.memory as memory_module

    monkeypatch.setattr(
        memory_module,
        "device_memory_info",
        lambda device: {"device": "cuda", "total_bytes": 8 * 1024**3, "free_bytes": 7 * 1024**3},
    )
    result = check_memory_budget(
        n_edges=1_131_909,
        model_config={"hidden_dim": 128, "num_classes": 10, "num_layers": 2},
        node_dim=8,
        edge_dim=50,
        device="cuda",
        guard="warn",
    )
    assert result["fits"] is True
    assert result["warning"] is None


def test_cpu_device_info_and_off_guard():
    info = device_memory_info("cpu")
    assert info["device"] == "cpu"
    result = check_memory_budget(
        n_edges=1_131_909,
        model_config={"hidden_dim": 128},
        node_dim=8,
        edge_dim=50,
        device="cpu",
        guard="off",
    )
    assert result["fits"] is None and result["warning"] is None
    assert result["estimated_peak_gib"] > 0


def test_guard_threshold_is_configurable(monkeypatch):
    import src.utils.memory as memory_module

    monkeypatch.setattr(
        memory_module,
        "device_memory_info",
        lambda device: {"device": "cuda", "total_bytes": 8 * 1024**3, "free_bytes": 8 * 1024**3},
    )
    strict = check_memory_budget(
        n_edges=1_131_909,
        model_config={"hidden_dim": 128, "num_classes": 10, "num_layers": 2},
        node_dim=8,
        edge_dim=50,
        device="cuda",
        reserve_fraction=0.99,
    )
    assert strict["fits"] is False
