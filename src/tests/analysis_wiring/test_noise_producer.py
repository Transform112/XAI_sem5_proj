"""Tests for the S8 noise-cache producer (D-048): repeat + null caches with fidelity."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from src.tests.helpers_torch import build_toy_graph, requires_torch

pytestmark = requires_torch


def _setup(tmp_path: Path):
    from src.models.edge_sage import build_model

    data = build_toy_graph(n_nodes=20, n_edges=120, seed=0)
    model = build_model(
        "egraphsage",
        node_dim=data.x.shape[1],
        edge_dim=data.edge_attr.shape[1],
        config={"model": {"hidden_dim": 16, "num_classes": int(data.y.max()) + 1}},
    )
    checkpoint = tmp_path / "M_1.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": {"hidden_dim": 16, "num_classes": int(data.y.max()) + 1},
        },
        checkpoint,
    )
    graph_file = tmp_path / "period_1.pt"
    torch.save(data, graph_file)
    # A minimal valid anchor set: one stored 2-hop-style subgraph.
    n_sub_edges = 8
    anchor = {
        "anchor_id": int(data.edge_row_ids[3]),
        "period": 0,
        "label": int(data.y[3]),
        "target_local_index": torch.tensor([0], dtype=torch.long),
        "sub_edge_index": data.edge_index[:, :n_sub_edges].clone(),
        "sub_edge_attr": data.edge_attr[:n_sub_edges].clone(),
        "n_sub_nodes": int(data.num_nodes),
        "n_sub_edges": int(n_sub_edges),
        "host_src_global": int(data.edge_host_src_global[3]),
        "host_dst_global": int(data.edge_host_dst_global[3]),
    }
    anchors_dir = tmp_path / "anchors"
    anchors_dir.mkdir()
    torch.save({"anchors": [anchor], "n_selected": 1}, anchors_dir / "period_1.pt")
    return graph_file, checkpoint, anchors_dir, data, model


def test_producer_emits_repeat_and_null_caches_with_fidelity(tmp_path):
    from src.analysis_wiring.frames import load_explanations_from_cache
    from scripts.build_noise_floor import combine_null_channels, produce_explainer_repeat, produce_model_null

    graph_file, checkpoint, anchors_dir, data, model = _setup(tmp_path)
    graph = torch.load(graph_file, weights_only=False)
    anchor_set = torch.load(anchors_dir / "period_1.pt", weights_only=False)
    anchors = anchor_set["anchors"]
    out_root = tmp_path / "explanations_noise"

    repeat = produce_explainer_repeat(
        model=model,
        node_feature_dim=int(data.x.shape[1]),
        anchors=anchors,
        out_dir=out_root,
        explainers=("ig",),
        repeats=2,
        seed=42,
        checkpoint_sha="abc",
        checkpoint_period=1,
        config_sha="cfg",
        device="cpu",
    )
    assert repeat["entries"] == 2
    repeat_index = Path(repeat["index_csv"])
    rows = list(csv.DictReader(repeat_index.open("r", encoding="utf-8")))
    assert len(rows) == 2
    assert all(float(row["runtime_s"]) >= 0.0 for row in rows)
    assert all(row["fidelity_sufficiency"] != "" for row in rows)
    assert all(row["fidelity_comprehensiveness"] != "" for row in rows)
    # The S8 loader can read the operational cache index directly.
    _, meta = load_explanations_from_cache(out_root / "explainer_repeat", explainer="ig", k=5)
    assert meta

    null = produce_model_null(
        graph=graph,
        node_feature_dim=int(graph.x.shape[1]),
        anchors=anchors,
        payload={"model_config": {"hidden_dim": 16, "num_classes": int(data.y.max()) + 1}},
        out_dir=out_root,
        seed=42,
        null_reseeds=2,
        null_epochs=1,
        explainers=("ig",),
        checkpoint_period=1,
        config_sha="cfg",
        device="cpu",
    )
    assert null["entries"] == 2
    assert len(null["null_checkpoints"]) == 2
    null_rows = list(csv.DictReader(Path(null["index_csv"]).open("r", encoding="utf-8")))
    assert len(null_rows) == 2

    combined = combine_null_channels(out_root)
    assert combined["status"] == "ok"
    assert combined["rows"] == 4
    assert Path(out_root / "combined_null.csv").is_file()


def test_producer_dry_run_plan(tmp_path, repo_root):
    from scripts.build_noise_floor import main

    exit_code = main(
        [
            "--dataset",
            "NF-UNSW-NB15-v3",
            "--strategy",
            "ft",
            "--period",
            "1",
            "--anchors-dir",
            str(tmp_path / "anchors"),
            "--graphs-dir",
            str(tmp_path / "graphs"),
            "--checkpoints-root",
            str(tmp_path / "runs"),
            "--out-root",
            str(tmp_path / "noise"),
            "--lane",
            "legacy",
            "--dry-run",
        ]
    )
    assert exit_code == 0


def test_canonical_noise_train_settings_uses_train_config_recipe():
    """T40/F2 re-audit: noise null reseeds use configs/train.yaml (graph_batch/30/5), not drift defaults."""
    from scripts.build_noise_floor import canonical_noise_train_settings
    from src.utils.config import load_config

    train_config = load_config("configs/train.yaml").to_dict()
    settings = canonical_noise_train_settings(train_config)
    assert str(settings.batching) == "graph_batch"
    train_section = train_config.get("train") or {}
    assert int(settings.max_epochs) == int(train_section.get("max_epochs", 30)) == 30
    assert int(settings.patience) == int(train_section.get("patience", 5)) == 5

    capped = canonical_noise_train_settings(train_config, null_epochs=10)
    assert int(capped.max_epochs) == 10
    assert str(capped.batching) == "graph_batch"

    with pytest.raises(ValueError):
        canonical_noise_train_settings({})


def test_required_index_fields_are_enforced(tmp_path):
    from src.explain.cache import ExplanationCache, cache_key

    cache = ExplanationCache(tmp_path / "cache")
    key = cache_key(checkpoint_sha="a", anchor_id=1, explainer="ig", seed=1, k=5, config_sha="c")
    with pytest.raises(ValueError):
        cache.put(
            key,
            arrays={"attributions": torch.zeros(3).numpy()},
            metadata={"explainer": "ig"},
            index_row={"checkpoint_sha": "a", "anchor_id": 1, "explainer": "ig", "seed": 1, "k": 5, "config_sha": "c"},
        )
