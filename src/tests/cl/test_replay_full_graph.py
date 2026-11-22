"""T41/D-072 tests: full-graph-consistent edge replay baseline."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.tests.helpers_torch import build_toy_graph, requires_torch

pytestmark = requires_torch


def _data():
    data = build_toy_graph(n_nodes=20, n_edges=180, n_classes=3, seed=11)
    data.period = 0
    return data


def _entries(data, indices, *, model=None):
    from src.cl.strategies import EdgeReplayEntry

    out = []
    for edge in indices:
        out.append(
            EdgeReplayEntry(
                period=1,
                src_global=int(data.edge_host_src_global[edge]),
                dst_global=int(data.edge_host_dst_global[edge]),
                edge_attr=data.edge_attr[edge].detach().cpu().clone(),
                label=int(data.y[edge]),
                source_edge=int(edge),
            )
        )
    return out


def _model(data):
    from src.models.edge_sage import build_model

    return build_model(
        "egraphsage",
        node_dim=data.x.shape[1],
        edge_dim=data.edge_attr.shape[1],
        config={"model": {"hidden_dim": 16, "num_classes": 3}},
    )


def test_edge_replay_buffer_stores_period_global_hosts_and_features():
    from src.cl.strategies import EdgeReplayEntry, ReplayBuffer

    data = _data()
    buffer = ReplayBuffer(capacity=30, representation="edge")
    stats = buffer.add_from_graph(data, seed=42)
    assert stats["representation"] == "edge"
    assert stats["entries"] > 0 and len(buffer) == stats["entries"]
    assert all(isinstance(entry, EdgeReplayEntry) for entry in buffer.entries)
    chosen = buffer.entries[0]
    source = int(chosen.source_edge)
    assert chosen.src_global == int(data.edge_host_src_global[source])
    assert chosen.dst_global == int(data.edge_host_dst_global[source])
    assert chosen.label == int(data.y[source])
    torch.testing.assert_close(chosen.edge_attr, data.edge_attr[source])
    assert chosen.period == 1
    # class-balanced storage; eviction continues to work on edge entries
    assert set(buffer.class_balance()) == {str(i) for i in range(int(data.y.max()) + 1)}


def test_edge_replay_loss_uses_current_embeddings_old_features_and_mean_ce():
    from src.train.trainer import _edge_replay_loss

    data = _data()
    model = _model(data)
    model.eval()
    indices = torch.tensor([0, 7, 31, 88, 120], dtype=torch.long)
    entries = _entries(data, indices.tolist())
    criterion = torch.nn.CrossEntropyLoss()
    alpha = 0.7
    with torch.no_grad():
        hidden = model.node_embeddings(data.x, data.edge_index, data.edge_attr)
        # manual: current node embeddings + stored old edge features, batch head only
        src = [int(data.edge_host_src_global[e]) for e in indices.tolist()]
        dst = [int(data.edge_host_dst_global[e]) for e in indices.tolist()]
        id_to_index = {int(g): i for i, g in enumerate(data.node_global_ids.tolist())}
        edge_index = torch.tensor([[id_to_index[s] for s in src], [id_to_index[d] for d in dst]])
        edge_attr = torch.stack([data.edge_attr[e] for e in indices.tolist()], dim=0)
        targets = torch.arange(len(indices))
        manual_logits = model.edge_logits(hidden, edge_index, edge_attr, targets)
        manual = alpha * criterion(manual_logits, data.y[indices])
        by_helper = _edge_replay_loss(
            model, hidden, entries, data, criterion, alpha, torch.device("cpu")
        )
        torch.testing.assert_close(by_helper, manual, rtol=0.0, atol=1e-6)


def test_edge_replay_loss_no_full_edge_head():
    from src.train.trainer import _edge_replay_loss

    data = _data()
    model = _model(data)
    model.eval()
    entries = _entries(data, list(range(0, 120, 4)))
    seen: list[int] = []

    def hook(module, inputs):
        seen.append(int(inputs[0].shape[0]))

    handle = model.head.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            hidden = model.node_embeddings(data.x, data.edge_index, data.edge_attr)
            _edge_replay_loss(model, hidden, entries, data, torch.nn.CrossEntropyLoss(), 1.0, torch.device("cpu"))
    finally:
        handle.remove()
    assert seen and max(seen) == len(entries)
    assert max(seen) < int(data.num_edges)


def test_edge_replay_provider_is_called_once_per_optimizer_step_and_fresh(monkeypatch):
    from src.train import trainer

    data = _data()
    model = _model(data)
    calls: list[int] = []

    def provider(step):
        calls.append(int(step))
        return _entries(data, [(step * 3 + i) % 100 for i in range(8)])

    result = trainer.train_edge_classifier(
        model,
        data,
        settings=trainer.TrainSettings(
            batching="graph_batch", max_epochs=1, patience=5, batch_size=64
        ),
        seed=42,
        device="cpu",
        replay_provider=provider,
    )
    expected_steps = int(np.ceil(len(torch.nonzero(data.train_mask).flatten()) / 64))
    assert len(calls) == expected_steps
    assert calls == list(range(expected_steps))
    assert result["timing"]["n_updates"] == expected_steps


def test_edge_replay_strategy_update_runs_with_provider():
    from src.cl.strategies import ReplayBuffer, ReplayStrategy

    data = _data()
    model = _model(data)
    strategy = ReplayStrategy(
        buffer=ReplayBuffer(capacity=40, representation="edge"), replay_batch=8
    )
    result = strategy.update(
        model,
        data,
        settings=__import__("src.train.trainer", fromlist=["TrainSettings"]).TrainSettings(
            batching="graph_batch", max_epochs=1, patience=5, batch_size=64
        ),
        seed=42,
        device="cpu",
    )
    assert result["replay_representation"] == "edge"
    assert result["buffer"]["representation"] == "edge"
    assert len(strategy.buffer) > 0


def test_edge_replay_requires_graph_batch_and_head_only_models():
    from src.train import trainer

    data = _data()
    model = _model(data)
    with pytest.raises(ValueError, match="graph_batch"):
        trainer.train_edge_classifier(
            model,
            data,
            settings=trainer.TrainSettings(batching="sampled", max_epochs=1),
            seed=42,
            device="cpu",
            replay_provider=lambda step: [],
        )


def test_subgraph_replay_mode_remains_available_for_diagnostics():
    from src.cl.strategies import ReplayBuffer

    data = _data()
    buffer = ReplayBuffer(capacity=10, representation="subgraph")
    stats = buffer.add_from_graph(data, seed=42)
    assert stats["representation"] == "subgraph"
    assert not hasattr(buffer.entries[0], "src_global")


def test_edge_replay_provider_counter_does_not_reset_between_epochs():
    """D-072: fresh batch every optimizer step, including later epochs."""
    from src.train import trainer

    data = _data()
    model = _model(data)
    calls: list[int] = []

    def provider(step):
        calls.append(int(step))
        return _entries(data, [(step * 3 + i) % 100 for i in range(8)])

    trainer.train_edge_classifier(
        model,
        data,
        settings=trainer.TrainSettings(
            batching="graph_batch", max_epochs=2, patience=5, batch_size=64
        ),
        seed=42,
        device="cpu",
        replay_provider=provider,
    )
    expected_steps = int(np.ceil(len(torch.nonzero(data.train_mask).flatten()) / 64))
    assert len(calls) >= expected_steps * 2
    assert calls == list(range(len(calls)))
