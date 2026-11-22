"""Tests for S4 strategies (incl. T17-S4 blocker regressions)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn
from torch_geometric.data import Data

from src.tests.helpers_torch import build_toy_graph, requires_torch

pytestmark = requires_torch


def _settings(max_epochs: int = 2):
    from src.train.trainer import TrainSettings

    return TrainSettings(max_epochs=max_epochs, patience=5, batch_size=64)


# ---------------------------------------------------------------------------
# FT
# ---------------------------------------------------------------------------
def test_fine_tune_strategy_update():
    from src.cl.strategies import FineTuneStrategy
    from src.models.edge_sage import build_model

    data = build_toy_graph()
    model = build_model(
        "egraphsage",
        node_dim=data.x.shape[1],
        edge_dim=data.edge_attr.shape[1],
        config={"model": {"hidden_dim": 16, "num_classes": int(data.y.max()) + 1}},
    )
    result = FineTuneStrategy().update(model, data, settings=_settings(), seed=42, device="cpu")
    assert result["strategy"] == "ft"
    assert 0.0 <= result["test"]["macro_f1"] <= 1.0


# ---------------------------------------------------------------------------
# Replay (T17-S4-2 alignment, T17-S4-3 eviction, T17-S4-4 fanout)
# ---------------------------------------------------------------------------
def test_replay_target_alignment_uses_stored_edge():
    """T17-S4-2 regression: stored label pairs with the stored edge's logits row."""
    from src.cl.strategies import ReplayBuffer

    data = build_toy_graph(n_nodes=20, n_edges=180, n_classes=3, seed=3)
    buffer = ReplayBuffer(capacity=30, fanout=(5, 5))
    buffer.add_from_graph(data, seed=42)
    assert len(buffer) > 0
    non_first = 0
    for entry in buffer.entries:
        source = int(entry.source_edge[0])
        target_index = int(entry.target_index[0])
        assert int(entry.y.reshape(-1)[0]) == int(data.y[source])
        torch.testing.assert_close(entry.edge_attr[target_index], data.edge_attr[source])
        target_src = int(entry.node_global_ids[entry.edge_index[0, target_index]])
        target_dst = int(entry.node_global_ids[entry.edge_index[1, target_index]])
        assert target_src == int(data.edge_host_src_global[source])
        assert target_dst == int(data.edge_host_dst_global[source])
        if target_index != 0:
            non_first += 1
    # The old bug always used target_index == 0; most entries must point elsewhere.
    assert non_first > len(buffer.entries) // 4


def test_replay_per_hop_fanout_cap():
    """T17-S4-4 regression: <= fanout neighbours sampled per hop."""
    from src.cl.strategies import ReplayBuffer

    data = build_toy_graph(n_nodes=40, n_edges=400, n_classes=3, seed=1)
    buffer = ReplayBuffer(capacity=10, fanout=(3, 3), num_hops=2)
    buffer.add_from_graph(data, seed=42)
    assert len(buffer) > 0
    upper_bound = 2 + 2 * 3 + (2 * 3) * 3
    for entry in buffer.entries:
        assert entry.n_sub_nodes <= upper_bound
        assert entry.n_sub_nodes < data.num_nodes


def test_replay_eviction_largest_class_random():
    """T17-S4-3: overflow evicts randomly from the largest class, not oldest."""
    from src.cl.strategies import ReplayBuffer

    def make_entry(label: int, entry_id: int) -> Data:
        entry = Data(
            x=torch.ones((2, 2)),
            edge_index=torch.tensor([[0], [1]], dtype=torch.long),
            edge_attr=torch.zeros((1, 3)),
            y=torch.tensor([label], dtype=torch.long),
        )
        entry.source_edge = torch.tensor([entry_id], dtype=torch.long)
        return entry

    def populated() -> "ReplayBuffer":
        buffer = ReplayBuffer(capacity=4, eviction="largest_class_random")
        buffer.entries = [make_entry(0, index) for index in range(5)] + [
            make_entry(1, 100 + index) for index in range(2)
        ]
        return buffer

    buffer = populated()
    buffer._enforce_capacity(seed=42)
    assert len(buffer.entries) == 4
    balance = buffer.class_balance()
    assert balance["1"] == 2  # minority class untouched
    assert balance["0"] == 2  # all evictions came from the largest class
    first_ids = sorted(int(entry.source_edge[0]) for entry in buffer.entries)
    buffer2 = populated()
    buffer2._enforce_capacity(seed=42)
    second_ids = sorted(int(entry.source_edge[0]) for entry in buffer2.entries)
    assert first_ids == second_ids


def test_replay_strategy_update_populates_buffer_after_update():
    from src.cl.strategies import ReplayBuffer, ReplayStrategy
    from src.models.edge_sage import build_model

    data = build_toy_graph()
    model = build_model(
        "egraphsage",
        node_dim=data.x.shape[1],
        edge_dim=data.edge_attr.shape[1],
        config={"model": {"hidden_dim": 16, "num_classes": int(data.y.max()) + 1}},
    )
    strategy = ReplayStrategy(buffer=ReplayBuffer(capacity=30), alpha=1.0)
    result = strategy.update(model, data, settings=_settings(), seed=42, device="cpu")
    assert result["strategy"] == "replay"
    assert len(strategy.buffer) > 0
    assert result["buffer"]["entries"] == len(strategy.buffer)


# ---------------------------------------------------------------------------
# EWC (T17-S4-1 order/accumulation, T17-S4-5 lambda grid)
# ---------------------------------------------------------------------------
def test_ewc_penalty_formula_hand_computed():
    from src.cl.strategies import EWCStrategy

    model = nn.Linear(2, 2, bias=False)
    strategy = EWCStrategy(fisher_edges=10, lam=1000.0)
    strategy.fisher = {"weight": torch.tensor([[1.0, 2.0], [3.0, 4.0]])}
    with torch.no_grad():
        model.weight.zero_()
        model.weight.add_(0.5)
    strategy.theta_star = {"weight": torch.zeros_like(model.weight)}
    fisher = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    expected = 0.5 * 1000.0 * float((fisher * (torch.full((2, 2), 0.5) ** 2)).sum())
    assert float(strategy.penalty(model)) == pytest.approx(expected)


def test_ewc_fisher_after_update_and_accumulated(monkeypatch):
    """T17-S4-1 regression: Fisher AFTER training, accumulated with sum."""
    import src.cl.strategies as strategies_module
    from src.cl.strategies import EWCStrategy

    calls: list[str] = []
    counter = {"fisher": 0}

    def fake_train(model, data, **kwargs):
        calls.append("train")
        return {
            "best_state": {name: parameter.detach().clone() for name, parameter in model.named_parameters()},
            "val": {"macro_f1": 0.0},
            "test": {"macro_f1": 0.0},
            "timing": {},
        }

    def fake_estimate(self, model, data, *, seed=42, device="cpu"):
        calls.append("fisher")
        counter["fisher"] += 1
        return {
            name: torch.full_like(parameter, float(counter["fisher"]))
            for name, parameter in model.named_parameters()
        }

    monkeypatch.setattr(strategies_module, "train_edge_classifier", fake_train)
    monkeypatch.setattr(EWCStrategy, "estimate_fisher", fake_estimate)
    strategy = EWCStrategy(fisher_edges=10, lam=1000.0)
    model = nn.Linear(3, 2, bias=True)
    first = strategy.update(model, object(), settings=_settings(), seed=42, device="cpu")
    assert calls == ["train", "fisher"]  # Fisher AFTER the update
    assert first["fisher"]["n_accumulated_updates"] == 1
    assert float(strategy.fisher["weight"].max()) == pytest.approx(1.0)
    second = strategy.update(model, object(), settings=_settings(), seed=42, device="cpu")
    assert calls == ["train", "fisher", "train", "fisher"]
    assert second["fisher"]["n_accumulated_updates"] == 2
    assert float(strategy.fisher["weight"].max()) == pytest.approx(3.0)  # sums 1 + 2
    assert len(strategy.fisher_history) == 2


def test_ewc_penalty_uses_previous_theta_star(monkeypatch):
    import src.cl.strategies as strategies_module
    from src.cl.strategies import EWCStrategy

    def fake_train(model, data, **kwargs):
        with torch.no_grad():
            model.weight.add_(1.0)
        return {
            "best_state": {name: p.detach().clone() for name, p in model.named_parameters()},
            "val": {"macro_f1": 0.0},
            "test": {"macro_f1": 0.0},
            "timing": {},
        }

    def fake_estimate(self, model, data, *, seed=42, device="cpu"):
        return {name: torch.ones_like(parameter) for name, parameter in model.named_parameters()}

    monkeypatch.setattr(strategies_module, "train_edge_classifier", fake_train)
    monkeypatch.setattr(EWCStrategy, "estimate_fisher", fake_estimate)
    strategy = EWCStrategy(fisher_edges=10, lam=1.0)
    model = nn.Linear(1, 1, bias=False)
    strategy.update(model, object(), settings=_settings(), seed=1, device="cpu")
    anchor_after_first = strategy.theta_star["weight"].clone()
    with torch.no_grad():
        model.weight.add_(1.0)
    assert float(strategy.penalty(model)) > 0.0
    strategy.update(model, object(), settings=_settings(), seed=1, device="cpu")
    assert not torch.equal(anchor_after_first, strategy.theta_star["weight"])
    assert strategy.n_updates == 2


def test_ewc_lambda_selection_on_old_period_validation():
    from src.cl.strategies import EWCStrategy

    strategy = EWCStrategy(lambda_candidates=(100.0, 1000.0, 10000.0))
    result = strategy.select_lambda({100.0: 0.40, 1000.0: 0.62, 10000.0: 0.58})
    assert result["selected_lambda"] == 1000.0
    assert strategy.lam == 1000.0
    assert result["selection_metric"].startswith("validation macro-F1 of old periods")
    with pytest.raises(ValueError):
        strategy.select_lambda({5.0: 0.9})
    scores = EWCStrategy.lambda_selection_scores(
        lambda lam: {100.0: 0.1, 1000.0: 0.9, 10000.0: 0.2}[lam],
        lambda_candidates=(100.0, 1000.0, 10000.0),
    )
    assert scores == {100.0: 0.1, 1000.0: 0.9, 10000.0: 0.2}


def test_ewc_estimate_fisher_returns_mean_squared_grads():
    from src.cl.strategies import EWCStrategy
    from src.models.edge_sage import build_model

    data = build_toy_graph()
    model = build_model(
        "egraphsage",
        node_dim=data.x.shape[1],
        edge_dim=data.edge_attr.shape[1],
        config={"model": {"hidden_dim": 16, "num_classes": int(data.y.max()) + 1}},
    )
    strategy = EWCStrategy(fisher_edges=8)
    fisher = strategy.estimate_fisher(model, data, seed=42)
    assert fisher and all(torch.isfinite(value).all() for value in fisher.values())
    assert all(float(value.mean()) >= 0.0 for value in fisher.values())


def test_build_strategy_factory():
    from src.cl.strategies import build_strategy

    assert build_strategy("ft").name == "ft"
    assert build_strategy("replay", config={"replay": {"buffer_edges": 10}}).name == "replay"
    assert build_strategy("ewc", config={"ewc": {"fisher_edges": 10, "lambda": 100.0}}).name == "ewc"
    with pytest.raises(KeyError):
        build_strategy("nope")


# ---------------------------------------------------------------------------
# T32: vectorised replay subgraph sampling (exact equivalence + perf regression)
# ---------------------------------------------------------------------------
def _reference_replay_adjacency(data):
    src = [int(v) for v in data.edge_index[0].tolist()]
    dst = [int(v) for v in data.edge_index[1].tolist()]
    adjacency = {}
    incidence = {}
    for pos, (source, target) in enumerate(zip(src, dst)):
        adjacency.setdefault(source, set()).add(target)
        adjacency.setdefault(target, set()).add(source)
        incidence.setdefault(source, []).append(pos)
        incidence.setdefault(target, []).append(pos)
    return adjacency, src, dst, incidence


def _reference_replay_subgraph(buffer, data, target_edge, *, adjacency, src_list, dst_list, incidence, rng):
    import torch

    src = int(data.edge_index[0, target_edge])
    dst = int(data.edge_index[1, target_edge])
    node_set = {src, dst}
    frontier = [src, dst]
    for hop in range(int(buffer.num_hops)):
        per_hop = buffer._fanout_for_hop(hop)
        next_frontier = []
        for node in frontier:
            candidates = [value for value in adjacency.get(node, set()) if value not in node_set]
            if len(candidates) > per_hop:
                pick = rng.choice(len(candidates), size=per_hop, replace=False)
                candidates = [candidates[int(position)] for position in pick]
            for neighbour in candidates:
                if neighbour not in node_set:
                    node_set.add(neighbour)
                    next_frontier.append(neighbour)
        frontier = next_frontier
        if not frontier:
            break
    candidate_edges = set()
    for node in node_set:
        candidate_edges.update(incidence.get(node, []))
    sub_edges = sorted(e for e in candidate_edges if src_list[e] in node_set and dst_list[e] in node_set)
    if target_edge not in sub_edges:
        sub_edges = sorted(set(sub_edges) | {target_edge})
    sub_edges_tensor = torch.as_tensor(sub_edges, dtype=torch.long)
    node_list = sorted(node_set)
    node_tensor = torch.as_tensor(node_list, dtype=torch.long)
    remap = {node: position for position, node in enumerate(node_list)}
    edge_index = data.edge_index[:, sub_edges_tensor]
    local_edge_index = torch.as_tensor(
        [[remap[int(v)] for v in edge_index[0]], [remap[int(v)] for v in edge_index[1]]], dtype=torch.long
    )
    target_local = int((sub_edges_tensor == target_edge).nonzero(as_tuple=False)[0])
    return {
        "sub_edge_index": local_edge_index,
        "sub_edge_attr": data.edge_attr[sub_edges_tensor],
        "sub_edge_ids": sub_edges_tensor,
        "sub_node_local_ids": node_tensor,
        "sub_node_global_ids": data.node_global_ids[node_tensor],
        "target_local_index": torch.tensor([target_local], dtype=torch.long),
        "target_edge": torch.tensor([target_edge], dtype=torch.long),
        "n_sub_nodes": int(node_tensor.numel()),
        "n_sub_edges": int(sub_edges_tensor.numel()),
    }


def test_replay_sampled_subgraph_matches_pre_t32_reference():
    from src.cl.strategies import ReplayBuffer

    data = build_toy_graph(n_nodes=40, n_edges=1200, n_classes=3, seed=5)
    buffer = ReplayBuffer(capacity=20, fanout=(10, 10))
    adjacency, src, dst, incidence = buffer._build_graph_index(data)
    ref_adjacency, ref_src, ref_dst, ref_incidence = _reference_replay_adjacency(data)
    for target in (0, 17, 301, 777, 1199):
        for seed in (42, 43, 7):
            new = buffer._sampled_subgraph(
                data, target, adjacency=adjacency, src_list=src, dst_list=dst,
                incidence=incidence, rng=np.random.default_rng(seed + target),
            )
            old = _reference_replay_subgraph(
                buffer, data, target, adjacency=ref_adjacency, src_list=ref_src, dst_list=ref_dst,
                incidence=ref_incidence, rng=np.random.default_rng(seed + target),
            )
            for key in ("sub_edge_index", "sub_edge_attr", "sub_edge_ids", "sub_node_local_ids",
                        "sub_node_global_ids", "target_local_index", "target_edge"):
                torch.testing.assert_close(new[key], old[key], rtol=0.0, atol=0.0)
            assert new["n_sub_nodes"] == old["n_sub_nodes"]
            assert new["n_sub_edges"] == old["n_sub_edges"]


def test_replay_dense_subgraph_capped_deterministic():
    from src.cl.strategies import ReplayBuffer

    data = build_toy_graph(n_nodes=20, n_edges=5000, n_classes=3, seed=9)
    buffer = ReplayBuffer(capacity=5, fanout=(10, 10), max_subgraph_edges=50)
    adjacency, src, dst, incidence = buffer._build_graph_index(data)
    first = buffer._sampled_subgraph(
        data, 0, adjacency=adjacency, src_list=src, dst_list=dst,
        incidence=incidence, rng=np.random.default_rng(42),
    )
    second = buffer._sampled_subgraph(
        data, 0, adjacency=adjacency, src_list=src, dst_list=dst,
        incidence=incidence, rng=np.random.default_rng(42),
    )
    assert first["n_sub_edges"] <= 50
    assert first["sub_edge_ids"][int(first["target_local_index"][0])] == 0
    for key in ("sub_edge_index", "sub_edge_attr", "sub_edge_ids", "target_local_index"):
        torch.testing.assert_close(first[key], second[key])


def test_replay_large_graph_sampling_is_vectorised_and_exact():
    import time

    from src.cl.strategies import ReplayBuffer

    rng = np.random.default_rng(11)
    n_nodes, n_edges = 40, 100_000
    src = rng.integers(0, n_nodes, n_edges)
    dst = rng.integers(0, n_nodes, n_edges)
    data = Data(
        x=torch.ones((n_nodes, 8)),
        edge_index=torch.as_tensor(np.stack([src, dst])),
        edge_attr=torch.as_tensor(rng.normal(size=(n_edges, 6)).astype("float32")),
        y=torch.as_tensor(rng.integers(0, 3, n_edges), dtype=torch.long),
    )
    data.node_global_ids = torch.arange(n_nodes, dtype=torch.long)
    buffer = ReplayBuffer(capacity=20, fanout=(10, 10), max_subgraph_edges=2000)
    adjacency, src_list, dst_list, incidence = buffer._build_graph_index(data)
    started = time.monotonic()
    for target in range(0, 20_000, 1_000):
        sub = buffer._sampled_subgraph(
            data, target, adjacency=adjacency, src_list=src_list, dst_list=dst_list,
            incidence=incidence, rng=np.random.default_rng(42 + target),
        )
        assert sub["sub_edge_ids"][int(sub["target_local_index"][0])] == target
    elapsed = time.monotonic() - started
    assert elapsed < 5.0, f"vectorised replay sampling too slow: {elapsed:.2f}s"


# ---------------------------------------------------------------------------
# T32: EWC head-only Fisher (same math as full-logit reference)
# ---------------------------------------------------------------------------
def _reference_fisher_full_logits(model, data, *, seed, fisher_edges):
    from src.utils.seed import seed_everything

    seed_everything(seed)
    model.eval()
    index = torch.nonzero(data.train_mask, as_tuple=False).flatten()
    if len(index) > fisher_edges:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(index), size=fisher_edges, replace=False)
        index = index[torch.as_tensor(np.sort(pick), dtype=torch.long)]
    fisher = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    for edge in index.tolist():
        model.zero_grad()
        logits = model(data.x, data.edge_index, data.edge_attr)[edge].unsqueeze(0)
        log_prob = torch.log_softmax(logits, dim=-1)[0, int(data.y[edge])]
        log_prob.backward()
        for name, p in model.named_parameters():
            if p.grad is not None:
                fisher[name] += p.grad.detach() ** 2
    count = max(len(index), 1)
    return {name: (value / count).detach() for name, value in fisher.items()}


@pytest.mark.parametrize("model_name", ["egraphsage", "gcn"])
def test_ewc_fisher_head_only_equals_full_logit_reference(model_name):
    from src.cl.strategies import EWCStrategy
    from src.models.edge_sage import build_model

    data = build_toy_graph(n_nodes=20, n_edges=220, n_classes=3, seed=2)
    model = build_model(
        model_name,
        node_dim=data.x.shape[1],
        edge_dim=data.edge_attr.shape[1],
        config={"model": {"hidden_dim": 16, "num_classes": 3}},
    )
    reference = _reference_fisher_full_logits(model, data, seed=42, fisher_edges=40)
    strategy = EWCStrategy(fisher_edges=40)
    head_only = strategy.estimate_fisher(model, data, seed=42, device="cpu")
    assert set(head_only) == set(reference)
    for name in reference:
        torch.testing.assert_close(head_only[name], reference[name], rtol=0.0, atol=1e-6)


def test_ewc_fisher_never_calls_full_forward():
    from src.cl.strategies import EWCStrategy
    from src.models.edge_sage import build_model

    data = build_toy_graph(n_nodes=20, n_edges=220, n_classes=3, seed=2)
    model = build_model(
        "egraphsage",
        node_dim=data.x.shape[1],
        edge_dim=data.edge_attr.shape[1],
        config={"model": {"hidden_dim": 16, "num_classes": 3}},
    )

    def _forbidden(*args, **kwargs):
        raise AssertionError("EWC Fisher must not call the full-edge forward")

    model.forward = _forbidden  # type: ignore[assignment]
    fisher = EWCStrategy(fisher_edges=32).estimate_fisher(model, data, seed=42, device="cpu")
    assert fisher and all(torch.isfinite(v).all() for v in fisher.values())


def test_ewc_running_mean_fisher_uses_after_count():
    """T34 (bug-report #4): k-th running-mean update must use weight 1/k."""
    from src.cl.strategies import EWCStrategy

    strategy = EWCStrategy(accumulate="mean")
    strategy.accumulate_fisher({"w": torch.tensor([1.0])}, n_edges=1)
    assert float(strategy.fisher["w"]) == pytest.approx(1.0)
    strategy.accumulate_fisher({"w": torch.tensor([2.0])}, n_edges=1)
    # second observation: mean(1, 2) = 1.5; the old bug produced 2.0
    assert float(strategy.fisher["w"]) == pytest.approx(1.5)
    strategy.accumulate_fisher({"w": torch.tensor([3.0])}, n_edges=1)
    assert float(strategy.fisher["w"]) == pytest.approx(2.0)
    # sum mode is unchanged (documented default)
    summed = EWCStrategy(accumulate="sum")
    summed.accumulate_fisher({"w": torch.tensor([1.0])}, n_edges=1)
    summed.accumulate_fisher({"w": torch.tensor([2.0])}, n_edges=1)
    assert float(summed.fisher["w"]) == pytest.approx(3.0)


def test_replay_eviction_matches_sequential_reference():
    """T34 (bug-report #5): fast eviction keeps the old deterministic victim sequence."""
    from types import SimpleNamespace

    from src.cl.strategies import ReplayBuffer

    entries = [SimpleNamespace(y=torch.tensor([i % 7])) for i in range(40)]
    buffer = ReplayBuffer(capacity=12)
    buffer.entries = list(entries)
    reference = list(entries)

    def _reference(buffer_entries, capacity, seed):
        evicted = 0
        rng = np.random.default_rng(seed + 1_000_003)
        while len(buffer_entries) > capacity:
            counts = {}
            for entry in buffer_entries:
                label = int(entry.y)
                counts[label] = counts.get(label, 0) + 1
            largest = max(counts, key=lambda label: (counts[label], -label))
            positions = [i for i, entry in enumerate(buffer_entries) if int(entry.y) == largest]
            victim = int(rng.choice(positions))
            buffer_entries.pop(victim)
            evicted += 1
        return evicted

    expected = _reference(reference, 12, 42)
    got = buffer._enforce_capacity(seed=42)
    assert got == expected
    assert [id(entry) for entry in buffer.entries] == [id(entry) for entry in reference]


def test_replay_eviction_is_fast_at_capacity():
    import time
    from types import SimpleNamespace

    from src.cl.strategies import ReplayBuffer

    buffer = ReplayBuffer(capacity=2000)
    buffer.entries = [SimpleNamespace(y=torch.tensor([i % 10])) for i in range(5000)]
    started = time.monotonic()
    evicted = buffer._enforce_capacity(seed=42)
    elapsed = time.monotonic() - started
    assert evicted == 3000 and len(buffer.entries) == 2000
    assert elapsed < 5.0, f"eviction too slow: {elapsed:.2f}s"


def test_batched_replay_loss_matches_per_entry_mean_sum():
    """T32: stacking replay subgraphs must equal the old sum of per-entry CE means."""
    from src.models.edge_sage import build_model
    from src.train.trainer import _replay_loss, _stack_replay_entries

    data = build_toy_graph(n_nodes=20, n_edges=180, n_classes=3, seed=8)
    model = build_model(
        "egraphsage",
        node_dim=data.x.shape[1],
        edge_dim=data.edge_attr.shape[1],
        config={"model": {"hidden_dim": 16, "num_classes": 3}},
    )
    model.eval()
    # Two tiny disjoint replay entries (Data-like objects are enough for the tuple API).
    entries = []
    for i in range(2):
        n = 6
        entries.append(
            (
                data.x[:n],
                torch.tensor([[0, 1], [2, 3]], dtype=torch.long),
                data.edge_attr[i * 4 : i * 4 + 2],
                torch.tensor([1], dtype=torch.long),
                torch.tensor([data.y[i]], dtype=torch.long),
            )
        )
    criterion = torch.nn.CrossEntropyLoss()
    alpha = 0.7
    with torch.no_grad():
        reference = None
        for entry_x, entry_edge_index, entry_edge_attr, target_index, entry_y in entries:
            logits = model(entry_x, entry_edge_index, entry_edge_attr, edge_batch=target_index)
            term = alpha * criterion(logits, entry_y)
            reference = term if reference is None else reference + term
        batched = _replay_loss(model, _stack_replay_entries(entries), criterion, alpha, supports_edge_batch=True)
    torch.testing.assert_close(batched, reference, rtol=0.0, atol=1e-6)
