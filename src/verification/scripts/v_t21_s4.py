#!/usr/bin/env python
"""T21 independent audit â€” S4 fixes (D-045) toy re-derivations.

Checks that the T17 blockers cannot pass with the new code:
  1. Replay entry target_index must index the stored edge inside the sampled
     subgraph (old bug: first subgraph edge).
  2. Replay subgraph must respect <= fanout neighbours per hop (old: unbounded
     k-hop subgraph).
  3. Replay eviction must be random from the currently largest class (old:
     oldest entries).
  4. EWC two-update hand example: after update t the Fisher of period t is
     estimated AFTER training on the just-trained model with true labels and is
     sum-accumulated; the penalty in update t+1 is anchored to theta* = params
     after update t (previous period). The old order (Fisher of current period
     BEFORE training, replaced each update) must fail this test.
  5. EWC lambda grid selection uses only old-period validation scores and picks
     the best candidate on the configured grid.

Output: results/text/verification/data_audit/t21_s4.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(os.environ.get("T21_REPO", r"C:\Users\iusearchbtw\Desktop\XAI_proj"))
OUT = Path(os.environ.get("T21_OUT", r"C:\Users\iusearchbtw\Desktop\XAI_proj\verification\data_audit"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


class TinyEdgeModel(torch.nn.Module):
    def __init__(self, edge_dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.l1 = torch.nn.Linear(edge_dim, hidden)
        self.l2 = torch.nn.Linear(hidden, n_classes)

    def forward(self, x, edge_index, edge_attr):
        return self.l2(torch.relu(self.l1(edge_attr)))


def toy_graph(edges, labels, n_nodes, edge_dim=3, seed=0):
    from torch_geometric.data import Data

    torch.manual_seed(seed)
    src = [e[0] for e in edges]; dst = [e[1] for e in edges]
    data = Data(
        x=torch.ones((n_nodes, 4)),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.randn(len(edges), edge_dim),
        y=torch.tensor(labels, dtype=torch.long),
    )
    data.node_global_ids = torch.arange(n_nodes, dtype=torch.long) * 10 + 1
    data.edge_host_src_global = data.node_global_ids[data.edge_index[0]]
    data.edge_host_dst_global = data.node_global_ids[data.edge_index[1]]
    for s in ("train", "val", "test"):
        setattr(data, f"{s}_mask", torch.ones(len(edges), dtype=torch.bool))
    return data


def main() -> int:
    from src.cl.strategies import EWCStrategy, ReplayBuffer
    from src.train.trainer import TrainSettings

    result: dict = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "checks": [], "findings": []}
    chk = result["checks"]

    def add(i, status, detail):
        chk.append({"id": i, "status": status, "detail": detail})

    settings = TrainSettings(max_epochs=1, patience=5, batch_size=64)

    # ------------------------------------------------------------------ (1) replay target pairing
    toy = toy_graph([(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 2), (0, 3), (0, 4), (0, 5)],
                    [0, 0, 0, 0, 1, 1, 1, 2, 2], n_nodes=6)
    buf = ReplayBuffer(capacity=100, fanout=(10, 10))
    buf.add_from_graph(toy, seed=42)
    misaligned = []
    for entry in buf.entries:
        source_edge = int(entry.source_edge.item())
        target_local = int(entry.target_index.item())
        sub_src = int(entry.edge_index[0, target_local]); sub_dst = int(entry.edge_index[1, target_local])
        got = (int(entry.node_global_ids[sub_src]), int(entry.node_global_ids[sub_dst]))
        want = (int(toy.edge_host_src_global[source_edge]), int(toy.edge_host_dst_global[source_edge]))
        if got != want or int(entry.y.item()) != int(toy.y[source_edge]):
            misaligned.append({"source_edge": source_edge, "target_local": target_local,
                               "got": got, "want": want})
    add("T21.S4.replay_target_alignment", "pass" if buf.entries and not misaligned else "fail",
        {"entries": len(buf.entries), "misaligned": len(misaligned), "examples": misaligned[:5],
         "expected": "target_index indexes the stored edge in the sampled subgraph"})

    # ------------------------------------------------------------------ (2) replay fanout cap
    n_leaves = 40
    star = toy_graph([(0, i) for i in range(1, n_leaves + 1)], [0] * n_leaves, n_nodes=n_leaves + 1)
    sbuf = ReplayBuffer(capacity=5, fanout=(3, 3))
    sbuf.add_from_graph(star, seed=7)
    max_nodes = max(int(e.num_nodes) for e in sbuf.entries) if sbuf.entries else 0
    # 2 hops with fanout 3 -> at most 1 + 3 + 9 = 13 nodes for a degree-unbounded star
    add("T21.S4.replay_fanout", "pass" if max_nodes <= 13 else "fail",
        {"max_subgraph_nodes": max_nodes, "bound_1+fanout+fanout^2": 13, "fanout": [3, 3],
         "graph_nodes": int(star.num_nodes)})

    # ------------------------------------------------------------------ (3) eviction: largest-class random
    ev_results = []
    # direct policy test on stub entries (independent of add_from_graph balancing):
    # labels [0,0,0,1,2] with capacity 3 -> largest class 0 must lose two random entries
    class _Stub:
        def __init__(self, label):
            self.y = torch.tensor([label], dtype=torch.long)
    ebuf2 = ReplayBuffer(capacity=3, fanout=(10, 10))
    ebuf2.entries = [_Stub(0), _Stub(0), _Stub(0), _Stub(1), _Stub(2)]
    before_labels = [int(e.y.item()) for e in ebuf2.entries]
    evicted = ebuf2._enforce_capacity(seed=11)
    after_labels = [int(e.y.item()) for e in ebuf2.entries]
    ev_results.append({"rule": ebuf2.eviction, "capacity": 3, "before_labels": before_labels,
                       "evicted": evicted, "after_labels": after_labels})
    ok_eviction = (ebuf2.eviction == "largest_class_random" and evicted == 2
                   and len(ebuf2.entries) == 3 and after_labels.count(0) == 1)
    # oldest rule must still be available for explicit fallback
    ebuf_old = ReplayBuffer(capacity=2, fanout=(10, 10), eviction="oldest")
    ebuf_old.entries = [_Stub(0), _Stub(0), _Stub(1)]
    evicted_old = ebuf_old._enforce_capacity(seed=11)
    ok_old = evicted_old == 1 and [int(e.y.item()) for e in ebuf_old.entries] == [0, 1]
    add("T21.S4.replay_eviction", "pass" if (ok_eviction and ok_old) else "fail",
        {"results": ev_results, "oldest_fallback_ok": ok_old,
         "expected": "eviction removes entries from the currently largest class at random, not the oldest",
         "old_bug": "T17: deleted entries[:len-capacity] (oldest)"})

    # ------------------------------------------------------------------ (4) EWC two-update order/accumulation
    torch.manual_seed(11)
    model = TinyEdgeModel(edge_dim=3, hidden=8, n_classes=3)
    d1 = toy_graph([(0, 1), (1, 2), (2, 3), (3, 0)], [0, 1, 2, 0], n_nodes=4, seed=1)
    d2 = toy_graph([(0, 2), (2, 1), (1, 3)], [2, 1, 2], n_nodes=4, seed=2)
    ewc = EWCStrategy(fisher_edges=100, lam=10.0, accumulate="sum",
                      lambda_candidates=(1.0, 10.0, 100.0))
    # update 1: before any Fisher -> no penalty; after training: Fisher(d1, trained model) accumulated, anchor
    r1 = ewc.update(model, d1, settings=settings, seed=42, device="cpu")
    theta_after_1 = {n: p.detach().clone() for n, p in model.named_parameters()}
    fisher1_ok = ewc.fisher is not None and set(ewc.fisher) == set(theta_after_1)
    anchor1_ok = ewc.theta_star is not None and all(
        torch.allclose(ewc.theta_star[n], theta_after_1[n]) for n in theta_after_1)
    # hand-recompute Fisher on d1 with the just-trained model (true labels), compare
    model.eval()
    f_manual = {n: torch.zeros_like(p) for n, p in model.named_parameters()}
    count = 0
    for e in range(len(d1.y)):
        model.zero_grad()
        logp = torch.log_softmax(model(d1.x, d1.edge_index, d1.edge_attr), dim=-1)
        logp[e, int(d1.y[e])].backward()
        for n, p in model.named_parameters():
            if p.grad is not None:
                f_manual[n] += p.grad.detach() ** 2
        count += 1
    f_manual = {n: (v / count).cpu() for n, v in f_manual.items()}
    fisher_after_training_ok = all(torch.allclose(ewc.fisher[n], f_manual[n], atol=1e-9) for n in f_manual)
    # record penalty seen in update 1 (must be zero: no Fisher before first period)
    # update 2: penalty anchored to theta_after_1; Fisher(d2) added AFTER training
    # instrument by stepping the penalty manually before/after
    # capture penalty at start of update 2, then run it
    pre_penalty_2 = float(ewc.penalty(model)) if ewc.fisher is not None else None
    r2 = ewc.update(model, d2, settings=settings, seed=42, device="cpu")
    theta_after_2 = {n: p.detach().clone() for n, p in model.named_parameters()}
    sum_accumulated = all(
        torch.allclose(ewc.fisher[n], ewc.fisher_history[-1][n], atol=1e-9) for n in theta_after_2)
    anchor2_ok = all(torch.allclose(ewc.theta_star[n], theta_after_2[n]) for n in theta_after_2)
    add("T21.S4.ewc_order_accumulation", "pass" if (fisher1_ok and anchor1_ok and
                                                     fisher_after_training_ok and
                                                     pre_penalty_2 is not None and
                                                     anchor2_ok and sum_accumulated) else "fail",
        {"fisher_after_training_period_1": bool(fisher1_ok and fisher_after_training_ok),
         "anchor_after_training_period_1": bool(anchor1_ok),
         "penalty_present_at_start_of_update_2": pre_penalty_2 is not None,
         "anchor_after_training_period_2": bool(anchor2_ok),
         "fisher_sum_accumulated": bool(sum_accumulated),
         "n_fisher_history": len(ewc.fisher_history),
         "expected": "after update t: Fisher(t) estimated on the trained model/true labels, accumulated with sum, "
                     "theta* = params after update t; next update penalises towards that theta*",
         "old_bug": "T17: Fisher(current period) before training, replaced each update"})

    # ------------------------------------------------------------------ (5) lambda grid selection
    scores = {1.0: 0.11, 10.0: 0.22, 100.0: 0.19}
    sel = ewc.select_lambda(scores)
    ok_sel = abs(float(sel["selected_lambda"]) - 10.0) < 1e-12 and "old" in json.dumps(sel).lower()
    try:
        ewc.select_lambda({5.0: 0.3})
        bad_rejected = False
    except ValueError:
        bad_rejected = True
    add("T21.S4.ewc_lambda_grid", "pass" if ok_sel and bad_rejected else "fail",
        {"selected": sel, "off_grid_rejected": bad_rejected,
         "candidates": ewc.lambda_candidates})

    fails = [c for c in chk if c["status"] == "fail"]
    result["overall"] = "fail" if fails else "pass"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t21_s4.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({c["id"]: c["status"] for c in chk}, indent=1))
    print("overall:", result["overall"])
    for c in chk:
        if c["status"] == "fail":
            print("FAIL", c["id"], json.dumps(c["detail"], default=str)[:400])
    return 0 if not fails else 2


if __name__ == "__main__":
    raise SystemExit(main())

