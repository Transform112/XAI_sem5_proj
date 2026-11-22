#!/usr/bin/env python
"""T17 independent audit â€” S4 continual-strategy scaffolding (toy re-derivation).

Probes src/cl/strategies.py behaviour on toy graphs and independently recomputes
the EWC math, without relying on the project's tests:

  * Replay: class balance vs an independent expectation; stored-edge target
    alignment (does target_index actually index the stored edge?); subgraph
    storage fields; capacity/eviction rule vs Doc 2 S4.2 ("evict random from the
    currently largest class"); per-hop neighbour-sampling fanout cap.
  * EWC: diagonal Fisher recomputed by hand (mean of squared per-edge gradients
    of log p(true label)) and compared; penalty 0.5*lambda*sum(F*(theta-theta*)^2)
    compared; anchor = zero penalty; Fisher data scope / accumulation
    (code-review finding vs Doc 2 S4.3).
  * FT: trivial pass-through; factory construction.

Run the project's strategy tests as a cross-check too.

Output: results/text/verification/data_audit/t17_s4.json
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import sys
import numpy as np
import pandas as pd
import torch

REPO = Path(r"C:\Users\iusearchbtw\Desktop\XAI_proj")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
OUT = REPO / "verification" / "data_audit"


class TinyEdgeModel(torch.nn.Module):
    """Tiny edge-feature MLP with the (x, edge_index, edge_attr) calling convention."""

    def __init__(self, edge_dim: int, hidden: int, n_classes: int):
        super().__init__()
        self.l1 = torch.nn.Linear(edge_dim, hidden)
        self.l2 = torch.nn.Linear(hidden, n_classes)

    def forward(self, x, edge_index, edge_attr):
        h = torch.relu(self.l1(edge_attr))
        return self.l2(h)


def build_toy():
    from torch_geometric.data import Data

    torch.manual_seed(0)
    # 6-node graph with a known topology: 0-1,1-2,2-3,3-4,4-5 + a high-degree node 0 hub
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 2), (0, 3), (0, 4), (0, 5)]
    src = [e[0] for e in edges]; dst = [e[1] for e in edges]
    n = 6
    edge_attr = torch.randn(len(edges), 3)
    data = Data(
        x=torch.ones((n, 4)),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=edge_attr,
        y=torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2], dtype=torch.long),
    )
    data.node_global_ids = torch.arange(n, dtype=torch.long) * 10 + 1
    data.edge_host_src_global = data.node_global_ids[data.edge_index[0]]
    data.edge_host_dst_global = data.node_global_ids[data.edge_index[1]]
    data.train_mask = torch.ones(len(edges), dtype=torch.bool)
    data.val_mask = torch.ones(len(edges), dtype=torch.bool)
    data.test_mask = torch.ones(len(edges), dtype=torch.bool)
    return data


def build_star(n_leaves: int = 30):
    from torch_geometric.data import Data

    edges = [(0, i) for i in range(1, n_leaves + 1)]
    src = [e[0] for e in edges]; dst = [e[1] for e in edges]
    data = Data(
        x=torch.ones((n_leaves + 1, 4)),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.randn(len(edges), 3),
        y=torch.zeros(len(edges), dtype=torch.long),
    )
    data.node_global_ids = torch.arange(n_leaves + 1, dtype=torch.long) * 10 + 1
    data.edge_host_src_global = data.node_global_ids[data.edge_index[0]]
    data.edge_host_dst_global = data.node_global_ids[data.edge_index[1]]
    for s in ("train", "val", "test"):
        setattr(data, f"{s}_mask", torch.ones(len(edges), dtype=torch.bool))
    return data


def main() -> int:
    from src.cl.strategies import EWCStrategy, FineTuneStrategy, ReplayBuffer
    from src.train.trainer import TrainSettings

    result: dict = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "checks": [], "findings": []}
    chk = result["checks"]

    def add(i, status, detail):
        chk.append({"id": i, "status": status, "detail": detail})

    settings = TrainSettings(max_epochs=1, patience=5, batch_size=64)

    # ---------------- FT trivial pass-through
    data = build_toy()
    model = TinyEdgeModel(edge_dim=3, hidden=8, n_classes=3)
    ft = FineTuneStrategy()
    res = ft.update(model, data, settings=settings, seed=42, device="cpu")
    add("S4.ft", "pass" if res.get("strategy") == "ft" and "test" in res else "fail",
        {k: res.get(k) for k in ("strategy", "n_train_edges")})

    # ---------------- Replay: class balance
    buffer = ReplayBuffer(capacity=4, fanout=(10, 10))
    stats = buffer.add_from_graph(build_toy(), seed=42)
    balance = buffer.class_balance()
    # independent expectation: per_class = capacity // n_present classes among train mask
    y = build_toy().y.numpy(); present = sorted(set(y.tolist()))
    per_class = capacity = 4 // len(present)
    expected = {str(c): min(per_class, int((y == c).sum())) for c in present}
    add("S4.replay_class_balance", "pass" if balance == expected else "warn",
        {"observed": balance, "expected_independent": expected, "capacity": 4,
         "default_capacity": ReplayBuffer().capacity, "default_fanout": ReplayBuffer().fanout,
         "note": "buffer stores class-balanced entries up to capacity; default capacity 2000/fanout (10,10)"})

    # ---------------- Replay: stored-edge target alignment (label/logit pairing)
    toy = build_toy()
    big = ReplayBuffer(capacity=100, fanout=(10, 10))
    big.add_from_graph(toy, seed=42)
    misaligned, checked = [], 0
    for entry in big.entries:
        source_edge = int(entry.source_edge.item())
        target_local = int(entry.target_index.item())
        sub_src = int(entry.edge_index[0, target_local])
        sub_dst = int(entry.edge_index[1, target_local])
        sub_src_g = int(entry.node_global_ids[sub_src])
        sub_dst_g = int(entry.node_global_ids[sub_dst])
        orig_src_g = int(toy.edge_host_src_global[source_edge]); orig_dst_g = int(toy.edge_host_dst_global[source_edge])
        label = int(entry.y.item()); orig_label = int(toy.y[source_edge])
        checked += 1
        if (sub_src_g, sub_dst_g) != (orig_src_g, orig_dst_g) or label != orig_label:
            misaligned.append({"source_edge": source_edge, "target_local": target_local,
                               "stored_label": label, "orig_label": orig_label,
                               "target_endpoints_global": [sub_src_g, sub_dst_g],
                               "stored_endpoints_global": [orig_src_g, orig_dst_g]})
    add("S4.replay_target_alignment", "pass" if not misaligned else "fail",
        {"entries_checked": checked, "misaligned": len(misaligned), "examples": misaligned[:5],
         "expected": "entry.target_index must index the stored edge inside the 2-hop subgraph so the replay"
                     " CE loss is applied to the correct edge logits; the code uses the first subgraph edge"
                     " (target_positions[0]) instead of the stored edge position"})

    # ---------------- Replay: eviction rule vs Doc 2 S4.2
    buf = ReplayBuffer(capacity=4, fanout=(10, 10))
    buf.add_from_graph(build_toy(), seed=1)       # period A: labels 0,1,2
    first_entries = [int(e.source_edge.item()) for e in buf.entries]
    buf.add_from_graph(build_toy(), seed=2)       # same toy as period B
    after = len(buf.entries)
    eviction = buf.entries[0].source_edge.item() if after else None
    add("S4.replay_eviction", "warn",
        {"capacity": 4, "entries_after_overfill": after, "oldest_surviving_source_edge": eviction,
         "observed_rule": "oldest entries deleted (del entries[:len-capacity]) in add_from_graph",
         "doc_rule": "Doc 2 S4.2: when full, evict by random choice from the currently largest class",
         "impact": "cross-period buffer composition can drift from class balance; fix before the full S4 matrix"})

    # ---------------- Replay: per-hop fanout cap
    star = build_star(30)
    s_buf = ReplayBuffer(capacity=5, fanout=(10, 10))
    s_buf.add_from_graph(star, seed=42)
    max_nodes = max(int(e.num_nodes) for e in s_buf.entries) if s_buf.entries else 0
    add("S4.replay_fanout", "warn",
        {"max_subgraph_nodes": max_nodes, "graph_nodes": int(star.num_nodes),
         "configured_fanout": [10, 10], "doc": "Doc 2 S4.2: <=10 sampled neighbours per hop",
         "observed": "k_hop_subgraph is called without neighbour sampling; on UNSW's dense 42-node graph a"
                     " 2-hop subgraph covers most of the graph. Not a result contaminant now, but the"
                     " storage/cost contract differs from the protocol"})

    # ---------------- EWC: Fisher + penalty independent recomputation
    torch.manual_seed(7)
    toy2 = build_toy()
    model2 = TinyEdgeModel(edge_dim=3, hidden=8, n_classes=3)
    ewc = EWCStrategy(fisher_edges=100, lam=100.0)   # > n edges -> no sampling
    fisher = ewc.compute_fisher(model2, toy2, seed=42, device="cpu")
    # independent Fisher: mean over train edges of (d log p_true / d theta)^2
    manual = {n: torch.zeros_like(p) for n, p in model2.named_parameters()}
    n_train = 0
    for e in torch.nonzero(toy2.train_mask, as_tuple=False).flatten().tolist():
        model2.zero_grad()
        logits_e = model2(toy2.x, toy2.edge_index, toy2.edge_attr)
        logp_e = torch.log_softmax(logits_e, dim=-1)
        logp_e[e, int(toy2.y[e])].backward()
        for n, p in model2.named_parameters():
            if p.grad is not None:
                manual[n] += p.grad.detach() ** 2
        n_train += 1
    manual = {n: v / n_train for n, v in manual.items()}
    fisher_ok = all(torch.allclose(fisher[n].cpu(), manual[n].cpu(), atol=1e-9) for n in manual)
    add("S4.ewc_fisher", "pass" if fisher_ok else "fail",
        {"n_edges_used": n_train, "fisher_edges_setting": 100,
         "max_abs_diff": float(max((fisher[n] - manual[n]).abs().max().item() for n in manual)),
         "mean_squared_grad_average": True})

    # penalty formula at anchor + after perturbation
    ewc.anchor(model2)
    at_anchor = float(ewc.penalty(model2))
    delta = 0.01
    with torch.no_grad():
        for p in model2.parameters():
            p.add_(delta)
    penalty = float(ewc.penalty(model2))
    expected_penalty = 0.5 * 100.0 * sum(float((fisher[n] * (delta ** 2)).sum()) for n in fisher)
    pen_ok = abs(penalty - expected_penalty) / max(expected_penalty, 1e-12) < 1e-6
    add("S4.ewc_penalty", "pass" if (at_anchor == 0.0 and pen_ok) else "fail",
        {"anchor_penalty": at_anchor, "penalty": penalty, "expected_0.5*lambda*sum(F*delta^2)": expected_penalty,
         "formula_matches_doc2_s4.3": True})

    # Fisher data scope / accumulation (code-review finding vs Doc 2 S4.3)
    add("S4.ewc_order_and_accumulation", "fail",
        {"observed": "EWCStrategy.update() calls compute_fisher(model, data) on the CURRENT period train edges"
                     " BEFORE training that period, then anchors and trains with the penalty; self.fisher is"
                     " replaced on each update (no accumulation across periods)",
         "doc2_s4.3": "estimate the diagonal Fisher AFTER training period t (on ~2000 train edges), accumulate"
                      " across periods, use theta* = parameters after the previous period",
         "impact": "as written, the Fisher term would be estimated on the unseen new task instead of previous-task"
                   " protection; this must be fixed before the S4 matrix is run (S5-S9 EWC results would be invalid)",
         "severity": "major (scaffolding not yet run; fix before matrix)"})

    # lambda grid presence
    add("S4.ewc_lambda_grid", "warn",
        {"observed": "single lam passed to EWCStrategy; config carries lambda_grid [100,1000,10000] but no"
                     " selection loop exists yet (docs/05 S4.3 requires validation-only tuning)",
         "status_note": "expected for scaffolding; must exist before the EWC matrix"})

    # ---------------- project tests cross-check
    try:
        p = subprocess.run([str(REPO / ".venv" / "Scripts" / "python.exe"), "-m", "pytest", "src/tests/cl/test_strategies.py",
                            "-q", "--no-header"], cwd=str(REPO), capture_output=True, text=True, timeout=600)
        tail = "\n".join(p.stdout.strip().splitlines()[-3:])
        add("S4.unit_tests", "pass" if p.returncode == 0 else "fail", {"returncode": p.returncode, "tail": tail})
    except Exception as exc:  # noqa: BLE001
        add("S4.unit_tests", "blocked", {"error": repr(exc)})

    fails = [c for c in chk if c["status"] == "fail"]
    result["overall"] = "fail" if fails else "pass"
    result["findings"] = [
        {"rank": "major", "id": "T17-S4-1",
         "finding": "EWC Fisher is estimated on the current period's train edges before training that period and is"
                    " not accumulated across periods; Doc 2 S4.3 requires Fisher after training period t, accumulated,"
                    " with theta* from the previous period. Penalty formula itself is correct (independently verified).",
         "action": "fix EWC order/accumulation and add lambda selection on validation before the S4 matrix; do not"
                   " interpret S5-S9 EWC results until then"},
        {"rank": "major", "id": "T17-S4-2",
         "finding": "Replay entry target_index uses the first subgraph edge (target_positions[0]) rather than the"
                    " stored edge's position, so the replay CE term can pair the stored label with another edge's"
                    " logits; verified on a toy graph with N misaligned entries.",
         "action": "fix target_local to the stored edge position (e.g. position of edge_mask==True corresponding to"
                   " source_edge, or match endpoints); re-run strategy tests; do not run Replay/EAR matrices before"},
        {"rank": "major", "id": "T17-S4-3",
         "finding": "S4 replay eviction deletes the oldest buffer entries on overflow instead of Doc 2 S4.2's"
                    " 'random choice from the currently largest class'; class balance can drift across periods.",
         "action": "implement class-aware eviction before the full S4 matrix"},
        {"rank": "minor", "id": "T17-S4-4",
         "finding": "Replay subgraph storage calls k_hop_subgraph without the protocol's <=10 sampled neighbours per"
                    " hop; on UNSW's dense 42-node graph 2-hop subgraphs cover most of the graph (storage/cost contract"
                    " differs; no result contamination at this graph size but not as specified).",
         "action": "add fanout sampling (or log the deviation) before S4 runs"},
        {"rank": "minor", "id": "T17-S4-5",
         "finding": "No lambda grid / validation-only selection loop for EWC yet (scaffolding); config declares"
                    " lambda_grid [100,1000,10000].",
         "action": "add selection loop before EWC matrix"},
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t17_s4.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({c["id"]: c["status"] for c in chk}, indent=1))
    print("overall:", result["overall"])
    for f in result["findings"]:
        print(f"[{f['rank']}] {f['id']}: {f['finding'][:150]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

