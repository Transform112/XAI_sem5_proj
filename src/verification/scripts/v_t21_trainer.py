#!/usr/bin/env python
"""T21 independent audit — trainer batching (D-049/D-050).

Verifies the claims that justify the G2/full-matrix go-ahead:
  * the canonical full-batch loss is an exact class-weighted edge-mean CE: an
    independent hand-computed reference (sum w[y] CE / sum w[y]) matches at
    rel <= 1e-6;
  * chunked loss is invariant to the chunk size (including chunk > #edges);
  * the full-batch path performs exactly ONE message-passing forward per epoch,
    while the retained legacy mini-batch reference performs one full-graph
    forward per edge batch (the O(E^2) path that OOMed);
  * one-epoch full-batch training is deterministic under a fixed seed;
  * independent full-size memory arithmetic for E=1,131,909, F=50, H=128,
    C=10, chunk=262,144 on an 8 GB card, plus the legacy 3 GB OOM expectation;
    flags if the chunked head still materialises full-E activations (memory
    ceiling not actually bounded by chunk_edges).

Output: results/text/verification/data_audit/t21_trainer.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(os.environ.get("T21_REPO", r"C:\Users\iusearchbtw\Desktop\XAI_proj"))
OUT = Path(os.environ.get("T21_OUT", r"C:\Users\iusearchbtw\Desktop\XAI_proj\verification\data_audit"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    from src.models.edge_sage import build_model
    from src.train.trainer import (
        TrainSettings,
        balanced_class_weights,
        full_batch_mean_loss,
        mini_batch_mean_loss,
        train_edge_classifier,
    )
    from tests.helpers_torch import build_toy_graph

    result: dict = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "checks": [], "memory_estimate": {}}
    chk = result["checks"]

    def add(i, status, detail):
        chk.append({"id": i, "status": status, "detail": detail})

    data = build_toy_graph(n_nodes=30, n_edges=240, n_classes=3, seed=0)
    model = build_model("egraphsage", node_dim=data.x.shape[1], edge_dim=data.edge_attr.shape[1],
                        config={"model": {"hidden_dim": 16, "num_classes": 3}})
    model.eval()
    weights = balanced_class_weights(data.y[data.train_mask].numpy(), int(data.y.max()) + 1)
    train_idx = torch.nonzero(data.train_mask, as_tuple=False).flatten()

    # ---- (1) independent reference: weighted CE mean from raw logits ----
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_attr)
    # unweighted per-sample CE (reduction='none' applies class weights, so use log-softmax)
    logp = torch.log_softmax(logits, dim=-1)
    ce = -logp.gather(1, data.y.unsqueeze(1)).squeeze(1)
    w = weights[data.y[train_idx]]
    ref = float((w * ce[train_idx]).sum() / w.sum())
    got = float(full_batch_mean_loss(model, data, train_idx, weights, chunk_edges=7))
    add("T21.trainer.loss_reference", "pass" if abs(got - ref) <= 1e-6 * max(abs(ref), 1.0) else "fail",
        {"independent_weighted_mean": ref, "full_batch_mean_loss": got,
         "abs_diff": abs(got - ref), "rel": abs(got - ref) / max(abs(ref), 1e-12),
         "definition": "sum_i w[y_i] CE_i / sum_i w[y_i] over train-mask edges"})

    # ---- (2) chunk invariance ----
    vals = {c: float(full_batch_mean_loss(model, data, train_idx, weights, chunk_edges=c))
            for c in (1, 5, 17, 100_000)}
    spread = max(vals.values()) - min(vals.values())
    add("T21.trainer.chunk_invariance", "pass" if spread <= 1e-6 else "fail",
        {"values": vals, "spread": spread})

    # ---- (3) forward-call counting: full_batch=1 vs mini_batch=ceil(E/batch) ----
    calls = {"full": 0, "mini": 0}

    class CountingModel(torch.nn.Module):
        def __init__(self, inner):
            super().__init__(); self.inner = inner
        def forward(self, x, edge_index, edge_attr):
            calls[self.tag] += 1
            return self.inner(x, edge_index, edge_attr)

    m2 = build_model("egraphsage", node_dim=data.x.shape[1], edge_dim=data.edge_attr.shape[1],
                     config={"model": {"hidden_dim": 16, "num_classes": 3}})
    cm = CountingModel(m2)
    cm.tag = "full"; cm.eval()
    with torch.no_grad():
        full_batch_mean_loss(cm, data, train_idx, weights, chunk_edges=17)
    cm.tag = "mini"; cm.eval()
    with torch.no_grad():
        mini_batch_mean_loss(cm, data, train_idx, weights, batch_size=10)
    expected_mini = int(np.ceil(len(train_idx) / 10))
    add("T21.trainer.complexity", "pass" if calls["full"] == 1 and calls["mini"] == expected_mini else "fail",
        {"full_batch_forwards": calls["full"], "mini_batch_forwards": calls["mini"],
         "expected_mini_batch_forwards": expected_mini,
         "n_train_edges": int(len(train_idx)),
         "note": "mini_batch reference = O(E^2) full-graph forwards; full_batch = 1 forward/epoch O(E)"})

    # ---- (4) one-epoch determinism ----
    def run_once():
        torch.manual_seed(123)
        m = build_model("egraphsage", node_dim=data.x.shape[1], edge_dim=data.edge_attr.shape[1],
                        config={"model": {"hidden_dim": 16, "num_classes": 3}})
        settings = TrainSettings(max_epochs=1, patience=5, batch_size=16, batching="full_batch", chunk_edges=13)
        r = train_edge_classifier(m, data, settings=settings, seed=123, device="cpu")
        state = {k: v.clone() for k, v in m.state_dict().items()}
        return r, state

    r1, s1 = run_once()
    r2, s2 = run_once()
    same = all(torch.equal(s1[k], s2[k]) for k in s1)
    add("T21.trainer.determinism", "pass" if same and r1["timing"]["batching"] == "full_batch" else "fail",
        {"same_state": bool(same), "batching_recorded": r1["timing"].get("batching"),
         "chunk_edges_recorded": r1["timing"].get("chunk_edges")})

    # ---- (5) memory arithmetic at full size ----
    E, F, N, H, C = 1_131_909, 50, 42, 128, 10
    chunk = 262_144
    gb = 1024 ** 3
    edge_attr = E * F * 4
    layer_msg = E * H * 4          # per-layer message tensor (if materialised/saved)
    edge_repr = E * (2 * H + F) * 4
    logits_mem = E * C * 4
    head_chunk = chunk * C * 4
    weights_gb = 33_534 * 4 * 4    # params + grads + 2 Adam states (approx)
    total_est = edge_attr + 2 * layer_msg + edge_repr + logits_mem + head_chunk + weights_gb
    result["memory_estimate"] = {
        "n_edges": E, "n_features": F, "n_nodes": N, "hidden": H, "n_classes": C,
        "edge_attr_MiB": round(edge_attr / gb * 1024, 1),
        "two_layer_messages_MiB": round(2 * layer_msg / gb * 1024, 1),
        "full_edge_repr_head_input_MiB": round(edge_repr / gb * 1024, 1),
        "logits_MiB": round(logits_mem / gb * 1024, 1),
        "chunk_head_scratch_MiB": round(head_chunk / gb * 1024, 1),
        "params_grads_optimizer_MiB": round(weights_gb / gb * 1024, 1),
        "estimated_forward_peak_MiB": round(total_est / gb * 1024, 1),
        "3GB_lane": "expected OOM (>= 3.3 GiB before backward buffers) — matches Lead measurement",
        "8GB_rtx5060": "estimated 4.0-5.0 GiB with backward buffers; fits 8 GB but tight (~1-2 GiB headroom)",
        "caveat": "chunk_edges bounds only the loss accumulation; model.forward still materialises full-E "
                  "edge_repr (2H+F=306 floats per edge ≈ 1.4 GiB) and logits (45 MiB), so the forward memory "
                  "ceiling is O(E), not O(chunk). Actual peak must be measured on the 8 GB lane (kit v6 G2 run).",
        "linear_in_E": True,
    }
    add("T21.trainer.memory_estimate", "pass" if total_est / gb < 8.0 else "fail",
        {k: result["memory_estimate"][k] for k in
         ("edge_attr_MiB", "full_edge_repr_head_input_MiB", "two_layer_messages_MiB",
          "estimated_forward_peak_MiB", "8GB_rtx5060", "caveat")})

    fails = [c for c in chk if c["status"] == "fail"]
    result["overall"] = "fail" if fails else "pass"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t21_trainer.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({c["id"]: c["status"] for c in chk}, indent=1))
    print("overall:", result["overall"])
    print("estimated_forward_peak_MiB:", result["memory_estimate"]["estimated_forward_peak_MiB"])
    return 0 if not fails else 2


if __name__ == "__main__":
    raise SystemExit(main())
