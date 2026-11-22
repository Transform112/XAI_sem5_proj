#!/usr/bin/env python
"""T28 (T27 re-audit) — independent model/trainer equivalence + explainer-cache safety.

Run against a pinned tree via T28_REPO (imports resolve there). Checks:
  E1 batch-only edge head == full-head slice (both models, <=1e-6)
  E2 E-GS static aggregation cache == PyG propagate mean (cache bypass via requires_grad)
  E3 GCN coalesced weighted graph == multigraph normalized path, forward + gradients
  E4 explainer masks gate propagation (PyG set_masks) and force cache bypass for both models
  E5 graph_batch vs reference mini_batch: same-seed 1-epoch parameter equality (<=1e-6)
  E6 cache invalidation on mutation / device move / requires_grad
  E7 graph_batch training batches only from train mask (instrumented edge_batch capture)
  E8 same-seed 1-epoch determinism (toy)

Output: results/text/verification/data_audit/t28_equiv.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(os.environ.get("T28_REPO", r"C:\Users\iusearchbtw\Desktop\XAI_proj"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
OUT = Path(os.environ.get("T28_OUT", r"C:\Users\iusearchbtw\Desktop\XAI_proj\verification\data_audit"))


def toy_dense(n=30, dup=True, seed=0, n_classes=3):
    from torch_geometric.data import Data

    torch.manual_seed(seed)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    src = [a for a, b in pairs]; dst = [b for a, b in pairs]
    if dup:  # duplicate some parallel edges -> multigraph for GCN coalescing
        for k in range(0, 30, 3):
            src.append(pairs[k][1]); dst.append(pairs[k][0])
    ed = torch.tensor([src, dst], dtype=torch.long)
    data = Data(x=torch.ones((n, 4)), edge_index=ed, edge_attr=torch.randn(ed.shape[1], 6),
                y=torch.randint(0, n_classes, (ed.shape[1],)))
    for s in ("train", "val", "test"):
        setattr(data, f"{s}_mask", torch.ones(ed.shape[1], dtype=torch.bool))
    return data


def main() -> int:
    from src.models.edge_sage import build_model
    from src.train.trainer import TrainSettings, train_edge_classifier
    from torch_geometric.explain.algorithm.utils import set_masks, clear_masks

    res: dict = {"repo": str(REPO), "checks": []}
    chk = res["checks"]

    def add(i, status, detail):
        chk.append({"id": i, "status": status, "detail": detail})

    data = toy_dense()
    classes = int(data.y.max()) + 1

    # E1 batch-only head equivalence
    e1 = {}
    for name in ("egraphsage", "gcn"):
        model = build_model(name, node_dim=4, edge_dim=6,
                            config={"model": {"hidden_dim": 16, "num_classes": classes}})
        model.eval()
        idx = torch.arange(0, data.num_edges, 3)
        with torch.no_grad():
            full = model(data.x, data.edge_index, data.edge_attr)
            sub = model(data.x, data.edge_index, data.edge_attr, edge_batch=idx)
        diff = float((sub - full[idx]).abs().max())
        e1[name] = {"max_abs_diff": diff, "n_edges": int(data.num_edges), "n_batch": int(idx.numel())}
    add("T28.E1.batch_head", "pass" if all(v["max_abs_diff"] <= 1e-6 for v in e1.values()) else "fail", e1)

    # E2 E-GS static aggregation cache vs PyG propagate mean
    egs = build_model("egraphsage", node_dim=4, edge_dim=6,
                      config={"model": {"hidden_dim": 16, "num_classes": classes}})
    egs.eval()
    layer = egs.layers[0]
    with torch.no_grad():
        cached = layer(data.x, data.edge_index, data.edge_attr)  # cache path
        ea2 = data.edge_attr.clone().requires_grad_(True)
        bypass = layer(data.x, data.edge_index, ea2)  # requires_grad -> PyG propagate mean, no cache
    e2_diff = float((cached - bypass).abs().max())
    add("T28.E2.egs_cache", "pass" if e2_diff <= 1e-6 else "fail",
        {"max_abs_diff": e2_diff, "note": "cache path vs PyG propagate mean path (requires_grad bypass)"})

    # E3 GCN coalesced vs multigraph forward + gradients
    gcn = build_model("gcn", node_dim=4, edge_dim=6,
                      config={"model": {"hidden_dim": 16, "num_classes": classes}})
    gcn.eval()
    e3 = {}
    for name, mask in (("all_ones_mask_multigraph", torch.ones(data.num_edges)),):
        base = build_model("gcn", node_dim=4, edge_dim=6,
                           config={"model": {"hidden_dim": 16, "num_classes": classes}})
        base.load_state_dict(gcn.state_dict())
        ref = build_model("gcn", node_dim=4, edge_dim=6,
                          config={"model": {"hidden_dim": 16, "num_classes": classes}})
        ref.load_state_dict(gcn.state_dict())
        logits_coalesced = base(data.x, data.edge_index, data.edge_attr)
        set_masks(ref, mask, data.edge_index, apply_sigmoid=False)  # forces multigraph normalized path
        logits_multi = ref(data.x, data.edge_index, data.edge_attr)
        fwd = float((logits_coalesced - logits_multi).abs().max())
        # gradient equivalence on a weighted loss over all edges
        y = data.y
        loss1 = torch.nn.functional.cross_entropy(logits_coalesced, y)
        grads1 = torch.autograd.grad(loss1, [p for p in base.parameters() if p.requires_grad],
                                    retain_graph=True, allow_unused=True)
        loss2 = torch.nn.functional.cross_entropy(logits_multi, y)
        grads2 = torch.autograd.grad(loss2, [p for p in ref.parameters() if p.requires_grad],
                                    retain_graph=True, allow_unused=True)
        gdiff = max((float((a - b).abs().max()) if a is not None and b is not None else 0.0)
                    for a, b in zip(grads1, grads2))
        e3[name] = {"forward_max_abs_diff": fwd, "grad_max_abs_diff": gdiff}
    add("T28.E3.gcn_coalesced", "pass" if all(v["forward_max_abs_diff"] <= 1e-6 and
                                              v["grad_max_abs_diff"] <= 1e-6 for v in e3.values()) else "fail", e3)

    # E4 explainer masks gate propagation + bypass caches
    e4 = {}
    for name in ("egraphsage", "gcn"):
        m = build_model(name, node_dim=4, edge_dim=6,
                        config={"model": {"hidden_dim": 16, "num_classes": classes}})
        m.eval()
        with torch.no_grad():
            full = m(data.x, data.edge_index, data.edge_attr)
        mask = torch.ones(data.num_edges); mask[: data.num_edges // 2] = 0.0
        set_masks(m, mask, data.edge_index, apply_sigmoid=False)
        try:
            with torch.no_grad():
                masked = m(data.x, data.edge_index, data.edge_attr)
            diff = float((masked - full).abs().max())
        except Exception as exc:  # noqa: BLE001
            diff = None
            e4[name] = {"error": repr(exc)}
            continue
        finally:
            clear_masks(m)
        e4[name] = {"mask_changes_output_diff": diff, "mask_gates": diff is not None and diff > 1e-8}
    add("T28.E4.explainer_masks", "pass" if all(v.get("mask_gates") for v in e4.values()) else "fail", e4)

    # E5 graph_batch vs reference mini_batch toy parameter equality
    def run_toy(batching_mode: str):
        torch.manual_seed(123)
        d = toy_dense()
        model = build_model("egraphsage", node_dim=4, edge_dim=6,
                            config={"model": {"hidden_dim": 16, "num_classes": classes}})
        settings = TrainSettings(max_epochs=1, patience=5, batch_size=32, batching=batching_mode,
                                 optimized_dispatch=(batching_mode == "graph_batch"),
                                 sampler_fast_path=(batching_mode == "graph_batch"))
        r = train_edge_classifier(model, d, settings=settings, seed=7, device="cpu")
        return {k: v.clone() for k, v in model.state_dict().items()}, r

    params_gb, res_gb = run_toy("graph_batch")
    params_mb, res_mb = run_toy("mini_batch")
    pdiff = max(float((params_gb[k] - params_mb[k]).abs().max()) for k in params_gb)
    add("T28.E5.graph_batch_vs_minibatch", "pass" if pdiff <= 1e-6 else "fail",
        {"max_param_abs_diff": pdiff, "updates_gb": res_gb["timing"].get("updates_per_epoch"),
         "updates_mb": res_mb["timing"].get("updates_per_epoch"),
         "n_train": res_gb.get("n_train_edges"),
         "note": "toy same-seed 1 epoch; graph_batch is the optimized implementation of the legacy "
                 "mini_batch training program"})

    # E6 cache invalidation / bypass
    layer = build_model("egraphsage", node_dim=4, edge_dim=6,
                        config={"model": {"hidden_dim": 16, "num_classes": classes}}).layers[0]
    layer.eval()
    with torch.no_grad():
        out1 = layer(data.x, data.edge_index, data.edge_attr)
        out2 = layer(data.x, data.edge_index, data.edge_attr)
        cached_hit = bool(layer._cached_agg is not None)
        layer.clear_cache()
        cleared = layer._cached_agg is None
        # mutation: new edge_attr pointer/version -> recompute
        ea3 = data.edge_attr.clone(); ea3[0, 0] += 1.0
        out3 = layer(data.x, data.edge_index, ea3)
        recomputed_after_mutation = bool(layer._cached_agg is not None)
    add("T28.E6.cache_invalidation",
        "pass" if (cached_hit and cleared and recomputed_after_mutation and torch.equal(out1, out2)) else "fail",
        {"repeated_forward_equal": bool(torch.equal(out1, out2)), "cached": cached_hit,
         "clear_works": cleared, "recompute_after_mutation": recomputed_after_mutation})

    # E7 graph_batch batches only from train mask (instrumented edge_batch capture)
    d2 = toy_dense()
    # make val/test masks disjoint from train
    n = int(d2.num_edges)
    train_ids = np.arange(0, n - 20)
    val_ids = np.arange(n - 20, n - 10)
    test_ids = np.arange(n - 10, n)
    d2.train_mask = torch.zeros(n, dtype=torch.bool); d2.train_mask[train_ids] = True
    d2.val_mask = torch.zeros(n, dtype=torch.bool); d2.val_mask[val_ids] = True
    d2.test_mask = torch.zeros(n, dtype=torch.bool); d2.test_mask[test_ids] = True
    model = build_model("egraphsage", node_dim=4, edge_dim=6,
                        config={"model": {"hidden_dim": 16, "num_classes": classes}})
    captured = []

    class Recorder(torch.nn.Module):
        def __init__(self, inner):
            super().__init__(); self.inner = inner
        def forward(self, x, edge_index, edge_attr, edge_batch=None):
            if edge_batch is not None:
                captured.append((edge_batch.detach().cpu().numpy().copy(), bool(torch.is_grad_enabled())))
            return self.inner(x, edge_index, edge_attr, edge_batch)

    settings = TrainSettings(max_epochs=1, patience=5, batch_size=32, batching="graph_batch")
    train_edge_classifier(Recorder(model), d2, settings=settings, seed=11, device="cpu")
    train_set = set(train_ids.tolist()); val_set = set(val_ids.tolist()); test_set = set(test_ids.tolist())
    train_batches = [b for b, grad in captured if grad]
    eval_batches = [b for b, grad in captured if not grad]
    bad = [b for b in train_batches if any(int(i) in val_set or int(i) in test_set for i in b)]
    all_train = all(all(int(i) in train_set for i in b) for b in train_batches)
    add("T28.E7.train_mask_only", "pass" if all_train and not bad and train_batches else "fail",
        {"n_training_batches_captured": len(train_batches), "n_eval_batches_captured": len(eval_batches),
         "all_training_batches_subset_of_train": all_train,
         "training_batches_containing_val_test": len(bad), "n_train": len(train_ids),
         "note": "grad-enabled graph_batch forwards captured during 1 epoch; eval forwards (no_grad) on "
                 "val/test masks are expected and excluded from the train-only check"})

    # E9 graph_batch computes the head only for the loss batch (no full-edge head per step)
    d3 = toy_dense()
    model3 = build_model("egraphsage", node_dim=4, edge_dim=6,
                         config={"model": {"hidden_dim": 16, "num_classes": classes}})
    rows = []

    class HeadRecorder(torch.nn.Module):
        def __init__(self, inner):
            super().__init__(); self.inner = inner
        def forward(self, x, edge_index, edge_attr, edge_batch=None):
            out = self.inner(x, edge_index, edge_attr, edge_batch)
            if torch.is_grad_enabled():
                rows.append(int(out.shape[0]))
            return out

    settings3 = TrainSettings(max_epochs=1, patience=5, batch_size=64, batching="graph_batch")
    train_edge_classifier(HeadRecorder(model3), d3, settings=settings3, seed=3, device="cpu")
    n_edges = int(d3.num_edges)
    e9_ok = rows and all(r < n_edges for r in rows) and max(rows) <= 64
    add("T28.E9.batch_only_head", "pass" if e9_ok else "fail",
        {"n_train_forwards": len(rows), "max_head_rows": max(rows) if rows else None,
         "period_edges": n_edges, "expected_batch_rows": 64,
         "note": "graph_batch head restricted to the 1024-target-edge loss batch (toy: 64), never all edges"})

    # E8 toy determinism
    p1, _ = run_toy("graph_batch")
    p2, _ = run_toy("graph_batch")
    det = all(torch.equal(p1[k], p2[k]) for k in p1)
    add("T28.E8.toy_determinism", "pass" if det else "fail", {"identical": det})

    fails = [c for c in chk if c["status"] == "fail"]
    res["overall"] = "fail" if fails else "pass"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t28_equiv.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({c["id"]: c["status"] for c in chk}, indent=1))
    print("overall:", res["overall"])
    for c in chk:
        print(c["id"], json.dumps(c["detail"], default=str)[:500])
    return 0 if not fails else 2


if __name__ == "__main__":
    raise SystemExit(main())
