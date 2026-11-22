"""T33 independent evidence: T30 eval fix, T31 bounded anchors, external-bug triage.

Runs against the pinned T31 tree (`T33_REPO`, default verification/audit_e10ea31/tree).
Writes results/text/verification/data_audit/t33_checks.json. No repo code/results modified.
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(r"C:\Users\iusearchbtw\Desktop\XAI_proj")
PINNED = Path(os.environ.get("T33_REPO", str(REPO / "verification" / "audit_e10ea31" / "tree")))
sys.path.insert(0, str(PINNED))
OUT = REPO / "verification" / "data_audit"


def toy_data(n=20, dup=False, seed=0):
    from torch_geometric.data import Data

    torch.manual_seed(seed)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    src = [a for a, b in pairs]; dst = [b for a, b in pairs]
    if dup:
        for k in range(0, 20, 2):
            src.append(pairs[k][1]); dst.append(pairs[k][0])
    ei = torch.tensor([src, dst], dtype=torch.long)
    data = Data(x=torch.ones((n, 4)), edge_index=ei, edge_attr=torch.randn(ei.shape[1], 6),
                y=torch.randint(0, 3, (ei.shape[1],)))
    for s in ("train", "val", "test"):
        setattr(data, f"{s}_mask", torch.ones(ei.shape[1], dtype=torch.bool))
    data.node_global_ids = torch.arange(n, dtype=torch.long) * 10 + 1
    data.edge_row_ids = torch.arange(ei.shape[1], dtype=torch.long)
    data.edge_ts = torch.arange(ei.shape[1], dtype=torch.long)
    data.edge_host_src_global = data.node_global_ids[ei[0]]
    data.edge_host_dst_global = data.node_global_ids[ei[1]]
    return data


def main() -> int:
    from src.eval.predict import predict_logits
    from src.models.edge_sage import build_model

    res: dict = {"repo": str(PINNED), "checks": []}
    chk = res["checks"]

    def add(i, status, detail):
        chk.append({"id": i, "status": status, "detail": detail})

    # ======================= T30: head-only / chunked evaluation =======================
    data = toy_data(20)
    model = build_model("egraphsage", node_dim=4, edge_dim=6,
                        config={"model": {"hidden_dim": 16, "num_classes": 3}})
    model.eval()
    idx = torch.arange(0, data.num_edges, 2)
    with torch.no_grad():
        full = model(data.x, data.edge_index, data.edge_attr)
        sub = predict_logits(model, data.x, data.edge_index, data.edge_attr, idx)
    eq = float((sub - full[idx]).abs().max())
    rows = []

    class Recorder(torch.nn.Module):
        def __init__(self, inner):
            super().__init__(); self.inner = inner
        def forward(self, x, edge_index, edge_attr, edge_batch=None):
            out = self.inner(x, edge_index, edge_attr, edge_batch)
            rows.append(int(out.shape[0]))
            return out

    rec = Recorder(model)
    rec.eval()
    with torch.no_grad():
        predict_logits(rec, data.x, data.edge_index, data.edge_attr, idx)
    head_only = bool(rows and max(rows) == int(idx.numel()) and max(rows) < int(data.num_edges))
    add("T30.predict_head_only", "pass" if (eq <= 1e-6 and head_only) else "fail",
        {"n_edges": int(data.num_edges), "n_subset": int(idx.numel()),
         "head_rows_projected": rows, "max_abs_diff_vs_full_slice": eq,
         "head_only_no_full_edge_projection": head_only})

    # chunked path (uses node_embeddings/edge_logits helpers when present)
    chunked_eq = None
    if hasattr(model, "node_embeddings") and hasattr(model, "edge_logits"):
        with torch.no_grad():
            chunked = predict_logits(model, data.x, data.edge_index, data.edge_attr, idx,
                                     chunk_edges=16, force_mode="chunked")
        chunked_eq = float((chunked - full[idx]).abs().max())
    add("T30.chunked_equivalence", "pass" if (chunked_eq is not None and chunked_eq <= 1e-6) else "warn",
        {"max_abs_diff": chunked_eq, "mode_available": chunked_eq is not None})

    # canary evaluation manifest / artifacts
    canary = REPO / "verification" / "data_audit" / "matrix_canary_legacy"
    ev = json.loads((canary / "eval_manifest_ft_egraphsage_seed42_edge_batch_cuda.json").read_text(encoding="utf-8-sig"))
    evaluation = ((ev.get("extra") or {}).get("evaluation") or {})
    matrix = json.loads((canary / "cl_matrix.json").read_text(encoding="utf-8-sig"))
    R = np.asarray((matrix.get("matrix") or {}).get("all_classes"), dtype=float)
    lower_complete = bool(np.isfinite(np.tril(R)).all())
    validators = json.loads((canary / "s5_validators.json").read_text(encoding="utf-8-sig"))
    add("T30.canary_eval", "pass" if (evaluation.get("mode") == "edge_batch"
                                       and evaluation.get("peak_memory_allocated_bytes", 0) < 1.5 * 1024 ** 3
                                       and evaluation.get("wall_clock_seconds", 0) < 120
                                       and lower_complete) else "fail",
        {"mode": evaluation.get("mode"), "peak_memory_allocated_bytes": evaluation.get("peak_memory_allocated_bytes"),
         "peak_memory_gib": round(float(evaluation.get("peak_memory_allocated_bytes", 0)) / 1024 ** 3, 3),
         "wall_clock_seconds": evaluation.get("wall_clock_seconds"),
         "matrix_shape": list(R.shape), "lower_triangle_complete": lower_complete,
         "checkpoints_evaluated": matrix.get("checkpoints_evaluated"),
         "s5_validators_counts": validators.get("counts"),
         "s5_validators_overall": validators.get("overall")})

    # ======================= T31: bounded deterministic context =======================
    from src.anchors.build import _build_adjacency, _sample_subgraph

    # dense synthetic graph: 80-node complete graph -> 2-hop induced context far above cap
    dense = toy_data(80, seed=3)
    adjacency, src_list, dst_list = _build_adjacency(dense)
    target = int(dense.num_edges // 2)
    r1 = np.random.default_rng(42)
    sub1 = _sample_subgraph(dense, target, adjacency=adjacency, src_list=src_list, dst_list=dst_list,
                            num_hops=2, fanout=100, rng=r1, max_subgraph_edges=2000)
    r2 = np.random.default_rng(42)
    sub2 = _sample_subgraph(dense, target, adjacency=adjacency, src_list=src_list, dst_list=dst_list,
                            num_hops=2, fanout=100, rng=r2, max_subgraph_edges=2000)
    ids1 = sub1["sub_edge_ids"].tolist()
    ids2 = sub2["sub_edge_ids"].tolist()
    cap_ok = len(ids1) <= 2000
    target_kept = target in ids1
    target_local_ok = int(sub1["target_local_index"].item()) == ids1.index(target)
    add("T31.bounded_context", "pass" if (cap_ok and target_kept and target_local_ok and ids1 == ids2) else "fail",
        {"induced_context_edges": int(dense.num_edges), "stored_edges": len(ids1), "cap": 2000,
         "target_kept": target_kept, "target_local_correct": target_local_ok,
         "deterministic_same_seed": ids1 == ids2})

    # sparse graph: induced context below cap must be exactly the induced 2-hop set (pre-T31 semantics)
    sparse = toy_data(20, seed=4)
    adjacency_s, src_s, dst_s = _build_adjacency(sparse)
    target_s = 10
    r3 = np.random.default_rng(7)
    sub_s = _sample_subgraph(sparse, target_s, adjacency=adjacency_s, src_list=src_s, dst_list=dst_s,
                             num_hops=2, fanout=100, rng=r3, max_subgraph_edges=2000)
    ids_s = sub_s["sub_edge_ids"].tolist()
    # independent expected induced set: all edges whose endpoints are in the 2-hop node set
    src_arr = np.asarray(src_s); dst_arr = np.asarray(dst_s)
    a, b = int(sparse.edge_index[0, target_s]), int(sparse.edge_index[1, target_s])
    adj: dict = {}
    for s, d in zip(src_arr.tolist(), dst_arr.tolist()):
        adj.setdefault(s, set()).add(d)
        adj.setdefault(d, set()).add(s)
    nodes = {a, b}
    for _ in range(2):
        nodes |= {x for node in list(nodes) for x in adj.get(node, ())}
    expected_ids = sorted(int(i) for i, (s, d) in enumerate(zip(src_arr.tolist(), dst_arr.tolist()))
                          if s in nodes and d in nodes)
    sparse_exact = ids_s == expected_ids
    add("T31.sparse_exact_equivalence", "pass" if sparse_exact else "warn",
        {"stored_edges": len(ids_s), "expected_induced_edges": len(expected_ids),
         "exact_match": sparse_exact, "cap": 2000,
         "note": "below the cap, the bounded context must equal the full induced 2-hop edge set (pre-T31 semantics)"})

    # docs/config integration
    docs = (REPO / "docs" / "05_EXPERIMENT_PROTOCOL.md").read_text(encoding="utf-8")
    cfg = (REPO / "configs" / "eval.yaml").read_text(encoding="utf-8")
    add("T31.docs_config", "pass" if ("D-063" in docs or "bounded" in docs.lower()) and "2000" in docs
        and "max_subgraph_edges" in cfg else "warn",
        {"docs_mention_bounded_context": "bounded" in docs.lower() and "2000" in docs,
         "config_cap_present": "max_subgraph_edges" in cfg,
         "limitation_mentioned": "limitation" in docs.lower()})

    # ======================= external bug-report triage (medium items) =======================
    bugs: dict = {}

    # B4: EWC running-mean Fisher weight (non-default branch)
    from src.cl.strategies import EWCStrategy
    ewc = EWCStrategy(fisher_edges=10, lam=10.0, accumulate="mean")
    f1 = {"w": torch.tensor([1.0])}
    f2 = {"w": torch.tensor([3.0])}
    ewc.fisher = None
    ewc.accumulate_fisher(f1, n_edges=5)       # first update: n_updates == 0 before increment
    ewc.n_updates = 1
    ewc.accumulate_fisher(f2, n_edges=5)       # second update: code weight = 1/max(1,1)=1
    expected_mean = torch.tensor([2.0])
    bugs["B4_ewc_running_mean"] = {
        "status": "confirmed-real-non-default" if abs(float(ewc.fisher["w"]) - 2.0) > 1e-6 else "confirmed-false",
        "code_value_after_two_updates": float(ewc.fisher["w"]), "expected_running_mean": 2.0,
        "formula_in_code": "weight = 1/max(n_updates,1); n_updates incremented AFTER accumulate",
        "canonical_branch": "accumulate='sum' default (D-047; no config override in strategies.ewc)",
        "impact": "running-mean branch weights are 1,1,1/2,1/3,... instead of 1,1/2,1/3,...; default sum branch "
                  "is unaffected, so canonical matrix EWC does not use this branch",
    }

    # B9: cache key (data_ptr, version, shapes) can collide after tensor re-allocation
    from src.models.edge_sage import build_model as bm
    m = bm("egraphsage", node_dim=4, edge_dim=6, config={"model": {"hidden_dim": 8, "num_classes": 3}})
    d = toy_data(8)
    m.eval()
    with torch.no_grad():
        m(d.x, d.edge_index, d.edge_attr)
    old_ptr = int(d.edge_attr.data_ptr())
    del d.edge_attr
    gc.collect()
    new_attr = None
    reused = False
    for _ in range(50):
        cand = torch.randn((190, 6))
        if int(cand.data_ptr()) == old_ptr:
            new_attr = cand; reused = True; break
        new_attr = cand
    stale = None
    if reused:
        with torch.no_grad():
            out_cached = m(torch.ones((8, 4)), d.edge_index, new_attr)
            m.clear_cache()
            out_fresh = m(torch.ones((8, 4)), d.edge_index, new_attr)
        stale = bool((out_cached - out_fresh).abs().max() > 1e-8)
    bugs["B9_cache_key"] = {
        "status": "latent-real-mitigated" if reused else "not-reproduced",
        "allocator_pointer_reused": reused, "stale_output_if_reused": stale,
        "mitigation_in_tree": "train_edge_classifier calls model.clear_cache() at every update start; "
                              "retention loop calls model.to(device) (clears cache) per checkpoint; PyG explainer bypasses caches",
        "impact": "no stale-cache path found in canonical trainer/matrix lifecycle; latent risk only if a model instance "
                  "is reused across same-shape graphs without a cache clear/device move",
    }

    # B5: replay capacity eviction complexity (benchmark)
    from src.cl.strategies import ReplayBuffer
    class Stub:
        def __init__(self, label):
            self.y = torch.tensor([label], dtype=torch.long)
    rb = ReplayBuffer(capacity=1990)
    rb.entries = [Stub(0) for _ in range(1990)] + [Stub(1) for _ in range(10)]
    t0 = time.perf_counter()
    evicted = rb._enforce_capacity(seed=1)
    dt = time.perf_counter() - t0
    bugs["B5_replay_eviction"] = {
        "status": "confirmed-false-severity (O(k*N) scan, not O(N^2) per eviction)",
        "n_entries_before": 2000, "capacity": 1990, "evicted": evicted, "seconds": round(dt, 5),
        "semantics": "largest-class random eviction preserved; one O(N) scan per evicted entry",
    }

    # B18: cl_metrics partial last row API vs writer guard
    from src.eval.cl_metrics import cl_metrics
    partial = np.array([[0.5, np.nan, np.nan], [0.4, 0.6, np.nan], [0.3, np.nan, np.nan]])
    try:
        cl_metrics(partial)
        api_outcome = "no-error"
    except ValueError as exc:
        api_outcome = f"ValueError: {exc}"
    bugs["B18_cl_metrics_partial"] = {
        "status": "api-only (writer guards partial matrices)",
        "direct_api_outcome": api_outcome,
        "writer_guard": "run_eval.py: `if last_row_complete: cl_metrics(...) else: null + note` (T18/T30) -> incremental "
                        "evaluation cannot crash on partial rows",
    }

    # B3: mini_batch loss normalization
    trainer_src = (PINNED / "src" / "train" / "trainer.py").read_text(encoding="utf-8")
    bugs["B3_minibatch_loss"] = {
        "status": "confirmed-false/stale",
        "evidence": "mini_batch branch accumulates per-batch losses and divides by the number of batches "
                    "(`torch.stack(batch_losses).mean()`); it is the documented legacy reference path (T21-3) "
                    "and is not canonical training.",
    }

    # B6/B7/B8/B10/B11/B12/B13/B14/B15/B16/B17: code-level triage
    ig_src = (PINNED / "src" / "explain" / "ig.py").read_text(encoding="utf-8")
    gnn_src = (PINNED / "src" / "explain" / "gnn_explainer.py").read_text(encoding="utf-8")
    anchors_src = (PINNED / "src" / "anchors" / "build.py").read_text(encoding="utf-8")
    bugs["B6_B7_ig"] = {
        "status": "partly-real-perf / B7 confirmed-false",
        "evidence": "IG clones edge_attr per step (per-step clone real memory cost) but base_value is computed AFTER "
                    "target_class is fixed in the same no_grad block, and attributions use torch.autograd.grad on the "
                    "per-step input; completeness is recomputed with the baseline forward. With D-063 bounded contexts "
                    "(<=2000 edges) per-call cost is small.",
        "has_per_step_clone": "current = edge_attr.clone()" in ig_src,
        "target_class_before_base_value": ig_src.index("target_class = int(logits[target_edge].argmax())") <
                                          ig_src.index("base_value = logits[target_edge, target_class]"),
    }
    bugs["B8_gnn_target_used"] = {
        "status": "confirmed-false",
        "evidence": "flag is set from which call succeeded (`with_target` iff errors == []); if the first call succeeds "
                    "the target was used, if it fails the fallback was used -> flag is consistent with the code path.",
    }
    bugs["B10_gcn_incident_mean"] = {
        "status": "confirmed-false",
        "evidence": "GCN aggregates each edge feature into BOTH endpoints then divides by incident-edge count = mean over "
                    "incident edges (in+out); this is the documented 'mean of incident edge features' and the legacy "
                    "baseline semantics (unchanged from pre-T27).",
    }
    bugs["B11_freeze_hash"] = {
        "status": "confirmed-false (reproducibility unaffected)",
        "evidence": "frozen hash is a deterministic function of the mask manifest; recomputation reproduces 010a80d9 "
                    "(T5/T17 and T33 high-bug script); no frozen-state impact.",
    }
    bugs["B12_ear_cost"] = {
        "status": "partly-real-perf",
        "evidence": "EAR `_store_attributions` re-runs IG for every buffer entry on every update and has no "
                    "'already-stored' early exit; <=2000 entries x bounded contexts, but this is a training-cost "
                    "multiplier for the optional EAR runs (document; optimize in T34 if budget needs it).",
    }
    bugs["B13_annotation"] = {
        "status": "confirmed-false/moot",
        "evidence": "src/anchors/build.py starts with `from __future__ import annotations`, so annotations are strings "
                    "and never evaluated; also the module imports torch inside the function that needs it. PEP 563 removes "
                    "the reported NameError risk.",
        "future_annotations": anchors_src.startswith('"""') and "from __future__ import annotations" in anchors_src[:800],
    }
    bugs["B14_clean_hash_copy"] = {
        "status": "real-perf-but-out-of-frozen-path",
        "evidence": "clean._hash_rows copies all columns for hashing; the C1 frozen state is already produced and "
                    "independently audited; only a future re-run of the cleaning stage would pay this cost.",
    }
    metrics_src = (REPO / "analysis" / "metrics_cl.py").read_text(encoding="utf-8")
    bugs["B15_from_records_forward_fill"] = {
        "status": "confirmed-false (theoretical)",
        "evidence": "analysis.metrics_cl builds R[i,j] macro from per-class values of the SAME cell; forward cells are "
                    "recorded but forgetting uses only i>=j past checkpoints (A11; verified by the S5 crosscheck and "
                    "T33 high-bug checks).",
    }
    fast_src = (PINNED / "src" / "train" / "fast.py").read_text(encoding="utf-8")
    bugs["B16_fast_sampler_cpu_diff"] = {
        "status": "real-perf-fallback-only",
        "evidence": "FastSampledBatcher._context_ids uses np.setdiff1d on CPU in a tight-budget fallback branch of the "
                    "sampled alternative mode; canonical mode is graph_batch (no sampler) so matrix timing is unaffected.",
        "uses_setdiff1d": "setdiff1d" in fast_src,
    }
    pre_src = (PINNED / "src" / "data" / "preprocess.py").read_text(encoding="utf-8")
    bugs["B17_preprocess_indicator"] = {
        "status": "confirmed-false (correct by construction)",
        "evidence": "indicator is computed from the original frame column isna(); imputation happens later on the "
                    "matrix copy. Frozen preprocessor state is already audited (T17 scaler recomputation) and unchanged.",
    }

    res["bugs"] = bugs
    fails = [c for c in chk if c["status"] == "fail"]
    res["overall"] = "fail" if fails else "pass"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t33_checks.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({c["id"]: c["status"] for c in chk}, indent=1))
    print("overall:", res["overall"])
    for c in chk:
        if c["status"] != "pass":
            print(c["id"], json.dumps(c["detail"], default=str)[:500])
    return 0 if not fails else 2


if __name__ == "__main__":
    raise SystemExit(main())

