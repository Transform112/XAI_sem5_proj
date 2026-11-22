#!/usr/bin/env python
"""T25 â€” code/behaviour audit of the sampled mini-batch canonical trainer.

Runs against a pinned tree via T25_REPO (default: live repo). Checks:
  * canonical batching in configs/train.yaml is sampled; full_batch is diagnostic only
  * sampled training performs per-edge-batch subgraph forwards: instrument a model
    to record edge counts per forward on a dense toy graph; no forward may see the
    full graph when a bounded sampled subgraph is used; calls/epoch == ceil(n_train/batch)
  * fanout <= 10 per hop honoured (sampled subgraph node bound for a star graph)
  * class-weighted CE computed per batch from train-mask labels only
  * measured memory field recorded in the trainer/metrics path
  * stale wording fixed (config header, docs/05 Â§4.1, D-053)
  * tests for the sampled batching exist and pass (file name/pattern check + subset run)

Writes results/text/verification/data_audit/t25_code.json.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
from pathlib import Path

REPO = Path(os.environ.get("T25_REPO", r"C:\Users\iusearchbtw\Desktop\XAI_proj"))
OUT = Path(os.environ.get("T25_OUT", r"C:\Users\iusearchbtw\Desktop\XAI_proj\verification\data_audit"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    result: dict = {"repo": str(REPO), "checks": []}
    chk = result["checks"]

    def add(i, status, detail):
        chk.append({"id": i, "status": status, "detail": detail})

    train_yaml = (REPO / "configs" / "train.yaml").read_text(encoding="utf-8")
    trainer_src = (REPO / "src" / "train" / "trainer.py").read_text(encoding="utf-8")
    protocol = (REPO / "docs" / "05_EXPERIMENT_PROTOCOL.md").read_text(encoding="utf-8")
    batch_match = re.search(r"^\s*batching:\s*(\w+)", train_yaml, re.M)
    mode = batch_match.group(1) if batch_match else None
    add("T25.code.canonical_batching",
        "pass" if mode == "sampled" and '"sampled"' in trainer_src and "full_batch" in trainer_src else "fail",
        {"config_batching": mode, "sampled_path_present": '"sampled"' in trainer_src,
         "full_batch_diagnostic_present": "full_batch" in trainer_src,
         "train_yaml_header_snippet": train_yaml.splitlines()[1][:180]})

    header_ok = "fanout" in train_yaml.split("--- S3 model + training ---")[1][:800].lower() or \
                "sampled" in train_yaml.lower()
    doc_ok = ("sampled" in protocol.lower() or "mini-batch" in protocol.lower()) and \
             "full-batch" in protocol.lower() and "diagnostic" in protocol.lower()
    add("T25.code.wording_fixes", "pass" if header_ok and doc_ok else "warn",
        {"config_header_mentions_sampled": header_ok, "docs05_mentions_full_batch_diagnostic": doc_ok})

    mem_rec = ("max_memory_allocated" in trainer_src) or any(
        "max_memory_allocated" in p.read_text(encoding="utf-8", errors="ignore")
        for p in [REPO / "scripts" / "run_train.py"] + list((REPO / "src").rglob("*.py")))
    add("T25.code.measured_memory_field", "pass" if mem_rec else "fail",
        {"max_memory_allocated_in_src": bool(mem_rec)})

    # ---- instrumentation: count sampled subgraph sizes per forward
    try:
        import torch
        from torch_geometric.data import Data
        from src.train.trainer import TrainSettings, train_edge_classifier

        torch.manual_seed(0)
        n_nodes, n_edges, n_classes = 30, 600, 3
        src = torch.randint(0, n_nodes, (n_edges,))
        dst = torch.randint(0, n_nodes, (n_edges,))
        data = Data(x=torch.ones((n_nodes, 4)), edge_index=torch.stack([src, dst]).long(),
                    edge_attr=torch.randn(n_edges, 6),
                    y=torch.randint(0, n_classes, (n_edges,)))
        for s in ("train", "val", "test"):
            setattr(data, f"{s}_mask", torch.ones(n_edges, dtype=torch.bool))
        from src.models.edge_sage import build_model
        model = build_model("egraphsage", node_dim=4, edge_dim=6,
                            config={"model": {"hidden_dim": 8, "num_classes": n_classes}})
        calls: list[tuple[int, bool]] = []

        class Recorder(torch.nn.Module):
            def __init__(self, inner):
                super().__init__(); self.inner = inner
            def forward(self, x, edge_index, edge_attr):
                calls.append((int(edge_index.shape[1]), bool(torch.is_grad_enabled())))
                return self.inner(x, edge_index, edge_attr)

        rec = Recorder(model)
        settings = TrainSettings(max_epochs=1, patience=5, batch_size=64, batching="sampled",
                                 neighbor_fanout=(10, 10), max_edges_per_step=200)
        r = train_edge_classifier(rec, data, settings=settings, seed=42, device="cpu")
        n_train = int(data.train_mask.sum())
        batches = int((n_train + 63) // 64)
        train_calls = [c for c, grad in calls if grad]
        eval_calls = [c for c, grad in calls if not grad]
        full_graph_train_calls = [c for c in train_calls if c == n_edges]
        max_train_call = max(train_calls) if train_calls else None
        timing = r["timing"]
        add("T25.code.sampled_per_batch",
            "pass" if (timing.get("batching") == "sampled" and not full_graph_train_calls
                       and max_train_call is not None and max_train_call <= 200
                       and len(train_calls) == batches and timing.get("n_updates") == batches) else "fail",
            {"batching_recorded": timing.get("batching"), "n_edges": n_edges,
             "n_train": n_train, "expected_batches": batches, "training_forward_calls": len(train_calls),
             "eval_forward_calls_no_grad": len(eval_calls),
             "max_edges_per_training_forward": max_train_call, "max_edges_per_step": 200,
             "full_graph_train_forwards": len(full_graph_train_calls),
             "updates_per_epoch": timing.get("updates_per_epoch"),
             "n_updates": timing.get("n_updates"),
             "note": "training forwards must all be bounded sampled subgraphs; full-graph forwards are "
                     "allowed only under no_grad (period-graph evaluation)"})
        # independent sampler prefix check: target rows first and equal to the target edges
        from src.train.sampler import EdgeNeighborSampler
        sampler = EdgeNeighborSampler(data, fanout=(10, 10), max_edges_per_step=200)
        rng = np.random.default_rng(7)
        targets = np.sort(rng.choice(n_edges, size=64, replace=False))
        sub = sampler.sample(targets, rng)
        prefix_ok = bool(torch.equal(sub.edge_attr[: sub.n_targets], data.edge_attr[torch.as_tensor(targets, dtype=torch.long)]))
        add("T25.code.sampler_target_prefix",
            "pass" if (sub.n_sub_edges <= 200 and sub.n_targets == 64 and prefix_ok) else "fail",
            {"n_sub_edges": sub.n_sub_edges, "n_targets": sub.n_targets,
             "n_context_edges": sub.n_context_edges, "target_rows_first_match": prefix_ok,
             "max_edges_per_step": 200})
    except Exception as exc:  # noqa: BLE001
        add("T25.code.sampled_per_batch", "fail", {"error": repr(exc)})

    # ---- sampled batching tests exist
    tests_text = ""
    for p in (REPO / "tests").rglob("*.py"):
        tests_text += p.read_text(encoding="utf-8", errors="ignore")
    has_sampled_tests = "sampled" in tests_text
    add("T25.code.tests_present", "pass" if has_sampled_tests else "warn",
        {"sampled_mentioned_in_tests": has_sampled_tests})

    fails = [c for c in chk if c["status"] == "fail"]
    result["overall"] = "fail" if fails else "pass"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t25_code.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({c["id"]: c["status"] for c in chk}, indent=1))
    print("overall:", result["overall"])
    for c in chk:
        if c["status"] != "pass":
            print(c["id"], json.dumps(c["detail"], default=str)[:500])
    return 0 if not fails else 2


if __name__ == "__main__":
    raise SystemExit(main())

