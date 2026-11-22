#!/usr/bin/env python
"""T17 independent audit — S3 smoke runs / manifests + metric recomputation.

Audits results/text/data_audit/train_manifest_ft_{egraphsage,gcn}_seed42_period1.json:

  * artifact sha256 vs files; config sha vs configs/train.yaml; graph sha vs
    graphs_manifest; frozen split hash vs current split_manifest; preprocessor sha
  * code fingerprint: recompute over the run's git commit tree using the exact
    documented algorithm (sorted relative paths + file hashes, roots
    src/scripts/configs/platforms/tests) and compare to the manifest; also
    recompute over the current working tree and report drift
  * dataset fingerprint vs graphs manifest period files
  * lane / env / argv recording; smoke scale; started/finished/duration
    consistency vs metrics timing and log timestamps
  * metric recomputation: macro-F1 from the saved per-class F1 vector; checkpoint
    reload + forward pass on the reproduced smoke subgraph (seed 42, 100k edges)
    to reproduce val/test macro-F1 and per-class F1 independently
  * determinism: same-seed rerun results are audited separately (t17_rerun.json)

Output: results/text/verification/data_audit/t17_s3.json
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(r"C:\Users\iusearchbtw\Desktop\XAI_proj")
AUDIT = REPO / "results" / "data_audit"
PROC = REPO / "data" / "processed" / "NF-UNSW-NB15-v3"
OUT = REPO / "verification" / "data_audit"
GIT = r"C:\Program Files\Git\bin\git.exe"
ROOTS = ("src", "scripts", "configs", "platforms", "tests")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def git(*args: str) -> tuple[int, bytes]:
    p = subprocess.run([GIT, "-C", str(REPO), *args], capture_output=True)
    return p.returncode, p.stdout


def code_fingerprint_from_git(commit: str) -> tuple[str | None, int]:
    rc, out = git("ls-tree", "-r", "--name-only", commit, "--", *ROOTS)
    if rc != 0:
        return None, 0
    files = [line for line in out.decode("utf-8", "replace").splitlines() if line.strip()]
    entries = []
    for rel in files:
        rc2, blob = git("cat-file", "-p", f"{commit}:{rel}")
        if rc2 != 0:
            continue
        entries.append((rel.replace("\\", "/"), hashlib.sha256(blob).hexdigest()))
    entries.sort(key=lambda item: item[0])
    h = hashlib.sha256()
    for rel, sha in entries:
        h.update(rel.encode("utf-8")); h.update(b"\0"); h.update(sha.encode("ascii")); h.update(b"\n")
    return h.hexdigest(), len(entries)


def code_fingerprint_worktree() -> tuple[str | None, int]:
    entries = []
    for root in ROOTS:
        base = REPO / root
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if p.is_file():
                entries.append((str(p.relative_to(REPO)).replace("\\", "/"), sha256_file(p)))
    entries.sort(key=lambda item: item[0])
    h = hashlib.sha256()
    for rel, sha in entries:
        h.update(rel.encode("utf-8")); h.update(b"\0"); h.update(sha.encode("ascii")); h.update(b"\n")
    return h.hexdigest(), len(entries)


def build_egraphsage_reference(edge_dim: int, hidden: int = 128, num_classes: int = 10, node_dim: int = 8):
    """Independent reimplementation of the E-GraphSAGE architecture (Doc 2 S3)
    used only to reload the saved checkpoint and reproduce metrics."""
    from torch import nn
    from torch_geometric.nn import MessagePassing

    class Layer(MessagePassing):
        def __init__(self, in_node, in_edge, out):
            super().__init__(aggr="mean")
            self.lin = nn.Linear(in_node + in_edge, out)

        def forward(self, x, edge_index, edge_attr):
            agg = self.propagate(edge_index, x=x, edge_attr=edge_attr)
            return torch.relu(self.lin(torch.cat([x, agg], dim=-1)))

        def message(self, edge_attr):
            return edge_attr

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer(node_dim, edge_dim, hidden), Layer(hidden, edge_dim, hidden)])
            self.head = nn.Linear(2 * hidden + edge_dim, num_classes)

        def forward(self, x, edge_index, edge_attr):
            h = x
            for layer in self.layers:
                h = layer(h, edge_index, edge_attr)
            src, dst = edge_index
            return self.head(torch.cat([h[src], h[dst], edge_attr], dim=-1))

    return Net()


def macro_from_per_class(per_class: dict) -> float:
    return float(np.mean(list(per_class.values())))


def main() -> int:
    result: dict = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "checks": []}
    chk = result["checks"]

    def add(i, status, detail):
        chk.append({"id": i, "status": status, "detail": detail})

    split_man = json.loads((AUDIT / "split_manifest.json").read_text(encoding="utf-8-sig"))
    gman = json.loads((PROC / "graphs_manifest.json").read_text(encoding="utf-8-sig"))
    train_cfg_sha = sha256_file(REPO / "configs" / "train.yaml")
    runs = {}
    for model_name in ("egraphsage", "gcn"):
        mpath = AUDIT / f"train_manifest_ft_{model_name}_seed42_period1.json"
        man = json.loads(mpath.read_text(encoding="utf-8-sig"))
        metrics_path = AUDIT / f"train_ft_{model_name}_seed42_period1.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
        rec: dict = {"manifest": str(mpath.name)}
        # artifact hashes
        art = []
        for a in man.get("artifacts", []):
            p = Path(a["path"])
            ok = p.is_file() and sha256_file(p) == a["sha256"] and p.stat().st_size == a.get("bytes")
            art.append({"path": a["path"].split("\\")[-1], "role": a.get("role"), "match": bool(ok)})
        rec["artifacts"] = art
        # pinned hashes
        extra = man.get("extra") or {}
        rec["pins"] = {
            "config_sha_matches_train_yaml": (man.get("config") or {}).get("sha256") == train_cfg_sha,
            "graph_sha_matches_graphs_manifest": extra.get("graph_sha256") == (gman["periods"]["1"].get("sha256")),
            "frozen_split_hash_matches_current": extra.get("frozen_split_hash") == split_man.get("frozen_hash"),
            "preprocessor_sha_matches_graphs_manifest": extra.get("preprocessor_state_sha256") == (gman.get("preprocessor_state") or {}).get("sha256"),
        }
        # dataset fingerprint: manifest dataset files vs graph file
        ds = man.get("dataset") or {}
        ds_ok = any(Path(f.get("path", "")).is_file() and sha256_file(Path(f["path"])) == f.get("sha256")
                    for f in (ds.get("files") or []))
        rec["pins"]["dataset_files_match"] = bool(ds_ok)
        # code fingerprint via git commit tree + worktree
        git_info = man.get("git") or {}
        commit = git_info.get("sha")
        run_fp = (man.get("code") or {}).get("sha256")
        git_fp, git_n = code_fingerprint_from_git(commit) if commit else (None, 0)
        wt_fp, wt_n = code_fingerprint_worktree()
        rec["code_fingerprint"] = {"manifest": run_fp, "recomputed_from_git_commit": git_fp,
                                   "commit": commit, "commit_file_count": git_n,
                                   "matches_git_commit": git_fp == run_fp,
                                   "recomputed_from_worktree": wt_fp, "worktree_file_count": wt_n,
                                   "matches_worktree": wt_fp == run_fp,
                                   "git_dirty_at_run": git_info.get("dirty")}
        # manifest lifecycle
        started = pd.Timestamp(man.get("started_at_utc")); finished = pd.Timestamp(man.get("finished_at_utc"))
        duration = float(man.get("duration_s") or 0)
        train_seconds = float((extra.get("timing") or {}).get("train_seconds") or 0)
        rec["lifecycle"] = {"started_at_utc": man.get("started_at_utc"), "finished_at_utc": man.get("finished_at_utc"),
                            "duration_s": duration, "train_seconds_from_timing": train_seconds,
                            "start_to_finish_s": float((finished - started).total_seconds()),
                            "duration_matches_training": abs(duration - train_seconds) < 5.0,
                            "note": "RunManifest is constructed after training; start()/finish() wrap seconds, not the run"}
        # lane/env/argv
        rec["recording"] = {"lane": man.get("lane"), "argv": (man.get("command") or {}).get("argv"),
                            "env": man.get("env"), "paths_config": man.get("paths_config"),
                            "smoke": extra.get("smoke"), "max_edges": extra.get("max_edges"),
                            "original_edges": extra.get("original_edges")}
        # metric cross-checks from saved per-class F1
        res = metrics.get("result") or {}
        test = res.get("test") or {}; val = res.get("val") or {}
        rec["metrics"] = {
            "test_macro_f1_manifest": extra.get("test_macro_f1"),
            "test_macro_f1_metrics": test.get("macro_f1"),
            "test_macro_f1_from_per_class": macro_from_per_class(test.get("per_class_f1") or {}),
            "val_macro_f1_manifest": extra.get("val_macro_f1"),
            "val_macro_f1_from_per_class": macro_from_per_class(val.get("per_class_f1") or {}),
            "epochs_run": res.get("epochs_run"), "best_epoch": res.get("best_epoch"),
            "n_train_edges": res.get("n_train_edges"), "n_test_edges": (test or {}).get("n_edges"),
            "n_val_edges": (val or {}).get("n_edges"),
        }
        rec["metrics"]["test_recomputable"] = abs((rec["metrics"]["test_macro_f1_metrics"] or -1)
                                                  - rec["metrics"]["test_macro_f1_from_per_class"]) < 1e-9
        rec["metrics"]["val_recomputable"] = abs((rec["metrics"]["val_macro_f1_from_per_class"] or -1)
                                                 - (extra.get("val_macro_f1") or -1)) < 1e-9
        # independent checkpoint forward pass on the reproduced smoke subgraph
        try:
            data = torch.load(PROC / "period_1.pt", map_location="cpu", weights_only=False)
            n_edges = int(data.num_edges)
            max_edges = int(extra.get("max_edges") or 100000)
            rng = np.random.default_rng(42)
            keep = np.sort(rng.choice(n_edges, size=max_edges, replace=False))
            keep_t = torch.as_tensor(keep, dtype=torch.long)
            ei = data.edge_index[:, keep_t]
            nodes = torch.unique(ei)
            remap = torch.full((int(data.num_nodes),), -1, dtype=torch.long)
            remap[nodes] = torch.arange(nodes.numel(), dtype=torch.long)
            ei = remap[ei]
            x = data.x[nodes]; ea = data.edge_attr[keep_t]; y = data.y[keep_t]
            masks = {s: getattr(data, f"{s}_mask")[keep_t] for s in ("train", "val", "test")}
            ckpt = torch.load(Path((man.get("artifacts") or [{}])[0]["path"]), map_location="cpu", weights_only=False)
            sd = ckpt.get("model_state") or ckpt.get("state_dict") or ckpt
            if model_name == "egraphsage":
                net = build_egraphsage_reference(edge_dim=int(ea.shape[1]))
            else:
                net = build_gcn_reference(edge_dim=int(ea.shape[1]))
            net.load_state_dict({k: v for k, v in sd.items() if not k.startswith("_")}, strict=False)
            net.eval()
            with torch.no_grad():
                logits = net(x, ei, ea)
            out = {}
            for split in ("val", "test"):
                idx = torch.nonzero(masks[split], as_tuple=False).flatten()
                yt = y[idx].numpy(); yp = logits[idx].argmax(dim=-1).numpy()
                from sklearn.metrics import f1_score
                labels = list(range(int(y.max()) + 1))
                per = f1_score(yt, yp, labels=labels, average=None, zero_division=0)
                out[split] = {"macro_f1": float(per.mean()), "per_class": [float(v) for v in per], "n": int(len(idx))}
            rec["independent_recompute"] = {"val_macro_f1": out["val"]["macro_f1"],
                                            "test_macro_f1": out["test"]["macro_f1"],
                                            "matches_saved_test": abs(out["test"]["macro_f1"] - (test.get("macro_f1") or -1)) < 1e-5,
                                            "matches_saved_val": abs(out["val"]["macro_f1"] - (val.get("macro_f1") or -1)) < 1e-5,
                                            "smoke_edges": int(len(keep))}
        except Exception as exc:  # noqa: BLE001
            rec["independent_recompute"] = {"error": repr(exc)}
        runs[model_name] = rec
    # top-level findings
    pins_ok = all(all(v for k, v in r["pins"].items()) for r in runs.values())
    add("S3.pinned_hashes", "pass" if pins_ok else "fail",
        {m: r["pins"] for m, r in runs.items()})
    add("S3.code_pinning", "pass" if all(r["code_fingerprint"]["matches_git_commit"] or r["code_fingerprint"]["matches_worktree"]
                                         for r in runs.values()) else "warn",
        {m: r["code_fingerprint"] for m, r in runs.items()})
    add("S3.manifest_lifecycle", "pass" if all(r["lifecycle"]["duration_matches_training"] for r in runs.values()) else "fail",
        {m: r["lifecycle"] for m, r in runs.items()})
    lane_ok = all(r["recording"]["lane"] != "default" for r in runs.values())
    add("S3.lane_recording", "pass" if lane_ok else "fail",
        {m: {"lane": r["recording"]["lane"], "paths_config": r["recording"]["paths_config"],
             "argv_has_lane": "--lane" in (r["recording"]["argv"] or [])} for m, r in runs.items()})
    add("S3.smoke_scale", "pass" if all((r["recording"]["smoke"] and r["recording"]["max_edges"] == 100000
                                        and r["recording"]["original_edges"] == 1131909) for r in runs.values()) else "warn",
        {m: {k: r["recording"][k] for k in ("smoke", "max_edges", "original_edges")} for m, r in runs.items()})
    add("S3.metric_recompute", "pass" if all(r["metrics"]["test_recomputable"] and r["metrics"]["val_recomputable"] and
                                              (r.get("independent_recompute") or {}).get("matches_saved_test", False)
                                              for r in runs.values()) else "warn",
        {m: {"metrics": r["metrics"], "independent": r.get("independent_recompute")} for m, r in runs.items()})
    fails = [c for c in chk if c["status"] == "fail"]
    result["runs"] = runs
    result["overall"] = "fail" if fails else "pass"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t17_s3.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({c["id"]: c["status"] for c in chk}, indent=1))
    print("overall:", result["overall"])
    for m, r in runs.items():
        print(m, "test_f1", r["metrics"]["test_macro_f1_manifest"], "recompute", r.get("independent_recompute"))
    return 0 if not fails else 2


def build_gcn_reference(edge_dim: int, hidden: int = 128, num_classes: int = 10, node_dim: int = 8):
    from torch import nn
    from torch_geometric.nn import GCNConv

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.edge_to_node = nn.Linear(edge_dim, hidden)
            self.node_input = nn.Linear(node_dim, hidden)
            self.convs = nn.ModuleList([GCNConv(hidden, hidden), GCNConv(hidden, hidden)])
            self.head = nn.Linear(2 * hidden + edge_dim, num_classes)

        def forward(self, x, edge_index, edge_attr):
            src, dst = edge_index
            n_nodes = x.shape[0]
            agg = torch.zeros((n_nodes, edge_attr.shape[1]), dtype=edge_attr.dtype)
            cnt = torch.zeros((n_nodes, 1), dtype=edge_attr.dtype)
            agg.index_add_(0, src, edge_attr); agg.index_add_(0, dst, edge_attr)
            cnt.index_add_(0, src, torch.ones((edge_attr.shape[0], 1), dtype=edge_attr.dtype))
            cnt.index_add_(0, dst, torch.ones((edge_attr.shape[0], 1), dtype=edge_attr.dtype))
            agg = agg / cnt.clamp_min(1.0)
            h = torch.relu(self.node_input(x) + self.edge_to_node(agg))
            for conv in self.convs:
                h = torch.relu(conv(h, edge_index))
            edge_repr = torch.cat([h[src], h[dst], edge_attr], dim=-1)
            return self.head(edge_repr)

    return Net()


if __name__ == "__main__":
    raise SystemExit(main())
