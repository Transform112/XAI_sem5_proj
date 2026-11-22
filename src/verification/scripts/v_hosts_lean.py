#!/usr/bin/env python
"""Lean single-pass host-graph audit for very large / host-rich datasets (CIC-IDS2018).

Memory-bounded alternative to v_hosts.py: reads only [ts, src, dst, Attack],
keeps global unique-pair keys as int64 arrays (not Python tuples), builds a CSR
adjacency at the end for 2-hop feasibility, and unions components from the unique
pairs. Produces the same metrics as v_hosts.py for the D-025 R2 screen.

Usage:
  .venv\\Scripts\\python.exe verification\\scripts\\v_hosts_lean.py --dataset NF-CSE-CIC-IDS2018-v3
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

VERIFIER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VERIFIER_ROOT.parents[1]
DEFAULT_RAW = REPO_ROOT / "data" / "raw"
DEFAULT_OUT = VERIFIER_ROOT / "data_audit"
DATE_COL = "FLOW_START_MILLISECONDS"
SRC_COL = "IPV4_SRC_ADDR"
DST_COL = "IPV4_DST_ADDR"
LABEL_CANDS = ["Attack", "Attack_cat", "attack_cat"]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class UF:
    __slots__ = ("parent", "size")

    def __init__(self, n: int):
        self.parent = np.arange(n, dtype=np.int64)
        self.size = np.ones(n, dtype=np.int64)

    def find(self, x: int) -> int:
        p = self.parent
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            p[x], x = root, p[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def components(self) -> list[int]:
        roots = {}
        p = self.parent
        for i in range(len(p)):
            r = self.find(i)
            roots[r] = roots.get(r, 0) + 1
        return sorted(roots.values(), reverse=True)


def iso(ms: int) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def pct(vals: np.ndarray, q: float) -> int:
    if len(vals) == 0:
        return 0
    return int(vals[min(len(vals) - 1, max(0, int(round(q * (len(vals) - 1))))) ])


def audit(ds: str, raw: Path, out: Path, periods: int = 5, chunksize: int = 400_000,
          max_pair_keys: int = 12_000_000, seed: int = 20261101) -> dict:
    t0 = time.time()
    csv = [p for p in (raw / ds).rglob("*.csv") if p.stat().st_size > 0][0]
    header = pd.read_csv(csv, nrows=0)
    cols = [str(c) for c in header.columns]
    label_col = next((c for c in LABEL_CANDS if c in cols), None)
    # ---- pass A: ts boundaries
    ts_parts = []
    dec = 0
    prev = None
    n = 0
    for chunk in pd.read_csv(csv, usecols=[DATE_COL], dtype={DATE_COL: "int64"}, chunksize=1_000_000):
        v = chunk[DATE_COL].to_numpy()
        if len(v):
            if len(v) > 1:
                dec += int((np.diff(v) < 0).sum())
            if prev is not None and v[0] < prev:
                dec += 1
            prev = int(v[-1])
            n += len(v)
            ts_parts.append(v)
    ts_all = np.concatenate(ts_parts)
    ts_min, ts_max = int(ts_all.min()), int(ts_all.max())
    ts_sorted = np.sort(ts_all)
    idx = [round(k * n / periods) for k in range(periods + 1)]
    ec_bounds = ts_sorted[np.clip(idx, 0, n - 1)].astype(np.int64)
    dur_bounds = np.array([ts_min + round((ts_max - ts_min) * k / periods) for k in range(periods + 1)], dtype=np.int64)
    del ts_sorted, ts_all
    print(f"[{ds}] pass A rows={n:,} dec={dec} ({time.time()-t0:.0f}s)")
    # ---- pass B
    t1 = time.time()
    host_id: dict[str, int] = {}
    host_names: list[str] = []
    ec = [dict(hosts=set(), edges=0, self_loops=0, labels=Counter()) for _ in range(periods)]
    pair_keys_chunks: list[np.ndarray] = []
    pair_capped = False
    rng = np.random.default_rng(seed)
    samples = [[] for _ in range(periods)]
    usecols = [DATE_COL, SRC_COL, DST_COL] + ([label_col] if label_col else [])
    dtype = {DATE_COL: "int64", SRC_COL: "str", DST_COL: "str"}
    if label_col:
        dtype[label_col] = "str"
    mid = ec_bounds[1:-1]
    for chunk in pd.read_csv(csv, usecols=usecols, chunksize=chunksize, dtype=dtype):
        t = chunk[DATE_COL].to_numpy(np.int64)
        src = chunk[SRC_COL].to_numpy(dtype=object)
        dst = chunk[DST_COL].to_numpy(dtype=object)
        lab = chunk[label_col].to_numpy(dtype=object) if label_col else None
        p_ec = np.searchsorted(mid, t, side="right")
        allh = pd.unique(np.concatenate([src, dst]))
        for h in allh:
            if h not in host_id:
                host_id[h] = len(host_names)
                host_names.append(h)
        sid = pd.Series(src).map(host_id).to_numpy(np.int64)
        did = pd.Series(dst).map(host_id).to_numpy(np.int64)
        for p in np.unique(p_ec):
            m = p_ec == p
            if lab is not None:
                ec[p]["labels"].update(pd.Series(lab[m]).value_counts().to_dict())
            ec[p]["edges"] += int(m.sum())
            ec[p]["self_loops"] += int((src[m] == dst[m]).sum())
            ids = np.unique(np.concatenate([sid[m], did[m]]))
            ec[p]["hosts"].update(ids.tolist())
            if ec[p]["edges"] and len(samples[p]) < 3000:
                take = min(3000 - len(samples[p]), max(1, int(m.sum()) // 100))
                ii = rng.integers(0, int(m.sum()), size=take)
                samples[p].extend(zip(sid[m][ii].tolist(), did[m][ii].tolist()))
        if not pair_capped:
            keys = (sid.astype(np.uint64) << np.uint64(32)) | did.astype(np.uint64)
            uniq = np.unique(keys)
            pair_keys_chunks.append(uniq.astype(np.uint64))
            total_uniq = sum(len(a) for a in pair_keys_chunks)
            if total_uniq > max_pair_keys:
                pair_capped = True
    print(f"[{ds}] pass B ({time.time()-t1:.0f}s) hosts={len(host_id)}")
    if pair_capped:
        keys = np.unique(np.concatenate(pair_keys_chunks))
    else:
        keys = np.unique(np.concatenate(pair_keys_chunks)) if pair_keys_chunks else np.array([], dtype=np.uint64)
    print(f"[{ds}] unique pair keys: {len(keys):,} (capped={pair_capped})")
    sid_u = (keys >> np.uint64(32)).astype(np.int64)
    did_u = (keys & np.uint64(0xFFFFFFFF)).astype(np.int64)
    # components over unique pairs
    uf = UF(len(host_names))
    for a, b in zip(sid_u.tolist(), did_u.tolist()):
        uf.union(a, b)
    comp_sizes = uf.components()
    # adjacency CSR (undirected) for 2-hop
    deg = np.bincount(sid_u, minlength=len(host_names)) + np.bincount(did_u, minlength=len(host_names))
    indptr = np.zeros(len(host_names) + 1, dtype=np.int64)
    np.cumsum(deg, out=indptr[1:])
    indices = np.empty(int(deg.sum()), dtype=np.int64)
    cursor = indptr[:-1].copy()
    for a, b in zip(sid_u.tolist(), did_u.tolist()):
        indices[cursor[a]] = b
        cursor[a] += 1
        indices[cursor[b]] = a
        cursor[b] += 1

    def neighbours(v: int) -> np.ndarray:
        return indices[indptr[v]:indptr[v + 1]]

    per_period = []
    prev_hosts = None
    for p, acc in enumerate(ec):
        degp = Counter()
        for a, b in samples[p]:
            degp[a] += 1
            degp[b] += 1
        n_hosts_p = len(acc["hosts"])
        rec = None
        if prev_hosts is not None and n_hosts_p:
            rec = round(len(acc["hosts"] & prev_hosts) / n_hosts_p, 6)
        entry = {"period": p + 1, "edges": acc["edges"], "hosts": n_hosts_p,
                 "self_loops": acc["self_loops"],
                 "label_counts": dict(acc["labels"].most_common()),
                 "label_classes": len(acc["labels"]),
                 "share_of_period_hosts_seen_in_prev": rec}
        if prev_hosts is not None:
            entry["consecutive_jaccard"] = round(len(acc["hosts"] & prev_hosts) / len(acc["hosts"] | prev_hosts), 6)
        sizes = []
        for a, b in samples[p]:
            nb = set(neighbours(a).tolist()) | set(neighbours(b).tolist())
            nb.discard(a); nb.discard(b)
            sizes.append(len(nb))
        if sizes:
            arr = np.array(sizes)
            entry["two_hop"] = {"sampled_edges": len(sizes),
                                "share_nonempty": round(float((arr > 0).mean()), 6),
                                "median_hosts": int(np.median(arr)), "p90_hosts": pct(np.sort(arr), 0.9),
                                "max_hosts": int(arr.max())}
        prev_hosts = acc["hosts"]
        per_period.append(entry)
    g_hosts = set().union(*[a["hosts"] for a in ec])
    edges = sum(a["edges"] for a in ec)
    recurrences = [p["share_of_period_hosts_seen_in_prev"] for p in per_period
                   if p.get("share_of_period_hosts_seen_in_prev") is not None]
    jaccs = [p["consecutive_jaccard"] for p in per_period if p.get("consecutive_jaccard") is not None]
    two_hop_medians = [p["two_hop"]["median_hosts"] for p in per_period if p.get("two_hop")]
    result = {
        "dataset": ds, "check": "structure_hosts_lean", "audited_utc": utc_now(),
        "script": "verification/src/scripts/v_hosts_lean.py", "raw_file": str(csv),
        "engine": "pandas-lean",
        "pass_a": {"rows": n, "decreasing_ts_steps": dec, "monotonic_nondecreasing": dec == 0,
                   "ts_min_iso": iso(ts_min), "ts_max_iso": iso(ts_max)},
        "boundaries": {"equal_count_ts_ms": ec_bounds.tolist(),
                       "equal_count_ts_iso": [iso(x) for x in ec_bounds],
                       "equal_duration_ts_ms": dur_bounds.tolist()},
        "global": {
            "hosts": len(g_hosts), "edges": edges,
            "self_loops": sum(a["self_loops"] for a in ec),
            "unique_host_pairs": int(len(keys)), "pairs_capped": bool(pair_capped),
            "repeated_pair_share_of_edges": round(1 - len(keys) / edges, 6) if edges else None,
            "n_components": len(comp_sizes), "component_size_top5": comp_sizes[:5],
            "giant_component_nodes": comp_sizes[0] if comp_sizes else 0,
            "giant_component_node_share": round(comp_sizes[0] / len(host_names), 6) if host_names else None,
            "degree": {"n_hosts": len(host_names),
                       "median": pct(np.sort(np.bincount(sid_u, minlength=len(host_names)) + np.bincount(did_u, minlength=len(host_names))), 0.5),
                       "mean": round(float((np.bincount(sid_u, minlength=len(host_names)) + np.bincount(did_u, minlength=len(host_names))).mean()), 4),
                       "max": int((np.bincount(sid_u, minlength=len(host_names)) + np.bincount(did_u, minlength=len(host_names))).max())},
        },
        "period_stats_equal_count": per_period,
        "host_persistence": {"hosts_per_period": [p["hosts"] for p in per_period],
                             "share_seen_prev": [p.get("share_of_period_hosts_seen_in_prev") for p in per_period],
                             "consecutive_jaccard": [p.get("consecutive_jaccard") for p in per_period]},
        "criteria": {
            "median_host_degree_ge_2": None,
            "giant_component_share_ge_0_90": {"value": None, "threshold": 0.90},
            "host_recurrence_ge_0_30": {"value": min(recurrences) if recurrences else None, "threshold": 0.30},
            "anchor_2hop_nonempty_ge_0_95": {"value": min((p["two_hop"]["share_nonempty"] for p in per_period if p.get("two_hop")), default=None), "threshold": 0.95},
            "repeated_host_pair_share_ge_0_05": {"value": None, "threshold": 0.05},
        },
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    med = result["global"]["degree"]["median"]
    result["criteria"]["median_host_degree_ge_2"] = {"value": med, "threshold": 2}
    result["criteria"]["giant_component_share_ge_0_90"]["value"] = result["global"]["giant_component_node_share"]
    result["criteria"]["repeated_host_pair_share_ge_0_05"]["value"] = result["global"]["repeated_pair_share_of_edges"]
    for k, c in result["criteria"].items():
        if c["value"] is None:
            c["status"] = "blocked"
        elif k == "median_host_degree_ge_2":
            c["status"] = "pass" if c["value"] >= 2 else "fail"
        elif k == "giant_component_share_ge_0_90":
            c["status"] = "pass" if c["value"] >= 0.90 else "fail"
        elif k == "host_recurrence_ge_0_30":
            c["status"] = "pass" if c["value"] >= 0.30 else "fail"
        elif k == "anchor_2hop_nonempty_ge_0_95":
            c["status"] = "pass" if c["value"] >= 0.95 else "fail"
        else:
            c["status"] = "info"
    fails = [k for k, c in result["criteria"].items() if c["status"] == "fail"]
    result["structure_verdict"] = "pass" if not fails else "fail"
    result["n_fail"] = len(fails)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"structure_hosts_{ds}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"[{ds}] hosts={len(host_names)} edges={edges:,} comps={len(comp_sizes)} "
          f"giant={result['global']['giant_component_node_share']} median_deg={med} "
          f"recurrence_min={result['criteria']['host_recurrence_ge_0_30']['value']} "
          f"2hop_medians={two_hop_medians} verdict={result['structure_verdict'].upper()} ({result['elapsed_seconds']}s)")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--raw", default=str(DEFAULT_RAW))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--periods", type=int, default=5)
    args = ap.parse_args()
    audit(args.dataset, Path(args.raw), Path(args.out), args.periods)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

