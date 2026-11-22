#!/usr/bin/env python
"""Fast host-graph structure audit for the whole NF-v3 family (verifier-owned).

Answers the G1c "is an alternate v3 dataset host-rich?" question on raw CSVs with
a chunked pandas implementation (no project code):

  * hosts per period (unique src/dst union), edges, self-loops
  * degree distribution (mean/median/p10/p90/max, share deg 1, share >=2)
  * connected components (union-find over unique host pairs) + giant share
  * repeated host-pair share (multi-edge structure), unique directed pairs
  * host recurrence across periods (Jaccard, share of period-t hosts seen in t-1)
  * 2-hop neighbourhood feasibility on sampled edges (adjacency from unique pairs)
  * per-period class counts (Attack) under equal-count periods; equal-duration
    boundaries also reported (they can be empty when the capture has a long gap)
  * timestamp gaps (capture-session evidence)

Definitions match docs/09 section 4 and verification/src/scripts/v_structure.py.
The heavier v_structure.py is the stdlib cross-check used for UNSW; this script
is the engine for the 17-27M-row datasets.

Usage (repo root, .venv python has pandas/numpy):
  .venv\\Scripts\\python.exe verification\\scripts\\v_hosts.py --dataset all
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
LABEL_CANDS = ["Attack", "Attack_cat", "attack_cat", "label", "Label"]
DATASETS = ["NF-UNSW-NB15-v3", "NF-ToN-IoT-v3", "NF-BoT-IoT-v3", "NF-CSE-CIC-IDS2018-v3"]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class UF:
    def __init__(self, n: int):
        self.p = np.arange(n, dtype=np.int64)
        self.sz = np.ones(n, dtype=np.int64)

    def find(self, x: int) -> int:
        p = self.p
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
        if self.sz[ra] < self.sz[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        self.sz[ra] += self.sz[rb]

    def component_sizes(self) -> list[int]:
        roots = np.array([self.find(i) for i in range(len(self.p))])
        uniq, cnt = np.unique(roots, return_counts=True)
        return sorted(cnt.tolist(), reverse=True)


def locate_csv(raw: Path, ds: str) -> Path:
    d = raw / ds
    if d.is_dir():
        csvs = [p for p in sorted(d.rglob("*.csv")) if p.stat().st_size > 0]
        if len(csvs) == 1:
            return csvs[0]
        if csvs:
            raise SystemExit(f"expected 1 CSV under {d}, found {len(csvs)}")
    zp = raw / f"{ds}.zip"
    if zp.is_file():
        return zp  # pandas reads a single-CSV zip via compression inference
    raise FileNotFoundError(f"no raw data for {ds}")


def header_of(path: Path) -> list[str]:
    h = pd.read_csv(path, nrows=0)
    return [str(c) for c in h.columns]


def pct(sorted_vals: np.ndarray, p: float) -> int:
    if len(sorted_vals) == 0:
        return 0
    return int(sorted_vals[min(len(sorted_vals) - 1, max(0, int(round(p * (len(sorted_vals) - 1))))) ])


def audit(ds: str, raw: Path, out: Path, periods: int, chunksize: int, seed: int = 20261101,
          max_unique_pairs: int = 5_000_000) -> dict:
    t_start = time.time()
    path = locate_csv(raw, ds)
    header = header_of(path)
    cols = {c: i for i, c in enumerate(header)}
    for need in (DATE_COL, SRC_COL, DST_COL):
        if need not in cols:
            raise SystemExit(f"{ds}: missing column {need}")
    label_col = next((c for c in LABEL_CANDS if c in cols), None)
    rng = np.random.default_rng(seed)
    result: dict = {"dataset": ds, "check": "structure_hosts", "audited_utc": utc_now(),
                    "script": "verification/src/scripts/v_hosts.py", "raw_file": str(path),
                    "n_columns": len(header), "label_column": label_col,
                    "periods": periods, "chunksize": chunksize, "seed": seed,
                    "engine": "pandas-chunked"}

    # ---------- pass A: timestamps + equal-count/equal-duration boundaries
    t0 = time.time()
    ts_parts = []
    dec = 0
    prev_last = None
    n_rows = 0
    for chunk in pd.read_csv(path, usecols=[DATE_COL], chunksize=chunksize,
                             dtype={DATE_COL: "int64"}):
        v = chunk[DATE_COL].to_numpy()
        n_rows += len(v)
        if len(v) > 1:
            dec += int((np.diff(v) < 0).sum())
        if prev_last is not None and v[0] < prev_last:
            dec += 1
        prev_last = int(v[-1]) if len(v) else prev_last
        ts_parts.append(v)
    ts = np.concatenate(ts_parts) if ts_parts else np.array([], dtype=np.int64)
    del ts_parts
    ts_min, ts_max = int(ts.min()), int(ts.max())
    ts_sorted = np.sort(ts)  # needed for quantile boundaries (raw order not chronological)
    idx = [round(k * n_rows / periods) for k in range(periods + 1)]
    ec_bounds = ts_sorted[np.clip(idx, 0, n_rows - 1)].astype(np.int64)
    dur_bounds = np.array([ts_min + round((ts_max - ts_min) * k / periods) for k in range(periods + 1)], dtype=np.int64)
    del ts_sorted, ts
    result["pass_a"] = {"rows": n_rows, "decreasing_ts_steps": dec,
                        "monotonic_nondecreasing": dec == 0,
                        "ts_min": ts_min, "ts_max": ts_max,
                        "ts_min_iso": datetime.fromtimestamp(ts_min / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "ts_max_iso": datetime.fromtimestamp(ts_max / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "seconds": round(time.time() - t0, 1)}
    result["boundaries"] = {"rule_primary": "equal_counts", "periods": periods,
                            "equal_count_ts_ms": ec_bounds.tolist(),
                            "equal_duration_ts_ms": dur_bounds.tolist(),
                            "equal_count_row_index_applicable": dec == 0}
    print(f"[{ds}] pass A: rows={n_rows:,} dec={dec} span {result['pass_a']['ts_min_iso']}..{result['pass_a']['ts_max_iso']} ({result['pass_a']['seconds']}s)")

    # ---------- pass B: per-period host graph
    t0 = time.time()
    host_id: dict[str, int] = {}
    n_hosts = 0
    ec = [dict(edges=0, hosts=set(), self_loops=0, labels=Counter(), src_deg=Counter(),
               dst_deg=Counter(), pair_keys=set(), pairs_capped=False, sample=[]) for _ in range(periods)]
    dur = [dict(edges=0, hosts=set(), labels=Counter()) for _ in range(periods)]
    usecols = [DATE_COL, SRC_COL, DST_COL] + ([label_col] if label_col else [])
    dtype = {DATE_COL: "int64", SRC_COL: "str", DST_COL: "str"}
    if label_col:
        dtype[label_col] = "str"
    bounds_mid = ec_bounds[1:-1]
    dur_mid = dur_bounds[1:-1]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize, dtype=dtype):
        t = chunk[DATE_COL].to_numpy(np.int64)
        src = chunk[SRC_COL].to_numpy(dtype=object)
        dst = chunk[DST_COL].to_numpy(dtype=object)
        lab = chunk[label_col].to_numpy(dtype=object) if label_col else None
        p_ec = np.searchsorted(bounds_mid, t, side="right")
        p_dur = np.searchsorted(dur_mid, t, side="right")
        # global host-id map (small number of distinct hosts per chunk)
        allh = pd.unique(np.concatenate([src, dst]))
        for h in allh:
            if h not in host_id:
                host_id[h] = n_hosts
                n_hosts += 1
        sid = pd.Series(src).map(host_id).to_numpy(np.int64)
        did = pd.Series(dst).map(host_id).to_numpy(np.int64)
        for p in np.unique(p_ec):
            m = p_ec == p
            acc = ec[p]
            acc["edges"] += int(m.sum())
            s, d = src[m], dst[m]
            acc["hosts"].update(pd.unique(np.concatenate([s, d])).tolist())
            acc["self_loops"] += int((s == d).sum())
            a_s = sid[m]; a_d = did[m]
            u, c = np.unique(a_s, return_counts=True)
            for k, v in zip(u.tolist(), c.tolist()):
                acc["src_deg"][k] += v
            u, c = np.unique(a_d, return_counts=True)
            for k, v in zip(u.tolist(), c.tolist()):
                acc["dst_deg"][k] += v
            if lab is not None:
                acc["labels"].update(pd.Series(lab[m]).value_counts().to_dict())
            if not acc["pairs_capped"]:
                keys = (a_s.astype(np.uint64) << np.uint64(32)) | a_d.astype(np.uint64)
                acc["pair_keys"].update(np.unique(keys).tolist())
                if len(acc["pair_keys"]) > max_unique_pairs:
                    acc["pairs_capped"] = True
            # random edge sample for 2-hop feasibility (cap 3000/period)
            room = 3000 - len(acc["sample"])
            if room > 0 and len(s):
                take = min(room, max(1, len(s) // 50))
                ii = rng.integers(0, len(s), size=take)
                acc["sample"].extend(zip(s[ii].tolist(), d[ii].tolist()))
        for p in np.unique(p_dur):
            m = p_dur == p
            acc = dur[p]
            acc["edges"] += int(m.sum())
            acc["hosts"].update(pd.unique(np.concatenate([src[m], dst[m]])).tolist())
            if lab is not None:
                acc["labels"].update(pd.Series(lab[m]).value_counts().to_dict())
    print(f"[{ds}] pass B: {time.time()-t0:.1f}s, hosts={n_hosts}")

    # ---------- global graph from unique pairs
    global_keys: set[int] = set()
    capped = False
    for acc in ec:
        if not acc["pairs_capped"]:
            global_keys |= acc["pair_keys"]
        else:
            capped = True
    uf = UF(n_hosts)
    for key in global_keys:
        uf.union(int(key >> 32), int(key & 0xFFFFFFFF))
    comp_sizes = uf.component_sizes()
    ids = np.array(sorted(host_id.values()), dtype=np.int64)
    host_names = {v: k for k, v in host_id.items()}

    def degree_stats(deg: dict[int, int]) -> dict:
        vals = np.array(sorted(deg.values()), dtype=np.int64) if deg else np.array([], dtype=np.int64)
        if len(vals) == 0:
            return {"n_hosts": 0}
        return {"n_hosts": int(len(vals)), "edges_incident": int(vals.sum()),
                "mean": round(float(vals.mean()), 4), "median": pct(vals, 0.5),
                "p10": pct(vals, 0.10), "p90": pct(vals, 0.90),
                "max": int(vals[-1]), "min": int(vals[0]),
                "share_degree_1": round(float((vals == 1).mean()), 6),
                "share_degree_ge_2": round(float((vals >= 2).mean()), 6)}

    comp_lists = None
    per_period = []
    for p, acc in enumerate(ec):
        deg = Counter()
        for k, v in acc["src_deg"].items():
            deg[k] += v
        for k, v in acc["dst_deg"].items():
            deg[k] += v
        n_hosts_p = len(acc["hosts"])
        unique_pairs = len(acc["pair_keys"]) if not acc["pairs_capped"] else None
        edges = acc["edges"]
        recurrence = None
        if p > 0 and n_hosts_p:
            prev = ec[p - 1]["hosts"]
            recurrence = round(len(acc["hosts"] & prev) / n_hosts_p, 6)
        per_period.append({
            "period": p + 1, "edges": edges, "hosts": n_hosts_p,
            "self_loops": acc["self_loops"],
            "unique_host_pairs": unique_pairs,
            "repeated_pair_share_of_edges": round(1 - unique_pairs / edges, 6) if unique_pairs is not None and edges else None,
            "pairs_capped": acc["pairs_capped"],
            "label_counts": dict(acc["labels"].most_common()),
            "label_classes": len(acc["labels"]),
            "share_of_period_hosts_seen_in_prev": recurrence,
            "degree": degree_stats(deg),
        })
    dur_period = [{"period": p + 1, "edges": a["edges"], "hosts": len(a["hosts"]),
                   "label_counts": dict(a["labels"].most_common()), "label_classes": len(a["labels"])}
                  for p, a in enumerate(dur)]

    # global degree
    gdeg: Counter = Counter()
    g_hosts: set[int] = set()
    for acc in ec:
        g_hosts |= {host_id[h] for h in acc["hosts"]}
        for k, v in acc["src_deg"].items():
            gdeg[k] += v
        for k, v in acc["dst_deg"].items():
            gdeg[k] += v
    gi = per_period[0]["period"] if per_period else 1
    # adjacency from global unique pairs for 2-hop feasibility
    adj: dict[int, set[int]] = {}
    for key in global_keys:
        a, b = int(key >> 32), int(key & 0xFFFFFFFF)
        adj.setdefault(a, set()).add(b)
        adj.setdefault(b, set()).add(a)
    feas = []
    for p, acc in enumerate(ec):
        sizes = []
        for (s, d) in acc["sample"]:
            nb = adj.get(host_id[s], set()) | adj.get(host_id[d], set())
            nb.discard(host_id[s]); nb.discard(host_id[d])
            sizes.append(len(nb))
        if sizes:
            sizes.sort()
            feas.append({"period": p + 1, "sampled_edges": len(sizes),
                         "share_nonempty_2hop": round(float(np.mean([x > 0 for x in sizes])), 6),
                         "median_2hop_hosts": int(np.median(sizes)),
                         "p90_2hop_hosts": pct(np.array(sizes), 0.9),
                         "max_2hop_hosts": int(max(sizes))})
        else:
            feas.append({"period": p + 1, "sampled_edges": 0, "share_nonempty_2hop": None})
    global_stats = {
        "hosts": len(g_hosts), "edges": sum(a["edges"] for a in ec),
        "self_loops": sum(a["self_loops"] for a in ec),
        "unique_host_pairs": (len(global_keys) if not capped else None),
        "pairs_capped": capped,
        "repeated_pair_share_of_edges": (round(1 - len(global_keys) / sum(a["edges"] for a in ec), 6)
                                         if not capped and sum(a["edges"] for a in ec) else None),
        "n_components": len(comp_sizes),
        "giant_component_nodes": comp_sizes[0] if comp_sizes else 0,
        "giant_component_node_share": round(comp_sizes[0] / len(g_hosts), 6) if g_hosts and comp_sizes else None,
        "component_size_top5": comp_sizes[:5],
        "degree": degree_stats(gdeg),
    }
    # component host membership samples (names, first 8 per top component)
    roots = np.array([uf.find(i) for i in range(n_hosts)])
    root_to_members: dict[int, list[int]] = {}
    for i in range(n_hosts):
        root_to_members.setdefault(int(roots[i]), []).append(i)
    top = sorted(root_to_members.values(), key=len, reverse=True)[:5]
    global_stats["component_host_samples_top5"] = [
        sorted(host_names[i] for i in members)[:8] for members in top
    ]

    persistence = {"hosts_per_period": [len(a["hosts"]) for a in ec],
                   "consecutive_jaccard": [
                       {"pair": [p + 1, p + 2],
                        "jaccard": round(len(ec[p]["hosts"] & ec[p + 1]["hosts"]) / len(ec[p]["hosts"] | ec[p + 1]["hosts"]), 6),
                        "intersection": len(ec[p]["hosts"] & ec[p + 1]["hosts"])}
                       for p in range(periods - 1)],
                   "per_period_share_seen_in_previous": [
                       {"period": p + 2,
                        "share_of_period_hosts_seen_in_prev": round(len(ec[p + 1]["hosts"] & ec[p]["hosts"]) / len(ec[p + 1]["hosts"]), 6),
                        "shared_hosts": len(ec[p + 1]["hosts"] & ec[p]["hosts"]),
                        "period_hosts": len(ec[p + 1]["hosts"])}
                       for p in range(periods - 1)]}
    recurring = sum(1 for h in g_hosts if sum(1 for a in ec if host_id[host_names[h]] in {host_id[x] for x in a["hosts"]}) >= 2)
    persistence["hosts_recurring_ge2_periods"] = recurring
    persistence["fraction_hosts_recurring_ge2_periods"] = round(recurring / len(g_hosts), 6) if g_hosts else None

    criteria = {
        "median_host_degree_ge_2": {"value": global_stats["degree"].get("median"), "threshold": 2,
                                    "status": "pass" if (global_stats["degree"].get("median") or 0) >= 2 else "fail"},
        "giant_component_share_ge_0_90": {"value": global_stats.get("giant_component_node_share"), "threshold": 0.90,
                                          "status": "pass" if (global_stats.get("giant_component_node_share") or 0) >= 0.90 else "fail"},
        "host_recurrence_ge_0_30": {"value": min((x["share_of_period_hosts_seen_in_prev"] or 0)
                                                 for x in persistence["per_period_share_seen_in_previous"]) if persistence["per_period_share_seen_in_previous"] else None,
                                    "threshold": 0.30,
                                    "definition": "min over periods of share of period-t hosts seen in t-1",
                                    "status": "pass" if (persistence["per_period_share_seen_in_previous"] and
                                                          all((x["share_of_period_hosts_seen_in_prev"] or 0) >= 0.30
                                                              for x in persistence["per_period_share_seen_in_previous"])) else "fail"},
        "anchor_2hop_nonempty_ge_0_95": {"value": min((f["share_nonempty_2hop"] or 0) for f in feas) if feas else None,
                                         "threshold": 0.95,
                                         "status": "pass" if feas and all((f["share_nonempty_2hop"] or 0) >= 0.95 for f in feas) else "fail"},
        "repeated_host_pair_share_ge_0_05": {"value": global_stats.get("repeated_pair_share_of_edges"), "threshold": 0.05, "status": "info"},
    }
    result["period_stats_equal_count"] = per_period
    result["period_stats_equal_duration"] = dur_period
    result["global"] = global_stats
    result["host_persistence"] = persistence
    result["two_hop_feasibility"] = feas
    result["criteria"] = criteria
    fails = [k for k, v in criteria.items() if v["status"] == "fail"]
    result["structure_verdict"] = "pass" if not fails else "fail"
    result["n_fail"] = len(fails)
    result["elapsed_seconds"] = round(time.time() - t_start, 1)
    result["note"] = ("unique pairs/components computed from unique host-pair keys (multi-edges collapsed only for "
                      "component/2-hop computations; degree/repeated-share use edge counts)")
    out.mkdir(parents=True, exist_ok=True)
    (out / f"structure_hosts_{ds}.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"[{ds}] hosts={len(g_hosts)} edges={global_stats['edges']:,} comps={len(comp_sizes)} "
          f"giant={global_stats['giant_component_node_share']} median_deg={global_stats['degree'].get('median')} "
          f"verdict={result['structure_verdict'].upper()} fails={fails} elapsed={result['elapsed_seconds']}s")
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all")
    ap.add_argument("--raw", default=str(DEFAULT_RAW))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--periods", type=int, default=5)
    ap.add_argument("--chunksize", type=int, default=500_000)
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        try:
            audit(ds, raw, out, args.periods, args.chunksize)
        except FileNotFoundError as exc:
            print(f"SKIP {ds}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
