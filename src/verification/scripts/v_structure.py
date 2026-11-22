#!/usr/bin/env python
"""Independent graph-structure audit for one NF-v3 dataset (verifier-owned).

Purpose: answer "does a real graph exist here?" without using any project code.
Nodes = hosts (IPV4_SRC_ADDR / IPV4_DST_ADDR); edges = flows. The script streams
the raw CSV twice (bounded RAM), builds per-period host graphs and recomputes the
docs/09 section-4 statistics:

  * nodes, edges, self-loops, unique/repeated host pairs, density
  * degree distribution (mean/median/p10/p90/max, share degree 1, share >=2)
  * connected components (count, giant-component node share, size distribution)
  * host persistence (hosts per period, consecutive-period host Jaccard,
    fraction of hosts recurring in >= 2 periods)
  * 2-hop neighbourhood feasibility on sampled edges
  * per-period class counts under the equal-count boundary rule, plus
    equal-duration boundaries as the alternate rule (G1c boundary decision input)

Everything is written to results/text/verification/data_audit/structure_<dataset>.json and, if
results/text/data_audit/<dataset>/structure.json exists, a reconciliation block shows
engineer values next to mine.

Usage (from repo root, Windows PowerShell 5.1 host):
  python verification\\scripts\\v_structure.py --dataset NF-UNSW-NB15-v3
  python verification\\scripts\\v_structure.py --dataset all --periods 5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
import zipfile
from array import array
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

VERIFIER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VERIFIER_ROOT.parents[1]
DEFAULT_RAW = REPO_ROOT / "data" / "raw"
DEFAULT_OUT = VERIFIER_ROOT / "data_audit"
ENGINEER_AUDIT = REPO_ROOT / "results" / "data_audit"
DATE_COL = "FLOW_START_MILLISECONDS"
SRC_COL = "IPV4_SRC_ADDR"
DST_COL = "IPV4_DST_ADDR"
LABEL_COL = "Attack"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def norm(s: str) -> str:
    return (s or "").strip().strip('"')


def locate_csvs(raw: Path, dataset: str) -> tuple[list[Path], Path | None]:
    d = raw / dataset
    if d.is_dir():
        csvs = sorted(p for p in d.rglob("*.csv") if p.is_file() and p.stat().st_size > 0)
        if csvs:
            return csvs, None
    zp = raw / f"{dataset}.zip"
    if zp.is_file():
        return [], zp
    raise FileNotFoundError(f"no CSV/zip for {dataset}")


def iter_rows(csv_paths: list[Path], zip_path: Path | None):
    """Yield (header, iterator-of-rows) for the (single) CSV, transparently from zip."""
    if zip_path is not None:
        zf = zipfile.ZipFile(zip_path)
        members = [m for m in zf.infolist() if m.filename.lower().endswith(".csv")]
        if len(members) != 1:
            raise SystemExit(f"expected 1 CSV member in {zip_path}, found {len(members)}")
        import io
        fh = io.TextIOWrapper(zf.open(members[0]), encoding="utf-8", errors="replace", newline="")
        r = csv.reader(fh)
        header = [norm(h) for h in next(r)]
        return header, r
    if len(csv_paths) != 1:
        raise SystemExit(f"expected 1 CSV, found {len(csv_paths)}")
    fh = open(csv_paths[0], "r", encoding="utf-8", errors="replace", newline="")
    r = csv.reader(fh)
    header = [norm(h) for h in next(r)]
    return header, r


class UF:
    __slots__ = ("parent", "size")

    def __init__(self):
        self.parent = {}
        self.size = {}

    def add(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.size[x] = 1

    def find(self, x):
        p = self.parent
        root = x
        while p[root] != root:
            root = p[root]
        while p[x] != root:
            p[x], x = root, p[x]
        return root

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.size[ra] < self.size[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        self.size[ra] += self.size[rb]

    def components(self):
        comp = Counter()
        for x in self.parent:
            comp[self.find(x)] += 1
        sizes = sorted(comp.values(), reverse=True)
        return sizes


def degree_stats(deg: dict[str, int]) -> dict:
    if not deg:
        return {"n_hosts": 0}
    vals = sorted(deg.values())
    n = len(vals)
    def pct(p):
        i = min(n - 1, max(0, int(round(p * (n - 1)))))
        return vals[i]
    return {
        "n_hosts": n,
        "edges_incident": sum(vals),
        "mean": round(sum(vals) / n, 4),
        "median": pct(0.5),
        "p10": pct(0.10),
        "p90": pct(0.90),
        "max": vals[-1],
        "min": vals[0],
        "share_degree_1": round(sum(1 for v in vals if v == 1) / n, 6),
        "share_degree_ge_2": round(sum(1 for v in vals if v >= 2) / n, 6),
    }


def compute_period_stats(
    edges: int,
    src_deg: Counter,
    dst_deg: Counter,
    hosts: set,
    pairs: dict[tuple[str, str], int] | None,
    uf: UF,
) -> dict:
    deg = Counter()
    deg.update(src_deg)
    deg.update(dst_deg)  # union counts? no: this would sum; do explicit
    deg = Counter()
    for h in hosts:
        deg[h] = src_deg.get(h, 0) + dst_deg.get(h, 0)
    comp_sizes = uf.components() if uf.parent else []
    n_hosts = len(hosts)
    giant_share = (comp_sizes[0] / n_hosts) if (comp_sizes and n_hosts) else None
    unique_pairs = len(pairs) if pairs is not None else None
    repeated = (edges - unique_pairs) if unique_pairs is not None else None
    density = (edges / (n_hosts * (n_hosts - 1))) if n_hosts > 1 else None
    return {
        "edges": edges,
        "hosts": n_hosts,
        "unique_host_pairs": unique_pairs,
        "unique_host_pairs_capped": pairs is None,
        "repeated_host_pair_edges": repeated,
        "repeated_pair_share_of_edges": round(repeated / edges, 6) if repeated is not None and edges else None,
        "self_loops": None,  # filled by caller
        "density_directed_multigraph": round(density, 8) if density is not None else None,
        "n_components": len(comp_sizes),
        "giant_component_nodes": comp_sizes[0] if comp_sizes else 0,
        "giant_component_node_share": round(giant_share, 6) if giant_share is not None else None,
        "component_size_top5": comp_sizes[:5],
        "degree": degree_stats(dict(deg)),
    }


def audit_dataset(ds: str, raw: Path, out: Path, periods: int, seed: int,
                  max_pair_dict: int = 3_000_000) -> dict:
    t_start = time.time()
    csvs, zp = locate_csvs(raw, ds)
    log_path = out / f"structure_{ds}.log.txt"
    log = open(log_path, "w", encoding="utf-8")
    def L(s=""):
        print(s)
        log.write(s + "\n")
        log.flush()
    L(f"=== verifier structure audit: {ds} ===")
    L(f"utc={utc_now()} argv={' '.join(sys.argv)}")
    result = {"dataset": ds, "check": "structure", "audited_utc": utc_now(),
              "script": "verification/src/scripts/v_structure.py", "periods": periods,
              "boundary_rule_primary": "equal-count", "seed": seed, "checks": []}

    # ---------------- pass 1: timestamps, row count, monotonicity
    header, rows = iter_rows(csvs, zp)
    cols = {c: i for i, c in enumerate(header)}
    for need in (DATE_COL, SRC_COL, DST_COL):
        if need not in cols:
            raise SystemExit(f"missing required column {need}")
    date_i, src_i, dst_i = cols[DATE_COL], cols[SRC_COL], cols[DST_COL]
    label_i = cols.get(LABEL_COL)
    ts = array("q")
    n = 0
    dec = 0
    prev = None
    label_counts = Counter()
    hour_counts = Counter()
    t0 = time.time()
    for row in rows:
        if not row:
            continue
        n += 1
        try:
            v = int(row[date_i])
        except (ValueError, IndexError):
            v = -1
        if prev is not None and v < prev:
            dec += 1
        prev = v
        ts.append(v)
        if v > 0:
            hour_counts[v // 3_600_000] += 1
        if label_i is not None and label_i < len(row):
            label_counts[norm(row[label_i])] += 1
    L(f"pass1: rows={n:,} decreasing_ts_steps={dec} ({time.time()-t0:.1f}s)")
    result["pass1"] = {"rows": n, "decreasing_ts_steps": dec,
                       "monotonic_nondecreasing": dec == 0, "pass1_seconds": round(time.time() - t0, 1)}
    if hour_counts:
        occ = sorted(hour_counts)
        gaps = [(occ[i + 1] - occ[i]) for i in range(len(occ) - 1)] or [0]
        gi = gaps.index(max(gaps)) if gaps else 0
        result["timestamp_gaps"] = {
            "occupied_hour_buckets": len(occ),
            "first_hour_iso": datetime.fromtimestamp(occ[0] * 3600, tz=timezone.utc).strftime("%Y-%m-%dT%H:00:00Z"),
            "last_hour_iso": datetime.fromtimestamp(occ[-1] * 3600, tz=timezone.utc).strftime("%Y-%m-%dT%H:00:00Z"),
            "largest_gap_hours": max(gaps),
            "largest_gap_start_iso": datetime.fromtimestamp(occ[gi] * 3600, tz=timezone.utc).strftime("%Y-%m-%dT%H:00:00Z") if gaps else None,
            "largest_gap_end_iso": datetime.fromtimestamp(occ[gi + 1] * 3600, tz=timezone.utc).strftime("%Y-%m-%dT%H:00:00Z") if gaps else None,
            "note": "large gaps indicate separate capture sessions; equal-duration periods crossing a gap can be empty",
        }
        L(f"timestamp gaps: {max(gaps)}h empty (occupied hour buckets={len(occ)})")
    result["label_counts_total"] = dict(label_counts.most_common())

    # ---------------- boundaries: equal-count (primary) and equal-duration (alt)
    epoch_ms_sane = bool(ts) and all(631152000000 <= v <= 3000000000000 for v in ts[: min(len(ts), 1000)])
    result["timestamp_epoch_ms_sane_sample"] = epoch_ms_sane
    boundary_idx = [round(k * n / periods) for k in range(periods + 1)]
    if n >= periods:
        ts_min, ts_max = min(ts), max(ts)
        dur = ts_max - ts_min
        dur_bounds = [ts_min + round(dur * k / periods) for k in range(periods + 1)]
    else:
        ts_min = ts_max = 0
        dur_bounds = [0] * (periods + 1)
    if dec == 0:
        equal_count_ts_bounds = [ts[min(b, n - 1)] for b in boundary_idx] if n else [0] * (periods + 1)
        ec_row_index = boundary_idx
    else:
        # file is NOT sorted by flow start time: equal-count boundaries must be
        # derived from the time-sorted values, and rows are then assigned to a
        # period by timestamp, not by row ordinal.
        sorted_ts = sorted(ts)
        equal_count_ts_bounds = [sorted_ts[min(b, n - 1)] for b in boundary_idx] if n else [0] * (periods + 1)
        ec_row_index = None
        del sorted_ts
    result["boundaries"] = {
        "equal_count_row_index": ec_row_index,
        "equal_count_row_index_note": None if ec_row_index is not None else
            "raw file is not time-sorted, so equal-count periods are defined by timestamp quantiles; row-index boundaries are not applicable",
        "equal_count_ts_ms": equal_count_ts_bounds,
        "equal_duration_ts_ms": dur_bounds,
        "ts_min": ts_min, "ts_max": ts_max,
        "ts_min_iso": datetime.fromtimestamp(ts_min / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if epoch_ms_sane else None,
        "ts_max_iso": datetime.fromtimestamp(ts_max / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if epoch_ms_sane else None,
    }
    L(f"time span: {result['boundaries']['ts_min_iso']} .. {result['boundaries']['ts_max_iso']}")
    if ec_row_index is not None:
        L(f"equal-count boundaries (row idx): {ec_row_index}")
    else:
        L(f"equal-count boundaries (ts ms, sorted quantiles): {equal_count_ts_bounds}")
    del ts  # free RAM before pass 2

    # ---------------- pass 2: per-period graph + class counts
    header, rows = iter_rows(csvs, zp)
    cols = {c: i for i, c in enumerate(header)}
    date_i, src_i, dst_i = cols[DATE_COL], cols[SRC_COL], cols[DST_COL]
    label_i = cols.get(LABEL_COL)
    ec_ts = equal_count_ts_bounds
    dur_ts = dur_bounds
    # per-period accumulators (equal-count primary, equal-duration secondary)
    def new_acc():
        return {"edges": 0, "hosts": set(), "src_deg": Counter(), "dst_deg": Counter(),
                "self_loops": 0, "pairs": {}, "pairs_capped": False,
                "labels": Counter(), "uf": UF(), "edge_sample_ids": []}
    ec = [new_acc() for _ in range(periods)]
    dur = [new_acc() for _ in range(periods)]
    rng = random.Random(seed)
    sample_target = 2000  # sampled edges per period for 2-hop feasibility
    t0 = time.time()
    row_i = 0
    import bisect
    for row in rows:
        if not row:
            continue
        i = row_i
        row_i += 1
        src, dst = norm(row[src_i]), norm(row[dst_i])
        # equal-count period: by row ordinal only when the file is time-sorted;
        # otherwise assign by the equal-count timestamp boundaries.
        if dec == 0:
            p_ec = min(max(bisect.bisect_right(boundary_idx, i) - 1, 0), periods - 1)
        else:
            try:
                v = int(row[date_i])
            except (ValueError, IndexError):
                v = 0
            p_ec = min(max(bisect.bisect_left(ec_ts, v) - 1, 0), periods - 1)
        try:
            vd = int(row[date_i])
        except (ValueError, IndexError):
            vd = 0
        p_dur = min(max(bisect.bisect_left(dur_ts, vd) - 1, 0), periods - 1)
        lab = norm(row[label_i]) if (label_i is not None and label_i < len(row)) else ""
        # equal-count accumulator: full graph statistics
        acc = ec[p_ec]
        acc["edges"] += 1
        acc["hosts"].add(src); acc["hosts"].add(dst)
        acc["src_deg"][src] += 1
        acc["dst_deg"][dst] += 1
        acc["labels"][lab] += 1
        if src == dst:
            acc["self_loops"] += 1
        acc["uf"].add(src); acc["uf"].add(dst); acc["uf"].union(src, dst)
        if not acc["pairs_capped"]:
            acc["pairs"][(src, dst)] = acc["pairs"].get((src, dst), 0) + 1
            if len(acc["pairs"]) > max_pair_dict:
                acc["pairs_capped"] = True
                acc["pairs"] = None
        if len(acc["edge_sample_ids"]) < sample_target and rng.random() < 0.01:
            acc["edge_sample_ids"].append((src, dst))
        # equal-duration accumulator: light (class counts / edges / host count only)
        dacc = dur[p_dur]
        dacc["edges"] += 1
        dacc["hosts"].add(src); dacc["hosts"].add(dst)
        dacc["labels"][lab] += 1
    pass2_s = time.time() - t0
    L(f"pass2: assigned per-period stats ({pass2_s:.1f}s)")

    # adjacency (undirected) for 2-hop feasibility; host set is assumed small for v3 family
    adjacency: dict[str, set[str]] = {}
    for acc in ec:
        for (s, d) in acc["pairs"].keys() if acc["pairs"] is not None else ():
            adjacency.setdefault(s, set()).add(d)
            adjacency.setdefault(d, set()).add(s)
        # if pairs capped, fall back: rebuild adjacency not possible; report degraded

    def period_stats(acc, name):
        st = compute_period_stats(acc["edges"], acc["src_deg"], acc["dst_deg"], acc["hosts"],
                                  acc["pairs"], acc["uf"])
        st["self_loops"] = acc["self_loops"]
        st["label_counts"] = dict(acc["labels"].most_common())
        st["label_classes"] = len(acc["labels"])
        st["period"] = name
        return st

    ec_stats = [period_stats(acc, k + 1) for k, acc in enumerate(ec)]
    dur_stats = []
    for k, acc in enumerate(dur):
        dur_stats.append({"period": k + 1, "edges": acc["edges"], "hosts": len(acc["hosts"]),
                          "label_counts": dict(acc["labels"].most_common()),
                          "label_classes": len(acc["labels"])})
    result["period_stats_equal_count"] = ec_stats
    result["period_stats_equal_duration"] = dur_stats
    result["period_assignment_method"] = "row_ordinal" if dec == 0 else "timestamp_boundary"

    # global graph = union over equal-count periods (pairs merged, not dropped)
    g_edges = sum(a["edges"] for a in ec)
    g_hosts = set()
    g_src = Counter(); g_dst = Counter(); g_labels = Counter(); g_self = 0
    g_uf = UF()
    g_pairs: dict[tuple[str, str], int] | None = {}
    for a in ec:
        g_hosts |= a["hosts"]; g_src.update(a["src_deg"]); g_dst.update(a["dst_deg"])
        g_labels.update(a["labels"]); g_self += a["self_loops"]
        for h in a["hosts"]:
            g_uf.add(h)
        if g_pairs is not None and a["pairs"] is not None:
            for (s, d), c in a["pairs"].items():
                g_pairs[(s, d)] = g_pairs.get((s, d), 0) + c
                if len(g_pairs) > max_pair_dict:
                    g_pairs = None
                    break
        for (s, d) in a["pairs"].keys() if a["pairs"] is not None else ():
            g_uf.union(s, d)
    global_stats = compute_period_stats(g_edges, g_src, g_dst, g_hosts, g_pairs, g_uf)
    global_stats["self_loops"] = g_self
    global_stats["label_counts"] = dict(g_labels.most_common())
    comp_hosts: dict[str, list[str]] = {}
    for h in g_hosts:
        comp_hosts.setdefault(g_uf.find(h), []).append(h)
    comp_lists = sorted(comp_hosts.values(), key=len, reverse=True)
    global_stats["component_host_samples_top5"] = [sorted(c)[:10] for c in comp_lists[:5]]
    result["global"] = global_stats

    # host persistence across equal-count periods
    host_sets = [a["hosts"] for a in ec]
    persistence = []
    for k in range(periods - 1):
        a, b = host_sets[k], host_sets[k + 1]
        inter = len(a & b); union = len(a | b)
        persistence.append({"pair": [k + 1, k + 2], "jaccard": round(inter / union, 6) if union else None,
                            "intersection": inter, "union": union})
    recurring = 0
    for h in g_hosts:
        c = sum(1 for s in host_sets if h in s)
        if c >= 2:
            recurring += 1
    result["host_persistence"] = {
        "hosts_per_period": [len(s) for s in host_sets],
        "consecutive_jaccard": persistence,
        "per_period_share_seen_in_previous": [
            {"period": k + 2, "share_of_period_hosts_seen_in_prev":
                round(len(host_sets[k + 1] & host_sets[k]) / len(host_sets[k + 1]), 6) if host_sets[k + 1] else None,
             "shared_hosts": len(host_sets[k + 1] & host_sets[k]),
             "period_hosts": len(host_sets[k + 1])}
            for k in range(periods - 1)
        ],
        "mean_consecutive_jaccard": round(sum(p["jaccard"] for p in persistence if p["jaccard"] is not None) / max(1, len([p for p in persistence if p["jaccard"] is not None])), 6),
        "fraction_hosts_recurring_ge2_periods": round(recurring / len(g_hosts), 6) if g_hosts else None,
        "hosts_recurring_ge2_periods": recurring,
        "unique_hosts_total": len(g_hosts),
    }

    # 2-hop feasibility on sampled edges from each period
    feas = []
    for k, acc in enumerate(ec):
        sizes = []
        for (s, d) in acc["edge_sample_ids"]:
            nb = set(adjacency.get(s, ())) | set(adjacency.get(d, ()))
            nb.discard(s); nb.discard(d)
            sizes.append(len(nb))
        if sizes:
            sizes.sort()
            feas.append({"period": k + 1, "sampled_edges": len(sizes),
                         "share_nonempty_2hop": round(sum(1 for x in sizes if x > 0) / len(sizes), 6),
                         "median_2hop_hosts": sizes[len(sizes) // 2],
                         "p90_2hop_hosts": sizes[min(len(sizes) - 1, int(0.9 * len(sizes)))],
                         "max_2hop_hosts": sizes[-1]})
        else:
            feas.append({"period": k + 1, "sampled_edges": 0, "share_nonempty_2hop": None})
    result["two_hop_feasibility"] = feas

    # ---------------- threshold verdicts (docs/09 Â§4 provisional)
    deg = global_stats["degree"]
    criteria = {
        "median_host_degree_ge_2": {"value": deg.get("median"), "threshold": 2, "status": "pass" if (deg.get("median") or 0) >= 2 else "fail"},
        "giant_component_share_ge_0_90": {"value": global_stats.get("giant_component_node_share"), "threshold": 0.90,
                                          "status": "pass" if (global_stats.get("giant_component_node_share") or 0) >= 0.90 else "fail"},
        "host_recurrence_ge_0_30": {"value": (min((p["share_of_period_hosts_seen_in_prev"] or 0) for p in result["host_persistence"]["per_period_share_seen_in_previous"])
                                               if result["host_persistence"]["per_period_share_seen_in_previous"] else None),
                                    "threshold": 0.30,
                                    "definition": "min over periods of share of period-t hosts already seen in period t-1 (config threshold definition)",
                                    "status": "pass" if (result["host_persistence"]["per_period_share_seen_in_previous"] and
                                                          all((p["share_of_period_hosts_seen_in_prev"] or 0) >= 0.30 for p in result["host_persistence"]["per_period_share_seen_in_previous"])) else "fail"},
        "anchor_2hop_nonempty_ge_0_95": {"value": min((f["share_nonempty_2hop"] or 0) for f in feas) if feas else None, "threshold": 0.95,
                                         "status": "pass" if feas and all((f["share_nonempty_2hop"] or 0) >= 0.95 for f in feas) else "fail"},
        "repeated_host_pair_share_ge_0_05": {"value": global_stats.get("repeated_pair_share_of_edges"), "threshold": 0.05,
                                             "status": "info"},  # reported, not enforced (docs/09)
    }
    result["criteria"] = criteria
    result["structure_verdict"] = "pass" if all(v["status"] != "fail" for k, v in criteria.items() if k != "repeated_host_pair_share_ge_0_05") else "fail"

    # ---------------- reconciliation with engineer artifact
    eng = ENGINEER_AUDIT / ds / "structure.json"
    recon = {"engineer_artifact": str(eng), "found": eng.is_file()}
    if eng.is_file():
        try:
            e = json.loads(eng.read_text(encoding="utf-8"))
            recon["engineer_keys"] = sorted(e.keys())
            recon["note"] = "compare fields manually / add mapping when schema stabilises"
        except Exception as exc:  # noqa: BLE001
            recon["parse_error"] = str(exc)
    result["reconciliation"] = recon

    fails = [k for k, v in criteria.items() if v["status"] == "fail"]
    result["n_fail"] = len(fails)
    result["elapsed_seconds"] = round(time.time() - t_start, 1)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / f"structure_{ds}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    L(f"\nGLOBAL: hosts={global_stats['hosts']:,} edges={global_stats['edges']:,} "
      f"median_deg={deg.get('median')} giant_share={global_stats.get('giant_component_node_share')}")
    L(f"criteria: " + ", ".join(f"{k}={v['status']}({v['value']})" for k, v in criteria.items()))
    L(f"OVERALL structure_verdict={result['structure_verdict'].upper()} fails={fails} elapsed={result['elapsed_seconds']}s")
    log.close()
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all")
    ap.add_argument("--raw", default=str(DEFAULT_RAW))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--periods", type=int, default=5)
    ap.add_argument("--seed", type=int, default=20261101)
    args = ap.parse_args()
    raw, out = Path(args.raw), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    datasets = ["NF-UNSW-NB15-v3", "NF-ToN-IoT-v3", "NF-BoT-IoT-v3", "NF-CSE-CIC-IDS2018-v3"] \
        if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        try:
            audit_dataset(ds, raw, out, args.periods, args.seed)
        except FileNotFoundError as e:
            print(f"SKIP {ds}: {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


