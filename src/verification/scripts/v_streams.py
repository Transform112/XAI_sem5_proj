#!/usr/bin/env python
"""Independent comparison of temporal stream constructions for NF-UNSW-NB15-v3.

Decision input for G1c / D-020. Uses only the raw CSV (no project code).

Variants:
  A   equal-count 5 periods over the whole file (current default)
  B   session-aware 2+3: session 1 -> 2 consecutive equal-count periods,
      session 2 -> 3 consecutive equal-count periods (T=5, never crossing the
      capture gap)
  C1  session-aware 1+4
  C2  session-aware 3+2
  C3  session-aware equal-duration within sessions (each session split into a
      number of periods proportional to its flow count, but by time span)

Per period and variant this reports: row count, time range, class histogram (all
classes), host count, share of hosts seen in the previous period, consecutive
host-set Jaccard, and the class counts in a simulated 70/10/20 within-period
split (seeded) so the "attack class with >=50 test edges in every period" test
can be evaluated before the real masks exist. Sessions are detected from
timestamp gaps (>= --gap-hours).

Output: results/text/verification/data_audit/stream_alternatives.json (+ .md note written by
the same run).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

VERIFIER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VERIFIER_ROOT.parents[1]
RAW = REPO_ROOT / "data" / "raw" / "NF-UNSW-NB15-v3" / "NF-UNSW-NB15-v3.csv"
OUT = VERIFIER_ROOT / "data_audit"
DATE_COL = "FLOW_START_MILLISECONDS"
SRC_COL = "IPV4_SRC_ADDR"
DST_COL = "IPV4_DST_ADDR"
LABEL_COL = "Attack"
SEED = 42
TEST_MIN = 50


def iso(ms: int) -> str:
    return datetime.fromtimestamp(int(ms) / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def period_stats(ts: np.ndarray, src: np.ndarray, dst: np.ndarray, lab: np.ndarray,
                 rng: np.random.Generator) -> dict:
    n = len(ts)
    classes, counts = np.unique(lab, return_counts=True)
    # simulated 70/10/20 split (random, seeded) to estimate test-mask class counts
    perm = rng.permutation(n)
    n_test = int(round(0.20 * n))
    test_idx = perm[:n_test]
    tc, tcnt = np.unique(lab[test_idx], return_counts=True)
    test_counts = {str(k): int(v) for k, v in zip(tc, tcnt)}
    hosts = set(src.tolist()) | set(dst.tolist())
    return {
        "rows": int(n),
        "ts_min": int(ts.min()), "ts_max": int(ts.max()),
        "ts_min_iso": iso(ts.min()), "ts_max_iso": iso(ts.max()),
        "duration_hours": round((int(ts.max()) - int(ts.min())) / 3_600_000, 2),
        "class_histogram": {str(k): int(v) for k, v in zip(classes, counts)},
        "n_attack_classes": int(len(classes) - (1 if "Benign" in classes else 0)),
        "hosts": len(hosts),
        "host_set": hosts,
        "simulated_test_counts": test_counts,
        "classes_with_ge50_test": sorted([c for c, v in test_counts.items() if v >= TEST_MIN and c != "Benign"]),
        "classes_with_ge250_total": sorted([str(c) for c, v in zip(classes, counts) if v >= 250 and c != "Benign"]),
    }


def boundaries(n: int, k: int) -> list[int]:
    return [round(i * n / k) for i in range(k + 1)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-hours", type=float, default=6.0)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    df = pd.read_csv(RAW, usecols=[DATE_COL, SRC_COL, DST_COL, LABEL_COL],
                     dtype={SRC_COL: "str", DST_COL: "str", LABEL_COL: "str"})
    df = df.sort_values(DATE_COL, kind="mergesort").reset_index(drop=True)
    ts_all = df[DATE_COL].to_numpy(np.int64)
    # sessions from timestamp gaps
    gaps = np.diff(ts_all)
    gap_ms = int(args.gap_hours * 3600 * 1000)
    cut = np.where(gaps >= gap_ms)[0] + 1
    session_starts = [0] + cut.tolist()
    session_ends = cut.tolist() + [len(ts_all)]
    sessions = [{"session": i + 1, "start_row": int(s), "end_row": int(e), "rows": int(e - s),
                 "ts_min_iso": iso(ts_all[s]), "ts_max_iso": iso(ts_all[e - 1])}
                for i, (s, e) in enumerate(zip(session_starts, session_ends))]
    result: dict = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "script": "verification/src/scripts/v_streams.py",
        "raw_file": str(RAW), "rows": int(len(df)),
        "gap_threshold_hours": args.gap_hours,
        "sessions": sessions,
        "split_simulation": {"fractions": {"train": 0.70, "val": 0.10, "test": 0.20}, "seed": SEED,
                             "note": "random within-period split proxy; real masks are group-aware (near-duplicate) and may shift a few counts"},
        "variants": {},
    }
    rng = np.random.default_rng(SEED)

    def build_from_boundaries(name: str, desc: str, row_bounds: list[int]) -> None:
        periods = []
        prev_hosts = None
        for k in range(len(row_bounds) - 1):
            a, b = row_bounds[k], row_bounds[k + 1]
            sub = df.iloc[a:b]
            st = period_stats(sub[DATE_COL].to_numpy(np.int64),
                              sub[SRC_COL].to_numpy(dtype=object),
                              sub[DST_COL].to_numpy(dtype=object),
                              sub[LABEL_COL].to_numpy(dtype=object), rng)
            hosts = st.pop("host_set")
            st["period"] = k + 1
            if prev_hosts is not None:
                st["share_of_period_hosts_seen_in_prev"] = round(len(hosts & prev_hosts) / len(hosts), 6) if hosts else None
                st["consecutive_jaccard"] = round(len(hosts & prev_hosts) / len(hosts | prev_hosts), 6)
            else:
                st["share_of_period_hosts_seen_in_prev"] = None
                st["consecutive_jaccard"] = None
            prev_hosts = hosts
            periods.append(st)
        common_ge50 = None
        for p in periods:
            s = set(p["classes_with_ge50_test"])
            common_ge50 = s if common_ge50 is None else (common_ge50 & s)
        result["variants"][name] = {"description": desc, "boundaries_row_index": row_bounds,
                                    "periods": periods,
                                    "attack_classes_with_ge50_test_in_every_period": sorted(common_ge50 or []),
                                    "n_such_classes": len(common_ge50 or [])}

    # A: whole-file equal count 5
    b = boundaries(len(df), 5)
    build_from_boundaries("A_equal_count_whole", "equal-count 5 periods over the whole file (current default)", b)
    # session-aware allocations
    s1, s2 = sessions[0], sessions[1]
    for name, alloc, desc in (
        ("B_session_2_3", [2, 3], "within-session equal count: session1 -> 2 periods, session2 -> 3 periods"),
        ("C1_session_1_4", [1, 4], "within-session equal count: session1 -> 1 period, session2 -> 4 periods"),
        ("C2_session_3_2", [3, 2], "within-session equal count: session1 -> 3 periods, session2 -> 2 periods"),
    ):
        bounds = [s1["start_row"]]
        for ses, k in zip(sessions, alloc):
            if k == 0:
                continue
            local = [round(i * ses["rows"] / k) + ses["start_row"] for i in range(k + 1)]
            bounds += local[1:]
        bounds[-1] = len(df)
        build_from_boundaries(name, desc, bounds)
    # C3: equal-duration within sessions (time-span split proportional allocation 2+3)
    bounds = [s1["start_row"]]
    for ses, k in zip(sessions, [2, 3]):
        t0 = int(ts_all[ses["start_row"]]); t1 = int(ts_all[ses["end_row"] - 1])
        for i in range(1, k + 1):
            target = t0 + round((t1 - t0) * i / k)
            row = int(np.searchsorted(ts_all[ses["start_row"]:ses["end_row"]], target, side="left")) + ses["start_row"]
            bounds.append(min(max(row, bounds[-1]), ses["end_row"]))
    bounds[-1] = len(df)
    build_from_boundaries("C3_session_equal_duration_2_3",
                          "within-session equal DURATION: session1 -> 2, session2 -> 3", bounds)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "stream_alternatives.json"
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    # compact markdown note
    lines = ["# Stream constructions for NF-UNSW-NB15-v3 (independent, raw timestamps)", "",
             f"Generated: {result['generated_utc']}; source: `{RAW.name}`; rows {len(df):,}", "",
             "Sessions detected from timestamp gaps >= " + str(args.gap_hours) + " h:", "",
             "| session | rows | start | end |", "| --- | --- | --- | --- |"]
    for s in sessions:
        lines.append(f"| {s['session']} | {s['rows']:,} | {s['ts_min_iso']} | {s['ts_max_iso']} |")
    lines += ["", "70/10/20 simulation: random within-period split (seed 42) as a proxy for the real group-aware masks.", ""]
    for name, v in result["variants"].items():
        lines += [f"## Variant {name}", "", v["description"], "",
                  "| period | rows | time range | classes | attack classes | hosts | seen-in-prev | classes with >=50 test edges |",
                  "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for p in v["periods"]:
            lines.append(f"| {p['period']} | {p['rows']:,} | {p['ts_min_iso'][:16]}..{p['ts_max_iso'][11:16]} | "
                         f"{len(p['class_histogram'])} | {p['n_attack_classes']} | {p['hosts']} | "
                         f"{p['share_of_period_hosts_seen_in_prev']} | {', '.join(p['classes_with_ge50_test']) or '—'} |")
        lines += ["", f"Attack classes with >=50 test edges in EVERY period: {', '.join(v['attack_classes_with_ge50_test_in_every_period']) or 'NONE'}", ""]
    (out_dir / "stream_alternatives.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {path} and stream_alternatives.md")
    for name, v in result["variants"].items():
        print(f"{name}: periods={[p['rows'] for p in v['periods']]} ge50={v['attack_classes_with_ge50_test_in_every_period']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
