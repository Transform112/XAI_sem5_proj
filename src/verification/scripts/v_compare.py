#!/usr/bin/env python
"""All-dataset comparison table for the G1c decision (verifier-owned).

Reads the independent artifacts only:
  results/text/verification/data_audit/identity_<ds>.json         (row/class/identity)
  results/text/verification/data_audit/structure_hosts_<ds>.json  (host graph + periods)
and evaluates the pre-registered D-025 / docs/08 section 10 replacement-primary
screen R2 (host-richness / neighbourhood variation) exactly as written:
  >=500 hosts; giant-component >=0.90; per-period host recurrence (share seen in
  previous period) >=0.10; consecutive-period host Jaccard >=0.05; median 2-hop
  host count in [10,300]; no benign-only periods or a policy leaving >=5
  evaluable classes (evaluable proxy used here: class has >=250 total edges in a
  period, i.e. expected >=50 test edges at a 20% test fraction).

Output: results/text/verification/data_audit/dataset_comparison.{json,md}
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

VERIFIER_ROOT = Path(__file__).resolve().parents[1]
OUT = VERIFIER_ROOT / "data_audit"
DATASETS = ["NF-UNSW-NB15-v3", "NF-ToN-IoT-v3", "NF-BoT-IoT-v3", "NF-CSE-CIC-IDS2018-v3"]
R2 = {"hosts_min": 500, "giant_min": 0.90, "recurrence_min": 0.10,
      "jaccard_min": 0.05, "two_hop_median_range": [10, 300], "min_evaluable_classes": 5,
      "evaluable_test_edges_min": 50, "test_fraction": 0.20}


def load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception:  # noqa: BLE001
        return None


def evaluable_classes(period: dict) -> int:
    """Classes with expected test edges >= 50 (count >= 250 at 20% test)."""
    hist = period.get("label_counts") or {}
    return sum(1 for k, v in hist.items() if k != "Benign" and v >= R2["evaluable_test_edges_min"] / R2["test_fraction"])


def main() -> int:
    rows = []
    for ds in DATASETS:
        ident = load(OUT / f"identity_{ds}.json")
        struct = load(OUT / f"structure_hosts_{ds}.json")
        rec: dict = {"dataset": ds, "identity_present": ident is not None, "structure_present": struct is not None}
        if ident:
            scans = (ident.get("scans") or [{}])[0]
            rec["identity"] = {
                "rows": ident.get("total_data_rows"),
                "published_rows": (ident.get("published_reference") or {}).get("paper_total_flows"),
                "overall": ident.get("overall"),
                "checks": {c["id"]: c["status"] for c in ident.get("checks", [])},
                "hosts_union": (scans.get("host_stats") or {}).get("unique_hosts_union"),
                "scan_light": ident.get("scan_light"),
                "time_sorted": (scans.get("timestamp") or {}).get("decreasing_steps") == 0,
                "ts_min_iso": (scans.get("timestamp") or {}).get("min_iso"),
                "ts_max_iso": (scans.get("timestamp") or {}).get("max_iso"),
                "class_counts": (ident.get("label_counts_merged") or {}).get("Attack"),
            }
        if struct:
            g = struct.get("global") or {}
            per = struct.get("period_stats_equal_count") or []
            pers = struct.get("host_persistence") or {}
            feas = struct.get("two_hop_feasibility") or []
            # normalise the lean (v_hosts_lean) schema to the v_hosts schema
            if "per_period_share_seen_in_previous" not in pers and "share_seen_prev" in pers:
                pers["per_period_share_seen_in_previous"] = [
                    {"share_of_period_hosts_seen_in_prev": v} for v in pers["share_seen_prev"] if v is not None]
            if "consecutive_jaccard" in pers and pers["consecutive_jaccard"] and all(v is None or isinstance(v, (int, float)) for v in pers["consecutive_jaccard"]):
                pers["consecutive_jaccard"] = [{"jaccard": v} for v in pers["consecutive_jaccard"] if v is not None]
            if not feas:
                feas = [{"period": q.get("period"), "median_2hop_hosts": (q.get("two_hop") or {}).get("median_hosts"),
                         "share_nonempty_2hop": (q.get("two_hop") or {}).get("share_nonempty")}
                        for q in (struct.get("period_stats_equal_count") or []) if q.get("two_hop")]
            rec["structure"] = {
                "hosts": g.get("hosts"), "edges": g.get("edges"),
                "components": g.get("n_components"),
                "component_sizes_top5": g.get("component_size_top5"),
                "giant_component_node_share": g.get("giant_component_node_share"),
                "median_degree": (g.get("degree") or {}).get("median"),
                "degree_p10": (g.get("degree") or {}).get("p10"),
                "degree_p90": (g.get("degree") or {}).get("p90"),
                "degree_max": (g.get("degree") or {}).get("max"),
                "unique_host_pairs": g.get("unique_host_pairs"),
                "repeated_pair_share_of_edges": g.get("repeated_pair_share_of_edges"),
                "self_loops": g.get("self_loops"),
                "hosts_per_period": pers.get("hosts_per_period"),
                "recurrence_min_seen_prev": min([x.get("share_of_period_hosts_seen_in_prev") or 0
                                                 for x in pers.get("per_period_share_seen_in_previous") or []] or [0]),
                "jaccard_min": min([x.get("jaccard") or 0 for x in pers.get("consecutive_jaccard") or [] if isinstance(x, dict)] or [0]),
                "two_hop_nonempty_min": min([f.get("share_nonempty_2hop") or 0 for f in feas] or [0]),
                "two_hop_median_hosts": sorted([f.get("median_2hop_hosts") for f in feas if f.get("median_2hop_hosts") is not None]),
                "periods": [{"period": p.get("period"), "rows": p.get("edges"), "hosts": p.get("hosts"),
                             "classes": p.get("label_classes"),
                             "class_histogram": p.get("label_counts"),
                             "benign_only": (p.get("label_classes") == 1),
                             "evaluable_attack_classes": evaluable_classes(p)}
                            for p in per],
                "structure_verdict": struct.get("structure_verdict"),
            }
            # R2 screen
            st = rec["structure"]
            fails = []
            if (st["hosts"] or 0) < R2["hosts_min"]:
                fails.append(f"hosts {st['hosts']} < {R2['hosts_min']}")
            if (st["giant_component_node_share"] or 0) < R2["giant_min"]:
                fails.append(f"giant {st['giant_component_node_share']} < {R2['giant_min']}")
            if (st["recurrence_min_seen_prev"] or 0) < R2["recurrence_min"]:
                fails.append(f"recurrence {st['recurrence_min_seen_prev']} < {R2['recurrence_min']} (D-023 flag threshold)")
            if (st["jaccard_min"] or 0) < R2["jaccard_min"]:
                fails.append(f"jaccard {st['jaccard_min']} < {R2['jaccard_min']}")
            if st["two_hop_median_hosts"]:
                lo = min(st["two_hop_median_hosts"]); hi = max(st["two_hop_median_hosts"])
                if lo < R2["two_hop_median_range"][0] or hi > R2["two_hop_median_range"][1]:
                    fails.append(f"median 2-hop hosts range [{lo},{hi}] outside {R2['two_hop_median_range']}")
            benign_only = [p["period"] for p in st["periods"] if p["benign_only"]]
            if benign_only:
                fails.append(f"benign-only period(s) {benign_only}")
            min_eval = min([p["evaluable_attack_classes"] for p in st["periods"]] or [0])
            if benign_only and min_eval < R2["min_evaluable_classes"]:
                fails.append(f"only {min_eval} evaluable attack classes in some period")
            rec["R2_screen"] = {"pass": not fails, "fails": fails,
                                "min_evaluable_attack_classes": min_eval}
        if not rec.get("structure_present") or not rec.get("identity_present"):
            rec["R2_screen"] = {"pass": None, "fails": [], "pending": True,
                                "reason": "identity/structure evidence not present yet"}
        rows.append(rec)
    out = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "rule_source": "docs/08 section 10 (D-025) R2, thresholds as pre-registered",
           "thresholds": R2, "datasets": rows}
    (OUT / "dataset_comparison.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    # markdown table
    lines = ["# All-four dataset comparison (G1c evidence core, verifier-independent)", "",
             f"Generated: {out['generated_utc']}; rule: docs/08 Ã‚Â§10 R2 (D-025).", "",
             "| metric | UNSW | ToN-IoT | BoT-IoT | CSE-CIC-IDS2018 |",
             "| --- | --- | --- | --- | --- |"]
    def cell(fn):
        vals = []
        for r in rows:
            try:
                vals.append(str(fn(r)))
            except Exception:  # noqa: BLE001
                vals.append("Ã¢â‚¬â€")
        return vals
    metric_rows = [
        ("rows (independent recount)", lambda r: r.get("identity", {}).get("rows")),
        ("published rows (paper)", lambda r: r.get("identity", {}).get("published_rows")),
        ("identity verdict", lambda r: r.get("identity", {}).get("overall")),
        ("hosts (union)", lambda r: r.get("structure", {}).get("hosts")),
        ("edges", lambda r: r.get("structure", {}).get("edges")),
        ("time-sorted raw file", lambda r: r.get("identity", {}).get("time_sorted")),
        ("time span", lambda r: f"{r.get('identity', {}).get('ts_min_iso')} .. {r.get('identity', {}).get('ts_max_iso')}"),
        ("components / sizes", lambda r: f"{r.get('structure', {}).get('components')} / {r.get('structure', {}).get('component_sizes_top5')}"),
        ("giant component share", lambda r: r.get("structure", {}).get("giant_component_node_share")),
        ("median host degree", lambda r: r.get("structure", {}).get("median_degree")),
        ("degree p10/p90/max", lambda r: f"{r.get('structure', {}).get('degree_p10')}/{r.get('structure', {}).get('degree_p90')}/{r.get('structure', {}).get('degree_max')}"),
        ("unique directed host pairs", lambda r: r.get("structure", {}).get("unique_host_pairs")),
        ("repeated-pair share of edges", lambda r: r.get("structure", {}).get("repeated_pair_share_of_edges")),
        ("self-loops", lambda r: r.get("structure", {}).get("self_loops")),
        ("hosts per period", lambda r: r.get("structure", {}).get("hosts_per_period")),
        ("recurrence min (share seen prev)", lambda r: r.get("structure", {}).get("recurrence_min_seen_prev")),
        ("consecutive host Jaccard min", lambda r: r.get("structure", {}).get("jaccard_min")),
        ("2-hop non-empty min", lambda r: r.get("structure", {}).get("two_hop_nonempty_min")),
        ("median 2-hop hosts (sorted periods)", lambda r: r.get("structure", {}).get("two_hop_median_hosts")),
        ("benign-only periods", lambda r: [p["period"] for p in r.get("structure", {}).get("periods", []) if p["benign_only"]] or "none"),
        ("min evaluable attack classes/period", lambda r: r.get("R2_screen", {}).get("min_evaluable_attack_classes")),
        ("**R2 replacement-primary screen**", lambda r: ("PENDING" if r.get("R2_screen", {}).get("pass") is None else ("PASS" if r.get("R2_screen", {}).get("pass") else "FAIL: " + "; ".join(r.get("R2_screen", {}).get("fails", []))))),
    ]
    for name, fn in metric_rows:
        lines.append(f"| {name} | " + " | ".join(cell(fn)) + " |")
    (OUT / "dataset_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



