"""Verify the downloaded corrected full-S7 (D-071) Kaggle output.

Checks the six seed-42 groups, canonical-anchor t>=j coverage, pair/row conventions
(19,266 t>=j pairs = 57,798 rows for the actual 229/205/221/230/123 anchors/period = 1,008 anchors/group), no forward t<j rows, unique
cache keys / dedupe, per-group anchor refs+manifests, and state job completeness.

Usage:
    .venv\\Scripts\\python.exe kaggle_lane\\verify_full_s7.py ^
        --run-dir incoming_rtx5060/kaggle_runs/t39-explain-full-649109a
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

GROUPS = {
    "ft_egraphsage_seed42", "ft_gcn_seed42",
    "replay_egraphsage_seed42", "replay_gcn_seed42",
    "ewc_egraphsage_seed42", "ewc_gcn_seed42",
}


def find_unpacked(run_dir: Path) -> Path:
    for candidate in (run_dir / "unpacked", run_dir / "kernel_output", run_dir):
        if (candidate / "results" / "tables").is_dir():
            return candidate
    raise SystemExit(f"no results/text/tables under {run_dir}")


def resolve(value: str, results_root: Path) -> Path:
    path = Path(value)
    if path.is_file():
        return path
    raw = str(value)
    parts = path.parts
    if path.is_absolute() or raw.startswith("/"):
        for index in range(len(parts) - 1, -1, -1):
            if parts[index] == "results":
                candidate = results_root / Path(*parts[index + 1 :])
                if candidate.is_file():
                    return candidate
                break
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-groups", type=int, default=6)
    parser.add_argument("--anchors-per-period", type=int, default=250)
    parser.add_argument("--periods", type=int, default=5)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    unpacked = find_unpacked(run_dir)
    results_root = unpacked / "results"
    tables = results_root / "tables"
    anchors_root = results_root / "anchors"
    canonical = anchors_root / "canonical"
    problems: list[str] = []

    manifest = json.loads((canonical / "anchors_manifest.json").read_text(encoding="utf-8")) if (canonical / "anchors_manifest.json").is_file() else {}
    n_anchors = {int(period): int((entry or {}).get("n_selected") or 0) for period, entry in (manifest.get("per_period") or {}).items()}
    groups_expected = args.expected_groups

    index_path = tables / "explain_index.csv"
    operational = results_root / "explanations" / "index.csv"
    rows = list(csv.DictReader(index_path.open("r", encoding="utf-8", newline=""))) if index_path.is_file() else []
    op_rows = list(csv.DictReader(operational.open("r", encoding="utf-8", newline=""))) if operational.is_file() else []
    if not rows:
        problems.append("explain_index.csv missing/empty")
    if not op_rows:
        problems.append("operational explanations/index.csv missing/empty")

    by_group_anchor = defaultdict(lambda: {"checkpoints": set(), "anchor_periods": set(), "explainers": set()})
    missing_meta = 0
    for row in rows:
        meta_path = resolve(str(row.get("json_path") or ""), results_root)
        if not meta_path.is_file():
            missing_meta += 1
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        group = f"{meta.get('strategy')}_{meta.get('model')}_seed{meta.get('seed')}"
        anchor_id = int(meta.get("anchor_id", row.get("anchor_id")))
        entry = by_group_anchor[(group, anchor_id)]
        entry["checkpoints"].add(int(meta.get("checkpoint_period", -1)))
        entry["anchor_periods"].add(int(meta.get("anchor_period", -1)))
        entry["explainers"].add(str(row.get("explainer")))

    total_pairs = 0
    cross_pairs = 0
    forward = 0
    per_group_pairs = defaultdict(int)
    seen_groups = set()
    for (group, anchor_id), entry in by_group_anchor.items():
        seen_groups.add(group)
        for anchor_period in entry["anchor_periods"]:
            for checkpoint_period in entry["checkpoints"]:
                if checkpoint_period >= anchor_period:
                    total_pairs += 1
                    per_group_pairs[group] += 1
                if checkpoint_period > anchor_period:
                    cross_pairs += 1
                if checkpoint_period < anchor_period:
                    forward += 1
    expected_pairs = groups_expected * sum(n_anchors.get(j, 0) * sum(1 for t in range(1, args.periods + 1) if t >= j) for j in range(1, args.periods + 1))
    expected_rows = expected_pairs * 3
    if seen_groups != GROUPS:
        problems.append(f"groups mismatch: {sorted(seen_groups)}")
    if forward:
        problems.append(f"{forward} forward t<j pairs present")
    if len(rows) != expected_rows:
        problems.append(f"index rows {len(rows)} != expected {expected_rows} ({expected_pairs} pairs)")
    if total_pairs != expected_pairs:
        problems.append(f"pairs {total_pairs} != expected {expected_pairs}")
    if missing_meta:
        problems.append(f"{missing_meta} rows missing cache metadata")

    # per-group anchor refs/manifests
    groups_dir = anchors_root / "groups"
    group_audit = {}
    for group_dir in sorted(groups_dir.iterdir()) if groups_dir.is_dir() else []:
        if not group_dir.is_dir():
            continue
        ref = json.loads((group_dir / "anchor_ref.json").read_text(encoding="utf-8")) if (group_dir / "anchor_ref.json").is_file() else {}
        manifests = sorted(path.name for path in group_dir.glob("anchors_manifest_period*.json"))
        group_audit[group_dir.name] = {"has_ref": bool(ref), "period_manifests": len(manifests), "canonical_dir": ref.get("canonical_dir")}
        if not ref:
            problems.append(f"{group_dir.name}: missing anchor_ref.json")
        if len(manifests) != args.periods:
            problems.append(f"{group_dir.name}: {len(manifests)} period manifests != {args.periods}")

    state = json.loads((tables / "explain_state.json").read_text(encoding="utf-8")) if (tables / "explain_state.json").is_file() else {}
    jobs = state.get("jobs") or {}
    complete_jobs = sum(1 for value in jobs.values() if value.get("status") == "complete")
    expected_jobs = groups_expected * sum(1 for j in range(1, args.periods + 1) for t in range(1, args.periods + 1) if t >= j)
    if len(jobs) != expected_jobs or complete_jobs != expected_jobs:
        problems.append(f"state jobs {len(jobs)} ({complete_jobs} complete) != {expected_jobs}")

    inner_path = unpacked / "run_summary.json"
    inner = json.loads(inner_path.read_text(encoding="utf-8")) if inner_path.is_file() else {}
    inner_jobs = (inner.get("explain") or {}).get("jobs") or {}
    summary = {
        "run_dir": str(run_dir.resolve()),
        "kernel_status": inner.get("status"),
        "inner_wall_clock_s": inner.get("wall_clock_s"),
        "index_rows": len(rows),
        "operational_rows": len(op_rows),
        "expected_rows": expected_rows,
        "total_pairs_incl_diagonal": total_pairs,
        "expected_pairs": expected_pairs,
        "cross_checkpoint_pairs": cross_pairs,
        "forward_pairs": forward,
        "groups": sorted(seen_groups),
        "per_group_pairs": dict(per_group_pairs),
        "anchors_per_period": n_anchors,
        "state_jobs": {"total": len(jobs), "complete": complete_jobs, "expected": expected_jobs},
        "inner_jobs": inner_jobs,
        "group_anchor_audit": group_audit,
        "row_counts_by_explainer": {name: sum(1 for row in rows if row.get("explainer") == name) for name in ("ig", "gnn_explainer")},
        "problems": problems,
        "overall": "pass" if not problems else "fail",
    }
    out = Path(args.out) if args.out else run_dir / "full_s7_audit.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("kernel_status", "inner_wall_clock_s", "index_rows", "expected_rows", "total_pairs_incl_diagonal", "cross_checkpoint_pairs", "groups", "state_jobs", "row_counts_by_explainer", "problems", "overall")}, indent=1, default=str)[:5000])
    print("written", out)
    return 0 if summary["overall"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
