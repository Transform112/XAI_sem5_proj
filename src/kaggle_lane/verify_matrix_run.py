"""Verify a downloaded Kaggle matrix run (per-group S5 + lane/gate/lifecycle + manifests).

Usage:
    .venv\\Scripts\\python.exe kaggle_lane\\verify_matrix_run.py ^
        --run-dir incoming_rtx5060/kaggle_runs/t42-replay-rerun-<sha> ^
        --expected-groups 10
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def find_unpacked(run_dir: Path) -> Path:
    for candidate in (run_dir / "unpacked", run_dir / "kernel_output", run_dir):
        if (candidate / "results" / "tables").is_dir():
            return candidate
    raise SystemExit(f"no results/text/tables under {run_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-groups", type=int, default=None)
    parser.add_argument("--strategies", default=None, help="comma list of expected strategies (optional)")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    unpacked = find_unpacked(run_dir)
    tables = unpacked / "results" / "tables"
    matrix_root = tables / "matrix"
    audit_root = unpacked / "results" / "data_audit"
    group_dirs = sorted(path for path in matrix_root.glob("*") if path.is_dir()) if matrix_root.is_dir() else []
    expected_strategies = {item.strip() for item in (args.strategies or "").split(",") if item.strip()}

    findings: list[dict] = []
    groups: list[dict] = []
    for group_dir in group_dirs:
        group = group_dir.name
        strategy = group.split("_")[0]
        record: dict = {"group": group, "strategy": strategy, "problems": []}
        # S5 artifacts
        matrix_path = group_dir / "cl_matrix.json"
        metrics_path = group_dir / "cl_metrics.csv"
        if not matrix_path.is_file():
            record["problems"].append("missing cl_matrix.json")
        else:
            payload = json.loads(matrix_path.read_text(encoding="utf-8"))
            matrix = (payload.get("matrix") or {}).get("all_classes") or []
            finite = sum(1 for row in matrix for value in row if isinstance(value, (int, float)) and value == value)
            record["r_matrix_cells"] = f"{finite}/{len(matrix) ** 2}"
            record["dual_macro"] = payload.get("cl_metrics_all_classes") is not None
            if len(matrix) != 5 or finite != 25:
                record["problems"].append(f"R matrix incomplete {finite}/{len(matrix) ** 2}")
            if payload.get("lane") not in (None, "kaggle"):
                record["problems"].append(f"matrix lane={payload.get('lane')}")
        if not metrics_path.is_file():
            record["problems"].append("missing cl_metrics.csv")
        else:
            import csv

            with metrics_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            metrics = {str(row.get("metric")) for row in rows}
            record["metrics"] = sorted(metrics)
            if metrics != {"all_classes", "supported_set"}:
                record["problems"].append(f"cl_metrics metric rows={sorted(metrics)}")
        # train manifests
        manifests = sorted(audit_root.glob(f"train_manifest_{group}_period*.json"))
        record["train_manifests"] = len(manifests)
        lifecycles = []
        lanes = set()
        gates = set()
        epoch_times = []
        peak_memory = []
        for path in manifests:
            payload = json.loads(path.read_text(encoding="utf-8"))
            extra = payload.get("extra") or {}
            timing = extra.get("timing") or {}
            lane = str(payload.get("lane") or extra.get("lane") or "")
            gate = extra.get("gate") or {}
            lanes.add(lane)
            gates.add(bool(gate.get("real_lane")))
            started = payload.get("started_at_utc")
            finished = payload.get("finished_at_utc")
            if started and finished:
                lifecycles.append(str(started) < str(finished))
            epoch_times.append(timing.get("epoch_seconds_mean"))
            peak_memory.append(timing.get("max_memory_allocated_bytes"))
            if payload.get("status") != "success":
                record["problems"].append(f"{path.name}: status={payload.get('status')}")
            if lane != "kaggle":
                record["problems"].append(f"{path.name}: lane={lane}")
            if not gate.get("real_lane"):
                record["problems"].append(f"{path.name}: gate.real_lane not true")
        record["lane_values"] = sorted(lanes)
        record["gate_real_lane"] = sorted(gates)
        record["lifecycle_ok"] = all(lifecycles) if lifecycles else None
        record["epoch_seconds_p1"] = next((value for value in epoch_times if value is not None), None)
        record["peak_memory_bytes"] = max((value for value in peak_memory if value), default=None)
        if manifests and not all(lifecycles):
            record["problems"].append("manifest lifecycle not started<finished for all periods")
        if len(manifests) != 5:
            record["problems"].append(f"expected 5 period manifests, found {len(manifests)}")
        groups.append(record)
        findings.extend({"scope": group, "problem": problem} for problem in record["problems"])

    # per-group strict s5 validators
    validator_results = []
    for group_dir in group_dirs:
        completed = subprocess.run(
            [sys.executable, "-m", "src.analysis.validators", "--results-tables", str(group_dir), "--stage", "s5", "--strict"],
            cwd=str(REPO_ROOT), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300,
        )
        text = completed.stdout or ""
        checks = None
        for line in text.splitlines():
            if line.startswith("checks:"):
                try:
                    checks = json.loads(line.split(":", 1)[1].strip().replace("'", '"'))
                except Exception:
                    checks = None
        validator_results.append({"group": group_dir.name, "returncode": completed.returncode, "checks": checks, "overall": "pass" if "validators: PASS" in text else "fail"})
    validator_pass = sum(1 for item in validator_results if item["overall"] == "pass")

    summary = {
        "run_dir": str(run_dir.resolve()),
        "group_count": len(group_dirs),
        "expected_groups": args.expected_groups,
        "strategies": sorted({item["strategy"] for item in groups}),
        "expected_strategies": sorted(expected_strategies) or None,
        "groups": groups,
        "validators": {"pass": validator_pass, "total": len(validator_results), "results": validator_results},
        "problems": findings,
        "overall": "pass" if not findings and (args.expected_groups in (None, len(group_dirs))) and (not expected_strategies or {item["strategy"] for item in groups} <= expected_strategies) and validator_pass == len(validator_results) else "fail",
    }
    out = Path(args.out) if args.out else run_dir / "matrix_run_audit.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("group_count", "groups", "validators", "overall") if key != "groups"}, indent=2)[:2500])
    print("problems:", json.dumps(findings[:10], indent=1))
    print("written", out)
    return 0 if summary["overall"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
