"""Lead-side audit of the full 30-group Kaggle matrix (T37/D-068).

Reads the unpacked matrix output, validates per-group S5 artifacts with the
strict validators, checks manifests/lifecycle/lane and builds the aggregate CL
metrics table used for the headline results. Evidence is written under
results/text/verification/data_audit/kaggle_matrix_full/.
"""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

RUN_DIR = Path(r"incoming_rtx5060/kaggle_runs/t37-matrix-full-90e7635")
UNPACKED = RUN_DIR / "unpacked"
OUT = Path(r"results/text/verification/data_audit/kaggle_matrix_full")
MATRIX = UNPACKED / "results" / "tables" / "matrix"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    groups = sorted(p for p in MATRIX.iterdir() if p.is_dir())
    summary = {"groups": [], "n_groups": len(groups)}
    metrics_rows: list[dict] = []
    validator_totals = {"pass": 0, "warn": 0, "fail": 0}
    for group in groups:
        name = group.name
        strategy, model, seed = name.rsplit("_", 2)
        seed = int(seed.replace("seed", ""))
        cl_matrix = group / "cl_matrix.json"
        cl_metrics = group / "cl_metrics.csv"
        entry = {"group": name, "strategy": strategy, "model": model, "seed": seed}
        # strict S5 validators per group
        proc = subprocess.run(
            [sys.executable, "-m", "src.analysis.validators", "--results-tables", str(group), "--stage", "s5", "--strict"],
            capture_output=True,
            text=True,
        )
        entry["validators_returncode"] = proc.returncode
        tail = (proc.stdout or proc.stderr or "").strip().splitlines()
        counts = {}
        for line in tail[-3:]:
            if line.strip().startswith("checks:"):
                try:
                    counts = eval(line.split("checks:", 1)[1].strip())  # noqa: S307 - local audit tool
                except Exception:
                    counts = {}
        entry["validators"] = counts
        for key in validator_totals:
            validator_totals[key] += int(counts.get(key, 0))
                # R completeness (matrix JSON stores a dict of matrices: all_classes/supported_set)
        if cl_matrix.is_file():
            payload = json.loads(cl_matrix.read_text(encoding="utf-8"))
            matrices = payload.get("matrix") or {}
            complete = True
            stats = {}
            if isinstance(matrices, dict):
                for name, matrix in matrices.items():
                    if not isinstance(matrix, list):
                        continue
                    n = len(matrix)
                    cells = 0
                    finite = 0
                    for i, row in enumerate(matrix):
                        for j, value in enumerate(row):
                            if j <= i:
                                cells += 1
                                if value is not None:
                                    finite += 1
                    stats[name] = {"rows": n, "lower_cells": cells, "finite_cells": finite}
                    complete = complete and cells > 0 and finite == cells
            entry.update({"matrix_stats": stats, "complete": complete})        # CL metrics rows
        if cl_metrics.is_file():
            with cl_metrics.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    row["group"] = name
                    row["strategy"] = strategy
                    row["model"] = model
                    row["seed"] = str(seed)
                    metrics_rows.append(row)
        summary["groups"].append(entry)

    # manifest / lifecycle / lane audit
    audit_dir = UNPACKED / "results" / "data_audit"
    manifests = sorted(audit_dir.glob("train_manifest_*.json"))
    lifecycle = []
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        lifecycle.append(
            {
                "manifest": path.name,
                "lane": payload.get("lane"),
                "lane_source": payload.get("lane_source"),
                "gate_real_lane": (payload.get("gate") or {}).get("real_lane"),
                "status": payload.get("status"),
                "duration_s": payload.get("duration_s"),
                "started_at_utc": payload.get("started_at_utc"),
                "finished_at_utc": payload.get("finished_at_utc"),
                "epoch_seconds_mean": ((payload.get("extra") or {}).get("timing") or {}).get("epoch_seconds_mean"),
                "max_memory_allocated_bytes": ((payload.get("extra") or {}).get("timing") or {}).get("max_memory_allocated_bytes"),
                "batching": ((payload.get("extra") or {}).get("timing") or {}).get("batching"),
            }
        )
    summary["manifests"] = lifecycle
    summary["n_manifests"] = len(lifecycle)
    summary["all_lane_kaggle"] = bool(lifecycle) and all(item["lane"] == "kaggle" for item in lifecycle)
    summary["all_gate_real"] = bool(lifecycle) and all(bool(item["gate_real_lane"]) for item in lifecycle)
    summary["all_status_success"] = bool(lifecycle) and all(item["status"] == "success" for item in lifecycle)
    summary["all_durations_positive"] = bool(lifecycle) and all(
        item["duration_s"] is not None and float(item["duration_s"]) > 0 for item in lifecycle
    )
    summary["epoch_seconds_mean_p1"] = sorted(
        round(float(item["epoch_seconds_mean"]), 4)
        for item in lifecycle
        if item["epoch_seconds_mean"] is not None and (item["manifest"] or "").endswith("period1.json")
    )
    summary["validator_totals"] = validator_totals
    summary["n_groups_complete"] = sum(1 for entry in summary["groups"] if entry.get("complete"))
    (OUT / "matrix_full_audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT.parent / "kaggle_matrix_full_metrics.csv").write_text(
        _csv_text(metrics_rows), encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in ("n_groups", "n_groups_complete", "n_manifests", "all_lane_kaggle", "all_gate_real", "all_status_success", "all_durations_positive", "validator_totals")}, indent=2))
    return 0


def _csv_text(rows: list[dict]) -> str:
    if not rows:
        return ""
    fieldnames = list(rows[0].keys())
    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


if __name__ == "__main__":
    raise SystemExit(main())


