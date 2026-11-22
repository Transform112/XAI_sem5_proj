"""Verify a downloaded Kaggle noise-floor stage output (F4 smoke + full noise stage).

Checks: both cache channels + combined null exist, cache schema columns/unique keys,
expected row counts for the configured anchors/periods/repeats/reseeds/explainers,
entry npz/json existence + finite arrays (relocated from /kaggle paths), IG-repeat canary,
and per-period noise manifests/summaries.

Usage:
    .venv\\Scripts\\python.exe kaggle_lane\\verify_noise_stage.py ^
        --run-dir incoming_rtx5060/kaggle_runs/t42-noise-smoke-649109a ^
        --anchors-per-period 100 --periods 5 --repeats 10 --null-reseeds 5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_COLUMNS = (
    "key", "checkpoint_sha", "anchor_id", "explainer", "seed", "k", "config_sha",
    "created_at_utc", "npz_path", "json_path",
)
PROVENANCE_COLUMNS = ("fidelity_sufficiency", "fidelity_comprehensiveness", "fidelity_over_random", "runtime_s")


def find_unpacked(run_dir: Path) -> Path:
    for candidate in (run_dir / "unpacked", run_dir / "kernel_output", run_dir):
        if (candidate / "results").is_dir():
            return candidate
    raise SystemExit(f"no results/ under {run_dir}")


def resolve_path(value: str, results_root: Path) -> Path:
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


def check_channel(results_root: Path, channel: str, expected_rows: int) -> dict:
    root = results_root / "explanations_noise" / channel
    index = root / "index.csv"
    record = {"channel": channel, "index": str(index), "problems": []}
    if not index.is_file():
        record["problems"].append("index.csv missing")
        return record
    with index.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    record["rows"] = len(rows)
    missing = [column for column in INDEX_COLUMNS + PROVENANCE_COLUMNS if column not in fields]
    if missing:
        record["problems"].append(f"missing columns {missing}")
    keys = [row.get("key") for row in rows]
    duplicates = len(keys) - len(set(keys))
    if duplicates:
        record["problems"].append(f"{duplicates} duplicate key(s)")
    if expected_rows and len(rows) != expected_rows:
        record["problems"].append(f"row count {len(rows)} != expected {expected_rows}")
    explainers = sorted({str(row.get("explainer")) for row in rows})
    record["explainers"] = explainers
    missing_entries = 0
    bad_arrays = 0
    metadata_provenance_warning = 0
    missing_index_provenance = 0
    class_metadata = 0
    over_random_empty = 0
    for row in rows:
        try:
            runtime = float(row.get("runtime_s"))
            numeric = all(float(row.get(column)) == float(row.get(column)) for column in ("fidelity_sufficiency", "fidelity_comprehensiveness"))
            if runtime <= 0 or not numeric:
                missing_index_provenance += 1
            try:
                float(row.get("fidelity_over_random"))
            except (TypeError, ValueError):
                over_random_empty += 1
        except (TypeError, ValueError):
            missing_index_provenance += 1
    for row in rows[:500]:
        npz = resolve_path(str(row.get("npz_path") or ""), results_root)
        meta = resolve_path(str(row.get("json_path") or ""), results_root)
        if not npz.is_file() or not meta.is_file():
            missing_entries += 1
            continue
        try:
            with np.load(npz) as archive:
                names = archive.files
                arrays = [archive[name] for name in names]
                if not names or not all(np.all(np.isfinite(array)) for array in arrays if getattr(array, "dtype", None) is not None and np.issubdtype(array.dtype, np.number)):
                    bad_arrays += 1
        except Exception:
            bad_arrays += 1
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except Exception:
            bad_arrays += 1
            continue
        if "fidelity" not in payload or "runtime_s" not in payload:
            metadata_provenance_warning += 1
        if "class" in payload or "label" in payload:
            class_metadata += 1
    if missing_entries:
        record["problems"].append(f"{missing_entries} entry pair(s) missing")
    if bad_arrays:
        record["problems"].append(f"{bad_arrays} unreadable/non-finite entry array(s)")
    if missing_index_provenance:
        record["problems"].append(f"{missing_index_provenance} index row(s) missing numeric fidelity/provenance columns")
    record["entries_checked"] = min(len(rows), 500)
    record["metadata_provenance_warning"] = metadata_provenance_warning
    record["fidelity_over_random_empty"] = over_random_empty
    record["class_metadata_present"] = class_metadata
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--anchors-per-period", type=int, default=100)
    parser.add_argument("--periods", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--null-reseeds", type=int, default=5)
    parser.add_argument("--explainers", default="ig,gnn_explainer")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    unpacked = find_unpacked(run_dir)
    results_root = unpacked / "results"
    explainers = [item.strip() for item in args.explainers.split(",") if item.strip()]
    expected_null = args.anchors_per_period * args.periods * args.null_reseeds * len(explainers)
    expected_repeat = args.anchors_per_period * args.periods * args.repeats * len(explainers)
    summary = {
        "run_dir": str(run_dir.resolve()),
        "anchors_per_period": args.anchors_per_period,
        "periods": args.periods,
        "repeats": args.repeats,
        "null_reseeds": args.null_reseeds,
        "explainers": explainers,
        "channels": [
            check_channel(results_root, "model_null", expected_null),
            check_channel(results_root, "explainer_repeat", expected_repeat),
        ],
    }
    combined = results_root / "explanations_noise" / "combined_null.csv"
    problems: list[str] = []
    if not combined.is_file():
        problems.append("combined_null.csv missing")
    else:
        with combined.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
        summary["combined"] = {"rows": len(rows), "has_channel_column": "channel" in fields}
        if "channel" not in fields:
            problems.append("combined_null.csv missing channel column")
        expected_combined = expected_null + expected_repeat
        if len(rows) != expected_combined:
            problems.append(f"combined rows {len(rows)} != expected {expected_combined}")
    manifests = sorted((results_root / "data_audit").glob("noise_floor_manifest_*.json"))
    summaries = sorted((results_root / "data_audit").glob("noise_floor_summary_*.json"))
    summary["manifests"] = len(manifests)
    summary["period_summaries"] = len(summaries)
    if len(manifests) != args.periods:
        problems.append(f"noise floor manifests {len(manifests)} != periods {args.periods}")
    if len(summaries) != args.periods:
        problems.append(f"noise floor summaries {len(summaries)} != periods {args.periods}")
    canaries = []
    for path in summaries:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        canary = ((payload.get("explainer_repeat") or {}).get("ig_repeat_canary_drift_max"))
        if canary is not None:
            canaries.append(float(canary))
    summary["ig_repeat_canary_max"] = max(canaries) if canaries else None
    if canaries and max(canaries) > 1e-9:
        problems.append(f"IG repeat canary non-zero: {max(canaries)}")
    for channel in summary["channels"]:
        problems.extend(f"{channel['channel']}: {problem}" for problem in channel["problems"])
    summary["problems"] = problems
    summary["overall"] = "pass" if not problems else "fail"
    out = Path(args.out) if args.out else run_dir / "noise_audit.json"
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: summary.get(key) for key in ("channels", "combined", "manifests", "period_summaries", "ig_repeat_canary_max", "problems", "overall")}, indent=1, default=str)[:4000])
    print("written", out)
    return 0 if summary["overall"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
