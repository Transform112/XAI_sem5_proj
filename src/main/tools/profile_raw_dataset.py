#!/usr/bin/env python
"""Profile the raw NF-UNSW-NB15-v3 CSV before writing the cleaning step.

Why this tool exists
--------------------
The cleaning step must be driven by what is actually in the CSV, not by assumptions
copied from the frozen artefacts. This script is a stdlib-only diagnostic (pandas is
not required) that streams the raw file once and records, per column:

* empty/NA counts and negative/zero/sentinel counts,
* numeric vs non-numeric values, min/max/mean/std,
* exact unique counts for low-cardinality columns and a bounded lower bound for
  high-cardinality columns,
* the label/attack distributions and obvious label inconsistencies,
* full-row and connection-key duplicate counts.

It writes JSON for reuse and prints a compact human summary. This is a one-off
diagnostic helper, not a numbered pipeline step; the pipeline step that consumes
it is the data-cleaning/preprocessing script.

Usage
-----
python src/main/tools/profile_raw_dataset.py \
    --csv datasets/raw/NF-UNSW-NB15-v3/NF-UNSW-NB15-v3.csv \
    --out results/text/data_audit/raw_dataset_profile.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# A bounded set keeps memory predictable on the 2.36M-row file. Columns above
# this cap report "at least N" rather than an exact unique count.
UNIQUE_CAP = 20_000
# Columns used as a connection fingerprint for duplicate-key diagnostics.
KEY_COLUMNS = (
    "FLOW_START_MILLISECONDS",
    "IPV4_SRC_ADDR",
    "L4_SRC_PORT",
    "IPV4_DST_ADDR",
    "L4_DST_PORT",
    "PROTOCOL",
)
MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none"}


def _is_missing(value: str) -> bool:
    return value.strip().casefold() in MISSING_TOKENS


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _new_column_state() -> dict[str, Any]:
    return {
        "rows": 0,
        "empty": 0,
        "nonnumeric": 0,
        "numeric": 0,
        "negative": 0,
        "zero": 0,
        "min": None,
        "max": None,
        "sum": 0.0,
        "sumsq": 0.0,
        "unique": set(),
        "unique_capped": False,
        "examples": [],
    }


def _update_numeric(state: dict[str, Any], value: float) -> None:
    state["numeric"] += 1
    if value < 0:
        state["negative"] += 1
    elif value == 0:
        state["zero"] += 1
    state["sum"] += value
    state["sumsq"] += value * value
    if state["min"] is None or value < state["min"]:
        state["min"] = value
    if state["max"] is None or value > state["max"]:
        state["max"] = value


def _finish_column(state: dict[str, Any]) -> dict[str, Any]:
    rows = state["rows"]
    numeric = state["numeric"]
    mean = state["sum"] / numeric if numeric else None
    variance = None
    std = None
    if numeric:
        variance = max(0.0, state["sumsq"] / numeric - mean * mean)
        std = math.sqrt(variance)
    unique = len(state["unique"])
    return {
        "rows": rows,
        "empty": state["empty"],
        "empty_pct": round(100.0 * state["empty"] / rows, 6) if rows else None,
        "numeric": numeric,
        "nonnumeric": state["nonnumeric"],
        "numeric_pct": round(100.0 * numeric / rows, 6) if rows else None,
        "negative": state["negative"],
        "zero": state["zero"],
        "min": state["min"],
        "max": state["max"],
        "mean": round(mean, 10) if mean is not None else None,
        "std": round(std, 10) if std is not None else None,
        "unique": unique,
        "unique_capped": bool(state["unique_capped"]),
        "examples": state["examples"],
    }


def profile(csv_path: Path, out_path: Path, *, hash_file: bool = False) -> dict[str, Any]:
    with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = [column.strip() for column in next(reader)]
        except StopIteration as exc:
            raise SystemExit(f"empty CSV: {csv_path}") from exc

        states = {column: _new_column_state() for column in header}
        row_digests: set[bytes] = set()
        key_digests: set[bytes] = set()
        label_counts: Counter[str] = Counter()
        attack_counts: Counter[str] = Counter()
        label_attack: Counter[str] = Counter()
        label_inconsistent = 0
        empty_label = 0
        empty_attack = 0
        start_gt_end = 0
        negative_start = 0
        row_count = 0
        bad_width = 0
        key_index = [header.index(name) for name in KEY_COLUMNS if name in header]

        for row in reader:
            if not row or (len(row) == 1 and not row[0].strip()):
                continue
            row_count += 1
            if len(row) != len(header):
                bad_width += 1
                # Pad/truncate only for statistics; the width count is reported.
                row = (row + [""] * len(header))[: len(header)]

            for column, value in zip(header, row):
                state = states[column]
                state["rows"] += 1
                raw = value.strip()
                if _is_missing(raw):
                    state["empty"] += 1
                    continue
                number = _to_float(raw)
                if number is None:
                    state["nonnumeric"] += 1
                else:
                    _update_numeric(state, number)
                if len(state["unique"]) < UNIQUE_CAP:
                    state["unique"].add(raw)
                else:
                    state["unique_capped"] = True
                if len(state["examples"]) < 3 and raw not in state["examples"]:
                    state["examples"].append(raw)

            # Full-row duplicates are exact repeats; the key digest is a
            # connection-level check (same start, hosts, ports and protocol).
            row_bytes = "\x1f".join(row).encode("utf-8", "replace")
            row_digests.add(hashlib.blake2b(row_bytes, digest_size=16).digest())
            if key_index:
                key_bytes = "\x1f".join(row[index] for index in key_index).encode("utf-8", "replace")
                key_digests.add(hashlib.blake2b(key_bytes, digest_size=16).digest())

            label = row[header.index("Label")].strip() if "Label" in header else ""
            attack = row[header.index("Attack")].strip() if "Attack" in header else ""
            label_counts[label] += 1
            attack_counts[attack] += 1
            if not label:
                empty_label += 1
            if not attack:
                empty_attack += 1
            if label and attack:
                label_attack[f"{label}|{attack}"] += 1
                is_benign = attack.casefold() == "benign"
                if (label == "0" and not is_benign) or (label == "1" and is_benign):
                    label_inconsistent += 1

            start = _to_float(row[header.index("FLOW_START_MILLISECONDS")]) if "FLOW_START_MILLISECONDS" in header else None
            end = _to_float(row[header.index("FLOW_END_MILLISECONDS")]) if "FLOW_END_MILLISECONDS" in header else None
            if start is not None and end is not None and start > end:
                start_gt_end += 1
            if start is not None and start < 0:
                negative_start += 1

    columns = {column: _finish_column(states[column]) for column in header}
    duplicates = row_count - len(row_digests)
    key_duplicates = row_count - len(key_digests) if key_index else None

    report: dict[str, Any] = {
        "schema_version": 1,
        "csv": str(csv_path),
        "size_bytes": csv_path.stat().st_size,
        "sha256": _sha256(csv_path) if hash_file else None,
        "rows": row_count,
        "columns": len(header),
        "header": header,
        "bad_width_rows": bad_width,
        "duplicates": {
            "full_row_rows_removed": duplicates,
            "full_row_unique": len(row_digests),
            "key_columns": list(KEY_COLUMNS),
            "key_rows_removed": key_duplicates,
        },
        "labels": {
            "Label_counts": dict(label_counts),
            "Attack_counts": dict(attack_counts),
            "label_attack_pairs": dict(label_attack),
            "label_inconsistent": label_inconsistent,
            "empty_Label": empty_label,
            "empty_Attack": empty_attack,
        },
        "time": {
            "rows_with_start_gt_end": start_gt_end,
            "rows_with_negative_start": negative_start,
        },
        "columns_profile": columns,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=True), encoding="utf-8")
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_summary(report: dict[str, Any], out_path: Path) -> None:
    print("=" * 88)
    print("RAW DATASET PROFILE | NF-UNSW-NB15-v3")
    print("=" * 88)
    print(f"rows={report['rows']:,}  columns={report['columns']}  bad_width={report['bad_width_rows']:,}")
    dup = report["duplicates"]
    print(f"full-row duplicate rows removed={dup['full_row_rows_removed']:,}")
    print(f"connection-key duplicate rows removed={dup['key_rows_removed']:,}")
    labels = report["labels"]
    print(f"Label counts={labels['Label_counts']}  inconsistent={labels['label_inconsistent']:,}")
    print(f"Attack classes={len(labels['Attack_counts'])}  empty_Attack={labels['empty_Attack']:,}")
    print("-" * 88)
    print(f"{'column':36s} {'empty':>8s} {'nonnum':>8s} {'neg':>8s} {'zero':>8s} {'unique':>8s}  range")
    for name, col in report["columns_profile"].items():
        unique = f">={col['unique']}" if col["unique_capped"] else str(col["unique"])
        range_text = ""
        if col["min"] is not None:
            range_text = f"[{col['min']:.4g}, {col['max']:.4g}]"
        print(
            f"{name:36.36s} {col['empty']:>8,} {col['nonnumeric']:>8,} "
            f"{col['negative']:>8,} {col['zero']:>8,} {unique:>8s}  {range_text}"
        )
    print("-" * 88)
    print(f"JSON: {out_path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile the raw NF-UNSW-NB15-v3 CSV.")
    parser.add_argument(
        "--csv",
        default="datasets/raw/NF-UNSW-NB15-v3/NF-UNSW-NB15-v3.csv",
        help="raw CSV path (relative to the repository root)",
    )
    parser.add_argument(
        "--out",
        default="results/text/data_audit/raw_dataset_profile.json",
        help="JSON output path (relative to the repository root)",
    )
    parser.add_argument("--hash", action="store_true", help="also compute the CSV SHA-256")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    csv_path = Path(args.csv)
    if not csv_path.is_absolute():
        csv_path = root / csv_path
    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = root / out_path
    if not csv_path.is_file():
        print(f"[FAIL] raw CSV not found: {csv_path}", file=sys.stderr)
        return 2
    report = profile(csv_path, out_path, hash_file=args.hash)
    print_summary(report, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
