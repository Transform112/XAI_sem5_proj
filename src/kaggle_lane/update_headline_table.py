"""Update the Kaggle-lane headline CL aggregate table from a set of matrix run dirs.

Reads per-group S5 artifacts (``results/text/tables/matrix/<run_key>/cl_metrics.csv``) from
downloaded Kaggle run directories and aggregates valid rows only (mean ± sd over seeds).
Explicitly records which runs/strategies are valid and which older rows are invalidated
(e.g. D-072 replay bug; EWC lambda=1000 superseded by the lambda=100 rerun).

Usage:
    .venv\\Scripts\\python.exe kaggle_lane\\update_headline_table.py ^
        --ft-run incoming_rtx5060/kaggle_runs/t37-matrix-full-90e7635 ^
        --replay-run incoming_rtx5060/kaggle_runs/t42-replay-rerun-<sha> ^
        --ewc-run incoming_rtx5060/kaggle_runs/t42-ewc-lambda100-<sha> ^
        --out-dir incoming_rtx5060/kaggle_runs/t42_headline
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
METRICS = ("all_classes", "supported_set")


def find_unpacked(run_dir: Path) -> Path:
    """Accept a driver run dir (contains unpacked/) or an unpacked/kernel-output dir."""
    candidates = [run_dir / "unpacked", run_dir / "kernel_output", run_dir]
    for candidate in candidates:
        if (candidate / "results" / "tables").is_dir():
            return candidate
    raise SystemExit(f"no results/text/tables found under {run_dir}")


def collect_groups(run_dir: Path, strategies: set[str]) -> dict[tuple[str, str, str], dict[str, float]]:
    unpacked = find_unpacked(run_dir)
    matrix_root = unpacked / "results" / "tables" / "matrix"
    groups: dict[tuple[str, str, str], dict[str, float]] = {}
    if not matrix_root.is_dir():
        return groups
    for group_dir in sorted(matrix_root.iterdir()):
        metrics_path = group_dir / "cl_metrics.csv"
        if not metrics_path.is_file():
            continue
        with metrics_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                strategy = str(row.get("strategy"))
                if strategies and strategy not in strategies:
                    continue
                model = str(row.get("model"))
                metric = str(row.get("metric"))
                key = (strategy, model, metric)
                values = groups.setdefault(key, {"n": 0, "final": [], "forgetting": [], "bwt": []})
                values["n"] += 1
                for column, name in (("final_average_macro_f1", "final"), ("forgetting", "forgetting"), ("bwt", "bwt")):
                    try:
                        values[name].append(float(row.get(column)))
                    except (TypeError, ValueError):
                        pass
    return groups


def aggregate(groups: dict[tuple[str, str, str], dict], source: str) -> list[dict]:
    rows = []
    for (strategy, model, metric), values in sorted(groups.items()):
        final = np.asarray(values["final"], dtype=float)
        forgetting = np.asarray(values["forgetting"], dtype=float)
        bwt = np.asarray(values["bwt"], dtype=float)
        if final.size == 0:
            continue
        rows.append(
            {
                "strategy": strategy,
                "model": model,
                "metric": metric,
                "n_seeds": int(final.size),
                "final_mean": round(float(final.mean()), 6),
                "final_sd": round(float(final.std(ddof=1)) if final.size > 1 else 0.0, 6),
                "forgetting_mean": round(float(forgetting.mean()), 6) if forgetting.size else "",
                "forgetting_sd": round(float(forgetting.std(ddof=1)) if forgetting.size > 1 else 0.0, 6) if forgetting.size else "",
                "bwt_mean": round(float(bwt.mean()), 6) if bwt.size else "",
                "bwt_sd": round(float(bwt.std(ddof=1)) if bwt.size > 1 else 0.0, 6) if bwt.size else "",
                "valid": "valid",
                "source_run": source,
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ft-run", required=True)
    parser.add_argument("--replay-run", default=None)
    parser.add_argument("--ewc-run", default=None)
    parser.add_argument("--out-dir", default="incoming_rtx5060/kaggle_runs/t42_headline")
    parser.add_argument("--invalidated-note", default=None)
    args = parser.parse_args(argv)

    sources = [("ft", Path(args.ft_run), args.ft_run)]
    if args.replay_run:
        sources.append(("replay", Path(args.replay_run), args.replay_run))
    if args.ewc_run:
        sources.append(("ewc", Path(args.ewc_run), args.ewc_run))

    all_rows: list[dict] = []
    provenance = {}
    for strategy, run_dir, label in sources:
        groups = collect_groups(run_dir, {strategy})
        strategy_rows = aggregate(groups, label)
        all_rows.extend(strategy_rows)
        provenance[strategy] = {"run": label, "groups": len(groups), "rows": len(strategy_rows)}
        print(f"[headline] {strategy}: {len(groups)} metric groups from {label}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table_path = out_dir / "headline_table_valid.csv"
    fields = [
        "strategy", "model", "metric", "n_seeds", "final_mean", "final_sd",
        "forgetting_mean", "forgetting_sd", "bwt_mean", "bwt_sd", "valid", "source_run",
    ]
    with table_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)

    note = args.invalidated_note or (
        "Invalidated rows (D-072/D-073): the original 30-group Kaggle matrix replay "
        "implementations are INVALID (fixed 32-entry batch reused every step, summed CE, "
        "subgraph-vs-full-graph mismatch) and the original EWC lambda=1000 rows are SUPERSEDED "
        "by the lambda=100 pilot rerun. FT rows are valid. Only the rows in "
        "headline_table_valid.csv (FT from the full matrix, replay from the rerun, EWC from the "
        "lambda=100 rerun) are headline-eligible; do not mix old replay/EWC rows into the paper. "
        "Re-run this script after any further rerun."
    )
    note_path = out_dir / "headline_table_valid.md"
    lines = [
        "# Kaggle-lane headline CL aggregate (valid rows only)",
        "",
        f"Generated by `src/kaggle_lane/update_headline_table.py`.",
        "",
        "| strategy | model | metric | n | final mean ± sd | forgetting mean ± sd | BWT mean ± sd | source run |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in all_rows:
        lines.append(
            f"| {row['strategy']} | {row['model']} | {row['metric']} | {row['n_seeds']} | "
            f"{row['final_mean']} ± {row['final_sd']} | {row['forgetting_mean']} ± {row['forgetting_sd']} | "
            f"{row['bwt_mean']} ± {row['bwt_sd']} | `{row['source_run']}` |"
        )
    lines += ["", "## Invalidated / provenance note", "", note, "", "```json", json.dumps(provenance, indent=2), "```", ""]
    note_path.write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "headline_provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[headline] wrote {table_path} ({len(all_rows)} rows) and {note_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
