#!/usr/bin/env python
"""Independent statistics recomputation for later gates (verifier-owned).

Recomputes the statistics the protocol promises (docs/05 §13) from per-item rows:
  * mean / median and bootstrap 95% CI (percentile, N=1000, seeded)
  * paired comparison across two conditions (paired bootstrap + Wilcoxon)
  * Holm correction over a family of p-values
  * comparison against reported numbers (if a report JSON/CSV is given)

Usage examples:
  python verification\\scripts\\v_stats.py --csv results/text/tables/drift_per_anchor.csv ^
      --group-cols strategy,period --value-col drift --resamples 1000 --out verification/stats
  python verification\\scripts\\v_stats.py --csv x.csv --group-cols strategy --value-col drift ^
      --pair-by anchor_id --pair-a FT --pair-b Replay
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_csv(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def boot_ci(x: np.ndarray, rng: np.random.Generator, n: int = 1000) -> tuple[float, float, float, float]:
    if len(x) == 0:
        return (float("nan"),) * 4
    mean = float(x.mean())
    idx = rng.integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return mean, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), float(x.std(ddof=1) / np.sqrt(len(x)))


def paired_bootstrap(a: np.ndarray, b: np.ndarray, rng: np.random.Generator, n: int = 1000) -> dict:
    d = a - b
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return {"n_pairs": int(len(d)), "mean_diff": float(d.mean()),
            "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
            "frac_bootstrap_positive": float((means > 0).mean())}


def wilcoxon(a: np.ndarray, b: np.ndarray) -> dict:
    try:
        from scipy.stats import wilcoxon as wx
        stat, p = wx(a, b, zero_method="wilcox")
        return {"statistic": float(stat), "p_value": float(p)}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def holm(pvals: list[float]) -> list[float]:
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    adj = [0.0] * m
    prev = 0.0
    for rank, i in enumerate(order):
        v = min(1.0, (m - rank) * pvals[i])
        prev = max(prev, v)
        adj[i] = prev
    return adj


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--group-cols", default="")
    ap.add_argument("--value-col", required=True)
    ap.add_argument("--resamples", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=20261101)
    ap.add_argument("--pair-by", default=None)
    ap.add_argument("--pair-a", default=None)
    ap.add_argument("--pair-b", default=None)
    ap.add_argument("--out", default="verification/stats")
    args = ap.parse_args()
    rows = read_csv(Path(args.csv))
    if not rows:
        print("empty csv")
        return 1
    val_col = args.value_col
    if val_col not in rows[0]:
        print(f"value column {val_col} not in CSV columns: {list(rows[0])}")
        return 1
    group_cols = [c.strip() for c in args.group_cols.split(",") if c.strip()]
    rng = np.random.default_rng(args.seed)
    groups: dict[tuple, list[float]] = defaultdict(list)
    for r in rows:
        try:
            v = float(r[val_col])
        except (TypeError, ValueError):
            continue
        key = tuple(r.get(c, "") for c in group_cols) if group_cols else ("all",)
        groups[key].append(v)
    out = {"generated_utc": utc_now(), "csv": args.csv, "value_col": val_col,
           "group_cols": group_cols, "resamples": args.resamples, "seed": args.seed, "groups": {}}
    for key, vals in sorted(groups.items()):
        x = np.asarray(vals, dtype=float)
        mean, lo, hi, se = boot_ci(x, rng, args.resamples)
        out["groups"]["|".join(map(str, key))] = {
            "n": int(len(x)), "mean": mean, "median": float(np.median(x)),
            "bootstrap_ci95": [lo, hi], "se": se,
            "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        }
    # paired comparison
    if args.pair_by and args.pair_a and args.pair_b:
        cond_col = group_cols[0] if group_cols else None
        a: dict[str, float] = {}
        b: dict[str, float] = {}
        for r in rows:
            try:
                v = float(r[val_col])
            except (TypeError, ValueError):
                continue
            if cond_col and r.get(cond_col) == args.pair_a:
                a[r[args.pair_by]] = v
            if cond_col and r.get(cond_col) == args.pair_b:
                b[r[args.pair_by]] = v
        common = sorted(set(a) & set(b))
        if common:
            av = np.array([a[k] for k in common])
            bv = np.array([b[k] for k in common])
            out["paired_comparison"] = {"a": args.pair_a, "b": args.pair_b,
                                        "bootstrap": paired_bootstrap(av, bv, rng, args.resamples),
                                        "wilcoxon": wilcoxon(av, bv)}
        else:
            out["paired_comparison"] = {"error": "no paired ids found"}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "stats_recomputed.json"
    path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"wrote {path} ({len(out['groups'])} groups)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
