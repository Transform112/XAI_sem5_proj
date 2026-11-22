#!/usr/bin/env python
"""Step 08C: calibrate the new-corpus IG drift against its matching null floor.

Reads the 08A drift table and the 08B model-null IG floor, attaches the
per-class q95 floor (global fallback when a class has too few null rows), and
reports how often real drift exceeds that floor.  GNNExplainer has no matching
floor yet and is deliberately left out of the H1-style summary.

This is a diagnostic calibration, not a replacement for signed S8.  Outputs
live under ``drift_calibrated`` and ``analysis/experiments/step08_calibrated.md``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--drift-root", default="results/text/tables/drift_main")
    parser.add_argument("--null-dir", default="results/text/tables/null_ig")
    parser.add_argument("--out-root", default="results/text/tables/drift_calibrated")
    parser.add_argument("--summary-root", default="results/text/data_audit/drift")
    parser.add_argument("--figures-root", default="results/images/drift")
    parser.add_argument("--report", default="analysis/experiments/step08_calibrated.md")
    parser.add_argument("--explainer", default="ig")
    parser.add_argument("--min-null-per-class", type=int, default=30)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args(argv)


def _floor_table(path: Path, min_count: int) -> tuple[dict[int, tuple[float, str]], float]:
    floor = pd.read_csv(path)
    global_rows = floor[floor["label"] == -1]
    if global_rows.empty:
        raise ValueError(f"no global floor row in {path}")
    global_q95 = float(global_rows.iloc[0]["q95"])
    per_class: dict[int, tuple[float, str]] = {}
    for row in floor[floor["label"] >= 0].itertuples(index=False):
        if int(row.count) >= min_count and np.isfinite(float(row.q95)):
            per_class[int(row.label)] = (float(row.q95), "class")
    return per_class, global_q95


def _cluster_bootstrap(frame: pd.DataFrame, *, n_boot: int, seed: int) -> dict[str, Any]:
    """Resample anchors (not rows) and recompute the two headline statistics."""
    anchor_ids = frame["anchor_id"].to_numpy()
    unique = np.unique(anchor_ids)
    if len(unique) < 2 or n_boot <= 0:
        return {"n_boot": 0}
    rng = np.random.default_rng(seed)
    by_anchor = {int(a): np.flatnonzero(anchor_ids == a) for a in unique}
    above = frame["above_floor"].to_numpy(dtype=bool)
    excess = frame["excess"].to_numpy(dtype=float)
    fractions, means = [], []
    for _ in range(n_boot):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_anchor[int(a)] for a in chosen])
        fractions.append(float(above[idx].mean()))
        means.append(float(excess[idx].mean()))
    return {
        "n_boot": int(n_boot),
        "above_floor_ci95": [float(np.percentile(fractions, 2.5)), float(np.percentile(fractions, 97.5))],
        "mean_excess_ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
    }


def _binomial_p(n_above: int, n: int, expected: float) -> float | None:
    if n == 0:
        return None
    try:
        from scipy.stats import binomtest

        return float(binomtest(n_above, n, p=expected, alternative="greater").pvalue)
    except Exception:
        # Normal approximation; only a fallback for environments without scipy.
        p_hat = n_above / n
        se = float(np.sqrt(expected * (1.0 - expected) / n))
        if se == 0:
            return None
        z = (p_hat - expected) / se
        return float(0.5 * np.erfc(z / np.sqrt(2.0)))


def _figures(frame: pd.DataFrame, floor_source: dict[str, float], out: Path, summary: dict[str, Any]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    fig, ax = plt.subplots(figsize=(6, 4))
    frac = summary["above_floor_fraction"]
    ax.bar([0, 1], [5.0, 100.0 * frac], color=["#8a9bb0", "#2f6f4e"], width=0.6)
    ax.set_xticks([0, 1], ["expected under no drift", "observed"])
    ax.set_ylim(0.0, max(6.0, 100.0 * frac + 1.0))
    ax.set_ylabel("% of cross-checkpoint rows above null q95")
    ax.set_title("Calibrated IG drift vs 5% expectation", pad=12)
    ax.text(1, 100.0 * frac + 0.2, f"{100.0 * frac:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    path = out / "fig01_above_floor_vs_expected.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(frame["excess"].to_numpy(), bins=50, color="#2f6f4e", alpha=0.8)
    ax.axvline(0.0, color="#b23b2e", linewidth=1)
    ax.set_xlabel("real drift - null q95 floor")
    ax.set_ylabel("rows")
    ax.set_title("Excess drift distribution (IG)")
    fig.tight_layout()
    path = out / "fig02_excess_distribution.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    written.append(str(path))

    if {"class", "above_floor"}.issubset(frame.columns):
        grouped = frame.groupby("class").agg(frac=("above_floor", "mean"), n=("above_floor", "size")).sort_values("frac")
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(grouped.index, 100.0 * grouped["frac"].to_numpy(), color="#2f6f4e")
        ax.axvline(5.0, color="#b23b2e", linestyle="--", linewidth=1)
        for y, (_, row) in enumerate(grouped.iterrows()):
            ax.text(100.0 * row["frac"] + 0.15, y, f"n={int(row['n'])}", va="center", fontsize=7)
        ax.set_xlim(0.0, max(7.0, 100.0 * grouped["frac"].max() + 1.5))
        ax.set_xlabel("% above null floor")
        ax.set_title("Calibrated IG drift by class (class sample on bar)")
        fig.tight_layout()
        path = out / "fig03_above_floor_by_class.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(str(path))
    return written


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    drift_root = Path(args.drift_root)
    null_dir = Path(args.null_dir)
    out_root = Path(args.out_root)
    summary_root = Path(args.summary_root)
    figures_root = Path(args.figures_root)
    out_root.mkdir(parents=True, exist_ok=True)
    summary_root.mkdir(parents=True, exist_ok=True)

    rows = pd.read_csv(drift_root / "drift_rows_cross.csv")
    rows = rows[rows["explainer"] == args.explainer].copy()
    if rows.empty:
        raise SystemExit(f"no {args.explainer} cross rows in {drift_root}")
    per_class, global_q95 = _floor_table(null_dir / "null_floor.csv", args.min_null_per_class)
    rows["floor_q95"] = [per_class.get(int(label), (global_q95, "global"))[0] for label in rows["label"]]
    rows["floor_scope"] = [per_class.get(int(label), (global_q95, "global"))[1] for label in rows["label"]]
    rows["above_floor"] = rows["drift_primary"] > rows["floor_q95"]
    rows["excess"] = rows["drift_primary"] - rows["floor_q95"]

    n = int(len(rows))
    n_above = int(rows["above_floor"].sum())
    summary: dict[str, Any] = {
        "explainer": args.explainer,
        "n_cross_rows": n,
        "n_anchors": int(rows["anchor_id"].nunique()),
        "above_floor": n_above,
        "above_floor_fraction": float(n_above / n) if n else float("nan"),
        "expected_fraction": 0.05,
        "binomial_p_greater_than_5pct": _binomial_p(n_above, n, 0.05),
        "mean_drift": float(rows["drift_primary"].mean()),
        "mean_excess": float(rows["excess"].mean()),
        "median_excess": float(rows["excess"].median()),
        "q05_excess": float(rows["excess"].quantile(0.05)),
        "q95_excess": float(rows["excess"].quantile(0.95)),
        "floor_scope_counts": rows["floor_scope"].value_counts().to_dict(),
        "global_q95": global_q95,
        "min_null_per_class": args.min_null_per_class,
        "note": "Diagnostic calibration on the new corpus; not signed S8 and not a formal H1 test.",
    }
    summary.update(_cluster_bootstrap(rows, n_boot=args.bootstrap, seed=args.seed))
    rows.to_csv(out_root / "calibrated_rows.csv", index=False)
    by_class = rows.groupby("class").agg(
        n=("above_floor", "size"),
        above_floor=("above_floor", "sum"),
        above_floor_fraction=("above_floor", "mean"),
        mean_drift=("drift_primary", "mean"),
        mean_excess=("excess", "mean"),
        floor_q95=("floor_q95", "first"),
    ).reset_index()
    by_class.to_csv(out_root / "calibrated_by_class.csv", index=False)
    rows.groupby(["anchor_period", "checkpoint_period"]).agg(
        n=("above_floor", "size"),
        above_floor_fraction=("above_floor", "mean"),
        mean_excess=("excess", "mean"),
    ).reset_index().to_csv(out_root / "calibrated_by_period_pair.csv", index=False)
    summary = {"dataset": str(rows["dataset"].iloc[0]), "run_id": str(rows["run_id"].iloc[0]), **summary}
    if not args.no_figures:
        summary["figures"] = _figures(rows, {"global": global_q95}, figures_root, summary)
    summary_path = summary_root / "drift_calibrated_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    floor_table = [
        f"| {row['class']} | {int(row['n'])} | {100.0 * row['above_floor_fraction']:.2f}% | {row['mean_excess']:.3f} | {row['floor_q95']:.3f} |"
        for _, row in by_class.iterrows()
    ]
    report.write_text(
        "\n".join([
            "# Step 08C: calibrated IG drift (new corpus)",
            "",
            f"- Rows: {summary['n_cross_rows']} cross-checkpoint IG rows from {summary['n_anchors']} anchors.",
            f"- Above matching null q95: **{summary['above_floor']} / {summary['n_cross_rows']} = {100.0 * summary['above_floor_fraction']:.2f}%** (5% expected).",
            f"- Mean excess over the floor: {summary['mean_excess']:.4f}; median {summary['median_excess']:.4f}.",
            f"- Cluster-bootstrap 95% CI for above-floor fraction: {summary.get('above_floor_ci95')}.",
            f"- Binomial p (greater than 5%): {summary['binomial_p_greater_than_5pct']}.",
            f"- Null floor: {summary['floor_scope_counts']}; global q95 {summary['global_q95']:.3f}; class floors require ≥{summary['min_null_per_class']} null rows.",
            "",
            "| class | rows | above floor | mean excess | floor q95 |",
            "| --- | ---: | ---: | ---: | ---: |",
            *floor_table,
            "",
            "This is a **diagnostic calibration**, not the signed S8 result. GNNExplainer has no matching null floor yet. The new corpus has a different anchor set and checkpoint from signed S7/S8, so these numbers cannot be mixed with those artifacts.",
            "",
        ]),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(f"[OK] wrote {out_root} and {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
