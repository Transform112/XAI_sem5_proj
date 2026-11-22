"""Step 08A: raw explanation drift from the diagnostic step-07 corpus.

This stage deliberately does **not** claim calibrated drift. It reads the
explanation cache built by step 07, pairs each anchor's period-j explanation
with every later checkpoint, and writes the full raw drift vector required by
the frozen protocol:

* IG: cosine, top-3/5/10 Jaccard, Spearman; primary = top-5 Jaccard drift.
* GNNExplainer: top-5 edge-mask Jaccard and mask cosine; primary = top-5 Jaccard.

The corpus stores each explanation separately from its metadata. The cache
index points at Kaggle paths, so the loader remaps files to the extracted
corpus root. IG has separate k=5 and k=10 cache rows for the same attribution
vector; only one row per (anchor, anchor period, checkpoint period) is used,
and all top-k sets are derived from that one vector. This avoids double
counting.

Inputs are the fixed step-07 corpus; this script never retrains, re-explains,
re-selects anchors, or modifies signed S7/S8 artifacts.

Usage:
    python src/main/08_drift.py \
        --corpus-root src/kaggle_lane/outputs/xdrift-step07-corpus/corpus \
        --dataset NF-UNSW-NB15-v3 \
        --run-id ft-none-d2-s42
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.analysis import metrics_drift as md

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "src/kaggle_lane/outputs/xdrift-step07-corpus/corpus"
CLASS_NAMES = (
    "Benign",
    "Analysis",
    "Backdoor",
    "DoS",
    "Exploits",
    "Fuzzers",
    "Generic",
    "Reconnaissance",
    "Shellcode",
    "Worms",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--dataset", default="NF-UNSW-NB15-v3")
    parser.add_argument("--run-id", default="ft-none-d2-s42")
    parser.add_argument("--out-root", type=Path, default=ROOT / "results/text/tables/drift_main")
    parser.add_argument("--summary-root", type=Path, default=ROOT / "results/text/data_audit/drift")
    parser.add_argument("--figures-root", type=Path, default=ROOT / "results/images/drift")
    parser.add_argument("--report", type=Path, default=ROOT / "analysis/experiments/step08_raw_drift.md")
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--fanout", type=int, default=10)
    parser.add_argument("--edge-cap", type=int, default=2000)
    parser.add_argument("--strict", action="store_true", help="fail if any anchor/checkpoint pair is missing")
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args(argv)


def load_index(corpus_root: Path) -> pd.DataFrame:
    path = corpus_root / "results/text/tables/explain_index_main.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing explanation index: {path}")
    index = pd.read_csv(path)
    required = {"key", "anchor_id", "explainer", "k"}
    missing = required - set(index.columns)
    if missing:
        raise ValueError(f"explanation index is missing columns: {sorted(missing)}")
    return index


def _record_paths(corpus_root: Path, key: str) -> tuple[Path, Path]:
    # The index stores absolute Kaggle paths; the extracted corpus keeps the
    # same relative layout under results/text/explanations.
    base = corpus_root / "results/text/explanations"
    return base / f"{key}.npz", base / f"{key}.json"


def _load_one(corpus_root: Path, row: pd.Series) -> tuple[np.ndarray, dict[str, Any]]:
    npz_path, json_path = _record_paths(corpus_root, str(row["key"]))
    if not npz_path.is_file() or not json_path.is_file():
        raise FileNotFoundError(f"missing explanation files for key {row['key']}: {npz_path}")
    meta = json.loads(json_path.read_text(encoding="utf-8"))
    with np.load(npz_path) as archive:
        if "attributions" in archive:
            values = np.asarray(archive["attributions"], dtype=float)
        elif "edge_mask" in archive:
            values = np.asarray(archive["edge_mask"], dtype=float)
        else:
            raise ValueError(f"{npz_path} has neither attributions nor edge_mask")
    values = np.asarray(values, dtype=float).reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite explanation in {npz_path}")
    return values, meta


def load_explanations(corpus_root: Path, index: pd.DataFrame) -> dict[tuple[int, int, int, str], dict[str, Any]]:
    """Load one explanation per (anchor_id, anchor_period, checkpoint_period, explainer).

    The IG cache has a separate row for k=5 and k=10; both reference the same
    attribution vector, so only the first (k=5 when available) is kept.
    """
    grouped: dict[tuple[int, int, int, str], list[pd.Series]] = defaultdict(list)
    for _, row in index.iterrows():
        npz_path, json_path = _record_paths(corpus_root, str(row["key"]))
        if not npz_path.is_file() or not json_path.is_file():
            continue
        meta = json.loads(json_path.read_text(encoding="utf-8"))
        anchor_id = int(meta["anchor_id"])
        anchor_period = int(meta["anchor_period"])
        checkpoint_period = int(meta["checkpoint_period"])
        explainer = str(meta["explainer"])
        grouped[(anchor_id, anchor_period, checkpoint_period, explainer)].append(row)

    records: dict[tuple[int, int, int, str], dict[str, Any]] = {}
    for signature, rows in grouped.items():
        # Prefer the k=5 row for IG; it is the primary metric's cache entry.
        chosen = sorted(rows, key=lambda item: int(item["k"]))[0]
        values, meta = _load_one(corpus_root, chosen)
        records[signature] = {
            "values": values,
            "meta": meta,
            "key": str(chosen["key"]),
            "explainer": signature[3],
            "label": int(meta.get("label", -1)),
            "fidelity": meta.get("fidelity", {}),
        }
    return records


def _drift_values(explainer: str, anchor_values: np.ndarray, checkpoint_values: np.ndarray) -> dict[str, float]:
    if explainer == "ig":
        return md.attribution_drift(anchor_values, checkpoint_values)
    if explainer == "gnn_explainer":
        return md.edge_mask_drift(anchor_values, checkpoint_values)
    raise ValueError(f"unknown explainer {explainer!r}")


def build_rows(
    records: dict[tuple[int, int, int, str], dict[str, Any]],
    *,
    dataset: str,
    run_id: str,
    strict: bool,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    anchors = sorted({(key[0], key[1]) for key in records})
    max_period = max(key[2] for key in records)
    rows: list[dict[str, Any]] = []
    missing: list[tuple[int, int, int, str]] = []

    by_anchor: dict[int, dict[str, dict[int, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for (anchor_id, anchor_period, checkpoint_period, explainer), rec in records.items():
        by_anchor[anchor_id][explainer][checkpoint_period] = rec

    for anchor_id, anchor_period in anchors:
        label = -1
        for explainer in ("ig", "gnn_explainer"):
            versions = by_anchor.get(anchor_id, {}).get(explainer, {})
            if not versions:
                missing.append((anchor_id, anchor_period, anchor_period, explainer))
                continue
            if anchor_period not in versions:
                missing.append((anchor_id, anchor_period, anchor_period, explainer))
                continue
            base = versions[anchor_period]
            label = int(base["label"])
            for checkpoint_period in range(anchor_period, max_period + 1):
                if checkpoint_period not in versions:
                    missing.append((anchor_id, anchor_period, checkpoint_period, explainer))
                    continue
                target = versions[checkpoint_period]
                drift = _drift_values(explainer, base["values"], target["values"])
                row: dict[str, Any] = {
                    "dataset": dataset,
                    "run_id": run_id,
                    "anchor_id": anchor_id,
                    "anchor_period": anchor_period,
                    "checkpoint_period": checkpoint_period,
                    "pair_type": "diagonal" if checkpoint_period == anchor_period else "cross",
                    "explainer": explainer,
                    "label": label,
                    "class": CLASS_NAMES[label] if 0 <= label < len(CLASS_NAMES) else str(label),
                    "context_hops": 2,
                    "context_fanout": 10,
                    "context_edge_cap": 2000,
                }
                for name, value in drift.items():
                    row[f"sim_{name}"] = 1.0 - float(value) if math.isfinite(value) else float("nan")
                    row[f"drift_{name}"] = float(value)
                row["drift_primary"] = float(drift[md.PRIMARY_FEATURE_METRIC if explainer == "ig" else md.PRIMARY_EDGE_METRIC])
                rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("no drift rows could be built from the corpus")
    if missing and strict:
        preview = ", ".join(f"{item}" for item in missing[:5])
        raise ValueError(f"corpus is missing {len(missing)} expected pairs (strict): {preview}")

    expected = 0
    for anchor_id, anchor_period in anchors:
        expected += 2 * (max_period - anchor_period + 1)  # two explainers, t = j..T
    summary: dict[str, Any] = {
        "dataset": dataset,
        "run_id": run_id,
        "n_anchors": len(anchors),
        "n_records": int(len(records)),
        "n_drift_rows": int(len(frame)),
        "n_cross_rows": int((frame["pair_type"] == "cross").sum()),
        "n_diagonal_rows": int((frame["pair_type"] == "diagonal").sum()),
        "expected_rows": int(expected),
        "missing_rows": int(len(missing)),
        "strict": bool(strict),
        "anchors_by_period": {
            int(period): int(sum(1 for _, anchor_period in anchors if anchor_period == period))
            for period in range(1, max_period + 1)
        },
        "explainers": {},
    }
    for explainer, group in frame.groupby("explainer"):
        cross = group[group["pair_type"] == "cross"]
        diagonal = group[group["pair_type"] == "diagonal"]
        values = cross["drift_primary"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        summary["explainers"][explainer] = {
            "n_rows": int(len(group)),
            "n_cross_rows": int(len(cross)),
            "n_diagonal_rows": int(len(diagonal)),
            "diagonal_max_abs_primary": float(np.max(np.abs(diagonal["drift_primary"]))) if len(diagonal) else float("nan"),
            "cross_mean_primary": float(np.mean(finite)) if finite.size else float("nan"),
            "cross_median_primary": float(np.median(finite)) if finite.size else float("nan"),
            "cross_q95_primary": float(np.quantile(finite, 0.95)) if finite.size else float("nan"),
            "cross_std_primary": float(np.std(finite, ddof=1)) if finite.size > 1 else float("nan"),
        }
    return frame, summary


def write_summary_tables(frame: pd.DataFrame, out_root: Path, summary_root: Path) -> None:
    out_root.mkdir(parents=True, exist_ok=True)
    summary_root.mkdir(parents=True, exist_ok=True)
    cross = frame[frame["pair_type"] == "cross"].copy()
    cross.to_csv(out_root / "drift_rows_cross.csv", index=False)
    frame.to_csv(out_root / "drift_rows_all.csv", index=False)

    grouped = (
        cross.groupby(["explainer", "class"], dropna=False)["drift_primary"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    grouped["q95"] = [
        float(np.quantile(cross[(cross["explainer"] == row.explainer) & (cross["class"] == row["class"])]["drift_primary"].to_numpy(dtype=float), 0.95))
        for _, row in grouped.iterrows()
    ]
    grouped.to_csv(out_root / "drift_by_class.csv", index=False)

    by_pair = (
        cross.groupby(["explainer", "anchor_period", "checkpoint_period"], dropna=False)["drift_primary"]
        .agg(["count", "mean", "median"])
        .reset_index()
    )
    by_pair.to_csv(out_root / "drift_by_period_pair.csv", index=False)


def make_figures(frame: pd.DataFrame, figures_root: Path) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_root.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []
    cross = frame[frame["pair_type"] == "cross"]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for explainer, group in cross.groupby("explainer"):
        ax.hist(group["drift_primary"].to_numpy(dtype=float), bins=30, alpha=0.55, label=explainer)
    ax.set_xlabel("primary drift (1 - top-5 Jaccard)")
    ax.set_ylabel("anchor-checkpoint pairs")
    ax.set_title("Raw explanation drift by explainer")
    ax.legend(frameon=False)
    path = figures_root / "fig01_drift_distribution.png"
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); out.append(path)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), squeeze=False)
    for ax, explainer in zip(axes[0], ("ig", "gnn_explainer")):
        group = cross[cross["explainer"] == explainer]
        pivot = group.pivot_table(index="anchor_period", columns="checkpoint_period", values="drift_primary", aggfunc="mean")
        im = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", cmap="viridis")
        ax.set_xticks(range(pivot.shape[1]), labels=[str(c) for c in pivot.columns])
        ax.set_yticks(range(pivot.shape[0]), labels=[str(i) for i in pivot.index])
        ax.set_xlabel("checkpoint period")
        ax.set_ylabel("anchor period")
        ax.set_title(explainer)
        fig.colorbar(im, ax=ax, label="mean drift")
    path = figures_root / "fig02_drift_period_heatmap.png"
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); out.append(path)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for explainer, group in cross.groupby("explainer"):
        means = group.groupby("class")["drift_primary"].mean()
        ax.plot(means.index, means.values, marker="o", label=explainer)
    ax.set_ylabel("mean primary drift")
    ax.set_title("Raw drift by class")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(frameon=False)
    path = figures_root / "fig03_drift_by_class.png"
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); out.append(path)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for explainer, group in cross.groupby("explainer"):
        means = group.groupby("checkpoint_period")["drift_primary"].mean()
        ax.plot(means.index, means.values, marker="o", label=explainer)
    ax.set_xlabel("checkpoint period")
    ax.set_ylabel("mean primary drift")
    ax.set_title("Raw drift by checkpoint")
    ax.legend(frameon=False)
    path = figures_root / "fig04_drift_by_checkpoint.png"
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig); out.append(path)
    return out


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Step 08A - raw explanation drift",
        "",
        "Status: **uncalibrated**. No noise floor or H1 claim is made here.",
        "",
        f"- dataset: `{summary['dataset']}`",
        f"- run: `{summary['run_id']}`",
        f"- anchors: **{summary['n_anchors']}**",
        f"- drift rows: **{summary['n_drift_rows']}** "
        f"({summary['n_cross_rows']} cross, {summary['n_diagonal_rows']} diagonal)",
        f"- missing expected rows: **{summary['missing_rows']}**",
        f"- context: 2-hop, fanout 10, edge cap 2,000",
        "",
        "## Primary drift by explainer (cross-checkpoint pairs)",
        "",
        "| explainer | n | mean | median | q95 | sd | diagonal max abs |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for explainer, item in summary["explainers"].items():
        lines.append(
            f"| {explainer} | {item['n_cross_rows']} | {item['cross_mean_primary']:.4f} | "
            f"{item['cross_median_primary']:.4f} | {item['cross_q95_primary']:.4f} | "
            f"{item['cross_std_primary']:.4f} | {item['diagonal_max_abs_primary']:.3g} |"
        )
    lines += [
        "",
        "The primary metric is `1 - Jaccard(top-5 |attribution|)` for IG and "
        "`1 - Jaccard(top-5 |edge mask|)` for GNNExplainer.",
        "",
        "## What this does not mean",
        "",
        "- These are raw drifts. They are not yet compared with a matching null floor.",
        "- The corpus has one GNNExplainer run per pair; explainer-repeat noise is not measured here.",
        "- No reseeded-model null explanations exist for this corpus yet.",
        "- Old F4/S8 floors were built for different checkpoints and context and are not reused.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    corpus_root = args.corpus_root.resolve()
    index = load_index(corpus_root)
    records = load_explanations(corpus_root, index)
    frame, summary = build_rows(records, dataset=args.dataset, run_id=args.run_id, strict=args.strict)
    write_summary_tables(frame, args.out_root, args.summary_root)
    figures = [] if args.no_figures else make_figures(frame, args.figures_root)
    summary["figures"] = [str(path) for path in figures]
    summary_path = args.summary_root / "drift_raw_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_report(args.report, summary)
    print(f"[OK] anchors {summary['n_anchors']}, rows {summary['n_drift_rows']} "
          f"({summary['n_cross_rows']} cross), missing {summary['missing_rows']}")
    print(f"[OK] tables {args.out_root}, summary {summary_path}, report {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
