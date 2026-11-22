#!/usr/bin/env python
"""Step 09: exploratory RQ2 on the new explanation corpus.

Question: does an anchor's calibrated IG drift follow the change in its local
host neighbourhood?

This is deliberately a *new* analysis, not the signed S10 result:
- one checkpoint (seed-42, constant node features, depth 2);
- the new 1,200-anchor corpus, not the signed S7 corpus;
- no matched null-update correlation yet.

It joins:
  results/text/tables/drift_calibrated/calibrated_rows.csv  (real IG drift)
  network neighbourhood change from the frozen period graphs + anchor bundles.

Outputs:
  results/text/tables/rq2_new/rq2_correlations.csv
  results/text/tables/rq2_new/rq2_anchor_level.csv
  results/text/data_audit/rq2_new/summary.json
  results/images/rq2_new/fig01_scatter.png
  results/images/rq2_new/fig02_spearman_by_pair.png
  analysis/experiments/step09_rq2_new_corpus.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.analysis import stats as st  # noqa: E402
from src.analysis_wiring.frames import build_neighbourhood_frame  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--dataset", default="NF-UNSW-NB15-v3")
    parser.add_argument("--run-id", default="ft-none-d2-s42")
    parser.add_argument("--calibrated", type=Path)
    parser.add_argument("--graph-root", type=Path)
    parser.add_argument("--anchor-root", type=Path)
    parser.add_argument("--out-root", type=Path)
    parser.add_argument("--summary-root", type=Path)
    parser.add_argument("--figures-root", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--fanout", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--no-figures", action="store_true")
    return parser.parse_args(argv)


def _resolve(root: Path, value: Path | None, default: Path) -> Path:
    if value is None:
        return root / default
    return value if value.is_absolute() else root / value


def _load_graphs(root: Path, dataset: str, graph_root: Path) -> dict[int, Any]:
    import torch

    graph_dir = graph_root / dataset / "constant"
    graphs: dict[int, Any] = {}
    for path in sorted(graph_dir.glob("period_*.pt")):
        period = int(path.stem.split("_")[1])
        graphs[period] = torch.load(path, map_location="cpu", weights_only=False)
    if not graphs:
        raise FileNotFoundError(f"no period graphs under {graph_dir}")
    return graphs


def _load_anchors(anchor_root: Path, dataset: str, run_id: str) -> dict[int, dict[str, Any]]:
    import torch

    anchor_dir = anchor_root / dataset / "explanations" / "anchors" / run_id
    anchor_sets: dict[int, dict[str, Any]] = {}
    for path in sorted(anchor_dir.glob("period_*.pt")):
        period = int(path.stem.split("_")[1])
        anchor_sets[period] = torch.load(path, map_location="cpu", weights_only=False)
    if not anchor_sets:
        raise FileNotFoundError(f"no anchor bundles under {anchor_dir}")
    return anchor_sets


def _figure_scatter(frame: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    pairs = sorted(frame.groupby(["checkpoint_j", "checkpoint_t"]).size().index)
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(pairs)))
    rng = np.random.default_rng(7)
    for color, (j, t) in zip(colors, pairs):
        part = frame[(frame.checkpoint_j == j) & (frame.checkpoint_t == t)]
        jitter = rng.uniform(-0.0012, 0.0012, size=len(part))
        ax.scatter(part.neighbourhood_change + jitter, part.drift_primary, s=9, alpha=0.55, color=color, label=f"P{j}->P{t}")
    ax.set_xlabel("host-neighbourhood change (1 - Jaccard; x jittered)")
    ax.set_ylabel("calibrated IG drift (1 - Jaccard top-5)")
    ax.set_title("RQ2 exploratory: structural change vs IG drift")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=6, ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _figure_bars(table: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    table = table.copy()
    table["pair"] = table["checkpoint_j"].astype(str) + "->" + table["checkpoint_t"].astype(str)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.errorbar(
        range(len(table)),
        table["spearman"],
        yerr=[table["spearman"] - table["ci_low"], table["ci_high"] - table["spearman"]],
        fmt="o",
        capsize=3,
        color="#1f4e79",
    )
    ax.set_xticks(range(len(table)), table["pair"], rotation=30, ha="right")
    ax.set_ylabel("Spearman rho (structural change vs drift)")
    ax.set_title("RQ2 exploratory: real IG drift vs neighbourhood change")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.root.resolve()
    corpus_root = _resolve(root, args.corpus_root, Path("src/kaggle_lane/outputs/xdrift-step07-corpus/step07_corpus"))
    calibrated_path = _resolve(root, args.calibrated, Path("results/text/tables/drift_calibrated/calibrated_rows.csv"))
    graph_root = _resolve(root, args.graph_root, Path("datasets/graphs"))
    anchor_root = _resolve(root, args.anchor_root, corpus_root / "datasets/processed")
    out_root = _resolve(root, args.out_root, Path("results/text/tables/rq2_new"))
    summary_root = _resolve(root, args.summary_root, Path("results/text/data_audit/rq2_new"))
    figures_root = _resolve(root, args.figures_root, Path("results/images/rq2_new"))
    report = _resolve(root, args.report, Path("analysis/experiments/step09_rq2_new_corpus.md"))
    for directory in (out_root, summary_root, figures_root, report.parent):
        directory.mkdir(parents=True, exist_ok=True)

    calibrated = pd.read_csv(calibrated_path)
    calibrated = calibrated[(calibrated["explainer"] == "ig") & (calibrated["pair_type"] == "cross")].copy()
    if calibrated.empty:
        raise SystemExit("no cross IG rows in calibrated table")

    graphs = _load_graphs(root, args.dataset, graph_root)
    anchor_sets = _load_anchors(anchor_root, args.dataset, args.run_id)
    structural = build_neighbourhood_frame(
        graphs,
        anchor_sets,
        strategy="ft",
        model="egraphsage",
        explainer="ig",
        seed=42,
        fanout=args.fanout,
        hops=args.hops,
    )
    if structural.empty:
        raise SystemExit("neighbourhood frame is empty")

    merged = calibrated.merge(
        structural[["anchor_id", "checkpoint_j", "checkpoint_t", "neighbourhood_change", "edge_set_change", "degree"]],
        left_on=["anchor_id", "anchor_period", "checkpoint_period"],
        right_on=["anchor_id", "checkpoint_j", "checkpoint_t"],
        how="inner",
        validate="one_to_one",
    )
    merged = merged.dropna(subset=["drift_primary", "neighbourhood_change", "degree"])
    merged["class_code"] = pd.factorize(merged["class"])[0]
    merged.to_csv(out_root / "rq2_anchor_level.csv", index=False)

    rows: list[dict[str, Any]] = []
    for (j, t), group in merged.groupby(["anchor_period", "checkpoint_period"], sort=True):
        x = group["neighbourhood_change"].to_numpy()
        y = group["drift_primary"].to_numpy()
        if x.size < 5 or np.ptp(x) == 0 or np.ptp(y) == 0:
            continue
        rho = st.spearman_correlation(x, y)
        ci = st.spearman_correlation_ci(x, y, n_resamples=args.bootstrap)
        partial = st.partial_spearman(x, y, group[["degree"]].to_numpy())
        perm = st.permutation_test(x, y, n_permutations=args.permutations)
        rows.append(
            {
                "checkpoint_j": int(j),
                "checkpoint_t": int(t),
                "n": int(x.size),
                "spearman": rho,
                "ci_low": ci.lo,
                "ci_high": ci.hi,
                "partial_spearman_degree": partial,
                "perm_p_greater": perm.p_value,
                "drift_mean": float(np.mean(y)),
                "change_mean": float(np.mean(x)),
            }
        )
    if not rows:
        raise SystemExit("no period pair had enough rows for RQ2")
    table = pd.DataFrame(rows)
    adjusted = st.holm_correction(table["perm_p_greater"].to_numpy())
    table["holm_p"] = adjusted
    table["survives_holm_0_05"] = table["holm_p"] < 0.05
    table.to_csv(out_root / "rq2_correlations.csv", index=False)

    summary = {
        "dataset": args.dataset,
        "run_id": args.run_id,
        "explainer": "ig",
        "context": {"hops": args.hops, "fanout": args.fanout, "edge_cap": 2000},
        "n_anchor_pairs": int(len(merged)),
        "n_period_pairs": int(len(table)),
        "pairs": table.to_dict("records"),
        "overall_spearman": st.spearman_correlation(merged["neighbourhood_change"], merged["drift_primary"]),
        "overall_partial_spearman_degree": st.partial_spearman(
            merged["neighbourhood_change"], merged["drift_primary"], merged[["degree"]].to_numpy()
        ),
        "note": "Exploratory new-corpus RQ2; not signed S10. No matched null-update correlation.",
    }
    (summary_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.no_figures:
        _figure_scatter(merged, figures_root / "fig01_scatter.png")
        _figure_bars(table, figures_root / "fig02_spearman_by_pair.png")

    lines = [
        "# Step 09 — exploratory RQ2 on the new corpus",
        "",
        "This is a new diagnostic analysis, not the signed S10 result. It uses one",
        f"checkpoint (`{args.run_id}`), the new 1,200-anchor corpus, and the 2-hop/10/2,000 context.",
        "No matched null-update correlation is included yet.",
        "",
        f"- Anchor–checkpoint pairs after merge: **{len(merged):,}**.",
        f"- Period pairs with >=5 rows: **{len(table)}**.",
        f"- Overall Spearman (structural change vs IG drift): **{summary['overall_spearman']:.4f}**.",
        f"- Overall partial Spearman controlling degree: **{summary['overall_partial_spearman_degree']:.4f}**.",
        f"- Distinct host-neighbourhood-change values: **{[float(value) for value in sorted(merged['neighbourhood_change'].round(6).unique())]}**.",
        "- `edge_set_change` is not used: temporal graphs contain different flow-row ids, so it is near-constant across checkpoints.",
        "",
        "| pair | n | Spearman [95% CI] | partial (degree) | perm p (greater) | Holm p |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for row in table.itertuples(index=False):
        lines.append(
            f"| P{row.checkpoint_j}->P{row.checkpoint_t} | {row.n} | "
            f"{row.spearman:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}] | "
            f"{row.partial_spearman_degree:.3f} | {row.perm_p_greater:.4f} | {row.holm_p:.4f} |"
        )
    lines += [
        "",
        "Interpretation guardrails:",
        "- One seed/checkpoint only; no H2 claim and no comparison with signed S10.",
        "- Class coverage is uneven; rare-class rows are tiny.",
        "- The 2-hop host-neighbourhood change is nearly discrete on this 40-host graph; degree controls most of the raw association.",
        "- These correlations are association only, not causal.",
        "",
    ]
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] wrote {out_root / 'rq2_correlations.csv'}")
    print(f"[OK] overall Spearman {summary['overall_spearman']:.4f}; {len(table)} period pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
