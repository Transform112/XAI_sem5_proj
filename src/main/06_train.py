"""Step 06: train continual GNN runs and compare reputation / depth variants.

This script is deliberately one linear pass:

1. load the frozen graph snapshots written by ``05_build_graphs.py``;
2. for each node-feature mode, depth and seed, train sequentially over periods;
3. evaluate the per-period test splits and build the R matrix;
4. write run JSON, a CSV table, five summary figures and a Markdown report.

The canonical protocol is the existing project trainer
(``src.train.trainer.train_edge_classifier``): class-weighted cross-entropy,
``graph_batch`` batching, early stopping on validation macro-F1.  This script does
not re-clean, re-split, re-fit the preprocessor, resample validation/test rows,
or rebuild graphs.

Performance contract
--------------------
* Step 05 must already have written the five ``.pt`` snapshots.  Steps 02/05 are
  never invoked here.
* Each mode's five graphs are loaded once, moved to the training device once, and
  reused for every depth and seed.  A depth/seed run only rebuilds the model.
* Completed runs are written to ``runs.json``/``runs.csv`` immediately; use
  ``--resume`` after an interrupted Kaggle kernel to skip finished run IDs.
* ``--diagonal-only`` is a pilot-speed option; full runs evaluate the whole lower
  triangle so A11 forgetting/BWT is defined.

Examples
--------
Smoke test on CPU (small masks, one period, one epoch)::

    python src/main/06_train.py --smoke --device cpu

Compare the reputation and depth variants on the local graph cache::

    python src/main/06_train.py --dataset NF-UNSW-NB15-v3 \
        --node-features constant,reputation,constant+reputation \
        --depths 2,3,4 --strategy ft --seeds 42

Full 5-seed Kaggle runs should be launched with ``--seeds 42,43,44,45,46``; smoke
and pilot runs are engineering tests, not paper evidence.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

# The scientific stack is required for this step.  Give a readable hint if the
# user runs the script before installing the project environment.
try:
    import pandas as pd
    import yaml
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "Step 06 needs pandas and pyyaml. Install the project environment:\n"
        "  pip install -r platforms/requirements/common.txt && pip install -r platforms/requirements/torch.txt"
    ) from exc

from src.cl.strategies import build_strategy
from src.data.registry import resolve_dataset_names
from src.eval.cl_metrics import cl_metrics
from src.models.edge_sage import build_model
from src.train.trainer import (
    TrainSettings,
    evaluate_graph_batch,
    macro_f1,
    train_edge_classifier,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = "NF-UNSW-NB15-v3"
NODE_MODES = ("constant", "reputation", "constant+reputation")
# Coarse diagnostic taxonomy: Benign, Exploits, Fuzzers, Generic, Reconnaissance,
# Shellcode, Rare(Analysis+Backdoor+DoS+Worms). This is an experiment, not the
# frozen 10-class label space.
COARSE_MAP = {0: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 1: 6, 2: 6, 3: 6, 9: 6}
NODE_LABELS = {
    "constant": "constant",
    "reputation": "reputation",
    "constant+reputation": "constant + reputation",
}


def cell(number: int, total: int, title: str) -> None:
    print()
    print("=" * 78)
    print(f"CELL {number:02d}/{total:02d} | {title}")
    print("=" * 78)


def line(tag: str, message: str) -> None:
    print(f"[{tag:<4}] {message}")


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------


def find_root(value: str | None) -> Path:
    """Find the repository root without importing the yaml-dependent resolver."""
    if value:
        root = Path(value).expanduser().resolve()
        if (root / "src").is_dir() and (root / "configs").is_dir():
            return root
        raise SystemExit(f"--root {root} does not look like the project root")
    candidates = [ROOT, *ROOT.parents]
    for candidate in candidates:
        if (candidate / "src").is_dir() and (candidate / "configs").is_dir():
            return candidate
    raise SystemExit(f"could not find project root from {ROOT}")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def choose_datasets(value: str, root: Path) -> list[str]:
    if value != "all":
        return [item.strip() for item in value.split(",") if item.strip()]
    audit = load_yaml(root / "configs" / "audit.yaml")
    names = audit.get("datasets") or [DEFAULT_DATASET]
    return resolve_dataset_names(names, root=root)


def graph_paths(root: Path, dataset: str, mode: str) -> list[Path]:
    return [
        root / "datasets" / "graphs" / dataset / mode / f"period_{period}.pt"
        for period in range(1, 6)
    ]


def load_graphs(paths: Sequence[Path], device: str) -> list[Any]:
    """Load each frozen snapshot once and keep it wherever training runs.

    Moving a graph to the GPU here matters: the trainer otherwise copies
    ``x``/``edge_index``/``edge_attr`` to the device at every period, for every
    depth and seed.  The largest graph is small enough to stay resident on a T4.
    """
    graphs: list[Any] = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"missing graph snapshot: {path}")
        graph = torch.load(path, map_location="cpu", weights_only=False)
        graph = graph.to(device)
        graphs.append(graph)
    return graphs


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------


def _keep_n(mask: torch.Tensor, limit: int | None, seed: int) -> torch.Tensor:
    """Return a boolean mask with at most ``limit`` true entries (deterministic)."""
    if limit is None or int(mask.sum()) <= int(limit):
        return mask
    indices = torch.nonzero(mask, as_tuple=False).flatten()
    generator = torch.Generator().manual_seed(int(seed))
    chosen = indices[torch.randperm(len(indices), generator=generator)[: int(limit)]]
    out = torch.zeros_like(mask)
    out[chosen] = True
    return out


def apply_masks(
    graph: Any,
    *,
    use_balance: bool,
    max_train: int | None,
    max_eval: int | None,
    seed: int,
) -> None:
    """Replace split masks for a smoke test; never touches the saved files."""
    natural_train = graph.train_mask
    natural_val = graph.val_mask
    natural_test = graph.test_mask

    # A graph from step 05 carries a selected training list from step 04.  Use it
    # only when it is actually a subset; otherwise keep the natural train mask.
    selected = getattr(graph, "train_indices", None)
    if use_balance and selected is not None and len(selected):
        selected = torch.as_tensor(selected, dtype=torch.long, device=natural_train.device)
        if int(selected.min()) < 0 or int(selected.max()) >= len(natural_train):
            raise ValueError("graph.train_indices is outside the edge range")
        balanced = torch.zeros_like(natural_train)
        balanced[selected] = True
        # The balance list can contain duplicates only if a future resampler is
        # enabled; this stage only under-samples, so .bool() is enough.
        graph.train_mask = balanced
    else:
        graph.train_mask = natural_train.clone()

    graph.val_mask = _keep_n(natural_val, max_eval, seed + 1)
    graph.test_mask = _keep_n(natural_test, max_eval, seed + 2)
    graph.train_mask = _keep_n(graph.train_mask, max_train, seed)


def restore_masks(graphs: Sequence[Any], saved: Sequence[tuple[Any, Any, Any]]) -> None:
    for graph, (train, val, test) in zip(graphs, saved):
        graph.train_mask = train
        graph.val_mask = val
        graph.test_mask = test


def save_masks(graphs: Sequence[Any]) -> list[tuple[Any, Any, Any]]:
    return [
        (graph.train_mask.clone(), graph.val_mask.clone(), graph.test_mask.clone())
        for graph in graphs
    ]


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _labels_for(graphs: Sequence[Any]) -> list[int]:
    labels: set[int] = set()
    for graph in graphs:
        labels.update(int(value) for value in graph.y.unique().tolist())
    return sorted(labels)


def evaluate_matrix(
    model: torch.nn.Module,
    graphs: Sequence[Any],
    *,
    device: torch.device,
    upto: int,
    eval_all: bool,
    labels: Sequence[int],
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Evaluate checkpoint ``upto`` on test rows of periods 0..upto.

    Returns macro-F1 plus per-class F1 for the same cells. The per-class values
    are needed to see whether the macro average is being dragged down by rare
    classes rather than by the common ones.
    """
    scores: dict[str, float] = {}
    per_class: dict[str, dict[str, float]] = {}
    for period in range(upto + 1):
        if not eval_all and period != upto:
            continue
        result = evaluate_graph_batch(
            model,
            graphs[period],
            "test",
            device=device,
            labels=labels,
        )
        key = str(period + 1)
        scores[key] = float(result["macro_f1"])
        per_class[key] = dict(result.get("per_class_f1", {}))
    return scores, per_class


def matrix_from_scores(scores: dict[str, dict[str, float]], n_periods: int) -> list[list[float | None]]:
    matrix: list[list[float | None]] = [[None] * n_periods for _ in range(n_periods)]
    for checkpoint_key, row in scores.items():
        checkpoint = int(checkpoint_key) - 1
        for period_key, value in row.items():
            matrix[checkpoint][int(period_key) - 1] = float(value)
    return matrix


def _matrix_numpy(matrix: Sequence[Sequence[float | None]]) -> np.ndarray:
    out = np.full((len(matrix), len(matrix)), np.nan, dtype="float64")
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            if value is not None:
                out[i, j] = float(value)
    return out


# ---------------------------------------------------------------------------
# One training run
# ---------------------------------------------------------------------------


def run_id(mode: str, depth: int, seed: int, strategy: str) -> str:
    short = {"constant": "none", "reputation": "rep", "constant+reputation": "both"}[mode]
    return f"{strategy}-{short}-d{depth}-s{seed}"


def run_signature(
    *,
    mode: str,
    depth: int,
    seed: int,
    strategy: str,
    max_epochs: int,
    max_train: int | None,
    max_eval: int | None,
    use_balance: bool,
    n_periods: int,
    recipe: str = "",
) -> str:
    """Identity of a completed run for ``--resume``.

    The human-readable run id is kept short, but resuming must not skip a run
    when the evaluation budget, mask size, or number of periods changed.
    """
    return "|".join(
        [
            run_id(mode, depth, seed, strategy),
            f"epochs={max_epochs}",
            f"train={max_train if max_train is not None else 'all'}",
            f"eval={max_eval if max_eval is not None else 'all'}",
            f"balance={int(use_balance)}",
            f"periods={n_periods}",
            f"recipe={recipe}",
        ]
    )


def train_one_run(
    *,
    graphs: Sequence[Any],
    dataset: str,
    mode: str,
    depth: int,
    seed: int,
    strategy_name: str,
    settings: TrainSettings,
    device: torch.device,
    max_train: int | None,
    max_eval: int | None,
    use_balance: bool,
    eval_all: bool,
    epochs: int | None,
    checkpoint_root: Path | None = None,
) -> dict[str, Any]:
    saved = save_masks(graphs)
    try:
        apply_masks(
            graphs[0],
            use_balance=use_balance,
            max_train=max_train,
            max_eval=max_eval,
            seed=seed,
        )
        # Only the first graph is masked here; later periods are masked in the loop.
        labels = _labels_for(graphs)
        num_classes = int(max(int(graph.y.max()) for graph in graphs)) + 1
        model = build_model(
            "egraphsage",
            node_dim=int(graphs[0].x.shape[1]),
            edge_dim=int(graphs[0].edge_attr.shape[1]),
            config={
                "model": {
                    "num_layers": int(depth),
                    "hidden_dim": 128,
                    "dropout": 0.0,
                    "num_classes": num_classes,
                }
            },
        ).to(device)
        strategy = build_strategy(strategy_name, config={}, seed=seed)
        scores: dict[str, dict[str, float]] = {}
        per_class: dict[str, dict[str, dict[str, float]]] = {}
        histories: list[dict[str, Any]] = []
        checkpoints: list[str] = []
        started = time.monotonic()
        for period, graph in enumerate(graphs):
            if period > 0:
                apply_masks(
                    graph,
                    use_balance=use_balance,
                    max_train=max_train,
                    max_eval=max_eval,
                    seed=seed + period,
                )
            update = strategy.update(
                model,
                graph,
                settings=settings,
                seed=seed,
                device=str(device),
                epochs_override=epochs,
            )
            histories.append(
                {
                    "period": period + 1,
                    "val_macro_f1": update.get("best_val_macro_f1", update.get("best_macro_f1")),
                    "best_epoch": update.get("best_epoch"),
                    "epochs_run": update.get("epochs_run"),
                    "seconds": update.get("seconds", update.get("timing", {}).get("seconds")),
                }
            )
            if checkpoint_root is not None:
                checkpoint_dir = checkpoint_root / run_id(mode, depth, seed, strategy_name)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                checkpoint_path = checkpoint_dir / f"period_{period + 1}.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model": "egraphsage",
                        "node_dim": int(graphs[0].x.shape[1]),
                        "edge_dim": int(graphs[0].edge_attr.shape[1]),
                        "num_layers": int(depth),
                        "hidden_dim": 128,
                        "dropout": 0.0,
                        "num_classes": num_classes,
                        "node_features": mode,
                        "period": period + 1,
                        "seed": int(seed),
                        "strategy": strategy_name,
                        "loss_type": settings.loss_type,
                        "focal_gamma": float(settings.focal_gamma),
                        "class_weight_cap": float(settings.class_weight_cap),
                        "balanced_batches": bool(settings.balanced_batches),
                        "node_noise_std": float(settings.node_noise_std),
                    },
                    checkpoint_path,
                )
                checkpoints.append(str(checkpoint_path))
            row_scores, row_per_class = evaluate_matrix(
                model,
                graphs,
                device=device,
                upto=period,
                eval_all=eval_all,
                labels=labels,
            )
            scores[str(period + 1)] = row_scores
            per_class[str(period + 1)] = row_per_class
            line("OK", f"period {period + 1} checkpoint; test F1 {row_scores}")
    finally:
        restore_masks(graphs, saved)

    matrix = matrix_from_scores(scores, len(graphs))
    cl = cl_metrics(_matrix_numpy(matrix))
    return {
        "run_id": run_id(mode, depth, seed, strategy_name),
        "run_key": run_signature(
            mode=mode,
            depth=depth,
            seed=seed,
            strategy=strategy_name,
            max_epochs=int(settings.max_epochs),
            max_train=max_train,
            max_eval=max_eval,
            use_balance=use_balance,
            n_periods=len(graphs),
        ),
        "dataset": dataset,
        "node_features": mode,
        "depth": int(depth),
        "seed": int(seed),
        "strategy": strategy_name,
        "n_periods": len(graphs),
        "eval_all_periods": bool(eval_all),
        "train_mask": "balanced" if use_balance else "natural",
        "max_train": max_train,
        "max_eval": max_eval,
        "matrix": matrix,
        "scores": scores,
        "matrix_per_class": per_class,
        "cl_metrics": cl,
        "history": histories,
        "checkpoints": checkpoints,
        "loss_type": settings.loss_type,
        "focal_gamma": float(settings.focal_gamma),
        "class_weight_cap": float(settings.class_weight_cap),
        "balanced_batches": bool(settings.balanced_batches),
        "node_noise_std": float(settings.node_noise_std),
        "seconds": round(time.monotonic() - started, 2),
        "split_manifest_hash": getattr(graphs[0], "split_manifest_hash", None),
    }


# ---------------------------------------------------------------------------
# Figures and report
# ---------------------------------------------------------------------------


def make_figures(
    runs: Sequence[dict[str, Any]],
    *,
    dataset: str,
    out_dir: Path,
) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - figures are optional
        return []
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    modes = [mode for mode in NODE_MODES if any(r["node_features"] == mode for r in runs)]
    depths = sorted({int(r["depth"]) for r in runs})

    def values(mode: str, key: str) -> list[float]:
        out: list[float] = []
        for depth in depths:
            vals = [
                float(r["cl_metrics"][key])
                for r in runs
                if r["node_features"] == mode
                and int(r["depth"]) == depth
                and r.get("cl_metrics")
                and r["cl_metrics"].get(key) is not None
            ]
            out.append(sum(vals) / len(vals) if vals else 0.0)
        return out

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    width = 0.8 / max(len(modes), 1)
    offsets = np.arange(len(depths))
    for index, mode in enumerate(modes):
        ax.bar(offsets + (index - (len(modes) - 1) / 2) * width, values(mode, "final_average_macro_f1"), width, label=NODE_LABELS[mode])
    ax.set_xticks(offsets)
    ax.set_xticklabels([f"d{depth}" for depth in depths])
    ax.set_ylabel("final average macro-F1")
    ax.set_title(f"{dataset}: depth vs final F1 by node features")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "fig01_depth_final_f1.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for index, mode in enumerate(modes):
        ax.bar(offsets + (index - (len(modes) - 1) / 2) * width, values(mode, "forgetting"), width, label=NODE_LABELS[mode])
    ax.set_xticks(offsets)
    ax.set_xticklabels([f"d{depth}" for depth in depths])
    ax.set_ylabel("forgetting (A11)")
    ax.set_title(f"{dataset}: depth vs forgetting")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "fig02_depth_forgetting.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # Final-row F1 by period (best depth per mode).
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    for mode in modes:
        subset = [r for r in runs if r["node_features"] == mode]
        if not subset:
            continue
        best = max(subset, key=lambda r: (r["cl_metrics"] or {}).get("final_average_macro_f1") or -1)
        row = best["matrix"][-1]
        ax.plot(range(1, len(row) + 1), [0.0 if value is None else value for value in row], marker="o", label=f"{NODE_LABELS[mode]} d{best['depth']}")
    ax.set_xlabel("evaluated period")
    ax.set_ylabel("test macro-F1 at final checkpoint")
    ax.set_title(f"{dataset}: final-row F1 by period")
    ax.legend(frameon=False)
    fig.tight_layout()
    path = out_dir / "fig03_final_row.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)

    # Reputation delta relative to constant at the same depth.
    # Average across seeds: comparing one arbitrary seed against another would
    # turn seed noise into an apparent feature effect.
    if "constant" in modes and len(modes) > 1:
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        base: dict[int, float] = {}
        for depth in depths:
            values = [
                float(r["cl_metrics"]["final_average_macro_f1"])
                for r in runs
                if r["node_features"] == "constant"
                and int(r["depth"]) == depth
                and r.get("cl_metrics")
            ]
            if values:
                base[depth] = sum(values) / len(values)
        for mode in modes:
            if mode == "constant":
                continue
            deltas = []
            for depth in depths:
                values = [
                    float(r["cl_metrics"]["final_average_macro_f1"])
                    for r in runs
                    if r["node_features"] == mode
                    and int(r["depth"]) == depth
                    and r.get("cl_metrics")
                ]
                if not values or depth not in base:
                    deltas.append(float("nan"))
                    continue
                deltas.append(sum(values) / len(values) - base[depth])
            ax.plot(depths, deltas, marker="o", label=NODE_LABELS[mode])
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_xlabel("depth")
        ax.set_ylabel("delta final macro-F1 vs constant")
        ax.set_title(f"{dataset}: reputation effect by depth (seed mean)")
        ax.legend(frameon=False)
        fig.tight_layout()
        path = out_dir / "fig04_reputation_delta.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    # Per-period training/validation curves for the first run (diagnostic).
    if runs:
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        for run in runs[: min(6, len(runs))]:
            history = run.get("history") or []
            ax.plot(
                [item["period"] for item in history],
                [item["val_macro_f1"] or 0.0 for item in history],
                marker="o",
                label=run["run_id"],
            )
        ax.set_xlabel("training period")
        ax.set_ylabel("best validation macro-F1")
        ax.set_title(f"{dataset}: validation curves")
        ax.legend(frameon=False, fontsize=7)
        fig.tight_layout()
        path = out_dir / "fig05_validation_curves.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)
    return paths


def write_report(runs: Sequence[dict[str, Any]], *, dataset: str, out_dir: Path, figures: Sequence[Path]) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{dataset}.md"
    modes = [mode for mode in NODE_MODES if any(r["node_features"] == mode for r in runs)]
    depths = sorted({int(r["depth"]) for r in runs})
    lines = [
        f"# Step 06 training results — {dataset}",
        "",
        "This report is generated by `src/main/06_train.py`. The protocol is the frozen",
        "C1 split, period-1 train-only preprocessing and the canonical `graph_batch` trainer.",
        "Smoke/pilot runs are engineering checks; when these tables contain five seeds per",
        "cell they are the full 5-seed matrix, not a pilot.",
        "",
        "## Runs",
        "",
        "| run | node features | depth | seed | final average F1 | forgetting | BWT |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for run in runs:
        block = run.get("cl_metrics") or {}
        def fmt(value: Any) -> str:
            return "—" if value is None else f"{float(value):.4f}"
        lines.append(
            f"| {run['run_id']} | {run['node_features']} | {run['depth']} | {run['seed']} | "
            f"{fmt(block.get('final_average_macro_f1'))} | {fmt(block.get('forgetting'))} | {fmt(block.get('bwt'))} |"
        )
    lines += [
        "",
        "## Summary by node mode and depth",
        "",
        "| node mode | depth | n | final F1 (mean ± sd) | forgetting (mean ± sd) |",
        "|---|---:|---:|---:|---:|",
    ]
    for mode in modes:
        for depth in depths:
            group = [
                r for r in runs
                if r["node_features"] == mode and int(r["depth"]) == depth
            ]
            if not group:
                continue
            f1 = [float(r["cl_metrics"]["final_average_macro_f1"]) for r in group if r.get("cl_metrics")]
            fg = [float(r["cl_metrics"]["forgetting"]) for r in group if r.get("cl_metrics")]
            def _ms(values: list[float]) -> str:
                if not values:
                    return "—"
                avg = sum(values) / len(values)
                if len(values) < 2:
                    return f"{avg:.4f}"
                return f"{avg:.4f} ± {statistics.stdev(values):.4f}"
            lines.append(
                f"| {mode} | {depth} | {len(group)} | {_ms(f1)} | {_ms(fg)} |"
            )

    lines += [
        "",
        "## Paired seed-level comparisons",
        "",
        "A mean is shown with the number of seeds where the first variant is higher;",
        "these are exploratory paired differences, not multiplicity-corrected tests.",
        "",
        "| comparison | n | mean delta | seeds higher |",
        "|---|---:|---:|---:|",
    ]
    by_key = {
        (r["node_features"], int(r["depth"]), int(r["seed"])): float(
            r["cl_metrics"]["final_average_macro_f1"]
        )
        for r in runs
        if r.get("cl_metrics")
    }
    comparisons: list[tuple[str, str, str, int | None, int | None]] = []
    for depth in depths:
        for label, a, b in (
            ("reputation − constant", "reputation", "constant"),
            ("constant+reputation − constant", "constant+reputation", "constant"),
        ):
            comparisons.append((f"{label} (d{depth})", a, b, depth, None))
    for mode in modes:
        for d1, d2 in ((2, 3), (2, 4), (3, 4)):
            comparisons.append((f"{mode} d{d2} − d{d1}", mode, mode, d1, d2))
    for label, a, b, d1, d2 in comparisons:
        pairs: list[float] = []
        for seed in sorted({int(r["seed"]) for r in runs}):
            if d2 is None:
                left = by_key.get((a, int(d1), seed))
                right = by_key.get((b, int(d1), seed))
            else:
                left = by_key.get((a, int(d2), seed))
                right = by_key.get((b, int(d1), seed))
            if left is not None and right is not None:
                pairs.append(left - right)
        if not pairs:
            continue
        avg = sum(pairs) / len(pairs)
        higher = sum(1 for value in pairs if value > 0)
        lines.append(f"| {label} | {len(pairs)} | {avg:+.4f} | {higher}/{len(pairs)} |")

    lines += [
        "",
        "## R matrices",
        "",
    ]
    for run in runs:
        lines.append(f"### {run['run_id']}")
        lines.append("")
        lines.append("| checkpoint \\ evaluated period | " + " | ".join(str(i + 1) for i in range(run["n_periods"])) + " |")
        lines.append("|---|" + "---|" * run["n_periods"])
        for i, row in enumerate(run["matrix"]):
            cells = ["—" if value is None else f"{value:.4f}" for value in row]
            lines.append(f"| after P{i + 1} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [
        "## Figures",
        "",
        *[f"- `{figure.name}`" for figure in figures],
        "",
        "## Reading notes",
        "",
        "- `final_average_macro_f1` is the mean over periods of the last checkpoint row.",
        "- A11 forgetting is `max_{i>=j} R[i,j] - R[T-1,j]`; forward cells are not used.",
        "- The node mode `constant` is the canonical E-GraphSAGE ones vector; `reputation` uses",
        "  causal prior-period host statistics only; `constant+reputation` concatenates both.",
        "- No training result here is a claim that reputation or depth is better; inspect the",
        "  paired seed runs before making an experiment-level conclusion.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare step-05 graph variants.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="dataset name or 'all'")
    parser.add_argument("--root", default=None, help="project root")
    parser.add_argument("--node-features", default="constant,reputation,constant+reputation")
    parser.add_argument("--depths", default="2,3,4")
    parser.add_argument("--strategy", default="ft", choices=("ft", "replay", "ewc"))
    parser.add_argument("--seeds", default="42", help="comma-separated seed list")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--periods", type=int, default=5, choices=range(1, 6))
    parser.add_argument("--epochs", type=int, default=None, help="override max epochs")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-eval", type=int, default=None)
    parser.add_argument("--use-balance", action="store_true", help="use step-04 selected train indices")
    parser.add_argument("--loss", default="ce", choices=("ce", "focal"), help="loss for imbalance experiments")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--class-weight-cap", type=float, default=0.0, help="clip class weights at this value (0=off)")
    parser.add_argument("--balanced-batches", action="store_true", help="graph_batch: sample training edges ∝ 1/class_count")
    parser.add_argument("--node-noise", type=float, default=0.0, help="std of Gaussian node-feature jitter (0=off)")
    parser.add_argument("--save-checkpoints", action="store_true", help="save per-period model payloads for explanations")
    parser.add_argument("--coarse-taxonomy", action="store_true", help="merge Analysis/Backdoor/DoS/Worms into one class")
    parser.add_argument("--resume", action="store_true", help="skip run IDs already present in runs.json")
    parser.add_argument(
        "--diagonal-only",
        action="store_true",
        help="evaluate only the current period at each checkpoint (faster, but no forgetting/BWT)",
    )
    parser.add_argument("--smoke", action="store_true", help="one period, 5000 train / 2000 eval edges, one epoch")
    return parser.parse_args(argv)


def _resolve_device(value: str) -> torch.device:
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(value)


def write_progress(out_dir: Path, dataset: str, runs: Sequence[dict[str, Any]]) -> None:
    """Write completed runs after every model, so a 12 h Kaggle cap cannot wipe
    the whole matrix. The final block below rewrites the same files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": dataset,
        "protocol": "frozen C1 split; period-1 train-only preprocessor; graph_batch",
        "runs": list(runs),
    }
    (out_dir / "runs.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    rows = []
    for run in runs:
        block = run["cl_metrics"]
        rows.append(
            {
                "run_id": run["run_id"],
                "node_features": run["node_features"],
                "depth": run["depth"],
                "seed": run["seed"],
                "final_average_macro_f1": block.get("final_average_macro_f1"),
                "forgetting": block.get("forgetting"),
                "bwt": block.get("bwt"),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "runs.csv", index=False)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = find_root(args.root)
    device = _resolve_device(args.device)
    datasets = choose_datasets(args.dataset, root)
    modes = [item.strip() for item in args.node_features.split(",") if item.strip()]
    for mode in modes:
        if mode not in NODE_MODES:
            raise SystemExit(f"unknown --node-features {mode!r}; choose from {NODE_MODES}")
    depths = [int(item) for item in args.depths.split(",") if item.strip()]
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    if args.smoke:
        args.periods = 1
        args.epochs = 1
        args.max_train = args.max_train or 5000
        args.max_eval = args.max_eval or 2000
    config = load_yaml(root / args.config)
    settings = TrainSettings.from_config(config)
    settings = replace(
        settings,
        loss_type=args.loss,
        focal_gamma=float(args.focal_gamma),
        class_weight_cap=float(args.class_weight_cap),
        balanced_batches=bool(args.balanced_batches),
        node_noise_std=float(args.node_noise),
    )
    if args.epochs is not None:
        settings = replace(settings, max_epochs=int(args.epochs))
    recipe = "|".join(
        [
            args.loss,
            f"g{args.focal_gamma:g}",
            f"cap{args.class_weight_cap:g}",
            f"bal{int(args.balanced_batches)}",
            f"noise{args.node_noise:g}",
            f"coarse{int(args.coarse_taxonomy)}",
        ]
    )

    out_root = root / "results" / "text" / "train"
    figure_root = root / "results" / "images" / "train"
    report_root = root / "analysis" / "train"

    all_runs: list[dict[str, Any]] = []
    for dataset in datasets:
        cell(1, 4, f"LOAD GRAPHS: {dataset}")
        out_dir = out_root / dataset
        checkpoint_root = root / "datasets" / "processed" / dataset / "models" if args.save_checkpoints else None
        dataset_runs: list[dict[str, Any]] = []
        completed_keys: set[str] = set()
        if args.resume and (out_dir / "runs.json").is_file():
            try:
                previous = json.loads((out_dir / "runs.json").read_text(encoding="utf-8")).get("runs", [])
            except (OSError, ValueError):
                previous = []
            dataset_runs.extend(previous)
            all_runs.extend(previous)
            completed_keys = {str(run.get("run_key") or run.get("run_id")) for run in previous}
            line("OK", f"{dataset}: resuming with {len(previous)} completed run(s)")
        for mode in modes:
            paths = graph_paths(root, dataset, mode)[: args.periods]
            if not all(path.is_file() for path in paths):
                line("SKIP", f"{dataset} / {mode}: graph snapshots missing; run 05_build_graphs.py first")
                continue
            graphs = load_graphs(paths, str(device))
            if args.coarse_taxonomy:
                for graph in graphs:
                    mapped = [COARSE_MAP[int(value)] for value in graph.y.detach().cpu().tolist()]
                    graph.y = torch.as_tensor(mapped, dtype=torch.long, device=graph.y.device)
                line("OK", f"{dataset} / {mode}: coarse 7-class taxonomy enabled")
            line("OK", f"{dataset} / {mode}: {len(graphs)} graphs; edges {[int(g.edge_index.shape[1]) for g in graphs]}")
            for depth in depths:
                for seed in seeds:
                    run_name = run_id(mode, depth, seed, args.strategy)
                    signature = run_signature(
                        mode=mode,
                        depth=depth,
                        seed=seed,
                        strategy=args.strategy,
                        max_epochs=int(settings.max_epochs),
                        max_train=args.max_train,
                        max_eval=args.max_eval,
                        use_balance=args.use_balance,
                        n_periods=len(graphs),
                        recipe=recipe,
                    )
                    if signature in completed_keys:
                        line("SKIP", f"{run_name}: already complete for this configuration")
                        continue
                    cell(2, 4, f"TRAIN {dataset} / {mode} / depth {depth} / seed {seed}")
                    t0 = time.monotonic()
                    result = train_one_run(
                        graphs=graphs,
                        dataset=dataset,
                        mode=mode,
                        depth=depth,
                        seed=seed,
                        strategy_name=args.strategy,
                        settings=settings,
                        device=device,
                        max_train=args.max_train,
                        max_eval=args.max_eval,
                        use_balance=args.use_balance,
                        eval_all=not args.diagonal_only,
                        epochs=args.epochs,
                        checkpoint_root=checkpoint_root,
                    )
                    result["seconds_total"] = round(time.monotonic() - t0, 2)
                    all_runs.append(result)
                    dataset_runs.append(result)
                    completed_keys.add(signature)
                    write_progress(out_dir, dataset, dataset_runs)
                    line("OK", f"{result['run_id']}: final F1 {result['cl_metrics'].get('final_average_macro_f1')}")
            del graphs
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
        if not dataset_runs:
            line("WARN", f"no runs for {dataset}")
            continue
        cell(3, 4, f"WRITE RESULTS: {dataset}")
        write_progress(out_dir, dataset, dataset_runs)
        figures = make_figures(dataset_runs, dataset=dataset, out_dir=figure_root / dataset)
        report = write_report(dataset_runs, dataset=dataset, out_dir=report_root, figures=figures)
        line("OK", f"{len(dataset_runs)} runs, {len(figures)} figures, report {report.relative_to(root)}")

    cell(4, 4, "SUMMARY")
    if not all_runs:
        line("FAIL", "no runs completed")
        return 1
    for run in all_runs:
        block = run["cl_metrics"]
        line(
            "OK",
            f"{run['dataset']} / {run['run_id']}: "
            f"F1={block.get('final_average_macro_f1')} forget={block.get('forgetting')} "
            f"({run['seconds_total']}s)",
        )
    print("\nNext: inspect the depth/reputation table, then run the same matrix on Kaggle")
    print("with 5 seeds if the smoke/pilot results are stable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
