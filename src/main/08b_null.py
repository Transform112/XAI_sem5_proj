#!/usr/bin/env python
"""Build a model-null IG floor on the fixed step-07 anchors.

This is the cheap half of step 08B.  It retrains the baseline model on each
period's own training split (same graph, same split, same recipe) with several
new seeds, explains the **same stored anchors** with each reseeded model, and
measures the pairwise drift among those null explanations.  That distribution
answers: "how much does an IG explanation move when only model randomness
changes?"  It is not a claim about real drift.

The expensive GNNExplainer half is intentionally not run here; add it once the
IG floor is understood.  The output is deliberately separate from signed S8:
``results/text/tables/null_ig/`` and ``results/text/data_audit/null_ig/``.
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch_geometric.data import Data

from src.analysis.metrics_drift import attribution_drift
from src.explain.ig import integrated_gradients
from src.models.edge_sage import build_model
from src.train.trainer import TrainSettings, train_edge_classifier
from src.utils.seed import seed_everything


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--corpus-root", default="src/kaggle_lane/outputs/xdrift-step07-corpus/corpus")
    parser.add_argument("--graph-root", default="datasets/graphs")
    parser.add_argument("--dataset", default="NF-UNSW-NB15-v3")
    parser.add_argument("--run-id", default="ft-none-d2-s42")
    parser.add_argument("--periods", default="1,2,3,4,5")
    parser.add_argument("--max-anchors-per-period", type=int, default=100)
    parser.add_argument("--null-seeds", type=int, default=5)
    parser.add_argument("--first-null-seed", type=int, default=142)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--out-root", default="results/text")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _anchor_data(anchor: dict[str, Any], node_dim: int) -> Data:
    return Data(
        x=torch.ones((int(anchor["n_sub_nodes"]), node_dim), dtype=torch.float32),
        edge_index=anchor["sub_edge_index"].long(),
        edge_attr=anchor["sub_edge_attr"].float(),
        y=torch.as_tensor([int(anchor["label"])], dtype=torch.long),
    )


def _model_from_checkpoint(payload: dict[str, Any]) -> torch.nn.Module:
    model_cfg = {
        "hidden_dim": int(payload["hidden_dim"]),
        "num_layers": int(payload["num_layers"]),
        "dropout": float(payload["dropout"]),
        "num_classes": int(payload["num_classes"]),
    }
    return build_model(
        str(payload["model"]),
        node_dim=int(payload["node_dim"]),
        edge_dim=int(payload["edge_dim"]),
        config={"model": model_cfg},
    )


def _anchors_by_period(corpus_root: Path, dataset: str, run_id: str) -> list[dict[str, Any]]:
    root = corpus_root / "datasets" / "processed" / dataset / "explanations" / "anchors" / run_id
    return [torch.load(root / f"period_{period}.pt", map_location="cpu", weights_only=False) for period in range(1, 6)]


def _train_null_model(graph: Any, payload: dict[str, Any], *, seed: int, epochs: int, device: torch.device):
    seed_everything(seed)
    model = _model_from_checkpoint(payload)
    settings = TrainSettings(
        lr=1e-3,
        batch_size=1024,
        max_epochs=int(epochs),
        patience=5,
        class_weights="balanced",
        eval_metric="macro_f1",
        batching="graph_batch",
        neighbor_fanout=(10, 10),
        max_edges_per_step=60_000,
        optimized_dispatch=True,
        sampler_fast_path=True,
    )
    train_edge_classifier(model, graph, settings=settings, seed=seed, device=str(device))
    model.to(device).eval()
    return model


def _q95(values: np.ndarray) -> float:
    return float(np.percentile(values, 95)) if len(values) else float("nan")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    corpus = Path(args.corpus_root)
    dataset = args.dataset
    periods = [int(p) for p in args.periods.split(",") if p.strip()]
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device if args.device != "auto" else "cpu")

    checkpoint_dir = corpus / "datasets" / "processed" / dataset / "models" / args.run_id
    graph_dir = Path(args.graph_root) / dataset / "constant"
    anchor_bundles = _anchors_by_period(corpus, dataset, args.run_id)

    plan = {
        "dataset": dataset,
        "run_id": args.run_id,
        "periods": periods,
        "checkpoint_dir": str(checkpoint_dir),
        "graph_dir": str(graph_dir),
        "max_anchors_per_period": args.max_anchors_per_period,
        "null_seeds": args.null_seeds,
        "epochs": args.epochs,
        "device": str(device),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return 0

    out_root = Path(args.out_root)
    table_dir = out_root / "tables" / "null_ig"
    audit_dir = out_root / "data_audit" / "null_ig"
    table_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    started_all = time.monotonic()
    for period in periods:
        checkpoint = checkpoint_dir / f"period_{period}.pt"
        graph_file = graph_dir / f"period_{period}.pt"
        if not checkpoint.is_file() or not graph_file.is_file():
            raise FileNotFoundError(f"missing checkpoint/graph for period {period}: {checkpoint}, {graph_file}")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        graph = torch.load(graph_file, map_location="cpu", weights_only=False)
        anchors = list(anchor_bundles[period - 1].get("anchors", []))[: args.max_anchors_per_period]
        if not anchors:
            print(f"[null] period {period}: no anchors, skipped")
            continue

        started = time.monotonic()
        null_models = []
        for index in range(args.null_seeds):
            seed = args.first_null_seed + index
            model = _train_null_model(graph, payload, seed=seed, epochs=args.epochs, device=device)
            null_models.append((seed, model))
            print(f"[null] period {period}: trained null seed {seed} ({time.monotonic() - started:.1f}s)")

        # Explain every anchor once per null model.  IG is deterministic given a
        # model, so this isolates model randomness rather than explainer noise.
        arrays: dict[int, list[np.ndarray]] = {}
        for anchor in anchors:
            anchor_id = int(anchor["anchor_id"])
            arrays[anchor_id] = []
            data = _anchor_data(anchor, node_dim=int(payload["node_dim"]))
            target_local = int(anchor["target_local_index"][0])
            for _, model in null_models:
                record = integrated_gradients(model, data, target_local, steps=32, device=str(device))
                arrays[anchor_id].append(np.asarray(record["attributions"], dtype=np.float64))

        for anchor in anchors:
            anchor_id = int(anchor["anchor_id"])
            label = int(anchor["label"])
            for (seed_a, _), (seed_b, _) in combinations(null_models, 2):
                drift = attribution_drift(arrays[anchor_id][seed_a - args.first_null_seed], arrays[anchor_id][seed_b - args.first_null_seed])
                rows.append({
                    "anchor_id": anchor_id,
                    "label": label,
                    "period": period,
                    "explainer": "ig",
                    "seed_a": seed_a,
                    "seed_b": seed_b,
                    "primary_drift": float(drift["jaccard_top5"]),
                    "cosine": float(drift["cosine"]),
                    "spearman": float(drift["spearman"]),
                })
        print(f"[null] period {period}: {len(anchors)} anchors, {len(rows)} accumulated rows ({time.monotonic() - started:.1f}s)")

    import pandas as pd

    frame = pd.DataFrame(rows)
    frame.to_csv(table_dir / "null_rows.csv", index=False)
    floor_rows = []
    if len(frame):
        for label, group in frame.groupby("label"):
            values = group["primary_drift"].to_numpy()
            floor_rows.append({"explainer": "ig", "label": int(label), "count": int(len(values)), "q95": _q95(values), "mean": float(np.mean(values))})
        all_values = frame["primary_drift"].to_numpy()
        floor_rows.append({"explainer": "ig", "label": -1, "count": int(len(all_values)), "q95": _q95(all_values), "mean": float(np.mean(all_values))})
    floor = pd.DataFrame(floor_rows)
    floor.to_csv(table_dir / "null_floor.csv", index=False)
    summary = {
        "dataset": dataset,
        "run_id": args.run_id,
        "periods": periods,
        "anchors_per_period": args.max_anchors_per_period,
        "null_seeds": args.null_seeds,
        "epochs": args.epochs,
        "rows": int(len(frame)),
        "floor_rows": int(len(floor)),
        "elapsed_s": round(time.monotonic() - started_all, 2),
        "device": str(device),
    }
    (audit_dir / "null_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[OK] null rows {len(frame)} -> {table_dir}")
    print(f"[OK] floors {len(floor)} -> {table_dir / 'null_floor.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
