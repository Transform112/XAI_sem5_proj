#!/usr/bin/env python
"""Step 07: build an explanation corpus from saved period checkpoints.

For each period ``j`` this script:

1. loads the frozen graph snapshot and checkpoint ``M_j``;
2. selects correct, confident test anchors with a fixed 2-hop context
   (``src.anchors.build.select_anchors``);
3. explains every anchor with every checkpoint ``M_t`` for ``t >= j``;
4. stores IG feature attributions and GNNExplainer edge masks in the
   explanation cache, plus a small JSON summary.

The model is never retrained.  The graph context is sampled once per anchor
and reused across checkpoints, which is what makes later drift comparisons
meaningful.

This is the corpus writer; drift/RQ analysis lives in later steps.

Examples
--------
Contract test (5 anchors, one period, both explainers)::

    python src/main/07_explain.py --dataset NF-UNSW-NB15-v3 \
        --run-id ft-none-d2-s42 --periods 1 --max-anchors 5

Full seed-42 corpus (all periods/checkpoints)::

    python src/main/07_explain.py --dataset NF-UNSW-NB15-v3 \
        --run-id ft-none-d2-s42 --periods 1,2,3,4,5

Context experiment (3-hop, 5,000-edge cap)::

    python src/main/07_explain.py ... --hops 3 --edge-cap 5000

Outputs (all new; old signed S7 files are never touched):

- ``datasets/processed/<dataset>/explanations/anchors/<run_id>/period_<j>.pt``
- ``results/text/explanations/`` (cache; gitignored)
- ``results/text/tables/explain_index_main.csv``
- ``results/text/data_audit/explanations/<dataset>/<run_id>/period_<j>.json``
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def find_root() -> Path:
    """Find the repository root (src/ + configs/ markers)."""
    candidate = _ROOT
    if (candidate / "src").is_dir() and (candidate / "configs").is_dir():
        return candidate
    raise RuntimeError(f"cannot find repository root from {_ROOT}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="NF-UNSW-NB15-v3")
    parser.add_argument("--run-id", default="ft-none-d2-s42",
                        help="checkpoint run id under datasets/processed/<dataset>/models/")
    parser.add_argument("--graph-mode", default="constant",
                        help="graph node-feature mode (constant for baseline)")
    parser.add_argument("--periods", default="1,2,3,4,5",
                        help="anchor periods, comma separated")
    parser.add_argument("--checkpoint-root", default=None,
                        help="optional root containing <run_id>/period_<i>.pt")
    parser.add_argument("--graph-root", default=None,
                        help="optional root containing <mode>/period_<i>.pt")
    parser.add_argument("--explainers", default="ig,gnn_explainer")
    parser.add_argument("--k", default="5,10", help="IG feature k values")
    parser.add_argument("--max-anchors", type=int, default=0,
                        help="cap anchors per period (0 = use selector cap)")
    parser.add_argument("--anchor-tag", default=None,
                        help="separate anchor/cache namespace for context experiments")
    parser.add_argument("--hops", type=int, default=2,
                        help="anchor context hops (2/3/4 for context experiments)")
    parser.add_argument("--fanout", type=int, default=10)
    parser.add_argument("--edge-cap", type=int, default=2000)
    parser.add_argument("--ig-steps", type=int, default=32)
    parser.add_argument("--gnn-epochs", type=int, default=100)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--max-explanations", type=int, default=0,
                        help="stop after this many explanations (0 = no cap)")
    parser.add_argument("--no-cache", action="store_true", help="do not write cache")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit("PyTorch is required for explanations; install platforms/requirements/common.txt") from exc
    return torch


def _numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("NumPy is required for explanations") from exc
    return np


def _checkpoint_path(root: Path, dataset: str, run_id: str, period: int) -> Path:
    return root / dataset / "models" / run_id / f"period_{int(period)}.pt"


def _graph_path(root: Path, dataset: str, mode: str, period: int) -> Path:
    return root / dataset / mode / f"period_{int(period)}.pt"


def load_model(path: Path, device: str):
    """Load one step-06 checkpoint into an eval model."""
    torch = _torch()
    from src.models.edge_sage import build_model

    payload = torch.load(path, map_location=torch.device(device), weights_only=False)
    model_cfg = {
        "hidden_dim": int(payload["hidden_dim"]),
        "num_layers": int(payload["num_layers"]),
        "dropout": float(payload["dropout"]),
        "num_classes": int(payload["num_classes"]),
    }
    model = build_model(
        str(payload["model"]),
        node_dim=int(payload["node_dim"]),
        edge_dim=int(payload["edge_dim"]),
        config={"model": model_cfg},
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(torch.device(device)).eval()
    return model, payload


def anchor_to_data(anchor: Mapping[str, Any], *, node_dim: int):
    """Turn a stored anchor subgraph into a PyG ``Data`` object."""
    torch = _torch()
    from torch_geometric.data import Data

    n_nodes = int(anchor["n_sub_nodes"])
    # The current baseline uses constant node features.  Reputation contexts
    # need stored sub_x from the builder; reject them rather than silently
    # explaining the wrong input.
    if node_dim != 8:
        raise ValueError("only constant 8-d node features are supported by this writer")
    return Data(
        x=torch.ones((n_nodes, node_dim), dtype=torch.float32),
        edge_index=anchor["sub_edge_index"].long(),
        edge_attr=anchor["sub_edge_attr"].float(),
        y=torch.as_tensor([int(anchor["label"])], dtype=torch.long),
    )


def select_period_anchors(
    *,
    root: Path,
    checkpoint_root: Path,
    graph_root: Path,
    dataset: str,
    run_id: str,
    graph_mode: str,
    period: int,
    hops: int,
    fanout: int,
    edge_cap: int,
    device: str,
    max_anchors: int,
    anchor_tag: str = "",
) -> tuple[dict[str, Any], Path]:
    """Select anchors for one period and save the tensor bundle."""
    torch = _torch()
    from src.anchors.build import saved_anchor_summary, select_anchors

    checkpoint = _checkpoint_path(checkpoint_root, dataset, run_id, period)
    graph_file = _graph_path(graph_root, dataset, graph_mode, period)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"missing checkpoint: {checkpoint}")
    if not graph_file.is_file():
        raise FileNotFoundError(f"missing graph: {graph_file}")

    model, payload = load_model(checkpoint, device)
    graph = torch.load(graph_file, map_location="cpu", weights_only=False)
    graph.period = int(period)

    anchor_cfg = {
        "num_hops": int(hops),
        "fanout": int(fanout),
        "max_subgraph_edges": int(edge_cap),
        "sampling_seed": int(payload.get("seed", 42)),
    }
    anchor_set = select_anchors(model, graph, config=anchor_cfg, device=device,
                                seed=int(anchor_cfg["sampling_seed"]))
    if max_anchors > 0:
        anchor_set["anchors"] = anchor_set["anchors"][: int(max_anchors)]
        anchor_set["n_selected"] = len(anchor_set["anchors"])

    out_dir = root / "datasets" / "processed" / dataset / "explanations" / "anchors" / run_id
    if anchor_tag:
        out_dir = out_dir / anchor_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"period_{period}.pt"
    torch.save(anchor_set, out_file)

    summary_dir = root / "results" / "text" / "data_audit" / "explanations" / dataset / run_id
    if anchor_tag:
        summary_dir = summary_dir / anchor_tag
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary = saved_anchor_summary(anchor_set)
    summary.update({
        "dataset": dataset,
        "run_id": run_id,
        "graph_mode": graph_mode,
        "checkpoint": str(checkpoint),
        "checkpoint_sha": _sha256(checkpoint),
        "graph": str(graph_file),
        "graph_sha": _sha256(graph_file),
        "context": {"hops": int(hops), "fanout": int(fanout), "edge_cap": int(edge_cap)},
        "anchor_file": str(out_file),
        "anchor_file_bytes": out_file.stat().st_size,
    })
    (summary_dir / f"period_{period}_anchors.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return anchor_set, out_file


def explain_anchor(
    *,
    model,
    anchor: Mapping[str, Any],
    node_dim: int,
    device: str,
    explainers: Sequence[str],
    k_values: Sequence[int],
    ig_steps: int,
    gnn_epochs: int,
    gnn_seed: int,
    config_sha: str,
    checkpoint_sha: str,
    checkpoint_period: int,
    anchor_period: int,
    dataset: str,
    run_id: str,
    cache,
    random_trials: int,
    fidelity_seed: int,
) -> list[dict[str, Any]]:
    """Explain one anchor and put IG/GNNExplainer results into the cache."""
    from src.explain.cache import cache_key
    from src.explain.fidelity import edge_mask_fidelity, feature_fidelity
    from src.explain.gnn_explainer import gnn_explainer
    from src.explain.ig import integrated_gradients, top_k_features

    data = anchor_to_data(anchor, node_dim=node_dim).to(device)
    target_local = int(anchor["target_local_index"][0])
    records: list[dict[str, Any]] = []

    if "ig" in explainers:
        start = time.monotonic()
        ig = integrated_gradients(model, data, target_local, steps=ig_steps, device=device)
        fidelity = feature_fidelity(
            model, data, target_local, ig["attributions"],
            k_values=k_values, random_trials=random_trials, seed=fidelity_seed, device=device,
        )
        runtime = round(time.monotonic() - start, 4)
        fidelity_by_k = {int(item["k"]): item for item in fidelity}
        base_record = {
            "anchor_id": int(anchor["anchor_id"]),
            "anchor_period": int(anchor_period),
            "checkpoint_period": int(checkpoint_period),
            "label": int(anchor["label"]),
            "explainer": "ig",
            "runtime_s": runtime,
            "fidelity": fidelity,
            "top_features": {str(k): top_k_features(ig["attributions"], k) for k in k_values},
        }
        records.append(base_record)
        if cache is not None:
            for k in k_values:
                key = cache_key(
                    checkpoint_sha=checkpoint_sha,
                    anchor_id=anchor["anchor_id"],
                    explainer="ig",
                    seed=int(gnn_seed),
                    k=int(k),
                    config_sha=config_sha,
                )
                fidelity_k = fidelity_by_k.get(int(k), {})
                cache.put(
                    key,
                    arrays={"attributions": ig["attributions"]},
                    metadata={
                        "explainer": "ig",
                        "anchor_id": int(anchor["anchor_id"]),
                        "dataset": dataset,
                        "run_id": run_id,
                        "checkpoint_period": int(checkpoint_period),
                        "anchor_period": int(anchor_period),
                        "label": int(anchor["label"]),
                        "target_class": int(ig["target_class"]),
                        "steps": int(ig_steps),
                        "config_sha": config_sha,
                        "seed": int(gnn_seed),
                        "k": int(k),
                        "fidelity": fidelity_k,
                        "runtime_s": runtime,
                    },
                    index_row={
                        "checkpoint_sha": checkpoint_sha,
                        "anchor_id": int(anchor["anchor_id"]),
                        "explainer": "ig",
                        "seed": int(gnn_seed),
                        "k": int(k),
                        "config_sha": config_sha,
                        "fidelity_sufficiency": float(fidelity_k.get("sufficiency_ratio", 0.0) or 0.0),
                        "fidelity_comprehensiveness": float(fidelity_k.get("comprehensiveness", 0.0) or 0.0),
                        "fidelity_over_random": float(fidelity_k.get("comprehensiveness_over_random", 0.0) or 0.0),
                        "runtime_s": float(runtime),
                    },
                )

    if "gnn_explainer" in explainers:
        start = time.monotonic()
        explanation = gnn_explainer(
            model, data, target_local, epochs=gnn_epochs, seed=gnn_seed, device=device,
        )
        fidelity = edge_mask_fidelity(
            model, data, target_local, explanation["edge_mask"],
            m=5, random_trials=random_trials, seed=fidelity_seed, device=device,
        )
        runtime = round(time.monotonic() - start, 4)
        records.append({
            "anchor_id": int(anchor["anchor_id"]),
            "anchor_period": int(anchor_period),
            "checkpoint_period": int(checkpoint_period),
            "label": int(anchor["label"]),
            "explainer": "gnn_explainer",
            "runtime_s": runtime,
            "fidelity": fidelity,
            "top_edges": explanation["top_edges"],
        })
        if cache is not None:
            key = cache_key(
                checkpoint_sha=checkpoint_sha,
                anchor_id=anchor["anchor_id"],
                explainer="gnn_explainer",
                seed=int(gnn_seed),
                k=0,
                config_sha=config_sha,
            )
            cache.put(
                key,
                arrays={"edge_mask": explanation["edge_mask"]},
                metadata={
                    "explainer": "gnn_explainer",
                    "anchor_id": int(anchor["anchor_id"]),
                    "dataset": dataset,
                    "run_id": run_id,
                    "checkpoint_period": int(checkpoint_period),
                    "anchor_period": int(anchor_period),
                    "label": int(anchor["label"]),
                    "config_sha": config_sha,
                    "seed": int(gnn_seed),
                    "fidelity": fidelity,
                    "runtime_s": runtime,
                },
                index_row={
                    "checkpoint_sha": checkpoint_sha,
                    "anchor_id": int(anchor["anchor_id"]),
                    "explainer": "gnn_explainer",
                    "seed": int(gnn_seed),
                    "k": 0,
                    "config_sha": config_sha,
                    "fidelity_sufficiency": float(fidelity.get("sufficiency_ratio", 0.0) or 0.0),
                    "fidelity_comprehensiveness": float(fidelity.get("comprehensiveness", 0.0) or 0.0),
                    "fidelity_over_random": float(fidelity.get("comprehensiveness_over_random", 0.0) or 0.0),
                    "runtime_s": float(runtime),
                },
            )
    return records


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    torch = _torch()
    root = find_root()
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve() if args.checkpoint_root else root / "datasets" / "processed"
    graph_root = Path(args.graph_root).expanduser().resolve() if args.graph_root else root / "datasets" / "graphs"
    periods = [int(value) for value in args.periods.split(",") if value.strip()]
    explainers = [value.strip() for value in args.explainers.split(",") if value.strip()]
    k_values = [int(value) for value in args.k.split(",") if value.strip()]

    if args.anchor_tag is not None:
        anchor_tag = str(args.anchor_tag).strip()
    elif (args.hops, args.fanout, args.edge_cap) == (2, 10, 2000):
        anchor_tag = ""
    else:
        anchor_tag = f"h{args.hops}-f{args.fanout}-e{args.edge_cap}"
    plan = {
        "dataset": args.dataset,
        "run_id": args.run_id,
        "anchor_tag": anchor_tag,
        "graph_mode": args.graph_mode,
        "periods": periods,
        "explainers": explainers,
        "k_values": k_values,
        "context": {"hops": args.hops, "fanout": args.fanout, "edge_cap": args.edge_cap},
        "device": device,
        "max_anchors": args.max_anchors,
        "max_explanations": args.max_explanations,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    import hashlib

    from src.explain.cache import ExplanationCache
    from src.utils.config import load_config

    paths = load_paths_compat(root)
    config = load_config("configs/explain.yaml", project_root=root)
    context = {"hops": args.hops, "fanout": args.fanout, "edge_cap": args.edge_cap}
    context_digest = hashlib.sha256(json.dumps(context, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    config_sha = f"{config.sha256}:{context_digest}"
    fidelity_cfg = dict((config.to_dict().get("fidelity") or {}))
    cache = None
    if not args.no_cache:
        cache = ExplanationCache(
            paths.explanations_root,
            index_csv=paths.tables_root / "explain_index_main.csv",
        )

    explanation_count = 0
    all_records: list[dict[str, Any]] = []
    started = time.monotonic()
    for period in periods:
        try:
            anchor_set, anchor_file = select_period_anchors(
                root=root, checkpoint_root=checkpoint_root, graph_root=graph_root,
                dataset=args.dataset, run_id=args.run_id,
                graph_mode=args.graph_mode, period=period, hops=args.hops,
                fanout=args.fanout, edge_cap=args.edge_cap, device=device,
                max_anchors=args.max_anchors, anchor_tag=anchor_tag,
            )
        except FileNotFoundError as exc:
            print(f"[SKIP] period {period}: {exc}", file=sys.stderr)
            continue
        anchors = list(anchor_set.get("anchors", []))
        print(f"[OK] period {period}: {len(anchors)} anchors -> {anchor_file}")
        for checkpoint_period in range(period, max(periods) + 1):
            checkpoint = _checkpoint_path(checkpoint_root, args.dataset, args.run_id, checkpoint_period)
            if not checkpoint.is_file():
                print(f"[SKIP] checkpoint period {checkpoint_period}: missing {checkpoint}", file=sys.stderr)
                continue
            model, _payload = load_model(checkpoint, device)
            checkpoint_sha = _sha256(checkpoint)
            for anchor in anchors:
                if args.max_explanations and explanation_count >= args.max_explanations:
                    break
                records = explain_anchor(
                    model=model, anchor=anchor, node_dim=int(anchor_set.get("node_features", 8)),
                    device=device, explainers=explainers, k_values=k_values,
                    ig_steps=args.ig_steps, gnn_epochs=args.gnn_epochs, gnn_seed=42,
                    config_sha=config_sha, checkpoint_sha=checkpoint_sha,
                    checkpoint_period=checkpoint_period, anchor_period=period,
                    dataset=args.dataset, run_id=args.run_id, cache=cache,
                    random_trials=int(fidelity_cfg.get("randomization_trials", 5)),
                    fidelity_seed=int(fidelity_cfg.get("seed", 42)),
                )
                all_records.extend(records)
                explanation_count += len(records)
            if args.max_explanations and explanation_count >= args.max_explanations:
                break
        if args.max_explanations and explanation_count >= args.max_explanations:
            break

    summary_dir = root / "results" / "text" / "data_audit" / "explanations" / args.dataset / args.run_id
    if anchor_tag:
        summary_dir = summary_dir / anchor_tag
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "plan": plan,
        "elapsed_s": round(time.monotonic() - started, 2),
        "n_explanations": explanation_count,
        "records": all_records,
        "cache_root": str(paths.explanations_root) if cache is not None else None,
        "canonical_index": str(paths.tables_root / "explain_index_main.csv") if cache is not None else None,
    }
    (summary_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({"explanations": explanation_count, "summary": str(summary_dir / "summary.json")},
                     indent=2, sort_keys=True))
    return 0


def load_paths_compat(root: Path):
    """Load path config from the repository root (small wrapper for readability)."""
    from src.utils.paths import load_paths

    return load_paths(project_root=root)


if __name__ == "__main__":
    raise SystemExit(main())
