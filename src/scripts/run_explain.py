#!/usr/bin/env python
"""S7 explanation CLI: IG + GNNExplainer on anchors, with fidelity and caching.

    python src/scripts/run_explain.py --dataset NF-UNSW-NB15-v3 --strategy ft --model egraphsage \
        --seed 42 --period 1 --max-anchors 5 --dry-run

At this stage the CLI is intentionally limited (``--max-anchors``) and no
full-scale explanation generation is run: it proves the S7 adapters, fidelity
checks and cache contract on a handful of anchors. Cache lives under
``XDRIFT_RESULTS/explanations`` (gitignored); each entry is keyed by
(checkpoint_sha, anchor_id, explainer, seed, k, config_sha) and indexed in a CSV.
"""

from __future__ import annotations

import argparse
import csv
import time
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.explain.cache import INDEX_COLUMNS, ExplanationCache, cache_from_config, cache_key  # noqa: E402
from src.explain.fidelity import edge_mask_fidelity, feature_fidelity  # noqa: E402
from src.explain.gnn_explainer import gnn_explainer  # noqa: E402
from src.explain.ig import integrated_gradients, top_k_features  # noqa: E402
from src.models.edge_sage import build_model  # noqa: E402
from src.utils.config import hash_config, load_config  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402
from src.utils.manifest import RunManifest, sha256_file, utc_now, write_json  # noqa: E402
from src.utils.paths import load_paths  # noqa: E402
from src.utils.torch_support import torch_available, torch_unavailable_reason  # noqa: E402


import math


def _unit(value):
    """D-048(c) provenance column: finite value clamped to [0,1] (raw kept in metadata)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(number):
        return 0.0
    return min(1.0, max(0.0, number))


def _numeric(value):
    """D-048(c) provenance column: finite numeric (NaN/inf -> 0.0; raw kept in metadata)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) else 0.0


def checkpoint_period_from_payload(payload: Mapping[str, Any], fallback: int) -> int:
    """True checkpoint period recorded in the checkpoint payload (D-072/T41).

    ``--period`` can be an anchor period in the corrected cross-checkpoint S7 plan; the
    checkpoint itself carries its training period (``M_t`` payload ``period``), which is
    what the cache/provenance must record. Fall back to ``--period`` when an older
    checkpoint does not carry it.
    """
    value = payload.get("period")
    try:
        return int(value) if value is not None else int(fallback)
    except (TypeError, ValueError):
        return int(fallback)

def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name) or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"config section {name!r} must be a mapping")
    return dict(value)


def anchor_to_data(anchor: Mapping[str, Any], *, node_feature_dim: int = 8):
    """Rebuild the anchor's stored 2-hop subgraph as a PyG ``Data`` object."""
    import torch
    from torch_geometric.data import Data

    n_nodes = int(anchor["n_sub_nodes"])
    data = Data(
        x=torch.ones((n_nodes, int(node_feature_dim)), dtype=torch.float32),
        edge_index=anchor["sub_edge_index"].long(),
        edge_attr=anchor["sub_edge_attr"].float(),
        y=torch.as_tensor([int(anchor["label"])], dtype=torch.long),
    )
    return data


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_explain", description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--strategy", default="ft")
    parser.add_argument("--model", default="egraphsage")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--period", type=int, default=1)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--anchors", default=None)
    parser.add_argument("--config", default="configs/explain.yaml")
    parser.add_argument("--explainers", default="ig,gnn_explainer")
    parser.add_argument("--k", default="5,10")
    parser.add_argument("--max-anchors", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lane", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    paths = load_paths(lane=args.lane)
    config = load_config(args.config, project_root=paths.project_root, lane=args.lane)
    config_dict = config.to_dict()
    logger = setup_logging(
        "xdrift.run_explain",
        level=str((config.get("logging") or {}).get("level", "INFO")),
        log_path=paths.logs_root / "run_explain.log",
    )
    dataset = args.dataset or str(config.get("dataset", "NF-UNSW-NB15-v3"))
    checkpoint_path = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else paths.run_dir(strategy=args.strategy, model=args.model, seed=args.seed) / f"M_{args.period}.pt"
    )
    anchors_path = (
        Path(args.anchors).expanduser().resolve()
        if args.anchors
        else paths.results_root / "anchors" / f"period_{args.period}.pt"
    )
    explainers = [name.strip() for name in args.explainers.split(",") if name.strip()]
    k_values = [int(value) for value in args.k.split(",") if value.strip()]
    cache = cache_from_config(config_dict, results_root=paths.results_root)
    plan = {
        "dataset": dataset,
        "strategy": args.strategy,
        "model": args.model,
        "seed": args.seed,
        "period": args.period,
        "checkpoint_period": int(args.period),
        "checkpoint": str(checkpoint_path),
        "checkpoint_present": checkpoint_path.is_file(),
        "anchors": str(anchors_path),
        "anchors_present": anchors_path.is_file(),
        "explainers": explainers,
        "k_values": k_values,
        "max_anchors": args.max_anchors,
        "cache_root": str(cache.root),
        "lane": paths.lane,
        "lane_source": paths.lane_source,
        "config_sha256": config.sha256,
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if not torch_available():
        print(f"[explain] ERROR: torch unavailable: {torch_unavailable_reason()}", file=sys.stderr)
        return 2
    import torch

    if not checkpoint_path.is_file():
        print(f"[explain] ERROR: checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 2
    if not anchors_path.is_file():
        print(f"[explain] ERROR: anchor file not found: {anchors_path}", file=sys.stderr)
        return 2
    if paths.lane_source == "default":
        logger.warning("lane is 'default' (no --lane/XDRIFT_LANE); gate runs should launch via run_lane.ps1")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    payload = torch.load(checkpoint_path, weights_only=False)
    anchor_set = torch.load(anchors_path, weights_only=False)
    checkpoint_period = checkpoint_period_from_payload(payload, int(args.period))
    anchor_period = int(anchor_set.get("period", int(args.period)))
    plan["checkpoint_period"] = checkpoint_period
    plan["anchor_period"] = anchor_period
    anchors = list(anchor_set.get("anchors", []))[: int(args.max_anchors)]
    if not anchors:
        print("[explain] ERROR: anchor file contains no anchors", file=sys.stderr)
        return 2
    manifest = RunManifest(
        "explanation_S7",
        paths=paths,
        config=config,
        lane=paths.lane,
        seed=args.seed,
        dataset=dataset,
        out_path=paths.audit_root
        / f"explain_manifest_{args.strategy}_{args.model}_seed{args.seed}_period{checkpoint_period}.json",
        extra={
            "checkpoint": str(checkpoint_path),
            "checkpoint_sha": sha256_file(checkpoint_path),
            "checkpoint_period": checkpoint_period,
            "anchor_period": anchor_period,
            "anchors_file": str(anchors_path),
            "anchors_sha": sha256_file(anchors_path),
            "explainers": explainers,
            "k_values": k_values,
            "max_anchors": int(args.max_anchors),
            "device": device,
            "lane_source": paths.lane_source,
        },
    )
    manifest.start()
    try:
        model = None
        records: list[dict[str, Any]] = []
        for anchor in anchors:
            data = anchor_to_data(anchor, node_feature_dim=8).to(device)
            if model is None:
                model_config = dict(payload.get("model_config") or {})
                model = build_model(
                    args.model,
                    node_dim=int(data.x.shape[1]),
                    edge_dim=int(data.edge_attr.shape[1]),
                    config={"model": model_config},
                )
                model.load_state_dict(payload["state_dict"])
                model.to(device).eval()
            target_local = int(anchor["target_local_index"][0])
            ig_steps = int(_section(_section(config_dict, "explainers"), "ig").get("steps", 32))
            record: dict[str, Any] = {
                "anchor_id": int(anchor["anchor_id"]),
                "period": int(anchor["period"]),
                "label": int(anchor["label"]),
                "explainers": {},
            }
            if "ig" in explainers:
                ig_start = time.monotonic()
                ig = integrated_gradients(
                    model, data, target_local, steps=ig_steps, device=device
                )
                fidelity = feature_fidelity(
                    model,
                    data,
                    target_local,
                    ig["attributions"],
                    k_values=k_values,
                    random_trials=int(_section(config_dict, "fidelity").get("randomization_trials", 5)),
                    seed=args.seed,
                    device=device,
                )
                ig_runtime = round(time.monotonic() - ig_start, 4)
                fidelity_by_k = {int(item["k"]): item for item in fidelity}
                record["explainers"]["ig"] = {
                    "target_class": ig["target_class"],
                    "attribution_sum": ig["attribution_sum"],
                    "completeness_delta": ig["completeness_delta"],
                    "top_features": {str(k): top_k_features(ig["attributions"], k) for k in k_values},
                    "fidelity": fidelity,
                    "runtime_s": ig_runtime,
                }
                for k in k_values:
                    key = cache_key(
                        checkpoint_sha=sha256_file(checkpoint_path),
                        anchor_id=anchor["anchor_id"],
                        explainer="ig",
                        seed=args.seed,
                        k=k,
                        config_sha=config.sha256,
                    )
                    fidelity_k = fidelity_by_k.get(int(k), {})
                    cache.put(
                        key,
                        arrays={"attributions": ig["attributions"]},
                        metadata={
                            "explainer": "ig",
                            "anchor_id": int(anchor["anchor_id"]),
                            "target_class": ig["target_class"],
                            "steps": ig["steps"],
                            "config_sha": config.sha256,
                            "seed": args.seed,
                            "k": k,
                            # S8 provenance: the drift wiring groups explanations by
                            # checkpoint and anchor period, and needs the class label.
                            "checkpoint_period": checkpoint_period,
                            "anchor_period": int(anchor["period"]),
                            "class": str(anchor.get("label")),
                            "label": int(anchor["label"]),
                            "strategy": args.strategy,
                            "model": args.model,
                            "dataset": dataset,
                            # D-048(c) provenance mirrored in per-entry metadata too.
                            "fidelity": fidelity_k,
                            "runtime_s": ig_runtime,
                        },
                        index_row={
                            "checkpoint_sha": sha256_file(checkpoint_path),
                            "anchor_id": anchor["anchor_id"],
                            "explainer": "ig",
                            "seed": args.seed,
                            "k": k,
                            "config_sha": config.sha256,
                            "fidelity_sufficiency": _unit(fidelity_k.get("sufficiency_ratio")),
                            "fidelity_comprehensiveness": _unit(fidelity_k.get("comprehensiveness")),
                            "fidelity_over_random": _numeric(fidelity_k.get("comprehensiveness_over_random")),
                            "runtime_s": ig_runtime,
                        },
                    )
            if "gnn_explainer" in explainers:
                gnn_cfg = _section(_section(config_dict, "explainers"), "gnn_explainer")
                gnn_start = time.monotonic()
                explanation = gnn_explainer(
                    model,
                    data,
                    target_local,
                    epochs=int(gnn_cfg.get("epochs", 100)),
                    seed=int(gnn_cfg.get("seed", 42)),
                    device=device,
                )
                edge_fidelity = edge_mask_fidelity(
                    model,
                    data,
                    target_local,
                    explanation["edge_mask"],
                    m=int(_section(_section(config_dict, "explainers"), "edge").get("top_m", 5)),
                    random_trials=int(_section(config_dict, "fidelity").get("randomization_trials", 5)),
                    seed=args.seed,
                    device=device,
                )
                gnn_runtime = round(time.monotonic() - gnn_start, 4)
                record["explainers"]["gnn_explainer"] = {
                    "target_edge_mask_value": explanation["target_edge_mask_value"],
                    "top_edges": explanation["top_edges"],
                    "edge_mask_size": int(len(explanation["edge_mask"])),
                    "seed": explanation["seed"],
                    "fidelity": edge_fidelity,
                    "runtime_s": gnn_runtime,
                }
                key = cache_key(
                    checkpoint_sha=sha256_file(checkpoint_path),
                    anchor_id=anchor["anchor_id"],
                    explainer="gnn_explainer",
                    seed=int(gnn_cfg.get("seed", 42)),
                    k=0,
                    config_sha=config.sha256,
                )
                cache.put(
                    key,
                    arrays={"edge_mask": explanation["edge_mask"]},
                    metadata={
                        "explainer": "gnn_explainer",
                        "anchor_id": int(anchor["anchor_id"]),
                        "seed": explanation["seed"],
                        "checkpoint_period": checkpoint_period,
                        "anchor_period": int(anchor["period"]),
                        "class": str(anchor.get("label")),
                        "label": int(anchor["label"]),
                        "strategy": args.strategy,
                        "model": args.model,
                        "dataset": dataset,
                        "fidelity": edge_fidelity,
                        "runtime_s": gnn_runtime,
                    },
                    index_row={
                        "checkpoint_sha": sha256_file(checkpoint_path),
                        "anchor_id": anchor["anchor_id"],
                        "explainer": "gnn_explainer",
                        "seed": explanation["seed"],
                        "k": 0,
                        "config_sha": config.sha256,
                        "fidelity_sufficiency": _unit(edge_fidelity.get("sufficiency_ratio")),
                        "fidelity_comprehensiveness": _unit(edge_fidelity.get("comprehensiveness")),
                        "fidelity_over_random": _numeric(edge_fidelity.get("comprehensiveness_over_random")),
                        "runtime_s": gnn_runtime,
                    },
                )
            records.append(record)
        # D-048: canonical committed evidence index (same rows as the operational
        # gitignored cache index). results/text/tables/explain_index.csv is what the
        # analysis wiring/validators read; results/text/explanations/index.csv stays
        # as the operational lookup.
        canonical_index = paths.tables_root / "explain_index.csv"
        canonical_index.parent.mkdir(parents=True, exist_ok=True)
        with canonical_index.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
            writer.writeheader()
            writer.writerows(cache.index_rows())
        summary_path = (
            paths.audit_root
            / f"explanations_summary_{args.strategy}_{args.model}_seed{args.seed}_period{checkpoint_period}.json"
        )
        write_json(summary_path, {
            "schema_version": 1,
            "generated_at_utc": utc_now(),
            "dataset": dataset,
            "strategy": args.strategy,
            "model": args.model,
            "seed": args.seed,
            "period": checkpoint_period,
            "checkpoint_period": checkpoint_period,
            "anchor_period": anchor_period,
            "plan": plan,
            "records": records,
            "cache_index": str(cache.index_path),
            "canonical_explain_index": str(canonical_index),
        })
        manifest.add_artifact(summary_path, role="explanation_summary")
        manifest.add_artifact(cache.index_path, role="cache_index")
        manifest.add_artifact(canonical_index, role="explain_index")
        manifest.finish("success")
        manifest.write()
        print(json.dumps({"anchors_explained": len(records), "summary": str(summary_path),
                          "cache_index": str(cache.index_path), "cache_root": str(cache.root)},
                         indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        manifest.finish("failed", error=f"{type(exc).__name__}: {exc}")
        manifest.write()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
