#!/usr/bin/env python
"""S8 noise-cache producer (D-048): explainer-repeat + model-null + combined null.

    python src/scripts/build_noise_floor.py --dataset NF-UNSW-NB15-v3 --strategy ft \
        --model egraphsage --seed 42 --period 1 --dry-run

Outputs (operational, gitignored):
    results/text/explanations_noise/explainer_repeat/index.csv   (R=10 repeats per anchor)
    results/text/explanations_noise/model_null/index.csv         (R=5 same-period FT reseeds)
    results/text/explanations_noise/combined_null.csv            (combined-null rows)
    results/text/data_audit/noise_floor_manifest.json            (pinned manifest)

The per-entry S7 cache schema (D-048/docs/05 ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â§6) is enforced: every entry carries
sufficiency/comprehensiveness fidelity plus runtime. The first real S8/S9 run only
needs the frozen S1 graphs + S6 anchors + S3 checkpoints; no data/split changes.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.explain.cache import INDEX_COLUMNS, ExplanationCache, cache_key  # noqa: E402
from src.explain.fidelity import edge_mask_fidelity, feature_fidelity  # noqa: E402
from src.explain.gnn_explainer import gnn_explainer  # noqa: E402
from src.explain.ig import integrated_gradients  # noqa: E402
from src.models.edge_sage import build_model  # noqa: E402
from src.utils.config import hash_config, load_config  # noqa: E402
from src.utils.gates import gate_record  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402
from src.utils.manifest import RunManifest, sha256_file, utc_now, write_json  # noqa: E402
from src.utils.paths import load_paths  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402
from src.utils.torch_support import torch_available, torch_unavailable_reason  # noqa: E402


def anchor_to_data(anchor: Mapping[str, Any], *, node_feature_dim: int = 8):
    import torch
    from torch_geometric.data import Data

    return Data(
        x=torch.ones((int(anchor["n_sub_nodes"]), int(node_feature_dim)), dtype=torch.float32),
        edge_index=anchor["sub_edge_index"].long(),
        edge_attr=anchor["sub_edge_attr"].float(),
        y=torch.as_tensor([int(anchor["label"])], dtype=torch.long),
    )


def _store_entry(
    cache: ExplanationCache,
    *,
    anchor: Mapping[str, Any],
    explainer: str,
    arrays: Mapping[str, Any],
    metadata: Mapping[str, Any],
    index_row: Mapping[str, Any],
) -> str:
    key = cache_key(
        checkpoint_sha=index_row["checkpoint_sha"],
        anchor_id=anchor["anchor_id"],
        explainer=explainer,
        seed=index_row["seed"],
        k=index_row["k"],
        config_sha=index_row["config_sha"],
    )
    cache.put(key, arrays=dict(arrays), metadata=dict(metadata), index_row=dict(index_row))
    return key


def produce_explainer_repeat(
    *,
    model: Any,
    node_feature_dim: int = 8,
    anchors: Sequence[Mapping[str, Any]],
    out_dir: str | Path,
    explainers: Sequence[str] = ("ig", "gnn_explainer"),
    repeats: int = 10,
    seed: int = 42,
    checkpoint_sha: str = "unknown",
    checkpoint_period: int = 1,
    config_sha: str = "unknown",
    strategy: str = "ft",
    model_name: str = "egraphsage",
    device: str = "cpu",
    random_trials: int = 2,
) -> dict[str, Any]:
    """Emit explainer-repeat caches (R=10) for every anchor ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã¢â‚¬Â explainer."""
    import numpy as np
    import torch

    out_path = Path(out_dir) / "explainer_repeat"
    cache = ExplanationCache(out_path)
    n_entries = 0
    canary: list[float] = []
    for anchor in anchors:
        data = anchor_to_data(anchor, node_feature_dim=node_feature_dim).to(device)
        target_local = int(anchor["target_local_index"][0])
        for explainer in explainers:
            first_array = None
            for repeat_index in range(int(repeats)):
                repeat_seed = int(seed) * 1000 + repeat_index
                started = time.monotonic()
                if explainer == "ig":
                    record = integrated_gradients(model, data, target_local, steps=16, device=device)
                    array = record["attributions"]
                    fidelity = feature_fidelity(
                        model, data, target_local, array, k_values=(5,), random_trials=random_trials,
                        seed=repeat_seed, device=device,
                    )[0]
                    arrays = {"attributions": array}
                else:
                    record = gnn_explainer(model, data, target_local, epochs=8, seed=repeat_seed, device=device)
                    array = record["edge_mask"]
                    fidelity = edge_mask_fidelity(
                        model, data, target_local, array, m=5, random_trials=random_trials,
                        seed=repeat_seed, device=device,
                    )
                    arrays = {"edge_mask": array}
                runtime = round(time.monotonic() - started, 4)
                if first_array is None:
                    first_array = np.asarray(array)
                _store_entry(
                    cache,
                    anchor=anchor,
                    explainer=explainer,
                    arrays=arrays,
                    metadata={
                        "explainer": explainer,
                        "anchor_id": int(anchor["anchor_id"]),
                        "seed": repeat_seed,
                        "repeat_index": int(repeat_index),
                        "checkpoint_period": int(checkpoint_period),
                        "anchor_period": int(anchor["period"]),
                        "class": str(anchor.get("label")),
                        "label": int(anchor["label"]),
                        "strategy": strategy,
                        "model": model_name,
                        "noise_channel": "explainer_repeat",
                        "config_sha": config_sha,
                    },
                    index_row={
                        "checkpoint_sha": checkpoint_sha,
                        "anchor_id": anchor["anchor_id"],
                        "explainer": explainer,
                        "seed": repeat_seed,
                        "k": 5 if explainer == "ig" else 0,
                        "config_sha": config_sha,
                        "fidelity_sufficiency": fidelity.get("sufficiency_ratio"),
                        "fidelity_comprehensiveness": fidelity.get("comprehensiveness"),
                        "fidelity_over_random": fidelity.get("comprehensiveness_over_random"),
                        "runtime_s": runtime,
                    },
                )
                n_entries += 1
            if first_array is not None and explainer == "ig":
                # IG-repetition drift canary must be exactly 0 (deterministic IG).
                from analysis.metrics_drift import attribution_drift

                last = integrated_gradients(model, data, target_local, steps=16, device=device)["attributions"]
                canary.append(float(attribution_drift(first_array, last)["cosine"]))
    return {
        "channel": "explainer_repeat",
        "entries": n_entries,
        "anchors": len(anchors),
        "repeats": int(repeats),
        "ig_repeat_canary_drift_max": float(max(canary)) if canary else None,
        "index_csv": str(cache.index_path),
    }


def canonical_noise_train_settings(train_config_dict: Mapping[str, Any], *, null_epochs: int | None = None) -> Any:
    """T40/F2: canonical FT training recipe for model-null reseeds.

    The null model must mirror the frozen FT recipe (D-059 ``graph_batch`` with
    early stopping on validation macro-F1); ``--null-epochs`` may only shorten
    the maximum budget, never change the batching/patience semantics.
    """
    from src.train.trainer import TrainSettings

    train_section = dict(train_config_dict.get("train") or {})
    if not train_section:
        raise ValueError("model-null reseeds require a config with a 'train' section (e.g. configs/train.yaml)")
    settings = TrainSettings.from_config(train_config_dict)
    if str(settings.batching) != "graph_batch":
        raise ValueError(
            f"model-null reseeds require canonical batching='graph_batch', got {settings.batching!r} "
            "from the supplied train config"
        )
    if null_epochs is not None and int(null_epochs) > 0:
        settings = replace(settings, max_epochs=min(int(null_epochs), int(settings.max_epochs)))
    return settings


def produce_model_null(
    *,
    graph: Any,
    node_feature_dim: int = 8,
    anchors: Sequence[Mapping[str, Any]],
    payload: Mapping[str, Any],
    out_dir: str | Path,
    seed: int = 42,
    null_reseeds: int = 5,
    null_epochs: int = 10,
    explainers: Sequence[str] = ("ig", "gnn_explainer"),
    checkpoint_period: int = 1,
    config_sha: str = "unknown",
    strategy: str = "ft",
    model_name: str = "egraphsage",
    device: str = "cpu",
    random_trials: int = 2,
    train_settings: Any | None = None,
) -> dict[str, Any]:
    """Retrain FT R=5 times on the same period and cache the null-model explanations."""
    from src.train.trainer import TrainSettings, train_edge_classifier

    out_path = Path(out_dir) / "model_null"
    cache = ExplanationCache(out_path)
    model_config = dict(payload.get("model_config") or {})
    settings = train_settings if train_settings is not None else TrainSettings(batching="graph_batch")
    # T40/F2: model-null reseeds must mirror the canonical FT recipe; only the
    # explicit --null-epochs cap may shorten the budget.
    if str(settings.batching) != "graph_batch":
        raise ValueError(
            f"model-null reseeds require canonical batching='graph_batch', got {settings.batching!r} "
            "(pass train settings from configs/train.yaml, see canonical_noise_train_settings)"
        )
    if int(null_epochs) > 0:
        settings = replace(settings, max_epochs=min(int(null_epochs), int(settings.max_epochs)))
    n_entries = 0
    null_checkpoints: list[dict[str, Any]] = []
    for reseed_index in range(int(null_reseeds)):
        null_seed = int(seed) + 100 + reseed_index
        seed_everything(null_seed)
        model = build_model(
            model_name,
            node_dim=int(graph.x.shape[1]),
            edge_dim=int(graph.edge_attr.shape[1]),
            config={"model": model_config},
        )
        train_result = train_edge_classifier(model, graph, settings=settings, seed=null_seed, device=device)
        null_checkpoints.append(
            {
                "null_seed": null_seed,
                "val_macro_f1": train_result["val"]["macro_f1"],
                "epochs_run": train_result["epochs_run"],
            }
        )
        checkpoint_sha = f"null_model_{null_seed}"
        for anchor in anchors:
            data = anchor_to_data(anchor, node_feature_dim=node_feature_dim).to(device)
            target_local = int(anchor["target_local_index"][0])
            for explainer in explainers:
                started = time.monotonic()
                if explainer == "ig":
                    record = integrated_gradients(model, data, target_local, steps=16, device=device)
                    array = record["attributions"]
                    fidelity = feature_fidelity(
                        model, data, target_local, array, k_values=(5,), random_trials=random_trials,
                        seed=null_seed, device=device,
                    )[0]
                    arrays = {"attributions": array}
                    k = 5
                else:
                    record = gnn_explainer(model, data, target_local, epochs=8, seed=null_seed, device=device)
                    array = record["edge_mask"]
                    fidelity = edge_mask_fidelity(
                        model, data, target_local, array, m=5, random_trials=random_trials,
                        seed=null_seed, device=device,
                    )
                    arrays = {"edge_mask": array}
                    k = 0
                runtime = round(time.monotonic() - started, 4)
                _store_entry(
                    cache,
                    anchor=anchor,
                    explainer=explainer,
                    arrays=arrays,
                    metadata={
                        "explainer": explainer,
                        "anchor_id": int(anchor["anchor_id"]),
                        "seed": null_seed,
                        "null_seed": null_seed,
                        "checkpoint_period": int(checkpoint_period),
                        "anchor_period": int(anchor["period"]),
                        "class": str(anchor.get("label")),
                        "label": int(anchor["label"]),
                        "strategy": strategy,
                        "model": model_name,
                        "noise_channel": "model_null",
                        "config_sha": config_sha,
                    },
                    index_row={
                        "checkpoint_sha": checkpoint_sha,
                        "anchor_id": anchor["anchor_id"],
                        "explainer": explainer,
                        "seed": null_seed,
                        "k": k,
                        "config_sha": config_sha,
                        "fidelity_sufficiency": fidelity.get("sufficiency_ratio"),
                        "fidelity_comprehensiveness": fidelity.get("comprehensiveness"),
                        "fidelity_over_random": fidelity.get("comprehensiveness_over_random"),
                        "runtime_s": runtime,
                    },
                )
                n_entries += 1
    return {
        "channel": "model_null",
        "entries": n_entries,
        "anchors": len(anchors),
        "null_reseeds": int(null_reseeds),
        "null_checkpoints": null_checkpoints,
        "index_csv": str(cache.index_path),
    }


def combine_null_channels(out_dir: str | Path) -> dict[str, Any]:
    """Combine the two null channels with the authoritative analysis function."""
    import pandas as pd

    root = Path(out_dir)
    repeat_path = root / "explainer_repeat" / "index.csv"
    null_path = root / "model_null" / "index.csv"
    summary: dict[str, Any] = {"index_csv": str(root / "combined_null.csv")}
    if not repeat_path.is_file() or not null_path.is_file():
        summary["status"] = "skipped_missing_channel"
        return summary
    repeat_rows = [{**row, "channel": "explainer_repeat"} for row in csv.DictReader(repeat_path.open("r", encoding="utf-8"))]
    null_rows = [{**row, "channel": "model_null"} for row in csv.DictReader(null_path.open("r", encoding="utf-8"))]
    frame = pd.DataFrame(repeat_rows + null_rows)
    summary["status"] = "ok"
    summary["rows"] = len(frame)
    summary["channel_counts"] = frame["channel"].value_counts().to_dict() if "channel" in frame else {}
    # The per-row cache indexes are combined here for the S8 consumer; the
    # authoritative channel-combination statistics are computed downstream by
    # analysis.noise_floor.combined_null_drift in tables.generate_all.
    summary["combined_statistics_owner"] = "analysis.noise_floor.combined_null_drift (D-044)"
    frame.to_csv(root / "combined_null.csv", index=False)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build_noise_floor", description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--strategy", default="ft")
    parser.add_argument("--model", default="egraphsage")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--period", type=int, default=1)
    parser.add_argument("--config", default="configs/drift.yaml")
    parser.add_argument("--train-config", default="configs/train.yaml")
    parser.add_argument("--anchors-dir", default=None)
    parser.add_argument("--graphs-dir", default=None)
    parser.add_argument("--checkpoints-root", default=None)
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--explainers", default="ig,gnn_explainer")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--null-reseeds", type=int, default=5)
    parser.add_argument("--null-epochs", type=int, default=None)
    parser.add_argument("--max-anchors", type=int, default=100)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--lane", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    paths = load_paths(lane=args.lane)
    config = load_config(args.config, project_root=paths.project_root, lane=args.lane)
    config_dict = config.to_dict()
    train_config = load_config(args.train_config, project_root=paths.project_root, lane=args.lane)
    train_config_dict = train_config.to_dict()
    train_section = dict(train_config_dict.get("train") or {})
    if not train_section:
        print(
            f"[noise] ERROR: --train-config {args.train_config} has no 'train' section; "
            "model-null reseeds must mirror the canonical training recipe",
            file=sys.stderr,
        )
        return 2
    train_config_sha = hash_config(train_config_dict)
    logger = setup_logging(
        "xdrift.build_noise_floor",
        level=str((config.get("logging") or {}).get("level", "INFO")),
        log_path=paths.logs_root / "build_noise_floor.log",
    )
    dataset = args.dataset or str(config.get("dataset", "NF-UNSW-NB15-v3"))
    explainers = [name.strip() for name in args.explainers.split(",") if name.strip()]
    anchors_dir = Path(args.anchors_dir).expanduser().resolve() if args.anchors_dir else paths.results_root / "anchors"
    graphs_dir = Path(args.graphs_dir).expanduser().resolve() if args.graphs_dir else paths.processed_dataset(dataset)
    checkpoints_root = (
        Path(args.checkpoints_root).expanduser().resolve()
        if args.checkpoints_root
        else paths.run_dir(strategy=args.strategy, model=args.model, seed=args.seed)
    )
    out_root = Path(args.out_root).expanduser().resolve() if args.out_root else paths.results_root / "explanations_noise"
    anchor_file = anchors_dir / f"period_{args.period}.pt"
    graph_file = graphs_dir / f"period_{args.period}.pt"
    checkpoint_file = checkpoints_root / f"M_{args.period}.pt"
    plan = {
        "dataset": dataset,
        "period": args.period,
        "anchors": str(anchor_file),
        "anchors_present": anchor_file.is_file(),
        "graph": str(graph_file),
        "graph_present": graph_file.is_file(),
        "checkpoint": str(checkpoint_file),
        "checkpoint_present": checkpoint_file.is_file(),
        "out_root": str(out_root),
        "explainers": explainers,
        "repeats": args.repeats,
        "null_reseeds": args.null_reseeds,
        "max_anchors": args.max_anchors,
        "lane": paths.lane,
        "lane_source": paths.lane_source,
        "gate": gate_record("noise_floor", paths),
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    if not torch_available():
        print(f"[noise] ERROR: torch unavailable: {torch_unavailable_reason()}", file=sys.stderr)
        return 2
    missing = [name for name in ("anchors", "graph", "checkpoint") if not plan[f"{name}_present"]]
    if missing:
        print(f"[noise] ERROR: missing inputs: {missing} (wait for G2 artifacts)", file=sys.stderr)
        return 2
    import torch

    if paths.lane_source == "default":
        logger.warning("lane is 'default' (no --lane/XDRIFT_LANE); gate runs should launch via run_lane.ps1")
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    manifest = RunManifest(
        "noise_floor",
        paths=paths,
        config=config,
        lane=paths.lane,
        seed=args.seed,
        dataset=dataset,
        out_path=paths.audit_root / f"noise_floor_manifest_{args.strategy}_{args.model}_seed{args.seed}_period{args.period}.json",
        extra={"plan": {key: value for key, value in plan.items() if key != "gate"}, "gate_preflight": plan["gate"]},
    )
    manifest.start()
    try:
        graph = torch.load(graph_file, weights_only=False)
        payload = torch.load(checkpoint_file, weights_only=False)
        anchor_set = torch.load(anchor_file, weights_only=False)
        anchors = list(anchor_set.get("anchors", []))[: int(args.max_anchors)]
        if not anchors:
            raise RuntimeError("anchor file contains no anchors")
        model_config = dict(payload.get("model_config") or {})
        model = build_model(
            args.model,
            node_dim=int(graph.x.shape[1]),
            edge_dim=int(graph.edge_attr.shape[1]),
            config={"model": model_config},
        )
        model.load_state_dict(payload["state_dict"])
        model.to(device).eval()
        out_root.mkdir(parents=True, exist_ok=True)
        checkpoint_sha = sha256_file(checkpoint_file)
        config_sha = hash_config(config_dict)
        start = time.monotonic()
        repeat_summary = produce_explainer_repeat(
            model=model,
            anchors=anchors,
            out_dir=out_root,
            explainers=explainers,
            repeats=int(args.repeats),
            seed=args.seed,
            checkpoint_sha=checkpoint_sha,
            checkpoint_period=args.period,
            config_sha=config_sha,
            strategy=args.strategy,
            model_name=args.model,
            device=device,
            node_feature_dim=int(graph.x.shape[1]),
        )
        null_epochs = args.null_epochs
        if null_epochs is None:
            null_epochs = int(train_section.get("max_epochs") or 30)
        null_settings = canonical_noise_train_settings(train_config_dict, null_epochs=int(null_epochs))
        null_summary = produce_model_null(
            graph=graph,
            node_feature_dim=int(graph.x.shape[1]),
            anchors=anchors,
            payload=payload,
            out_dir=out_root,
            seed=args.seed,
            null_reseeds=int(args.null_reseeds),
            null_epochs=int(null_epochs),
            train_settings=null_settings,
            explainers=explainers,
            checkpoint_period=args.period,
            config_sha=config_sha,
            strategy=args.strategy,
            model_name=args.model,
            device=device,
        )
        combined = combine_null_channels(out_root)
        summary_path = paths.audit_root / f"noise_floor_summary_{args.strategy}_{args.model}_seed{args.seed}_period{args.period}.json"
        write_json(
            summary_path,
            {
                "schema_version": 1,
                "generated_at_utc": utc_now(),
                "dataset": dataset,
                "period": args.period,
                "strategy": args.strategy,
                "model": args.model,
                "seed": args.seed,
                "elapsed_s": round(time.monotonic() - start, 3),
                "explainer_repeat": repeat_summary,
                "model_null": null_summary,
                "combined": combined,
                "train_config": {
                    "path": str(args.train_config),
                    "sha256": train_config_sha,
                    "batching": str(null_settings.batching),
                    "max_epochs": int(null_settings.max_epochs),
                    "patience": int(null_settings.patience),
                    "null_epochs": int(null_epochs),
                },
            },
        )
        manifest.add_artifact(summary_path, role="noise_floor_summary")
        manifest.add_artifact(Path(repeat_summary["index_csv"]), role="explainer_repeat_index")
        manifest.add_artifact(Path(null_summary["index_csv"]), role="model_null_index")
        if Path(combined["index_csv"]).is_file():
            manifest.add_artifact(Path(combined["index_csv"]), role="combined_null")
        manifest.finish("success")
        manifest.write()
        print(json.dumps({"explainer_repeat": repeat_summary, "model_null": null_summary, "combined": combined}, indent=2, sort_keys=True, default=str))
        print(f"[noise] manifest: {manifest.resolve_out_path()}")
        return 0
    except Exception as exc:  # noqa: BLE001
        manifest.finish("failed", error=f"{type(exc).__name__}: {exc}")
        manifest.write()
        raise


if __name__ == "__main__":
    raise SystemExit(main())


