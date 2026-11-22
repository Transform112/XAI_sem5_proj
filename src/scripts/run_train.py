#!/usr/bin/env python
"""S3 training CLI: train one model on one frozen period graph, with pins.

    python src/scripts/run_train.py --dataset NF-UNSW-NB15-v3 --config configs/train.yaml \
        --stage S3 --strategy ft --period 1 --seed 42 --smoke

Paths come from ``XDRIFT_DATA`` / ``XDRIFT_RUNS`` / ``XDRIFT_RESULTS`` (remote-kit
contract); no absolute paths. The graph artefact, frozen split manifest and
preprocessor state are hashed and embedded in the run manifest so every
checkpoint is reproducible.

Checkpoints: ``runs/<strategy>/<model>/seed<seed>/M_<period>.pt``
Metrics:     ``results/text/data_audit/train_<strategy>_<model>_seed<seed>_period<period>.json``
Manifest:    ``results/text/data_audit/train_manifest_<strategy>_<model>_seed<seed>_period<period>.json``

Only single-period fine-tuning is run at this stage (T14 smoke). Replay/EWC
strategies are CLI-selectable but the full matrix waits for the G2 compute plan.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data.graphs import require_torch_geometric, subsample_data  # noqa: E402
from src.models.edge_sage import build_model  # noqa: E402
from src.train.trainer import TrainSettings, train_edge_classifier  # noqa: E402
from src.utils.config import Config, hash_config, load_config  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402
from src.utils.manifest import RunManifest, sha256_file  # noqa: E402
from src.utils.memory import check_memory_budget  # noqa: E402
from src.utils.paths import gate_preflight, load_paths  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402
from src.utils.torch_support import torch_available, torch_status, torch_unavailable_reason  # noqa: E402


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name) or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"config section {name!r} must be a mapping")
    return dict(value)


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is unavailable")
    return requested


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_train", description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--stage", default="S3")
    parser.add_argument("--strategy", default="ft", choices=["ft", "replay", "ewc", "ear_a", "ear_b"])
    parser.add_argument("--model", default=None, help="default from config model.name")
    parser.add_argument("--period", type=int, default=1, help="1-based period to train on")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--smoke", action="store_true", help="subsample the graph for a legacy-lane smoke run")
    parser.add_argument("--max-edges", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--memory-guard",
        default="warn",
        choices=["warn", "error", "off"],
        help="pre-flight full-batch memory estimate: warn (default) / error (abort) / off",
    )
    parser.add_argument("--epochs", type=int, default=None, help="override max epochs (smoke only)")
    parser.add_argument(
        "--max-edges-per-step",
        type=int,
        default=None,
        help="override sampled-subgraph per-step edge cap (memory tuning for the legacy lane)",
    )
    parser.add_argument(
        "--batching",
        default=None,
        choices=["sampled", "graph_batch", "full_batch", "mini_batch"],
        help="override config batching (D-059 canonical is graph_batch; sampled is the T24 alternative mode)",
    )
    parser.add_argument(
        "--reference-path",
        action="store_true",
        help="force the pre-T27 reference pipeline: batching=mini_batch, optimized_dispatch=False, sampler_fast_path=False",
    )
    parser.add_argument("--root", default=None)
    parser.add_argument("--lane", default=None)
    parser.add_argument("--graphs-dir", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    paths = load_paths(lane=args.lane, project_root=args.root)
    root = paths.project_root
    config = load_config(args.config, project_root=root, lane=args.lane)
    config_dict = config.to_dict()
    logger = setup_logging(
        "xdrift.run_train",
        level=str((config.get("logging") or {}).get("level", "INFO")),
        log_path=paths.logs_root / "run_train.log",
    )
    dataset = args.dataset or str(config.get("dataset", "NF-UNSW-NB15-v3"))
    model_name = args.model or str(_section(config_dict, "model").get("name", "egraphsage"))
    seed = int(args.seed if args.seed is not None else config_dict.get("seed", 42))
    seed_everything(seed)
    graphs_dir = Path(args.graphs_dir).expanduser().resolve() if args.graphs_dir else paths.processed_dataset(dataset)
    period = int(args.period)
    graph_file = graphs_dir / f"period_{period}.pt"
    if not graph_file.is_file():
        print(
            f"[train] ERROR: graph artefact not found: {graph_file}. Run src/scripts/build_graphs.py first.",
            file=sys.stderr,
        )
        return 2
    graphs_manifest_path = graphs_dir / "graphs_manifest.json"
    graphs_manifest = (
        json.loads(graphs_manifest_path.read_text(encoding="utf-8")) if graphs_manifest_path.is_file() else {}
    )
    train_cfg = _section(config_dict, "train")
    max_edges = args.max_edges
    if max_edges is None and args.smoke:
        max_edges = int(train_cfg.get("max_edges_smoke", 100000))
    device = "cpu"
    if args.dry_run:
        print(json.dumps({
            "dataset": dataset,
            "graph_file": str(graph_file),
            "model": model_name,
            "strategy": args.strategy,
            "period": period,
            "seed": seed,
            "smoke": bool(args.smoke),
            "max_edges": max_edges,
            "batching": (
                "mini_batch"
                if args.reference_path
                else str(args.batching or train_cfg.get("batching", "graph_batch"))
            ),
            "reference_path": bool(args.reference_path),
            "optimized_dispatch": bool(train_cfg.get("optimized_dispatch", True)) and not args.reference_path,
            "sampler_fast_path": bool(train_cfg.get("sampler_fast_path", True)) and not args.reference_path,
            "neighbor_fanout": train_cfg.get("neighbor_fanout", [10, 10]),
            "max_edges_per_step": int(train_cfg.get("max_edges_per_step", 60000)),
            "chunk_edges": int(train_cfg.get("chunk_edges", 262144)),
            "lane": paths.lane,

            "lane_source": paths.lane_source,

            "config_sha256": config.sha256,
            "graphs_manifest": str(graphs_manifest_path),
            "device": args.device,
        }, indent=2, sort_keys=True))
        return 0
    if not torch_available():
        print(f"[train] ERROR: torch unavailable on this lane: {torch_unavailable_reason()}", file=sys.stderr)
        return 2

    torch, _ = require_torch_geometric()
    import torch as torch_mod

    device = _resolve_device(args.device)
    # T17/T18 gate-grade manifest lifecycle: create + start BEFORE the expensive
    # work so started_at/finished_at/duration reflect real wall clock.
    preflight = gate_preflight(paths)
    for warning in preflight["warnings"]:
        logger.warning("[gate] %s", warning)
    manifest = RunManifest(
        "train_S3",
        paths=paths,
        config=config,
        lane=paths.lane,
        seed=seed,
        dataset=dataset,
        out_path=paths.audit_root
        / f"train_manifest_{args.strategy}_{model_name}_seed{seed}_period{period}.json",
        extra={
            "stage": args.stage,
            "strategy": args.strategy,
            "model": model_name,
            "period": period,
            "smoke": bool(args.smoke),
            "max_edges": max_edges,
            "device": device,
            "gate_preflight": preflight,
        },
    )
    manifest.start()
    logger.info("loading graph %s (device=%s)", graph_file, device)
    data = torch_mod.load(graph_file, weights_only=False)
    original_edges = int(data.num_edges)
    if max_edges is not None and original_edges > int(max_edges):
        data = subsample_data(data, int(max_edges), seed=seed)
        logger.info("smoke subgraph: %s of %s edges", f"{data.num_edges:,}", f"{original_edges:,}")

    # Frozen-state pins: graph file hash + graph manifest fingerprint + split hash.
    graph_sha = sha256_file(graph_file)
    frozen_split_hash = graphs_manifest.get("frozen_split_hash")
    preprocessor_sha = (graphs_manifest.get("preprocessor_state") or {}).get("sha256")
    if graphs_manifest.get("periods") and period in graphs_manifest["periods"]:
        recorded_sha = graphs_manifest["periods"][str(period)]["sha256"]
        if recorded_sha != graph_sha:
            raise RuntimeError(
                f"graph file hash {graph_sha} does not match graphs_manifest hash {recorded_sha}; "
                "frozen artefacts were modified"
            )

    model_cfg = _section(config_dict, "model")
    model_cfg.setdefault("num_classes", int(data.y.max()) + 1)
    model = build_model(
        model_name,
        node_dim=int(data.x.shape[1]),
        edge_dim=int(data.edge_attr.shape[1]),
        config={"model": model_cfg},
    )
    settings = TrainSettings.from_config(config_dict)
    if args.epochs is not None:
        settings.max_epochs = int(args.epochs)
    if args.max_edges_per_step is not None:
        settings.max_edges_per_step = int(args.max_edges_per_step)
    if args.batching is not None:
        settings.batching = str(args.batching)
    if args.reference_path:
        settings.batching = "mini_batch"
        settings.optimized_dispatch = False
        settings.sampler_fast_path = False
    from src.cl.strategies import build_strategy

    strategy = build_strategy(args.strategy, config=_section(config_dict, "strategies"), seed=seed)
    # D-049/T24 memory guard. Canonical sampled batching only allocates one bounded
    # per-step subgraph, so the guard estimates that cap; the full_batch diagnostic
    # estimates the whole-graph activations (which is why it needs the 8 GB lane).
    guard_edges = int(data.num_edges)
    guard_basis = "full_graph"
    if settings.batching == "sampled":
        guard_edges = min(int(data.num_edges), int(settings.max_edges_per_step))
        guard_basis = "per_step_sampled_subgraph"
    memory_report = check_memory_budget(
        n_edges=guard_edges,
        model_config=model_cfg,
        node_dim=int(data.x.shape[1]),
        edge_dim=int(data.edge_attr.shape[1]),
        device=device,
        guard=str(args.memory_guard),
        chunk_edges=int(train_cfg.get("chunk_edges", 262144)),
    )
    memory_report["estimate_basis"] = guard_basis
    memory_report["graph_edges"] = int(data.num_edges)
    if memory_report.get("warning"):
        logger.warning("[memory] %s", memory_report["warning"])
    print(
        json.dumps(
            {
                "memory_guard": {
                    "estimated_peak_gib": memory_report["estimated_peak_gib"],
                    "fits": memory_report["fits"],
                    "device": memory_report["device"].get("device"),
                }
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.memory_guard == "error" and memory_report["fits"] is False:
        print(f"[train] ERROR: {memory_report['warning']}", file=sys.stderr)
        return 2
    logger.info(
        "training %s/%s period=%s seed=%s edges=%s", args.strategy, model_name, period, seed, f"{data.num_edges:,}"
    )
    if args.strategy == "ft":
        result = train_edge_classifier(model, data, settings=settings, seed=seed, device=device)
    else:
        result = strategy.update(model, data, settings=settings, seed=seed, device=device)

    run_dir = paths.run_dir(strategy=args.strategy, model=model_name, seed=seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / f"M_{period}.pt"
    checkpoint = {
        "state_dict": result.get("best_state", model.state_dict()),
        "model_name": model_name,
        "model_config": model_cfg,
        "strategy": args.strategy,
        "period": period,
        "seed": seed,
        "dataset": dataset,
        "graph_file": str(graph_file),
        "graph_sha256": graph_sha,
        "frozen_split_hash": frozen_split_hash,
        "lane": paths.lane,

        "lane_source": paths.lane_source,

        "config_sha256": config.sha256,
    }
    torch_mod.save(checkpoint, checkpoint_path)

    metrics_path = (
        paths.audit_root
        / f"train_{args.strategy}_{model_name}_seed{seed}_period{period}.json"
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "schema_version": 1,
        "dataset": dataset,
        "stage": args.stage,
        "strategy": args.strategy,
        "model": model_name,
        "period": period,
        "seed": seed,
        "smoke": bool(args.smoke),
        "max_edges": max_edges,
        "original_edges": original_edges,
        "frozen_split_hash": frozen_split_hash,
        "graph_sha256": graph_sha,
        "preprocessor_state_sha256": preprocessor_sha,
        "lane": paths.lane,

        "lane_source": paths.lane_source,

        "config_sha256": config.sha256,
        "device": device,
        "torch": torch_status(),
        "result": {key: value for key, value in result.items() if key != "best_state"},
        "memory_guard": memory_report,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")

    manifest.payload.setdefault("extra", {}).update(
        {
            "original_edges": original_edges,
            "frozen_split_hash": frozen_split_hash,
            "graph_sha256": graph_sha,
            "preprocessor_state_sha256": preprocessor_sha,
            "audit_config_sha256": hash_config(config_dict),
            "timing": result.get("timing"),
            "test_macro_f1": (result.get("test") or {}).get("macro_f1"),
            "val_macro_f1": (result.get("val") or {}).get("macro_f1"),
        }
    )
    manifest.add_artifact(checkpoint_path, role="checkpoint")
    manifest.add_artifact(metrics_path, role="metrics")
    manifest.set_dataset(dataset, graph_file.parent, patterns=(graph_file.name,))
    manifest.finish("success")
    manifest.write()

    print(json.dumps({
        "strategy": args.strategy,
        "model": model_name,
        "period": period,
        "seed": seed,
        "device": device,
        "smoke": bool(args.smoke),
        "edges": int(data.num_edges),
        "val_macro_f1": (result.get("val") or {}).get("macro_f1"),
        "test_macro_f1": (result.get("test") or {}).get("macro_f1"),
        "timing": result.get("timing"),
        "checkpoint": str(checkpoint_path),
        "metrics": str(metrics_path),
    }, indent=2, sort_keys=True))
    print(f"[train] manifest: {manifest.resolve_out_path()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
