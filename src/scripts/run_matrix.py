#!/usr/bin/env python
"""T23 resume-friendly strategy-matrix driver (sequential continual updates).

One Python process per run group (model, strategy, seed): period 1 trains M_1,
periods 2..5 continue the SAME model through the strategy update so replay buffers
and EWC Fisher/theta* persist across the frozen C1 period sequence. Every period
writes a checkpoint, metrics JSON and a pinned per-run manifest; the group finishes
with an S5 evaluation into ``results/text/tables/matrix/<run_key>/`` (R-matrix inputs),
a group manifest, a resumable state file and ``results/text/tables/matrix_index.csv``.

    # inspect the full grid (no runs):
    python src/scripts/run_matrix.py --models egraphsage,gcn --strategies ft,replay,ewc \
        --seeds 42,43,44,45,46 --dry-run
    # staged execution:
    python src/scripts/run_matrix.py --lane rtx5060 --strategies ft --seeds 42 --max-runs 1

Completed period steps (checkpoint sha matches the pinned manifest) are skipped;
incomplete groups restart from period 1 so the continual sequence stays consistent.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.orchestration.matrix_plan import (  # noqa: E402
    DEFAULT_MODELS,
    DEFAULT_PERIODS,
    DEFAULT_SEEDS,
    DEFAULT_STRATEGIES,
    ESTIMATE_EPOCHS_PER_PERIOD,
    MatrixRun,
    build_plan,
    checkpoint_path,
    eval_completion,
    eval_out_dir,
    filter_plan,
    group_completion,
    group_manifest_path,
    load_index,
    load_state,
    metrics_path,
    parse_csv,
    run_dir,
    run_manifest_path,
    save_state,
    sha256_file,
    step_completion,
    train_command,
    upsert_index_row,
    wall_clock_estimate,
    write_index,
    INDEX_COLUMNS,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.gates import gate_record  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402
from src.utils.manifest import RunManifest, utc_now, write_json  # noqa: E402
from src.utils.memory import check_memory_budget, estimate_training_peak_bytes  # noqa: E402
from src.utils.paths import load_paths, require_gate_lane  # noqa: E402


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name) or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"config section {name!r} must be a mapping")
    return dict(value)


def _graph_edge_counts(graphs_dir: Path, graphs_manifest_path: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    if graphs_manifest_path.is_file():
        try:
            payload = json.loads(graphs_manifest_path.read_text(encoding="utf-8"))
            for period, entry in (payload.get("periods") or {}).items():
                counts[int(period)] = int(entry.get("n_edges", 0))
        except (OSError, json.JSONDecodeError, ValueError):
            counts = {}
    if not counts:
        for path in sorted(graphs_dir.glob("period_*.pt")):
            counts[int(path.stem.split("_")[1])] = 0
    return counts


def _memory_estimate(
    *,
    run: MatrixRun,
    period: int,
    edge_counts: Mapping[int, int],
    model_config: Mapping[str, Any],
    guard: str,
    device: str = "cpu",
    batching: str = "sampled",
    max_edges_per_step: int = 60_000,
) -> dict[str, Any]:
    graph_edges = int(edge_counts.get(int(period)) or 0)
    if graph_edges <= 0:
        return {"estimated_peak_gib": None, "fits": None, "guard": guard, "note": "edge count unknown"}
    # D-052 canonical sampled batching allocates one bounded subgraph per step;
    # the full_batch diagnostic is the one that needs whole-graph activations.
    n_edges = min(graph_edges, int(max_edges_per_step)) if batching == "sampled" else graph_edges
    report = check_memory_budget(
        n_edges=n_edges,
        model_config=model_config,
        node_dim=int(model_config.get("node_dim", 8)),
        edge_dim=int(model_config.get("edge_dim", 50)),
        device=device,
        guard=guard,
    )
    report.update(
        {
            "estimate_basis": "per_step_sampled_subgraph" if batching == "sampled" else "full_graph",
            "graph_edges": graph_edges,
            "batching": batching,
        }
    )
    return report


def _resolve_epoch_anchor(args: argparse.Namespace) -> tuple[float | None, str]:
    """Measured epoch-time anchor: explicit flag or a metrics JSON (no silent default)."""
    if args.epoch_seconds_full_size is not None:
        value = float(args.epoch_seconds_full_size)
        if value <= 0:
            raise ValueError("--epoch-seconds-full-size must be > 0")
        return value, "flag"
    if args.anchor_metrics:
        path = Path(args.anchor_metrics).expanduser().resolve()
        if not path.is_file():
            raise ValueError(f"--anchor-metrics file not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        timing = (payload.get("result") or {}).get("timing") if isinstance(payload, dict) else None
        timing = timing or payload.get("timing") or {}
        value = timing.get("epoch_seconds_mean")
        if value is None:
            raise ValueError(f"--anchor-metrics has no timing.epoch_seconds_mean: {path}")
        return float(value), f"metrics:{path}"
    return None, "missing"


def plan_summary(
    plan: Sequence[MatrixRun],
    paths: Any,
    *,
    edge_counts: Mapping[int, int] | None = None,
    model_config: Mapping[str, Any] | None = None,
    epoch_seconds_full_size: float = 20.0,
) -> dict[str, Any]:
    groups = [group_completion(paths, run) for run in plan]
    commands = {}
    for run in plan:
        commands[run.key] = {
            "period_commands": [
                train_command(run, period, dataset="NF-UNSW-NB15-v3")
                for period in run.steps
            ],
            "eval_out_dir": str(eval_out_dir(paths, run)),
            "execution": "in-process sequential continual updates (strategy state preserved)",
        }
    estimates = wall_clock_estimate(
        plan,
        epoch_seconds_full_size=epoch_seconds_full_size,
        period_edge_counts=edge_counts,
    )
    return {
        "n_groups": len(plan),
        "n_period_updates": sum(len(run.steps) for run in plan),
        "n_eval_groups": len(plan),
        "complete_groups": sum(1 for group in groups if group["status"] == "complete"),
        "partial_groups": sum(1 for group in groups if group["status"] == "partial"),
        "pending_groups": sum(1 for group in groups if group["status"] == "missing"),
        "groups": groups,
        "commands": commands,
        "wall_clock": estimates,
        "config": "configs/train.yaml + configs/eval.yaml",
        "batching": "sampled (D-052 canonical; full_batch diagnostic only)",
        "s5_eval": "src/scripts/run_eval.py --out-dir results/text/tables/matrix/<run_key>",
    }


def _run_group_eval(
    run: MatrixRun,
    *,
    paths: Any,
    dataset: str,
    device: str,
    lane: str | None,
    runner: Any = subprocess.run,
    eval_chunk_edges: int = 0,
) -> dict[str, Any]:
    """Run (or skip) the S5 evaluation for one completed group.

    T30: the eval CLI is memory-bounded (edge_batch head), so a group whose
    checkpoints are complete can be evaluated on the 3 GB lane without retraining.
    """
    eval_state = eval_completion(paths, run)
    if eval_state["status"] == "complete":
        return {"eval_seconds": 0.0, "eval_complete": True}
    out_dir = eval_out_dir(paths, run)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    eval_started = time.monotonic()
    command = [
        sys.executable,
        str(_ROOT / "scripts" / "run_eval.py"),
        "--dataset",
        dataset,
        "--config",
        "configs/eval.yaml",
        "--strategy",
        run.strategy,
        "--model",
        run.model,
        "--seed",
        str(int(run.seed)),
        "--device",
        str(device),
        "--out-dir",
        str(out_dir),
    ]
    if int(eval_chunk_edges or 0) > 0:
        command += ["--eval-chunk-edges", str(int(eval_chunk_edges))]
    if lane:
        command += ["--lane", str(lane)]
    completed = runner(command, cwd=str(_ROOT), capture_output=True, text=True)
    eval_seconds = time.monotonic() - eval_started
    if int(getattr(completed, "returncode", 1)) != 0:
        raise RuntimeError(
            f"S5 eval failed for {run.key}: {getattr(completed, 'stderr', '')[-2000:]}"
        )
    return {
        "eval_seconds": round(eval_seconds, 2),
        "eval_complete": eval_completion(paths, run)["status"] == "complete",
    }


def _execute_group(
    run: MatrixRun,
    *,
    paths: Any,
    dataset: str,
    config_path: str,
    config_dict: Mapping[str, Any],
    device: str,
    lane: str | None,
    memory_guard: str,
    epochs: int | None,
    logger: Any,
    runner: Any = subprocess.run,
    eval_chunk_edges: int = 0,
) -> dict[str, Any]:
    """Run one group: sequential continual updates + S5 evaluation."""
    import torch

    from src.cl.strategies import build_strategy
    from src.models.edge_sage import build_model
    from src.train.trainer import TrainSettings
    from src.utils.seed import seed_everything

    group_start = time.monotonic()
    graphs_dir = Path(paths.processed_dataset(dataset))
    graphs_manifest = graphs_dir / "graphs_manifest.json"
    edge_counts = _graph_edge_counts(graphs_dir, graphs_manifest)
    train_cfg = _section(config_dict, "train")
    model_cfg = _section(config_dict, "model")
    settings = TrainSettings.from_config(config_dict)
    if epochs is not None:
        settings.max_epochs = int(epochs)
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(int(run.seed))
    # Only a fully-complete prefix may be resumed; any partial prefix restarts at
    # period 1 so the continual sequence and strategy state stay consistent.
    completed_prefix = 0
    for period in run.steps:
        if step_completion(paths, run, period)["status"] == "complete":
            completed_prefix += 1
        else:
            break
    if 0 < completed_prefix < len(run.steps):
        logger.warning(
            "matrix %s has %s/%s completed periods; restarting the group from period 1 for a consistent continual sequence",
            run.key,
            completed_prefix,
            len(run.steps),
        )
        completed_prefix = 0
    model = None
    strategy = None
    completed_periods: list[int] = []
    for period in run.steps:
        started = time.monotonic()
        existing = step_completion(paths, run, period)
        checkpoint = checkpoint_path(paths, run, period)
        graph_file = graphs_dir / f"period_{period}.pt"
        if not graph_file.is_file():
            raise FileNotFoundError(f"frozen graph missing: {graph_file}")
        if model is None:
            first_graph = torch.load(graph_file, weights_only=False)
            model_cfg.setdefault("num_classes", int(first_graph.y.max()) + 1)
            model = build_model(
                run.model,
                node_dim=int(first_graph.x.shape[1]),
                edge_dim=int(first_graph.edge_attr.shape[1]),
                config={"model": model_cfg},
            )
            strategy = build_strategy(
                run.strategy,
                config=_section(config_dict, "strategies"),
                seed=int(run.seed),
            )
            memory = _memory_estimate(
                run=run,
                period=period,
                edge_counts=edge_counts,
                model_config={**model_cfg, "node_dim": int(first_graph.x.shape[1]), "edge_dim": int(first_graph.edge_attr.shape[1])},
                guard=memory_guard,
                device=device,
                batching=settings.batching,
                max_edges_per_step=int(settings.max_edges_per_step),
            )
            if memory.get("warning"):
                logger.warning("[memory] %s", memory["warning"])
            if memory_guard == "error" and memory.get("fits") is False:
                raise RuntimeError(memory["warning"])
            del first_graph
        # T36 fix: start the per-period manifest lifecycle BEFORE training so
        # started/finished/duration reflect the real wall clock (verifier
        # KAG.manifest_lifecycle). The manifest is written after the artifacts
        # exist; execution failures still leave the parent manifest failed.
        run_manifest = RunManifest(
            "train_S3",
            paths=paths,
            config=load_config(config_path, project_root=paths.project_root, lane=lane),
            lane=paths.lane,
            seed=int(run.seed),
            dataset=dataset,
            out_path=run_manifest_path(paths, run, period),
            extra={
                "driver": "run_matrix",
                "run_key": run.key,
                "strategy": run.strategy,
                "model": run.model,
                "period": int(period),
                "device": str(device),
                "gate": gate_record("matrix_run", paths),
            },
        )
        run_manifest.start()
        data = torch.load(graph_file, weights_only=False)
        result = strategy.update(model, data, settings=settings, seed=int(run.seed), device=device)
        run_dir_path = run_dir(paths, run)
        run_dir_path.mkdir(parents=True, exist_ok=True)
        checkpoint_payload = {
            "state_dict": result.get("best_state", model.state_dict()),
            "model_name": run.model,
            "model_config": model_cfg,
            "strategy": run.strategy,
            "period": int(period),
            "seed": int(run.seed),
            "dataset": dataset,
            "graph_file": str(graph_file),
            "graph_sha256": sha256_file(graph_file),
            "frozen_split_hash": (json.loads(graphs_manifest.read_text(encoding="utf-8"))).get("frozen_split_hash") if graphs_manifest.is_file() else None,
            "lane": paths.lane,
            "lane_source": paths.lane_source,
            "config_sha256": sha256_file(Path(config_path).resolve()),
        }
        torch.save(checkpoint_payload, checkpoint)
        metrics_path_value = metrics_path(paths, run, period)
        metrics_path_value.parent.mkdir(parents=True, exist_ok=True)
        metrics = {
            "schema_version": 1,
            "dataset": dataset,
            "stage": "S3",
            "driver": "run_matrix",
            "strategy": run.strategy,
            "model": run.model,
            "period": int(period),
            "seed": int(run.seed),
            "device": str(device),
            "batching": settings.batching,
            "chunk_edges": int(settings.chunk_edges),
            "result": {key: value for key, value in result.items() if key != "best_state"},
        }
        write_json(metrics_path_value, metrics)
        run_manifest.add_artifact(checkpoint, role="checkpoint")
        run_manifest.add_artifact(metrics_path_value, role="metrics")
        run_manifest.add_artifact(graph_file, role="graph", hash_file=False)
        run_manifest.finish("success", extra={"timing": result.get("timing")})
        run_manifest.write()
        completed_periods.append(period)
        logger.info(
            "matrix %s period %s done in %.1fs (val_macro_f1=%s)",
            run.key,
            period,
            time.monotonic() - started,
            (result.get("val") or {}).get("macro_f1"),
        )
        del data
    # T32: release the parent process's training graph/model and cached blocks before
    # the S5 eval subprocess starts; on a 3 GB lane the parent's residual allocations
    # otherwise make the (head-only) eval OOM.
    if "model" in locals():
        del model
    if "strategy" in locals():
        del strategy
    if "device_graph" in locals():
        del device_graph
    import gc

    gc.collect()
    if str(device).startswith("cuda"):
        try:
            import torch as torch_mod

            torch_mod.cuda.empty_cache()
        except Exception:  # noqa: BLE001 - cleanup must never break the group
            pass
    # S5 evaluation per run group (R-matrix inputs for this group).
    eval_result = _run_group_eval(
        run,
        paths=paths,
        dataset=dataset,
        device=device,
        lane=lane,
        runner=runner,
        eval_chunk_edges=eval_chunk_edges,
    )
    group_seconds = time.monotonic() - group_start
    return {
        "key": run.key,
        "periods_completed": completed_periods,
        "eval_seconds": eval_result["eval_seconds"],
        "duration_s": round(group_seconds, 2),
        "eval_complete": eval_result["eval_complete"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_matrix", description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--models", default=None, help="comma list, default egraphsage,gcn")
    parser.add_argument("--strategies", default=None, help="comma list, default ft,replay,ewc")
    parser.add_argument("--seeds", default=None, help="comma list, default 42,43,44,45,46")
    parser.add_argument("--periods", type=int, default=DEFAULT_PERIODS)
    parser.add_argument("--include-ear", action="store_true", help="add ear_a,ear_b strategies")
    parser.add_argument("--max-runs", type=int, default=None, help="staged execution cap")
    parser.add_argument("--epochs", type=int, default=None, help="override max epochs (smoke only)")
    parser.add_argument("--eval-chunk-edges", type=int, default=0, help="T30 eval head chunk cap (0 = edge_batch path)")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--memory-guard", default="warn", choices=["warn", "error", "off"])
    parser.add_argument("--lane", default=None)
    parser.add_argument(
        "--epoch-seconds-full-size",
        type=float,
        default=None,
        help="measured sampled-batch epoch time at period-1 size (REQUIRED for non-dry-run; "
        "kit-v7 RTX 5060 value)",
    )
    parser.add_argument(
        "--anchor-metrics",
        default=None,
        help="metrics JSON to derive the epoch anchor from (timing.epoch_seconds_mean)",
    )
    parser.add_argument("--out-state", default=None)
    parser.add_argument("--out-index", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    paths = load_paths(lane=args.lane)
    config = load_config(args.config, project_root=paths.project_root, lane=args.lane)
    config_dict = config.to_dict()
    logger = setup_logging(
        "xdrift.run_matrix",
        level=str((config.get("logging") or {}).get("level", "INFO")),
        log_path=paths.logs_root / "run_matrix.log",
    )
    dataset = args.dataset or str(config.get("dataset", "NF-UNSW-NB15-v3"))
    models = parse_csv(args.models) or DEFAULT_MODELS
    strategies = parse_csv(args.strategies) or DEFAULT_STRATEGIES
    seeds = parse_csv(args.seeds, cast=int) or DEFAULT_SEEDS
    unknown_models = [model for model in models if str(model) not in ("egraphsage", "gcn")]
    if unknown_models:
        print(f"[matrix] ERROR: unknown models {unknown_models}", file=sys.stderr)
        return 2
    known = {"ft", "replay", "ewc", "ear_a", "ear_b"}
    unknown_strategies = [name for name in strategies if str(name) not in known]
    if unknown_strategies:
        print(f"[matrix] ERROR: unknown strategies {unknown_strategies}", file=sys.stderr)
        return 2
    plan = filter_plan(
        build_plan(
            models=[str(model) for model in models],
            strategies=[str(name) for name in strategies],
            seeds=[int(seed) for seed in seeds],
            periods=int(args.periods),
            include_ear=bool(args.include_ear),
        ),
        max_runs=args.max_runs,
    )
    state_path = Path(args.out_state) if args.out_state else Path(paths.tables_root) / "matrix_state.json"
    index_path = Path(args.out_index) if args.out_index else Path(paths.tables_root) / "matrix_index.csv"
    graphs_dir = Path(paths.processed_dataset(dataset))
    edge_counts = _graph_edge_counts(graphs_dir, graphs_dir / "graphs_manifest.json")
    model_cfg = {**_section(config_dict, "model"), "node_dim": 8, "edge_dim": 50}
    try:
        anchor_seconds, anchor_source = _resolve_epoch_anchor(args)
    except ValueError as exc:
        print(f"[matrix] ERROR: {exc}", file=sys.stderr)
        return 2
    if anchor_seconds is None:
        if not args.dry_run:
            print(
                "[matrix] ERROR: a measured epoch anchor is REQUIRED for non-dry-run execution. "
                "Pass --epoch-seconds-full-size <kit-v7 measured epoch s> or --anchor-metrics <metrics.json>; "
                "refusing to budget or run with a placeholder anchor.",
                file=sys.stderr,
            )
            return 2
        anchor_source = "placeholder"
    summary = plan_summary(
        plan,
        paths,
        edge_counts=edge_counts,
        model_config=model_cfg,
        epoch_seconds_full_size=anchor_seconds,
    )
    wall_clock = summary["wall_clock"]
    budget = {
        "anchor_seconds": wall_clock["epoch_seconds_full_size"],
        "anchor_source": anchor_source,
        "placeholder_anchor": bool(wall_clock["placeholder_anchor"]),
        "projected_gpu_hours": wall_clock["total_gpu_hours"],
        "budget_gpu_hours": wall_clock["budget_gpu_hours"],
        "over_budget": wall_clock["over_budget"],
        "by_strategy_gpu_hours": wall_clock["by_strategy_gpu_hours"],
        "by_model_gpu_hours": wall_clock["by_model_gpu_hours"],
        "warnings": list(wall_clock["warnings"]),
    }
    for warning in budget["warnings"]:
        print(f"[matrix] WARNING: {warning}", file=sys.stderr)
    plan_payload = {
        "dataset": dataset,
        "lane": paths.lane,
        "lane_source": paths.lane_source,
        "gate": gate_record("matrix_run", paths),
        "models": list(models),
        "strategies": list(strategies),
        "seeds": list(seeds),
        "periods": int(args.periods),
        "max_runs": args.max_runs,
        "state_path": str(state_path),
        "index_path": str(index_path),
        "summary": summary,
        "budget": budget,
        "filters": {"models": list(models), "strategies": list(strategies), "seeds": list(seeds)},
    }
    if args.dry_run:
        print(json.dumps(plan_payload, indent=2, sort_keys=True, default=str))
        return 0
    # Real execution is a gate artifact: require an explicit lane.
    try:
        require_gate_lane(paths, allow_default_lane=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[matrix] ERROR: {exc}", file=sys.stderr)
        return 2
    state = load_state(state_path)
    index_rows = load_index(index_path)
    manifest = RunManifest(
        "matrix_run",
        paths=paths,
        config=config,
        lane=paths.lane,
        seed=int(seeds[0]),
        dataset=dataset,
        out_path=Path(paths.audit_root) / "matrix_manifest.json",
        extra={"plan": {key: value for key, value in plan_payload.items() if key != "summary"}, "gate": gate_record("matrix_run", paths)},
    )
    manifest.start()
    failures = 0
    try:
        for run in plan:
            completion = group_completion(paths, run)
            if completion["status"] == "complete":
                logger.info("skip complete run %s", run.key)
                upsert_index_row(index_rows, run=run, kind="group", period=0, status="complete", eval_out_dir=str(eval_out_dir(paths, run)))
                state["groups"][run.key] = {
                    "status": "complete",
                    "model": run.model,
                    "strategy": run.strategy,
                    "seed": int(run.seed),
                    "periods_completed": list(run.steps),
                    "eval_complete": True,
                    "updated_at_utc": utc_now(),
                }
                save_state(state_path, state)
                write_index(index_path, index_rows)
                continue
            group_started = utc_now()
            steps_complete = all(step["status"] == "complete" for step in completion["steps"])
            if steps_complete and completion["eval"]["status"] != "complete":
                # T30 resume path: periods are complete, only the S5 eval is missing
                # (the canary case). Evaluate without retraining anything.
                logger.info("matrix %s: all periods complete; evaluating only (resume)", run.key)
                try:
                    eval_result = _run_group_eval(
                        run,
                        paths=paths,
                        dataset=dataset,
                        device=args.device,
                        lane=args.lane,
                        eval_chunk_edges=int(args.eval_chunk_edges),
                    )
                    status = "complete" if eval_result["eval_complete"] else "partial"
                    for period in run.steps:
                        upsert_index_row(
                            index_rows, run=run, kind="train", period=period, status="complete",
                            checkpoint=str(checkpoint_path(paths, run, period)),
                            manifest=str(run_manifest_path(paths, run, period)),
                        )
                    upsert_index_row(
                        index_rows, run=run, kind="eval", period=0,
                        status="complete" if eval_result["eval_complete"] else "partial",
                        eval_out_dir=str(eval_out_dir(paths, run)),
                        duration_s=eval_result["eval_seconds"],
                    )
                    upsert_index_row(
                        index_rows, run=run, kind="group", period=0, status=status,
                        eval_out_dir=str(eval_out_dir(paths, run)),
                        started_at_utc=group_started, finished_at_utc=utc_now(),
                        duration_s=eval_result["eval_seconds"],
                    )
                    state["groups"][run.key] = {
                        "status": status,
                        "model": run.model,
                        "strategy": run.strategy,
                        "seed": int(run.seed),
                        "periods_completed": list(run.steps),
                        "eval_complete": eval_result["eval_complete"],
                        "eval_only_resume": True,
                        "duration_s": eval_result["eval_seconds"],
                        "updated_at_utc": utc_now(),
                    }
                    logger.info("matrix run %s eval-only resume -> %s", run.key, status)
                except Exception as exc:  # noqa: BLE001
                    failures += 1
                    logger.exception("matrix %s eval-only resume failed", run.key)
                    upsert_index_row(
                        index_rows, run=run, kind="group", period=0, status="failed",
                        started_at_utc=group_started, finished_at_utc=utc_now(),
                        error=f"{type(exc).__name__}: {exc}",
                    )
                save_state(state_path, state)
                write_index(index_path, index_rows)
                continue
            try:
                result = _execute_group(
                    run,
                    paths=paths,
                    dataset=dataset,
                    config_path=args.config,
                    config_dict=config_dict,
                    device=args.device,
                    lane=args.lane,
                    memory_guard=str(args.memory_guard),
                    epochs=args.epochs,
                    logger=logger,
                    eval_chunk_edges=int(args.eval_chunk_edges),
                )
                status = "complete" if result["eval_complete"] else "partial"
                for period in run.steps:
                    step = step_completion(paths, run, period)
                    upsert_index_row(
                        index_rows,
                        run=run,
                        kind="train",
                        period=period,
                        status=step["status"],
                        checkpoint=str(checkpoint_path(paths, run, period)),
                        manifest=str(run_manifest_path(paths, run, period)),
                    )
                upsert_index_row(
                    index_rows,
                    run=run,
                    kind="eval",
                    period=0,
                    status="complete" if result["eval_complete"] else "partial",
                    eval_out_dir=str(eval_out_dir(paths, run)),
                    duration_s=result["eval_seconds"],
                )
                upsert_index_row(
                    index_rows,
                    run=run,
                    kind="group",
                    period=0,
                    status=status,
                    eval_out_dir=str(eval_out_dir(paths, run)),
                    started_at_utc=group_started,
                    finished_at_utc=utc_now(),
                    duration_s=result["duration_s"],
                )
                state["groups"][run.key] = {
                    "status": status,
                    "model": run.model,
                    "strategy": run.strategy,
                    "seed": int(run.seed),
                    "periods_completed": result["periods_completed"],
                    "eval_complete": result["eval_complete"],
                    "duration_s": result["duration_s"],
                    "updated_at_utc": utc_now(),
                }
                group_manifest = {
                    "schema_version": 1,
                    "kind": "matrix_run_group",
                    "generated_at_utc": utc_now(),
                    "run_key": run.key,
                    "model": run.model,
                    "strategy": run.strategy,
                    "seed": int(run.seed),
                    "periods": int(run.periods),
                    "status": status,
                    "checkpoints": {
                        str(period): {
                            "path": str(checkpoint_path(paths, run, period)),
                            "sha256": sha256_file(checkpoint_path(paths, run, period)),
                        }
                        for period in run.steps
                        if checkpoint_path(paths, run, period).is_file()
                    },
                    "eval_out_dir": str(eval_out_dir(paths, run)),
                    "eval_complete": bool(result["eval_complete"]),
                    "timing": {"duration_s": result["duration_s"], "eval_seconds": result["eval_seconds"]},
                    "lane": paths.lane,
                    "lane_source": paths.lane_source,
                }
                write_json(group_manifest_path(paths, run), group_manifest)
                manifest.add_artifact(group_manifest_path(paths, run), role="group_manifest")
                for period in run.steps:
                    manifest.add_artifact(checkpoint_path(paths, run, period), role="checkpoint")
                logger.info("matrix run %s -> %s", run.key, status)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                logger.exception("matrix run %s failed", run.key)
                state["groups"][run.key] = {
                    "status": "failed",
                    "model": run.model,
                    "strategy": run.strategy,
                    "seed": int(run.seed),
                    "error": f"{type(exc).__name__}: {exc}",
                    "updated_at_utc": utc_now(),
                }
                upsert_index_row(
                    index_rows,
                    run=run,
                    kind="group",
                    period=0,
                    status="failed",
                    started_at_utc=group_started,
                    finished_at_utc=utc_now(),
                    error=f"{type(exc).__name__}: {exc}",
                )
                save_state(state_path, state)
                write_index(index_path, index_rows)
                if args.max_runs and failures >= 1:
                    break
            save_state(state_path, state)
            write_index(index_path, index_rows)
        manifest.finish("success" if failures == 0 else "failed", error=None if failures == 0 else f"{failures} run(s) failed")
        manifest.write()
        print(
            json.dumps(
                {
                    "groups_planned": len(plan),
                    "failures": failures,
                    "state": str(state_path),
                    "index": str(index_path),
                    "manifest": str(manifest.resolve_out_path()),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0 if failures == 0 else 1
    except Exception as exc:  # noqa: BLE001
        manifest.finish("failed", error=f"{type(exc).__name__}: {exc}")
        manifest.write()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
