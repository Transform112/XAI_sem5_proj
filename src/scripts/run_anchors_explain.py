#!/usr/bin/env python
"""T23 anchors + explanations batch driver (resume-friendly, cache-aware).

For every planned run group (model, strategy, seed) and period j:
1. build the S6 anchor set from the frozen test mask + checkpoint ``M_j``;
2. generate IG + GNNExplainer explanations through the S7 cache.

Completed anchor sets (manifest sha matches the anchor file) and completed
explanation batches (canonical ``results/text/tables/explain_index.csv`` already has the
expected rows for that checkpoint/explainer) are skipped, so the driver can be
re-run safely on the canonical lane. Job state is written to
``results/text/tables/explain_state.json``; the batch writes one pinned manifest.

    python src/scripts/run_anchors_explain.py --models egraphsage --strategies ft \
        --seeds 42 --periods 1 --max-jobs 1 --dry-run
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

from src.orchestration.explain_plan import (  # noqa: E402
    DEFAULT_ANCHORS_PER_PERIOD,
    DEFAULT_EXPLAINERS,
    DEFAULT_K_VALUES,
    anchors_command,
    anchors_dir,
    anchors_manifest_check,
    build_jobs,
    canonical_explain_index,
    checkpoint_path,
    eval_out_dir,
    explain_command,
    explain_completion,
    wall_clock_estimate,
)
from src.orchestration.matrix_plan import (  # noqa: E402
    DEFAULT_MODELS,
    DEFAULT_PERIODS,
    DEFAULT_SEEDS,
    DEFAULT_STRATEGIES,
    build_plan,
    filter_plan,
    load_state,
    parse_csv,
    save_state,
    sha256_file,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.gates import gate_record  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402
from src.utils.manifest import RunManifest, utc_now, write_json  # noqa: E402
from src.utils.memory import check_memory_budget  # noqa: E402
from src.utils.paths import load_paths, require_gate_lane  # noqa: E402


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name) or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"config section {name!r} must be a mapping")
    return dict(value)


def _graph_edge_counts(graphs_dir: Path, manifest_path: Path) -> dict[int, int]:
    counts: dict[int, int] = {}
    if manifest_path.is_file():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            for period, entry in (payload.get("periods") or {}).items():
                counts[int(period)] = int(entry.get("n_edges", 0))
        except (OSError, json.JSONDecodeError, ValueError):
            counts = {}
    return counts


def _memory_report(
    *,
    period: int,
    edge_counts: Mapping[int, int],
    model_config: Mapping[str, Any],
    guard: str,
    device: str = "cpu",
) -> dict[str, Any]:
    n_edges = int(edge_counts.get(int(period)) or 0)
    if n_edges <= 0:
        return {"estimated_peak_gib": None, "fits": None, "guard": guard, "note": "edge count unknown"}
    return check_memory_budget(
        n_edges=n_edges,
        model_config=model_config,
        node_dim=int(model_config.get("node_dim", 8)),
        edge_dim=int(model_config.get("edge_dim", 50)),
        device=device,
        guard=guard,
    )


def batch_summary(
    *,
    paths: Any,
    anchor_jobs: Sequence[Any],
    explain_jobs: Sequence[Any],
    explainers: Sequence[str],
    k_values: Sequence[int],
    edge_counts: Mapping[int, int],
    model_config: Mapping[str, Any],
    epochs: int = 100,
) -> dict[str, Any]:
    anchor_states = []
    for job in anchor_jobs:
        state = anchors_manifest_check(paths, job)
        anchor_states.append({"job": job.key, "status": state["status"], "n_selected": state.get("n_selected")})
    explain_states = []
    for job, anchor_job in zip(explain_jobs, anchor_jobs):
        anchor_state = anchors_manifest_check(paths, anchor_job)
        expected = int(anchor_state.get("n_selected") or DEFAULT_ANCHORS_PER_PERIOD)
        state = explain_completion(paths, job, expected_anchors=expected)
        explain_states.append(
            {
                "job": job.key,
                "status": state["status"],
                "expected_anchors": expected,
                "counts": state.get("counts"),
                "expected": state.get("expected"),
            }
        )
    effective_epochs = int(((model_config or {}).get("gnn_explainer_epochs")) or epochs)
    estimate = wall_clock_estimate(
        anchor_jobs,
        explain_jobs,
        explainers=tuple(explainers),
        anchors_per_period=DEFAULT_ANCHORS_PER_PERIOD,
    )
    return {
        "n_anchor_jobs": len(anchor_jobs),
        "n_explain_jobs": len(explain_jobs),
        "anchor_complete": sum(1 for item in anchor_states if item["status"] == "complete"),
        "anchor_pending": sum(1 for item in anchor_states if item["status"] != "complete"),
        "explain_complete": sum(1 for item in explain_states if item["status"] == "complete"),
        "explain_pending": sum(1 for item in explain_states if item["status"] != "complete"),
        "anchors_total_estimate": int(
            sum(int(item.get("n_selected") or DEFAULT_ANCHORS_PER_PERIOD) for item in anchor_states)
        ),
        "explainers": list(explainers),
        "k_values": [int(value) for value in k_values],
        "gnn_explainer_epochs": effective_epochs,
        "anchor_jobs": anchor_states,
        "explain_jobs": explain_states,
        "wall_clock": estimate,
        "commands": {
            "anchors": {
                job.key: anchors_command(job, dataset="NF-UNSW-NB15-v3", device="auto")
                for job in anchor_jobs
            },
            "explain": {
                job.key: explain_command(
                    job,
                    dataset="NF-UNSW-NB15-v3",
                    anchors=str(anchors_dir(paths) / f"period_{int(job.period)}.pt"),
                    device="auto",
                )
                for job in explain_jobs
            },
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_anchors_explain", description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--config", default="configs/explain.yaml")
    parser.add_argument("--eval-config", default="configs/eval.yaml")
    parser.add_argument("--models", default=None)
    parser.add_argument("--strategies", default=None)
    parser.add_argument("--seeds", default=None)
    parser.add_argument("--periods", type=int, default=DEFAULT_PERIODS)
    parser.add_argument("--include-ear", action="store_true")
    parser.add_argument("--max-runs", type=int, default=None, help="cap run groups (staged)")
    parser.add_argument("--max-jobs", type=int, default=None, help="cap total jobs (staged)")
    parser.add_argument("--explainers", default="ig,gnn_explainer")
    parser.add_argument("--k", default="5,10")
    parser.add_argument("--max-anchors", type=int, default=10_000, help="max anchors per period (S6 caps apply)")
    parser.add_argument("--gnn-epochs", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--memory-guard", default="warn", choices=["warn", "error", "off"])
    parser.add_argument("--lane", default=None)
    parser.add_argument("--state-path", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    paths = load_paths(lane=args.lane)
    config = load_config(args.config, project_root=paths.project_root, lane=args.lane)
    config_dict = config.to_dict()
    logger = setup_logging(
        "xdrift.run_anchors_explain",
        level=str((config.get("logging") or {}).get("level", "INFO")),
        log_path=paths.logs_root / "run_anchors_explain.log",
    )
    dataset = args.dataset or str(config.get("dataset", "NF-UNSW-NB15-v3"))
    models = parse_csv(args.models) or DEFAULT_MODELS
    strategies = parse_csv(args.strategies) or DEFAULT_STRATEGIES
    seeds = parse_csv(args.seeds, cast=int) or DEFAULT_SEEDS
    explainers = parse_csv(args.explainers) or DEFAULT_EXPLAINERS
    k_values = parse_csv(args.k, cast=int) or DEFAULT_K_VALUES
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
    anchor_jobs, explain_jobs = build_jobs(
        plan, periods=int(args.periods), explainers=explainers, k_values=k_values
    )
    if args.max_jobs is not None:
        anchor_jobs = anchor_jobs[: int(args.max_jobs)]
        explain_jobs = explain_jobs[: int(args.max_jobs)]
    graphs_dir = Path(paths.processed_dataset(dataset))
    edge_counts = _graph_edge_counts(graphs_dir, graphs_dir / "graphs_manifest.json")
    model_cfg = {**_section(load_config("configs/train.yaml", project_root=paths.project_root, lane=args.lane).to_dict(), "model"), "node_dim": 8, "edge_dim": 50}
    summary = batch_summary(
        paths=paths,
        anchor_jobs=anchor_jobs,
        explain_jobs=explain_jobs,
        explainers=explainers,
        k_values=k_values,
        edge_counts=edge_counts,
        model_config=model_cfg,
        epochs=int(args.gnn_epochs or 100),
    )
    state_path = (
        Path(args.state_path) if args.state_path else Path(paths.tables_root) / "explain_state.json"
    )
    payload = {
        "dataset": dataset,
        "lane": paths.lane,
        "lane_source": paths.lane_source,
        "gate": gate_record("anchor_explain_batch", paths),
        "models": list(models),
        "strategies": list(strategies),
        "seeds": list(seeds),
        "periods": int(args.periods),
        "max_runs": args.max_runs,
        "max_jobs": args.max_jobs,
        "max_anchors": int(args.max_anchors),
        "state_path": str(state_path),
        "summary": summary,
    }
    if args.dry_run:
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
        return 0
    try:
        require_gate_lane(paths, allow_default_lane=False)
    except Exception as exc:  # noqa: BLE001
        print(f"[explain-batch] ERROR: {exc}", file=sys.stderr)
        return 2
    state = load_state(state_path)
    manifest = RunManifest(
        "anchor_explain_batch",
        paths=paths,
        config=config,
        lane=paths.lane,
        seed=int(seeds[0]) if seeds else 42,
        dataset=dataset,
        out_path=Path(paths.audit_root) / "anchor_explain_batch_manifest.json",
        extra={"plan": {key: value for key, value in payload.items() if key != "summary"}, "gate": gate_record("anchor_explain_batch", paths)},
    )
    manifest.start()
    failures = 0
    try:
        for job in anchor_jobs:
            state_key = f"anchors:{job.key}"
            anchor_state = anchors_manifest_check(paths, job)
            if anchor_state["status"] == "complete":
                state["groups"][state_key] = {"status": "complete", "updated_at_utc": utc_now()}
                continue
            if not checkpoint_path(paths, job.run, job.period).is_file():
                logger.warning("skip anchors %s: checkpoint missing", job.key)
                state["groups"][state_key] = {"status": "blocked", "reason": "checkpoint missing"}
                continue
            memory = _memory_report(
                period=job.period,
                edge_counts=edge_counts,
                model_config=model_cfg,
                guard=str(args.memory_guard),
                device=args.device,
            )
            if memory.get("warning"):
                logger.warning("[memory] %s", memory["warning"])
            if args.memory_guard == "error" and memory.get("fits") is False:
                logger.error("anchors %s aborted by memory guard", job.key)
                failures += 1
                continue
            command = anchors_command(
                job,
                dataset=dataset,
                checks_root=None,
                graphs_dir=None,
                out_dir=None,
                device=args.device,
                lane=args.lane,
            )
            import scripts.build_anchors as build_anchors_script

            code = build_anchors_script.main([str(item) for item in command[2:]])
            state["groups"][state_key] = {
                "status": "complete" if int(code) == 0 else "failed",
                "exit_code": int(code),
                "updated_at_utc": utc_now(),
            }
            if int(code) != 0:
                failures += 1
            save_state(state_path, state)
        for job in explain_jobs:
            state_key = f"explain:{job.key}"
            anchor_job = next((item for item in anchor_jobs if item.run.key == job.run.key and item.period == job.period), None)
            expected = DEFAULT_ANCHORS_PER_PERIOD
            if anchor_job is not None:
                anchor_state = anchors_manifest_check(paths, anchor_job)
                expected = int(anchor_state.get("n_selected") or DEFAULT_ANCHORS_PER_PERIOD)
            explain_state = explain_completion(paths, job, expected_anchors=expected)
            if explain_state["status"] == "complete":
                state["groups"][state_key] = {"status": "complete", "updated_at_utc": utc_now()}
                continue
            command = explain_command(
                job,
                dataset=dataset,
                anchors=str(anchors_dir(paths) / f"period_{int(job.period)}.pt"),
                device=args.device,
                lane=args.lane,
                max_anchors=int(args.max_anchors),
            )
            import scripts.run_explain as run_explain_script

            code = run_explain_script.main([str(item) for item in command[2:]])
            state["groups"][state_key] = {
                "status": "complete" if int(code) == 0 else "failed",
                "exit_code": int(code),
                "expected_anchors": expected,
                "updated_at_utc": utc_now(),
            }
            if int(code) != 0:
                failures += 1
            save_state(state_path, state)
        summary_path = Path(paths.audit_root) / "anchor_explain_batch_summary.json"
        write_json(summary_path, {"schema_version": 1, "generated_at_utc": utc_now(), "summary": summary, "state": str(state_path)})
        manifest.add_artifact(summary_path, role="batch_summary")
        canonical_index = canonical_explain_index(paths)
        if canonical_index.is_file():
            manifest.add_artifact(canonical_index, role="explain_index")
        manifest.finish("success" if failures == 0 else "failed", error=None if failures == 0 else f"{failures} job(s) failed")
        manifest.write()
        print(
            json.dumps(
                {
                    "batch": summary.get("n_anchor_jobs"),
                    "explain_jobs": summary.get("n_explain_jobs"),
                    "failures": failures,
                    "state": str(state_path),
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
