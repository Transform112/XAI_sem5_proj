#!/usr/bin/env python
"""S5 evaluation CLI: R[i,j] checkpoint x period matrix + CL metrics (D-033 dual F1).

    python src/scripts/run_eval.py --dataset NF-UNSW-NB15-v3 --strategy ft --model egraphsage --seed 42

Reads pinned checkpoints ``<runs>/<strategy>/<model>/seed<seed>/M_<i>.pt`` and
the frozen per-period graphs, evaluates every available checkpoint on every
period's frozen test mask, and writes:

    results/text/tables/cl_metrics.csv          (one row per run; all-class + supported macro-F1)
    results/text/tables/cl_per_class_f1.json    (per-class F1 tensor for RQ3)
    results/text/tables/cl_matrix.json          (R matrices + dual-F1 details)
    results/text/data_audit/eval_manifest_*.json (pinned run manifest)

No training is performed and the frozen data state is not touched.
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

from src.eval.cl_metrics import (  # noqa: E402
    DEFAULT_SUPPORTED_SET,
    FORGETTING_CONVENTION,
    cl_metrics,
    dual_macro_f1,
    matrix_from_scores,
    per_class_f1,
    supported_ids_from_names,
)
from src.eval.predict import predict_logits, predict_mode  # noqa: E402
from src.models.edge_sage import build_model  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402
from src.utils.manifest import RunManifest, sha256_file, utc_now, write_json  # noqa: E402
from src.utils.paths import load_paths  # noqa: E402
from src.utils.torch_support import torch_available, torch_unavailable_reason  # noqa: E402


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name) or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"config section {name!r} must be a mapping")
    return dict(value)


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def load_checkpoint_model(torch, checkpoint_path: Path, *, device: str):
    payload = torch.load(checkpoint_path, weights_only=False)
    model_config = dict(payload.get("model_config") or {})
    data_x_dim = int(payload.get("node_dim", 8)) if "node_dim" in payload else 8
    node_dim = int(payload.get("node_feature_dim", data_x_dim))
    edge_dim = None
    if "edge_attr_dim" in payload:
        edge_dim = int(payload["edge_attr_dim"])
    return payload, model_config, node_dim, edge_dim


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_eval", description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--strategy", default="ft")
    parser.add_argument("--model", default="egraphsage")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", default="configs/eval.yaml")
    parser.add_argument("--checkpoints-root", default=None)
    parser.add_argument("--graphs-dir", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--lane", default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--eval-chunk-edges",
        type=int,
        default=0,
        help="fallback head chunk size (0 = model's edge_batch path when available; T30 memory bound)",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    paths = load_paths(lane=args.lane, project_root=None)
    config = load_config(args.config, project_root=paths.project_root, lane=args.lane)
    config_dict = config.to_dict()
    logger = setup_logging(
        "xdrift.run_eval",
        level=str((config.get("logging") or {}).get("level", "INFO")),
        log_path=paths.logs_root / "run_eval.log",
    )
    dataset = args.dataset or str(config.get("dataset", "NF-UNSW-NB15-v3"))
    from src.data.registry import load_dataset_spec

    spec = load_dataset_spec(dataset, project_root=paths.project_root)
    s5 = _section(config_dict, "s5")
    graphs_dir = Path(args.graphs_dir).expanduser().resolve() if args.graphs_dir else paths.processed_dataset(dataset)
    checkpoints_root = (
        Path(args.checkpoints_root).expanduser().resolve()
        if args.checkpoints_root
        else paths.run_dir(strategy=args.strategy, model=args.model, seed=args.seed)
    )
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else paths.tables_root
    graph_files = sorted(graphs_dir.glob("period_*.pt"), key=lambda path: int(path.stem.split("_")[1]))
    n_periods = len(graph_files)
    checkpoints = {index + 1: checkpoints_root / f"M_{index + 1}.pt" for index in range(n_periods)}
    present = {period: path for period, path in checkpoints.items() if path.is_file()}
    if args.dry_run:
        print(json.dumps({
            "dataset": dataset,
            "graphs_dir": str(graphs_dir),
            "checkpoints_root": str(checkpoints_root),
            "checkpoints_present": sorted(present),
            "checkpoints_missing": sorted(set(checkpoints) - set(present)),
            "out_dir": str(out_dir),
            "lane": paths.lane,
            "lane_source": paths.lane_source,
            "supported_set": _section(_section(config_dict, "s5"), "dual_macro_f1").get("supported_set"),
            "n_periods": n_periods,
        }, indent=2, sort_keys=True))
        return 0
    if not torch_available():
        print(f"[eval] ERROR: torch unavailable: {torch_unavailable_reason()}", file=sys.stderr)
        return 2
    import torch

    device = _resolve_device(args.device)
    class_order = list(spec.class_order or [])
    class_ids = list(range(len(class_order)))
    dual_cfg = _section(s5, "dual_macro_f1")
    supported_names = list(dual_cfg.get("supported_set") or [])
    supported_ids = supported_ids_from_names(supported_names, class_order) if supported_names else class_ids
    if not present:
        print(f"[eval] ERROR: no checkpoints under {checkpoints_root}", file=sys.stderr)
        return 2
    if paths.lane_source == "default":
        logger.warning("lane is 'default' (no --lane/XDRIFT_LANE); gate runs should launch via run_lane.ps1")
    manifest = RunManifest(
        "cl_evaluation",
        paths=paths,
        config=config,
        lane=paths.lane,
        seed=args.seed,
        dataset=dataset,
        out_path=paths.audit_root / f"eval_manifest_{args.strategy}_{args.model}_seed{args.seed}.json",
        extra={
            "strategy": args.strategy,
            "model": args.model,
            "checkpoints": {str(period): str(path) for period, path in checkpoints.items()},
            "checkpoints_present": sorted(present),
            "supported_set": supported_names,
            "lane_source": paths.lane_source,
            "eval_chunk_edges": int(args.eval_chunk_edges),
        },
    )
    manifest.start()
    try:
        # T30: memory-bounded evaluation. Stage the frozen graph tensors once per
        # period and project only the split-mask edges through the head (edge_batch
        # path) or chunked fallback; the full-edge head is never materialised.
        eval_started = time.monotonic()
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        graphs = {}
        graph_devices: dict[int, dict[str, Any]] = {}
        for period, path in enumerate(graph_files, start=1):
            graph = torch.load(path, weights_only=False)
            graphs[period] = graph
            test_mask = getattr(graph, "test_mask")
            graph_devices[period] = {
                "x": graph.x.to(device),
                "edge_index": graph.edge_index.to(device),
                "edge_attr": graph.edge_attr.to(device),
                "test_idx": torch.nonzero(test_mask, as_tuple=False).flatten().to(device),
            }
        per_class_tensor: dict[str, Any] = {}
        dual_scores: dict[str, dict[tuple[int, int], float]] = {
            "all_classes": {},
            "supported_set": {},
        }
        eval_mode: str | None = None
        checkpoints_meta: dict[str, Any] = {}
        for row_period, checkpoint_path in sorted(present.items()):
            payload = torch.load(checkpoint_path, weights_only=False)
            model_config = dict(payload.get("model_config") or {})
            model = None
            for column_period, graph in sorted(graphs.items()):
                if model is None:
                    model = build_model(
                        args.model,
                        node_dim=int(graph.x.shape[1]),
                        edge_dim=int(graph.edge_attr.shape[1]),
                        config={"model": model_config},
                    )
                    try:
                        model.load_state_dict(payload["state_dict"])
                    except RuntimeError as exc:  # pragma: no cover - checkpoint mismatch
                        raise RuntimeError(
                            f"checkpoint {checkpoint_path} does not match graph features: {exc}"
                        ) from exc
                    model.to(device).eval()
                    eval_mode = predict_mode(model, chunk_edges=int(args.eval_chunk_edges))
                staged = graph_devices[column_period]
                logits = predict_logits(
                    model,
                    staged["x"],
                    staged["edge_index"],
                    staged["edge_attr"],
                    staged["test_idx"],
                    device=device,
                    chunk_edges=int(args.eval_chunk_edges),
                )
                test_mask = getattr(graph, "test_mask")
                y_true = graph.y[test_mask].cpu().numpy()
                y_pred = logits.argmax(dim=-1).cpu().numpy()
                dual = dual_macro_f1(
                    y_true, y_pred, class_ids=class_ids, supported_ids=supported_ids
                )
                dual_scores["all_classes"][(row_period - 1, column_period - 1)] = dual["macro_f1_all_classes"]
                dual_scores["supported_set"][(row_period - 1, column_period - 1)] = dual["macro_f1_supported_set"]
                per_class_tensor.setdefault(str(row_period), {})[str(column_period)] = {
                    "dual_macro_f1": dual,
                    "per_class_f1": per_class_f1(y_true, y_pred, class_ids),
                }
            checkpoints_meta[str(row_period)] = {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "seed": payload.get("seed"),
                "period": payload.get("period"),
            }
        eval_seconds = round(time.monotonic() - eval_started, 3)
        if eval_mode is None:
            eval_mode = "edge_batch" if present else "not_run"
        if str(device).startswith("cuda") and torch.cuda.is_available():
            peak_memory = int(torch.cuda.max_memory_allocated())
            memory_device = "cuda"
            cuda_total = int(torch.cuda.get_device_properties(0).total_memory)
        else:
            try:
                import psutil

                peak_memory = int(psutil.Process().memory_info().rss)
            except Exception:  # noqa: BLE001
                peak_memory = None
            memory_device = "cpu"
            cuda_total = None
        evaluation = {
            "mode": eval_mode,
            "chunk_edges": int(args.eval_chunk_edges),
            "wall_clock_seconds": eval_seconds,
            "peak_memory_allocated_bytes": peak_memory,
            "memory_device": memory_device,
            "cuda_total_memory_bytes": cuda_total,
            "checkpoints_evaluated": sorted(present),
        }
        matrix_all = matrix_from_scores(dual_scores["all_classes"], n_periods)
        matrix_supported = matrix_from_scores(dual_scores["supported_set"], n_periods)
        last_row_complete = not any(
            value not in present for value in range(1, n_periods + 1)
        )
        metrics: dict[str, Any] = {
            "schema_version": 1,
            "generated_at_utc": utc_now(),
            "dataset": dataset,
            "strategy": args.strategy,
            "model": args.model,
            "seed": args.seed,
            "class_order": class_order,
            "supported_set": supported_names,
            "checkpoints": checkpoints_meta,
            "matrix": {
                "all_classes": matrix_all.tolist(),
                "supported_set": matrix_supported.tolist(),
            },
            "rows": {str(index + 1): "M_%d" % (index + 1) for index in range(n_periods)},
            "columns": {str(index + 1): "period_%d" % (index + 1) for index in range(n_periods)},
            "dual_macro_f1_per_cell": per_class_tensor,
            # T30: memory-bounded evaluation provenance (mode/peak/wall clock) and a
            # top-level completeness field for the matrix driver's resume checks.
            "checkpoints_evaluated": sorted(present),
            "evaluation": evaluation,
        }
        if last_row_complete:
            metrics["cl_metrics_all_classes"] = cl_metrics(matrix_all)
            metrics["cl_metrics_supported_set"] = cl_metrics(matrix_supported)
        else:
            metrics["cl_metrics_all_classes"] = None
            metrics["cl_metrics_supported_set"] = None
            metrics["note"] = "last checkpoint row incomplete; CL metrics not computed yet"
        out_dir.mkdir(parents=True, exist_ok=True)
        csv_path = out_dir / str(_section(s5, "outputs").get("metrics_csv", "cl_metrics.csv"))
        per_class_path = out_dir / str(_section(s5, "outputs").get("per_class_json", "cl_per_class_f1.json"))
        # D-047/D-044 provenance: the writer uses the analysis reference convention
        # (triangular best-so-far; 7-class supported set incl. Benign) and records it
        # in the artifact so the verifier can audit the definition directly.
        metrics["forgetting_convention"] = FORGETTING_CONVENTION
        metrics["supported_set_definition"] = "D-033/D-047: 6 attack classes + Benign (7)"
        metrics["supported_set_names"] = supported_names or list(DEFAULT_SUPPORTED_SET)
        matrix_path = out_dir / str(_section(s5, "outputs").get("matrix_json", "cl_matrix.json"))
        write_json(matrix_path, metrics)
        write_json(per_class_path, {"per_class_f1": per_class_tensor, "class_order": class_order})
        header = (
            "strategy,model,seed,metric,final_average_macro_f1,forgetting,bwt,note\n"
        )
        lines = []
        for metric_name, key in (("all_classes", "cl_metrics_all_classes"), ("supported_set", "cl_metrics_supported_set")):
            block = metrics[key]
            if block is None:
                lines.append(f"{args.strategy},{args.model},{args.seed},{metric_name},,,,incomplete_matrix\n")
            else:
                lines.append(
                    f"{args.strategy},{args.model},{args.seed},{metric_name},"
                    f"{block['final_average_macro_f1']:.6f},{block['forgetting']:.6f},{block['bwt']:.6f},\n"
                )
        import csv

        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(header)
            handle.write("".join(lines))
        manifest.add_artifact(csv_path, role="cl_metrics")
        manifest.add_artifact(per_class_path, role="per_class_f1")
        manifest.add_artifact(matrix_path, role="cl_matrix")
        for period, path in present.items():
            manifest.add_artifact(path, role="checkpoint")
        for period, graph_path in zip(range(1, len(graph_files) + 1), graph_files):
            manifest.add_artifact(graph_path, role="graph", hash_file=False)
        manifest.payload.setdefault("extra", {}).update({"evaluation": evaluation})
        manifest.finish("success")
        manifest.write()
        print(json.dumps({
            "dataset": dataset,
            "strategy": args.strategy,
            "model": args.model,
            "seed": args.seed,
            "checkpoints_evaluated": sorted(present),
            "evaluation": evaluation,
            "cl_metrics_all_classes": metrics["cl_metrics_all_classes"],
            "cl_metrics_supported_set": metrics["cl_metrics_supported_set"],
            "outputs": {"metrics_csv": str(csv_path), "per_class": str(per_class_path), "matrix": str(matrix_path)},
        }, indent=2, sort_keys=True))
        return 0
    except Exception as exc:  # noqa: BLE001
        manifest.finish("failed", error=f"{type(exc).__name__}: {exc}")
        manifest.write()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
