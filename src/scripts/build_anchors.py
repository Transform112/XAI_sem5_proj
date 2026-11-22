#!/usr/bin/env python
"""S6 anchor builder CLI (test-mask, correct + confident, stratified, D-033 rare stratum).

    python src/scripts/build_anchors.py --dataset NF-UNSW-NB15-v3 --strategy ft --model egraphsage --seed 42

For every period j with a pinned checkpoint ``M_j`` this selects anchors from the
frozen test mask, stores sampled 2-hop subgraphs + host ids + anchor ids in
``results/text/anchors/period_<j>.pt``, and writes the retained-fraction table across
later checkpoints. No training, no frozen-state changes.
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

from src.anchors.build import (  # noqa: E402
    retained_fraction_table,
    saved_anchor_summary,
    select_anchors,
)
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


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build_anchors", description=__doc__.splitlines()[0])
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
    parser.add_argument("--period", type=int, default=None, help="build anchors for one period only (batch driver)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    paths = load_paths(lane=args.lane)
    config = load_config(args.config, project_root=paths.project_root, lane=args.lane)
    config_dict = config.to_dict()
    logger = setup_logging(
        "xdrift.build_anchors",
        level=str((config.get("logging") or {}).get("level", "INFO")),
        log_path=paths.logs_root / "build_anchors.log",
    )
    dataset = args.dataset or str(config.get("dataset", "NF-UNSW-NB15-v3"))
    from src.data.registry import load_dataset_spec

    spec = load_dataset_spec(dataset, project_root=paths.project_root)
    graphs_dir = Path(args.graphs_dir).expanduser().resolve() if args.graphs_dir else paths.processed_dataset(dataset)
    checkpoints_root = (
        Path(args.checkpoints_root).expanduser().resolve()
        if args.checkpoints_root
        else paths.run_dir(strategy=args.strategy, model=args.model, seed=args.seed)
    )
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else paths.results_root / "anchors"
    graph_files = sorted(graphs_dir.glob("period_*.pt"), key=lambda path: int(path.stem.split("_")[1]))
    if args.period is not None:
        graph_files = [path for path in graph_files if int(path.stem.split("_")[1]) == int(args.period)]
        if not graph_files:
            graph_files = [graphs_dir / f"period_{int(args.period)}.pt"]
        checkpoints = {int(args.period): checkpoints_root / f"M_{int(args.period)}.pt"}
    else:
        checkpoints = {period: checkpoints_root / f"M_{period}.pt" for period in range(1, len(graph_files) + 1)}
    graph_by_period = {int(path.stem.split("_")[1]): path for path in graph_files}
    present = {period: path for period, path in checkpoints.items() if path.is_file()}
    s6 = _section(config_dict, "s6")
    if args.dry_run:
        print(json.dumps({
            "dataset": dataset,
            "graphs_dir": str(graphs_dir),
            "checkpoints_root": str(checkpoints_root),
            "checkpoints_present": sorted(present),
            "checkpoints_missing": sorted(set(checkpoints) - set(present)),
            "out_dir": str(out_dir),
            "period": args.period,
            "lane": paths.lane,
            "lane_source": paths.lane_source,
            "anchor_config": _section(s6, "anchors"),
        }, indent=2, sort_keys=True))
        return 0
    if not torch_available():
        print(f"[anchors] ERROR: torch unavailable: {torch_unavailable_reason()}", file=sys.stderr)
        return 2
    import torch

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if paths.lane_source == "default":
        logger.warning("lane is 'default' (no --lane/XDRIFT_LANE); gate runs should launch via run_lane.ps1")
    manifest = RunManifest(
        "anchor_build",
        paths=paths,
        config=config,
        lane=paths.lane,
        seed=args.seed,
        dataset=dataset,
        out_path=paths.audit_root / f"anchors_manifest_{args.strategy}_{args.model}_seed{args.seed}.json",
        extra={
            "strategy": args.strategy,
            "model": args.model,
            "checkpoints": {str(period): str(path) for period, path in checkpoints.items()},
            "checkpoints_present": sorted(present),
            "lane_source": paths.lane_source,
        },
    )
    manifest.start()
    try:
        if not present:
            raise RuntimeError(f"no checkpoints under {checkpoints_root}; cannot build anchors")
        out_dir.mkdir(parents=True, exist_ok=True)
        anchor_config = _section(s6, "anchors")
        graphs: dict[int, Any] = {}
        models: dict[int, Any] = {}
        anchor_sets: dict[int, dict[str, Any]] = {}
        retention_pairs: dict[int, tuple[Any, Any]] = {}
        for period, checkpoint_path in sorted(present.items()):
            graph = torch.load(graph_by_period[period], weights_only=False)
            payload = torch.load(checkpoint_path, weights_only=False)
            model_config = dict(payload.get("model_config") or {})
            model = build_model(
                args.model,
                node_dim=int(graph.x.shape[1]),
                edge_dim=int(graph.edge_attr.shape[1]),
                config={"model": model_config},
            )
            model.load_state_dict(payload["state_dict"])
            # Processed graphs store a 0-based period field; the S6 contract and
            # validators use 1-based period ids, so pin the artifact period here.
            graph.period = int(period)
            anchor_set = select_anchors(
                model,
                graph,
                config=anchor_config,
                device=device,
                seed=int(anchor_config.get("sampling_seed", 42)),
            )
            anchor_sets[period - 1] = anchor_set
            graphs[period - 1] = graph
            models[period - 1] = model
            anchor_file = out_dir / f"period_{period}.pt"
            torch.save(anchor_set, anchor_file)
            logger.info(
                "period %s anchors: %s selected / %s eligible (rare classes %s)",
                period,
                anchor_set["n_selected"],
                anchor_set["n_eligible"],
                anchor_set["rare_classes"],
            )
        # Retained-fraction table: for each anchor period, evaluate later checkpoints.
        checkpoints_for_retention: dict[int, tuple[Any, Any]] = {}
        for period in models:
            checkpoints_for_retention[period] = (models[period], graphs[period])
        retained_rows = retained_fraction_table(
            anchor_sets, checkpoints_for_retention, graphs=graphs, device=device
        )
        retained_json = out_dir / "retained_fraction.json"
        write_json(retained_json, {
            "schema_version": 1,
            "generated_at_utc": utc_now(),
            "dataset": dataset,
            "strategy": args.strategy,
            "model": args.model,
            "seed": args.seed,
            "rows": retained_rows,
        })
        retained_csv = out_dir / "retained_fraction.csv"
        with retained_csv.open("w", encoding="utf-8", newline="") as handle:
            handle.write("anchor_period,later_checkpoint,retained,total,retained_fraction\n")
            for row in retained_rows:
                handle.write(
                    f"{row['anchor_period']},{row['later_checkpoint']},{row['retained']},"
                    f"{row['total']},{row['retained_fraction']:.6f}\n"
                )
        # D-048 canonical per-class retention artifact (analysis contract schema:
        # strategy,model,class,checkpoint_t,retained_fraction). Cumulative retention
        # guarantees the required non-increasing property per class across later
        # checkpoints (an anchor counts as retained only if every checkpoint up to t
        # still classifies it correctly).
        class_names = list(getattr(spec, "class_order", []))
        canonical_rows: list[dict[str, Any]] = []
        status_by_pair = {
            (int(row["anchor_period"]), int(row["later_checkpoint"])): list(row.get("per_anchor") or [])
            for row in retained_rows
        }
        # Canonical table cohort: anchors selected at the first checkpoint, tracked
        # cumulatively across later checkpoints (single row per class/checkpoint_t and
        # non-increasing retention; schema has no anchor-period column so pooling
        # would duplicate keys).
        if anchor_sets:
            first_period = min(anchor_sets)
            first_anchors = list(anchor_sets[first_period].get("anchors", []))
            if first_anchors:
                labels = [int(anchor["label"]) for anchor in first_anchors]
                later_checkpoints = sorted(
                    int(row["later_checkpoint"])
                    for row in retained_rows
                    if int(row["anchor_period"]) == int(first_period) + 1
                )
                cumulative = [True] * len(first_anchors)
                for checkpoint_t in later_checkpoints:
                    statuses = status_by_pair.get((int(first_period) + 1, checkpoint_t), [])
                    if len(statuses) != len(first_anchors):
                        continue
                    cumulative = [bool(a and b) for a, b in zip(cumulative, statuses)]
                    by_class: dict[int, list[bool]] = {}
                    for label, ok in zip(labels, cumulative):
                        by_class.setdefault(label, []).append(bool(ok))
                    for label, values in sorted(by_class.items()):
                        canonical_rows.append(
                            {
                                "strategy": args.strategy,
                                "model": args.model,
                                "class": class_names[label] if 0 <= label < len(class_names) else str(label),
                                "checkpoint_t": int(checkpoint_t),
                                "retained_fraction": float(sum(values) / max(len(values), 1)),
                            }
                        )
        canonical_retention = paths.tables_root / "anchor_retention.csv"
        if args.period is None:
            canonical_retention.parent.mkdir(parents=True, exist_ok=True)
            with canonical_retention.open("w", encoding="utf-8", newline="") as handle:
                handle.write("strategy,model,class,checkpoint_t,retained_fraction\n")
                for row in canonical_rows:
                    handle.write(
                        f"{row['strategy']},{row['model']},{row['class']},"
                        f"{row['checkpoint_t']},{row['retained_fraction']:.6f}\n"
                    )
        anchor_summaries = {str(period + 1): saved_anchor_summary(anchor_set) for period, anchor_set in anchor_sets.items()}
        anchors_manifest = {
            "schema_version": 1,
            "kind": "s6_anchor_sets",
            "dataset": dataset,
            "generated_at_utc": utc_now(),
            "strategy": args.strategy,
            "model": args.model,
            "seed": args.seed,
            "anchor_config": anchor_config,
            "rare_min_eligible": int(anchor_config.get("rare_min_eligible", 10)),
            "rare_stratum": str(anchor_config.get("rare_stratum", "rare")),
            "anchor_files": {},
            "periods": {},
            "per_period": anchor_summaries,
            "retained_fraction": {"path": str(retained_json), "csv": str(retained_csv), "rows": retained_rows},
            "checkpoints": {},
        }
        for period, checkpoint_path in sorted(present.items()):
            anchors_path = out_dir / f"period_{period}.pt"
            anchors_manifest["anchor_files"][str(period)] = {
                "path": str(anchors_path),
                "bytes": anchors_path.stat().st_size,
                "sha256": sha256_file(anchors_path),
            }
            anchors_manifest["checkpoints"][str(period)] = {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
            }
            anchors_manifest["periods"][str(period)] = dict(anchors_manifest["anchor_files"][str(period)])
            manifest.add_artifact(anchors_path, role="anchor_set")
        manifest_name = (
            f"anchors_manifest_{args.strategy}_{args.model}_seed{args.seed}_period{int(args.period)}.json"
            if args.period is not None
            else f"anchors_manifest_{args.strategy}_{args.model}_seed{args.seed}.json"
        )
        out_manifest = (
            out_dir / f"anchors_manifest_period{int(args.period)}.json"
            if args.period is not None
            else out_dir / "anchors_manifest.json"
        )
        write_json(out_manifest, anchors_manifest)
        write_json(paths.audit_root / manifest_name, anchors_manifest)
        manifest.add_artifact(out_manifest, role="anchors_manifest")
        manifest.add_artifact(retained_json, role="retained_fraction")
        manifest.add_artifact(retained_csv, role="retained_fraction")
        # D-048 canonical evidence path (results/text/tables/anchor_retention.csv).
        if args.period is None:
            anchors_manifest["canonical_anchor_retention"] = str(canonical_retention)
        write_json(out_manifest, anchors_manifest)
        manifest.add_artifact(canonical_retention, role="anchor_retention")
        manifest.finish("success")
        manifest.write()
        print(json.dumps({
            "dataset": dataset,
            "strategy": args.strategy,
            "model": args.model,
            "seed": args.seed,
            "per_period": anchor_summaries,
            "retained_fraction_rows": retained_rows,
            "out_dir": str(out_dir),
        }, indent=2, sort_keys=True, default=str))
        return 0
    except Exception as exc:  # noqa: BLE001
        manifest.finish("failed", error=f"{type(exc).__name__}: {exc}")
        manifest.write()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
