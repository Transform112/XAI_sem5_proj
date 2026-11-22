#!/usr/bin/env python
"""Run the graph-lift pilot (T12 / D-024 decisive G1c item).

Runs the three-way comparison on a stratified sample of period-1 edges:
MLP-on-edge-features vs E-GraphSAGE vs E-GraphSAGE with randomised structure,
over the seeds configured in ``configs/audit.yaml`` (``graph_lift.seeds``).
Writes ``results/text/data_audit/graph_lift_pilot.{json,md}`` plus a run manifest.

Usage:
    python src/scripts/run_graph_lift_pilot.py --dataset NF-UNSW-NB15-v3 --config configs/audit.yaml
    python src/scripts/run_graph_lift_pilot.py --dataset NF-UNSW-NB15-v3 --sample-edges 50000 --epochs 20
    python src/scripts/run_graph_lift_pilot.py --dataset NF-UNSW-NB15-v3 --dry-run

This is a pilot, not a continual-learning experiment: no strategy is trained and
no frozen split is modified. It exists to decide whether graph structure carries
signal on the primary dataset before the model stage is built.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

_LAUNCHER_ROOT = Path(__file__).resolve().parents[2]
if str(_LAUNCHER_ROOT) not in sys.path:
    sys.path.insert(0, str(_LAUNCHER_ROOT))

from src.data.clean import clean_frame, load_raw_frame  # noqa: E402
from src.data.periods import assign_periods  # noqa: E402
from src.data.preprocess import Preprocessor  # noqa: E402
from src.data.registry import load_dataset_spec  # noqa: E402
from src.data.splits import build_splits, freeze_split_masks  # noqa: E402
from src.pilot.graph_lift import resolve_config, run_graph_lift_pilot  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402
from src.utils.manifest import RunManifest  # noqa: E402
from src.utils.paths import load_paths  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="run_graph_lift_pilot", description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=None, help="dataset name (default: config dataset)")
    parser.add_argument("--config", default="configs/audit.yaml")
    parser.add_argument("--root", default=None)
    parser.add_argument("--lane", default=None)
    parser.add_argument("--out-dir", default=None, help="artifact directory (default: results/text/data_audit)")
    parser.add_argument("--sample-edges", type=int, default=None)
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", default=None, help="comma-separated seeds (default: config)")
    parser.add_argument("--run-label", default=None, help="free-text label recorded in the artifact (e.g. D-037 R1b)")
    parser.add_argument("--bootstrap", type=int, default=None, help="bootstrap resamples")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name) or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"config section {name!r} must be a mapping")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve() if args.root else None
    paths = load_paths(lane=args.lane, project_root=root)
    root = paths.project_root
    config = load_config(args.config, project_root=root, lane=args.lane)
    config_dict = config.to_dict()
    logger = setup_logging(
        "xdrift.pilot_runner",
        level=str((config.get("logging") or {}).get("level", "INFO")),
    )
    dataset_name = args.dataset or str(config.get("dataset", "NF-UNSW-NB15-v3"))
    spec = load_dataset_spec(dataset_name, project_root=root)
    seed = int(config.get("seed", 42))
    seed_everything(seed)

    graph_lift_cfg = _section(config_dict, "graph_lift")
    if args.sample_edges is not None:
        graph_lift_cfg["sample_edges"] = args.sample_edges
    if args.sample_frac is not None:
        graph_lift_cfg["sample_frac"] = args.sample_frac
    if args.epochs is not None:
        graph_lift_cfg["epochs"] = args.epochs
    if args.seeds:
        graph_lift_cfg["seeds"] = [int(value) for value in args.seeds.split(",") if value.strip()]
    if args.bootstrap is not None:
        graph_lift_cfg["bootstrap_samples"] = args.bootstrap
    resolved = resolve_config(graph_lift_cfg)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else paths.audit_root
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        plan = {
            "dataset": dataset_name,
            "period": resolved["period"],
            "sample_edges": resolved["sample_edges"],
            "seeds": resolved["seeds"],
            "epochs": resolved["epochs"],
            "out_dir": str(out_dir),
            "thresholds": {key: config_dict["thresholds"][key] for key in (
                "graph_lift_macro_f1_min",
                "graph_lift_vs_randomised_min",
            )},
        }
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0

    csv_files = spec.csv_files(paths.raw_dataset(dataset_name))
    if not csv_files:
        print(f"[pilot] ERROR: no CSV for {dataset_name} under {paths.raw_dataset(dataset_name)}", file=sys.stderr)
        return 2
    csv_path = csv_files[0]

    manifest = RunManifest(
        "graph_lift_pilot",
        paths=paths,
        config=config,
        lane=paths.lane,
        seed=seed,
        dataset=dataset_name,
        out_path=out_dir / "graph_lift_pilot_manifest.json",
        extra={"graph_lift_config": resolved, "csv": str(csv_path)},
    )
    manifest.start()
    try:
        logger.info("loading %s", csv_path)
        raw = load_raw_frame(spec, csv_path)
        clean = clean_frame(
            raw,
            spec,
            sentinel_policy=_section(config_dict, "sentinels"),
            near_duplicate=_section(config_dict, "near_duplicate"),
            dedupe=bool(_section(config_dict, "dedupe").get("exact", True)),
        )
        frame = clean.frame
        del raw
        periods_cfg = _section(config_dict, "periods")
        periods = assign_periods(
            frame,
            n_periods=int(periods_cfg["count"]),
            strategy=str(periods_cfg["strategy"]),
            min_flows_per_period=int(periods_cfg.get("min_flows_per_period", 1)),
        )
        frame = periods.frame
        splits_cfg = _section(config_dict, "splits")
        splits = build_splits(
            frame,
            fractions=dict(splits_cfg["fractions"]),
            seed=seed,
            mode="grouped",
            min_groups_per_split=int(
                (_section(config_dict, "splits").get("host_pair_disjoint") or {}).get("min_groups_per_split", 10)
            ),
            min_share=float(
                (_section(config_dict, "splits").get("host_pair_disjoint") or {}).get("min_share", 0.05)
            ),
        )
        train_period = min(splits.masks)
        train_positions = splits.masks[train_period]["train"]
        preprocessor = Preprocessor(spec, config=_section(config_dict, "preprocess")).fit(
            frame.iloc[train_positions]
        )
        report = run_graph_lift_pilot(
            frame,
            splits,
            preprocessor,
            spec,
            config=graph_lift_cfg,
            seed=seed,
            dataset=dataset_name,
            out_dir=out_dir,
            thresholds=_section(config_dict, "thresholds"),
            period_strategy=str(periods_cfg["strategy"]),
            config_sha256=config.sha256,
            run_label=args.run_label,
        )
        manifest.set_dataset(dataset_name, csv_path.parent, patterns=(csv_path.name,))
        for name in ("graph_lift_pilot.json", "graph_lift_pilot.md"):
            artifact = out_dir / name
            if artifact.is_file():
                manifest.add_artifact(artifact, role="graph_lift_pilot")
        manifest.finish(
            "success" if report.get("status") == "success" else "failed",
            extra={
                "run_label": args.run_label,
                "seeds": report.get("seeds"),
                "verdict": report.get("verdict"),
                "subset_summaries": report.get("subset_summaries"),
                "mean_over_seeds": report.get("mean_over_seeds"),
                "sample": report.get("sample"),
                "pinning": report.get("pinning"),
            },
        )
        manifest.write()
        verdict = report.get("verdict", {})
        print(
            "\n[pilot] mean over seeds: "
            + ", ".join(
                f"{name}={value:.4f}" for name, value in report.get("mean_over_seeds", {}).get("models", {}).items()
            )
        )
        print(
            f"[pilot] Delta(E-GS-MLP)={report.get('mean_over_seeds', {}).get('delta_macro_f1_egsage_minus_mlp', float('nan')):+.4f} "
            f"CI95={verdict.get('graph_lift_macro_f1', {}).get('ci95')}"
        )
        print(
            f"[pilot] Delta(E-GS-random)={report.get('mean_over_seeds', {}).get('delta_macro_f1_egsage_minus_randomised', float('nan')):+.4f} "
            f"CI95={verdict.get('graph_lift_vs_randomised', {}).get('ci95')}"
        )
        print(f"[pilot] D-024 overall verdict: {verdict.get('overall', 'n/a')}")
        print(f"[pilot] artifacts: {out_dir / 'graph_lift_pilot.json'} , {out_dir / 'graph_lift_pilot.md'}")
        return 0 if report.get("status") == "success" else 1
    except Exception as exc:  # noqa: BLE001 - manifest records the failure
        manifest.finish("failed", error=f"{type(exc).__name__}: {exc}")
        manifest.write()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
