#!/usr/bin/env python
"""S1: build per-period graph artefacts from the frozen C1 state.

    python src/scripts/build_graphs.py --dataset NF-UNSW-NB15-v3 --config configs/train.yaml

Reads the frozen data state produced by ``src/scripts/run_data_audit.py``
(periods/splits/preprocessor), verifies that the derived split hash still equals
the frozen ``split_manifest.json`` hash (D-042: no frozen-state changes), and
writes ``datasets/processed/<dataset>/period_{1..5}.pt`` + ``graphs_manifest.json``
(with per-file SHA-256) plus a small copy under ``results/text/data_audit/``.

Paths are resolved through ``XDRIFT_*`` environment variables (remote-kit
contract): ``XDRIFT_DATA`` for processed graphs, ``XDRIFT_RESULTS`` for the small
manifest copy, ``XDRIFT_RUNS`` for the run manifest.
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

from src.data.clean import clean_frame, load_raw_frame  # noqa: E402
from src.data.graphs import build_graphs  # noqa: E402
from src.data.periods import DEFAULT_SESSION_MIN_GAP_SECONDS, assign_periods  # noqa: E402
from src.data.preprocess import Preprocessor  # noqa: E402
from src.data.registry import load_dataset_spec  # noqa: E402
from src.data.splits import build_splits, freeze_split_masks  # noqa: E402
from src.utils.config import Config, hash_config, load_config  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402
from src.utils.manifest import RunManifest, sha256_file, utc_now, write_json  # noqa: E402
from src.utils.paths import load_paths  # noqa: E402
from src.utils.seed import seed_everything  # noqa: E402


def _section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name) or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"config section {name!r} must be a mapping")
    return dict(value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build_graphs", description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--audit-config", default="configs/audit.yaml")
    parser.add_argument("--root", default=None)
    parser.add_argument("--lane", default=None)
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--results-dir", default=None)
    parser.add_argument("--preprocessor-state", default=None, help="default: <results>/data_audit/preprocessor_state.json")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    paths = load_paths(lane=args.lane, project_root=args.root)
    root = paths.project_root
    graph_config = load_config(args.config, project_root=root, lane=args.lane)
    audit_config = load_config(args.audit_config, project_root=root, lane=args.lane)
    logger = setup_logging(
        "xdrift.build_graphs",
        level=str((audit_config.get("logging") or {}).get("level", "INFO")),
        log_path=paths.logs_root / "build_graphs.log",
    )
    dataset = args.dataset or str(graph_config.get("dataset", "NF-UNSW-NB15-v3"))
    spec = load_dataset_spec(dataset, project_root=root)
    data_cfg = audit_config.to_dict()
    seed = int(data_cfg.get("seed", graph_config.get("seed", 42)))
    seed_everything(seed)
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else paths.processed_dataset(dataset)
    results_dir = Path(args.results_dir).expanduser().resolve() if args.results_dir else paths.audit_root
    results_dir.mkdir(parents=True, exist_ok=True)
    preprocessor_path = (
        Path(args.preprocessor_state).expanduser().resolve()
        if args.preprocessor_state
        else results_dir / "preprocessor_state.json"
    )
    split_manifest_path = results_dir / "split_manifest.json"
    graph_cfg = _section(graph_config.to_dict(), "graph")

    if args.dry_run:
        print(json.dumps({
            "dataset": dataset,
            "out_dir": str(out_dir),
            "preprocessor_state": str(preprocessor_path),
            "split_manifest": str(split_manifest_path),
            "periods": _section(data_cfg, "periods"),
            "lane": paths.lane,

            "lane_source": paths.lane_source,

            "node_feature_dim": graph_cfg.get("node_feature_dim", 8),
            "overwrite": args.overwrite,
        }, indent=2, sort_keys=True))
        return 0

    raw_dir = paths.raw_dataset(dataset)
    csv_files = spec.csv_files(raw_dir) if raw_dir.is_dir() else []
    if not csv_files:
        print(f"[graphs] ERROR: no raw CSV for {dataset} under {raw_dir}", file=sys.stderr)
        return 2
    csv_path = csv_files[0]

    manifest = RunManifest(
        "graph_build",
        paths=paths,
        config=graph_config,
        lane=paths.lane,
        seed=seed,
        dataset=dataset,
        out_path=results_dir / "graph_build_manifest.json",
        extra={"audit_config_sha256": hash_config(data_cfg), "out_dir": str(out_dir)},
    )
    manifest.start()
    try:
        logger.info("loading %s", csv_path)
        raw = load_raw_frame(spec, csv_path)
        clean = clean_frame(
            raw,
            spec,
            sentinel_policy=_section(data_cfg, "sentinels"),
            near_duplicate=_section(data_cfg, "near_duplicate"),
            dedupe=bool(_section(data_cfg, "dedupe").get("exact", True)),
            label_conflict_policy=str(_section(data_cfg, "dedupe").get("label_conflict_policy", "report_only")),
        )
        frame = clean.frame
        del raw
        periods_cfg = _section(data_cfg, "periods")
        periods = assign_periods(
            frame,
            n_periods=int(periods_cfg["count"]),
            strategy=str(periods_cfg["strategy"]),
            min_flows_per_period=int(periods_cfg.get("min_flows_per_period", 1)),
            min_session_gap_seconds=float(periods_cfg.get("min_session_gap_seconds", DEFAULT_SESSION_MIN_GAP_SECONDS)),
        )
        frame = periods.frame
        splits_cfg = _section(data_cfg, "splits")
        host_pair_cfg = dict(splits_cfg.get("host_pair_disjoint") or {})
        splits = build_splits(
            frame,
            fractions=dict(splits_cfg.get("fractions", {"train": 0.7, "val": 0.1, "test": 0.2})),
            seed=seed,
            mode="grouped",
            min_groups_per_split=int(host_pair_cfg.get("min_groups_per_split", 10)),
            min_share=float(host_pair_cfg.get("min_share", 0.05)),
        )
        frozen = freeze_split_masks(splits, frame, dataset=dataset)
        expected_frozen_hash = None
        if split_manifest_path.is_file():
            frozen_manifest = json.loads(split_manifest_path.read_text(encoding="utf-8"))
            expected_frozen_hash = frozen_manifest.get("frozen_hash") or frozen_manifest.get("grouped", {}).get("frozen_hash")
        if expected_frozen_hash and expected_frozen_hash != frozen["frozen_hash"]:
            raise RuntimeError(
                "derived split hash != frozen split_manifest.json hash; refusing to build graphs "
                f"({frozen['frozen_hash']} vs {expected_frozen_hash})"
            )
        # Frozen preprocessing state (fitted on period-1 train only) must be reused.
        if preprocessor_path.is_file():
            preprocessor = Preprocessor.load(spec, preprocessor_path)
            if preprocessor.feature_names != list(preprocessor.state.get("feature_names", preprocessor.feature_names)):
                raise RuntimeError("preprocessor state feature names are inconsistent")
            logger.info("reusing frozen preprocessor state %s", preprocessor_path)
        else:
            logger.warning("no frozen preprocessor state at %s; fitting on period-1 train (should not happen post-G1)", preprocessor_path)
            train_positions = splits.masks[min(splits.masks)]["train"]
            preprocessor = Preprocessor(spec, config=_section(data_cfg, "preprocess")).fit(frame.iloc[train_positions])
        graph_manifest = build_graphs(
            frame=frame,
            splits=splits,
            preprocessor=preprocessor,
            spec=spec,
            out_dir=out_dir,
            node_feature_dim=int(graph_cfg.get("node_feature_dim", 8)),
            overwrite=bool(args.overwrite or graph_cfg.get("overwrite", False)),
            expected_frozen_hash=expected_frozen_hash,
            frozen_hash=frozen["frozen_hash"],
            raw_csv=csv_path,
            preprocessor_state_path=preprocessor_path,
            config_pins={
                "graph_config_sha256": graph_config.sha256,
                "audit_config_sha256": hash_config(data_cfg),
                "seed": seed,
                "period_strategy": periods.stats["strategy"],
                "lane": paths.lane,

                "lane_source": paths.lane_source,

                "node_feature_dim": int(graph_cfg.get("node_feature_dim", 8)),
            },
            dataset=dataset,
        )
        write_json(results_dir / "graph_manifest.json", graph_manifest)
        manifest.set_dataset(dataset, csv_path.parent, patterns=(csv_path.name,))
        manifest.add_artifact(out_dir / "graphs_manifest.json", role="graph_manifest")
        manifest.add_artifact(results_dir / "graph_manifest.json", role="graph_manifest_copy")
        for record in graph_manifest["periods"].values():
            manifest.add_artifact(record["file"], role="period_graph", hash_file=False)
        manifest.finish("success")
        manifest.write()
        print(json.dumps({
            "dataset": dataset,
            "artifact_fingerprint_sha256": graph_manifest["artifact_fingerprint_sha256"],
            "frozen_split_hash": graph_manifest["frozen_split_hash"],
            "split_hash_matches_frozen": graph_manifest["split_hash_matches_frozen"],
            "periods": {
                key: {"edges": block["n_edges"], "nodes": block["n_nodes"], "sha256": block["sha256"]}
                for key, block in graph_manifest["periods"].items()
            },
            "out_dir": str(out_dir),
        }, indent=2, sort_keys=True))
        print(f"[graphs] manifest: {out_dir / 'graphs_manifest.json'}")
        print(f"[graphs] run manifest: {results_dir / 'graph_build_manifest.json'}")
        return 0
    except Exception as exc:  # noqa: BLE001
        manifest.finish("failed", error=f"{type(exc).__name__}: {exc}")
        manifest.write()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
