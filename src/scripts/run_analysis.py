#!/usr/bin/env python
"""S8-S12 analysis wiring CLI (authoritative src/analysis/ modules; D-044).

    python src/scripts/run_analysis.py --mode synthetic --lane legacy
    python src/scripts/run_analysis.py --mode real --lane rtx5060 --dry-run

``synthetic`` mode proves the wiring end-to-end on the analysis toolkit's
fixtures by calling the same ``analysis.tables.generate_all`` path the real run
uses; it writes under ``runs/`` by default and is explicitly banner-marked as NOT
a result. ``real`` mode assembles the S8-S12 frames from frozen artifacts and
refuses to write partial tables: missing artifacts are listed by ``--dry-run``.
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

from src.analysis_wiring.frames import AnalysisWiringError  # noqa: E402
from src.analysis_wiring.pipeline import (  # noqa: E402
    required_real_inputs,
    run_real_wiring,
    run_synthetic_wiring,
)
from src.utils.config import load_config  # noqa: E402
from src.utils.gates import gate_record  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402
from src.utils.manifest import RunManifest, utc_now, write_json  # noqa: E402
from src.utils.paths import gate_preflight, load_paths  # noqa: E402


def resolve_groups(
    groups_arg: str | None,
    strategies_arg: str | None,
    models_arg: str | None,
    seed: int,
) -> list[tuple[str, str, int]] | None:
    """Resolve ``--groups seed42-all`` / cartesian strategyÃƒâ€”model options (T49).

    Accepts either an explicit comma list ``strategy:model[:seed]`` or the
    ``seed42-all`` shorthand; ``--strategies ft,replay,ewc --models ...`` builds
    the cartesian product at the common ``--seed``.  Returns ``None`` for the
    legacy single-group path.
    """
    valid_strategies = {"ft", "replay", "ewc"}
    valid_models = {"egraphsage", "gcn"}
    if groups_arg is not None:
        if groups_arg.strip().lower() in {"seed42-all", "all", "seed42_all"}:
            selected_strategies = ["ft", "replay", "ewc"]
            selected_models = ["egraphsage", "gcn"]
            return [(s, m, int(seed)) for s in selected_strategies for m in selected_models]
        groups: list[tuple[str, str, int]] = []
        for entry in groups_arg.split(","):
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split(":")
            if len(parts) not in (2, 3):
                raise ValueError(f"invalid --groups entry {entry!r}; expected strategy:model[:seed]")
            strategy, model = parts[0], parts[1]
            group_seed = int(parts[2]) if len(parts) == 3 else int(seed)
            if strategy not in valid_strategies or model not in valid_models:
                raise ValueError(f"invalid --groups entry {entry!r}")
            if (strategy, model, group_seed) not in groups:
                groups.append((strategy, model, group_seed))
        if not groups:
            raise ValueError("--groups was given but resolved to an empty list")
        return groups
    if strategies_arg or models_arg:
        selected_strategies = [
            value.strip() for value in (strategies_arg or "ft,replay,ewc").split(",") if value.strip()
        ]
        selected_models = [
            value.strip() for value in (models_arg or "egraphsage,gcn").split(",") if value.strip()
        ]
        unknown_s = [value for value in selected_strategies if value not in valid_strategies]
        unknown_m = [value for value in selected_models if value not in valid_models]
        if unknown_s or unknown_m:
            raise ValueError(f"invalid strategies/models: {unknown_s + unknown_m}")
        return [(s, m, int(seed)) for s in selected_strategies for m in selected_models]
    return None


def discover_group_cl_matrices(
    cl_matrix_arg: str | None,
    groups: Sequence[tuple[str, str, int]] | None,
) -> dict[tuple[str, str, int], Path] | None:
    """Find per-group S5 matrices under ``<cl_matrix>/../matrix/<group>/cl_matrix.json``."""
    if not cl_matrix_arg or not groups:
        return None
    matrix_root = Path(cl_matrix_arg).parent / "matrix"
    paths: dict[tuple[str, str, int], Path] = {}
    for strategy, model, seed in groups:
        candidate = matrix_root / f"{strategy}_{model}_seed{seed}" / "cl_matrix.json"
        if not candidate.is_file():
            return None
        paths[(strategy, model, seed)] = candidate
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="run_analysis", description=__doc__.splitlines()[0])
    parser.add_argument("--mode", default="synthetic", choices=["synthetic", "real"])
    parser.add_argument("--config", default="configs/drift.yaml")
    parser.add_argument("--lane", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--strategy", default="ft")
    parser.add_argument("--model", default="egraphsage")
    parser.add_argument(
        "--groups",
        default=None,
        help="multi-group S8/S9 mode: 'seed42-all' or explicit strategy:model[:seed] list",
    )
    parser.add_argument("--strategies", default=None, help="multi-group cartesian strategies (default ft,replay,ewc)")
    parser.add_argument("--models", default=None, help="multi-group cartesian models (default egraphsage,gcn)")
    parser.add_argument("--explainers", default="ig,gnn_explainer")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--out-root", default=None)
    parser.add_argument("--anchors-dir", default=None)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--cl-matrix", default=None)
    parser.add_argument("--graphs-dir", default=None)
    parser.add_argument("--noise-cache-root", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    paths = load_paths(lane=args.lane)
    config = load_config(args.config, project_root=paths.project_root, lane=args.lane)
    logger = setup_logging(
        "xdrift.run_analysis",
        level=str((config.get("logging") or {}).get("level", "INFO")),
        log_path=paths.logs_root / "run_analysis.log",
    )
    out_root = (
        Path(args.out_root).expanduser().resolve()
        if args.out_root
        else paths.runs_root / "analysis_wiring" / args.mode
    )
    try:
        resolved_groups = resolve_groups(args.groups, args.strategies, args.models, args.seed)
    except ValueError as exc:
        parser.error(str(exc))
        return 2
    cl_matrix_paths = discover_group_cl_matrices(args.cl_matrix, resolved_groups)
    explainers = [name.strip() for name in args.explainers.split(",") if name.strip()]
    preflight = gate_preflight(paths)
    plan = {
        "mode": args.mode,
        "out_root": str(out_root),
        "config": str(config.path) if config.path else None,
        "config_sha256": config.sha256,
        "lane": paths.lane,
        "lane_source": paths.lane_source,
        "explainers": explainers,
        "k": args.k,
        "groups": [f"{s}:{m}:{seed}" for s, m, seed in resolved_groups] if resolved_groups else None,
        "gate_preflight": preflight,
    }
    if args.mode == "real":
        plan["missing_artifacts"] = required_real_inputs(
            out_root=out_root,
            anchors_dir=args.anchors_dir,
            cache_root=args.cache_root,
            cl_matrix_path=args.cl_matrix,
            graphs_dir=args.graphs_dir,
            noise_cache_root=args.noise_cache_root,
        )
    if args.dry_run:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    for warning in preflight["warnings"]:
        logger.warning("[gate] %s", warning)
    manifest = RunManifest(
        "drift_analysis",
        paths=paths,
        config=config,
        lane=paths.lane,
        seed=args.seed,
        dataset=str(config.get("dataset", "NF-UNSW-NB15-v3")),
        out_path=out_root / "analysis_wiring_manifest.json",
        extra={"mode": args.mode, "explainers": explainers, "k": args.k, "groups": plan.get("groups"), "gate_preflight": preflight},
    )
    manifest.start()
    try:
        if args.mode == "synthetic":
            summary = run_synthetic_wiring(out_root=out_root, seed=args.seed)
        else:
            summary = run_real_wiring(
                out_root=out_root,
                strategy=args.strategy,
                model=args.model,
                seed=args.seed,
                anchors_dir=args.anchors_dir,
                cache_root=args.cache_root,
                cl_matrix_path=args.cl_matrix,
                graphs_dir=args.graphs_dir,
                noise_cache_root=args.noise_cache_root,
                explainers=explainers,
                k=args.k,
                groups=resolved_groups,
                cl_matrix_paths=cl_matrix_paths,
            )
        summary["generated_at_utc"] = utc_now()
        summary["plan"] = plan
        summary_path = out_root / f"analysis_summary_{args.mode}.json"
        write_json(summary_path, summary)
        manifest.add_artifact(summary_path, role="analysis_summary")
        validation = summary.get("validation") or {}
        manifest.payload.setdefault("extra", {}).update(
            {"validation_overall": validation.get("overall"), "generated_at_utc": summary["generated_at_utc"]}
        )
        manifest.finish("success")
        manifest.write()
        printable = {key: value for key, value in summary.items() if key not in ("outputs",)}
        print(json.dumps(printable, indent=2, sort_keys=True, default=str))
        print(f"[analysis] tables root: {summary.get('tables_root')}")
        print(f"[analysis] manifest: {manifest.resolve_out_path()}")
        return 0
    except AnalysisWiringError as exc:
        manifest.finish("failed", error=str(exc))
        manifest.write()
        print(f"[analysis] ERROR: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:  # noqa: BLE001
        manifest.finish("failed", error=f"{type(exc).__name__}: {exc}")
        manifest.write()
        raise


if __name__ == "__main__":
    raise SystemExit(main())
