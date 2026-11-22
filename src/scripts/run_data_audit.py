#!/usr/bin/env python
"""One-command S-D1 data audit (T4 acceptance command).

    python src/scripts/run_data_audit.py --dataset NF-UNSW-NB15-v3 --config configs/audit.yaml

Produces (defaults under results/text/data_audit/):
    scorecard.json / scorecard.md
    feature_manifest.csv
    split_manifest.json          (+ frozen npz masks under datasets/splits/<dataset>/)
    structure.json               (D-020/D-023 per-period structure facts)
    quality_audit.json
    preprocessor_state.json      (period-1-train-only fitted state)
    manifest.json                (reproducibility record)
    graph_lift_pilot.{json,md}   (only with --with-pilot; decisive G1c item)

Exit code: 0 when the scorecard is ``pass`` or ``flag``; 1 when it is ``fail``
(a fail blocks the G1 gate). ``flag`` requires an explanation and is resolved at
G1c; the scorecard always records it.

The raw data is never modified. Thresholds come from the config file and are
never hard-coded (missing threshold keys abort with an error).
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

from src.data.audit import run_audit  # noqa: E402
from src.data.registry import load_dataset_spec  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402
from src.utils.paths import load_paths  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="run_data_audit", description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default=None, help="dataset name (default: config dataset)")
    parser.add_argument("--config", default="configs/audit.yaml")
    parser.add_argument("--root", default=None)
    parser.add_argument("--lane", default=None)
    parser.add_argument("--out-dir", default=None, help="artifact directory (default: results/text/data_audit)")
    parser.add_argument(
        "--splits-dir", default=None, help="frozen mask directory (default: datasets/splits/<dataset>)"
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--with-pilot",
        action="store_true",
        help="also run the graph-lift pilot (T12/D-024 decisive G1c item)",
    )
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(args.root).expanduser().resolve() if args.root else None
    paths = load_paths(lane=args.lane, project_root=root)
    root = paths.project_root
    config = load_config(args.config, project_root=root, lane=args.lane)
    config_dict = config.to_dict()
    logger = setup_logging(
        "xdrift.run_data_audit",
        level=str((config.get("logging") or {}).get("level", "INFO")),
        log_path=paths.logs_root / "run_data_audit.log",
    )

    dataset = args.dataset or str(config.get("dataset", "NF-UNSW-NB15-v3"))
    spec = load_dataset_spec(dataset, project_root=root)
    raw_dir = paths.raw_dataset(dataset)
    csv_files = spec.csv_files(raw_dir) if raw_dir.is_dir() else []
    if not csv_files:
        print(
            f"[audit] ERROR: no CSV for {dataset} under {raw_dir}. "
            "Run src/scripts/download_data.ps1 first (and src/scripts/verify_datasets.py for identity).",
            file=sys.stderr,
        )
        return 2
    csv_path = csv_files[0]
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else paths.audit_root
    period_strategy = str((config_dict.get("periods") or {}).get("strategy", "equal_counts"))
    default_splits_dir = paths.dataset_splits(dataset)
    if period_strategy != "equal_counts":
        # Stream-A masks (equal_counts) stay at the dataset root as an
        # ablation/fallback; named streams get their own frozen-mask directory.
        default_splits_dir = default_splits_dir / period_strategy
    splits_dir = (
        Path(args.splits_dir).expanduser().resolve()
        if args.splits_dir
        else default_splits_dir
    )
    thresholds = config_dict.get("thresholds") or {}

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dataset": dataset,
                    "csv": str(csv_path),
                    "out_dir": str(out_dir),
                    "splits_dir": str(splits_dir),
                    "periods": (config_dict.get("periods") or {}).get("count"),
                    "with_pilot": bool(args.with_pilot),
                    "thresholds": thresholds,
                    "lane": paths.lane,
                    "lane_source": paths.lane_source,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    logger.info(
        "data audit: dataset=%s csv=%s out=%s splits=%s", dataset, csv_path, out_dir, splits_dir
    )
    from src.utils.gates import GateRequirementError

    try:
        scorecard = run_audit(
            spec=spec,
            csv_path=csv_path,
            config=config,
            out_dir=out_dir,
            splits_dir=splits_dir,
            dataset=dataset,
            seed=args.seed if args.seed is not None else int(config_dict.get("seed", 42)),
            with_pilot=bool(args.with_pilot),
            paths=paths,
        )
    except GateRequirementError as exc:
        print(f"[audit] ERROR: {exc}", file=sys.stderr)
        return 2
    status = str(scorecard["status"])
    counts = {
        state: sum(1 for check in scorecard["checks"] if check["status"] == state)
        for state in ("pass", "flag", "fail")
    }
    print("\n[audit] dataset :", dataset)
    print(f"[audit] verdict : {status.upper()}  (checks: {counts['pass']} pass / "
          f"{counts['flag']} flag / {counts['fail']} fail)")
    for check in scorecard["checks"]:
        if check["status"] != "pass":
            print(f"  - {check['status'].upper():4s} {check['name']}: {check.get('detail', '')[:160]}")
    artifacts = scorecard.get("artifacts", {})
    for name, path in artifacts.items():
        print(f"[audit] {name}: {path}")
    print(f"[audit] scorecard: {out_dir / 'scorecard.json'} , {out_dir / 'scorecard.md'}")
    print(f"[audit] feature manifest: {out_dir / 'feature_manifest.csv'}")
    print(f"[audit] split manifest: {out_dir / 'split_manifest.json'} (masks: {splits_dir})")
    print(f"[audit] structure: {out_dir / 'structure.json'}")
    if scorecard.get("graph_lift"):
        print(f"[audit] graph-lift: {out_dir / 'graph_lift_pilot.json'}")
    return 0 if status != "fail" else 1


if __name__ == "__main__":
    raise SystemExit(main())
