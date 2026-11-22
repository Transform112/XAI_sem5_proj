"""One-command synthetic demo of the analysis toolkit (never writes real results).

What it does
------------
1. builds the deterministic synthetic bundle (:mod:`analysis.synthetic`) —
   protocol-shaped S5 records, S8 explanation cache/drift/noise data, S10
   neighbourhood change, S12 EAR rows;
2. runs the full S5 → S8 → S9 → S10 → S11 → S12 → S13 pipeline through the real
   analysis modules;
3. writes every registered table/figure to ``src/analysis/examples/results/`` with
   the exact skeleton filenames, plus a schema-valid ``paper_numbers.json``
   under ``src/analysis/examples/``;
4. writes ``src/analysis/examples/manifest.json`` recording input seed, output
   sha256s and a clear "SYNTHETIC — not a result" banner.

Why it exists: it proves the artifact contract (paths, columns, statistics)
end-to-end while the real S1–S7 pipeline is still being built, and it gives the
verifier a runnable reference for recomputing the analysis. It is **not** a
result: synthetic numbers must never be quoted in the paper (status defaults to
``pending`` in the numbers file).

Usage
-----
    .\\.venv\\Scripts\\python.exe -m analysis.run_synthetic_demo
    .\\.venv\\Scripts\\python.exe -m analysis.run_synthetic_demo --out-dir src/analysis/examples --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from src.analysis import artifacts
from src.analysis import figures as figures_module
from src.analysis import numbers as numbers_module
from src.analysis import pipeline_fixtures, synthetic, tables, validators

DEFAULT_OUT_DIR = Path("analysis") / "examples"

SYNTHETIC_BANNER = (
    "SYNTHETIC DEMONSTRATION OUTPUT — generated from analysis.synthetic fixtures, "
    "NOT from the frozen NF-UNSW-NB15-v3 run artifacts. Never quote these numbers as results."
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_sha() -> str:
    """Best-effort git SHA (analysis is honest about a missing pin in demo mode)."""
    try:
        from src.utils.manifest import git_info

        info = git_info()
        return str(info.get("sha") or "unavailable-synthetic-demo")
    except Exception:  # pragma: no cover - demo only
        return "unavailable-synthetic-demo"


def run_demo(out_dir: str | Path = DEFAULT_OUT_DIR, *, seed: int = synthetic.SEED, status: str = "pending") -> dict[str, Any]:
    """Generate the synthetic artifact package and return a summary dictionary."""
    output_root = Path(out_dir)
    layout = artifacts.ArtifactLayout(output_root / "results")
    layout.ensure()

    bundle = synthetic.make_bundle(seed=seed)
    # S6/S7 schema-faithful fixtures so the validators exercise the anchor/cache
    # contract end-to-end (still synthetic; demo root only).
    pipeline_fixtures.write_synthetic_s6_s7(layout, seed=seed)
    outputs = tables.generate_all(
        cl_tensors=bundle.cl_tensors,
        real_drift=bundle.real_drift,
        model_null_noise=bundle.model_null_noise,
        explainer_repeat_noise=bundle.explainer_repeat_noise,
        retention=bundle.retention,
        neighbourhood=bundle.neighbourhood,
        ear_comparison=bundle.ear_comparison,
        ear_ablations=bundle.ear_ablations,
        layout=layout,
        supported_classes=synthetic.SUPPORTED_CLASSES,
    )
    validation = validators.validate_artifact_set(layout.tables_root, stage="all")
    figure_paths = figures_module.generate_all_figures(
        tensors=bundle.cl_tensors,
        excess_frame=outputs["rq1"]["excess"],
        noise_floor_table=outputs["noise_floor"],
        rq1_aggregated=outputs["rq1"]["aggregated"],
        h1_tests=outputs["rq1"]["h1_tests"],
        rq2_anchor_level=outputs["rq2"]["anchor_level"],
        rq4_comparison=outputs["rq4"]["comparison"],
        layout=layout,
    )
    payload = numbers_module.numbers_from_tables(
        {
            "rq1_h1_tests": outputs["rq1"]["h1_tests"],
            "rq1_strategy_comparison": outputs["rq1"]["strategy_comparison"],
            "rq2_correlations": outputs["rq2"]["correlations"],
            "rq3_auc": outputs["rq3"]["auc"],
            "rq4_comparison": outputs["rq4"]["comparison"],
        },
        git_sha=_git_sha(),
        status=status,
        graph_lift={
            "delta_rand": 0.0,
            "ci95": [0.0, 0.0],
            "artifact": "results/text/data_audit/graph_lift_pilot.json",
        },
    )
    numbers_path = numbers_module.write_numbers_json(payload, output_root / "paper_numbers.json")

    table_paths = sorted({str(Path(path).resolve()) for path in outputs["paths"].values()})
    summary: dict[str, Any] = {
        "banner": SYNTHETIC_BANNER,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "status_of_numbers": status,
        "layout": {
            "tables_root": str(layout.tables_root.resolve()),
            "figures_root": str(layout.figures_root.resolve()),
        },
        "tables": [
            {"path": str(Path(path).resolve().relative_to(Path.cwd())), "sha256": _sha256(Path(path))}
            for path in table_paths
            if Path(path).is_file()
        ],
        "figures": [
            {"path": str(Path(path).resolve().relative_to(Path.cwd())), "sha256": _sha256(Path(path))}
            for path in sorted(figure_paths.values())
            if Path(path).is_file()
        ],
        "paper_numbers": {"path": str(numbers_path.resolve().relative_to(Path.cwd())), "issues": numbers_module.validate_numbers_json(payload)},
        "validation": {
            "overall": validation.overall,
            "counts": validation.counts,
            "non_pass": [
                finding.as_dict() for finding in validation.findings if finding.status != "pass"
            ],
        },
        "s6_s7_fixtures": {
            "anchors": [str(path.resolve().relative_to(Path.cwd())) for path in sorted((layout.results_root / "anchors").glob("period_*.pt"))],
            "explanation_index": str((layout.results_root / "explanations" / "index.csv").resolve().relative_to(Path.cwd())),
        },
        "rq3_notes": outputs["rq3"]["notes"],
        "rq1_h1_configurations": int(len(outputs["rq1"]["h1_tests"])),
        "rq2_correlation_groups": int(len(outputs["rq2"]["correlations"])),
        "rq3_auc_rows": int(len(outputs["rq3"]["auc"])),
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary["manifest_path"] = str(manifest_path)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="output directory (default src/analysis/examples)")
    parser.add_argument("--seed", type=int, default=synthetic.SEED, help="synthetic fixture seed")
    parser.add_argument("--status", default="pending", choices=numbers_module.STATUS_VOCABULARY, help="status recorded for synthetic numbers")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run_demo(args.out_dir, seed=args.seed, status=args.status)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
