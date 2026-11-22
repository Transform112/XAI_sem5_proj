"""Pipeline readiness map: every frozen-protocol artifact → producer → validator (T20).

Purpose
-------
Answer one question before the first real S8/S9 run: *which artifact is missing,
who produces it, and which validator stage checks it?* The report lists the full
C1–C7 / S5–S13 artifact chain with:

* the artifact path(s) — canonical skeleton path first, engineer convention second;
* the producing CLI (engineer-owned pipeline);
* the validator stage / check that applies;
* the current status: ``ready`` / ``present-warn`` / ``present-FAIL`` / ``waiting``;
* the blocking inputs for the first real S8–S12 run (same list the engineer's
  ``src/analysis_wiring/pipeline.required_real_inputs`` refuses to run without);
* config/protocol reconciliation checks (D-043 filenames, D-044 import migration,
  D-047 supported set, docs/05 §8 primary metric, frozen δ/β grids).

Authority: ``research/lit/claims_evidence_skeleton.md`` (artifact chain),
``docs/05_EXPERIMENT_PROTOCOL.md`` §8–§13, Doc 2b S7–S13, ``docs/DECISIONS.md``
D-043/D-044/D-047, and the engineer pipeline CLIs.

Usage
-----
    .\\.venv\\Scripts\\python.exe -m analysis.pipeline_readiness
    .\\.venv\\Scripts\\python.exe -m analysis.pipeline_readiness --json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from src.analysis import validators

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_TABLES = REPO_ROOT / "results" / "tables"
REPORT_PATH = REPO_ROOT / "analysis" / "reports" / "pipeline_readiness.md"

STATUS_ORDER = {"present-FAIL": 4, "present-warn": 3, "waiting": 2, "ready": 1, "not-applicable": 0}


@dataclass
class ArtifactRow:
    """One artifact in the frozen protocol chain."""

    name: str
    paths: tuple[str, ...]
    stage: str
    claims: str
    producer: str
    validator: str = "none"  # "table:<file>" | "anchors" | "index" | "none"
    notes: str = ""


#: Frozen artifact chain (skeleton + docs/05 + Doc 2b S7–S13; D-043 filenames).
ARTIFACTS: tuple[ArtifactRow, ...] = (
    ArtifactRow("cl_matrix.json", ("results/text/tables/cl_matrix.json",), "S5", "S5; RQ3 input; C1–C5 input", "src/scripts/run_eval.py", "table:cl_matrix.json"),
    ArtifactRow("cl_metrics.csv", ("results/text/tables/cl_metrics.csv",), "S5", "S5; accuracy side of C1–C5", "src/scripts/run_eval.py", "table:cl_metrics.csv"),
    ArtifactRow("cl_per_class_f1.json", ("results/text/tables/cl_per_class_f1.json",), "S5", "S5; C4/RQ3", "src/scripts/run_eval.py", "table:cl_per_class_f1.json"),
    ArtifactRow(
        "anchor sets",
        ("results/text/anchors/period_1.pt", "results/text/anchors/period_2.pt", "results/text/anchors/period_3.pt", "results/text/anchors/period_4.pt", "results/text/anchors/period_5.pt"),
        "S6",
        "C1–C5 input; RQ2 anchor set",
        "src/scripts/build_anchors.py",
        "anchors",
    ),
    ArtifactRow(
        "retained-fraction table",
        ("results/text/tables/anchor_retention.csv", "results/text/anchors/retained_fraction.csv"),
        "S6",
        "S9 retained fraction; S6 check",
        "src/scripts/build_anchors.py",
        "none",
        notes="D-048: canonical results/text/tables/anchor_retention.csv (engineer internal copy may exist under results/text/anchors/)",
    ),
    ArtifactRow(
        "explanation cache index",
        ("results/text/tables/explain_index.csv", "results/text/explanations/index.csv"),
        "S7",
        "C1–C5 input; S7 cache",
        "src/scripts/run_explain.py",
        "index",
        notes="D-048: canonical committed results/text/tables/explain_index.csv + operational results/text/explanations/index.csv from the same run",
    ),
    ArtifactRow(
        "noise caches (model_null + explainer_repeat + combined)",
        (
            "results/text/explanations_noise/model_null/index.csv",
            "results/text/explanations_noise/explainer_repeat/index.csv",
            "results/text/explanations_noise/combined_null.csv",
        ),
        "S8",
        "C1/C2 null floor",
        "src/scripts/build_noise_floor.py",
        "noise_cache",
        notes="D-048(d): R=5 reseeds / R=10 repeats per anchor x explainer + combined_null.csv",
    ),
    ArtifactRow("drift metrics (IG)", ("results/text/tables/drift_metrics_ig.csv",), "S8", "C1, C2", "src/scripts/run_analysis.py --mode real", "table:drift_metrics_ig.csv"),
    ArtifactRow("drift metrics (GNNExplainer)", ("results/text/tables/drift_metrics_gnnexplainer.csv",), "S8", "C1, C2", "src/scripts/run_analysis.py --mode real", "table:drift_metrics_gnnexplainer.csv"),
    ArtifactRow("noise floor table", ("results/text/tables/noise_floor.csv",), "S8", "C1; S8 output", "src/scripts/run_analysis.py --mode real", "table:noise_floor.csv"),
    ArtifactRow("rq1 anchor level", ("results/text/tables/rq1_anchor_level.csv",), "S9", "C1", "src/scripts/run_analysis.py --mode real", "table:rq1_anchor_level.csv"),
    ArtifactRow("rq1 aggregated", ("results/text/tables/rq1_aggregated.csv",), "S9", "C1, C2", "src/scripts/run_analysis.py --mode real", "table:rq1_aggregated.csv"),
    ArtifactRow("rq1 H1 tests", ("results/text/tables/rq1_h1_tests.csv",), "S9", "C1", "src/scripts/run_analysis.py --mode real", "table:rq1_h1_tests.csv"),
    ArtifactRow("rq1 strategy comparison", ("results/text/tables/rq1_strategy_comparison.csv",), "S9", "C2", "src/scripts/run_analysis.py --mode real", "table:rq1_strategy_comparison.csv"),
    ArtifactRow("rq2 anchor level", ("results/text/tables/rq2_anchor_level.csv",), "S10", "C3", "src/scripts/run_analysis.py --mode real", "table:rq2_anchor_level.csv"),
    ArtifactRow("rq2 correlations", ("results/text/tables/rq2_correlations.csv",), "S10", "C3", "src/scripts/run_analysis.py --mode real", "table:rq2_correlations.csv"),
    ArtifactRow("rq3 features", ("results/text/tables/rq3_features.csv",), "S11", "C4", "src/scripts/run_analysis.py --mode real", "table:rq3_features.csv"),
    ArtifactRow("rq3 forgetting labels", ("results/text/tables/rq3_forgetting_labels.csv",), "S11", "C4", "src/scripts/run_analysis.py --mode real", "table:rq3_forgetting_labels.csv"),
    ArtifactRow("rq3 AUC", ("results/text/tables/rq3_auc.csv",), "S11", "C4", "src/scripts/run_analysis.py --mode real", "table:rq3_auc.csv"),
    ArtifactRow("rq3 coefficients (extra, D-043)", ("results/text/tables/rq3_coefficients.csv",), "S11", "C4 support", "src/scripts/run_analysis.py --mode real", "table:rq3_coefficients.csv"),
    ArtifactRow("rq4 comparison", ("results/text/tables/rq4_comparison.csv",), "S12", "C5", "src/scripts/run_analysis.py --mode real", "table:rq4_comparison.csv"),
    ArtifactRow("rq4 ablations", ("results/text/tables/rq4_ablations.csv",), "S12", "C5", "src/scripts/run_analysis.py --mode real", "table:rq4_ablations.csv"),
    ArtifactRow(
        "graph-lift pilot (C6)",
        ("results/text/data_audit/graph_lift_pilot_r1b.json", "results/text/data_audit/graph_lift_pilot.json"),
        "S-D1",
        "C6 setup validity",
        "src/scripts/run_graph_lift_pilot.py",
        "none",
        notes="G1c PASS evidence (D-042); not regenerated by S8–S12",
    ),
    ArtifactRow("per_class_f1 npz/csv (analysis-materialised)", ("results/text/tables/per_class_f1.csv", "results/text/tables/per_class_f1.npz"), "S13", "C4 input", "analysis.tables.generate_per_class_f1", "table:per_class_f1.csv"),
    ArtifactRow("figures", ("results/images",), "S13", "C1–C5 figures", "analysis.figures", "none"),
    ArtifactRow("paper numbers (S13 writer)", ("paper_writing/manuscript/numbers.json",), "S13", "C1–C6 headline numbers", "S13 writer (Lead/verifier) + analysis.numbers", "none"),
)

#: Blocking inputs for the first real S8–S12 run (mirrors
#: ``src/analysis_wiring/pipeline.required_real_inputs``; T18 guard).
BLOCKING_INPUTS: tuple[str, ...] = (
    "results/text/tables/cl_matrix.json",
    "results/text/explanations/index.csv",
    "results/text/anchors/period_*.pt",
    "datasets/processed/NF-UNSW-NB15-v3/period_*.pt",
    "results/text/explanations_noise/model_null/",
    "results/text/explanations_noise/explainer_repeat/",
)


@dataclass
class RowStatus:
    row: ArtifactRow
    existing: list[str]
    status: str
    detail: str


@dataclass
class ReadinessReport:
    generated_at_utc: str
    results_tables: str
    rows: list[RowStatus] = field(default_factory=list)
    blocking: list[dict[str, Any]] = field(default_factory=list)
    config_checks: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def counts(self) -> dict[str, int]:
        counts = {status: 0 for status in STATUS_ORDER}
        for row in self.rows:
            counts[row.status] = counts.get(row.status, 0) + 1
        return {status: counts.get(status, 0) for status in STATUS_ORDER}

    def as_dict(self) -> dict[str, Any]:
        return {
            "generated_at_utc": self.generated_at_utc,
            "results_tables": self.results_tables,
            "counts": self.counts,
            "rows": [
                {
                    "artifact": row.row.name,
                    "stage": row.row.stage,
                    "claims": row.row.claims,
                    "producer": row.row.producer,
                    "validator": row.row.validator,
                    "existing": row.existing,
                    "status": row.status,
                    "detail": row.detail,
                    "notes": row.row.notes,
                }
                for row in self.rows
            ],
            "blocking_inputs": self.blocking,
            "config_checks": self.config_checks,
            "notes": self.notes,
        }


def _validate_row(row: ArtifactRow, existing: list[str], results_tables: Path) -> tuple[str, str]:
    results_root = results_tables.parent
    repo_root = results_root.parent
    if row.validator == "table":
        spec = row.validator.split(":", 1)[1]
        report = validators.validate_artifact(results_tables, spec)
        status = {validators.PASS: "ready", validators.WARN: "present-warn", validators.FAIL: "present-FAIL"}[report.overall]
        return status, f"validator {report.overall} ({report.counts})"
    if row.validator == "anchors":
        report = validators.ValidationReport(results_tables=str(results_tables), stage="s6")
        validators.validate_anchor_artifacts(report, results_root / "anchors")
        status = {validators.PASS: "ready", validators.WARN: "present-warn", validators.FAIL: "present-FAIL"}[report.overall]
        return status, f"anchor validator {report.overall}"
    if row.validator == "noise_cache":
        report = validators.ValidationReport(results_tables=str(results_tables), stage="s8")
        validators.validate_noise_cache_artifacts(report, results_root)
        status = {validators.PASS: "ready", validators.WARN: "present-warn", validators.FAIL: "waiting" if not (results_root / "explanations_noise" / "model_null" / "index.csv").is_file() else "present-FAIL"}[report.overall]
        return status, f"noise-cache validator {report.overall}"
    if row.validator == "index":
        index = results_tables / "explain_index.csv"  # D-048 canonical evidence index
        if not index.is_file():
            index = results_root / "explanations" / "index.csv"
        report = validators.ValidationReport(results_tables=str(results_tables), stage="s7")
        validators.validate_explain_index_csv(report, index, require_entries=False)
        empty = any("no entries" in finding.detail or "zero rows" in finding.detail for finding in report.findings)
        if empty:
            return "waiting", "index exists but has no cached explanations yet — S7 explanation run pending"
        status = {validators.PASS: "ready", validators.WARN: "present-warn", validators.FAIL: "present-FAIL"}[report.overall]
        return status, f"index validator {report.overall}"
    return "ready", "present (no dedicated validator)"


def build_readiness(results_tables: Path = RESULTS_TABLES) -> ReadinessReport:
    results_tables = Path(results_tables)
    results_root = results_tables.parent
    repo_root = results_root.parent
    report = ReadinessReport(
        generated_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        results_tables=str(results_tables),
    )
    for row in ARTIFACTS:
        existing = [path for path in row.paths if (repo_root / path).exists()]
        if not existing:
            report.rows.append(RowStatus(row, [], "waiting", "artifact not produced yet"))
            continue
        status, detail = _validate_row(row, existing, results_tables)
        report.rows.append(RowStatus(row, existing, status, detail))
    for pattern in BLOCKING_INPUTS:
        target = repo_root / pattern.replace("period_*.pt", "period_*").rstrip("/")
        if pattern.endswith("/"):
            found = target.is_dir() and any(target.iterdir())
        elif "*" in pattern:
            directory = (repo_root / pattern).parent
            found = directory.is_dir() and any(directory.glob("period_*.pt"))
        else:
            found = (repo_root / pattern).is_file()
        report.blocking.append({"input": pattern, "present": bool(found)})
    report.config_checks = _config_checks(repo_root)
    report.notes.append(
        "S8–S12 tables are produced by `src/scripts/run_analysis.py --mode real` only after every blocking input exists "
        "(the wiring refuses to write partial tables; T18)."
    )
    return report


def _config_checks(repo_root: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append({"check": name, "status": status, "detail": detail})

    def load(name: str) -> Mapping[str, Any]:
        path = repo_root / "configs" / name
        if not path.is_file():
            add(f"configs/{name}", validators.FAIL, "config missing")
            return {}
        try:
            text = path.read_text(encoding="utf-8")
            return yaml.safe_load(text) or {}
        except Exception as exc:  # pragma: no cover - corrupt config
            add(f"configs/{name}", validators.FAIL, f"unreadable YAML: {exc}")
            return {}

    eval_cfg = load("eval.yaml")
    supported = list((((eval_cfg.get("s5") or {}).get("dual_macro_f1") or {}).get("supported_set")) or [])
    add(
        "configs/eval.yaml supported_set (D-047)",
        validators.PASS if "Benign" in supported else validators.FAIL,
        f"supported_set={supported}; D-047 requires the 6 attack classes + Benign",
    )
    drift_cfg = load("drift.yaml")
    metrics = drift_cfg.get("metrics") or {}
    feature_primary = ((metrics.get("feature") or {}).get("primary"))
    edge_primary = ((metrics.get("edge") or {}).get("primary"))
    add(
        "configs/drift.yaml primary metric (docs/05 §8)",
        validators.PASS if feature_primary == "jaccard_top5" and edge_primary == "edge_jaccard_top5" else validators.FAIL,
        f"feature.primary={feature_primary!r}, edge.primary={edge_primary!r}; frozen values jaccard_top5 / edge_jaccard_top5",
    )
    floor = drift_cfg.get("noise_floor") or {}
    add(
        "configs/drift.yaml noise floor",
        validators.PASS
        if floor.get("quantile") == 0.95
        and ((floor.get("pooling") or {}).get("min_pairs_per_class") == 20)
        and ((floor.get("components") or {}).get("model_null") or {}).get("reseeds") == 5
        and ((floor.get("components") or {}).get("explainer_repeat") or {}).get("repeats") == 10
        else validators.FAIL,
        f"quantile={floor.get('quantile')}, pooling={floor.get('pooling')}, components set per docs/05 §9",
    )
    configured_index = str(((drift_cfg.get("inputs") or {}).get("explanation_index")) or "")
    canonical_index = "results/text/tables/explain_index.csv"
    add(
        "configs/drift.yaml explanation_index path (D-048)",
        validators.PASS if configured_index == canonical_index else validators.WARN,
        f"config input={configured_index!r}; D-048 canonical evidence index = {canonical_index!r} "
        "(writer also emits results/text/explanations/index.csv in the same run; both must carry the same cache-key rows)",
    )
    rq3_cfg = load("rq3.yaml")
    labels = rq3_cfg.get("labels") or {}
    add(
        "configs/rq3.yaml labels (docs/05 §11)",
        validators.PASS if labels.get("delta") == 0.05 and labels.get("horizon") == "next_or_final" else validators.FAIL,
        f"delta={labels.get('delta')}, horizon={labels.get('horizon')!r}",
    )
    cv = rq3_cfg.get("cross_validation") or {}
    add(
        "configs/rq3.yaml CV group (A9)",
        validators.PASS if cv.get("group_primary") == "checkpoint_t" else validators.WARN,
        f"group_primary={cv.get('group_primary')!r}",
    )
    ear_cfg = load("ear.yaml")
    training = ear_cfg.get("training") or {}
    ablations = ear_cfg.get("ablations") or {}
    add(
        "configs/ear.yaml grids (docs/05 §12)",
        validators.PASS
        if training.get("beta_grid") == [0.1, 0.5, 1.0] and ablations.get("buffer_edges") == [500, 2000, 5000]
        else validators.WARN,
        f"beta_grid={training.get('beta_grid')}, buffer ablations={ablations.get('buffer_edges')}",
    )
    cl_metrics_path = repo_root / "src" / "eval" / "cl_metrics.py"
    if cl_metrics_path.is_file():
        source = cl_metrics_path.read_text(encoding="utf-8", errors="replace")
        add(
            "D-044 import migration (src/eval/cl_metrics.py)",
            validators.PASS if "analysis.metrics_cl" in source else validators.FAIL,
            "writer must import the authoritative analysis.metrics_cl (D-044/D-047) — currently duplicated implementation",
        )
    add(
        "D-048 canonical evidence paths",
        validators.PASS
        if configured_index == "results/text/tables/explain_index.csv"
        else validators.WARN,
        "canonical S7 index results/text/tables/explain_index.csv; canonical S6 retention results/text/tables/anchor_retention.csv (D-048)",
    )
    report = repo_root / "analysis" / "reports" / "s5_crosscheck.md"
    if report.is_file():
        text = report.read_text(encoding="utf-8", errors="replace")
        verdict = "unknown"
        for line in text.splitlines():
            if line.startswith("## Verdict:"):
                verdict = line.replace("## Verdict:", "").replace("*", "").strip()
                break
        add("S5 cross-check (src/analysis/reports/s5_crosscheck.md)", validators.WARN if verdict.startswith("PASS-WITH") else validators.PASS, f"last verdict: {verdict}")
    return checks


def render_markdown(readiness: ReadinessReport) -> str:
    lines: list[str] = []
    lines.append("# Pipeline readiness — frozen-protocol artifact chain (T20)")
    lines.append("")
    lines.append(
        "*Owner: analyst. Generated by `python -m src.analysis.pipeline_readiness`. Authority: "
        "`research/lit/claims_evidence_skeleton.md`, `docs/05` §8–§13, Doc 2b S7–S13, "
        "`docs/DECISIONS.md` D-043/D-044/D-047. Regenerate after every pipeline milestone; "
        "statuses are derived from the artifacts on disk + `src.analysis.validators`.*"
    )
    lines.append("")
    counts = readiness.counts
    lines.append(f"*Generated: {readiness.generated_at_utc} · results root: `{readiness.results_tables}`*")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append("| status | count |")
    lines.append("| --- | --- |")
    for status in STATUS_ORDER:
        lines.append(f"| `{status}` | {counts.get(status, 0)} |")
    lines.append("")
    lines.append("Status vocabulary: `ready` = present and validator passes; `present-warn` = present with non-blocking warnings; "
                 "`present-FAIL` = present but fails a schema/invariant check; `waiting` = not produced yet.")
    lines.append("")
    lines.append("## Artifact chain")
    lines.append("")
    lines.append("| # | artifact | stage | claims / analysis | paths | producer CLI | validator | status | detail |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for index, item in enumerate(readiness.rows, start=1):
        paths = "<br>".join(f"`{path}`" for path in item.row.paths)
        detail = str(item.detail).replace("|", "\\|")
        notes = f" {item.row.notes}" if item.row.notes else ""
        lines.append(
            f"| {index} | **{item.row.name}** | {item.row.stage} | {item.row.claims} | {paths} | "
            f"`{item.row.producer}` | `{item.row.validator}` | **{item.status}** | {detail}{notes} |"
        )
    lines.append("")
    lines.append("## Blocking inputs for the first real S8–S12 run")
    lines.append("")
    lines.append("These mirror `src/analysis_wiring/pipeline.required_real_inputs`; the wiring refuses to write partial tables until all exist (T18).")
    lines.append("")
    lines.append("| input | present |")
    lines.append("| --- | --- |")
    for item in readiness.blocking:
        lines.append(f"| `{item['input']}` | {'yes' if item['present'] else '**no**'} |")
    lines.append("")
    lines.append("## Config / protocol reconciliation")
    lines.append("")
    lines.append("| check | status | detail |")
    lines.append("| --- | --- | --- |")
    for check in readiness.config_checks:
        lines.append(f"| {check['check']} | **{check['status']}** | {str(check['detail']).replace('|', '&#124;')} |")
    lines.append("")
    ticks = chr(96) * 3
    lines.append("## Exact first real S6 -> S7 -> S8 -> S9 command sequence")
    lines.append("")
    lines.append(
        "Run from the repository root on the canonical lane (--lane rtx5060) after G2 checkpoints exist. "
        "Each step is gated by the validators; stop on any FAIL. --dry-run is safe and lists missing inputs without writing."
    )
    lines.append("")
    sequence = [
        ("S5 (already produced): validate the writer artifacts",
         "python -m src.analysis.validators --stage s5 --strict",
         "python -m src.analysis.s5_crosscheck"),
        ("S6: anchors + canonical retention table",
         "python src/scripts/build_anchors.py --dataset NF-UNSW-NB15-v3 --strategy ft --model egraphsage --seed 42 --lane rtx5060",
         "python -m src.analysis.validators --stage s6 --strict"),
        ("S7: explanations + canonical evidence index",
         "python src/scripts/run_explain.py --dataset NF-UNSW-NB15-v3 --strategy ft --model egraphsage --seed 42 --period 1 --lane rtx5060",
         "python -m src.analysis.validators --stage s7 --strict"),
        ("S8 noise caches: required before any S8/S9 table run (D-048d)",
         "python src/scripts/build_noise_floor.py --dataset NF-UNSW-NB15-v3 --strategy ft --model egraphsage --seed 42 --period 1 --lane rtx5060",
         "python -m src.analysis.validators --stage s8 --strict"),
        ("S8/S9 wiring: drift metrics, noise floor, RQ1 tables (refuses partial output)",
         "python src/scripts/run_analysis.py --mode real --lane rtx5060 --out-root . --anchors-dir results/text/anchors --cache-root results/text/explanations --noise-cache-root results/text/explanations_noise --cl-matrix results/text/tables/cl_matrix.json --graphs-dir datasets/processed/NF-UNSW-NB15-v3 --dry-run",
         "python -m src.analysis.validators --stage s9 --strict"),
        ("Independent H1 arithmetic + S8/S9 report",
         "python -m src.analysis.s8_validation",
         "python -m src.analysis.pipeline_readiness"),
    ]
    for title, command, gate in sequence:
        lines.append("### " + title)
        lines.append("")
        lines.append(ticks + "powershell")
        lines.append(str(command))
        lines.append(ticks)
        lines.append("Gate: " + chr(96) + gate + chr(96))
        lines.append("")
    lines.append(
        "Note: run_analysis --mode real writes into --out-root; pass --out-root . (or a staging root) so the frozen "
        "results tree is used, and re-run with --dry-run first to confirm the missing-input list is empty."
    )
    lines.append("")
    if readiness.notes:
        lines.append("## Notes")
        lines.append("")
        for note in readiness.notes:
            lines.append("- " + note)
        lines.append("")
    return "\n".join(lines)


def write_report(readiness: ReadinessReport, path: Path = REPORT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_markdown(readiness), encoding="utf-8")
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pipeline readiness report (T20).")
    parser.add_argument("--results-tables", default=str(RESULTS_TABLES))
    parser.add_argument("--out", default=str(REPORT_PATH))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    readiness = build_readiness(Path(args.results_tables))
    path = write_report(readiness, Path(args.out))
    if args.json:
        print(json.dumps(readiness.as_dict(), indent=2, sort_keys=True))
    else:
        print(json.dumps({"report": str(path), "counts": readiness.counts, "blocking_missing": [item["input"] for item in readiness.blocking if not item["present"]], "config_checks": readiness.config_checks}, indent=2))
    failed = [check for check in readiness.config_checks if check["status"] == validators.FAIL]
    return 1 if failed else 0


__all__ = ["ARTIFACTS", "BLOCKING_INPUTS", "ReadinessReport", "build_readiness", "main", "render_markdown", "write_report"]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    sys.exit(main())
