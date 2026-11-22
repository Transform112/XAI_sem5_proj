"""Job planning / resume checks for the T23 anchors+explain batch driver.

A run group (model, strategy, seed) produces 5 checkpoints ``M_1..M_5``; for each
period the batch driver (1) builds the S6 anchor set from the frozen test mask and
``M_j``, then (2) generates IG + GNNExplainer explanations through the S7 cache.
Both stages are resume-friendly: anchors are skipped when the per-period anchor
manifest exists and its stored sha matches the anchor file on disk; explanations
are skipped when the canonical evidence index already holds the expected number of
rows for that checkpoint/explainer.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.orchestration.matrix_plan import MatrixRun, eval_out_dir, run_dir, sha256_file

DEFAULT_EXPLAINERS: tuple[str, ...] = ("ig", "gnn_explainer")
DEFAULT_K_VALUES: tuple[int, ...] = (5, 10)
DEFAULT_ANCHORS_PER_PERIOD = 1000  # Doc 2 S6 cap (≤100/class, ≤1000/period)


@dataclass(frozen=True)
class AnchorJob:
    run: MatrixRun
    period: int

    @property
    def key(self) -> str:
        return f"anchors_{self.run.strategy}_{self.run.model}_seed{self.run.seed}_period{self.period}"


@dataclass(frozen=True)
class ExplainJob:
    run: MatrixRun
    period: int
    explainers: tuple[str, ...] = DEFAULT_EXPLAINERS
    k_values: tuple[int, ...] = DEFAULT_K_VALUES

    @property
    def key(self) -> str:
        return f"explain_{self.run.strategy}_{self.run.model}_seed{self.run.seed}_period{self.period}"


def build_jobs(
    plan: Sequence[MatrixRun],
    *,
    periods: int | None = None,
    explainers: Sequence[str] = DEFAULT_EXPLAINERS,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> tuple[list[AnchorJob], list[ExplainJob]]:
    anchor_jobs: list[AnchorJob] = []
    explain_jobs: list[ExplainJob] = []
    for run in plan:
        steps = range(1, int(periods or run.periods) + 1)
        for period in steps:
            anchor_jobs.append(AnchorJob(run=run, period=int(period)))
            explain_jobs.append(
                ExplainJob(
                    run=run,
                    period=int(period),
                    explainers=tuple(str(name) for name in explainers),
                    k_values=tuple(int(value) for value in k_values),
                )
            )
    return anchor_jobs, explain_jobs


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def anchors_dir(paths: Any) -> Path:
    return Path(paths.results_root) / "anchors"


def anchor_file(paths: Any, job: AnchorJob) -> Path:
    return anchors_dir(paths) / f"period_{int(job.period)}.pt"


def anchor_manifest_path(paths: Any, job: AnchorJob) -> Path:
    return (
        Path(paths.audit_root)
        / (
            f"anchors_manifest_{job.run.strategy}_{job.run.model}_seed{job.run.seed}"
            f"_period{int(job.period)}.json"
        )
    )


def checkpoint_path(paths: Any, run: MatrixRun, period: int) -> Path:
    return run_dir(paths, run) / f"M_{int(period)}.pt"


def canonical_explain_index(paths: Any) -> Path:
    return Path(paths.tables_root) / "explain_index.csv"


def anchors_manifest_check(paths: Any, job: AnchorJob) -> dict[str, Any]:
    """Resume check for one anchor-set build."""
    anchor = anchor_file(paths, job)
    manifest = anchor_manifest_path(paths, job)
    result: dict[str, Any] = {
        "status": "missing",
        "anchor_file": str(anchor),
        "manifest": str(manifest),
    }
    if not anchor.is_file():
        result["reason"] = "anchor file missing"
        return result
    if not manifest.is_file():
        result["reason"] = "anchor manifest missing"
        return result
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        result["reason"] = "anchor manifest unreadable"
        return result
    entry = (payload.get("anchor_files") or {}).get(str(int(job.period)))
    if not entry:
        result["reason"] = "period missing from anchor manifest"
        return result
    recorded = str(entry.get("sha256") or "")
    actual = sha256_file(anchor)
    if recorded != actual:
        result.update(status="stale", reason="anchor file sha mismatch vs manifest")
        return result
    result.update(
        status="complete",
        sha256=actual,
        n_selected=int((payload.get("per_period") or {}).get(str(int(job.period)), {}).get("n_selected", 0)),
    )
    return result


def count_explain_index_rows(
    index_path: str | Path,
    *,
    checkpoint_sha: str,
    explainers: Sequence[str] = DEFAULT_EXPLAINERS,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    path = Path(index_path)
    if not path.is_file():
        return counts
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("checkpoint_sha")) != str(checkpoint_sha):
                continue
            explainer = str(row.get("explainer"))
            try:
                k = int(row.get("k") or 0)
            except ValueError:
                continue
            if explainer == "ig" and k not in {int(value) for value in k_values}:
                continue
            counts[explainer] = counts.get(explainer, 0) + 1
    return counts


def explain_completion(
    paths: Any,
    job: ExplainJob,
    *,
    expected_anchors: int = DEFAULT_ANCHORS_PER_PERIOD,
) -> dict[str, Any]:
    """Resume check for one explanation batch (cache-aware; canonical index counts)."""
    checkpoint = checkpoint_path(paths, job.run, job.period)
    result: dict[str, Any] = {
        "status": "missing",
        "checkpoint": str(checkpoint),
        "expected_anchors": int(expected_anchors),
    }
    if not checkpoint.is_file():
        result["reason"] = "checkpoint missing"
        return result
    checkpoint_sha = sha256_file(checkpoint)
    index_path = canonical_explain_index(paths)
    counts = count_explain_index_rows(
        index_path,
        checkpoint_sha=checkpoint_sha,
        explainers=job.explainers,
        k_values=job.k_values,
    )
    expected_ig = int(expected_anchors) * len(job.k_values)
    expected_gnn = int(expected_anchors)
    expected = {}
    if "ig" in job.explainers:
        expected["ig"] = expected_ig
    if "gnn_explainer" in job.explainers:
        expected["gnn_explainer"] = expected_gnn
    complete = all(counts.get(name, 0) >= needs for name, needs in expected.items())
    result.update(
        status="complete" if complete else "partial",
        checkpoint_sha=checkpoint_sha,
        counts=counts,
        expected=expected,
        explain_index=str(index_path),
    )
    return result


# ---------------------------------------------------------------------------
# Commands (kit stage reference)
# ---------------------------------------------------------------------------
def anchors_command(
    job: AnchorJob,
    *,
    python: str | None = None,
    dataset: str,
    config: str = "configs/eval.yaml",
    checks_root: str | None = None,
    graphs_dir: str | None = None,
    out_dir: str | None = None,
    device: str = "auto",
    lane: str | None = None,
) -> list[str]:
    command = [
        python or sys.executable,
        "src/scripts/build_anchors.py",
        "--dataset",
        dataset,
        "--config",
        config,
        "--strategy",
        job.run.strategy,
        "--model",
        job.run.model,
        "--seed",
        str(int(job.run.seed)),
        "--period",
        str(int(job.period)),
        "--device",
        device,
    ]
    if checks_root:
        command += ["--checkpoints-root", checks_root]
    if graphs_dir:
        command += ["--graphs-dir", graphs_dir]
    if out_dir:
        command += ["--out-dir", out_dir]
    if lane:
        command += ["--lane", str(lane)]
    return command


def explain_command(
    job: ExplainJob,
    *,
    python: str | None = None,
    dataset: str,
    config: str = "configs/explain.yaml",
    anchors: str | None = None,
    device: str = "auto",
    lane: str | None = None,
    max_anchors: int = 10_000,
) -> list[str]:
    command = [
        python or sys.executable,
        "src/scripts/run_explain.py",
        "--dataset",
        dataset,
        "--config",
        config,
        "--strategy",
        job.run.strategy,
        "--model",
        job.run.model,
        "--seed",
        str(int(job.run.seed)),
        "--period",
        str(int(job.period)),
        "--explainers",
        ",".join(job.explainers),
        "--k",
        ",".join(str(int(value)) for value in job.k_values),
        "--max-anchors",
        str(int(max_anchors)),
        "--device",
        device,
    ]
    if anchors:
        command += ["--anchors", anchors]
    if lane:
        command += ["--lane", str(lane)]
    return command


def wall_clock_estimate(
    anchor_jobs: Sequence[AnchorJob],
    explain_jobs: Sequence[ExplainJob],
    *,
    anchors_per_period: int = DEFAULT_ANCHORS_PER_PERIOD,
    anchor_model_forward_s: float = 0.05,
    anchor_subgraph_s: float = 0.002,
    ig_per_anchor_s: float = 0.03,
    gnn_explainer_per_anchor_s: float = 0.6,
    explainers: Sequence[str] = DEFAULT_EXPLAINERS,
) -> dict[str, Any]:
    """Projected anchors+explain wall-clock (per-anchor constants are estimates)."""
    anchors_cost = float(len(anchor_jobs)) * (
        float(anchor_model_forward_s) + float(anchors_per_period) * float(anchor_subgraph_s)
    )
    ing = float(len(explain_jobs)) * float(anchors_per_period)
    explain_cost = 0.0
    if "ig" in explainers:
        explain_cost += ing * float(ig_per_anchor_s)
    if "gnn_explainer" in explainers:
        explain_cost += ing * float(gnn_explainer_per_anchor_s)
    total = anchors_cost + explain_cost
    return {
        "formula": (
            "anchors = jobs * (model_forward_s + anchors_per_period * subgraph_s); "
            "explain = jobs * anchors_per_period * (IG_s + GNNExplainer_s)"
        ),
        "anchor_jobs": len(anchor_jobs),
        "explain_jobs": len(explain_jobs),
        "anchors_per_period": int(anchors_per_period),
        "anchors_total_estimate": int(len(anchor_jobs) * int(anchors_per_period)),
        "anchors_seconds": round(anchors_cost, 1),
        "explain_seconds": round(explain_cost, 1),
        "total_seconds": round(total, 1),
        "total_gpu_hours": round(total / 3600.0, 3),
    }


__all__ = [
    "AnchorJob",
    "DEFAULT_ANCHORS_PER_PERIOD",
    "DEFAULT_EXPLAINERS",
    "DEFAULT_K_VALUES",
    "ExplainJob",
    "anchor_file",
    "anchor_manifest_path",
    "anchors_command",
    "anchors_dir",
    "anchors_manifest_check",
    "build_jobs",
    "canonical_explain_index",
    "checkpoint_path",
    "count_explain_index_rows",
    "eval_out_dir",
    "explain_command",
    "explain_completion",
    "wall_clock_estimate",
]
