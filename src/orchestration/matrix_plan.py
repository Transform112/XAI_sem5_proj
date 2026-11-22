"""Planning, resume-state and completion checks for the T23 matrix driver.

Pure helpers (no torch/training) so scheduling/resume/filter logic is unit-testable
with tiny synthetic fixtures. The CLI ``src/scripts/run_matrix.py`` executes the plan
in one Python process per run group so the continual-update strategy state
(replay buffer, EWC Fisher/theta*) survives the period sequence.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_MODELS: tuple[str, ...] = ("egraphsage", "gcn")
DEFAULT_STRATEGIES: tuple[str, ...] = ("ft", "replay", "ewc")
OPTIONAL_STRATEGIES: tuple[str, ...] = ("ear_a", "ear_b")
DEFAULT_SEEDS: tuple[int, ...] = (42, 43, 44, 45, 46)
DEFAULT_PERIODS = 5
FULL_SIZE_EDGES = 1_131_909  # period-1 frozen C1 count (wall-clock formula anchor)
PLACEHOLDER_EPOCH_SECONDS = 20.0  # only for dry-run placeholder; never a measured value

#: Mean epochs per period used only for the dry-run wall-clock estimate; the
#: actual epoch count is decided by early stopping (patience 5, cap 30).
ESTIMATE_EPOCHS_PER_PERIOD = 12


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_csv(value: str | None, *, cast: type = str) -> tuple[Any, ...]:
    """Parse a comma-separated CLI filter value; ``None`` means "all"."""
    if value is None or str(value).strip() == "" or str(value).strip().lower() == "all":
        return ()
    return tuple(cast(item.strip()) for item in str(value).split(",") if item.strip())


@dataclass(frozen=True)
class MatrixRun:
    """One (model, strategy, seed) run: P sequential continual updates."""

    model: str
    strategy: str
    seed: int
    periods: int = DEFAULT_PERIODS

    @property
    def key(self) -> str:
        return f"{self.strategy}_{self.model}_seed{self.seed}"

    @property
    def steps(self) -> tuple[int, ...]:
        return tuple(range(1, int(self.periods) + 1))


def build_plan(
    *,
    models: Sequence[str] = DEFAULT_MODELS,
    strategies: Sequence[str] = DEFAULT_STRATEGIES,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    periods: int = DEFAULT_PERIODS,
    include_ear: bool = False,
) -> list[MatrixRun]:
    """Deterministic grid in execution order (strategy, model, seed)."""
    selected_strategies = list(strategies) + ([*OPTIONAL_STRATEGIES] if include_ear else [])
    seen: set[str] = set()
    plan: list[MatrixRun] = []
    for strategy in selected_strategies:
        for model in models:
            for seed in seeds:
                run = MatrixRun(model=str(model), strategy=str(strategy), seed=int(seed), periods=int(periods))
                if run.key in seen:
                    continue
                seen.add(run.key)
                plan.append(run)
    return plan


def filter_plan(plan: Sequence[MatrixRun], *, max_runs: int | None = None) -> list[MatrixRun]:
    """Staged execution: keep at most ``max_runs`` run groups."""
    if max_runs is None:
        return list(plan)
    if int(max_runs) < 0:
        raise ValueError("--max-runs must be >= 0")
    return list(plan)[: int(max_runs)]


# ---------------------------------------------------------------------------
# Frozen paths / hashes
# ---------------------------------------------------------------------------
def artifacts_root(paths: Any) -> Path:
    return Path(paths.results_root) / Path(paths.tables_root).name if False else Path(paths.tables_root)


def run_dir(paths: Any, run: MatrixRun) -> Path:
    return Path(paths.run_dir(strategy=run.strategy, model=run.model, seed=run.seed))


def checkpoint_path(paths: Any, run: MatrixRun, period: int) -> Path:
    return run_dir(paths, run) / f"M_{int(period)}.pt"


def metrics_path(paths: Any, run: MatrixRun, period: int) -> Path:
    return Path(paths.audit_root) / f"train_{run.strategy}_{run.model}_seed{run.seed}_period{int(period)}.json"


def run_manifest_path(paths: Any, run: MatrixRun, period: int) -> Path:
    return (
        Path(paths.audit_root)
        / f"train_manifest_{run.strategy}_{run.model}_seed{run.seed}_period{int(period)}.json"
    )


def group_manifest_path(paths: Any, run: MatrixRun) -> Path:
    return Path(paths.audit_root) / f"matrix_manifest_{run.key}.json"


def eval_out_dir(paths: Any, run: MatrixRun) -> Path:
    return Path(paths.tables_root) / "matrix" / run.key


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def manifest_checkpoint_sha(manifest_path: str | Path) -> str | None:
    """Extract the completed checkpoint sha256 from a per-run manifest."""
    path = Path(manifest_path)
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if str(payload.get("status")) != "success":
        return None
    for artifact in payload.get("artifacts") or []:
        if str(artifact.get("role")) == "checkpoint" and artifact.get("sha256"):
            return str(artifact["sha256"])
    return None


def step_completion(paths: Any, run: MatrixRun, period: int) -> dict[str, Any]:
    """Resume check for one period of one run group.

    Completed requires the checkpoint, the metrics JSON, a successful pinned
    per-run manifest, and a checkpoint sha256 matching the manifest hash.
    """
    checkpoint = checkpoint_path(paths, run, period)
    metrics = metrics_path(paths, run, period)
    manifest = run_manifest_path(paths, run, period)
    result: dict[str, Any] = {
        "period": int(period),
        "checkpoint": str(checkpoint),
        "manifest": str(manifest),
        "status": "missing",
    }
    if not checkpoint.is_file():
        result["reason"] = "checkpoint missing"
        return result
    if not metrics.is_file():
        return result
    if not manifest.is_file():
        return result
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        result["reason"] = "manifest unreadable"
        return result
    if str(payload.get("status")) != "success":
        result["reason"] = f"manifest status {payload.get('status')!r}"
        return result
    recorded = manifest_checkpoint_sha(manifest)
    if recorded is None:
        result.update(status="complete", verified=False, reason="no checkpoint sha in manifest")
        return result
    actual = sha256_file(checkpoint)
    if actual != recorded:
        result.update(status="stale", reason="checkpoint sha mismatch vs manifest")
        return result
    result.update(status="complete", verified=True, sha256=actual)
    return result


def eval_completion(paths: Any, run: MatrixRun) -> dict[str, Any]:
    out_dir = eval_out_dir(paths, run)
    matrix = out_dir / "cl_matrix.json"
    metrics_csv = out_dir / "cl_metrics.csv"
    result: dict[str, Any] = {"status": "missing", "out_dir": str(out_dir)}
    if not out_dir.is_dir():
        result["reason"] = "eval out-dir missing"
        return result
    if not matrix.is_file() or not metrics_csv.is_file():
        result["reason"] = "cl_matrix.json / cl_metrics.csv missing"
        return result
    try:
        payload = json.loads(matrix.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        result["reason"] = "cl_matrix.json unreadable"
        return result
    checkpoints = payload.get("checkpoints_evaluated")
    if checkpoints is None:
        # T30 writer also stores it under evaluation.checkpoints_evaluated; accept
        # either location so resume detection is robust across writer versions.
        checkpoints = (payload.get("evaluation") or {}).get("checkpoints_evaluated")
    if checkpoints is None:
        result["reason"] = "cl_matrix.json missing checkpoints_evaluated"
        return result
    result.update(status="complete", checkpoints=checkpoints)
    return result


def group_completion(paths: Any, run: MatrixRun) -> dict[str, Any]:
    steps = [step_completion(paths, run, period) for period in run.steps]
    complete_steps = [step for step in steps if step["status"] == "complete"]
    pending = [step["period"] for step in steps if step["status"] != "complete"]
    eval_state = eval_completion(paths, run)
    status = "missing"
    if len(complete_steps) == len(steps) and eval_state["status"] == "complete":
        status = "complete"
    elif complete_steps:
        status = "partial"
    return {
        "key": run.key,
        "model": run.model,
        "strategy": run.strategy,
        "seed": run.seed,
        "periods": int(run.periods),
        "status": status,
        "steps": steps,
        "pending_periods": pending,
        "eval": eval_state,
    }


# ---------------------------------------------------------------------------
# Commands (for dry-run rendering and the kit wrapper)
# ---------------------------------------------------------------------------
def train_command(
    run: MatrixRun,
    period: int,
    *,
    python: str | None = None,
    dataset: str,
    config: str = "configs/train.yaml",
    device: str = "auto",
    lane: str | None = None,
    init_checkpoint: str | None = None,
) -> list[str]:
    """Reference CLI command for one period (the driver executes this in-process)."""
    command = [
        python or sys.executable,
        "src/scripts/run_train.py",
        "--dataset",
        dataset,
        "--config",
        config,
        "--stage",
        "S3",
        "--strategy",
        run.strategy,
        "--model",
        run.model,
        "--period",
        str(int(period)),
        "--seed",
        str(int(run.seed)),
        "--device",
        device,
    ]
    if lane:
        command += ["--lane", str(lane)]
    if init_checkpoint:
        command += ["--init-checkpoint", str(init_checkpoint)]
    return command


def eval_command(
    run: MatrixRun,
    *,
    python: str | None = None,
    dataset: str,
    config: str = "configs/eval.yaml",
    device: str = "auto",
    lane: str | None = None,
) -> list[str]:
    command = [
        python or sys.executable,
        "src/scripts/run_eval.py",
        "--dataset",
        dataset,
        "--config",
        config,
        "--strategy",
        run.strategy,
        "--model",
        run.model,
        "--seed",
        str(int(run.seed)),
        "--device",
        device,
        "--out-dir",
        "__OUT_DIR__",
    ]
    if lane:
        command += ["--lane", str(lane)]
    return command


def wall_clock_estimate(
    plan: Sequence[MatrixRun],
    *,
    epoch_seconds_full_size: float | None = None,
    epochs_per_period: int = ESTIMATE_EPOCHS_PER_PERIOD,
    period_edge_counts: Mapping[int, int] | None = None,
    eval_forward_fraction: float = 0.35,
    eval_passes: int = 25,
    budget_gpu_hours: float = 150.0,
) -> dict[str, Any]:
    """Projected matrix wall-clock from a measured sampled-batch epoch time.

    Formula per run: ``periods * epochs_per_period * epoch_s(edges_j)`` training
    plus ``eval_passes * epoch_s(period_1) * eval_forward_fraction`` for the S5
    evaluation group. ``epoch_seconds_full_size`` MUST come from measured timing
    (kit-v7 RTX 5060, or a `--anchor-metrics` file); a missing anchor uses a
    clearly flagged placeholder so budgets can never be mistaken for measured.
    """
    counts = dict(period_edge_counts or {})
    full_edges = float(counts.get(1, FULL_SIZE_EDGES)) or float(FULL_SIZE_EDGES)
    placeholder = epoch_seconds_full_size is None
    base_epoch = (
        float(epoch_seconds_full_size) if epoch_seconds_full_size is not None else PLACEHOLDER_EPOCH_SECONDS
    )

    def epoch_seconds(period: int) -> float:
        edges = float(counts.get(int(period), FULL_SIZE_EDGES if period == 1 else counts.get(1, FULL_SIZE_EDGES)))
        return base_epoch * (edges / full_edges)

    per_run_seconds: list[float] = []
    by_strategy_seconds: dict[str, float] = {}
    by_model_seconds: dict[str, float] = {}
    for run in plan:
        train = sum(int(epochs_per_period) * epoch_seconds(period) for period in run.steps)
        eval_seconds = float(eval_passes) * epoch_seconds(1) * float(eval_forward_fraction)
        seconds = train + eval_seconds
        per_run_seconds.append(seconds)
        by_strategy_seconds[run.strategy] = by_strategy_seconds.get(run.strategy, 0.0) + seconds
        by_model_seconds[run.model] = by_model_seconds.get(run.model, 0.0) + seconds
    total = float(sum(per_run_seconds))
    total_gpu_hours = total / 3600.0
    warnings: list[str] = []
    if placeholder:
        warnings.append(
            "PLACEHOLDER_ANCHOR: no measured epoch_seconds_full_size supplied; budget uses "
            f"{PLACEHOLDER_EPOCH_SECONDS:.0f} s/epoch placeholder and MUST NOT be treated as a real projection"
        )
    if total_gpu_hours > float(budget_gpu_hours):
        warnings.append(
            f"BUDGET_OVERRUN: projected {total_gpu_hours:.1f} GPU-h > {float(budget_gpu_hours):.0f} GPU-h cap "
            "(at this anchor; consider staging or the sampler fast path)"
        )
    return {
        "formula": (
            "per run = sum_{j=1..P} E_epoch * epoch_s(edges_j) "
            "+ eval_passes * epoch_s(P1) * eval_forward_fraction; "
            "epoch_s(edges) = epoch_seconds_full_size * edges / edges(P1)"
        ),
        "epoch_seconds_full_size": base_epoch,
        "epoch_seconds_source": "placeholder" if placeholder else "measured",
        "placeholder_anchor": placeholder,
        "epochs_per_period": int(epochs_per_period),
        "eval_passes": int(eval_passes),
        "n_runs": len(plan),
        "per_run_seconds_mean": round(total / max(len(plan), 1), 2),
        "total_seconds": round(total, 2),
        "total_gpu_hours": round(total_gpu_hours, 3),
        "budget_gpu_hours": float(budget_gpu_hours),
        "over_budget": bool(total_gpu_hours > float(budget_gpu_hours)),
        "by_strategy_gpu_hours": {
            strategy: round(seconds / 3600.0, 3) for strategy, seconds in sorted(by_strategy_seconds.items())
        },
        "by_model_gpu_hours": {
            model: round(seconds / 3600.0, 3) for model, seconds in sorted(by_model_seconds.items())
        },
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Resumable state + index
# ---------------------------------------------------------------------------
def load_state(path: str | Path) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file():
        return {"schema_version": 1, "groups": {}}
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {"schema_version": 1, "groups": {}}
    payload.setdefault("groups", {})
    return payload


def save_state(path: str | Path, state: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    state = dict(state)
    state["updated_at_utc"] = utc_now()
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


INDEX_COLUMNS = [
    "run_key",
    "model",
    "strategy",
    "seed",
    "kind",
    "period",
    "status",
    "checkpoint",
    "manifest",
    "eval_out_dir",
    "started_at_utc",
    "finished_at_utc",
    "duration_s",
    "error",
]


def load_index(path: str | Path) -> dict[tuple[str, str, int], dict[str, Any]]:
    candidate = Path(path)
    rows: dict[tuple[str, str, int], dict[str, Any]] = {}
    if not candidate.is_file():
        return rows
    with candidate.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (str(row.get("run_key")), str(row.get("kind")), int(row.get("period") or 0))
            rows[key] = dict(row)
    return rows


def write_index(path: str | Path, rows: Mapping[tuple[str, str, int], Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        for key in sorted(rows, key=lambda item: (item[0], item[1], int(item[2]))):
            row = {column: rows[key].get(column, "") for column in INDEX_COLUMNS}
            writer.writerow(row)


def upsert_index_row(
    rows: dict[tuple[str, str, int], dict[str, Any]],
    *,
    run: MatrixRun,
    kind: str,
    period: int,
    status: str,
    **extra: Any,
) -> None:
    key = (run.key, str(kind), int(period))
    row = rows.get(key, {})
    row.update(
        {
            "run_key": run.key,
            "model": run.model,
            "strategy": run.strategy,
            "seed": run.seed,
            "kind": str(kind),
            "period": int(period),
            "status": str(status),
        }
    )
    row.update({name: value for name, value in extra.items() if name in INDEX_COLUMNS})
    rows[key] = row


__all__ = [
    "DEFAULT_MODELS",
    "DEFAULT_PERIODS",
    "DEFAULT_SEEDS",
    "DEFAULT_STRATEGIES",
    "ESTIMATE_EPOCHS_PER_PERIOD",
    "FULL_SIZE_EDGES",
    "INDEX_COLUMNS",
    "MatrixRun",
    "OPTIONAL_STRATEGIES",
    "build_plan",
    "checkpoint_path",
    "eval_command",
    "eval_completion",
    "eval_out_dir",
    "filter_plan",
    "group_completion",
    "group_manifest_path",
    "load_index",
    "load_state",
    "metrics_path",
    "parse_csv",
    "run_dir",
    "run_manifest_path",
    "save_state",
    "sha256_file",
    "step_completion",
    "train_command",
    "upsert_index_row",
    "wall_clock_estimate",
    "write_index",
]
