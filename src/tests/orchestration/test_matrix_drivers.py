"""Tiny-fixture tests for the T23 matrix/explain batch drivers (no full runs)."""

from __future__ import annotations

import csv
import hashlib
import json
import types
from pathlib import Path

import pytest

from src.orchestration import explain_plan, matrix_plan


class FakePaths:
    """Minimal paths object for scheduler/resume tests (no repo access)."""

    def __init__(self, root: Path):
        self.root = Path(root)
        self.results_root = self.root / "results"
        self.tables_root = self.results_root / "tables"
        self.audit_root = self.results_root / "data_audit"
        self.lane = "legacy"
        self.lane_source = "cli"
        for child in (self.results_root, self.tables_root, self.audit_root):
            child.mkdir(parents=True, exist_ok=True)

    def run_dir(self, *, strategy: str, model: str, seed: int) -> Path:
        return self.root / "runs" / strategy / model / f"seed{int(seed)}"

    def processed_dataset(self, dataset: str) -> Path:
        return self.root / "data" / "processed" / dataset


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_completed_step(paths: FakePaths, run: matrix_plan.MatrixRun, period: int = 1) -> None:
    checkpoint = matrix_plan.checkpoint_path(paths, run, period)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint-bytes-" + str(period).encode())
    metrics = matrix_plan.metrics_path(paths, run, period)
    metrics.write_text('{"ok": true}\n', encoding="utf-8")
    manifest = matrix_plan.run_manifest_path(paths, run, period)
    manifest.write_text(
        json.dumps(
            {
                "status": "success",
                "artifacts": [
                    {"role": "checkpoint", "path": str(checkpoint), "sha256": _sha(checkpoint)}
                ],
            }
        ),
        encoding="utf-8",
    )


def _write_eval(paths: FakePaths, run: matrix_plan.MatrixRun) -> None:
    out_dir = matrix_plan.eval_out_dir(paths, run)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cl_matrix.json").write_text('{"checkpoints_evaluated": [1]}\n', encoding="utf-8")
    (out_dir / "cl_metrics.csv").write_text("strategy,model\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Plan / filters
# ---------------------------------------------------------------------------
def test_build_plan_counts_and_default_grid():
    plan = matrix_plan.build_plan()
    assert len(plan) == len(matrix_plan.DEFAULT_MODELS) * len(matrix_plan.DEFAULT_STRATEGIES) * len(
        matrix_plan.DEFAULT_SEEDS
    )
    assert all(run.periods == 5 for run in plan)
    assert len({run.key for run in plan}) == len(plan)
    assert plan[0].strategy == "ft" and plan[0].model == "egraphsage" and plan[0].seed == 42


def test_build_plan_include_ear_and_filters():
    base = matrix_plan.build_plan(models=["egraphsage"], strategies=["ft"], seeds=[42])
    assert len(base) == 1
    with_ear = matrix_plan.build_plan(models=["egraphsage"], strategies=["ft"], seeds=[42], include_ear=True)
    assert [run.strategy for run in with_ear] == ["ft", "ear_a", "ear_b"]
    filtered = matrix_plan.filter_plan(matrix_plan.build_plan(), max_runs=3)
    assert len(filtered) == 3
    assert matrix_plan.filter_plan(matrix_plan.build_plan(), max_runs=0) == []
    with pytest.raises(ValueError):
        matrix_plan.filter_plan(matrix_plan.build_plan(), max_runs=-1)


def test_parse_csv():
    assert matrix_plan.parse_csv("a,b , c") == ("a", "b", "c")
    assert matrix_plan.parse_csv(None) == ()
    assert matrix_plan.parse_csv("all") == ()
    assert matrix_plan.parse_csv("42,43", cast=int) == (42, 43)


def test_train_command_is_explicit_and_lane_tagged():
    run = matrix_plan.build_plan(models=["gcn"], strategies=["ewc"], seeds=[43])[0]
    command = matrix_plan.train_command(run, 2, dataset="NF-UNSW-NB15-v3", lane="rtx5060", device="cuda")
    assert "src/scripts/run_train.py" in command[0] or command[1].endswith("run_train.py")
    assert "--period" in command and command[command.index("--period") + 1] == "2"
    assert command[command.index("--strategy") + 1] == "ewc"
    assert command[command.index("--model") + 1] == "gcn"
    assert "--lane" in command and command[command.index("--lane") + 1] == "rtx5060"


# ---------------------------------------------------------------------------
# Resume / completion checks
# ---------------------------------------------------------------------------
def test_step_completion_and_staleness(tmp_path):
    paths = FakePaths(tmp_path)
    run = matrix_plan.build_plan(models=["egraphsage"], strategies=["ft"], seeds=[42])[0]
    assert matrix_plan.step_completion(paths, run, 1)["status"] == "missing"
    _write_completed_step(paths, run, 1)
    state = matrix_plan.step_completion(paths, run, 1)
    assert state["status"] == "complete" and state["verified"] is True
    # Tamper with the checkpoint -> stale, must be rerun.
    matrix_plan.checkpoint_path(paths, run, 1).write_bytes(b"tampered")
    assert matrix_plan.step_completion(paths, run, 1)["status"] == "stale"


def test_group_completion_partial_and_complete(tmp_path):
    paths = FakePaths(tmp_path)
    run = matrix_plan.build_plan(models=["egraphsage"], strategies=["ft"], seeds=[42])[0]
    _write_completed_step(paths, run, 1)
    group = matrix_plan.group_completion(paths, run)
    assert group["status"] == "partial" and group["pending_periods"] == [2, 3, 4, 5]
    for period in (2, 3, 4, 5):
        _write_completed_step(paths, run, period)
    assert matrix_plan.group_completion(paths, run)["status"] == "partial"  # eval missing
    _write_eval(paths, run)
    group = matrix_plan.group_completion(paths, run)
    assert group["status"] == "complete" and group["eval"]["status"] == "complete"


def test_eval_completion_accepts_t30_nested_evaluation_schema(tmp_path):
    """T30 writer stores checkpoints_evaluated under evaluation.{...}; resume must see it."""
    paths = FakePaths(tmp_path)
    run = matrix_plan.build_plan(models=["egraphsage"], strategies=["ft"], seeds=[42])[0]
    out_dir = matrix_plan.eval_out_dir(paths, run)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "evaluation": {
            "mode": "edge_batch",
            "peak_memory_allocated_bytes": 1082511360,
            "wall_clock_seconds": 14.75,
            "checkpoints_evaluated": [1, 2, 3, 4, 5],
        }
    }
    (out_dir / "cl_matrix.json").write_text(json.dumps(payload), encoding="utf-8")
    (out_dir / "cl_metrics.csv").write_text("strategy,model\n", encoding="utf-8")
    state = matrix_plan.eval_completion(paths, run)
    assert state["status"] == "complete"
    assert state["checkpoints"] == [1, 2, 3, 4, 5]


def test_state_and_index_roundtrip(tmp_path):
    paths = FakePaths(tmp_path)
    state_path = paths.tables_root / "matrix_state.json"
    index_path = paths.tables_root / "matrix_index.csv"
    state = matrix_plan.load_state(state_path)
    assert state["groups"] == {}
    state["groups"]["ft_egraphsage_seed42"] = {"status": "complete"}
    matrix_plan.save_state(state_path, state)
    assert matrix_plan.load_state(state_path)["groups"]["ft_egraphsage_seed42"]["status"] == "complete"
    run = matrix_plan.build_plan(models=["egraphsage"], strategies=["ft"], seeds=[42])[0]
    rows = matrix_plan.load_index(index_path)
    matrix_plan.upsert_index_row(rows, run=run, kind="train", period=1, status="complete")
    matrix_plan.upsert_index_row(rows, run=run, kind="eval", period=0, status="complete")
    matrix_plan.write_index(index_path, rows)
    reloaded = matrix_plan.load_index(index_path)
    assert set(reloaded) == {("ft_egraphsage_seed42", "train", 1), ("ft_egraphsage_seed42", "eval", 0)}
    with index_path.open("r", encoding="utf-8", newline="") as handle:
        header = next(csv.reader(handle))
    assert header == matrix_plan.INDEX_COLUMNS


# ---------------------------------------------------------------------------
# Explain batch planning
# ---------------------------------------------------------------------------
def test_build_jobs_counts_and_commands():
    plan = matrix_plan.build_plan(models=["egraphsage"], strategies=["ft"], seeds=[42], periods=2)
    anchor_jobs, explain_jobs = explain_plan.build_jobs(plan, periods=2, explainers=("ig", "gnn_explainer"), k_values=(5, 10))
    assert len(anchor_jobs) == 2 and len(explain_jobs) == 2
    command = explain_plan.explain_command(explain_jobs[0], dataset="NF-UNSW-NB15-v3", lane="rtx5060")
    assert "--explainers" in command and command[command.index("--k") + 1] == "5,10"
    anchor_command = explain_plan.anchors_command(anchor_jobs[0], dataset="NF-UNSW-NB15-v3", lane="rtx5060")
    assert "--period" in anchor_command
    assert anchor_command[anchor_command.index("--period") + 1] == "1"


def test_anchor_manifest_resume_check(tmp_path):
    paths = FakePaths(tmp_path)
    plan = matrix_plan.build_plan(models=["egraphsage"], strategies=["ft"], seeds=[42], periods=1)
    anchor_jobs, _ = explain_plan.build_jobs(plan, periods=1)
    job = anchor_jobs[0]
    assert explain_plan.anchors_manifest_check(paths, job)["status"] == "missing"
    anchor_file = explain_plan.anchor_file(paths, job)
    anchor_file.parent.mkdir(parents=True, exist_ok=True)
    anchor_file.write_bytes(b"anchor-bytes")
    manifest = explain_plan.anchor_manifest_path(paths, job)
    manifest.write_text(
        json.dumps(
            {
                "anchor_files": {"1": {"path": str(anchor_file), "sha256": _sha(anchor_file)}},
                "per_period": {"1": {"n_selected": 17}},
            }
        ),
        encoding="utf-8",
    )
    state = explain_plan.anchors_manifest_check(paths, job)
    assert state["status"] == "complete" and state["n_selected"] == 17
    anchor_file.write_bytes(b"tampered")
    assert explain_plan.anchors_manifest_check(paths, job)["status"] == "stale"


def test_explain_completion_cache_aware(tmp_path):
    paths = FakePaths(tmp_path)
    plan = matrix_plan.build_plan(models=["egraphsage"], strategies=["ft"], seeds=[42], periods=1)
    anchor_jobs, explain_jobs = explain_plan.build_jobs(plan, periods=1, explainers=("ig", "gnn_explainer"), k_values=(5, 10))
    run = plan[0]
    checkpoint = explain_plan.checkpoint_path(paths, run, 1)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"checkpoint")
    checkpoint_sha = _sha(checkpoint)
    index = explain_plan.canonical_explain_index(paths)
    rows = []
    for anchor in range(3):
        for k in (5, 10):
            rows.append({"checkpoint_sha": checkpoint_sha, "explainer": "ig", "k": k, "anchor_id": anchor})
        rows.append({"checkpoint_sha": checkpoint_sha, "explainer": "gnn_explainer", "k": 0, "anchor_id": anchor})
    with index.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["checkpoint_sha", "explainer", "k", "anchor_id"])
        writer.writeheader()
        writer.writerows(rows)
    state = explain_plan.explain_completion(paths, explain_jobs[0], expected_anchors=3)
    assert state["status"] == "complete"
    assert state["counts"] == {"ig": 6, "gnn_explainer": 3}
    partial = explain_plan.explain_completion(paths, explain_jobs[0], expected_anchors=5)
    assert partial["status"] == "partial"


def test_wall_clock_formula_sanity():
    plan = matrix_plan.build_plan()
    counts = {1: 1_131_909, 2: 304_675, 3: 304_675, 4: 304_675, 5: 304_675}
    estimate = matrix_plan.wall_clock_estimate(plan, epoch_seconds_full_size=2.0, period_edge_counts=counts)
    assert estimate["n_runs"] == len(plan)
    assert estimate["total_seconds"] > 0
    assert estimate["per_run_seconds_mean"] > 0
    expensive = matrix_plan.wall_clock_estimate(plan, epoch_seconds_full_size=4.0, period_edge_counts=counts)
    assert expensive["total_seconds"] == pytest.approx(2 * estimate["total_seconds"], rel=1e-4)
    anchor_jobs, explain_jobs = explain_plan.build_jobs(plan)
    explain_estimate = explain_plan.wall_clock_estimate(anchor_jobs, explain_jobs)
    assert explain_estimate["anchors_total_estimate"] == len(anchor_jobs) * explain_plan.DEFAULT_ANCHORS_PER_PERIOD
    assert explain_estimate["total_seconds"] > 0


# ---------------------------------------------------------------------------
# CLI dry-runs (no execution)
# ---------------------------------------------------------------------------
def test_run_matrix_dry_run_default_grid(capsys):
    import scripts.run_matrix as run_matrix

    code = run_matrix.main(["--models", "egraphsage,gcn", "--strategies", "ft,replay,ewc", "--seeds", "42,43,44,45,46", "--dry-run"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["n_groups"] == 30
    assert payload["summary"]["n_period_updates"] == 150
    assert payload["summary"]["wall_clock"]["n_runs"] == 30


def test_run_matrix_dry_run_filters(capsys):
    import scripts.run_matrix as run_matrix

    code = run_matrix.main(["--models", "gcn", "--strategies", "ft", "--seeds", "42", "--periods", "2", "--max-runs", "1", "--dry-run"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["n_groups"] == 1
    assert payload["summary"]["n_period_updates"] == 2


def test_run_anchors_explain_dry_run(capsys):
    import scripts.run_anchors_explain as driver

    code = driver.main(["--models", "egraphsage", "--strategies", "ft", "--seeds", "42", "--periods", "1", "--max-jobs", "1", "--dry-run"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["summary"]["n_anchor_jobs"] == 1
    assert payload["summary"]["n_explain_jobs"] == 1
    assert payload["summary"]["wall_clock"]["total_seconds"] > 0


# ---------------------------------------------------------------------------
# T26: measured epoch anchor is mandatory for real runs; dry-run budget guards
# ---------------------------------------------------------------------------
def test_matrix_dry_run_uses_placeholder_anchor_with_loud_warning(capsys):
    import scripts.run_matrix as run_matrix

    code = run_matrix.main(["--models", "egraphsage", "--strategies", "ft", "--seeds", "42", "--periods", "1", "--dry-run"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    budget = payload["budget"]
    assert budget["anchor_source"] == "placeholder"
    assert budget["placeholder_anchor"] is True
    assert any("PLACEHOLDER_ANCHOR" in warning for warning in budget["warnings"])
    assert "by_strategy_gpu_hours" in budget and "by_model_gpu_hours" in budget
    assert set(budget["by_strategy_gpu_hours"]) == {"ft"}
    assert set(budget["by_model_gpu_hours"]) == {"egraphsage"}


def test_matrix_dry_run_measured_anchor_and_overrun_warning(capsys):
    import scripts.run_matrix as run_matrix

    code = run_matrix.main(
        ["--models", "egraphsage", "--strategies", "ft", "--seeds", "42", "--periods", "1",
         "--epoch-seconds-full-size", "60000", "--dry-run"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    budget = payload["budget"]
    assert budget["anchor_source"] == "flag" and budget["placeholder_anchor"] is False
    assert budget["anchor_seconds"] == 60000
    assert budget["over_budget"] is True
    assert any("BUDGET_OVERRUN" in warning for warning in budget["warnings"])


def test_matrix_anchor_can_be_derived_from_metrics_file(capsys, tmp_path):
    import scripts.run_matrix as run_matrix

    metrics = tmp_path / "train_metrics.json"
    metrics.write_text(json.dumps({"timing": {"epoch_seconds_mean": 42.5}}), encoding="utf-8")
    code = run_matrix.main(
        ["--models", "egraphsage", "--strategies", "ft", "--seeds", "42", "--periods", "1",
         "--anchor-metrics", str(metrics), "--dry-run"]
    )
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["budget"]["anchor_source"].startswith("metrics:")
    assert payload["budget"]["anchor_seconds"] == 42.5


def test_matrix_real_run_refuses_missing_measured_anchor(capsys):
    import scripts.run_matrix as run_matrix

    code = run_matrix.main(
        ["--models", "egraphsage", "--strategies", "ft", "--seeds", "42", "--periods", "1", "--lane", "legacy"]
    )
    assert code == 2
    captured = capsys.readouterr()
    assert "measured epoch anchor is REQUIRED" in captured.err


def test_wall_clock_grouping_sums_to_total():
    plan = matrix_plan.build_plan()
    estimate = matrix_plan.wall_clock_estimate(plan, epoch_seconds_full_size=20.0)
    by_strategy = sum(estimate["by_strategy_gpu_hours"].values())
    by_model = sum(estimate["by_model_gpu_hours"].values())
    assert by_strategy == pytest.approx(estimate["total_gpu_hours"], rel=1e-3)
    assert by_model == pytest.approx(estimate["total_gpu_hours"], rel=1e-3)
    assert set(estimate["by_strategy_gpu_hours"]) == {"ft", "replay", "ewc"}
    assert set(estimate["by_model_gpu_hours"]) == {"egraphsage", "gcn"}
    assert estimate["over_budget"] is False
