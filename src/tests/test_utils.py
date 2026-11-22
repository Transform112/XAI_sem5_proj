"""Tests for the portable scaffolding (T2): paths, config, manifest, launcher."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from src.utils.config import Config, hash_config, load_config, load_yaml
from src.utils.manifest import RunManifest, directory_fingerprint, git_info, write_json
from src.utils.paths import PathConfigError, find_project_root, load_paths
from src.utils.seed import describe_determinism_state, seed_everything
from src.utils.torch_support import torch_status


# ---------------------------------------------------------------------------
# Project root and path resolution
# ---------------------------------------------------------------------------
def test_project_root_contains_layout(repo_root):
    root = find_project_root()
    assert root == repo_root
    assert (root / "platforms").is_dir()
    assert (root / "src").is_dir()
    assert (root / "configs" / "paths" / "default.yaml").is_file()


def test_load_paths_default_layout(repo_root):
    paths = load_paths(project_root=repo_root, environ={})
    assert paths.project_root == repo_root
    assert paths.data_root == (repo_root / "data").resolve()
    assert paths.raw_root == paths.data_root / "raw"
    assert paths.audit_root == paths.results_root / "data_audit"
    assert paths.manifests_root == paths.runs_root / "manifests"


def test_env_variables_override_roots(repo_root, tmp_path):
    environ = {
        "XDRIFT_DATA": str(tmp_path / "d"),
        "XDRIFT_RUNS": str(tmp_path / "r"),
        "XDRIFT_CACHE": str(tmp_path / "c"),
        "XDRIFT_RESULTS": str(tmp_path / "res"),
    }
    paths = load_paths(project_root=repo_root, environ=environ)
    assert paths.data_root == tmp_path / "d"
    assert paths.runs_root == tmp_path / "r"
    assert paths.cache_root == tmp_path / "c"
    assert paths.results_root == tmp_path / "res"
    assert paths.audit_root == tmp_path / "res" / "data_audit"


def test_lane_overrides_and_lane_env(repo_root):
    paths = load_paths(lane="kaggle", project_root=repo_root, environ={})
    # Kaggle lane defaults to /kaggle/working roots when no env var is set.
    assert "kaggle" in str(paths.data_root)
    assert paths.lane_env.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    assert paths.to_env()["XDRIFT_RESULTS"] == str(paths.results_root)


def test_unknown_lane_raises(repo_root):
    with pytest.raises(PathConfigError) as excinfo:
        load_paths(lane="not-a-lane", project_root=repo_root, environ={})
    assert "unknown lane" in str(excinfo.value)


def test_paths_reject_unsafe_dataset_name(repo_root):
    paths = load_paths(project_root=repo_root, environ={})
    with pytest.raises(PathConfigError):
        paths.raw_dataset("../escape")


# ---------------------------------------------------------------------------
# Config layering and hashing
# ---------------------------------------------------------------------------
def test_config_layers_base_under_stage(repo_root):
    config = load_config("configs/audit.yaml", project_root=repo_root)
    assert isinstance(config, Config)
    assert config.get("schema_version") == 1
    assert config.get("seed") == 42  # from base.yaml
    assert config.get("dataset") == "NF-UNSW-NB15-v3"  # from audit.yaml
    assert config.path is not None and config.path.name == "audit.yaml"


def test_config_overrides_and_hash_stability(repo_root):
    first = load_config("configs/audit.yaml", project_root=repo_root)
    second = load_config("configs/audit.yaml", project_root=repo_root)
    assert first.sha256 == second.sha256 == hash_config(first.data)
    overridden = load_config(
        "configs/audit.yaml", project_root=repo_root, overrides=["periods.count=2"]
    )
    assert overridden.get("periods")["count"] == 2
    assert overridden.sha256 != first.sha256


def test_config_value_parsing(repo_root):
    config = load_config(
        "configs/audit.yaml",
        project_root=repo_root,
        overrides=["thresholds.nan_inf_fraction_max=0.5", "graph_lift.seeds=[1,2]"],
    )
    assert config.get("thresholds")["nan_inf_fraction_max"] == 0.5
    assert config.get("graph_lift")["seeds"] == [1, 2]


def test_yaml_loader_rejects_missing_file(tmp_path):
    from src.utils.config import ConfigError

    with pytest.raises(ConfigError):
        load_yaml(tmp_path / "nope.yaml")


# ---------------------------------------------------------------------------
# Manifests
# ---------------------------------------------------------------------------
def test_run_manifest_records_reproducibility_fields(repo_root, tmp_path):
    paths = load_paths(project_root=repo_root, environ={})
    config = load_config("configs/audit.yaml", project_root=repo_root)
    out = tmp_path / "manifest.json"
    artifact = write_json(tmp_path / "artifact.json", {"ok": True})
    with RunManifest("unit_test", paths=paths, config=config, seed=42, out_path=out) as manifest:
        manifest.set_dataset("tiny", tmp_path)
        manifest.add_artifact(artifact, role="test")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "success"
    assert payload["config"]["sha256"] == config.sha256
    assert payload["seed"] == 42
    assert payload["git"]["sha"] and len(payload["git"]["sha"]) >= 7
    assert payload["versions"]["pandas"] is not None
    assert payload["artifacts"][0]["role"] == "test"
    assert payload["dataset"]["name"] == "tiny"
    assert payload["started_at_utc"] and payload["finished_at_utc"]


def test_manifest_records_failure(tmp_path):
    out = tmp_path / "failed.json"
    with pytest.raises(RuntimeError):
        with RunManifest("unit_test_fail", out_path=out):
            raise RuntimeError("boom")
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "failed"
    assert "boom" in payload["error"]


def test_directory_fingerprint_detects_change(tmp_path):
    (tmp_path / "a.csv").write_text("x,y\n1,2\n", encoding="utf-8")
    first = directory_fingerprint(tmp_path)
    assert first["file_count"] == 1 and first["fingerprint_sha256"]
    (tmp_path / "a.csv").write_text("x,y\n1,3\n", encoding="utf-8")
    second = directory_fingerprint(tmp_path)
    assert first["fingerprint_sha256"] != second["fingerprint_sha256"]


# ---------------------------------------------------------------------------
# Seed / torch support
# ---------------------------------------------------------------------------
def test_seed_everything_is_repeatable():
    first = seed_everything(7)
    second = seed_everything(7)
    assert first["seed"] == second["seed"] == 7
    assert first["torch"] == second["torch"]
    assert isinstance(describe_determinism_state(), dict)


def test_torch_status_is_cached_and_informative():
    first = torch_status()
    assert torch_status() is first  # lru_cache
    assert set(first) >= {"available", "reason", "version"}


# ---------------------------------------------------------------------------
# Lane launcher
# ---------------------------------------------------------------------------
def test_run_lane_dry_run_plan(repo_root):
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_lane.py"),
            "--root",
            str(repo_root),
            "--lane",
            "legacy",
            "--config",
            "configs/audit.yaml",
            "--entry",
            "src/scripts/run_data_audit.py",
            "--dry-run",
            "--",
            "--dataset",
            "NF-UNSW-NB15-v3",
        ],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=str(repo_root),
    )
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    assert plan["lane"] == "legacy"
    assert plan["seed"] == 42
    assert plan["command"][-2:] == ["--dataset", "NF-UNSW-NB15-v3"]
    assert plan["manifest"].endswith(".json")
    assert "XDRIFT_DATA" in plan["env"]
    assert plan["env"]["XDRIFT_LANE"] == "legacy"  # lane propagates to child processes


def test_run_lane_lists_lanes(repo_root):
    result = subprocess.run(
        [sys.executable, str(repo_root / "scripts" / "run_lane.py"), "--list-lanes", "--root", str(repo_root)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(repo_root),
    )
    assert result.returncode == 0, result.stderr
    for lane in ("legacy", "rtx5060", "kaggle", "cpu"):
        assert lane in result.stdout


def test_run_lane_rejects_unknown_lane(repo_root):
    result = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "run_lane.py"),
            "--root",
            str(repo_root),
            "--lane",
            "nope",
            "--entry",
            "src/scripts/run_data_audit.py",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(repo_root),
    )
    assert result.returncode != 0
    assert "unknown lane" in result.stderr


def test_lane_propagation_cli_env_and_manifest(repo_root, tmp_path):
    """--lane / XDRIFT_LANE must propagate into resolved paths and run manifests."""
    cli_paths = load_paths(lane="legacy", project_root=repo_root, environ={"XDRIFT_LANE": "kaggle"})
    assert cli_paths.lane == "legacy" and cli_paths.lane_source == "cli"
    env_paths = load_paths(project_root=repo_root, environ={"XDRIFT_LANE": "legacy"})
    assert env_paths.lane == "legacy" and env_paths.lane_source == "env"
    assert env_paths.to_env()["XDRIFT_LANE"] == "legacy"
    config = load_config("configs/audit.yaml", project_root=repo_root)
    manifest_path = tmp_path / "lane_manifest.json"
    with RunManifest("lane_test", paths=env_paths, config=config, seed=42, out_path=manifest_path):
        pass
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["lane"] == "legacy"
    assert payload["lane_source"] == "env"


def test_manifest_lifecycle_real_wall_clock_and_env_block(repo_root, tmp_path):
    """T17-S3-2: started/finished/duration must reflect real wall clock + env populated."""
    import time

    env_paths = load_paths(
        project_root=repo_root,
        environ={"XDRIFT_LANE": "legacy", "XDRIFT_DATA": str(tmp_path / "data")},
    )
    config = load_config("configs/audit.yaml", project_root=repo_root)
    manifest_path = tmp_path / "lifecycle.json"
    with RunManifest("unit_test", paths=env_paths, config=config, seed=42, out_path=manifest_path):
        time.sleep(0.05)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["duration_s"] is not None and payload["duration_s"] >= 0.04
    assert payload["started_at_utc"] <= payload["finished_at_utc"]
    # Gate-grade env block records the resolved XDRIFT_* layout.
    assert payload["env"]["XDRIFT_DATA"] == str(env_paths.data_root)
    assert payload["env"]["XDRIFT_LANE"] == "legacy"
    assert payload["git"]["clean_commit"] is not None
    assert payload["code"]["from_dirty_tree"] is not None


def test_gate_manifest_requires_real_lane(repo_root, tmp_path, monkeypatch):
    """T18: gate artifact kinds refuse lane=default unless the dev escape hatch is set."""
    from src.utils.gates import ALLOW_DEFAULT_LANE_ENV, GateRequirementError

    monkeypatch.delenv(ALLOW_DEFAULT_LANE_ENV, raising=False)
    default_paths = load_paths(project_root=repo_root, environ={})
    assert default_paths.lane_source == "default"
    with pytest.raises(GateRequirementError):
        RunManifest("data_audit", paths=default_paths, out_path=tmp_path / "gate_bad.json")
    env_paths = load_paths(project_root=repo_root, environ={"XDRIFT_LANE": "legacy"})
    manifest = RunManifest("data_audit", paths=env_paths, out_path=tmp_path / "gate_ok.json")
    assert manifest.payload["gate"]["real_lane"] is True
    # Non-gate kinds stay usable from a default lane (local development).
    local = RunManifest("unit_test", paths=default_paths, out_path=tmp_path / "local.json")
    assert local.payload["gate"]["real_lane"] is False
    # Explicit escape hatch for local development.
    monkeypatch.setenv(ALLOW_DEFAULT_LANE_ENV, "1")
    escaped = RunManifest("data_audit", paths=default_paths, out_path=tmp_path / "gate_dev.json")
    assert escaped.payload["gate"]["allow_default_lane_env"] is True
    default_paths = load_paths(project_root=repo_root, environ={})
    assert default_paths.lane == "default" and default_paths.lane_source == "default"


def test_no_absolute_paths_in_portable_code(repo_root):
    """src/ and platforms/ code must not hard-code Windows/POSIX absolute paths."""
    import re

    pattern = re.compile(r"[A-Za-z]:\\\\|[A-Za-z]:/|/kaggle/|/home/")
    offenders = []
    for base in ("src", "platforms", "scripts"):
        for path in (repo_root / base).rglob("*"):
            if path.suffix not in {".py", ".sh"} or not path.is_file():
                continue
            for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                stripped = line.strip()
                if stripped.startswith("#") or stripped.startswith("//"):
                    continue
                if pattern.search(stripped):
                    offenders.append(f"{path.relative_to(repo_root)}:{number}: {stripped}")
    assert not offenders, offenders
