#!/usr/bin/env python
"""XDRIFT Kaggle full-run kernel template (T37, D-065/D-066).

`src/kaggle_lane/run_full.py` substitutes the ``noise`` and ``mysteriousavailable/xdrift-noise``
markers and pushes the generated single-file script as a private Kaggle kernel:

* stage ``matrix`` — the full base strategy matrix (resumable via a resume dataset);
* stage ``explain`` — S6 anchors + S7 IG/GNNExplainer (+ optional S8 noise caches),
  staged by jobs with cache/resume.

Inputs (mounted read-only under /kaggle/input):
* the private inputs dataset (raw CSV, frozen C1 artifacts, code snapshot, env manifest);
* the run control dataset (``runctl.json``: stage parameters, dataset version, commit);
* an optional resume dataset (``resume/results`` + ``resume/runs`` from earlier pushes).

The kernel verifies every hash it can (frozen protocol D-042, CSV, code snapshot),
runs the committed code unmodified from the code snapshot, writes a compact summary
JSON and zips evidence artifacts (results + runs + logs) into /kaggle/working.
The stage tree (including large processed graphs) is removed after zipping.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

STAGE_KIND = "noise"
KERNEL_REF = "mysteriousavailable/xdrift-noise"
WORK_ROOT = Path("/kaggle/working")
INPUT_ROOT = Path("/kaggle/input")
STAGE = WORK_ROOT / "stage"
LOGS_DIR = WORK_ROOT / "logs"
SUMMARY_PATH = WORK_ROOT / "run_summary.json"
OUTPUT_ZIP = WORK_ROOT / "xdrift_run_output.zip"
EXPECTED_CSV_BYTES = 577_360_958
REQUIRED_PYG = "torch-geometric==2.8.1"

STARTED_AT = datetime.now(timezone.utc)
SUMMARY: dict[str, Any] = {
    "kind": "xdrift-kaggle-run",
    "stage": STAGE_KIND,
    "kernel_ref": KERNEL_REF,
    "schema_version": 1,
    "started_at_utc": STARTED_AT.isoformat(timespec="seconds"),
    "status": "running",
    "steps": {},
    "input_verification": {},
    "environment": {},
    "runctl": {},
    "resume": {},
    "matrix": {},
    "explain": {},
    "noise_floor": {},
    "artifacts": {},
    "risks": [],
}
STEP_RECORDS: list[dict[str, Any]] = []


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[xdrift-{STAGE_KIND}] {message}", flush=True)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def run_step(
    name: str,
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str],
    timeout: int,
    allow_failure: bool = False,
    tail_lines: int = 30,
) -> dict[str, Any]:
    started = time.monotonic()
    log(f"STEP {name}: {' '.join(str(part) for part in command)}")
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOGS_DIR / f"{name}.log"
    try:
        completed = subprocess.run(
            [str(part) for part in command],
            cwd=str(cwd),
            env=dict(env),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
        output = completed.stdout or ""
        returncode = completed.returncode
    except subprocess.TimeoutExpired as exc:
        output = exc.stdout if isinstance(exc.stdout, str) else ""
        returncode = -9
        output += f"\n[TIMEOUT after {timeout}s]\n"
    duration = round(time.monotonic() - started, 2)
    log_path.write_text(output, encoding="utf-8")
    tail = "\n".join(output.strip().splitlines()[-tail_lines:])
    print("\n".join(f"[{name}] {line}" for line in tail.splitlines()), flush=True)
    record = {
        "command": [str(part) for part in command],
        "returncode": int(returncode),
        "duration_s": duration,
        "log": str(log_path),
        "tail": tail[-6000:],
    }
    STEP_RECORDS.append({"step": name, **record})
    SUMMARY["steps"][name] = record
    if returncode != 0 and not allow_failure:
        raise RuntimeError(f"step {name!r} failed with exit code {returncode} (see {log_path})")
    return record


def find_input(pattern: str) -> Path:
    matches = sorted(INPUT_ROOT.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"input {pattern!r} not found under {INPUT_ROOT}")
    return matches[0]


def extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(destination)


def load_runctl() -> dict[str, Any]:
    path = find_input("runctl.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"runctl is not a JSON object: {path}")
    SUMMARY["runctl"] = {"path": str(path), "stage": payload.get("stage"), "run_id": payload.get("run_id"), "dataset_version": payload.get("dataset_version"), "commit": payload.get("commit")}
    return payload


def verify_and_stage_inputs(frozen_fallback_entries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    env_manifest_path = find_input("env_manifest.json")
    env_manifest = json.loads(env_manifest_path.read_text(encoding="utf-8"))
    dataset_root = env_manifest_path.parent
    frozen_fallback = [entry for entry in (env_manifest.get("frozen_zip") or {}).get("files") or [] if entry.get("path") and entry.get("sha256")]
    SUMMARY["input_verification"]["env_manifest"] = str(env_manifest_path)
    SUMMARY["input_verification"]["dataset_slug"] = env_manifest.get("dataset_slug")
    SUMMARY["input_verification"]["commit"] = env_manifest.get("git_commit")

    csv_source = find_input("NF-UNSW-NB15-v3.csv")
    csv_bytes = csv_source.stat().st_size
    csv_sha = sha256_file(csv_source)
    if csv_bytes != int(env_manifest["csv"]["bytes"]) or csv_sha != env_manifest["csv"]["sha256"]:
        raise RuntimeError(f"raw CSV mismatch: {csv_bytes}/{csv_sha} != manifest")
    if csv_bytes != EXPECTED_CSV_BYTES:
        raise RuntimeError(f"raw CSV size {csv_bytes} != {EXPECTED_CSV_BYTES}")

    dataset_version = SUMMARY["runctl"].get("dataset_version")
    manifest_commit = env_manifest.get("git_commit")
    runctl_commit = SUMMARY["runctl"].get("commit")
    if runctl_commit and manifest_commit and str(runctl_commit) != str(manifest_commit):
        raise RuntimeError(f"runctl commit {runctl_commit} != dataset code snapshot {manifest_commit}")

    code_dirs = sorted(p for p in dataset_root.iterdir() if p.is_dir() and p.name.startswith("xdrift_repo_"))
    code_zips = sorted(dataset_root.glob("xdrift_repo_*.zip"))
    frozen_dir = dataset_root / "frozen"
    frozen_zips = sorted(dataset_root.glob("frozen.zip"))
    if not (code_dirs or code_zips):
        raise RuntimeError("no code snapshot in input dataset")
    if not (frozen_dir.is_dir() or frozen_zips):
        raise RuntimeError("no frozen artifacts in input dataset")

    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True, exist_ok=True)
    if code_dirs:
        shutil.copytree(code_dirs[0], STAGE, dirs_exist_ok=True)
        code_delivery = "extracted_tree"
        code_name = code_dirs[0].name
    else:
        extract_zip(code_zips[0], STAGE)
        code_delivery = "zip"
        code_name = code_zips[0].name
    code_files = [path for path in STAGE.rglob("*") if path.is_file()]
    code_tree = hashlib.sha256()
    for path in sorted(code_files):
        code_tree.update(path.relative_to(STAGE).as_posix().encode("utf-8"))
        code_tree.update(b"\0")
        code_tree.update(sha256_file(path).encode("ascii"))
        code_tree.update(b"\n")

    if frozen_dir.is_dir():
        for path in sorted(frozen_dir.rglob("*")):
            if path.is_file():
                target = STAGE / path.relative_to(frozen_dir)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        frozen_delivery = "extracted_tree"
        frozen_root = frozen_dir
    else:
        extract_zip(frozen_zips[0], STAGE)
        frozen_delivery = "zip"
        frozen_root = frozen_zips[0]
    manifest_source = dataset_root / "frozen_artifacts.sha256"
    if manifest_source.is_file():
        shutil.copy2(manifest_source, STAGE / "frozen_artifacts.sha256")
    expected: dict[str, str] = {}
    manifest_path = STAGE / "frozen_artifacts.sha256"
    if manifest_path.is_file():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "  " in line:
                digest, rel = line.split("  ", 1)
                expected[rel.strip()] = digest.strip()
    if not expected:
        expected = {str(entry["path"]): str(entry["sha256"]) for entry in (frozen_fallback or frozen_fallback_entries)}
    problems = [rel for rel, digest in sorted(expected.items()) if not (STAGE / rel).is_file() or sha256_file(STAGE / rel) != digest]
    if problems:
        raise RuntimeError("frozen verification failed: " + "; ".join(problems[:5]))
    SUMMARY["input_verification"].update(
        {
            "csv_bytes": csv_bytes,
            "csv_sha256": csv_sha,
            "csv_verified": True,
            "code_delivery": code_delivery,
            "code_name": code_name,
            "code_files": len(code_files),
            "code_tree_sha256": code_tree.hexdigest(),
            "frozen_delivery": frozen_delivery,
            "frozen_files_verified": len(expected),
            "dataset_version": dataset_version,
        }
    )
    log(f"inputs verified: csv={csv_bytes}B code={code_name} frozen={len(expected)} files")

    raw_dir = STAGE / "data" / "raw" / "NF-UNSW-NB15-v3"
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(csv_source, raw_dir / "NF-UNSW-NB15-v3.csv")
    (STAGE / "results").mkdir(parents=True, exist_ok=True)
    (STAGE / "runs").mkdir(parents=True, exist_ok=True)
    return env_manifest


def restore_resume() -> dict[str, Any]:
    candidates = [path for path in INPUT_ROOT.rglob("resume") if path.is_dir() and ((path / "results").is_dir() or (path / "runs").is_dir())]
    if not candidates:
        # Kaggle may deliver the resume dataset as a raw zip rather than an extracted tree.
        resume_zips = sorted(INPUT_ROOT.rglob("resume.zip"))
        if resume_zips:
            unpack_dir = STAGE / "_resume_unzip"
            if unpack_dir.exists():
                shutil.rmtree(unpack_dir)
            extract_zip(resume_zips[0], unpack_dir)
            source = unpack_dir / "resume" if (unpack_dir / "resume").is_dir() else unpack_dir
            if not ((source / "results").is_dir() or (source / "runs").is_dir()):
                candidates = [path for path in unpack_dir.rglob("resume") if path.is_dir() and ((path / "results").is_dir() or (path / "runs").is_dir())]
                source = candidates[0] if candidates else None
            if source is None:
                SUMMARY["resume"] = {"found": False}
                return SUMMARY["resume"]
            candidates = [source]
    source = candidates[0]
    copied_results = copied_runs = 0
    for name in ("results", "runs"):
        root = source / name
        if not root.is_dir():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            target = STAGE / name / path.relative_to(root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            if name == "results":
                copied_results += 1
            else:
                copied_runs += 1
    state_path = STAGE / "results" / "tables" / "matrix_state.json"
    state = {}
    if state_path.is_file():
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            state = {}
    completed = [key for key, value in (state.get("groups") or {}).items() if isinstance(value, dict) and value.get("status") == "complete"]
    SUMMARY["resume"] = {"found": True, "path": str(source), "results_files": copied_results, "run_files": copied_runs, "completed_groups": sorted(completed)}
    log(f"resume restored: {copied_results} results files, {copied_runs} run files, {len(completed)} completed groups")
    return SUMMARY["resume"]


def prepare_environment() -> tuple[dict[str, str], dict[str, Any]]:
    env = dict(os.environ)
    env.update(
        {
            "XDRIFT_ROOT": str(STAGE),
            "XDRIFT_DATA": str(STAGE / "data"),
            "XDRIFT_RUNS": str(STAGE / "runs"),
            "XDRIFT_CACHE": str(STAGE / "cache"),
            "XDRIFT_RESULTS": str(STAGE / "results"),
            "XDRIFT_LANE": "kaggle",
            "PYTHONPATH": str(STAGE),
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
        }
    )
    os.environ.update({key: env[key] for key in env if key.startswith(("XDRIFT_", "CUBLAS", "PYTORCH_", "PYTHONPATH"))})
    SUMMARY["environment"]["env_vars"] = {key: env[key] for key in env if key.startswith(("XDRIFT_", "CUBLAS", "PYTORCH_", "PYTHONPATH"))}
    probe = (
        "import json, platform, torch, torch_geometric\n"
        "info = {'python': platform.python_version(), 'torch': torch.__version__,"
        "'torch_cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),"
        "'pyg': torch_geometric.__version__}\n"
        "if torch.cuda.is_available():\n"
        "    info['device'] = torch.cuda.get_device_name(0)\n"
        "    info['capability'] = list(torch.cuda.get_device_capability(0))\n"
        "    info['device_count'] = torch.cuda.device_count()\n"
        "    info['total_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2)\n"
        "print(json.dumps(info))\n"
    )
    check = subprocess.run([sys.executable, "-c", "import torch_geometric"], cwd=str(STAGE), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120)
    if check.returncode != 0:
        run_step("pip_pyg", [sys.executable, "-m", "pip", "install", "--quiet", REQUIRED_PYG], cwd=STAGE, env=env, timeout=1200)
    completed = subprocess.run([sys.executable, "-c", probe], cwd=str(STAGE), env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300)
    print(completed.stdout or "", flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"environment probe failed: {completed.stdout}")
    info = json.loads([line for line in completed.stdout.splitlines() if line.startswith("{")][-1])
    SUMMARY["environment"].update(info)
    run_step("check_env", [sys.executable, "src/scripts/check_env.py", "--require-cuda", "--json-out", str(STAGE / "results" / "env_check_kaggle.json")], cwd=STAGE, env=env, timeout=900)
    env_check = STAGE / "results" / "env_check_kaggle.json"
    if env_check.is_file():
        SUMMARY["environment"]["check_env"] = json.loads(env_check.read_text(encoding="utf-8"))
    return env, SUMMARY["environment"]


def ensure_graphs(env: Mapping[str, str], *, rebuild: bool = False) -> dict[str, Any]:
    dataset = "NF-UNSW-NB15-v3"
    processed = STAGE / "data" / "processed" / dataset
    manifest = processed / "graphs_manifest.json"
    if rebuild or not manifest.is_file():
        run_step("build_graphs", [sys.executable, "src/scripts/build_graphs.py", "--lane", "kaggle"], cwd=STAGE, env=env, timeout=7200)
    payload = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
    periods = {}
    for path in sorted(processed.glob("period_*.pt")):
        periods[path.stem.split("_")[1]] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    SUMMARY["graph_build"] = {"processed_dir": str(processed), "manifest": payload, "periods": periods}
    return SUMMARY["graph_build"]


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _timing_from_manifests() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audit = STAGE / "results" / "data_audit"
    for path in sorted(audit.glob("train_manifest_*.json")):
        payload = _read_json(path) or {}
        extra = payload.get("extra") or {}
        timing = extra.get("timing") or {}
        rows.append(
            {
                "manifest": path.name,
                "strategy": extra.get("strategy"),
                "model": extra.get("model"),
                "period": extra.get("period"),
                "device": extra.get("device"),
                "timing": timing,
                "max_memory_allocated_bytes": timing.get("max_memory_allocated_bytes"),
                "max_memory_allocated_gib": round((timing.get("max_memory_allocated_bytes") or 0) / 1024 ** 3, 3) if timing.get("max_memory_allocated_bytes") else None,
                "epoch_seconds_mean": timing.get("epoch_seconds_mean"),
                "updates_per_epoch": timing.get("updates_per_epoch"),
            }
        )
    return rows


def _metrics_from_manifests() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    audit = STAGE / "results" / "data_audit"
    for pattern in ("train_*.json", "eval_manifest_*.json", "matrix_manifest_*.json"):
        for path in sorted(audit.glob(pattern)):
            payload = _read_json(path) or {}
            extra = payload.get("extra") or {}
            rows.append(
                {
                    "manifest": path.name,
                    "kind": payload.get("kind"),
                    "duration_s": payload.get("duration_s"),
                    "extra": {key: extra.get(key) for key in ("strategy", "model", "seed", "period", "run_key", "status", "eval_complete", "checkpoints", "timing") if key in extra},
                }
            )
    return rows


def run_matrix_stage(runctl: Mapping[str, Any], env: Mapping[str, str]) -> None:
    cfg = dict(runctl.get("matrix") or {})
    models = ",".join(str(item) for item in cfg.get("models", ["egraphsage", "gcn"]))
    strategies = ",".join(str(item) for item in cfg.get("strategies", ["ft", "replay", "ewc"]))
    seeds = ",".join(str(item) for item in cfg.get("seeds", [42, 43, 44, 45, 46]))
    periods = int(cfg.get("periods", 5))
    anchor = float(cfg.get("epoch_seconds_full_size", 2.4021))
    max_runs = cfg.get("max_runs")
    rebuild_graphs = bool(cfg.get("rebuild_graphs", False))
    SUMMARY["matrix"]["config"] = {**cfg, "models": models.split(","), "strategies": strategies.split(","), "seeds": seeds.split(",")}
    ensure_graphs(env, rebuild=rebuild_graphs)
    command = [
        sys.executable, "src/scripts/run_matrix.py",
        "--lane", "kaggle", "--device", "cuda",
        "--config", str(cfg.get("config") or "configs/train.yaml"),
        "--models", models, "--strategies", strategies, "--seeds", seeds,
        "--periods", str(periods), "--epoch-seconds-full-size", repr(anchor),
    ]
    if cfg.get("include_ear"):
        command.append("--include-ear")
    if max_runs:
        command.extend(["--max-runs", str(int(max_runs))])
    if cfg.get("eval_chunk_edges"):
        command.extend(["--eval-chunk-edges", str(int(cfg["eval_chunk_edges"]))])
    if cfg.get("memory_guard"):
        command.extend(["--memory-guard", str(cfg["memory_guard"])])
    result = run_step("run_matrix", command, cwd=STAGE, env=env, timeout=int(cfg.get("timeout_s", 10 * 3600)), allow_failure=True)
    state = _read_json(STAGE / "results" / "tables" / "matrix_state.json") or {}
    groups = state.get("groups") or {}
    complete = sorted(key for key, value in groups.items() if isinstance(value, dict) and value.get("status") == "complete")
    failed = sorted(key for key, value in groups.items() if isinstance(value, dict) and value.get("status") not in {"complete"})
    SUMMARY["matrix"].update(
        {
            "groups_planned": len(groups) or None,
            "groups_complete": len(complete),
            "groups_complete_keys": complete,
            "groups_not_complete": failed,
            "matrix_state": state,
            "train_timings": _timing_from_manifests(),
            "manifests": _metrics_from_manifests(),
            "returncode": result["returncode"],
            "duration_s": result["duration_s"],
        }
    )
    eval_artifacts: list[dict[str, Any]] = []
    for table_dir in sorted((STAGE / "results" / "tables" / "matrix").glob("*")):
        matrix_json = table_dir / "cl_matrix.json"
        if matrix_json.is_file():
            payload = _read_json(matrix_json) or {}
            rows = (payload.get("matrix") or {}).get("all_classes") or []
            counts = [sum(1 for value in row if isinstance(value, (int, float)) and value == value) for row in rows]
            eval_artifacts.append({"run_key": table_dir.name, "cl_matrix": str(matrix_json), "n_periods": len(rows), "defined_cells": sum(counts), "dual_macro": payload.get("cl_metrics_all_classes") is not None})
    SUMMARY["matrix"]["eval_artifacts"] = eval_artifacts
    if failed or (result["returncode"] != 0 and not complete):
        raise RuntimeError(f"matrix stage: {len(failed)} incomplete group(s) of {len(groups)}; failed={failed[:5]}")


def _explain_groups(cfg: Mapping[str, Any]) -> list[tuple[str, str, int]]:
    """Normalize runctl explain groups to (strategy, model, seed) tuples.

    Accepted forms: a 3-list of selections ``[[strategies], [models], [seeds]]``
    (cross product), an explicit list of ``[strategy, model, seed]`` triples, or the
    legacy single strategy/model/seed keys.
    """
    raw = cfg.get("groups")
    selections = [raw[0], raw[1], raw[2]] if isinstance(raw, (list, tuple)) and len(raw) == 3 and all(isinstance(item, (list, tuple)) for item in raw) else None
    if selections is not None:
        strategies = [str(item) for item in selections[0]]
        models = [str(item) for item in selections[1]]
        seeds = [int(item) for item in selections[2]]
        return [(strategy, model, seed) for strategy in strategies for model in models for seed in seeds]
    if raw and all(isinstance(item, (list, tuple)) and len(item) == 3 for item in raw):
        return [(str(item[0]), str(item[1]), int(item[2])) for item in raw]
    strategies = [str(item) for item in (cfg.get("strategies") or [cfg.get("strategy", "ft")])]
    models = [str(item) for item in (cfg.get("models") or [cfg.get("model", "egraphsage")])]
    seeds = [int(item) for item in (cfg.get("seeds") or [cfg.get("seed", 42)])]
    return [(strategy, model, seed) for strategy in strategies for model in models for seed in seeds]


def _group_key(strategy: str, model: str, seed: int) -> str:
    return f"{strategy}_{model}_seed{seed}"


def _apply_anchor_config(runctl: Mapping[str, Any]) -> Path:
    """Apply runctl anchor caps to configs/eval.yaml (the T23 driver uses the default eval config).

    eval.yaml nests the anchor caps under ``s6.anchors``; overriding a root ``anchors``
    key has no effect on ``build_anchors.py``. Update the real nested section and keep a
    root-level copy for compatibility.
    """
    base = STAGE / "configs" / "eval.yaml"
    import yaml  # available in the Kaggle image / check_env

    original = base.read_text(encoding="utf-8")
    config = yaml.safe_load(original)
    overrides = dict(runctl.get("explain") or {}).get("anchors") or {}
    if overrides:
        s6 = config.get("s6")
        if not isinstance(s6, dict):
            s6 = {}
            config["s6"] = s6
        anchors = s6.setdefault("anchors", {})
        if not isinstance(anchors, dict):
            anchors = {}
            s6["anchors"] = anchors
        for key, value in overrides.items():
            anchors[str(key)] = value
        config.setdefault("anchors", {}).update({str(key): value for key, value in overrides.items()})
        backup = STAGE / "results" / "data_audit" / "eval_config_kaggle.yaml"
        backup.parent.mkdir(parents=True, exist_ok=True)
        backup.write_text(original, encoding="utf-8")
        base.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
        SUMMARY["explain"]["anchor_config_override"] = {
            "path": str(base),
            "backup": str(backup),
            "section": "s6.anchors",
            "overrides": {str(key): value for key, value in overrides.items()},
        }
    return base


def _backfill_canonical_s5(groups: Sequence[tuple[str, str, int]], env: Mapping[str, str]) -> dict[str, Any]:
    """Ensure top-level canonical S5 artifacts exist (validators s6/s7 are cumulative over S5)."""
    tables = STAGE / "results" / "tables"
    needed = ("cl_matrix.json", "cl_metrics.csv", "cl_per_class_f1.json")
    if all((tables / name).is_file() for name in needed):
        return {"status": "present", "tables": str(tables)}
    for strategy, model, seed in list(groups) + [("ft", "egraphsage", 42)]:
        group_dir = tables / "matrix" / _group_key(strategy, model, seed)
        if not (group_dir / "cl_matrix.json").is_file():
            continue
        for name in needed:
            source = group_dir / name
            if source.is_file():
                shutil.copy2(source, tables / name)
        return {"status": "copied_from_group", "group": _group_key(strategy, model, seed), "tables": str(tables)}
    # last resort: evaluate the first group directly (writes the top-level canonical artifacts)
    strategy, model, seed = groups[0]
    run_step(
        "canonical_s5_eval",
        [sys.executable, "src/scripts/run_eval.py", "--lane", "kaggle", "--device", "cuda", "--strategy", strategy, "--model", model, "--seed", str(seed)],
        cwd=STAGE, env=env, timeout=1800,
    )
    return {"status": "evaluated_group", "group": _group_key(strategy, model, seed), "tables": str(tables)}


def _merge_anchor_retention() -> dict[str, Any]:
    """Merge canonical anchor_retention rows across staged groups (build_anchors overwrites per group)."""
    canonical = STAGE / "results" / "tables" / "anchor_retention.csv"
    merged = STAGE / "results" / "data_audit" / "anchor_retention_merged.csv"
    rows: dict[tuple[str, ...], list[str]] = {}
    fields: list[str] = []
    for source in (merged, canonical):
        if not source.is_file():
            continue
        with source.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not fields and reader.fieldnames:
                fields = list(reader.fieldnames)
            for row in reader:
                key = tuple(str(row.get(name, "")) for name in ("strategy", "model", "class", "checkpoint_t"))
                rows[key] = [str(row.get(name, "")) for name in fields]
    if not rows or not fields:
        return {"status": "empty", "rows": 0}
    merged.parent.mkdir(parents=True, exist_ok=True)
    for target in (merged, canonical):
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(fields)
            writer.writerows(rows[key] for key in sorted(rows))
    return {"status": "merged", "rows": len(rows), "canonical": str(canonical), "merged": str(merged)}


def _anchor_counts() -> dict[str, Any]:
    anchors_dir = STAGE / "results" / "anchors"
    files: dict[str, Any] = {}
    for path in sorted(anchors_dir.glob("period_*.pt")):
        period = path.stem.split("_")[1]
        files[period] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    retention = STAGE / "results" / "tables" / "anchor_retention.csv"
    ret_rows: list[dict[str, Any]] = []
    if retention.is_file():
        with retention.open("r", encoding="utf-8", newline="") as handle:
            ret_rows = list(csv.DictReader(handle))
    manifest = anchors_dir / "anchors_manifest.json"
    return {
        "anchor_files": files,
        "retention_csv": str(retention) if retention.is_file() else None,
        "retention_rows": len(ret_rows),
        "manifest_path": str(manifest) if manifest.is_file() else None,
        "manifest": _read_json(manifest) if manifest.is_file() else None,
    }


def _explain_index_stats() -> dict[str, Any]:
    canonical = STAGE / "results" / "tables" / "explain_index.csv"
    rows: list[dict[str, Any]] = []
    if canonical.is_file():
        with canonical.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    by_explainer: dict[str, int] = {}
    by_k: dict[str, int] = {}
    runtime_total = 0.0
    runtime_by_explainer: dict[str, float] = {}
    fidelity: dict[str, list[float]] = {}
    runtime_values: list[float] = []
    for row in rows:
        explainer = str(row.get("explainer"))
        by_explainer[explainer] = by_explainer.get(explainer, 0) + 1
        k = str(row.get("k") or "")
        by_k[k] = by_k.get(k, 0) + 1
        try:
            runtime = float(row.get("runtime_s") or 0.0)
        except ValueError:
            runtime = 0.0
        runtime_values.append(runtime)
        runtime_total += runtime
        runtime_by_explainer[explainer] = runtime_by_explainer.get(explainer, 0.0) + runtime
        value = row.get("fidelity_comprehensiveness")
        if value not in (None, ""):
            try:
                fidelity.setdefault(explainer, []).append(float(value))
            except ValueError:
                pass
    runtime_values.sort()
    return {
        "path": str(canonical) if canonical.is_file() else None,
        "rows": len(rows),
        "rows_by_explainer": by_explainer,
        "rows_by_k": by_k,
        "runtime_total_s": round(runtime_total, 3),
        "runtime_by_explainer_s": {key: round(value, 3) for key, value in runtime_by_explainer.items()},
        "runtime_median_s": round(runtime_values[len(runtime_values) // 2], 4) if runtime_values else None,
        "runtime_p95_s": round(runtime_values[int(0.95 * (len(runtime_values) - 1))], 4) if runtime_values else None,
        "fidelity_comprehensiveness_mean": {key: round(sum(values) / len(values), 5) for key, values in fidelity.items() if values},
        "columns": list(rows[0].keys()) if rows else [],
        "operational_index": str(STAGE / "results" / "explanations" / "index.csv") if (STAGE / "results" / "explanations" / "index.csv").is_file() else None,
    }


def _explain_stage_diagonal(runctl: Mapping[str, Any], env: Mapping[str, str]) -> None:
    cfg = dict(runctl.get("explain") or {})
    groups = _explain_groups(cfg)
    periods = int(cfg.get("periods", 5))
    explainers = [str(item) for item in cfg.get("explainers", ["ig", "gnn_explainer"])]
    k_values = [int(item) for item in cfg.get("k", [5, 10])]
    max_anchors = int(cfg.get("max_anchors", 10_000))
    max_jobs = cfg.get("max_jobs")
    max_runs = cfg.get("max_runs")
    SUMMARY["explain"]["config"] = {
        **cfg,
        "groups": [_group_key(*group) for group in groups],
        "periods": periods,
        "explainers": explainers,
        "k": k_values,
        "max_anchors": max_anchors,
    }
    ensure_graphs(env)
    SUMMARY["explain"]["canonical_s5"] = _backfill_canonical_s5(groups, env)
    _apply_anchor_config(runctl)

    checkpoints: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for strategy, model, seed in groups:
        for period in range(1, periods + 1):
            path = STAGE / "runs" / strategy / model / f"seed{seed}" / f"M_{period}.pt"
            if not path.is_file():
                missing.append(f"{_group_key(strategy, model, seed)} M_{period}.pt")
            else:
                checkpoints[f"{_group_key(strategy, model, seed)}/M_{period}.pt"] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    if missing:
        raise FileNotFoundError(f"missing checkpoints for explain stage: {missing[:8]} (run the matrix stage / restore resume first)")
    SUMMARY["explain"]["checkpoints"] = checkpoints

    command = [
        sys.executable, "src/scripts/run_anchors_explain.py",
        "--lane", "kaggle", "--device", "cuda",
        "--models", ",".join(sorted({group[1] for group in groups})),
        "--strategies", ",".join(sorted({group[0] for group in groups})),
        "--seeds", ",".join(str(group[2]) for group in groups),
        "--periods", str(periods),
        "--explainers", ",".join(explainers),
        "--k", ",".join(str(value) for value in k_values),
        "--max-anchors", str(max_anchors),
        "--state-path", str(STAGE / "results" / "tables" / "explain_state.json"),
    ]
    if cfg.get("include_ear"):
        command.append("--include-ear")
    if max_runs:
        command.extend(["--max-runs", str(int(max_runs))])
    if max_jobs:
        command.extend(["--max-jobs", str(int(max_jobs))])
    if cfg.get("gnn_epochs"):
        command.extend(["--gnn-epochs", str(int(cfg["gnn_epochs"]))])
    driver = run_step("run_anchors_explain", command, cwd=STAGE, env=env, timeout=int(cfg.get("timeout_s", 10 * 3600)), allow_failure=True)
    SUMMARY["explain"]["driver"] = {"returncode": driver["returncode"], "duration_s": driver["duration_s"], "command": driver["command"]}

    # Finalize canonical S6 artifacts (combined anchors manifest + cumulative retention table) and
    # merge per-group retention rows. This runs the anchor builder once per group without --period.
    finalize: list[dict[str, Any]] = []
    for strategy, model, seed in groups:
        record = run_step(
            f"finalize_anchors_{_group_key(strategy, model, seed)}",
            [sys.executable, "src/scripts/build_anchors.py", "--lane", "kaggle", "--device", "cuda", "--strategy", strategy, "--model", model, "--seed", str(seed)],
            cwd=STAGE, env=env, timeout=3600, allow_failure=True,
        )
        merged = _merge_anchor_retention()
        finalize.append({"group": _group_key(strategy, model, seed), "returncode": record["returncode"], "duration_s": record["duration_s"], "retention": merged})
    SUMMARY["explain"]["finalize_anchors"] = finalize
    SUMMARY["explain"]["anchors"] = _anchor_counts()
    state = _read_json(STAGE / "results" / "tables" / "explain_state.json") or {}
    state_groups = state.get("groups") or {}
    failed = sorted(key for key, value in state_groups.items() if isinstance(value, dict) and value.get("status") in {"failed", "blocked"})
    SUMMARY["explain"]["state"] = state
    SUMMARY["explain"]["failed_jobs"] = failed
    SUMMARY["explain"]["jobs_planned"] = len(groups) * periods
    SUMMARY["explain"]["explain_index"] = _explain_index_stats()

    SUMMARY["explain"]["index_dedup"] = _dedupe_explain_indexes()
    SUMMARY["explain"]["explain_index"] = _explain_index_stats()
    index = SUMMARY["explain"]["explain_index"]
    duration = float(driver["duration_s"]) or None
    # anchors generated = GNNExplainer rows (one per anchor); IG has one row per (anchor, k)
    rows_by_explainer = index.get("rows_by_explainer") or {}
    anchors_total = int(rows_by_explainer.get("gnn_explainer") or 0)
    if not anchors_total and k_values:
        anchors_total = int(rows_by_explainer.get("ig", 0) // max(len(k_values), 1))
    expected_ig = anchors_total * len(k_values) if "ig" in explainers else 0
    expected_gnn = anchors_total if "gnn_explainer" in explainers else 0
    SUMMARY["explain"]["expected_rows"] = {"ig": expected_ig, "gnn_explainer": expected_gnn, "total": expected_ig + expected_gnn}
    SUMMARY["explain"]["rate"] = {
        "total_anchors": anchors_total,
        "groups": len(groups),
        "period_jobs": len(groups) * periods,
        "driver_duration_s": driver["duration_s"],
        "anchors_per_minute": round(anchors_total / (duration / 60.0), 2) if duration else None,
        "explain_rows_per_minute": round(index["rows"] / (duration / 60.0), 2) if duration else None,
        "explain_runtime_total_s": index["runtime_total_s"],
        "planning_note": "rates are per whole explain-stage wall clock (anchors + both explainers + finalize); index runtime_s covers the explainer calls only",
    }

    if driver["returncode"] != 0 or failed:
        raise RuntimeError(f"explain stage failed: driver rc={driver['returncode']}, failed jobs={failed[:6]}")


def _canonical_anchor_dir() -> Path:
    return STAGE / "results" / "anchors" / "canonical"


def _anchor_ids(path: Path) -> list[int]:
    payload = _load_anchor_file(path)
    return [int(anchor["anchor_id"]) for anchor in payload.get("anchors", [])]


def _explain_row_multiplier(explainers: Sequence[str], k_values: Sequence[int]) -> int:
    rows = 0
    if "ig" in explainers:
        rows += len(k_values)
    if "gnn_explainer" in explainers:
        rows += 1
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def _ensure_canonical_anchors(runctl: Mapping[str, Any], env: Mapping[str, str], cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Build (or verify) the canonical first-seed anchor set under results/text/anchors/canonical.

    D-071: one canonical anchor set from ft/egraphsage/seed42 per period, 250/period cap
    (stratified/rare policy unchanged); never overwritten by other groups. Root copies are
    kept for the validator/wiring contract (anchors/period_*.pt); per-group dirs only carry
    refs/manifests (payloads referenced unless small or explicitly requested).
    """
    periods = [int(item) for item in cfg.get("periods", [1, 2, 3, 4, 5])]
    reference = dict(cfg.get("anchor_reference") or {"strategy": "ft", "model": "egraphsage", "seed": 42})
    strategy = str(reference.get("strategy", "ft"))
    model = str(reference.get("model", "egraphsage"))
    seed = int(reference.get("seed", 42))
    canonical = _canonical_anchor_dir()
    canonical.mkdir(parents=True, exist_ok=True)
    manifest_path = canonical / "anchors_manifest.json"
    missing = [period for period in periods if not (canonical / f"period_{period}.pt").is_file()]
    rebuild = bool(cfg.get("rebuild_canonical_anchors", False)) or missing or not manifest_path.is_file()
    if rebuild:
        ensure_graphs(env)
        _apply_anchor_config(runctl)
        overrides = dict((runctl.get("explain") or {}).get("anchors") or {})
        command = [
            sys.executable, "src/scripts/build_anchors.py", "--lane", "kaggle", "--device", "cuda",
            "--strategy", strategy, "--model", model, "--seed", str(seed),
            "--out-dir", str(canonical),
        ]
        record = run_step("canonical_anchors", command, cwd=STAGE, env=env, timeout=3600)
        SUMMARY["explain"].setdefault("canonical_anchor_build", {})["duration_s"] = record["duration_s"]
    manifest = _read_json(manifest_path) or {}
    periods_info: dict[str, Any] = {}
    anchor_ids: dict[str, list[int]] = {}
    for period in periods:
        path = canonical / f"period_{period}.pt"
        if not path.is_file():
            raise FileNotFoundError(f"canonical anchor file missing after build: {path}")
        ids = _anchor_ids(path)
        entry = ((manifest.get("per_period") or {}).get(str(period)) or {})
        periods_info[str(period)] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "n_selected": int(entry.get("n_selected") or len(ids)),
            "anchor_ids": ids,
        }
        anchor_ids[str(period)] = ids
    # root compatibility copies + refreshed combined/per-period manifests
    root = STAGE / "results" / "anchors"
    root.mkdir(parents=True, exist_ok=True)
    per_period_manifest: dict[str, dict[str, Any]] = {}
    for period in periods:
        source = canonical / f"period_{period}.pt"
        target = root / f"period_{period}.pt"
        shutil.copy2(source, target)
        clean_entry = {key: value for key, value in periods_info[str(period)].items() if key != "anchor_ids"}
        payload = {
            "schema_version": 1,
            "kind": "s6_anchor_period",
            "dataset": manifest.get("dataset", "NF-UNSW-NB15-v3"),
            "strategy": strategy,
            "model": model,
            "seed": seed,
            "period": int(period),
            "canonical_dir": str(canonical),
            "anchor_file": str(canonical / f"period_{period}.pt"),
            "refreshed_at_utc": utc_now(),
            **clean_entry,
        }
        per_period_manifest[str(period)] = payload
        _write_json(canonical / f"anchors_manifest_period{period}.json", payload)
        _write_json(root / f"anchors_manifest_period{period}.json", payload)
    root_manifest = {**manifest}
    root_manifest["canonical_dir"] = str(canonical)
    root_manifest["refreshed_at_utc"] = utc_now()
    root_manifest["periods"] = {period: {key: value for key, value in periods_info[period].items() if key != "anchor_ids"} for period in periods_info}
    if isinstance(root_manifest.get("anchor_files"), dict):
        root_manifest["anchor_files"] = {
            period: {**root_manifest["periods"][period], "path": str(root / f"period_{period}.pt")}
            for period in root_manifest["periods"]
        }
    _write_json(root / "anchors_manifest.json", root_manifest)
    # per-group dirs: refs + refreshed per-period manifests (payload copies only when small/explicit)
    groups = _explain_groups((runctl.get("explain") or {}))
    copy_payloads = bool(cfg.get("copy_group_payloads", False))
    total_bytes = sum(item["bytes"] for item in periods_info.values())
    if total_bytes <= int(cfg.get("group_payload_copy_max_bytes", 20 * 1024 * 1024)):
        copy_payloads = True
    group_dirs: dict[str, Any] = {}
    for strategy_name, model_name, seed_value in groups:
        group_key = _group_key(strategy_name, model_name, seed_value)
        group_dir = STAGE / "results" / "anchors" / "groups" / group_key
        group_dir.mkdir(parents=True, exist_ok=True)
        ref = {
            "schema_version": 1,
            "group": group_key,
            "reference": {"strategy": strategy_name, "model": model_name, "seed": int(seed_value)},
            "canonical_dir": str(canonical),
            "refreshed_at_utc": utc_now(),
            "periods": {period: {key: value for key, value in periods_info[period].items() if key != "anchor_ids"} for period in periods_info},
            "payload_copies": copy_payloads,
        }
        _write_json(group_dir / "anchor_ref.json", ref)
        for period in periods:
            payload = dict(per_period_manifest[str(period)])
            payload["group"] = group_key
            payload["group_dir"] = str(group_dir)
            _write_json(group_dir / f"anchors_manifest_period{period}.json", payload)
        if copy_payloads:
            for period in periods:
                shutil.copy2(canonical / f"period_{period}.pt", group_dir / f"period_{period}.pt")
        group_dirs[group_key] = ref
    SUMMARY["explain"]["canonical_anchors"] = {
        "canonical_dir": str(canonical),
        "reference": {"strategy": strategy, "model": model, "seed": seed},
        "periods": {period: {key: value for key, value in info.items() if key != "anchor_ids"} for period, info in periods_info.items()},
        "root_manifest": str(root_manifest),
        "group_dirs": list(group_dirs),
        "payload_copies": copy_payloads,
    }
    return {"canonical_dir": str(canonical), "periods": periods_info, "root_manifest": root_manifest, "groups": group_dirs}


def _explain_index_state() -> dict[tuple[str, str], list[tuple[str, str]]]:
    canonical = STAGE / "results" / "tables" / "explain_index.csv"
    state: dict[tuple[str, str], list[tuple[str, str]]] = {}
    if not canonical.is_file():
        return state
    with canonical.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (str(row.get("checkpoint_sha")), str(row.get("anchor_id")))
            state.setdefault(key, []).append((str(row.get("explainer")), str(row.get("k"))))
    return state


def _explain_job_complete(index_state: Mapping[tuple[str, str], list[tuple[str, str]]], *, checkpoint_sha: str, anchor_ids: Sequence[int], explainers: Sequence[str], k_values: Sequence[int]) -> bool:
    ig_per_anchor = len(k_values) if "ig" in explainers else 0
    gnn_per_anchor = 1 if "gnn_explainer" in explainers else 0
    for anchor_id in anchor_ids:
        rows = index_state.get((str(checkpoint_sha), str(anchor_id)), [])
        ig = sum(1 for explainer, k in rows if explainer == "ig" and str(k) in {str(value) for value in k_values})
        gnn = sum(1 for explainer, _ in rows if explainer == "gnn_explainer")
        if ig < ig_per_anchor or gnn < gnn_per_anchor:
            return False
    return True


def _filter_explain_index_scope(allowed_checkpoint_shas: set[str], allowed_anchor_ids: set[str]) -> dict[str, Any]:
    """Keep only corrected-S7 rows (expected checkpoints, canonical anchor ids) in both index copies."""
    summary: dict[str, Any] = {"files": []}
    for path in (STAGE / "results" / "tables" / "explain_index.csv", STAGE / "results" / "explanations" / "index.csv"):
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
        kept: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for row in rows:
            if str(row.get("checkpoint_sha")) not in allowed_checkpoint_shas:
                continue
            if str(row.get("anchor_id")) not in allowed_anchor_ids:
                continue
            key = str(row.get("key") or "")
            if key not in kept:
                order.append(key)
            kept[key] = row
        filtered = [kept[key] for key in order]
        if len(filtered) != len(rows):
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(filtered)
        summary["files"].append({"path": str(path), "rows_before": len(rows), "rows_after": len(filtered), "rows_filtered": len(rows) - len(filtered)})
    return summary


def _snapshot_explain_manifest(job: Mapping[str, Any], audit_root: Path) -> str | None:
    source = audit_root / f"explain_manifest_{job['strategy']}_{job['model']}_seed{job['seed']}_period{job['anchor_period']}.json"
    if not source.is_file():
        return None
    target = audit_root / "explain_manifests" / _group_key(job["strategy"], job["model"], job["seed"]) / f"period{job['anchor_period']}_ckpt{job['checkpoint_period']}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return str(target)


def _normalize_explain_metadata(checkpoint_period_by_sha: Mapping[str, int]) -> dict[str, Any]:
    """Rewrite per-entry cache metadata checkpoint_period from the actual checkpoint sha.

    run_explain sets metadata checkpoint_period=args.period, i.e. the anchor period, even
    when an explicit later --checkpoint is used. The corrected-S7 cross-checkpoint rows
    are therefore mislabelled for the S8 wiring; normalize them from the checkpoint sha
    (authoritative, part of the cache key) after the jobs complete.
    """
    canonical = STAGE / "results" / "tables" / "explain_index.csv"
    summary = {"rows": 0, "normalized": 0, "already_correct": 0, "missing_metadata": 0}
    if not canonical.is_file():
        return summary
    for row in csv.DictReader(canonical.open("r", encoding="utf-8", newline="")):
        expected = checkpoint_period_by_sha.get(str(row.get("checkpoint_sha")))
        if expected is None:
            continue
        summary["rows"] += 1
        json_path = Path(str(row.get("json_path") or ""))
        if not json_path.is_absolute():
            json_path = STAGE / "results" / "tables" / json_path
        if not json_path.is_file():
            summary["missing_metadata"] += 1
            continue
        try:
            metadata = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception:
            summary["missing_metadata"] += 1
            continue
        current = metadata.get("checkpoint_period")
        if current == expected:
            summary["already_correct"] += 1
            continue
        metadata["checkpoint_period_legacy"] = current
        metadata["checkpoint_period"] = int(expected)
        metadata["checkpoint_period_source"] = "captured_from_checkpoint_sha"
        json_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        summary["normalized"] += 1
    return summary


def _prune_unreferenced_explain_entries(referenced_names: set[str]) -> dict[str, Any]:
    root = STAGE / "results" / "explanations"
    if not root.is_dir():
        return {"removed": 0, "bytes": 0}
    removed = 0
    freed = 0
    for path in list(root.glob("*")):
        if path.suffix not in {".npz", ".json"} or not path.is_file():
            continue
        if path.name in referenced_names:
            continue
        freed += path.stat().st_size
        path.unlink()
        removed += 1
    return {"removed": removed, "bytes": freed}


def _explain_stage_cross(runctl: Mapping[str, Any], env: Mapping[str, str]) -> None:
    cfg = dict(runctl.get("explain") or {})
    mode = str(cfg.get("mode", "cross_checkpoint"))
    if mode != "cross_checkpoint":
        _explain_stage_diagonal(runctl, env)
        return
    groups = _explain_groups(cfg)
    raw_periods = cfg.get("periods", [1, 2, 3, 4, 5])
    if isinstance(raw_periods, int):
        periods = list(range(1, int(raw_periods) + 1))
    else:
        periods = sorted(int(item) for item in raw_periods)
    raw_checkpoints = cfg.get("checkpoint_periods", periods)
    if isinstance(raw_checkpoints, int):
        checkpoint_periods = list(range(1, int(raw_checkpoints) + 1))
    else:
        checkpoint_periods = sorted(int(item) for item in raw_checkpoints)
    horizon = str(cfg.get("horizon", "all"))
    explainers = [str(item) for item in cfg.get("explainers", ["ig", "gnn_explainer"])]
    k_values = [int(item) for item in cfg.get("k", [5, 10])]
    SUMMARY["explain"]["config"] = {
        **cfg,
        "groups": [_group_key(*group) for group in groups],
        "periods": periods,
        "checkpoint_periods": checkpoint_periods,
        "explainers": explainers,
        "k": k_values,
    }
    SUMMARY["explain"]["canonical_s5"] = _backfill_canonical_s5(groups, env)
    anchors = _ensure_canonical_anchors(runctl, env, {**cfg, "periods": periods, "checkpoint_periods": checkpoint_periods})
    n_anchors = {int(period): int(info["n_selected"]) for period, info in anchors["periods"].items()}
    # checkpoints for every group/t
    checkpoints: dict[str, dict[str, Any]] = {}
    missing: list[str] = []
    for strategy, model, seed in groups:
        for period in checkpoint_periods:
            path = STAGE / "runs" / strategy / model / f"seed{seed}" / f"M_{period}.pt"
            key = f"{_group_key(strategy, model, seed)}/M_{period}.pt"
            if path.is_file():
                checkpoints[key] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            else:
                missing.append(key)
    if missing:
        raise FileNotFoundError(f"checkpoints missing for corrected S7: {missing[:8]} (run the matrix/resume first)")
    SUMMARY["explain"]["checkpoints"] = checkpoints
    jobs: list[dict[str, Any]] = []
    for strategy, model, seed in groups:
        group_key = _group_key(strategy, model, seed)
        for anchor_period in periods:
            for checkpoint_period in checkpoint_periods:
                if checkpoint_period < anchor_period:
                    continue
                if horizon == "adjacent" and checkpoint_period not in {anchor_period, anchor_period + 1}:
                    continue
                if horizon == "final" and checkpoint_period not in {anchor_period, max(checkpoint_periods)}:
                    continue
                jobs.append(
                    {
                        "strategy": strategy,
                        "model": model,
                        "seed": int(seed),
                        "group": group_key,
                        "anchor_period": int(anchor_period),
                        "checkpoint_period": int(checkpoint_period),
                        "n_anchors": n_anchors[int(anchor_period)],
                        "rows_expected": int(n_anchors[int(anchor_period)] * _explain_row_multiplier(explainers, k_values)),
                        "anchor_file": str(Path(anchors["periods"][str(anchor_period)]["path"])),
                        "checkpoint": str(STAGE / "runs" / strategy / model / f"seed{seed}" / f"M_{checkpoint_period}.pt"),
                        "checkpoint_sha": checkpoints[f"{group_key}/M_{checkpoint_period}.pt"]["sha256"],
                    }
                )
    state_path = STAGE / "results" / "tables" / "explain_state.json"
    state = _read_json(state_path) or {"schema_version": 2, "jobs": {}}
    state.setdefault("jobs", {})
    state.setdefault("schema_version", 2)
    index_state = _explain_index_state()
    pending: list[dict[str, Any]] = []
    already_complete = 0
    for job in jobs:
        job_key = f"{job['group']}|anchor{job['anchor_period']}|ckpt{job['checkpoint_period']}"
        recorded_payload = state["jobs"].get(job_key) or {}
        recorded = (
            recorded_payload.get("status") == "complete"
            and int(recorded_payload.get("n_anchors") or 0) >= int(job["n_anchors"])
            and int(recorded_payload.get("rows_expected") or 0) >= int(job["rows_expected"])
        )
        complete = recorded or _explain_job_complete(
            index_state,
            checkpoint_sha=job["checkpoint_sha"],
            anchor_ids=anchors["periods"][str(job["anchor_period"])]["anchor_ids"],
            explainers=explainers,
            k_values=k_values,
        )
        if complete:
            already_complete += 1
            continue
        job["key"] = job_key
        pending.append(job)
    total_pending = len(pending)
    offset = int(cfg.get("job_offset") or 0)
    pending = pending[offset:]
    max_jobs = cfg.get("max_jobs")
    this_slice = pending if max_jobs is None else pending[: int(max_jobs)]
    SUMMARY["explain"]["slicing"] = {"jobs_total": len(jobs), "already_complete": already_complete, "pending_before": total_pending, "job_offset": offset, "max_jobs": max_jobs, "selected": len(this_slice)}
    audit_root = STAGE / "results" / "data_audit"
    executed: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for job in this_slice:
        command = [
            sys.executable, "src/scripts/run_explain.py",
            "--lane", "kaggle", "--device", "cuda",
            "--strategy", job["strategy"], "--model", job["model"], "--seed", str(job["seed"]),
            "--period", str(job["anchor_period"]),
            "--checkpoint", job["checkpoint"],
            "--anchors", job["anchor_file"],
            "--max-anchors", str(job["n_anchors"]),
            "--explainers", ",".join(explainers),
            "--k", ",".join(str(value) for value in k_values),
        ]
        name = f"explain_{job['group']}_j{job['anchor_period']}_ckpt{job['checkpoint_period']}"
        record = run_step(name, command, cwd=STAGE, env=env, timeout=int(cfg.get("job_timeout_s", 3600)), allow_failure=True)
        manifest_snapshot = _snapshot_explain_manifest(job, audit_root)
        job_record = {
            "job": job["key"],
            "duration_s": record["duration_s"],
            "returncode": record["returncode"],
            "n_anchors": job["n_anchors"],
            "rows_expected": job["rows_expected"],
            "manifest": manifest_snapshot,
        }
        state["jobs"][job["key"]] = {
            "status": "complete" if record["returncode"] == 0 else "failed",
            "duration_s": record["duration_s"],
            "n_anchors": job["n_anchors"],
            "rows_expected": job["rows_expected"],
            "updated_at_utc": utc_now(),
            "manifest": manifest_snapshot,
        }
        _write_json(state_path, state)
        if record["returncode"] != 0:
            failures.append(job_record)
            break
        executed.append(job_record)
    checkpoint_period_by_sha: dict[str, int] = {}
    for strategy, model, seed in groups:
        for period in checkpoint_periods:
            key = f"{_group_key(strategy, model, seed)}/M_{period}.pt"
            if key in checkpoints:
                checkpoint_period_by_sha[checkpoints[key]["sha256"]] = int(period)
    SUMMARY["explain"]["metadata_normalization"] = _normalize_explain_metadata(checkpoint_period_by_sha)
    index_dedup = _dedupe_explain_indexes()
    allowed_shas = {info["sha256"] for info in checkpoints.values()}
    allowed_anchor_ids: set[str] = set()
    for info in anchors["periods"].values():
        allowed_anchor_ids.update(str(anchor_id) for anchor_id in info["anchor_ids"])
    scope_filter = _filter_explain_index_scope(allowed_shas, allowed_anchor_ids)
    SUMMARY["explain"]["index_dedup"] = index_dedup
    SUMMARY["explain"]["scope_filter"] = scope_filter
    referenced: set[str] = set()
    canonical_index = STAGE / "results" / "tables" / "explain_index.csv"
    if canonical_index.is_file():
        for row in csv.DictReader(canonical_index.open("r", encoding="utf-8", newline="")):
            for column in ("npz_path", "json_path"):
                value = str(row.get(column) or "")
                if value:
                    referenced.add(Path(value).name)
            referenced.add(str(row.get("key") or ""))
    SUMMARY["explain"]["pruned_entries"] = _prune_unreferenced_explain_entries(referenced)
    SUMMARY["explain"]["explain_index"] = _explain_index_stats()
    SUMMARY["explain"]["anchors"] = {
        "canonical_dir": anchors["canonical_dir"],
        "root_manifest": anchors["root_manifest"],
        "periods": {period: {key: value for key, value in info.items() if key != "anchor_ids"} for period, info in anchors["periods"].items()},
        "group_dirs": anchors["groups"],
    }
    planned_pairs = 0
    planned_pairs_total = 0
    for strategy, model, seed in groups:
        for anchor_period in periods:
            planned_pairs += n_anchors[anchor_period] * sum(1 for t in checkpoint_periods if t > anchor_period)
            planned_pairs_total += n_anchors[anchor_period] * sum(1 for t in checkpoint_periods if t >= anchor_period)
    completed_pairs = 0
    completed_pairs_total = 0
    for job_key, job_state in state["jobs"].items():
        if job_state.get("status") != "complete":
            continue
        try:
            anchor_period = int(str(job_key).split("|anchor")[1].split("|")[0])
            checkpoint_period = int(str(job_key).split("|ckpt")[1])
        except Exception:
            continue
        n_anchors_job = int(job_state.get("n_anchors") or 0)
        completed_pairs_total += n_anchors_job
        if checkpoint_period > anchor_period:
            completed_pairs += n_anchors_job
    pending_after = max(0, total_pending - len(executed))
    SUMMARY["explain"]["jobs"] = {
        "planned": len(jobs),
        "already_complete": already_complete,
        "pending_before_push": total_pending,
        "executed_this_push": len(executed),
        "pending_after_push": pending_after,
        "failures": failures,
        "pairs_expected": planned_pairs,
        "pairs_expected_cross_checkpoint": planned_pairs,
        "pairs_expected_total_incl_diagonal": planned_pairs_total,
        "pairs_completed_state": completed_pairs,
        "pairs_completed_state_total_incl_diagonal": completed_pairs_total,
        "rows_expected_total": planned_pairs_total * int(_explain_row_multiplier(explainers, k_values)),
        "index_rows": SUMMARY["explain"]["explain_index"].get("rows"),
    }
    duration = sum(float(record["duration_s"]) for record in executed) or None
    if completed_pairs:
        SUMMARY["explain"]["rate"] = {
            "anchor_explanations_completed": sum(int(record["n_anchors"]) for record in executed),
            "job_wall_seconds": round(sum(float(record["duration_s"]) for record in executed), 2),
            "anchor_pairs_per_second": round(completed_pairs / duration, 3) if duration else None,
        }
    if failures:
        raise RuntimeError(f"corrected S7 job failed: {failures[0]['job']} rc={failures[0]['returncode']} (see logs)")


def explain_stage(runctl: Mapping[str, Any], env: Mapping[str, str]) -> None:
    cfg = dict(runctl.get("explain") or {})
    if str(cfg.get("mode", "cross_checkpoint")) == "cross_checkpoint":
        _explain_stage_cross(runctl, env)
    else:
        _explain_stage_diagonal(runctl, env)


def _load_anchor_file(path: Path) -> dict[str, Any]:
    import torch

    return torch.load(path, weights_only=False)


def _stratified_anchor_subset(anchors: Sequence[Mapping[str, Any]], max_n: int, *, rare_min_eligible: int = 10) -> list[Mapping[str, Any]]:
    """Deterministic class-stratified anchor subset (D-033 strata; rare merge only for reporting)."""
    if max_n >= len(anchors):
        return list(anchors)
    from collections import OrderedDict

    buckets: "OrderedDict[str, list[Mapping[str, Any]]]" = OrderedDict()
    for anchor in anchors:
        label = int(anchor.get("label", -1))
        buckets.setdefault(str(label), []).append(anchor)
    counts = {key: len(value) for key, value in buckets.items()}
    strata: "OrderedDict[str, list[Mapping[str, Any]]]" = OrderedDict()
    rare: list[Mapping[str, Any]] = []
    for key, values in buckets.items():
        if counts[key] < int(rare_min_eligible):
            rare.extend(values)
        else:
            strata[key] = list(values)
    if rare:
        strata["rare"] = rare
    total = sum(len(values) for values in strata.values())
    if total == 0:
        return list(anchors)[:max_n]
    quotas = {key: max(1, int(max_n * len(values) / total)) for key, values in strata.items()}
    while sum(quotas.values()) > max_n:
        for key in sorted(quotas, key=lambda item: quotas[item], reverse=True):
            if sum(quotas.values()) <= max_n:
                break
            if quotas[key] > 1:
                quotas[key] -= 1
    selected: list[Mapping[str, Any]] = []
    for key, values in strata.items():
        selected.extend(values[: quotas[key]])
    index = 0
    while len(selected) < max_n and index < len(anchors):
        anchor = anchors[index]
        if anchor not in selected:
            selected.append(anchor)
        index += 1
    selected_ids = {int(anchor.get("anchor_id", -1)) for anchor in selected}
    return [anchor for anchor in anchors if int(anchor.get("anchor_id", -1)) in selected_ids]


def run_noise_floor_stage(runctl: Mapping[str, Any], env: Mapping[str, str]) -> None:
    """S8 noise-floor producer stage: R=5 model-null reseeds + R=10 explainer repeats on a stratified subset."""
    cfg = dict(runctl.get("noise_floor") or {})
    if not cfg.get("enabled"):
        SUMMARY["noise_floor"] = {"enabled": False}
        return
    reference = dict(cfg.get("reference") or {})
    strategy = str(reference.get("strategy", "ft"))
    model = str(reference.get("model", "egraphsage"))
    seed = int(reference.get("seed", 42))
    periods = [int(item) for item in cfg.get("periods", [1, 2, 3, 4, 5])]
    anchors_per_period = int(cfg.get("anchors_per_period", 100))
    repeats = int(cfg.get("repeats", 10))
    null_reseeds = int(cfg.get("null_reseeds", 5))
    null_epochs = cfg.get("null_epochs")
    null_epochs = None if null_epochs in (None, "") else int(null_epochs)
    explainers = ",".join(str(item) for item in cfg.get("explainers", ["ig", "gnn_explainer"]))
    SUMMARY["noise_floor"] = {
        "enabled": True,
        "reference": {"strategy": strategy, "model": model, "seed": seed},
        "periods": periods,
        "anchors_per_period": anchors_per_period,
        "repeats": repeats,
        "null_reseeds": null_reseeds,
        "null_epochs": null_epochs,
        "explainers": explainers.split(","),
    }
    ensure_graphs(env)
    checkpoints = {}
    missing = []
    for period in periods:
        path = STAGE / "runs" / strategy / model / f"seed{seed}" / f"M_{period}.pt"
        if path.is_file():
            checkpoints[str(period)] = {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}
        else:
            missing.append(str(path))
    if missing:
        raise FileNotFoundError(f"noise-floor reference checkpoints missing: {missing[:4]}")
    SUMMARY["noise_floor"]["checkpoints"] = checkpoints

    # Rebuild the reference anchor sets once (all periods) so the noise floor and the
    # explanation stage share one common, protocol-anchored edge set.
    explain_cfg = dict(runctl.get("explain") or {})
    canonical = _canonical_anchor_dir()
    if not (canonical / "anchors_manifest.json").is_file():
        _ensure_canonical_anchors(runctl, env, {**explain_cfg, "periods": periods, "rebuild_canonical_anchors": True})
    anchors_dir = canonical if canonical.is_dir() else STAGE / "results" / "anchors"
    noise_anchors = STAGE / "results" / "anchors_noise"
    noise_anchors.mkdir(parents=True, exist_ok=True)
    subset_summary: dict[str, Any] = {}
    for period in periods:
        source = anchors_dir / f"period_{period}.pt"
        if not source.is_file():
            raise FileNotFoundError(f"reference anchors missing: {source}")
        payload = _load_anchor_file(source)
        anchors = list(payload.get("anchors", []))
        selected = _stratified_anchor_subset(anchors, anchors_per_period, rare_min_eligible=int(cfg.get("rare_min_eligible", 10)))
        target = noise_anchors / f"period_{period}.pt"
        import torch

        torch.save({"anchors": selected, "period": int(period), "source": str(source), "subset": "stratified", "max_n": anchors_per_period}, target)
        subset_summary[str(period)] = {"available": len(anchors), "selected": len(selected), "path": str(target), "sha256": sha256_file(target)}
    SUMMARY["noise_floor"]["anchor_subsets"] = subset_summary

    runs: list[dict[str, Any]] = []
    for period in periods:
        command = [
            sys.executable, "src/scripts/build_noise_floor.py",
            "--lane", "kaggle", "--device", "cuda",
            "--strategy", strategy, "--model", model, "--seed", str(seed), "--period", str(period),
            "--anchors-dir", str(noise_anchors),
            "--explainers", explainers,
            "--repeats", str(repeats),
            "--null-reseeds", str(null_reseeds),
            "--max-anchors", str(anchors_per_period),
        ]
        if null_epochs is not None:
            command.extend(["--null-epochs", str(int(null_epochs))])
        record = run_step(f"noise_floor_p{period}", command, cwd=STAGE, env=env, timeout=int(cfg.get("timeout_s", 3600)), allow_failure=True)
        runs.append({"period": period, "returncode": record["returncode"], "duration_s": record["duration_s"]})
    SUMMARY["noise_floor"]["runs"] = runs
    SUMMARY["noise_floor"]["index_dedup"] = _dedupe_noise_indexes()
    noise_root = STAGE / "results" / "explanations_noise"
    stats: dict[str, Any] = {}
    for channel in ("model_null", "explainer_repeat"):
        index = noise_root / channel / "index.csv"
        if index.is_file():
            with index.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            stats[channel] = {"index": str(index), "rows": len(rows), "entry_files": len([p for p in (noise_root / channel).glob("*") if p.suffix in {".npz", ".json"}])}
    combined = noise_root / "combined_null.csv"
    if combined.is_file():
        with combined.open("r", encoding="utf-8", newline="") as handle:
            stats["combined_null"] = {"path": str(combined), "rows": sum(1 for _ in handle) - 1}
    SUMMARY["noise_floor"]["indexes"] = stats
    failed = [run for run in runs if run["returncode"] != 0]
    if failed:
        raise RuntimeError(f"noise-floor stage: {len(failed)} period(s) failed: {failed[:3]}")


def _dedupe_explain_indexes() -> dict[str, Any]:
    """Drop duplicate cache-key rows from the canonical + operational evidence indexes.

    Re-running a group on a resumed staged run can append a second row for the same
    cache key with a newer created_at_utc; the S7 validator requires key uniqueness.
    Keep the newest row per key and rewrite both copies in place.
    """
    targets = [STAGE / "results" / "tables" / "explain_index.csv", STAGE / "results" / "explanations" / "index.csv"]
    summary: dict[str, Any] = {"files": []}
    for path in targets:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
        by_key: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for row in rows:
            key = str(row.get("key") or "")
            if key not in by_key:
                order.append(key)
            by_key[key] = row
        deduped = [by_key[key] for key in order]
        if len(deduped) != len(rows):
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(deduped)
        summary["files"].append({"path": str(path), "rows_before": len(rows), "rows_after": len(deduped), "duplicates_removed": len(rows) - len(deduped)})
    return summary


def _dedupe_noise_indexes() -> dict[str, Any]:
    root = STAGE / "results" / "explanations_noise"
    summary: dict[str, Any] = {"files": []}
    for path in sorted(root.glob("*/index.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            fields = list(reader.fieldnames or [])
            rows = list(reader)
        by_key: dict[str, dict[str, Any]] = {}
        order: list[str] = []
        for row in rows:
            key = str(row.get("key") or "")
            if key not in by_key:
                order.append(key)
            by_key[key] = row
        deduped = [by_key[key] for key in order]
        if len(deduped) != len(rows):
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(deduped)
        summary["files"].append({"path": str(path), "rows_before": len(rows), "rows_after": len(deduped), "duplicates_removed": len(rows) - len(deduped)})
    return summary


def make_output_zip() -> dict[str, Any]:
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for prefix, root in (("results", STAGE / "results"), ("runs", STAGE / "runs"), ("logs", LOGS_DIR)):
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                if path.suffix in {".pt", ".pth", ".npz"} and path.stat().st_size > 200 * 1024 * 1024:
                    continue
                archive.write(path, arcname=f"{prefix}/{path.relative_to(root).as_posix()}")
        if (STAGE / "data" / "processed" / "NF-UNSW-NB15-v3" / "graphs_manifest.json").is_file():
            archive.writestr("graphs_manifest.json", (STAGE / "data" / "processed" / "NF-UNSW-NB15-v3" / "graphs_manifest.json").read_text(encoding="utf-8"))
        archive.writestr("runctl.json", json.dumps(SUMMARY.get("runctl", {}), indent=2, sort_keys=True, default=str))
        archive.writestr("run_summary.json", json.dumps(SUMMARY, indent=2, sort_keys=True, default=str))
    digest = sha256_file(OUTPUT_ZIP)
    log(f"output zip: {OUTPUT_ZIP} ({OUTPUT_ZIP.stat().st_size} bytes, sha256 {digest})")
    return {"path": str(OUTPUT_ZIP), "bytes": OUTPUT_ZIP.stat().st_size, "sha256": digest, "excluded_large_pt_over_mb": 200}


def gpu_snapshot() -> dict[str, Any]:
    try:
        completed = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
        return {"nvidia_smi": (completed.stdout or "").strip(), "returncode": completed.returncode}
    except Exception as exc:  # pragma: no cover
        return {"nvidia_smi": None, "error": str(exc)}


def main() -> int:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY["gpu"] = gpu_snapshot()
    try:
        runctl = load_runctl()
        runctl_stage = str(runctl.get("stage") or STAGE_KIND)
        if runctl_stage not in {STAGE_KIND, "all"}:
            raise RuntimeError(f"runctl stage {runctl_stage!r} does not match kernel stage {STAGE_KIND!r}")
        stage = runctl
        verify_and_stage_inputs(frozen_fallback_entries=[])
        restore_resume()
        env, env_info = prepare_environment()
        SUMMARY["environment"] = env_info
        if STAGE_KIND in {"matrix", "all"}:
            run_matrix_stage(stage, env)
        if STAGE_KIND in {"explain", "all"}:
            explain_stage(stage, env)
        if STAGE_KIND in {"noise", "all"}:
            run_noise_floor_stage(stage, env)
        SUMMARY["gpu_after"] = gpu_snapshot()
        SUMMARY["status"] = "success"
        SUMMARY["finished_at_utc"] = utc_now()
        SUMMARY["wall_clock_s"] = round((datetime.now(timezone.utc) - STARTED_AT).total_seconds(), 1)
    except Exception as exc:  # noqa: BLE001
        SUMMARY["status"] = "failed"
        SUMMARY["error"] = f"{type(exc).__name__}: {exc}"
        SUMMARY["traceback"] = traceback.format_exc()[-6000:]
        SUMMARY["finished_at_utc"] = utc_now()
        SUMMARY["wall_clock_s"] = round((datetime.now(timezone.utc) - STARTED_AT).total_seconds(), 1)
        log(f"FAILED: {SUMMARY['error']}")
    finally:
        try:
            if STAGE.exists():
                try:
                    SUMMARY["artifacts"] = make_output_zip()
                except Exception as zip_exc:  # pragma: no cover
                    SUMMARY["zip_error"] = f"{type(zip_exc).__name__}: {zip_exc}"
            SUMMARY["per_step_logs"] = [{"step": item.get("step"), "returncode": item.get("returncode"), "duration_s": item.get("duration_s"), "log": item.get("log")} for item in STEP_RECORDS]
            SUMMARY_PATH.write_text(json.dumps(SUMMARY, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
            print("XDRIFT_RUN_SUMMARY=" + json.dumps({key: SUMMARY.get(key) for key in ("status", "stage", "kernel_ref", "error", "wall_clock_s", "artifacts", "matrix", "explain")}, default=str)[:12000], flush=True)
            if STAGE.exists():
                shutil.rmtree(STAGE, ignore_errors=True)
                SUMMARY["stage_removed_after_zip"] = True
                SUMMARY_PATH.write_text(json.dumps(SUMMARY, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        except Exception as exc:  # pragma: no cover
            print(f"[xdrift-{STAGE_KIND}] FINAL REPORTING FAILED: {exc}", flush=True)
    return 0 if SUMMARY.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
