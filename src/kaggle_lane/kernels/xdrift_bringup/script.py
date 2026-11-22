#!/usr/bin/env python
"""XDRIFT Kaggle T4 bring-up kernel (D-065 canonical lane).

End-to-end smoke of the Kaggle execution path, WITHOUT running anchors,
explanations or the full matrix:

1. locate/mount the private inputs dataset (raw NF-UNSW-NB15-v3 CSV, frozen C1
   artifacts zip, runtime code-snapshot zip, env manifest with all hashes);
2. verify every input hash/size and fail loudly on mismatch (frozen protocol
   D-042 must be byte-identical);
3. extract the code snapshot to ``/kaggle/working/stage`` and seed the frozen
   artifacts exactly like ``remote_rtx5060/src/scripts/03_get_code.ps1``;
4. set ``XDRIFT_*`` to the stage tree, install the pinned PyG if the image lacks
   it, run ``check_env.py --require-cuda``;
5. run ``src/scripts/build_graphs.py`` on the frozen C1 state;
6. run ONE canary matrix group with
   ``src/scripts/run_matrix.py --lane kaggle --device cuda --models egraphsage
   --strategies ft --seeds 42 --periods 5 --max-runs 1
   --epoch-seconds-full-size 4.0`` (diagnostic wall-clock anchor only);
7. run ``src/scripts/run_eval.py`` for that group and collect the R matrix;
8. write ``bringup_summary.json`` (env, verified hashes, per-period timings,
   eval completeness, GPU info, peak memory, OOM check) and zip the artifacts
   (results + runs/manifests + logs) into ``/kaggle/working``.

The stage tree itself is removed after zipping so the downloaded output stays
small (processed graphs are large and are not evidence). The kernel exits
non-zero on any failure and still writes a partial summary + zip.
"""

from __future__ import annotations

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

WORK_ROOT = Path("/kaggle/working")
INPUT_ROOT = Path("/kaggle/input")
STAGE = WORK_ROOT / "stage"
OUTPUT_ZIP = WORK_ROOT / "xdrift_bringup_output.zip"
SUMMARY_PATH = WORK_ROOT / "bringup_summary.json"
LOGS_DIR = WORK_ROOT / "logs"
EXPECTED_CSV_BYTES = 577_360_958
REQUIRED_PYG = "torch-geometric==2.8.1"

STARTED_AT = datetime.now(timezone.utc)
SUMMARY: dict[str, Any] = {
    "kind": "xdrift-kaggle-bringup",
    "schema_version": 1,
    "started_at_utc": STARTED_AT.isoformat(timespec="seconds"),
    "status": "running",
    "kernel_ref": "mysteriousavailable/xdrift-t4-bring-up-canonical-lane-d-065",
    "stage_path": str(STAGE),
    "steps": {},
    "input_verification": {},
    "environment": {},
    "frozen": {},
    "graphs": {},
    "matrix": {},
    "eval": {},
    "gpu": {},
    "risks": [],
}
STEP_LOGS: list[dict[str, Any]] = []


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[bringup] {message}", flush=True)


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
) -> dict[str, Any]:
    """Run one subprocess, stream its output to the kernel log, record timing."""
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
        output = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        returncode = -9
        output += f"\n[bringup] TIMEOUT after {timeout}s\n"
    duration = round(time.monotonic() - started, 2)
    log_path.write_text(output, encoding="utf-8")
    tail_lines = output.strip().splitlines()[-40:]
    print("\n".join(f"[{name}] {line}" for line in tail_lines), flush=True)
    record = {
        "command": [str(part) for part in command],
        "returncode": int(returncode),
        "duration_s": duration,
        "log": str(log_path),
        "tail": "\n".join(tail_lines)[-8000:],
    }
    STEP_LOGS.append({"step": name, **record})
    SUMMARY["steps"][name] = record
    if returncode != 0 and not allow_failure:
        raise RuntimeError(f"step {name!r} failed with exit code {returncode} (see {log_path})")
    return record


def find_input(name_pattern: str) -> Path:
    matches = sorted(INPUT_ROOT.rglob(name_pattern))
    if not matches:
        raise FileNotFoundError(f"dataset input {name_pattern!r} not found under {INPUT_ROOT}")
    log(f"input {name_pattern}: {len(matches)} match(es), using {matches[0]}")
    return matches[0]


def extract_zip(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as handle:
        handle.extractall(destination)


def seeded_frozen_hashes(fallback_entries: Sequence[Mapping[str, Any]] = ()) -> Path:
    """Verify the frozen tree against the shipped manifest (03_get_code.ps1 semantics)."""
    manifest_path = STAGE / "frozen_artifacts.sha256"
    expected: dict[str, str] = {}
    if manifest_path.is_file():
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            digest, rel = line.split("  ", 1)
            expected[rel.strip()] = digest.strip()
    else:
        for entry in fallback_entries:
            expected[str(entry["path"])] = str(entry["sha256"])
    if not expected:
        raise RuntimeError("no frozen hash manifest available (neither frozen_artifacts.sha256 nor env_manifest entries)")
    problems: list[str] = []
    verified = 0
    for rel, digest in sorted(expected.items()):
        target = STAGE / rel
        if not target.is_file():
            problems.append(f"missing {rel}")
            continue
        actual = sha256_file(target)
        if actual != digest:
            problems.append(f"{rel}: expected {digest}, got {actual}")
        else:
            verified += 1
    SUMMARY["frozen"] = {
        "manifest": str(manifest_path),
        "files_expected": len(expected),
        "files_verified": verified,
        "problems": problems,
        "verified": not problems,
    }
    if problems:
        raise RuntimeError("frozen artifact verification failed: " + "; ".join(problems[:5]))
    log(f"frozen artifacts verified: {verified}/{len(expected)} files")
    return manifest_path


def query_environment(env: Mapping[str, str]) -> dict[str, Any]:
    """Collect python/torch/PyG/CUDA/GPU facts in a child interpreter (clean import state)."""
    script = (
        "import json, platform, torch\n"
        "import torch_geometric\n"
        "info = {'python': platform.python_version(), 'torch': torch.__version__,"
        "'torch_cuda_build': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),"
        "'pyg': torch_geometric.__version__}\n"
        "if torch.cuda.is_available():\n"
        "    info['device'] = torch.cuda.get_device_name(0)\n"
        "    info['capability'] = list(torch.cuda.get_device_capability(0))\n"
        "    info['total_memory_gb'] = round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2)\n"
        "    x = torch.randn(512, 512, device='cuda'); info['matmul_ok'] = bool((x @ x).isfinite().all().item())\n"
        "print(json.dumps(info))\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], cwd=str(STAGE), env=dict(env), text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=300,
    )
    print(completed.stdout or "", flush=True)
    if completed.returncode != 0:
        raise RuntimeError(f"environment query failed: {completed.stdout}")
    return json.loads([line for line in completed.stdout.splitlines() if line.startswith("{")][-1])


def collect_metrics() -> dict[str, Any]:
    """Collect per-period training metrics, matrix state and group manifest facts."""
    metrics: list[dict[str, Any]] = []
    runs_root = STAGE / "runs"
    for path in sorted(runs_root.rglob("metrics*.json")) if runs_root.is_dir() else []:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        result = payload.get("result") or {}
        metrics.append(
            {
                "path": str(path.relative_to(STAGE)),
                "strategy": payload.get("strategy"),
                "model": payload.get("model"),
                "period": payload.get("period"),
                "batching": payload.get("batching"),
                "training": {key: result.get(key) for key in ("epochs_run", "best_val_macro_f1", "timing", "batches_per_epoch")},
                "val": result.get("val"),
            }
        )
    state_path = STAGE / "results" / "tables" / "matrix_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    index_path = STAGE / "results" / "tables" / "matrix_index.csv"
    audit_root = STAGE / "results" / "data_audit"
    matrix_manifest = audit_root / "matrix_manifest.json"
    matrix_plan = json.loads(matrix_manifest.read_text(encoding="utf-8")) if matrix_manifest.is_file() else {}
    group_manifests: dict[str, Any] = {}
    for path in sorted(audit_root.glob("matrix_manifest_*.json")):
        try:
            group_manifests[path.stem] = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
    run_manifests: list[dict[str, Any]] = []
    for path in sorted(runs_root.rglob("manifest*.json")) if runs_root.is_dir() else []:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        extra = payload.get("extra") or {}
        run_manifests.append(
            {
                "path": path.name,
                "duration_s": payload.get("duration_s"),
                "timing": extra.get("timing"),
                "lane": payload.get("lane"),
                "device": extra.get("device"),
                "gate": extra.get("gate"),
            }
        )
    return {
        "period_metrics": metrics,
        "matrix_state": state,
        "matrix_index": str(index_path) if index_path.is_file() else None,
        "matrix_plan": {key: matrix_plan.get(key) for key in ("extra", "duration_s", "lane", "config")},
        "group_manifests": group_manifests,
        "run_manifests": run_manifests,
    }


def collect_eval() -> dict[str, Any]:
    """Check the S5 R matrix completeness from the run_eval artifacts."""
    results_root = STAGE / "results"
    preferred = results_root / "tables" / "cl_matrix.json"
    if preferred.is_file():
        candidates = [preferred]
    else:
        candidates = sorted(results_root.rglob("cl_matrix.json")) if results_root.is_dir() else []
    if not candidates:
        return {"found": False, "reason": "no cl_matrix.json under stage/results"}
    chosen = candidates[0]
    payload = json.loads(chosen.read_text(encoding="utf-8"))
    matrix = (payload.get("matrix") or {}).get("all_classes") or []
    finite = 0
    total = 0
    for row in matrix:
        for value in row:
            total += 1
            if isinstance(value, (int, float)) and value == value:  # not NaN
                finite += 1
    cl_metrics = payload.get("cl_metrics_all_classes")
    audit_root = STAGE / "results" / "data_audit"
    eval_manifest = audit_root / "eval_manifest.json"
    if not eval_manifest.is_file():
        matches = sorted(audit_root.glob("eval_manifest_*.json"))
        eval_manifest = matches[0] if matches else None
    eval_duration = None
    if eval_manifest is not None and eval_manifest.is_file():
        try:
            eval_duration = json.loads(eval_manifest.read_text(encoding="utf-8")).get("duration_s")
        except Exception:
            eval_duration = None
    return {
        "eval_manifest": None if eval_manifest is None else str(eval_manifest),
        "eval_duration_s": eval_duration,
        "gpu_memory_after": SUMMARY.get("gpu_after", {}),
        "peak_memory_note": "peak memory is not instrumented in-process; post-run nvidia-smi used-MiB and the train/eval durations are reported instead",
        "found": True,
        "path": str(chosen),
        "strategy": payload.get("strategy"),
        "model": payload.get("model"),
        "seed": payload.get("seed"),
        "n_periods": len(matrix),
        "cell_total": total,
        "cell_defined": finite,
        "matrix_complete": total == 25 and finite == 25,
        "dual_macro_ok": payload.get("cl_metrics_all_classes") is not None,
        "cl_metrics_all_classes": cl_metrics,
        "supported_set": payload.get("supported_set"),
        "per_class_file": str(chosen.parent / "cl_per_class_f1.json"),
    }


def collect_graphs() -> dict[str, Any]:
    processed = STAGE / "data" / "processed" / "NF-UNSW-NB15-v3"
    manifest = processed / "graphs_manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() else {}
    periods = []
    for path in sorted(processed.glob("period_*.pt")):
        periods.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"processed_dir": str(processed), "manifest": payload, "periods": periods}


def make_output_zip() -> dict[str, Any]:
    """Zip evidence artifacts (results + runs + logs + summary), excluding large processed graphs."""
    if OUTPUT_ZIP.exists():
        OUTPUT_ZIP.unlink()
    include_roots = [("results", STAGE / "results"), ("runs", STAGE / "runs"), ("logs", LOGS_DIR)]
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for prefix, root in include_roots:
            if not root.exists():
                continue
            for path in sorted(root.rglob("*")):
                if not path.is_file():
                    continue
                # exclude checkpoint binaries > 50 MB to keep the download small; manifests/metrics stay
                if path.suffix in {".pt", ".pth", ".npz"} and path.stat().st_size > 50 * 1024 * 1024:
                    continue
                archive.write(path, arcname=f"{prefix}/{path.relative_to(root).as_posix()}")
        graphs = collect_graphs()
        archive.writestr("graphs_manifest.json", json.dumps(graphs, indent=2, sort_keys=True, default=str))
        summary_bytes = json.dumps(SUMMARY, indent=2, sort_keys=True, default=str)
        archive.writestr("bringup_summary.json", summary_bytes)
        archive.writestr("env_manifest.json", (STAGE / "env_manifest.json").read_text(encoding="utf-8") if (STAGE / "env_manifest.json").is_file() else "{}")
    digest = sha256_file(OUTPUT_ZIP)
    log(f"output zip: {OUTPUT_ZIP} ({OUTPUT_ZIP.stat().st_size} bytes, sha256 {digest})")
    return {"path": str(OUTPUT_ZIP), "bytes": OUTPUT_ZIP.stat().st_size, "sha256": digest}


def gpu_snapshot() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used", "--format=csv,noheader"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60,
        )
        return {"nvidia_smi": (completed.stdout or "").strip(), "returncode": completed.returncode}
    except Exception as exc:  # pragma: no cover - no GPU tools
        return {"nvidia_smi": None, "error": str(exc)}


def main() -> int:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY["gpu"] = gpu_snapshot()
    try:
        # --- inputs ---------------------------------------------------------
        env_manifest_path = find_input("env_manifest.json")
        env_manifest = json.loads(env_manifest_path.read_text(encoding="utf-8"))
        SUMMARY["input_verification"]["env_manifest"] = str(env_manifest_path)
        SUMMARY["input_verification"]["dataset_slug"] = env_manifest.get("dataset_slug")
        SUMMARY["input_verification"]["git_commit"] = env_manifest.get("git_commit")

        dataset_root = env_manifest_path.parent
        csv_source = find_input("NF-UNSW-NB15-v3.csv")

        csv_bytes = csv_source.stat().st_size
        if csv_bytes != int(env_manifest["csv"]["bytes"]):
            raise RuntimeError(f"raw CSV size mismatch: dataset {csv_bytes} != manifest {env_manifest['csv']['bytes']}")
        csv_sha = sha256_file(csv_source)
        if csv_sha != env_manifest["csv"]["sha256"]:
            raise RuntimeError(f"raw CSV sha256 mismatch: {csv_sha} != {env_manifest['csv']['sha256']}")
        if csv_bytes != EXPECTED_CSV_BYTES:
            raise RuntimeError(f"raw CSV size {csv_bytes} != expected {EXPECTED_CSV_BYTES}")

        # Kaggle auto-extracts uploaded .zip files: the code snapshot and frozen tree may
        # arrive either as zips or as extracted directories. Support both deliveries and
        # record which one was used.
        code_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir() and path.name.startswith("xdrift_repo_"))
        code_zips = sorted(dataset_root.glob("xdrift_repo_*.zip"))
        frozen_dir = dataset_root / "frozen"
        frozen_zips = sorted(dataset_root.glob("frozen.zip"))

        code_delivery = "extracted_tree" if code_dirs else "zip"
        code_name = (code_dirs[0].name if code_dirs else code_zips[0].name) if (code_dirs or code_zips) else None
        if code_name is None:
            raise RuntimeError("no code snapshot found in dataset (expected xdrift_repo_* dir or zip)")
        snapshot_info = dataset_root / "snapshot_info.txt"
        snapshot_meta = {}
        if snapshot_info.is_file():
            for line in snapshot_info.read_text(encoding="utf-8").splitlines():
                if "=" in line:
                    key, value = line.split("=", 1)
                    snapshot_meta[key.strip()] = value.strip()
        if snapshot_meta.get("snapshot_sha") and snapshot_meta["snapshot_sha"] != env_manifest.get("git_commit"):
            raise RuntimeError(f"snapshot_info commit {snapshot_meta['snapshot_sha']} != env_manifest {env_manifest.get('git_commit')}")

        # --- extract/copy code + seed frozen --------------------------------
        if STAGE.exists():
            shutil.rmtree(STAGE)
        STAGE.mkdir(parents=True, exist_ok=True)
        if code_dirs:
            shutil.copytree(code_dirs[0], STAGE, dirs_exist_ok=True)
        else:
            extract_zip(code_zips[0], STAGE)
        code_files = [path for path in STAGE.rglob("*") if path.is_file()]
        code_tree = hashlib.sha256()
        for path in sorted(code_files):
            code_tree.update(path.relative_to(STAGE).as_posix().encode("utf-8"))
            code_tree.update(b"\0")
            code_tree.update(sha256_file(path).encode("ascii"))
            code_tree.update(b"\n")
        log(f"code ({code_delivery}) copied to {STAGE}: {len(code_files)} files, tree sha256 {code_tree.hexdigest()[:12]}…")
        if frozen_dir.is_dir():
            frozen_delivery = "extracted_tree"
            frozen_root = frozen_dir
            for path in sorted(frozen_root.rglob("*")):
                if not path.is_file():
                    continue
                target = STAGE / path.relative_to(frozen_root)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        else:
            if not frozen_zips:
                raise RuntimeError("frozen artifacts not found in dataset (neither frozen/ dir nor frozen.zip)")
            frozen_delivery = "zip"
            frozen_root = frozen_zips[0]
            extract_zip(frozen_root, STAGE)
        manifest_source = dataset_root / "frozen_artifacts.sha256"
        if manifest_source.is_file():
            shutil.copy2(manifest_source, STAGE / "frozen_artifacts.sha256")
        seeded_frozen_hashes(env_manifest["frozen_zip"].get("files") or [])
        SUMMARY["input_verification"].update(
            {
                "csv_bytes": csv_bytes,
                "csv_sha256": csv_sha,
                "csv_verified": True,
                "code_delivery": code_delivery,
                "code_name": code_name,
                "code_files": len(code_files),
                "code_tree_sha256": code_tree.hexdigest(),
                "code_snapshot_sha": env_manifest["code_zip"].get("snapshot_sha"),
                "dirty_worktree_files_not_in_snapshot": env_manifest["code_zip"].get("dirty_worktree_files_not_in_snapshot"),
                "frozen_delivery": frozen_delivery,
                "frozen_source": str(frozen_root),
            }
        )
        log(f"inputs verified: CSV {csv_bytes} B, code={code_name} ({code_delivery}), frozen={frozen_delivery}")

        # --- raw data into the lane layout ---------------------------------
        raw_dir = STAGE / "data" / "raw" / "NF-UNSW-NB15-v3"
        raw_dir.mkdir(parents=True, exist_ok=True)
        raw_target = raw_dir / "NF-UNSW-NB15-v3.csv"
        shutil.copy2(csv_source, raw_target)
        if raw_target.stat().st_size != EXPECTED_CSV_BYTES:
            raise RuntimeError(f"staged CSV size {raw_target.stat().st_size} != {EXPECTED_CSV_BYTES}")
        (STAGE / "results").mkdir(parents=True, exist_ok=True)

        # --- environment ----------------------------------------------------
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

        needs_pyg = subprocess.run(
            [sys.executable, "-c", "import torch_geometric"], cwd=str(STAGE), env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=120,
        )
        if needs_pyg.returncode != 0:
            log("torch_geometric missing in the image; installing pinned PyG")
            run_step(
                "pip_pyg",
                [sys.executable, "-m", "pip", "install", "--quiet", REQUIRED_PYG],
                cwd=STAGE, env=env, timeout=1200,
            )
        else:
            SUMMARY["steps"]["pip_pyg"] = {"skipped": True, "reason": "torch_geometric already importable"}

        SUMMARY["environment"].update(query_environment(env))
        run_step(
            "check_env",
            [sys.executable, "src/scripts/check_env.py", "--require-cuda", "--json-out", str(STAGE / "results/env_check_kaggle.json")],
            cwd=STAGE, env=env, timeout=900,
        )
        env_check_path = STAGE / "results/env_check_kaggle.json"
        SUMMARY["environment"]["check_env"] = json.loads(env_check_path.read_text(encoding="utf-8")) if env_check_path.is_file() else {}

        # --- graphs ---------------------------------------------------------
        run_step(
            "build_graphs",
            [sys.executable, "src/scripts/build_graphs.py", "--lane", "kaggle"],
            cwd=STAGE, env=env, timeout=7200,
        )
        SUMMARY["graphs"] = collect_graphs()

        # --- one canary matrix group ---------------------------------------
        run_step(
            "canary_matrix",
            [
                sys.executable, "src/scripts/run_matrix.py",
                "--lane", "kaggle", "--device", "cuda",
                "--models", "egraphsage", "--strategies", "ft",
                "--seeds", "42", "--periods", "5", "--max-runs", "1",
                "--epoch-seconds-full-size", "4.0",  # diagnostic RTX-5060 anchor supplied by the task; T4 timings are measured below
            ],
            cwd=STAGE, env=env, timeout=8 * 3600,
        )
        SUMMARY["matrix"] = collect_metrics()

        # --- S5 evaluation for the group ------------------------------------
        run_step(
            "run_eval",
            [
                sys.executable, "src/scripts/run_eval.py",
                "--lane", "kaggle", "--device", "cuda",
                "--strategy", "ft", "--model", "egraphsage", "--seed", "42",
            ],
            cwd=STAGE, env=env, timeout=3600,
        )
        SUMMARY["eval"] = collect_eval()
        if not SUMMARY["eval"].get("matrix_complete"):
            raise RuntimeError(f"R matrix incomplete after run_eval: {json.dumps(SUMMARY['eval'])[:800]}")

        group_status = (SUMMARY["matrix"].get("matrix_state") or {}).get("groups") or {}
        SUMMARY["matrix"]["group_status"] = group_status
        SUMMARY["gpu_after"] = gpu_snapshot()
        SUMMARY["status"] = "success"
        SUMMARY["finished_at_utc"] = utc_now()
        SUMMARY["wall_clock_s"] = round((datetime.now(timezone.utc) - STARTED_AT).total_seconds(), 1)
        SUMMARY["oom"] = False
    except Exception as exc:  # noqa: BLE001 - report every failure explicitly
        SUMMARY["status"] = "failed"
        SUMMARY["error"] = f"{type(exc).__name__}: {exc}"
        SUMMARY["traceback"] = traceback.format_exc()[-6000:]
        SUMMARY["finished_at_utc"] = utc_now()
        SUMMARY["wall_clock_s"] = round((datetime.now(timezone.utc) - STARTED_AT).total_seconds(), 1)
        log(f"FAILED: {SUMMARY['error']}")
    finally:
        try:
            SUMMARY["gpu"] = SUMMARY.get("gpu") or gpu_snapshot()
            summary_step_logs = [{"step": item.get("step"), "returncode": item.get("returncode"), "duration_s": item.get("duration_s"), "log": item.get("log")} for item in STEP_LOGS]
            SUMMARY["per_step_logs"] = summary_step_logs
            if STAGE.exists():
                try:
                    SUMMARY["artifacts"] = make_output_zip()
                except Exception as zip_exc:  # pragma: no cover - zip failure path
                    SUMMARY["zip_error"] = f"{type(zip_exc).__name__}: {zip_exc}"
            SUMMARY_PATH.write_text(json.dumps(SUMMARY, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
            print("XDRIFT_BRINGUP_SUMMARY=" + json.dumps({k: SUMMARY.get(k) for k in ("status", "error", "wall_clock_s", "artifacts", "matrix", "eval")}, default=str)[:8000], flush=True)
            # keep the download small: remove the (large) stage tree after evidence is zipped
            if STAGE.exists():
                shutil.rmtree(STAGE, ignore_errors=True)
                SUMMARY["stage_removed_after_zip"] = True
                SUMMARY_PATH.write_text(json.dumps(SUMMARY, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        except Exception as exc:  # pragma: no cover - final reporting failure
            print(f"[bringup] FINAL REPORTING FAILED: {exc}", flush=True)
    return 0 if SUMMARY.get("status") == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
