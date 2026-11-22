"""Kaggle full-matrix run for the new main pipeline (steps 02 -> 04 -> 05 -> 06).

Run as a script kernel with two private datasets mounted:
  * ``mysteriousavailable/xdrift-nf-unsw-c1-inputs`` (raw CSV + frozen splits)
  * ``mysteriousavailable/xdrift-main-code`` (repository snapshot)

The kernel keeps the working tree outside ``/kaggle/working`` so Kaggle only
exports the small result bundle at the end.  It trains all node-feature/depth seed combinations (45 runs) in one kernel.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from pathlib import Path


ROOT = Path("/kaggle/temp/xdrift")
INPUT = Path("/kaggle/input")
CODE_ROOT = INPUT / "xdrift-main-code" / "xdrift_repo"
CODE_ZIP = INPUT / "xdrift-main-code" / "xdrift_repo.zip"
DATA_DIR = INPUT / "xdrift-nf-unsw-c1-inputs"
DATASET = "NF-UNSW-NB15-v3"
EXPORT = Path("/kaggle/working/step06_matrix")
SUMMARY: dict[str, object] = {"steps": [], "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}

# Kaggle's mount names can change with dataset versions; discover the actual
# code tree, raw CSV and frozen splits instead of assuming one prefix.
def _discover_code_root() -> Path:
    if (CODE_ROOT / "src" / "main" / "06_train.py").is_file():
        return CODE_ROOT
    matches = sorted(INPUT.rglob("06_train.py"))
    for match in matches:
        if match.parent.name == "main" and (match.parents[2] / "configs").is_dir():
            return match.parents[2]
    raise FileNotFoundError(f"cannot find src/main/06_train.py under {INPUT}")


def _discover_data_dir() -> Path:
    direct = DATA_DIR / f"{DATASET}.csv"
    if direct.is_file():
        return DATA_DIR
    matches = sorted(INPUT.rglob(f"{DATASET}.csv"))
    if not matches:
        raise FileNotFoundError(f"cannot find {DATASET}.csv under {INPUT}")
    return matches[0].parent


def _discover_frozen_splits() -> Path:
    direct = DATA_DIR / "frozen" / "data" / "splits"
    if direct.is_dir():
        return direct
    candidates = [p for p in INPUT.rglob("splits") if p.is_dir() and (p / DATASET).is_dir()]
    if not candidates:
        raise FileNotFoundError(f"cannot find frozen splits under {INPUT}")
    return candidates[0]


def log(message: str) -> None:
    print(f"[step06-matrix] {message}", flush=True)


def run(name: str, args: list[str], *, timeout: int = 7200) -> dict[str, object]:
    log(f"START {name}: {' '.join(args)}")
    started = time.monotonic()
    result = subprocess.run(
        [sys.executable, *args],
        cwd=str(ROOT),
        env=os.environ.copy(),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    elapsed = round(time.monotonic() - started, 2)
    print(result.stdout or "", flush=True)
    entry = {"step": name, "seconds": elapsed, "returncode": result.returncode, "tail": (result.stdout or "")[-3000:]}
    SUMMARY["steps"].append(entry)
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")
    log(f"DONE {name} in {elapsed}s")
    return entry


def prepare_workspace() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    code_root = _discover_code_root()
    if code_root.is_dir():
        shutil.copytree(code_root, ROOT, dirs_exist_ok=True)
        log(f"code staged from extracted dataset folder {code_root}")
    elif CODE_ZIP.is_file():
        with zipfile.ZipFile(CODE_ZIP) as archive:
            archive.extractall(ROOT)
        log(f"code extracted from {CODE_ZIP}")
    else:
        raise FileNotFoundError(f"missing code snapshot folder {code_root} or zip {CODE_ZIP}")

    data_dir = _discover_data_dir()
    raw = data_dir / f"{DATASET}.csv"
    if not raw.is_file():
        raise FileNotFoundError(f"missing raw CSV: {raw}")
    raw_dir = ROOT / "datasets" / "raw" / DATASET
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw, raw_dir / f"{DATASET}.csv")
    log(f"raw CSV staged from {raw}: {(raw_dir / f'{DATASET}.csv').stat().st_size} bytes")

    frozen_splits = _discover_frozen_splits()
    if not frozen_splits.is_dir():
        raise FileNotFoundError(f"missing frozen splits: {frozen_splits}")
    target = ROOT / "datasets" / "splits"
    shutil.copytree(frozen_splits, target, dirs_exist_ok=True)
    log(f"frozen splits staged from {frozen_splits} under {target}")

    env = os.environ.copy()
    env.update(
        {
            "XDRIFT_ROOT": str(ROOT),
            "XDRIFT_DATA": str(ROOT / "datasets"),
            "XDRIFT_RUNS": str(ROOT / "results" / "text" / "runs"),
            "XDRIFT_CACHE": str(ROOT / "cache"),
            "XDRIFT_RESULTS": str(ROOT / "results"),
            "XDRIFT_LANE": "kaggle",
            "PYTHONPATH": str(ROOT),
            "PYTHONUNBUFFERED": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        }
    )
    os.environ.update(env)
    log(f"workspace ready: {ROOT}")


def ensure_pyg() -> None:
    probe = subprocess.run(
        [sys.executable, "-c", "import torch, torch_geometric; print(torch.__version__, torch_geometric.__version__)"],
        cwd=str(ROOT),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    print(probe.stdout or "", flush=True)
    if probe.returncode == 0:
        return
    log("torch_geometric missing; installing a compatible wheel")
    run("pip_torch_geometric", ["-m", "pip", "install", "--quiet", "torch-geometric==2.8.1"], timeout=1200)


def main() -> int:
    started = time.monotonic()
    prepare_workspace()
    ensure_pyg()
    run("01_setup_environment", ["src/main/01_setup_environment.py"], timeout=600)
    run("02_clean_preprocess", ["src/main/02_clean_preprocess.py", "--dataset", DATASET], timeout=1800)
    run("04_split_balance", ["src/main/04_split_balance.py", "--dataset", DATASET], timeout=1800)
    run(
        "05_build_graphs",
        [
            "src/main/05_build_graphs.py",
            "--dataset", DATASET,
            "--node-features", "constant,reputation,constant+reputation",
            "--overwrite",
        ],
        timeout=3600,
    )
    run(
        "06_train_matrix",
        [
            "src/main/06_train.py",
            "--dataset", DATASET,
            "--node-features", "constant,reputation,constant+reputation",
            "--depths", "2,3,4",
            "--seeds", "42,43,44,45,46",
            "--strategy", "ft",
        ],
        timeout=10800,
    )

    # Keep only the small, useful artifacts in the exported bundle.
    if EXPORT.exists():
        shutil.rmtree(EXPORT)
    EXPORT.mkdir(parents=True, exist_ok=True)
    for rel in (
        "results/text/train",
        "results/text/data_audit/graphs",
        "results/images/train",
        "results/images/graphs",
        "analysis/train",
        "results/text/env_checks",
    ):
        src = ROOT / rel
        if src.exists():
            shutil.copytree(src, EXPORT / rel, dirs_exist_ok=True)
    SUMMARY["elapsed_seconds"] = round(time.monotonic() - started, 2)
    SUMMARY["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (EXPORT / "matrix_summary.json").write_text(json.dumps(SUMMARY, indent=2), encoding="utf-8")
    log(f"exported pilot artifacts to {EXPORT}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - Kaggle-side diagnostics
        EXPORT.mkdir(parents=True, exist_ok=True)
        (EXPORT / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        (EXPORT / "matrix_summary.json").write_text(json.dumps(SUMMARY, indent=2), encoding="utf-8")
        log(f"FAILED: {exc}")
        raise
