"""Kaggle full seed-42 explanation corpus for the new main pipeline.

Stages the code/raw/frozen splits, builds cleaned tables, splits, constant
graphs and one checkpointed depth-2 run, then writes the full five-period
IG + GNNExplainer corpus.

Run as a script kernel with:
  * ``mysteriousavailable/xdrift-nf-unsw-c1-inputs`` (raw CSV + frozen splits)
  * ``mysteriousavailable/xdrift-main-code`` (repository snapshot at step 07)
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
DATASET = "NF-UNSW-NB15-v3"
EXPORT = Path("/kaggle/working/step07_corpus")
SUMMARY: dict[str, object] = {
    "steps": [],
    "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
}


def log(message: str) -> None:
    print(f"[step07-corpus] {message}", flush=True)


def run(name: str, args: list[str], *, timeout: int = 10800) -> dict[str, object]:
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
    entry = {
        "step": name,
        "seconds": elapsed,
        "returncode": result.returncode,
        "tail": (result.stdout or "")[-3000:],
    }
    SUMMARY["steps"].append(entry)
    if result.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {result.returncode}")
    log(f"DONE {name} in {elapsed}s")
    return entry


def discover_code_root() -> Path:
    candidates = sorted(INPUT.rglob("07_explain.py"))
    for candidate in candidates:
        if candidate.parent.name == "main" and (candidate.parents[2] / "configs").is_dir():
            return candidate.parents[2]
    zips = sorted(INPUT.rglob("xdrift_repo.zip"))
    if zips:
        with zipfile.ZipFile(zips[0]) as archive:
            archive.extractall(ROOT)
        return ROOT
    raise FileNotFoundError("cannot find src/main/07_explain.py or xdrift_repo.zip")


def discover_data_file() -> Path:
    matches = sorted(INPUT.rglob(f"{DATASET}.csv"))
    if not matches:
        raise FileNotFoundError(f"cannot find {DATASET}.csv under {INPUT}")
    return matches[0]


def discover_frozen_splits() -> Path:
    candidates = [p for p in INPUT.rglob("splits") if p.is_dir() and (p / DATASET).is_dir()]
    if not candidates:
        raise FileNotFoundError("cannot find frozen splits under /kaggle/input")
    return candidates[0]


def prepare_workspace() -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    code_root = discover_code_root()
    if code_root != ROOT:
        shutil.copytree(code_root, ROOT, dirs_exist_ok=True)
    log(f"code staged from {code_root}")
    raw = discover_data_file()
    raw_dir = ROOT / "datasets" / "raw" / DATASET
    raw_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw, raw_dir / f"{DATASET}.csv")
    log(f"raw CSV staged from {raw}: {(raw_dir / f'{DATASET}.csv').stat().st_size} bytes")
    splits = discover_frozen_splits()
    shutil.copytree(splits, ROOT / "datasets" / "splits", dirs_exist_ok=True)
    log(f"frozen splits staged from {splits}")
    os.environ.update(
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
    run("pip_torch_geometric", ["-m", "pip", "install", "--quiet", "torch-geometric==2.8.1"], timeout=1200)


def export_tree() -> None:
    if EXPORT.exists():
        shutil.rmtree(EXPORT)
    EXPORT.mkdir(parents=True, exist_ok=True)
    for rel in (
        "results/text/train",
        "results/text/data_audit/explanations",
        "results/text/tables/explain_index_main.csv",
        "results/text/explanations",
        "results/text/env_checks",
        "datasets/processed/NF-UNSW-NB15-v3/explanations/anchors",
        "datasets/processed/NF-UNSW-NB15-v3/models",
    ):
        src = ROOT / rel
        if src.is_file():
            target = EXPORT / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
        elif src.exists():
            shutil.copytree(src, EXPORT / rel, dirs_exist_ok=True)
    archive = shutil.make_archive(str(EXPORT / "explanation_bundle"), "zip", root_dir=ROOT, base_dir="results/text")
    log(f"wrote bundle {archive}")


def main() -> int:
    started = time.monotonic()
    prepare_workspace()
    ensure_pyg()
    run("01_setup_environment", ["src/main/01_setup_environment.py"], timeout=600)
    run("02_clean_preprocess", ["src/main/02_clean_preprocess.py", "--dataset", DATASET], timeout=1800)
    run("04_split_balance", ["src/main/04_split_balance.py", "--dataset", DATASET], timeout=1800)
    run("05_build_graphs", ["src/main/05_build_graphs.py", "--dataset", DATASET,
                            "--node-features", "constant", "--overwrite"], timeout=3600)
    run("06_train_checkpoint", ["src/main/06_train.py", "--dataset", DATASET,
                                "--node-features", "constant", "--depths", "2",
                                "--seeds", "42", "--strategy", "ft",
                                "--save-checkpoints"], timeout=9000)
    run("07_explain_corpus", ["src/main/07_explain.py", "--dataset", DATASET,
                              "--run-id", "ft-none-d2-s42", "--graph-mode", "constant",
                              "--periods", "1,2,3,4,5",
                              "--explainers", "ig,gnn_explainer",
                              "--gnn-epochs", "100"], timeout=10800)
    export_tree()
    SUMMARY["elapsed_seconds"] = round(time.monotonic() - started, 2)
    SUMMARY["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (EXPORT / "corpus_summary.json").write_text(json.dumps(SUMMARY, indent=2), encoding="utf-8")
    log(f"exported corpus artifacts to {EXPORT}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # pragma: no cover - Kaggle-side diagnostics
        EXPORT.mkdir(parents=True, exist_ok=True)
        (EXPORT / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        (EXPORT / "corpus_summary.json").write_text(json.dumps(SUMMARY, indent=2), encoding="utf-8")
        log(f"FAILED: {exc}")
        raise
