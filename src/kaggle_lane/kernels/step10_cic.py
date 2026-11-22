#!/usr/bin/env python3
"""CICIDS2018 cross-dataset diagnostic for the step-06 classifiers.

The full 20.1M-row CIC CSV is 4.2 GB and does not fit the notebook memory
budget together with the cleaned frame.  This kernel therefore builds the
same deterministic 20% class-stratified sample that is committed under
``datasets/raw/NF-CSE-CIC-IDS2018-v3-sample``, runs steps 02/04/05, trains
three one-seed step-06 variants, then runs the post-hoc decoder comparison.

All outputs are diagnostic only: this is not the paper protocol and not the
signed S10/S11 analysis.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

WORK = Path("/kaggle/working/step10_cic")
TEMP = Path("/kaggle/temp/xdrift-cic")
DATASET = "NF-CSE-CIC-IDS2018-v3-sample"
SAMPLE_FRACTION = 0.20
CHUNK_SIZE = 1_000_000
SEED = 42


def log(message: str) -> None:
    print(f"[step10-cic] {message}", flush=True)


def find_one(pattern: str, root: Path = Path("/kaggle/input")) -> Path:
    matches = sorted(root.rglob(pattern))
    if not matches:
        raise FileNotFoundError(f"missing {pattern} under {root}")
    return matches[0]


def discover_code() -> Path:
    for candidate in Path("/kaggle/input").rglob("src/main/06_train.py"):
        return candidate.parents[2]
    raise FileNotFoundError("missing xdrift code snapshot")


def run(command: list[str], *, cwd: Path) -> None:
    log("$ " + " ".join(command))
    env = os.environ.copy()
    env["PYTHONPATH"] = str(cwd)
    env["XDRIFT_ROOT"] = str(TEMP)
    env["XDRIFT_LANE"] = "default"
    env["XDRIFT_DATA"] = str(cwd / "datasets")
    env["XDRIFT_RESULTS"] = str(cwd / "results")
    env["XDRIFT_RUNS"] = str(cwd / "results/text/runs")
    env["XDRIFT_CACHE"] = str(cwd / "results/text/cache")
    proc = subprocess.run(command, cwd=cwd, env=env, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(command)}")


def make_sample(raw_csv: Path, output: Path) -> dict[str, int]:
    """Deterministic 20% per-class sample, preserving chronological order."""
    import numpy as np
    import pandas as pd

    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    parts: list[pd.DataFrame] = []
    counts: dict[str, int] = {}
    for chunk in pd.read_csv(raw_csv, chunksize=CHUNK_SIZE, low_memory=False):
        for label, group in chunk.groupby("Attack", sort=False):
            take = max(int(round(len(group) * SAMPLE_FRACTION)), 1)
            take = min(take, len(group))
            chosen = np.sort(rng.choice(len(group), size=take, replace=False))
            parts.append(group.iloc[chosen])
            counts[str(label)] = counts.get(str(label), 0) + int(take)
    sample = pd.concat(parts, ignore_index=True)
    sample = sample.sort_values("FLOW_START_MILLISECONDS", kind="stable").reset_index(drop=True)
    sample.to_csv(output, index=False)
    log(f"sample rows={len(sample):,} bytes={output.stat().st_size:,}")
    return counts


def main() -> int:
    started = time.monotonic()
    WORK.mkdir(parents=True, exist_ok=True)
    TEMP.mkdir(parents=True, exist_ok=True)

    code = discover_code()
    log(f"code root {code}")
    repo = TEMP
    if (repo / "src").exists():
        shutil.rmtree(repo / "src")
    shutil.copytree(code / "src", repo / "src")
    shutil.copytree(code / "configs", repo / "configs")
    shutil.copytree(code / "platforms", repo / "platforms")
    shutil.copy2(code / "README.md", repo / "README.md")

    raw = find_one("NF-CICIDS2018-v3.csv", Path("/kaggle/input"))
    log(f"raw CIC {raw} ({raw.stat().st_size:,} bytes)")
    sample_csv = repo / "datasets/raw" / DATASET / f"{DATASET}.csv"
    sample_counts = make_sample(raw, sample_csv)

    # Kaggle's base image can omit PyG; install it only when missing.
    try:
        import torch_geometric  # noqa: F401
    except ImportError:
        log("installing torch-geometric==2.8.1")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "torch-geometric==2.8.1"],
            check=True,
        )

    run(
        [sys.executable, "src/main/01_setup_environment.py", "--dataset", DATASET, "--allow-cpu"],
        cwd=repo,
    )
    run([sys.executable, "src/main/02_clean_preprocess.py", "--dataset", DATASET, "--format", "pickle"], cwd=repo)
    run([sys.executable, "src/main/04_split_balance.py", "--dataset", DATASET], cwd=repo)
    run([sys.executable, "src/main/05_build_graphs.py", "--dataset", DATASET, "--node-features", "constant", "--overwrite"], cwd=repo)

    variants = [
        ("baseline", []),
        ("class_weight_cap_100", ["--class-weight-cap", "100"]),
        ("focal", ["--loss", "focal", "--focal-gamma", "2"]),
    ]
    training_runs = []
    for name, extra in variants:
        command = [
            sys.executable,
            "src/main/06_train.py",
            "--dataset",
            DATASET,
            "--node-features",
            "constant",
            "--depths",
            "2",
            "--seeds",
            "42",
            "--strategy",
            "ft",
            "--save-checkpoints",
        ] + extra
        run(command, cwd=repo)
        runs_csv = repo / "results/text/train" / DATASET / "runs.csv"
        training_runs.append({"variant": name, "runs_csv": str(runs_csv)})

    # Compare fine vs prior decoding on the saved baseline checkpoints.
    run(
        [
            sys.executable,
            "src/main/10_decoder_compare.py",
            "--dataset",
            DATASET,
            "--checkpoint-root",
            str(repo / "datasets/processed" / DATASET / "models"),
            "--graph-root",
            str(repo / "datasets/graphs"),
            "--methods",
            "fine,prior",
            "--seeds",
            "42",
            "--out",
            str(repo / "results/text/tables/decoder_compare_cic"),
        ],
        cwd=repo,
    )

    # Collect only small artifacts; raw/sample CSV and model weights stay out.
    export = WORK
    export.mkdir(parents=True, exist_ok=True)
    for relative in [
        "results/text/train",
        "results/text/tables/decoder_compare_cic",
        "results/text/data_audit/clean",
        "results/text/data_audit/split",
        "results/text/data_audit/graphs",
        "results/images/train",
        "results/images/split",
        "results/images/graphs",
        "analysis/train",
    ]:
        source = repo / relative
        if source.exists():
            target = export / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target, dirs_exist_ok=True)
    (export / "sample_counts.json").write_text(json.dumps(sample_counts, indent=2), encoding="utf-8")
    (export / "kernel_summary.json").write_text(
        json.dumps(
            {
                "dataset": DATASET,
                "sample_fraction": SAMPLE_FRACTION,
                "seed": SEED,
                "variants": [name for name, _ in variants],
                "seconds": round(time.monotonic() - started, 2),
                "note": "20% stratified diagnostic sample; not the paper protocol",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log(f"done in {time.monotonic() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
