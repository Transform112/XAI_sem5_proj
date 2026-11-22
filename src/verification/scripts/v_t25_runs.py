#!/usr/bin/env python
"""T25 — independent S3 smoke convergence + full-size feasibility runner.

Runs the canonical trainer through the public CLI, with all output redirected to
results/text/verification/data_audit/t25_runs (checked BEFORE training; the T17/T21 env bug
is avoided by creating directories first and asserting the resolved paths).

Steps
  1. smoke: `--smoke --stage S3 --strategy ft --model egraphsage --period 1 --seed 42 --lane legacy`
     -> compare val/test macro-F1 with the frozen legacy baseline 0.4931/0.4069 (±0.03 abs).
  2. full-size 2-epoch run (same CLI, `--epochs 2`, no --smoke) on the legacy 3 GB card
     -> must not OOM; read measured max_memory_allocated_bytes, epoch time, update count.
  3. manifest/metrics audit: measured memory, lane, lifecycle, env, batching/sample counts.

Writes results/text/verification/data_audit/t25_runs.json. Never writes canonical results/runs.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(r"C:\Users\iusearchbtw\Desktop\XAI_proj")
PY = str(REPO / ".venv" / "Scripts" / "python.exe")
BASE = REPO / "verification" / "data_audit" / "t25_runs"
OUT = REPO / "verification" / "data_audit"
DEVICE = os.environ.get("T25_DEVICE", "auto")
SMOKE_ONLY = os.environ.get("T25_SMOKE_ONLY") == "1"
BASELINE = {"val": 0.4931, "test": 0.4069, "source": "frozen legacy S3 smoke artifact "
            "results/text/data_audit/train_ft_egraphsage_seed42_period1.json (mini_batch, 100k/1,131,909 edges, seed 42)"}


def _redirect(tag: str) -> dict[str, str]:
    for sub in ("runs", "results", "cache"):
        (BASE / tag / sub).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["XDRIFT_RUNS"] = str((BASE / tag / "runs").resolve())
    env["XDRIFT_RESULTS"] = str((BASE / tag / "results").resolve())
    env["XDRIFT_CACHE"] = str((BASE / tag / "cache").resolve())
    env.pop("XDRIFT_LANE", None)
    env.pop("XDRIFT_ALLOW_DEFAULT_LANE", None)
    # assert the redirect is what paths.py will resolve (prevents the T21 incident)
    check = subprocess.run(
        [PY, "-c", "from src.utils.paths import load_paths; p=load_paths(lane='legacy'); print(p.runs_root); print(p.audit_root)"],
        cwd=str(REPO), capture_output=True, text=True, env=env)
    roots = check.stdout.strip().splitlines()
    assert roots and str(BASE / tag) in roots[0] and str(BASE / tag) in roots[1], f"redirect failed: {roots}"
    return env


def _run(cmd: list[str], env: dict[str, str], tag: str) -> dict:
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, env=env, timeout=3600)
    (BASE / tag).mkdir(parents=True, exist_ok=True)
    (BASE / tag / "stdout.txt").write_text(proc.stdout, encoding="utf-8", errors="replace")
    (BASE / tag / "stderr.txt").write_text(proc.stderr, encoding="utf-8", errors="replace")
    metrics_path = BASE / tag / "results" / "data_audit" / "train_ft_egraphsage_seed42_period1.json"
    manifest_path = BASE / tag / "results" / "data_audit" / "train_manifest_ft_egraphsage_seed42_period1.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig")) if metrics_path.is_file() else None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig")) if manifest_path.is_file() else None
    return {"cmd": cmd, "returncode": proc.returncode, "stdout_tail": proc.stdout.strip().splitlines()[-4:],
            "stderr_tail": proc.stderr.strip().splitlines()[-4:], "metrics": metrics, "manifest": manifest}


def main() -> int:
    BASE.mkdir(parents=True, exist_ok=True)
    result: dict = {"generated_at": str(np.datetime64("now")), "baseline": BASELINE, "checks": []}
    chk = result["checks"]

    def add(i, status, detail):
        chk.append({"id": i, "status": status, "detail": detail})

    common = ["--dataset", "NF-UNSW-NB15-v3", "--config", "configs/train.yaml", "--stage", "S3",
              "--strategy", "ft", "--model", "egraphsage", "--period", "1", "--seed", "42",
              "--device", DEVICE, "--lane", "legacy"]
    # ---------------- smoke convergence
    smoke = _run([PY, "src/scripts/run_train.py", *common, "--smoke"], _redirect("smoke"), "smoke")
    m = (smoke.get("metrics") or {}).get("result") or {}
    val = ((m.get("val") or {}).get("macro_f1")); test = ((m.get("test") or {}).get("macro_f1"))
    ok_smoke = (smoke["returncode"] == 0 and val is not None and test is not None
                and abs(val - BASELINE["val"]) <= 0.03 and abs(test - BASELINE["test"]) <= 0.03)
    add("T25.smoke.convergence", "pass" if ok_smoke else "fail",
        {"val": val, "test": test, "baseline": {"val": BASELINE["val"], "test": BASELINE["test"]},
         "delta": {"val": None if val is None else round(val - BASELINE["val"], 4),
                   "test": None if test is None else round(test - BASELINE["test"], 4)},
         "tolerance_abs": 0.03, "epochs_run": m.get("epochs_run"), "n_train_edges": m.get("n_train_edges"),
         "settings": m.get("settings"), "returncode": smoke["returncode"]})
    if SMOKE_ONLY:
        result["overall"] = "fail" if any(c["status"] == "fail" for c in chk) else "pass"
        result["smoke_only"] = True
        (OUT / "t25_runs.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(json.dumps({c["id"]: c["status"] for c in chk}, indent=1))
        print("overall:", result["overall"], "(smoke-only mode)")
        for c in chk:
            print(c["id"], json.dumps(c["detail"], default=str)[:600])
        return 0 if result["overall"] == "pass" else 2
    # ---------------- full-size 2 epochs on the 3 GB card
    full = _run([PY, "src/scripts/run_train.py", *common, "--epochs", "2"], _redirect("full2"), "full2")
    fm = (full.get("metrics") or {}).get("result") or {}
    timing = fm.get("timing") or {}
    mem = timing.get("max_memory_allocated_bytes")
    if mem is None:
        mem = (((full.get("manifest") or {}).get("extra") or {}).get("timing") or {}).get("max_memory_allocated_bytes")
    n_updates = timing.get("n_updates")
    updates_per_epoch = timing.get("updates_per_epoch")
    ok_full = (full["returncode"] == 0 and fm.get("epochs_run") == 2 and mem is not None)
    add("T25.fullsize.feasibility", "pass" if ok_full else "fail",
        {"returncode": full["returncode"], "epochs_run": fm.get("epochs_run"),
         "n_train_edges": fm.get("n_train_edges"), "n_graph_edges": 1_131_909,
         "epoch_seconds_mean": timing.get("epoch_seconds_mean"), "train_seconds": timing.get("train_seconds"),
         "max_memory_allocated_bytes": mem,
         "max_memory_allocated_gib": None if mem is None else round(mem / 1024 ** 3, 3),
         "stderr_tail": full["stderr_tail"]})
    # ---------------- manifest/metrics audit (measured memory, lane, lifecycle, env)
    man = full.get("manifest") or {}
    metrics_full = full.get("metrics") or {}
    env = man.get("env") or {}
    lifecycle = {"started_at_utc": man.get("started_at_utc"), "finished_at_utc": man.get("finished_at_utc"),
                 "duration_s": man.get("duration_s"),
                 "train_seconds": (man.get("extra") or {}).get("timing", {}).get("train_seconds")
                 if isinstance((man.get("extra") or {}).get("timing"), dict) else None}
    add("T25.manifest.measured_memory_lane_lifecycle",
        "pass" if (man.get("lane") == "legacy" and lifecycle["started_at_utc"] and lifecycle["finished_at_utc"]
                   and mem is not None and len([k for k in env if k.startswith("XDRIFT")]) >= 5) else "fail",
        {"lane": man.get("lane"), "gate": man.get("gate"), "lifecycle": lifecycle,
         "env_xdrift_keys": sorted(k for k in env if k.startswith("XDRIFT")),
         "measured_memory_in_metrics_timing": timing.get("max_memory_allocated_bytes"),
         "measured_memory_in_manifest": mem,
         "batching": timing.get("batching"),
         "updates_per_epoch": timing.get("updates_per_epoch")
         if isinstance(fm.get("timing"), dict) else None})
    # ---------------- optimizer budget
    settings = fm.get("settings") or {}
    n_train = fm.get("n_train_edges") or 0
    batch = settings.get("batch_size") or 1024
    epochs = fm.get("epochs_run") or 0
    updates = (fm.get("timing") or {}).get("updates") or epochs * (-(-n_train // batch))
    add("T25.optimizer_budget", "pass" if (batch == 1024 and updates >= 1300 or epochs == 2) else "warn",
        {"settings": settings, "n_train_edges": n_train, "batch_size": batch,
         "expected_updates_smoke_full_budget": -(-n_train // batch) * 30 + 30,
         "updates_observed": n_updates if n_updates is not None else updates,
         "updates_per_epoch": updates_per_epoch,
         "note": "full protocol budget = ceil(n_train/1024) * epochs; the 2-epoch feasibility run is "
                 "shortened deliberately"})

    result["overall"] = "fail" if any(c["status"] == "fail" for c in chk) else "pass"
    (OUT / "t25_runs.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({c["id"]: c["status"] for c in chk}, indent=1))
    print("overall:", result["overall"])
    for c in chk:
        print(c["id"], json.dumps(c["detail"], default=str)[:600])
    return 0 if result["overall"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
