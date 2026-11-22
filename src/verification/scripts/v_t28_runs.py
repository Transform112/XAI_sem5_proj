#!/usr/bin/env python
"""T28 (T27 re-audit) — independent CLI evidence on the pinned tree.

Runs, with all outputs redirected to verification/ (asserted before each run):
  1. canonical `graph_batch` smoke (CPU, 100k, seed 42, 30 epochs)  -> expect exact
     frozen reproduction val 0.493075 / test 0.406881
  2. `sampled` T24 mode smoke (CPU)                                  -> within ±0.03 of
     0.4715/0.4162 (T24/T25 sampled reference)
  3. full-size P1 2-epoch GPU determinism/timing for E-GS (twice)    -> checkpoint SHA
     matches engineer reference DC93B08... and both repeats
  4. same for GCN                                                    -> 29419F4...
Plus updates/epoch (774), measured max_memory_allocated_bytes, phase timings,
local (sampled) vs full (graph_batch) evaluation mode, and training-artefact fields.

Env: T28_REPO=<pinned tree>, T28_BASE=<output dir>, T28_OUT=<evidence dir>
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(os.environ.get("T28_REPO", r"C:\Users\iusearchbtw\Desktop\XAI_proj"))
PY = str(Path(r"C:\Users\iusearchbtw\Desktop\XAI_proj") / ".venv" / "Scripts" / "python.exe")
BASE = Path(os.environ.get("T28_BASE", r"C:\Users\iusearchbtw\Desktop\XAI_proj\verification\data_audit\t28_runs"))
OUT = Path(os.environ.get("T28_OUT", r"C:\Users\iusearchbtw\Desktop\XAI_proj\verification\data_audit"))
EXPECTED = {
    "sampled_ref": {"val": 0.4715, "test": 0.4162},
    "graph_batch_smoke": {"val": 0.493075, "test": 0.406881},
    "egs_full": {"epoch_seconds_mean": 3.664, "max_memory_allocated_bytes": 615_159_296,
                 "checkpoint_sha256": "dc93b0868cb44fe889b4c41a1f7fb4d3ba86432c430d196db9c86d590dbeb757"},
    "gcn_full": {"epoch_seconds_mean": 7.352, "max_memory_allocated_bytes": 638_001_664,
                 "checkpoint_sha256": "29419f465028d62b57901f56f64095415bac6205f8769a9822e480264f6721f0"},
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def redirect(tag: str) -> dict:
    for sub in ("runs", "results", "cache"):
        (BASE / tag / sub).mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["XDRIFT_DATA"] = str(Path(r"C:\Users\iusearchbtw\Desktop\XAI_proj") / "data")
    env["XDRIFT_RUNS"] = str((BASE / tag / "runs").resolve())
    env["XDRIFT_RESULTS"] = str((BASE / tag / "results").resolve())
    env["XDRIFT_CACHE"] = str((BASE / tag / "cache").resolve())
    env["XDRIFT_LANE"] = "legacy"
    env.pop("XDRIFT_ALLOW_DEFAULT_LANE", None)
    # verify redirect before any training
    proc = subprocess.run(
        [PY, "-c", "from src.utils.paths import load_paths; p=load_paths(lane='legacy'); "
                    "print(p.runs_root); print(p.audit_root)"],
        cwd=str(REPO), capture_output=True, text=True, env=dict(env, PYTHONPATH=str(REPO)))
    if str(BASE / tag) not in proc.stdout:
        raise RuntimeError(f"redirect not honoured: {proc.stdout} {proc.stderr}")
    return env


def run(tag: str, args: list[str]) -> dict:
    env = redirect(tag)
    cmd = [PY, str(REPO / "scripts" / "run_train.py"), "--dataset", "NF-UNSW-NB15-v3",
           "--config", str(REPO / "configs" / "train.yaml"), "--stage", "S3", "--strategy", "ft",
           "--model", args_model(args), "--period", "1", "--seed", "42", "--lane", "legacy",
           "--device", args_device(args)] + args
    proc = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True, env=env, timeout=3600)
    (BASE / tag).mkdir(parents=True, exist_ok=True)
    (BASE / tag / "stdout.txt").write_text(proc.stdout, encoding="utf-8", errors="replace")
    (BASE / tag / "stderr.txt").write_text(proc.stderr, encoding="utf-8", errors="replace")
    model = args_model(args)
    res_dir = BASE / tag / "results" / "data_audit"
    met = res_dir / f"train_ft_{model}_seed42_period1.json"
    man = res_dir / f"train_manifest_ft_{model}_seed42_period1.json"
    metrics = json.loads(met.read_text(encoding="utf-8-sig")) if met.is_file() else None
    manifest = json.loads(man.read_text(encoding="utf-8-sig")) if man.is_file() else None
    ckpt = BASE / tag / "runs" / "ft" / model / "seed42" / "M_1.pt"
    return {"cmd": cmd, "returncode": proc.returncode, "metrics": metrics, "manifest": manifest,
            "checkpoint_sha256": sha(ckpt) if ckpt.is_file() else None,
            "stderr_tail": proc.stderr.strip().splitlines()[-4:]}


def args_model(args: list[str]) -> str:
    try:
        return args[args.index("--model") + 1]
    except (ValueError, IndexError):
        return "egraphsage"


def args_device(args: list[str]) -> str:
    try:
        return args[args.index("--device") + 1]
    except (ValueError, IndexError):
        return "auto"


def summarize(run: dict) -> dict:
    m = (run.get("metrics") or {}).get("result") or {}
    t = m.get("timing") or {}
    val = (m.get("val") or {}); test = (m.get("test") or {})
    return {"returncode": run["returncode"], "epochs_run": m.get("epochs_run"),
            "n_train_edges": m.get("n_train_edges"),
            "updates_per_epoch": t.get("updates_per_epoch"), "n_updates": t.get("n_updates"),
            "batching": t.get("batching"), "training_mode": t.get("training_mode"),
            "epoch_seconds_mean": t.get("epoch_seconds_mean"), "train_seconds": t.get("train_seconds"),
            "sample_seconds": t.get("sample_seconds"), "forward_backward_seconds": t.get("forward_backward_seconds"),
            "eval_seconds": t.get("eval_seconds"),
            "max_memory_allocated_bytes": t.get("max_memory_allocated_bytes"),
            "memory_device": t.get("memory_device"),
            "val_eval_mode": val.get("evaluation_mode"), "test_eval_mode": test.get("evaluation_mode"),
            "val_macro_f1": val.get("macro_f1"), "test_macro_f1": test.get("macro_f1"),
            "checkpoint_sha256": run.get("checkpoint_sha256"),
            "lane": (run.get("manifest") or {}).get("lane"),
            "gate_real_lane": ((run.get("manifest") or {}).get("gate") or {}).get("real_lane"),
            "manifest_status": (run.get("manifest") or {}).get("status"),
            "static_aggregation_cache": t.get("static_aggregation_cache"),
            "graph_batch_optimized": t.get("graph_batch_optimized")}


def main() -> int:
    BASE.mkdir(parents=True, exist_ok=True)
    res: dict = {"repo": str(REPO), "checks": [], "runs": {}}
    chk = res["checks"]

    def add(i, status, detail):
        chk.append({"id": i, "status": status, "detail": detail})

    # 1. canonical graph_batch smoke (CPU)
    gb_smoke = run("smoke_graph_batch", ["--smoke", "--device", "cpu", "--epochs", "30"])
    s = summarize(gb_smoke)
    res["runs"]["graph_batch_smoke"] = s
    exact = (s["val_macro_f1"] is not None and s["test_macro_f1"] is not None
             and abs(s["val_macro_f1"] - EXPECTED["graph_batch_smoke"]["val"]) <= 1e-6
             and abs(s["test_macro_f1"] - EXPECTED["graph_batch_smoke"]["test"]) <= 1e-6)
    add("T28.smoke.graph_batch_exact", "pass" if exact else "fail",
        {"observed": {"val": s["val_macro_f1"], "test": s["test_macro_f1"]},
         "expected": EXPECTED["graph_batch_smoke"], "match_1e-6": exact})

    # 2. sampled T24-mode smoke (CPU)
    sm_smoke = run("smoke_sampled", ["--smoke", "--device", "cpu", "--epochs", "30", "--batching", "sampled"])
    s = summarize(sm_smoke)
    res["runs"]["sampled_smoke"] = s
    within = (s["val_macro_f1"] is not None and s["test_macro_f1"] is not None
              and abs(s["val_macro_f1"] - EXPECTED["sampled_ref"]["val"]) <= 0.03
              and abs(s["test_macro_f1"] - EXPECTED["sampled_ref"]["test"]) <= 0.03)
    add("T28.smoke.sampled_within_0.03", "pass" if within else "fail",
        {"observed": {"val": s["val_macro_f1"], "test": s["test_macro_f1"]},
         "reference_t24": EXPECTED["sampled_ref"], "within_0.03": within,
         "note": "sampled mode retained as alternative/robustness mode (D-059)"})

    # 3. full-size graph_batch 2 epochs GPU, two repeats, E-GS then GCN
    for model, key in (("egraphsage", "egs_full"), ("gcn", "gcn_full")):
        reps = []
        for rep in ("a", "b"):
            r = run(f"full_{model}_{rep}", ["--model", model, "--device", "cuda", "--epochs", "2"])
            s = summarize(r)
            reps.append(s)
            res["runs"][f"full_{model}_{rep}"] = s
        exp = EXPECTED[key]
        det = reps[0]["checkpoint_sha256"] == reps[1]["checkpoint_sha256"] == exp["checkpoint_sha256"]
        timing_ok = (reps[0]["epoch_seconds_mean"] is not None
                     and abs(reps[0]["epoch_seconds_mean"] - exp["epoch_seconds_mean"])
                     <= max(0.5, 0.15 * exp["epoch_seconds_mean"]))
        mem_ok = (reps[0]["max_memory_allocated_bytes"] is not None
                  and abs(reps[0]["max_memory_allocated_bytes"] - exp["max_memory_allocated_bytes"]) <= 10 * 1024 ** 2)
        budget_ok = reps[0]["updates_per_epoch"] == 774 and reps[0]["n_updates"] == 1548
        add(f"T28.fullsize.{model}", "pass" if (det and timing_ok and mem_ok and budget_ok) else "fail",
            {"expected": exp, "repeats": reps, "deterministic_sha": det,
             "timing_within_tolerance": timing_ok, "memory_within_10MiB": mem_ok,
             "updates_774_and_1548": budget_ok,
             "note": "full-size P1, seed 42, 2-epoch timing runs; same-seed repeats must be bit-identical"})

    fails = [c for c in chk if c["status"] == "fail"]
    res["overall"] = "fail" if fails else "pass"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t28_runs.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({c["id"]: c["status"] for c in chk}, indent=1))
    print("overall:", res["overall"])
    for c in chk:
        if c["status"] != "pass":
            print(c["id"], json.dumps(c["detail"], default=str)[:800])
    return 0 if not fails else 2


if __name__ == "__main__":
    raise SystemExit(main())
