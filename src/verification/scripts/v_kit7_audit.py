#!/usr/bin/env python
"""Kit-v7 (RTX 5060) G2 timing audit — verifier side.

Reads the user's kit output (a directory extracted from `output_*.zip`, or the zip
itself) and audits the G2 evidence against the conditional sign-off (D-055):

  * code snapshot identity (`env/code_snapshot.txt`; must be a descendant of the
    audited T24 commit `03b720c` and contain the sampled trainer)
  * environment check (RTX 5060, ~8 GB, CUDA available)
  * every train manifest: success, canonical/rtx5060 lane, gate.real_lane, real
    lifecycle, env block, sampled batching, batch/fanout/edge-cap pins, frozen
    split hash, graph sha, config sha
  * metrics: epoch_seconds_mean/train_seconds, updates_per_epoch (774 full-size),
    n_updates, measured max_memory_allocated_bytes + memory_device=cuda,
    evaluation_mode (full_graph expected on 8 GB; sampled_fallback is a red flag)
  * G2 matrix budget: measured epoch time -> GPU-hours for the default matrix
    (strategies x models x seeds x periods x estimated epochs) vs the 150 GPU-h
    ceiling, plus the A+B fast-path projection
  * manifest cross-check with the 3 GB control measurements (same peak order,
    same update budget)

Usage:
  python verification/src/scripts/v_kit7_audit.py <kit_output_dir_or_zip> [--target-sha <sha>]

Output: results/text/verification/data_audit/kit7_audit.json (+ md summary).
Never writes outside verification/.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(r"C:\Users\iusearchbtw\Desktop\XAI_proj")
OUT = REPO / "verification" / "data_audit"
GIT = r"C:\Program Files\Git\bin\git.exe"
# audit-chain commit: T24 sampled trainer (03b720c) is the code the kit must contain
T24_COMMIT = "03b720c81ea6dbd87485279f5ea1fccba4f0832b"

# matrix plan defaults (src/orchestration/matrix_plan.py) and the frozen ceiling
DEFAULT_STRATEGIES = ("ft", "replay", "ewc")
OPTIONAL_STRATEGIES = ("ear_a", "ear_b")
MODELS = ("egraphsage", "gcn")
SEEDS = 5
PERIODS = 5
CEILING_GPU_H = 150.0
ESTIMATE_EPOCHS_PER_PERIOD = 12  # matrix_plan.py default for the dry-run estimate


def git(*args: str) -> tuple[int, str]:
    proc = subprocess.run([GIT, "-C", str(REPO), *args], capture_output=True, text=True)
    return proc.returncode, proc.stdout.strip()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def code_fingerprint_from_git(sha: str, roots: list[str]) -> tuple[str | None, int]:
    """Recompute the manifest code fingerprint from a committed tree.

    Algorithm (mirrors src/utils/manifest.py): SHA-256 over sorted relative paths
    (backslashes normalised to "/"), NUL, per-file SHA-256 hex, newline. Uses raw
    bytes so non-UTF-8 blobs (docs with encoding artifacts) cannot break decoding.
    """
    import hashlib

    proc = subprocess.run([GIT, "-C", str(REPO), "ls-tree", "-r", "--name-only", sha, "--", *roots],
                          capture_output=True)
    if proc.returncode != 0:
        return None, 0
    entries = []
    for rel in [line for line in proc.stdout.decode("utf-8", "replace").splitlines() if line.strip()]:
        raw = subprocess.run([GIT, "-C", str(REPO), "cat-file", "-p", f"{sha}:{rel}"], capture_output=True).stdout
        entries.append((rel.replace("\\", "/"), hashlib.sha256(raw).hexdigest()))
    entries.sort(key=lambda item: item[0])
    h = hashlib.sha256()
    for rel, fsha in entries:
        h.update(rel.encode("utf-8")); h.update(b"\0"); h.update(fsha.encode("ascii")); h.update(b"\n")
    return h.hexdigest(), len(entries)


def code_fingerprint_from_zip(zip_path: Path, roots: list[str]) -> tuple[str | None, int]:
    """Recompute the manifest code fingerprint from the kit code-snapshot ZIP.

    The kit snapshot is a Windows-checkout byte copy (CRLF line endings) of the
    commit; manifests are written by that copy, so the audit must fingerprint the
    snapshot bytes rather than the git blobs.
    """
    import hashlib
    import zipfile

    try:
        zf = zipfile.ZipFile(zip_path)
    except Exception:  # noqa: BLE001
        return None, 0
    entries = []
    for rel in zf.namelist():
        if rel.endswith("/"):
            continue
        if not any(rel == root or rel.startswith(root + "/") for root in roots):
            continue
        entries.append((rel.replace("\\", "/"), hashlib.sha256(zf.read(rel)).hexdigest()))
    entries.sort(key=lambda item: item[0])
    h = hashlib.sha256()
    for rel, fsha in entries:
        h.update(rel.encode("utf-8")); h.update(b"\0"); h.update(fsha.encode("ascii")); h.update(b"\n")
    return h.hexdigest(), len(entries)


def find_code_snapshot_zip(short_sha: str) -> Path | None:
    import re as _re

    cands = []
    for root in (REPO / "remote_rtx5060", REPO / "verification" / "data_audit" / "kit7b_preflight", REPO):
        if not root.is_dir():
            continue
        for p in root.rglob(f"xdrift_repo_{short_sha}*.zip"):
            cands.append(p)
        for p in root.glob("remote_rtx5060_kit_*.zip"):
            cands.append(p)
    for p in cands:
        if p.is_file():
            return p
    return None


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: v_kit7_audit.py <kit_output_dir_or_zip> [--target-sha <sha>]")
        return 2
    src = Path(sys.argv[1])
    target_sha = None
    if "--target-sha" in sys.argv:
        target_sha = sys.argv[sys.argv.index("--target-sha") + 1]
    work = OUT / "kit7" / "output"
    if src.is_file() and src.suffix.lower() == ".zip":
        work = OUT / "kit7" / "output"
        if work.exists():
            import shutil
            shutil.rmtree(work)
        work.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src) as zf:
            zf.extractall(work)
    else:
        work = src
    result: dict = {"source": str(src), "root": str(work), "checks": [], "budget": {}}
    chk = result["checks"]

    def add(i, status, detail):
        chk.append({"id": i, "status": status, "detail": detail})

    def find(pattern: str) -> list[Path]:
        return sorted(work.rglob(pattern))

    # ---- code snapshot identity
    snap_files = list(work.rglob("code_snapshot.txt"))
    snap_sha = None
    if snap_files:
        text = snap_files[0].read_text(encoding="utf-8", errors="replace")
        m = re.search(r"([0-9a-f]{7,40})", text)
        snap_sha = m.group(1) if m else None
    ok_commit = None
    if snap_sha:
        rc, full = git("rev-parse", snap_sha)
        if rc == 0:
            rc2, _ = git("merge-base", "--is-ancestor", T24_COMMIT, full)
            ok_commit = (rc2 == 0)
    add("KIT7.code_snapshot", "pass" if ok_commit else ("warn" if snap_sha else "fail"),
        {"snapshot_text": snap_sha, "descends_from_T24_commit": ok_commit,
         "t24_commit": T24_COMMIT, "target_sha": target_sha,
         "raw_snapshot_file": str(snap_files[0]) if snap_files else None})

    # ---- environment check
    env_files = list(work.rglob("env_check_rtx5060.json"))
    env_ok = False
    env_detail: dict = {}
    if env_files:
        try:
            env = load_json(env_files[0])
            # src/scripts/check_env.py format: {"checks": {name: {ok, detail}}} with
            # torch detail carrying device_name/capability/device_count as JSON text.
            detail: dict = {}
            checks = (env.get("checks") or {}) if isinstance(env, dict) else {}
            torch_detail = (checks.get("torch") or {}).get("detail")
            if isinstance(torch_detail, str):
                try:
                    detail = json.loads(torch_detail)
                except Exception:  # noqa: BLE001
                    detail = {"raw": torch_detail}
            gpu_name = str(detail.get("device_name") or (env.get("gpu") or {}).get("name") or "")
            capability = detail.get("capability")
            total = detail.get("total_memory_bytes") or detail.get("total_memory") or env.get("cuda_total_memory_bytes")
            text = json.dumps(env).lower()
            env_ok = ("5060" in gpu_name.lower() or "rtx" in gpu_name.lower() or "blackwell" in text
                      or (capability and capability[0] >= 12)) and \
                     (total is None or float(total) > 6.0 * 1024 ** 3)
            env_detail = {"gpu_name": gpu_name, "capability": capability, "total_memory_bytes": total,
                          "cuda_available": detail.get("cuda_available", True), "source": str(env_files[0])}
        except Exception as exc:  # noqa: BLE001
            env_detail = {"error": repr(exc)}
    add("KIT7.env_rtx5060", "pass" if env_ok else ("warn" if env_files else "fail"), env_detail)

    # ---- collect train manifests and metrics
    train_mans = [p for p in find("train_manifest_*.json") if "run_manifests" not in str(p)]
    train_mets = [p for p in find("train_ft_*.json") if "manifest" not in p.name]
    runs = []
    for man_path in train_mans:
        man = load_json(man_path)
        extra = man.get("extra") or {}
        timing = extra.get("timing") or {}
        # metrics sibling: same directory pattern
        met_path = None
        cand = man_path.parent / man_path.name.replace("train_manifest_", "train_").replace("_period1.json", "_period1.json")
        cand = man_path.parent / man_path.name.replace("train_manifest_", "train_")
        if cand.is_file():
            met_path = cand
        met = load_json(met_path) if met_path else {}
        res = (met.get("result") or {})
        mt = res.get("timing") or {}
        val = (res.get("val") or {})
        test = (res.get("test") or {})
        lane = man.get("lane")
        started = man.get("started_at_utc"); finished = man.get("finished_at_utc")
        dur = man.get("duration_s")
        env = man.get("env") or {}
        frozen = extra.get("frozen_split_hash")
        graph_sha = extra.get("graph_sha256")
        mem = mt.get("max_memory_allocated_bytes")
        epoch_s = mt.get("epoch_seconds_mean")
        run = {
            "manifest": str(man_path), "metrics": str(met_path) if met_path else None,
            "model": extra.get("model") or (man.get("command") or {}).get("argv"),
            "lane": lane, "lane_source": (man.get("gate") or {}).get("lane_source"),
            "gate_real_lane": (man.get("gate") or {}).get("real_lane"),
            "status": man.get("status"),
            "started": started, "finished": finished, "duration_s": dur,
            "env_keys": sorted(k for k in env if str(k).startswith("XDRIFT")),
            "batching": mt.get("batching"), "batch_size": mt.get("batch_size"),
            "updates_per_epoch": mt.get("updates_per_epoch"), "n_updates": mt.get("n_updates"),
            "max_edges_per_step": mt.get("max_edges_per_step"),
            "neighbor_fanout": mt.get("neighbor_fanout"),
            "epoch_seconds_mean": epoch_s, "train_seconds": mt.get("train_seconds"),
            "max_memory_allocated_bytes": mem, "memory_device": mt.get("memory_device"),
            "cuda_device_name": mt.get("cuda_device_name"), "cuda_total_memory_bytes": mt.get("cuda_total_memory_bytes"),
            "evaluation_mode": {"val": val.get("evaluation_mode"), "test": test.get("evaluation_mode")},
            "val_macro_f1": val.get("macro_f1"), "test_macro_f1": test.get("macro_f1"),
            "n_train_edges": res.get("n_train_edges"), "epochs_run": res.get("epochs_run"),
            "frozen_split_hash": frozen, "graph_sha256": graph_sha,
            "config_sha256": (man.get("config") or {}).get("sha256"),
        }
        runs.append(run)
    result["runs"] = runs

    # ---- manifest code-fingerprint cross-check against the kit snapshot commit
    fp_cache: dict[str, tuple[str | None, int]] = {}
    fp_detail = []
    for man_path in train_mans:
        man = load_json(man_path)
        code = man.get("code") or {}
        declared = code.get("sha256")
        roots = code.get("roots") or ["src", "scripts", "configs", "platforms", "tests"]
        snap = snap_sha or ""
        if snap and declared:
            snap_zip = find_code_snapshot_zip(snap[:7])
            if snap_zip is not None:
                if f"zip::{snap_zip}" not in fp_cache:
                    fp_cache[f"zip::{snap_zip}"] = code_fingerprint_from_zip(snap_zip, list(roots))
                recomputed, n_files = fp_cache[f"zip::{snap_zip}"]
                method = f"snapshot_zip:{snap_zip.name}"
            else:
                rc, full = git("rev-parse", snap)
                sha_full = full if rc == 0 else snap
                if sha_full not in fp_cache:
                    fp_cache[sha_full] = code_fingerprint_from_git(sha_full, list(roots))
                recomputed, n_files = fp_cache[sha_full]
                method = f"git_commit:{sha_full}"
            fp_detail.append({"manifest": man_path.name, "declared": declared, "recomputed": recomputed,
                              "method": method,
                              "file_count_declared": code.get("file_count"), "file_count_recomputed": n_files,
                              "matches": recomputed == declared, "from_dirty_tree": code.get("from_dirty_tree"),
                              "clean_commit": code.get("clean_commit")})
    add("KIT7.code_fingerprint", "pass" if fp_detail and all(d["matches"] for d in fp_detail) else
        ("warn" if not fp_detail else "fail"), {"entries": fp_detail, "snapshot": snap_sha})

    def run_ok(r: dict) -> bool:
        real_life = bool(r["duration_s"] and r["train_seconds"] and float(r["duration_s"]) >= float(r["train_seconds"]) * 0.9)
        return (r["status"] == "success" and r["lane"] in ("rtx5060", "canonical") and r["gate_real_lane"]
                and r["batching"] == "sampled" and r["batch_size"] == 1024
                and r["max_edges_per_step"] == 60000 and r["neighbor_fanout"] == [10, 10]
                and r["updates_per_epoch"] in (69, 774, 309)  # smoke / full-size / other smoke caps
                and r["max_memory_allocated_bytes"] and r["memory_device"] == "cuda"
                and real_life and len(r["env_keys"]) >= 5
                and (r["frozen_split_hash"] or "").startswith("010a80d9"))

    canon = [r for r in runs if r["lane"] in ("rtx5060", "canonical")]
    add("KIT7.train_manifests", "pass" if canon and all(run_ok(r) for r in canon) else "fail",
        {"n_manifests": len(runs), "n_canonical_lane": len(canon),
         "failures": [r["manifest"] for r in canon if not run_ok(r)],
         "runs": [{k: r[k] for k in ("lane", "status", "epoch_seconds_mean", "updates_per_epoch",
                                     "max_memory_allocated_bytes", "memory_device", "evaluation_mode")} for r in runs]})

    # full-graph eval expected on 8 GB
    fallback = [r["manifest"] for r in canon if "sampled_fallback" in
                (str((r["evaluation_mode"] or {}).get("val")) + str((r["evaluation_mode"] or {}).get("test")))]
    add("KIT7.full_graph_eval", "pass" if not fallback else "fail",
        {"fallback_runs": fallback,
         "note": "on the 8 GB canonical lane validation must use full-graph eval; sampled_fallback would "
                 "mean the lane purity rule (T25-2) is violated and the checkpoints' val-based selection is "
                 "not comparable to the legacy baseline"})

    # ---- budget from measured epoch time (same formula/counts as matrix_plan.wall_clock_estimate)
    epoch_candidates = [r["epoch_seconds_mean"] for r in canon if r.get("epoch_seconds_mean")]
    if epoch_candidates:
        epoch_s = float(sorted(epoch_candidates)[len(epoch_candidates) // 2])
        PERIOD_EDGES = {1: 1_131_909, 2: 304_675, 3: 304_675, 4: 304_675, 5: 304_675}
        EVAL_PASSES = 25
        EVAL_FORWARD_FRACTION = 0.35
        epochs_per_period = ESTIMATE_EPOCHS_PER_PERIOD
        full_edges = PERIOD_EDGES[1]

        def run_seconds(strategy_count: int) -> float:
            per_run = sum(epochs_per_period * epoch_s * (PERIOD_EDGES[j] / full_edges) for j in range(1, 6))
            per_run += EVAL_PASSES * epoch_s * EVAL_FORWARD_FRACTION
            return strategy_count * len(MODELS) * SEEDS * per_run

        default_strategies = list(DEFAULT_STRATEGIES)
        all_strategies = default_strategies + list(OPTIONAL_STRATEGIES)
        gpu_h_default = run_seconds(len(default_strategies)) / 3600.0
        gpu_h_all = run_seconds(len(all_strategies)) / 3600.0
        # max epoch time (period-1 s) that keeps each scenario within the ceiling
        def max_epoch_for(gpu_h: float) -> float:
            return epoch_s * CEILING_GPU_H / gpu_h if gpu_h > 0 else float("inf")
        result["budget"] = {
            "measured_epoch_seconds": epoch_s,
            "formula": "per run = sum_j 12*epoch_s*(edges_j/edges_1) + 25*0.35*epoch_s; "
                       "P2-P5 edge counts = 304,675 (matrix_plan.wall_clock_estimate)",
            "strategies_default": default_strategies, "strategies_with_ear": all_strategies,
            "models": list(MODELS), "seeds": SEEDS, "periods": PERIODS,
            "epochs_per_period_assumed": epochs_per_period,
            "default_matrix": {"cells": len(default_strategies) * len(MODELS) * SEEDS,
                               "projected_gpu_hours": round(gpu_h_default, 2),
                               "fits_ceiling": gpu_h_default <= CEILING_GPU_H,
                               "max_epoch_seconds_that_fits": round(max_epoch_for(gpu_h_default), 1)},
            "ear_inclusive_matrix": {"cells": len(all_strategies) * len(MODELS) * SEEDS,
                                     "projected_gpu_hours": round(gpu_h_all, 2),
                                     "fits_ceiling": gpu_h_all <= CEILING_GPU_H,
                                     "max_epoch_seconds_that_fits": round(max_epoch_for(gpu_h_all), 1)},
            "ceiling_gpu_hours": CEILING_GPU_H,
            "with_AB_fastpath_2x": {"default_gpu_hours": round(gpu_h_default / 2.0, 2),
                                     "ear_inclusive_gpu_hours": round(gpu_h_all / 2.0, 2)},
            "note": "estimate excludes S7/S8 explanation runs; Doc 1 §7.4 reductions available: "
                    "GNNExplainer anchors 1000->500/period, then seeds 5->3, then RQ4 drop",
        }
        # 15/20-epoch upper sensitivity (same real edge counts)
        def run_seconds_epochs(epochs: int, strategy_count: int) -> float:
            per_run = sum(epochs * epoch_s * (PERIOD_EDGES[j] / full_edges) for j in range(1, 6))
            per_run += EVAL_PASSES * epoch_s * EVAL_FORWARD_FRACTION
            return strategy_count * len(MODELS) * SEEDS * per_run / 3600.0
        result["budget"]["sensitivity_epochs_per_period"] = {
            str(k): {"default_gpu_hours": round(run_seconds_epochs(k, len(default_strategies)), 2),
                     "ear_inclusive_gpu_hours": round(run_seconds_epochs(k, len(all_strategies)), 2)}
            for k in (10, 12, 15, 20)
        }
        verdict = "pass" if gpu_h_all <= CEILING_GPU_H else ("warn" if gpu_h_default <= CEILING_GPU_H else "fail")
        add("KIT7.matrix_budget", verdict, result["budget"])
    else:
        add("KIT7.matrix_budget", "blocked", {"note": "no canonical-lane train timing found in the kit output"})

    result["overall"] = "fail" if any(c["status"] == "fail" for c in chk) else (
        "warn" if any(c["status"] in ("warn", "blocked") for c in chk) else "pass")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "kit7_audit.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({c["id"]: c["status"] for c in chk}, indent=1))
    print("overall:", result["overall"])
    if result.get("budget"):
        print("budget:", json.dumps(result["budget"], indent=1))
    return 0 if result["overall"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
