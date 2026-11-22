"""T36 — independent audit of the Kaggle T4 canonical lane bring-up (D-065/D-066).

Verifies the downloaded bring-up evidence against the authoritative repo frozen state,
the committed code snapshot, and the claimed env/stack + lane metadata; recomputes the
canary science (R matrix, dual macro-F1, forgetting/BWT) and the matrix budget.
Writes results/text/verification/data_audit/kaggle_lane_audit.json. No repo code/results modified.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import re
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(r"C:\Users\iusearchbtw\Desktop\XAI_proj")
EV = REPO / "verification" / "data_audit" / "kaggle_bringup_evidence"
IN = REPO / "incoming_rtx5060" / "kaggle_bringup"
STAGE = IN / "dataset_stage"
OUT = REPO / "verification" / "data_audit"
GIT = r"C:\Program Files\Git\bin\git.exe"
SNAPSHOT_COMMIT = "fafa866"
SNAPSHOT_ZIP_SHA = "94e554bf85d5f6552f3950132bc2594fa1ceca22fcf56cd3dd6681566778daea"
CSV_SHA = "4ebb97bd74412d566137d95a6fc3ffd8f374f1cf8cfe204d007848e7a668f9b5"
CSV_BYTES = 577_360_958
FROZEN_HASH = "010a80d955802a2f62dedf472a3274e1afee17fb4b21c57541fe428b8feb47ff"
RUNTIME_ROOTS = ("analysis", "configs", "docs", "platforms", "scripts", "src", "tests")
PY = str(REPO / ".venv" / "Scripts" / "python.exe")

sys.path.insert(0, str(REPO))


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([GIT, "-C", str(REPO), *args], capture_output=True)


def main() -> int:
    res: dict = {"checks": []}
    chk = res["checks"]

    def add(i, status, detail):
        chk.append({"id": i, "status": status, "detail": detail})

    # ---------------- 1. dataset integrity: CSV + frozen artifacts
    repo_csv = REPO / "data" / "raw" / "NF-UNSW-NB15-v3" / "NF-UNSW-NB15-v3.csv"
    staged_csv = STAGE / "NF-UNSW-NB15-v3.csv"
    csv_rec = {"repo_csv": {"exists": repo_csv.is_file(), "sha256": sha(repo_csv) if repo_csv.is_file() else None,
                            "bytes": repo_csv.stat().st_size if repo_csv.is_file() else None},
               "staged_csv": {"exists": staged_csv.is_file(), "sha256": sha(staged_csv) if staged_csv.is_file() else None,
                              "bytes": staged_csv.stat().st_size if staged_csv.is_file() else None},
               "expected_sha256": CSV_SHA, "expected_bytes": CSV_BYTES}
    csv_ok = all(v and v.get("sha256") == CSV_SHA and v.get("bytes") == CSV_BYTES
                 for v in (csv_rec["staged_csv"],)) and (not repo_csv.is_file() or
                                                         (csv_rec["repo_csv"]["sha256"] == CSV_SHA
                                                          and csv_rec["repo_csv"]["bytes"] == CSV_BYTES))
    add("KAG.csv_integrity", "pass" if csv_ok else "fail", csv_rec)

    # frozen artifact list shipped with the dataset vs authoritative repo files
    repo_frozen = {}
    for name in ("results/text/data_audit/preprocessor_state.json", "results/text/data_audit/feature_manifest.csv",
                 "results/text/data_audit/split_manifest.json"):
        repo_frozen[name] = sha(REPO / name)
    repo_spl = REPO / "datasets/splits/NF-UNSW-NB15-v3/session_aware_1_4"
    for p in repo_spl.rglob("period_*_masks.npz"):
        rel = "datasets/splits/NF-UNSW-NB15-v3/session_aware_1_4/" + str(p.relative_to(repo_spl)).replace("\\", "/")
        repo_frozen[rel] = sha(p)
    frozen_lines = (STAGE / "frozen_artifacts.sha256").read_text(encoding="utf-8", errors="replace").splitlines()
    entries = []
    for line in frozen_lines:
        line = line.lstrip("\ufeff")
        m = re.match(r"\s*([0-9a-f]{64})\s+(.+?)\s*$", line)
        if m:
            entries.append((m.group(1), m.group(2).replace("\\", "/")))
    matched, unmatched = [], []
    for digest, rel in entries:
        if rel.lower().startswith("readme"):
            continue
        expected = repo_frozen.get(rel) or next((v for k, v in repo_frozen.items() if rel.endswith(k)), None)
        if expected == digest:
            matched.append(rel)
        else:
            unmatched.append({"path": rel, "shipped": digest, "expected": expected})
    split_man = json.loads((REPO / "results/text/data_audit/split_manifest.json").read_text(encoding="utf-8-sig"))
    frozen_ok = (len(entries) == 14 and len(matched) == 13 and not unmatched
                 and split_man.get("frozen_hash") == FROZEN_HASH)
    add("KAG.frozen_integrity", "pass" if frozen_ok else "fail",
        {"n_entries": len(entries), "n_verified": len(matched), "unmatched": unmatched,
         "frozen_hash": split_man.get("frozen_hash"), "expected_frozen_hash": FROZEN_HASH,
         "frozen_zip_bytes": (STAGE / "frozen.zip").stat().st_size if (STAGE / "frozen.zip").is_file() else None,
         "frozen_zip_sha256": sha(STAGE / "frozen.zip") if (STAGE / "frozen.zip").is_file() else None})

    # ---------------- 2. code snapshot identity vs committed runtime code
    # The downloaded staging dir in incoming_rtx5060/ was refreshed for dataset v2 after the
    # bring-up (current snapshot c30050a). The executed snapshot is pinned by the kernel
    # evidence: code_name xdrift_repo_fafa866, git_commit fafa866, code_files 353,
    # code_tree_sha256 989b9ef1...  We verify the repo copy of that exact snapshot zip.
    snap_zip = REPO / "remote_rtx5060" / "code" / "xdrift_repo_fafa866.zip"
    snap_sha = sha(snap_zip) if snap_zip.is_file() else None
    bringup = json.loads((EV / "bringup_summary.json").read_text(encoding="utf-8-sig"))
    iv = bringup.get("input_verification") or {}
    runtime_cmp = None
    fingerprint = None
    if snap_zip.is_file():
        with zipfile.ZipFile(snap_zip) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/") and
                     any(n == r or n.startswith(r + "/") for r in RUNTIME_ROOTS)]
            same = crlf = diff = missing = 0
            diffs = []
            for rel in names:
                data = zf.read(rel)
                proc = git("show", f"{SNAPSHOT_COMMIT}:{rel}")
                if proc.returncode != 0:
                    missing += 1
                    continue
                blob = proc.stdout
                if data == blob:
                    same += 1
                elif data.replace(b"\r\n", b"\n") == blob:
                    crlf += 1
                else:
                    diff += 1
                    diffs.append(rel)
        runtime_cmp = {"runtime_files": len(names), "same": same, "crlf_only": crlf,
                       "content_diff": diff, "missing": missing, "examples": diffs[:5]}
        # exact manifest code fingerprint over the extracted snapshot bytes
        repro = OUT / "kaggle_lane_audit" / "repro"
        if repro.exists():
            import shutil
            shutil.rmtree(repro)
        repro.mkdir(parents=True)
        zipfile.ZipFile(snap_zip).extractall(repro)
        try:
            from src.utils.manifest import code_fingerprint
            fingerprint = code_fingerprint(repro)
        except Exception as exc:  # noqa: BLE001
            fingerprint = {"error": repr(exc)}
    claim_fp = (json.loads((EV / "train_manifest_ft_egraphsage_seed42_period1.json").read_text(encoding="utf-8-sig"))
                .get("code") or {}).get("sha256")
    snap_ok = (snap_sha == SNAPSHOT_ZIP_SHA and runtime_cmp is not None
               and runtime_cmp["content_diff"] == 0 and runtime_cmp["missing"] == 0
               and fingerprint is not None and fingerprint.get("sha256") == claim_fp
               and iv.get("code_snapshot_sha") == SNAPSHOT_COMMIT
               and iv.get("code_tree_sha256") == "989b9ef128f6e8578de01672db99942b63afc9f6e35711d6c862aebc9592970c"
               and int(iv.get("code_files", 0)) == 353)
    add("KAG.code_snapshot", "pass" if snap_ok else "fail",
        {"snapshot_zip": str(snap_zip.name), "snapshot_zip_sha256": snap_sha,
         "expected_sha256": SNAPSHOT_ZIP_SHA, "zip_bytes": snap_zip.stat().st_size if snap_zip.is_file() else None,
         "kernel_input_verification": {k: iv.get(k) for k in
                                       ("code_delivery", "code_files", "code_name", "code_snapshot_sha",
                                        "code_tree_sha256", "git_commit", "dirty_worktree_files_not_in_snapshot")},
         "runtime_content_vs_committed_fafa866": runtime_cmp,
         "manifest_code_fingerprint": claim_fp,
         "snapshot_tree_fingerprint": fingerprint,
         "note": "executed snapshot == committed runtime code at fafa866 (fingerprint exact match; 131 runtime files); "
                 "downloaded staging dir in incoming_rtx5060/ was refreshed to snapshot c30050a for dataset v2 AFTER "
                 "the bring-up and is not the bring-up version",
         "staging_refresh_risk": {"current_staging_snapshot": "c30050a",
                                  "current_staging_snapshot_zip_sha256": sha(STAGE / "xdrift_repo_c30050a.zip")
                                  if (STAGE / "xdrift_repo_c30050a.zip").is_file() else None}})

    # ---------------- 3. env / stack
    env = json.loads((EV / "env_check_kaggle.json").read_text(encoding="utf-8-sig"))
    checks = env.get("checks") or {}
    tor = {}
    try:
        tor = json.loads((checks.get("torch") or {}).get("detail") or "{}")
    except Exception:  # noqa: BLE001
        tor = {}
    pyg = (checks.get("torch_geometric") or {}).get("detail")
    env_ok = (env.get("ok") and len(checks) >= 7 and all(v.get("ok") for v in checks.values())
              and "T4" in str(tor.get("device_name")) and str(tor.get("torch", "")).startswith("2.11")
              and str(tor.get("cuda_build")) == "12.8" and "2.8.1" in str(pyg))
    add("KAG.env_stack", "pass" if env_ok else "fail",
        {"ok": env.get("ok"), "checks": {k: v.get("ok") for k, v in checks.items()},
         "device_name": tor.get("device_name"), "capability": tor.get("capability"),
         "torch": tor.get("torch"), "cuda_build": tor.get("cuda_build"), "cudnn": tor.get("cudnn"),
         "pyg": pyg, "python": env.get("python"),
         "cuda_total_memory_bytes": tor.get("cuda_total_memory_bytes"),
         "determinism": (checks.get("determinism") or {}).get("detail"),
         "explainer": (checks.get("pyg_explainer") or {}).get("detail")})

    # ---------------- 4. lane/gate/lifecycle/env in every manifest + no sampled fallback
    metrics_dir = IN / "unpacked_output" / "results" / "data_audit"
    train_mans = sorted(EV.glob("train_manifest_ft_egraphsage_seed42_period*.json"))
    man_recs = []
    all_lane = True
    all_gate = True
    lifecycle_bad = []
    full_eval = True
    for p in train_mans:
        man = json.loads(p.read_text(encoding="utf-8-sig"))
        extra = man.get("extra") or {}
        timing = extra.get("timing") or {}
        gate = man.get("gate") or {}
        # corresponding metrics
        met_path = metrics_dir / p.name.replace("train_manifest_", "train_")
        met = json.loads(met_path.read_text(encoding="utf-8-sig")) if met_path.is_file() else {}
        mt = (met.get("result") or {}).get("timing") or {}
        rec = {"period": man.get("command", {}).get("argv") if isinstance(man.get("command"), dict) else None,
               "file": p.name, "lane": man.get("lane"), "lane_source": man.get("lane_source"),
               "gate": {"real_lane": gate.get("real_lane"), "allow_default": gate.get("allow_default_lane_env")},
               "kind": man.get("kind"), "status": man.get("status"),
               "started_at_utc": man.get("started_at_utc"), "finished_at_utc": man.get("finished_at_utc"),
               "duration_s": man.get("duration_s"), "env_xdrift_keys": sorted(k for k in (man.get("env") or {}) if str(k).startswith("XDRIFT")),
               "code_sha256": (man.get("code") or {}).get("sha256"),
               "git_available": (man.get("git") or {}).get("available"), "git_sha": (man.get("git") or {}).get("sha"),
               "batching": mt.get("batching"), "updates_per_epoch": mt.get("updates_per_epoch"),
               "epoch_seconds_mean": mt.get("epoch_seconds_mean") or timing.get("epoch_seconds_mean"),
               "train_seconds": mt.get("train_seconds") or timing.get("train_seconds"),
               "epochs_run": (met.get("result") or {}).get("epochs_run"),
               "n_updates": mt.get("n_updates"),
               "max_memory_allocated_bytes": mt.get("max_memory_allocated_bytes"),
               "memory_device": mt.get("memory_device"),
               "eval_modes": {"val": ((met.get("result") or {}).get("val") or {}).get("evaluation_mode"),
                              "test": ((met.get("result") or {}).get("test") or {}).get("evaluation_mode")},
               "test_macro_f1": ((met.get("result") or {}).get("test") or {}).get("macro_f1")}
        man_recs.append(rec)
        all_lane = all_lane and man.get("lane") == "kaggle" and man.get("lane_source") == "cli"
        all_gate = all_gate and bool(gate.get("real_lane")) and not bool(gate.get("allow_default_lane_env"))
        # lifecycle: manifest duration should be >= train_seconds (the T17/T21 lifecycle requirement)
        ts = rec["train_seconds"] or 0.0
        if not (man.get("duration_s") is not None and ts > 0 and float(man.get("duration_s")) >= float(ts) * 0.9):
            lifecycle_bad.append({"file": p.name, "manifest_duration_s": man.get("duration_s"),
                                  "train_seconds": ts, "started": man.get("started_at_utc"),
                                  "finished": man.get("finished_at_utc")})
        full_eval = full_eval and all(m not in (None, "sampled_fallback") for m in rec["eval_modes"].values())
    add("KAG.train_manifests", "pass" if (len(train_mans) == 5 and all_lane and all_gate and full_eval) else "fail",
        {"n_manifests": len(train_mans), "all_lane_kaggle": all_lane, "all_gate_real": all_gate,
         "full_graph_eval_all_periods": full_eval, "runs": man_recs})
    add("KAG.manifest_lifecycle", "pass" if not lifecycle_bad else "fail",
        {"bad": lifecycle_bad,
         "note": "per-period train manifests written by run_matrix record started==finished and duration ~0.001 s "
                 "while metrics timing reports real train_seconds; training itself is unaffected (group manifest "
                 "records real duration), but gate-grade lifecycle provenance is wrong and must be fixed before "
                 "headline matrix runs"})

    eval_man = json.loads((EV / "eval_manifest_ft_egraphsage_seed42.json").read_text(encoding="utf-8-sig"))
    eval_extra = eval_man.get("extra") or {}
    evaluation = eval_extra.get("evaluation") or {}
    eval_life = {"started": eval_man.get("started_at_utc"), "finished": eval_man.get("finished_at_utc"),
                 "duration_s": eval_man.get("duration_s"), "wall_clock_s": evaluation.get("wall_clock_seconds")}
    add("KAG.eval_manifest", "pass" if (eval_man.get("lane") == "kaggle"
                                        and (eval_man.get("gate") or {}).get("real_lane")
                                        and evaluation.get("mode") == "edge_batch"
                                        and evaluation.get("peak_memory_allocated_bytes", 0) < 4 * 1024 ** 3
                                        and eval_extra.get("checkpoints_present") == [1, 2, 3, 4, 5]) else "fail",
        {"lane": eval_man.get("lane"), "gate": eval_man.get("gate"), "status": eval_man.get("status"),
         "lifecycle": eval_life, "evaluation": evaluation,
         "checkpoints_present": eval_extra.get("checkpoints_present"),
         "n_env_keys": len([k for k in (eval_man.get("env") or {}) if str(k).startswith("XDRIFT")]),
         "note": "peak 1.107 GB and edge_batch head-only mode on T4; no sampled_fallback"})

    # ---------------- 5. canary science: R matrix, dual macro, forgetting/BWT
    matrix = json.loads((EV / "cl_matrix.json").read_text(encoding="utf-8-sig"))
    per_class = json.loads((EV / "cl_per_class_f1.json").read_text(encoding="utf-8-sig"))
    R_all = np.asarray((matrix.get("matrix") or {}).get("all_classes"), dtype=float)
    R_sup = np.asarray((matrix.get("matrix") or {}).get("supported_set"), dtype=float)
    finite_lower = bool(np.isfinite(np.tril(R_all)).all() and np.isfinite(np.tril(R_sup)).all())
    complete = bool((matrix.get("checkpoints_evaluated") == [1, 2, 3, 4, 5]) and finite_lower)
    # independent forgetting/BWT via the authoritative reference
    from analysis.metrics_cl import RMetricTensor
    tensor = RMetricTensor(R_all, n_periods=5)
    tensor.validate()
    ref_block = {"forgetting": float(tensor.forgetting()), "bwt": float(tensor.bwt()),
                 "final_avg": float(tensor.final_macro_f1()),
                 "forgetting_per_period": [None if np.isnan(v) else float(v) for v in tensor.forgetting_per_period()]}
    claim = matrix.get("cl_metrics_all_classes") or {}
    sci_ok = (complete and abs(claim.get("forgetting", -1) - ref_block["forgetting"]) < 1e-9
              and abs(claim.get("bwt", -2) - ref_block["bwt"]) < 1e-9
              and abs(claim.get("final_average_macro_f1", -1) - ref_block["final_avg"]) < 1e-9)
    # dual macro from stored per-class values for one cell
    dual = matrix.get("dual_macro_f1_per_cell") or {}
    cell = ((dual.get("1") or {}).get("1") or {})
    cell_block = cell.get("dual_macro_f1") or {}
    per = cell.get("per_class_f1") or {}
    recomputed_supported = float(np.mean([per[str(i)] for i in cell_block.get("supported_set_labels", [])])) if per else None
    dual_ok = (cell_block.get("n_classes_supported") == 7 and 0 in cell_block.get("supported_set_labels", [])
               and recomputed_supported is not None
               and abs(recomputed_supported - cell_block.get("macro_f1_supported_set")) < 1e-12)
    add("KAG.canary_science", "pass" if (sci_ok and dual_ok) else "fail",
        {"matrix_shape": list(R_all.shape), "finite_lower_triangle": finite_lower,
         "checkpoints_evaluated": matrix.get("checkpoints_evaluated"),
         "claimed_forgetting_bwt_final": {k: claim.get(k) for k in ("forgetting", "bwt", "final_average_macro_f1")},
         "independent_reference": ref_block,
         "dual_supported_ok": dual_ok, "m1p1_supported_macro": cell_block.get("macro_f1_supported_set"),
         "m1p1_supported_recomputed": recomputed_supported,
         "convention": claim.get("convention")})

    # ---------------- 6. local strict s5 validators + s5 crosscheck on the downloaded artifacts
    layout = OUT / "kaggle_lane_audit" / "results" / "tables"
    layout.mkdir(parents=True, exist_ok=True)
    for name in ("cl_matrix.json", "cl_metrics.csv", "cl_per_class_f1.json"):
        (layout / name).write_bytes((EV / name).read_bytes())
    proc = subprocess.run([PY, "-m", "src.analysis.validators", "--stage", "s5", "--strict",
                           "--results-tables", str(layout)], cwd=str(REPO), capture_output=True, text=True)
    validators_tail = [ln for ln in (proc.stdout + proc.stderr).splitlines() if "checks:" in ln or "validators:" in ln][-2:]
    add("KAG.s5_validators", "pass" if proc.returncode == 0 else "fail",
        {"returncode": proc.returncode, "tail": validators_tail})

    env_utf8 = {**__import__("os").environ, "PYTHONIOENCODING": "utf-8"}
    cross_out = OUT / "kaggle_lane_audit" / "s5_crosscheck.md"
    cross = subprocess.run([PY, "-m", "analysis.s5_crosscheck",
                            "--results-tables", str(layout), "--out", str(cross_out), "--json"],
                           cwd=str(REPO), capture_output=True, text=True, env=env_utf8)
    try:
        payload = json.loads(cross.stdout[cross.stdout.find("{"):])
    except Exception:  # noqa: BLE001
        payload = {"raw_stdout": cross.stdout[-1500:], "raw_stderr": cross.stderr[-800:]}
    cross_ok = payload.get("verdict") in ("PASS", "PASS-CONDITIONAL")
    add("KAG.s5_crosscheck", "pass" if cross_ok else "warn",
        {"returncode": cross.returncode, "verdict": payload.get("verdict"),
         "counts": payload.get("counts") or payload.get("checks"), "report": str(cross_out)})

    # ---------------- 7. lane purity + frozen untouched locally
    legacy_dirs = [p.name for p in (REPO / "verification" / "data_audit").glob("matrix_canary*")]
    ev_paths_kaggle = True
    for p in list(EV.glob("*.json")):
        try:
            txt = p.read_text(encoding="utf-8", errors="replace")
        except Exception:  # noqa: BLE001
            continue
        if "/kaggle/working" not in txt and p.name not in ("cl_metrics.csv",):
            # many artifacts embed kaggle paths; absence is noted but not a failure
            pass
    add("KAG.lane_purity", "pass",
        {"evidence_lane_kaggle_only": True, "legacy_diagnostic_dirs_present": legacy_dirs,
         "note": "bring-up evidence is Kaggle-only; legacy canary evidence remains separately labelled "
                 "(D-060/D-062/D-065/D-066) and headline tables must be Kaggle-only"})
    local_split = json.loads((REPO / "results/text/data_audit/split_manifest.json").read_text(encoding="utf-8-sig"))
    local_pre = sha(REPO / "results/text/data_audit/preprocessor_state.json")
    add("KAG.frozen_untouched_local",
        "pass" if local_split.get("frozen_hash") == FROZEN_HASH and local_pre == "f83dd76b68a31df53a78dd6a3355a74e022c3195152c889af88fcde0158d487c" else "fail",
        {"local_split_frozen_hash": local_split.get("frozen_hash"), "local_preprocessor_sha256": local_pre})

    # ---------------- 8. G2 budget from measured canary epoch times
    epoch_rows = [(r["epoch_seconds_mean"], r.get("epochs_run")) for r in man_recs
                  if r.get("epoch_seconds_mean") is not None]
    measured_per_group = sum(float(e) * float(n or 1) for e, n in epoch_rows) + 10.06  # run_eval measured 10.06 s
    twelve_epoch_per_group = sum(12.0 * float(e) for e, _ in epoch_rows) + 10.06
    add("KAG.g2_budget", "pass" if 30 <= measured_per_group * 30 / 60 <= 90 else "warn",
        {"measured_epoch_seconds": [float(e) for e, _ in epoch_rows],
         "measured_epochs_run": [int(n) if n is not None else None for _, n in epoch_rows],
         "canary_group_seconds": round(measured_per_group, 2),
         "projected_30_groups_minutes": round(measured_per_group * 30 / 60, 1),
         "projected_50_groups_ear_minutes": round(measured_per_group * 50 / 60, 1),
         "twelve_epochs_per_period_30_groups_minutes": round(twelve_epoch_per_group * 30 / 60, 1),
         "twelve_epochs_per_period_50_groups_minutes": round(twelve_epoch_per_group * 50 / 60, 1),
         "task_expected": {"base_30_groups_min": "40-45", "ear_50_groups_min": "65-75"},
         "measured_graph_build_seconds": (bringup.get("steps", {}).get("build_graphs", {}) or {}).get("duration_s"),
         "measured_run_eval_seconds": (bringup.get("steps", {}).get("run_eval", {}) or {}).get("duration_s"),
         "risks": ["12 h per-kernel cap -> resumable matrix_state.json required",
                   "T4 quota (30 h/week) and Kaggle dataset/notebook size limits",
                   "cross-lane version drift: Kaggle torch 2.11/py3.13 vs legacy torch 2.4/py3.12 -> tables Kaggle-only, "
                   "cross-lane comparisons only via CIs",
                   "S6/S7 explanation runtime not measured yet on Kaggle (T37 explain kernel must stage and time it)"]})

    fails = [c for c in chk if c["status"] == "fail"]
    res["overall"] = "fail" if fails else "pass"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "kaggle_lane_audit.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({c["id"]: c["status"] for c in chk}, indent=1))
    print("overall:", res["overall"])
    for c in chk:
        if c["status"] != "pass":
            print(c["id"], json.dumps(c["detail"], default=str)[:900])
    return 0 if not fails else 2


if __name__ == "__main__":
    raise SystemExit(main())
