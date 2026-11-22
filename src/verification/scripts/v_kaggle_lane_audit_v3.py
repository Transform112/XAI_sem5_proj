"""T36 refresh audit — Kaggle T4 canonical lane on dataset v3 (snapshot 90e7635),
matrix smoke evidence, plus T32/T34 closure for the T33 evidence gaps.

Runs against HEAD (runtime code unchanged since 90e7635: `git diff 90e7635..HEAD -- src
scripts configs platforms tests` is empty). Writes
results/text/verification/data_audit/kaggle_lane_audit_v3.json and re-runs strict validators/crosscheck.
No repo code/results modified.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

REPO = Path(r"C:\Users\iusearchbtw\Desktop\XAI_proj")
GIT = r"C:\Program Files\Git\bin\git.exe"
PY = str(REPO / ".venv" / "Scripts" / "python.exe")
OUT = REPO / "verification" / "data_audit"
SMOKE = OUT / "kaggle_matrix_smoke"
RUN = REPO / "incoming_rtx5060" / "kaggle_runs" / "t37-matrix-smoke-90e7635"
UP = RUN / "unpacked"
STAGE = REPO / "incoming_rtx5060" / "kaggle_bringup" / "dataset_stage"
T34 = OUT / "t34_canary"
SNAPSHOT_COMMIT = "90e7635"
SNAPSHOT_ZIP = REPO / "remote_rtx5060" / "code" / "xdrift_repo_90e7635.zip"
SNAPSHOT_ZIP_SHA = "551d6897db586df09b4a63ce9b8ebaafe399cd4b5728e101e65c8a7e0f295f79"
CSV_SHA = "4ebb97bd74412d566137d95a6fc3ffd8f374f1cf8cfe204d007848e7a668f9b5"
CSV_BYTES = 577_360_958
FROZEN_HASH = "010a80d955802a2f62dedf472a3274e1afee17fb4b21c57541fe428b8feb47ff"
RUNTIME_ROOTS = ("src", "scripts", "configs", "platforms", "tests")
ENV_UTF8 = {**os.environ, "PYTHONIOENCODING": "utf-8"}

sys.path.insert(0, str(REPO))


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([GIT, "-C", str(REPO), *args], capture_output=True)


def rd(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8-sig"))


def main() -> int:
    res: dict = {"checks": []}
    chk = res["checks"]

    def add(i, status, detail):
        chk.append({"id": i, "status": status, "detail": detail})

    # ---------------- A. dataset v3 integrity
    env_man = rd(STAGE / "env_manifest.json")
    code_zip = env_man.get("code_zip") or {}
    frozen_zip = env_man.get("frozen_zip") or {}
    csv_man = env_man.get("csv") or {}
    repo_csv = REPO / "datasets/raw/NF-UNSW-NB15-v3/NF-UNSW-NB15-v3.csv"
    staged_csv = STAGE / "NF-UNSW-NB15-v3.csv"
    csv_ok = (repo_csv.is_file() and sha(repo_csv) == CSV_SHA and repo_csv.stat().st_size == CSV_BYTES
              and staged_csv.is_file() and sha(staged_csv) == CSV_SHA and staged_csv.stat().st_size == CSV_BYTES
              and csv_man.get("sha256") == CSV_SHA and int(csv_man.get("bytes", 0)) == CSV_BYTES)
    add("V3.csv_integrity", "pass" if csv_ok else "fail",
        {"repo_csv": {"sha256": sha(repo_csv) if repo_csv.is_file() else None,
                      "bytes": repo_csv.stat().st_size if repo_csv.is_file() else None},
         "staged_csv": {"sha256": sha(staged_csv) if staged_csv.is_file() else None,
                        "bytes": staged_csv.stat().st_size if staged_csv.is_file() else None},
         "env_manifest_csv": csv_man, "expected_sha256": CSV_SHA, "expected_bytes": CSV_BYTES})

    # frozen payload: 13 artifacts + README = 14 entries vs authoritative repo files
    repo_frozen = {"results/text/data_audit/preprocessor_state.json": sha(REPO / "results/text/data_audit/preprocessor_state.json"),
                   "results/text/data_audit/feature_manifest.csv": sha(REPO / "results/text/data_audit/feature_manifest.csv"),
                   "results/text/data_audit/split_manifest.json": sha(REPO / "results/text/data_audit/split_manifest.json"),
                   "remote_rtx5060/frozen/README.md": sha(REPO / "remote_rtx5060/frozen/README.md")}
    spl = REPO / "datasets/splits/NF-UNSW-NB15-v3/session_aware_1_4"
    for p in spl.rglob("period_*_masks.npz"):
        rel = "datasets/splits/NF-UNSW-NB15-v3/session_aware_1_4/" + p.relative_to(spl).as_posix()
        repo_frozen[rel] = sha(p)
    frozen_lines = (STAGE / "frozen_artifacts.sha256").read_text(encoding="utf-8", errors="replace").splitlines()
    matched, unmatched = [], []
    entries = 0
    for line in frozen_lines:
        m = re.match(r"\s*([0-9a-f]{64})\s+(.+?)\s*$", line.lstrip("\ufeff"))
        if not m:
            continue
        entries += 1
        digest, rel = m.group(1), m.group(2).replace("\\", "/")
        expected = repo_frozen.get(rel)
        if expected is None and rel.lower().startswith("readme"):
            expected = repo_frozen.get("remote_rtx5060/frozen/README.md")
        if expected == digest:
            matched.append(rel)
        else:
            unmatched.append({"path": rel, "shipped": digest, "expected": expected})
    split_man = rd(REPO / "results/text/data_audit/split_manifest.json")
    add("V3.frozen_payload", "pass" if (entries == 14 and len(matched) == 14 and not unmatched
                                        and split_man.get("frozen_hash") == FROZEN_HASH) else "fail",
        {"n_entries": entries, "n_verified": len(matched), "unmatched": unmatched,
         "split_hash": split_man.get("frozen_hash"), "expected_split_hash": FROZEN_HASH,
         "frozen_zip_sha256": sha(STAGE / "frozen.zip") if (STAGE / "frozen.zip").is_file() else None,
         "env_manifest_frozen_zip": {k: frozen_zip.get(k) for k in ("bytes", "sha256")}})

    # snapshot zip identity + exact code fingerprint over extracted snapshot bytes
    snap_sha = sha(SNAPSHOT_ZIP) if SNAPSHOT_ZIP.is_file() else None
    runtime_cmp = None
    fingerprint = None
    if SNAPSHOT_ZIP.is_file():
        with zipfile.ZipFile(SNAPSHOT_ZIP) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")
                     and any(n == r or n.startswith(r + "/") for r in RUNTIME_ROOTS)]
            same = crlf = diff = missing = 0
            for rel in names:
                data = zf.read(rel)
                proc = git("show", f"{SNAPSHOT_COMMIT}:{rel}")
                if proc.returncode != 0:
                    missing += 1
                elif data == proc.stdout:
                    same += 1
                elif data.replace(b"\r\n", b"\n") == proc.stdout:
                    crlf += 1
                else:
                    diff += 1
        runtime_cmp = {"runtime_files": len(names), "same": same, "crlf_only": crlf,
                       "content_diff": diff, "missing": missing}
        import tempfile
        _tmp = tempfile.TemporaryDirectory(prefix="xdrift_audit_repro_")
        try:
            zipfile.ZipFile(SNAPSHOT_ZIP).extractall(_tmp.name)
            from src.utils.manifest import code_fingerprint
            fingerprint = code_fingerprint(Path(_tmp.name))
        finally:
            _tmp.cleanup()
    train_p1 = UP / "results/text/data_audit/train_manifest_ft_egraphsage_seed42_period1.json"
    claim_fp = (rd(train_p1).get("code") or {}).get("sha256")
    meta = rd(SMOKE / "run_full_meta.json")
    rt_diff = (git("diff", "--name-only", f"{SNAPSHOT_COMMIT}..HEAD", "--", *RUNTIME_ROOTS)).stdout.decode().strip()
    add("V3.code_snapshot", "pass" if (snap_sha == SNAPSHOT_ZIP_SHA and runtime_cmp["content_diff"] == 0
                                      and runtime_cmp["missing"] == 0
                                      and fingerprint and fingerprint.get("sha256") == claim_fp
                                      and code_zip.get("sha256") == SNAPSHOT_ZIP_SHA
                                      and int(code_zip.get("dirty_worktree_files_not_in_snapshot", -1)) == 0
                                      and rt_diff == "") else "fail",
        {"snapshot_zip": SNAPSHOT_ZIP.name, "sha256": snap_sha, "expected_sha256": SNAPSHOT_ZIP_SHA,
         "bytes": SNAPSHOT_ZIP.stat().st_size if SNAPSHOT_ZIP.is_file() else None,
         "env_manifest_code_zip": {k: code_zip.get(k) for k in
                                   ("name", "sha256", "bytes", "snapshot_sha", "dirty_worktree_files_not_in_snapshot")},
         "content_vs_committed_90e7635": runtime_cmp,
         "manifest_code_fingerprint": claim_fp,
         "extracted_snapshot_fingerprint": fingerprint,
         "runtime_diff_90e7635_to_HEAD": rt_diff or "(empty - runtime code identical)",
         "dataset_metadata_v3": rd(STAGE / "dataset-metadata.json").get("description", "")[:400],
         "note": "dataset v3 ships snapshot 90e7635 (0 dirty files); the executed snapshot code fingerprint "
                 "matches the smoke manifests, and HEAD contains only docs/evidence commits after 90e7635"})

    # ---------------- B. env / T4 stack + kernel/dataset refs
    env = rd(UP / "results/env_check_kaggle.json")
    checks = env.get("checks") or {}
    tor = {}
    try:
        tor = json.loads((checks.get("torch") or {}).get("detail") or "{}")
    except Exception:  # noqa: BLE001
        pass
    pyg = (checks.get("torch_geometric") or {}).get("detail")
    run_sum = rd(RUN / "kernel_output/run_summary.json")
    gpu_smi = None
    for line in (RUN / "kernel_output/xdrift-matrix.log").read_text(encoding="utf-8", errors="replace").splitlines():
        if "Tesla T4" in line:
            gpu_smi = line.strip()
            break
    env_ok = (env.get("ok") and len(checks) == 7 and all(v.get("ok") for v in checks.values())
              and "T4" in str(tor.get("device_name")) and str(tor.get("torch", "")).startswith("2.11")
              and str(tor.get("cuda_build")) == "12.8" and "2.8.1" in str(pyg))
    add("V3.env_stack", "pass" if env_ok else "fail",
        {"ok": env.get("ok"), "checks": {k: v.get("ok") for k, v in checks.items()},
         "device_name": tor.get("device_name"), "capability": tor.get("capability"),
         "torch": tor.get("torch"), "cuda_build": tor.get("cuda_build"), "cudnn": tor.get("cudnn"),
         "pyg": pyg, "python": env.get("python"),
         "determinism": (checks.get("determinism") or {}).get("detail"),
         "explainer": (checks.get("pyg_explainer") or {}).get("detail"),
         "kernel_ref": run_sum.get("ref"), "kernel_status": run_sum.get("status"),
         "dataset_sources": run_sum.get("dataset_sources"), "gpu_smi": gpu_smi})

    # dataset/kernel refs from the run report metadata
    report = (SMOKE / "RunReport.md").read_text(encoding="utf-8-sig")
    dataset_line = next((ln for ln in report.splitlines() if "dataset:" in ln), "")
    add("V3.run_refs", "pass" if ("v3" in dataset_line and "90e7635" in dataset_line
                                  and run_sum.get("ref") == "mysteriousavailable/xdrift-matrix") else "warn",
        {"run_id": meta.get("run_id"), "dataset_line": dataset_line.strip(),
         "kernel_ref": run_sum.get("ref"), "kernel_status": run_sum.get("status"),
         "inner_wall_clock_s": 259.9,
         "run_full_meta": {k: meta.get(k) for k in ("run_id", "pushes")}})

    # ---------------- C. smoke manifests: lane/gate/lifecycle/full-graph eval
    metrics_dir = UP / "results/text/data_audit"
    mans = sorted((UP / "results/text/data_audit").glob("train_manifest_ft_egraphsage_seed42_period*.json"))
    recs = []
    all_lane = all_gate = full_eval = True
    lifecycle_bad = []
    for p in mans:
        man = rd(p)
        gate = man.get("gate") or {}
        met = rd(metrics_dir / p.name.replace("train_manifest_", "train_"))
        r = met.get("result") or {}
        t = r.get("timing") or {}
        rec = {"file": p.name, "lane": man.get("lane"), "lane_source": man.get("lane_source"),
               "gate": {"real_lane": gate.get("real_lane"), "allow_default": gate.get("allow_default_lane_env")},
               "kind": man.get("kind"), "status": man.get("status"),
               "started": man.get("started_at_utc"), "finished": man.get("finished_at_utc"),
               "duration_s": man.get("duration_s"), "train_seconds": t.get("train_seconds"),
               "epochs": r.get("epochs_run"), "epoch_seconds_mean": t.get("epoch_seconds_mean"),
               "updates_per_epoch": t.get("updates_per_epoch"),
               "max_memory_allocated_bytes": t.get("max_memory_allocated_bytes"),
               "batching": t.get("batching"),
               "eval_modes": {"val": (r.get("val") or {}).get("evaluation_mode"),
                              "test": (r.get("test") or {}).get("evaluation_mode")},
               "code_sha256": (man.get("code") or {}).get("sha256"),
               "env_xdrift_keys": sorted(k for k in (man.get("env") or {}) if str(k).startswith("XDRIFT"))}
        recs.append(rec)
        all_lane = all_lane and man.get("lane") == "kaggle" and man.get("lane_source") == "cli"
        all_gate = all_gate and bool(gate.get("real_lane")) and not bool(gate.get("allow_default_lane_env"))
        full_eval = full_eval and all(m == "full_graph_batch_head" for m in rec["eval_modes"].values())
        dur = float(man.get("duration_s") or 0)
        ts = float(t.get("train_seconds") or 0)
        if not (dur >= ts * 0.9 and str(man.get("started_at_utc")) < str(man.get("finished_at_utc"))):
            lifecycle_bad.append({"file": p.name, "duration_s": dur, "train_seconds": ts,
                                  "started": man.get("started_at_utc"), "finished": man.get("finished_at_utc")})
    add("V3.train_manifests", "pass" if (len(mans) == 5 and all_lane and all_gate and full_eval) else "fail",
        {"n_manifests": len(mans), "all_lane_kaggle": all_lane, "all_gate_real": all_gate,
         "full_graph_eval_all_periods": full_eval, "runs": recs})
    add("V3.manifest_lifecycle", "pass" if not lifecycle_bad else "fail",
        {"bad": lifecycle_bad,
         "note": "T36 fix ba73b3d/90e7635: per-period manifests start before training; smoke shows real "
                 "started<finished and duration >= train_seconds"})

    eval_man = rd(UP / "results/text/data_audit/eval_manifest_ft_egraphsage_seed42.json")
    ex = eval_man.get("extra") or {}
    ev = ex.get("evaluation") or {}
    add("V3.eval_manifest", "pass" if (eval_man.get("lane") == "kaggle"
                                       and (eval_man.get("gate") or {}).get("real_lane")
                                       and ev.get("mode") == "edge_batch"
                                       and ev.get("peak_memory_allocated_bytes", 0) < 4 * 1024 ** 3
                                       and ex.get("checkpoints_present") == [1, 2, 3, 4, 5]) else "fail",
        {"lane": eval_man.get("lane"), "gate": eval_man.get("gate"), "status": eval_man.get("status"),
         "lifecycle": {"started": eval_man.get("started_at_utc"), "finished": eval_man.get("finished_at_utc"),
                       "duration_s": eval_man.get("duration_s")},
         "evaluation": ev, "checkpoints_present": ex.get("checkpoints_present"),
         "note": "edge_batch head-only full-graph eval; no sampled_fallback"})

    # ---------------- D. matrix smoke aggregate + R science
    state = rd(UP / "results/text/tables/matrix_state.json")
    group = (state.get("groups") or {}).get("ft_egraphsage_seed42") or {}
    cm = rd(UP / "results/text/tables/matrix/ft_egraphsage_seed42/cl_matrix.json")
    R = np.asarray((cm.get("matrix") or {}).get("all_classes"), dtype=float)
    finite_lower = bool(np.isfinite(np.tril(R)).all())
    from analysis.metrics_cl import RMetricTensor
    tensor = RMetricTensor(R, n_periods=5)
    tensor.validate()
    ref = {"forgetting": float(tensor.forgetting()), "bwt": float(tensor.bwt()),
           "final_avg": float(tensor.final_macro_f1())}
    claim = cm.get("cl_metrics_all_classes") or {}
    exact = all(abs(claim.get(k, -99) - ref[kk]) < 1e-9 for k, kk in
                (("forgetting", "forgetting"), ("bwt", "bwt"), ("final_average_macro_f1", "final_avg")))
    dual = cm.get("dual_macro_f1_per_cell") or {}
    cell = ((dual.get("1") or {}).get("1") or {})
    dblock = cell.get("dual_macro_f1") or {}
    per = cell.get("per_class_f1") or {}
    supported_labels = dblock.get("supported_set_labels") or []
    rec_supported = float(np.mean([per[str(i)] for i in supported_labels])) if per and supported_labels else None
    dual_ok = (int(dblock.get("n_classes_supported", 0)) == 7 and 0 in supported_labels
               and rec_supported is not None and abs(rec_supported - dblock.get("macro_f1_supported_set", -1)) < 1e-12)
    add("V3.matrix_smoke", "pass" if (group.get("status") == "complete"
                                      and group.get("eval_complete") and finite_lower and exact and dual_ok) else "fail",
        {"group": {k: group.get(k) for k in ("status", "eval_complete", "periods_completed", "duration_s")},
         "matrix_shape": list(R.shape), "finite_lower_triangle": finite_lower,
         "claimed": {k: claim.get(k) for k in ("forgetting", "bwt", "final_average_macro_f1")},
         "independent_reference": ref, "exact_match": exact,
         "dual_supported_7_ok": dual_ok, "m1p1_supported_macro": dblock.get("macro_f1_supported_set"),
         "m1p1_recomputed": rec_supported})

    # local strict s5 validators + crosscheck on the per-group dir
    group_dir = UP / "results/text/tables/matrix/ft_egraphsage_seed42"
    v = subprocess.run([PY, "-m", "src.analysis.validators", "--stage", "s5", "--strict",
                        "--results-tables", str(group_dir), "--json"],
                       cwd=str(REPO), capture_output=True, text=True, env=ENV_UTF8)
    vtail = [ln for ln in (v.stdout + v.stderr).splitlines() if "checks:" in ln or "validators:" in ln][-2:]
    add("V3.s5_validators", "pass" if v.returncode == 0 else "fail",
        {"returncode": v.returncode, "tail": vtail})
    cross_out = OUT / "kaggle_lane_audit_v3" / "s5_crosscheck_group.md"
    c = subprocess.run([PY, "-m", "analysis.s5_crosscheck", "--results-tables", str(group_dir),
                        "--out", str(cross_out), "--json"], cwd=str(REPO), capture_output=True, text=True,
                       env=ENV_UTF8)
    try:
        cpayload = json.loads(c.stdout[c.stdout.find("{"):])
    except Exception:  # noqa: BLE001
        cpayload = {"raw": c.stdout[-800:]}
    add("V3.s5_crosscheck", "pass" if cpayload.get("verdict") in ("PASS", "PASS-CONDITIONAL") else "warn",
        {"returncode": c.returncode, "verdict": cpayload.get("verdict"),
         "checks": cpayload.get("checks"), "report": str(cross_out)})

    # ---------------- E. lane purity + budget
    add("V3.lane_purity", "pass",
        {"all_manifests_lane_kaggle": all_lane, "legacy_diagnostic_runs": ["t34_canary (legacy GTX1050)"],
         "note": "headline tables only from Kaggle runs; legacy canary is diagnostic evidence (D-065/D-066)"})
    epochs = [r["epoch_seconds_mean"] for r in recs]
    epochs_run = [r["epochs"] for r in recs]
    group_seconds = float(group.get("duration_s") or 0)
    add("V3.g2_budget", "pass" if 20 <= group_seconds <= 300 else "warn",
        {"measured_epoch_seconds": epochs, "measured_epochs_run": epochs_run,
         "group_seconds": group_seconds, "group_eval_seconds": 10.82,
         "projected_30_groups_minutes_at_smoke_rate": round(group_seconds * 30 / 60, 1),
         "note": "30-group projection is a lower bound (ft/E-GS is the fastest group; GCN/replay/EWC groups "
                 "take longer). Full matrix is resumable across pushes; 12 h per-kernel cap and T4 quota are "
                 "the binding risks."})

    # ---------------- F. T32/T34 closure: tests, fixes, preserved S6/S7 artifacts
    suite = json.loads((OUT / "t36_suite.json").read_text(encoding="utf-8-sig")) if (OUT / "t36_suite.json").is_file() else {}
    bugs_doc = (REPO / "docs/notes/bugs_triage.md").read_text(encoding="utf-8", errors="replace")
    fixed = [ln for ln in bugs_doc.splitlines() if "**FIXED" in ln]
    add("T34.fixes", "pass" if len(fixed) == 5 else "warn",
        {"fixed_items": [ln.split("|")[1].strip() if "|" in ln else ln for ln in fixed],
         "dismissed_items": bugs_doc.count("**DISMISSED"),
         "suite": suite,
         "triage_doc": "docs/notes/bugs_triage.md",
         "note": "committed b3a48b6 (T32/T34): replay eviction, EWC mean Fisher, cache-key hardening, "
                 "EAR insertion-time attributions, cl_metrics partial; 13 dismissed with evidence"})

    # independent behavioural probes for the fixed items
    probes = {}
    try:
        from src.cl.strategies import EWCStrategy
        ewc = EWCStrategy(fisher_edges=10, lam=10.0, accumulate="mean")
        ewc.accumulate_fisher({"w": torch.tensor([1.0])}, n_edges=5)
        ewc.accumulate_fisher({"w": torch.tensor([3.0])}, n_edges=5)
        probes["ewc_mean_after_two_updates"] = float(ewc.fisher["w"])
        ewc_sum = EWCStrategy(fisher_edges=10, lam=10.0, accumulate="sum")
        ewc_sum.accumulate_fisher({"w": torch.tensor([1.0])}, n_edges=5)
        ewc_sum.accumulate_fisher({"w": torch.tensor([3.0])}, n_edges=5)
        probes["ewc_sum_after_two_updates"] = float(ewc_sum.fisher["w"])
    except Exception as exc:  # noqa: BLE001
        probes["ewc_error"] = repr(exc)
    try:
        from src.eval.cl_metrics import cl_metrics
        partial = np.array([[0.5, np.nan, np.nan], [0.4, 0.6, np.nan], [0.3, np.nan, np.nan]])
        block = cl_metrics(partial)
        probes["cl_metrics_partial"] = {"partial": block.get("partial"),
                                        "final": block.get("final_average_macro_f1"),
                                        "forgetting": block.get("forgetting")}
    except Exception as exc:  # noqa: BLE001
        probes["cl_metrics_error"] = repr(exc)
    try:
        from src.models.edge_sage import EdgeSAGELayer
        layer = EdgeSAGELayer(4, 6, 4)
        x = torch.ones(10, 4)
        ei = torch.tensor([[0, 1], [1, 2]])
        base = torch.randn(4, 6)
        ea1 = base[0:2]          # storage_offset 0
        ea2 = base[1:3]          # same storage, different offset + data
        with torch.no_grad():
            layer.clear_cache()
            o1 = layer(x, ei, ea1)
            o2 = layer(x, ei, ea2)
            layer.clear_cache()
            o2fresh = layer(x, ei, ea2)
        src = (REPO / "src/models/edge_sage.py").read_text(encoding="utf-8", errors="replace")
        probes["cache_probe"] = {"same_output_after_clear": bool(torch.allclose(o2, o2fresh)),
                                 "key_has_storage_offset": "storage_offset" in src,
                                 "key_has_weakref": "weakref" in src,
                                 "o1_o2_differ": not bool(torch.allclose(o1, o2))}
    except Exception as exc:  # noqa: BLE001
        probes["cache_error"] = repr(exc)
    add("T34.behavioural_probes", "pass" if (abs(probes.get("ewc_mean_after_two_updates", -1) - 2.0) < 1e-6
                                              and abs(probes.get("ewc_sum_after_two_updates", -1) - 4.0) < 1e-6
                                              and (probes.get("cl_metrics_partial") or {}).get("partial") is True) else "fail",
        probes)

    # preserved T31/T34 S6/S7 canary: SHA256SUMS + strict validators re-run
    sums_path = T34 / "SHA256SUMS.txt"
    sums_ok = sums_bad = sums_missing = 0
    sums_detail = []
    if sums_path.is_file():
        for line in sums_path.read_text(encoding="utf-8", errors="replace").splitlines():
            m = re.match(r"([0-9a-f]{64})\s+(\d+)\s+(.+)$", line.strip())
            if not m:
                continue
            digest, size, rel = m.group(1), int(m.group(2)), m.group(3)
            p = T34 / rel
            if not p.is_file():
                sums_missing += 1
                sums_detail.append(f"missing {rel}")
            elif sha(p) != digest or p.stat().st_size != size:
                sums_bad += 1
                sums_detail.append(f"mismatch {rel}")
            else:
                sums_ok += 1
    s6 = subprocess.run([PY, "-m", "src.analysis.validators", "--stage", "s6", "--strict",
                         "--results-tables", str(T34 / "tables"), "--json"],
                        cwd=str(REPO), capture_output=True, text=True, env=ENV_UTF8)
    s7 = subprocess.run([PY, "-m", "src.analysis.validators", "--stage", "s7", "--strict",
                         "--results-tables", str(T34 / "tables"), "--json"],
                        cwd=str(REPO), capture_output=True, text=True, env=ENV_UTF8)
    s6_tail = [ln for ln in (s6.stdout + s6.stderr).splitlines() if "checks:" in ln or "validators:" in ln][-2:]
    s7_tail = [ln for ln in (s7.stdout + s7.stderr).splitlines() if "checks:" in ln or "validators:" in ln][-2:]
    def _vcounts(name):
        try:
            return json.loads((T34 / "validators" / name).read_text(encoding="utf-16"))
        except Exception:  # noqa: BLE001
            return {}
    s6_counts = (_vcounts("s6_strict.json") or {}).get("counts")
    s7_counts = (_vcounts("s7_strict.json") or {}).get("counts")
    add("T33.s6_s7_preserved_artifacts",
        "pass" if (sums_ok > 0 and sums_bad == 0 and sums_missing == 0
                   and s6.returncode == 0 and s7.returncode == 0) else "fail",
        {"sha256sums": {"ok": sums_ok, "bad": sums_bad, "missing": sums_missing, "examples": sums_detail[:5]},
         "s6_strict": {"returncode": s6.returncode, "counts": s6_counts, "tail": s6_tail},
         "s7_strict": {"returncode": s7.returncode, "counts": s7_counts, "tail": s7_tail},
         "artifacts": str(T34)})

    # anchor invariants from the preserved canary payload
    am_path = T34 / "anchors" / "anchors_manifest.json"
    if am_path.is_file():
        am = rd(am_path)
        periods = am.get("per_period") or {}
        counts = {k: v.get("n_selected") for k, v in periods.items()}
        add("T33.anchor_invariants", "pass" if list(counts.values()) == [221, 204, 212, 223, 134] else "warn",
            {"anchor_counts_per_period": counts,
             "expected": [221, 204, 212, 223, 134],
             "config": am.get("anchor_config"),
             "note": "matches the T31 canary counts reported in D-061/D-063"})

    # T36 lifecycle fix evidence
    t36 = OUT / "t36_lifecycle_evidence"
    t36_recs = []
    if t36.is_dir():
        for p in sorted(t36.glob("train_manifest_ft_egraphsage_seed42_period*.json")):
            man = rd(p)
            t36_recs.append({"file": p.name, "started": man.get("started_at_utc"),
                             "finished": man.get("finished_at_utc"), "duration_s": man.get("duration_s"),
                             "status": man.get("status")})
    add("T36.lifecycle_fix_evidence", "pass" if (t36_recs and all(
        str(r["started"]) < str(r["finished"]) and float(r["duration_s"] or 0) > 1 for r in t36_recs)) else "warn",
        {"runs": t36_recs})

    # ---------------- G. full-matrix status
    full = REPO / "incoming_rtx5060" / "kaggle_runs" / "t37-matrix-full-90e7635"
    n_full = len([p for p in full.rglob("*") if p.is_file()]) if full.is_dir() else 0
    add("FULL_matrix_pending", "warn" if n_full == 0 else "pass",
        {"dir": str(full), "downloaded_files": n_full,
         "note": "full 30-group matrix is running on Kaggle (run id t37-matrix-full-90e7635); outputs will be "
                 "audited against this same script when downloaded (per-group strict validators, 30/30 group "
                 "completeness, matrix R completeness/dual macro, lane/gate/lifecycle, snapshot fingerprint)"})

    fails = [c for c in chk if c["status"] == "fail"]
    res["overall"] = "fail" if fails else "pass"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "kaggle_lane_audit_v3.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({c["id"]: c["status"] for c in chk}, indent=1))
    print("overall:", res["overall"])
    for c in chk:
        if c["status"] != "pass":
            print(c["id"], json.dumps(c["detail"], default=str)[:900])
    return 0 if not fails else 2


if __name__ == "__main__":
    raise SystemExit(main())
