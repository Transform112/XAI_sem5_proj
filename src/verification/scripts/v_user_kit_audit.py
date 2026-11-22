"""Independent audit of the user's RTX 5060 kit output (G0/env + integrity only).

User output: incoming_rtx5060/output_20261102_001738.zip (sha256 465a47a2...),
unpacked at incoming_rtx5060/unpacked_output/. Custom code at
incoming_rtx5060/output/custom_code/.

Scope per Lead: verify env_check + frozen artifact SHA-256 list + output structure.
Training artifacts are DIAGNOSTIC-ONLY (kit snapshot 1810b07 != audited v7b
8187bb7; user modified code). No G2 conversion.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path

REPO = Path(r"C:\Users\iusearchbtw\Desktop\XAI_proj")
OUT = REPO / "verification" / "data_audit"
ROOT = REPO / "incoming_rtx5060" / "unpacked_output"
CUSTOM = REPO / "incoming_rtx5060" / "output" / "custom_code"
ZIP = REPO / "incoming_rtx5060" / "output_20261102_001738.zip"
GIT = r"C:\Program Files\Git\bin\git.exe"
AUDITED_SNAPSHOT = "8187bb7"
T24_COMMIT = "03b720c81ea6dbd87485279f5ea1fccba4f0832b"


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def known_frozen() -> dict[str, str]:
    known = {
        "results/text/data_audit/preprocessor_state.json": sha(REPO / "results/text/data_audit/preprocessor_state.json"),
        "results/text/data_audit/feature_manifest.csv": sha(REPO / "results/text/data_audit/feature_manifest.csv"),
        "results/text/data_audit/split_manifest.json": sha(REPO / "results/text/data_audit/split_manifest.json"),
    }
    repo_spl = REPO / "datasets/splits/NF-UNSW-NB15-v3/session_aware_1_4"
    for p in repo_spl.rglob("period_*_masks.npz"):
        rel = "datasets/splits/NF-UNSW-NB15-v3/session_aware_1_4/" + str(p.relative_to(repo_spl)).replace("\\", "/")
        known[rel] = sha(p)
    return known


def main() -> int:
    res: dict = {"zip": str(ZIP), "zip_sha256": sha(ZIP), "root": str(ROOT), "checks": []}
    chk = res["checks"]

    def add(i, status, detail):
        chk.append({"id": i, "status": status, "detail": detail})

    # ---------------- structure
    files = [p for p in ROOT.rglob("*") if p.is_file()]
    top = {}
    for p in files:
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        top[rel.split("/")[0]] = top.get(rel.split("/")[0], 0) + 1
    add("USERKIT.structure", "pass" if {"env", "artifacts", "logs"}.issubset(top) else "warn",
        {"file_count": len(files), "top_level": top,
         "has_summary": (ROOT / "SUMMARY.md").is_file(), "has_inventory": (ROOT / "INVENTORY.txt").is_file()})

    # ---------------- env check
    env_files = list(ROOT.rglob("env_check_rtx5060.json"))
    env = json.loads(env_files[0].read_text(encoding="utf-8-sig")) if env_files else {}
    checks = env.get("checks") or {}
    td = {}
    try:
        td = json.loads((checks.get("torch") or {}).get("detail") or "{}")
    except Exception:  # noqa: BLE001
        td = {}
    pg = {}
    try:
        pg = json.loads((checks.get("pyg_explainer") or {}).get("detail") or "{}")
    except Exception:  # noqa: BLE001
        pg = {}
    det = (checks.get("determinism") or {}).get("detail")
    all_ok = bool(env.get("ok")) and all(bool(v.get("ok")) for v in checks.values())
    env_ok = (td.get("capability") == [12, 0] and "RTX 5060" in str(td.get("device_name"))
              and str(td.get("torch", "")).startswith("2.7.1+cu128") and str(td.get("cuda_build")) == "12.8"
              and "torch_geometric 2.7.0" in str((checks.get("torch_geometric") or {}).get("detail"))
              and pg.get("pyg_explainer") == "ok" and "edge" in str(pg.get("task_level_used"))
              and det and "identical losses" in str(det) and all_ok)
    add("USERKIT.env_rtx5060", "pass" if env_ok else "fail",
        {"python": env.get("python"), "platform": env.get("platform"),
         "torch": td.get("torch"), "cuda_build": td.get("cuda_build"),
         "device_name": td.get("device_name"), "capability": td.get("capability"),
         "device_count": td.get("device_count"), "gpu_matmul": td.get("gpu_matmul"),
         "pyg": (checks.get("torch_geometric") or {}).get("detail"),
         "pyg_explainer": pg, "determinism": det,
         "all_checks_ok": all_ok, "checks": {k: v.get("ok") for k, v in checks.items()},
         "source": str(env_files[0]) if env_files else None})

    # ---------------- frozen artifacts
    frozen_files = list(ROOT.rglob("frozen_artifacts.txt"))
    frozen_lines = frozen_files[0].read_text(encoding="utf-8", errors="replace").splitlines() if frozen_files else []
    entries = []
    for line in frozen_lines:
        line = line.lstrip("\ufeff")
        m = re.match(r"\s*(\S.*?)\s+sha256=([0-9a-f]{64})\s*$", line)
        if m:
            rel = m.group(1).replace("\\", "/").lstrip("\ufeff")
            entries.append((m.group(2), rel))
    known = known_frozen()
    matched, unmatched = [], []
    artifact_entries = 0
    for digest, rel in entries:
        if rel.lstrip("\ufeff").lower().startswith("readme"):
            continue
        artifact_entries += 1
        expected = known.get(rel)
        if expected is None:
            cands = [k for k in known if rel.endswith(k)]
            expected = known[cands[0]] if cands else None
        if expected == digest:
            matched.append(rel)
        else:
            unmatched.append({"path": rel, "sha256": digest, "expected": expected})
    split_candidates = list(ROOT.rglob("split_manifest.json"))
    split_man = {}
    split_sha_actual = None
    if split_candidates:
        split_path = split_candidates[0]
        split_man = json.loads(split_path.read_text(encoding="utf-8-sig"))
        split_sha_actual = sha(split_path)
    add("USERKIT.frozen_artifacts",
        "pass" if (artifact_entries == 13 and len(matched) == 13 and not unmatched
                   and split_man.get("frozen_hash") == "010a80d955802a2f62dedf472a3274e1afee17fb4b21c57541fe428b8feb47ff"
                   and split_sha_actual == known.get("results/text/data_audit/split_manifest.json"))
        else "fail" if unmatched else "warn",
        {"n_entries_total": len(entries), "n_artifact_entries": artifact_entries, "n_verified": len(matched),
         "readme_present": any(rel.lower().startswith("readme") for _, rel in entries),
         "unmatched": unmatched, "split_manifest_frozen_hash": split_man.get("frozen_hash"),
         "split_manifest_file_sha256": split_sha_actual,
         "split_manifest_sha_matches_repo": split_sha_actual == known.get("results/text/data_audit/split_manifest.json"),
         "verified_masks": [r for r in matched if r.endswith("_masks.npz")],
         "source": str(frozen_files[0]) if frozen_files else None})

    # ---------------- snapshot identity
    snap_files = list(ROOT.rglob("code_snapshot.txt"))
    snap_text = snap_files[0].read_text(encoding="utf-8", errors="replace") if snap_files else ""
    snap_zip_sha = None
    m = re.search(r"code_snapshot_sha256\s*=\s*([0-9a-f]{64})", snap_text)
    if m:
        snap_zip_sha = m.group(1)
    snap_kb = None
    m = re.search(r"xdrift_repo_([0-9a-f]{7,40})\.zip", snap_text)
    if m:
        snap_kb = m.group(1)
    snap_commit = None
    if snap_kb:
        rc, full = subprocess.run([GIT, "-C", str(REPO), "rev-parse", snap_kb],
                                  capture_output=True, text=True).returncode, \
            subprocess.run([GIT, "-C", str(REPO), "rev-parse", snap_kb],
                           capture_output=True, text=True).stdout.strip()
        if rc == 0:
            snap_commit = full
    contains_t24 = None
    if snap_commit:
        contains_t24 = subprocess.run([GIT, "-C", str(REPO), "merge-base", "--is-ancestor", T24_COMMIT, snap_commit],
                                      capture_output=True).returncode == 0
    add("USERKIT.snapshot_identity", "warn",
        {"snapshot_label": snap_kb, "snapshot_zip_sha256": snap_zip_sha,
         "snapshot_commit_resolves": bool(snap_commit), "snapshot_commit": snap_commit,
         "audited_v7b_snapshot": AUDITED_SNAPSHOT,
         "is_audited_v7b": bool(snap_kb == AUDITED_SNAPSHOT),
         "contains_T24_sampled_trainer": contains_t24,
         "extracted_files_reported": (re.search(r"extracted_files\s*=\s*(\d+)", snap_text).group(1)
                                      if re.search(r"extracted_files\s*=\s*(\d+)", snap_text) else None),
         "note": "kit snapshot is 1810b07 (pre-T24 code), NOT the audited kit-v7b 8187bb7; the user also "
                 "modified the code (custom_code/CHANGES.md). Training artefacts in this output are "
                 "DIAGNOSTIC-ONLY and cannot be used as the canonical G2 anchor."})

    # ---------------- custom code (noted, not audited)
    changes = CUSTOM / "CHANGES.md"
    custom_files = [p for p in CUSTOM.rglob("*") if p.is_file()] if CUSTOM.is_dir() else []
    add("USERKIT.custom_code_noted", "pass" if changes.is_file() else "warn",
        {"custom_dir": str(CUSTOM), "file_count": len(custom_files),
         "changes_md_head": changes.read_text(encoding="utf-8", errors="replace")[:800] if changes.is_file() else None,
         "note": "user speed edits described in CHANGES.md (cached static aggregation, batch-only edge head, "
                 "GCN coalesced normalisation, cached eval tensors, set_to_none); engineer diffs/integrates "
                 "into T27; verifier does not audit user code directly; timings diagnostic-only"})

    # ---------------- training artefacts classification
    train_mans = [p for p in ROOT.rglob("train_manifest_*.json")]
    man_info = []
    for p in train_mans:
        try:
            man = json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:  # noqa: BLE001
            continue
        timing = ((man.get("extra") or {}).get("timing") or {})
        man_info.append({"file": str(p.relative_to(ROOT)).replace("\\", "/"),
                         "lane": man.get("lane"), "status": man.get("status"),
                         "started": man.get("started_at_utc"), "git": (man.get("git") or {}).get("sha"),
                         "device": timing.get("device"), "epoch_seconds_mean": timing.get("epoch_seconds_mean"),
                         "test_macro_f1": (man.get("extra") or {}).get("test_macro_f1"),
                         "code_fingerprint": (man.get("code") or {}).get("sha256")})
    add("USERKIT.training_artifacts_diagnostic_only", "warn",
        {"n_train_manifests": len(man_info), "manifests": man_info,
         "note": "stale repo/GTX-1050 smoke artefacts copied into the kit (not new RTX 5060 runs); "
                 "training numbers are DIAGNOSTIC-ONLY: snapshot mismatch + custom code + non-canonical lane. "
                 "No G2 conversion from this output."})

    # ---------------- diagnostic timing logs
    log_dir = ROOT / "logs"
    log_files = sorted(log_dir.glob("05_train_*.log")) if log_dir.is_dir() else []
    diag = []
    for log_path in log_files:
        raw = log_path.read_bytes()
        if raw.startswith(b"\xff\xfe"):
            text = raw.decode("utf-16-le", errors="replace")
        elif raw.startswith(b"\xfe\xff"):
            text = raw.decode("utf-16-be", errors="replace")
        else:
            text = raw.decode("utf-8-sig", errors="replace")

        def grab(pat):
            m = re.search(pat, text)
            return m.group(1) if m else None

        model = grab(r'"model":\s*"(\w+)"')
        epoch_s = grab(r'"epoch_seconds_mean":\s*([0-9.]+)')
        epochs_n = grab(r'"epochs":\s*(\d+)')
        edges = grab(r'"edges":\s*(\d+)')
        test_f1 = grab(r'"test_macro_f1":\s*([0-9.]+)')
        val_f1 = grab(r'"val_macro_f1":\s*([0-9.]+)')
        train_s = grab(r'"train_seconds":\s*([0-9.]+)')
        device = grab(r'"device":\s*"([^"]+)"')
        diag.append({
            "log": log_path.name,
            "model": model,
            "device": device,
            "epoch_seconds_mean": float(epoch_s) if epoch_s else None,
            "epochs": int(epochs_n) if epochs_n else None,
            "edges": int(edges) if edges else None,
            "train_seconds": float(train_s) if train_s else None,
            "test_macro_f1": float(test_f1) if test_f1 else None,
            "val_macro_f1": float(val_f1) if val_f1 else None,
            "line_count": len(text.splitlines()),
        })
    fullsize = [d for d in diag if d.get("edges") and d["edges"] >= 1_000_000]
    add("USERKIT.diagnostic_timing_logs", "pass" if diag else "warn",
        {"logs": diag,
         "full_size_diagnostic_runs": fullsize,
         "note": "RTX 5060 diagnostic runs of the user's modified kit code (snapshot 1810b07 + custom speed "
                 "edits); full-size P1. Numbers are DIAGNOSTIC-ONLY: non-canonical code path/semantics and "
                 "snapshot, NOT comparable to the frozen sampled-trainer baselines and NOT usable for G2. "
                 "Engineer reviews/integrates the custom code into T27; a clean kit v8 will produce the "
                 "canonical RTX 5060 G2 anchor."})

    # ---------------- overall
    has_fail = any(c["status"] == "fail" for c in chk)
    res["overall"] = "fail" if has_fail else "pass-with-caveats"
    res["g2_conversion"] = ("NOT converted. Kit snapshot 1810b07 != audited kit-v7b 8187bb7 and user code is "
                            "modified; training artefacts are diagnostic-only. G2 remains conditional on a clean "
                            "kit-v8 run of the audited T27 optimized trainer on the RTX 5060.")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "user_kit_1810b07_audit.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({c["id"]: c["status"] for c in chk}, indent=1))
    print("overall:", res["overall"])
    for c in chk:
        print("---", c["id"], "---")
        print(json.dumps(c["detail"], indent=1, default=str)[:2500])
    return 0 if not has_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
