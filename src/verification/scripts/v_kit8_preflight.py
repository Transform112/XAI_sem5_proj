#!/usr/bin/env python
"""Kit v8 (remote_rtx5060_kit_7eae31c.zip) pre-flight audit.

Verifies the kit the user will run: archive/lane/script fixes, frozen artifacts,
code snapshot identity vs the audited T27 runtime code (commit 0af6c91), and the
new-run output collector. Evidence: results/text/verification/data_audit/kit8_preflight.json.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path

REPO = Path(r"C:\Users\iusearchbtw\Desktop\XAI_proj")
KIT_ZIP = REPO / "remote_rtx5060_kit_7eae31c.zip"
KIT = REPO / "verification" / "data_audit" / "kit8_preflight" / "kit"
OUT = REPO / "verification" / "data_audit"
GIT = r"C:\Program Files\Git\bin\git.exe"
AUDITED_COMMIT = "0af6c91"
SNAPSHOT_COMMIT = "7eae31c"
RUNTIME_ROOTS = ("src", "scripts", "configs", "tests")


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([GIT, "-C", str(REPO), *args], capture_output=True)


def main() -> int:
    res: dict = {"kit_zip": str(KIT_ZIP), "kit_zip_sha256": sha(KIT_ZIP), "checks": []}
    chk = res["checks"]

    def add(i, status, detail):
        chk.append({"id": i, "status": status, "detail": detail})

    # ---- lane / script fixes
    ps1 = [(p, p.read_text(encoding="utf-8", errors="replace")) for p in KIT.rglob("*.ps1")]
    canonical_refs = [{"file": str(p.relative_to(KIT)), "line": line.strip()}
                      for p, text in ps1 for line in text.splitlines()
                      if re.search(r"lane['\",\s]+canonical|--lane\s+canonical|canonical['\"]", line, re.I)]
    rtx_refs = sum(1 for _, text in ps1 for line in text.splitlines() if "rtx5060" in line.lower())
    gb_refs = sum(1 for _, text in ps1 for line in text.splitlines() if "graph_batch" in line.lower())
    stage5 = (KIT / "scripts" / "05_run_stage.ps1")
    stage5_text = stage5.read_text(encoding="utf-8", errors="replace") if stage5.is_file() else ""
    add("KIT8.lane_scripts", "pass" if not canonical_refs and rtx_refs >= 5 and gb_refs >= 1 else "fail",
        {"canonical_lane_refs": canonical_refs, "rtx5060_refs": rtx_refs, "graph_batch_refs": gb_refs,
         "train_stage_has_graph_batch": "--batching', 'graph_batch'" in stage5_text or "graph_batch" in stage5_text,
         "matrix_stage_present": "run_matrix.py" in stage5_text,
         "explain_stage_present": "run_anchors_explain.py" in stage5_text})

    # ---- output collector (new-run collection + timing summary)
    stage6 = (KIT / "scripts" / "06_make_output.ps1")
    s6 = stage6.read_text(encoding="utf-8", errors="replace") if stage6.is_file() else ""
    collector_ok = all(k in s6 for k in ("run_manifests", "timing_summary", "repro", "results"))
    add("KIT8.output_collector", "pass" if collector_ok else "warn",
        {"collects_run_manifests": "run_manifests" in s6,
         "generates_timing_summary": "timing_summary" in s6,
         "copies_repro_new_runs": "repro" in s6,
         "collects_results": "results" in s6})

    # ---- frozen artifacts
    known = {
        "results/text/data_audit/preprocessor_state.json": sha(REPO / "results/text/data_audit/preprocessor_state.json"),
        "results/text/data_audit/feature_manifest.csv": sha(REPO / "results/text/data_audit/feature_manifest.csv"),
        "results/text/data_audit/split_manifest.json": sha(REPO / "results/text/data_audit/split_manifest.json"),
    }
    repo_spl = REPO / "datasets/splits/NF-UNSW-NB15-v3/session_aware_1_4"
    for p in repo_spl.rglob("period_*_masks.npz"):
        known["datasets/splits/NF-UNSW-NB15-v3/session_aware_1_4/" + str(p.relative_to(repo_spl)).replace("\\", "/")] = sha(p)
    matched, unmatched = [], []
    for p in sorted((KIT / "frozen").rglob("*")):
        if not p.is_file() or p.name.lower().startswith("readme"):
            continue
        rel = str(p.relative_to(KIT / "frozen")).replace("\\", "/")
        expected = known.get(rel)
        if expected is None:
            candidates = [k for k in known if rel.endswith(k)]
            expected = known[candidates[0]] if candidates else None
        if expected == sha(p):
            matched.append(rel)
        else:
            unmatched.append({"path": rel, "sha256": sha(p), "expected": expected})
    split_man_path = next((KIT / "frozen").rglob("split_manifest.json"), None)
    split_man = json.loads(split_man_path.read_text(encoding="utf-8-sig")) if split_man_path else {}
    add("KIT8.frozen_artifacts",
        "pass" if len(matched) == 13 and not unmatched and split_man.get("frozen_hash") ==
        "010a80d955802a2f62dedf472a3274e1afee17fb4b21c57541fe428b8feb47ff" else "fail",
        {"n_matched": len(matched), "n_total": len(matched) + len(unmatched), "unmatched": unmatched,
         "split_manifest_frozen_hash": split_man.get("frozen_hash")})

    # ---- code snapshot identity
    snap_zip = next((KIT / "code").glob("xdrift_repo_*.zip"), None)
    snap_sha = sha(snap_zip) if snap_zip else None
    repo_snap = REPO / "remote_rtx5060" / "code" / "xdrift_repo_7eae31c.zip"
    repo_snap_sha = sha(repo_snap) if repo_snap.is_file() else None
    content_cmp = None
    if snap_zip:
        with zipfile.ZipFile(snap_zip) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/") and
                     any(n == r or n.startswith(r + "/") for r in RUNTIME_ROOTS)]
            same = crlf = diff = missing = 0
            diffs = []
            for rel in names:
                data = zf.read(rel)
                proc = git("show", f"{AUDITED_COMMIT}:{rel}")
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
        content_cmp = {"runtime_files": len(names), "same": same, "crlf_only": crlf,
                       "content_diff": diff, "missing": missing, "examples": diffs[:5]}
    # snapshot commit ancestry: 7eae31c must descend from audited 0af6c91
    rc = git("rev-parse", SNAPSHOT_COMMIT)
    snap_commit = rc.stdout.strip().decode() if rc.returncode == 0 else None
    contains_audited = (subprocess.run([GIT, "-C", str(REPO), "merge-base", "--is-ancestor",
                                       AUDITED_COMMIT, SNAPSHOT_COMMIT], capture_output=True).returncode == 0
                        if snap_commit else None)
    add("KIT8.code_snapshot_identity",
        "pass" if (snap_sha and snap_sha == repo_snap_sha and content_cmp and content_cmp["content_diff"] == 0
                   and content_cmp["missing"] == 0 and contains_audited) else "fail",
        {"snapshot_zip": snap_zip.name if snap_zip else None, "snapshot_zip_sha256": snap_sha,
         "repo_snapshot_zip_sha256": repo_snap_sha, "matches_repo_copy": snap_sha == repo_snap_sha,
         "snapshot_commit": snap_commit, "descends_from_audited_0af6c91": contains_audited,
         "runtime_content_vs_0af6c91": content_cmp,
         "note": "snapshot is the runtime-only code snapshot at kit HEAD 7eae31c; runtime code must be "
                 "identical to audited 0af6c91 (CRLF-only differences expected from the Windows snapshot)"})

    # ---- config sanity inside snapshot
    cfg_ok = None
    if snap_zip:
        with zipfile.ZipFile(snap_zip) as zf:
            train_yaml = zf.read("configs/train.yaml").decode("utf-8", "replace") if "configs/train.yaml" in zf.namelist() else ""
            cfg_ok = ("batching: graph_batch" in train_yaml and "optimized_dispatch: true" in train_yaml)
            has_fast = "src/train/fast.py" in zf.namelist()
            has_sampler = "src/train/sampler.py" in zf.namelist()
            has_models = "src/models/edge_sage.py" in zf.namelist()
    else:
        has_fast = has_sampler = has_models = False
    add("KIT8.snapshot_config", "pass" if (cfg_ok and has_fast and has_sampler and has_models) else "fail",
        {"train_yaml_graph_batch": bool(cfg_ok), "has_fast_py": bool(has_fast),
         "has_sampler_py": bool(has_sampler), "has_edge_sage": bool(has_models)})

    # ---- expected timings documented
    readme = (KIT / "README.md").read_text(encoding="utf-8", errors="replace") if (KIT / "README.md").is_file() else ""
    add("KIT8.expected_timing_docs", "pass" if re.search(r"2.?3\s*s/epoch|2–3 s|2-3 s", readme) else "warn",
        {"readme_mentions_expected_epoch": bool(re.search(r"2.?3\s*s/epoch|2–3 s|2-3 s", readme)),
         "readme_head": readme[:300]})

    has_fail = any(c["status"] == "fail" for c in chk)
    res["overall"] = "fail" if has_fail else "pass"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "kit8_preflight.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({c["id"]: c["status"] for c in chk}, indent=1))
    print("overall:", res["overall"])
    for c in chk:
        if c["status"] != "pass":
            print(c["id"], json.dumps(c["detail"], default=str)[:700])
    return 0 if not has_fail else 2


if __name__ == "__main__":
    raise SystemExit(main())
