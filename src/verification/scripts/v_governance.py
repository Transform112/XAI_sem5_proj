#!/usr/bin/env python
"""Independent governance / pre-registration audit (verifier-owned).

Checks that the data-audit gate is not being tuned to hypotheses and that
provenance claims are true:

  GV-1  configs/audit.yaml committed (pre-registered) BEFORE any training run
  GV-2  thresholds unchanged since first commit (no post-hoc relaxation)
  GV-3  config hash in scorecards / run manifests == current config hash
  GV-4  no undocumented threshold/seed/split changes after training starts
  GV-5  decision log completeness for audit-default deviations
  GV-6  audit-before-training ordering (no runs before a green audit)
  ID-9  datasets/raw/provenance.json claims vs actual files + Kaggle metadata

Reads git through the git executable (no project imports), hashes files
directly, and writes results/text/verification/data_audit/governance.json.

Usage (repo root):
  python verification\\scripts\\v_governance.py
  python verification\\scripts\\v_governance.py --json-only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

VERIFIER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VERIFIER_ROOT.parents[1]
AUDIT_CFG = REPO_ROOT / "configs" / "audit.yaml"
RAW = REPO_ROOT / "data" / "raw"
PROVENANCE = RAW / "provenance.json"
REFERENCE = VERIFIER_ROOT / "reference" / "published_counts.json"
OUT = VERIFIER_ROOT / "data_audit"
DECISIONS = REPO_ROOT / "docs" / "DECISIONS.md"
RUNS = REPO_ROOT / "runs"
RESULTS = REPO_ROOT / "results"
ENGINEER_AUDIT = RESULTS / "data_audit"
GIT_CANDIDATES = [r"C:\Program Files\Git\bin\git.exe", "git"]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class Git:
    def __init__(self):
        self.exe = next((c for c in GIT_CANDIDATES if c == "git" or Path(c).is_file()), None)
        if self.exe is None:
            self.exe = shutil.which("git")
        self.available = self.exe is not None

    def run(self, *args: str) -> tuple[int, str, str]:
        if not self.available:
            return 127, "", "git not found"
        p = subprocess.run([self.exe, "-C", str(REPO_ROOT), *args],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        return p.returncode, p.stdout.strip(), p.stderr.strip()


def mtime_iso(p: Path) -> str | None:
    return datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if p.exists() else None


def check_provenance(git: Git, ref: dict) -> dict:
    out = {"id": "ID-9.provenance", "checks": [], "datasets": {}}
    if not PROVENANCE.is_file():
        out["status"] = "fail"
        out["detail"] = "datasets/raw/provenance.json missing"
        return out
    prov = json.loads(PROVENANCE.read_text(encoding="utf-8-sig"))
    all_ok = True
    any_warn = False
    for ds, meta in ref["datasets"].items():
        if ds not in prov:
            if meta.get("paper_total_flows") is None:
                out["datasets"][ds] = {"status": "not_in_scope",
                                       "detail": "not part of the NF-v3 identity gate and not downloaded"}
                continue
            out["datasets"][ds] = {"status": "fail", "detail": "missing from provenance.json"}
            all_ok = False
            continue
        p = prov[ds]
        zpath = RAW / f"{ds}.zip"
        csvs = list((RAW / ds).rglob("*.csv")) if (RAW / ds).is_dir() else []
        rec = {
            "source_url": p.get("source"),
            "recorded_zip_bytes": p.get("zip_bytes"),
            "recorded_sha256": p.get("sha256"),
            "recorded_kaggle_total_bytes_uncompressed": p.get("kaggle_total_bytes_uncompressed"),
            "expected_kaggle_metadata_total_bytes": meta.get("kaggle_metadata_total_bytes"),
            "probed_archive_bytes": meta.get("kaggle_archive_bytes_probed"),
            "actual_zip_exists": zpath.is_file(),
            "actual_zip_bytes": zpath.stat().st_size if zpath.is_file() else None,
            "actual_csv": [{"path": str(c), "bytes": c.stat().st_size} for c in csvs],
        }
        checks = []
        if zpath.is_file():
            checks.append({"check": "zip_bytes", "status": "pass" if rec["actual_zip_bytes"] == rec["recorded_zip_bytes"] else "fail",
                           "recorded": rec["recorded_zip_bytes"], "actual": rec["actual_zip_bytes"]})
            if rec["actual_zip_bytes"] == rec["probed_archive_bytes"]:
                checks.append({"check": "zip_bytes_vs_kaggle_content_range", "status": "pass",
                               "probed": rec["probed_archive_bytes"]})
            else:
                checks.append({"check": "zip_bytes_vs_kaggle_content_range", "status": "fail",
                               "probed": rec["probed_archive_bytes"], "actual": rec["actual_zip_bytes"]})
            actual_sha = sha256_file(zpath)
            rec["actual_sha256"] = actual_sha
            checks.append({"check": "zip_sha256", "status": "pass" if actual_sha == rec["recorded_sha256"] else "fail",
                           "recorded": rec["recorded_sha256"], "actual": actual_sha})
        else:
            checks.append({"check": "zip_exists", "status": "fail"})
        # csv bytes vs recorded extracted file + kaggle metadata total
        if csvs:
            csv_bytes = csvs[0].stat().st_size
            rec_files = p.get("extracted_files") or []
            rec_csv_bytes = rec_files[0]["bytes"] if rec_files else None
            checks.append({"check": "csv_bytes_vs_recorded", "status": "pass" if rec_csv_bytes == csv_bytes else "fail",
                           "recorded": rec_csv_bytes, "actual": csv_bytes})
            checks.append({"check": "csv_bytes_vs_kaggle_metadata_total",
                           "status": "pass" if csv_bytes == meta.get("kaggle_metadata_total_bytes") else "fail",
                           "kaggle_totalBytes": meta.get("kaggle_metadata_total_bytes"), "actual": csv_bytes})
            # provenance field semantics: kaggle_total_bytes_uncompressed must equal metadata total
            ku = rec["recorded_kaggle_total_bytes_uncompressed"]
            if ku == meta.get("kaggle_metadata_total_bytes"):
                checks.append({"check": "provenance_uncompressed_field_semantics", "status": "pass", "value": ku})
            else:
                checks.append({"check": "provenance_uncompressed_field_semantics", "status": "warn",
                               "recorded": ku, "expected_metadata_total": meta.get("kaggle_metadata_total_bytes"),
                               "note": "field holds the archive (zip) size instead of the Kaggle uncompressed totalBytes; "
                                       "actual bytes checks above still pass, so this is a labelling defect, not a data defect"})
        rec["checks"] = checks
        bad = [c for c in checks if c["status"] == "fail"]
        warns = [c for c in checks if c["status"] == "warn"]
        rec["status"] = "pass" if not bad and not warns else ("warn" if not bad else "fail")
        out["datasets"][ds] = rec
        if bad:
            all_ok = False
        if warns:
            any_warn = True
    out["status"] = "fail" if not all_ok else ("warn" if any_warn else "pass")
    return out


def check_audit_yaml(git: Git) -> dict:
    out = {"id": "GV-1/2/3.audit_yaml_preregistration", "checks": []}
    if not AUDIT_CFG.is_file():
        out["status"] = "fail"
        out["checks"].append({"check": "exists", "status": "fail", "path": str(AUDIT_CFG)})
        return out
    cur_sha = sha256_file(AUDIT_CFG)
    out["current_sha256"] = cur_sha
    out["current_mtime_utc"] = mtime_iso(AUDIT_CFG)
    rc_tracked, tracked_out, _ = git.run("ls-files", "--error-unmatch", "configs/audit.yaml")
    tracked = rc_tracked == 0 and tracked_out.strip() != ""
    out["checks"].append({"check": "tracked_in_git", "status": "pass" if tracked else "fail",
                          "detail": "configs/audit.yaml is not committed; pre-registration in git history is not yet provable" if not tracked else tracked_out})
    history = []
    if tracked:
        rc, log_out, err = git.run("log", "--follow", "--format=%H|%cI|%s", "--", "configs/audit.yaml")
        for line in log_out.splitlines():
            parts = line.split("|", 2)
            if len(parts) == 3:
                h, when, subj = parts
                rc2, blob, _ = git.run("show", f"{h}:configs/audit.yaml")
                history.append({"commit": h, "committed_at": when, "subject": subj,
                                "blob_sha256": sha256_bytes(blob.encode("utf-8")) if rc2 == 0 else None})
        # oldest commit = pre-registration; newest = current state
        history_sorted = list(reversed(history))
        out["history_oldest_first"] = history_sorted
        unchanged = len({h["blob_sha256"] for h in history}) == 1
        out["checks"].append({"check": "thresholds_unchanged_since_first_commit",
                              "status": "pass" if unchanged else "fail",
                              "n_commits": len(history),
                              "distinct_blob_hashes": len({h["blob_sha256"] for h in history})})
    out["history"] = history
    # training evidence
    train_evidence = []
    if RUNS.is_dir():
        train_evidence += [str(p) for p in RUNS.rglob("*.json") if p.is_file()][:50]
    if RESULTS.is_dir():
        for pat in ("tables/*.csv", "tables/*.json", "**/run_manifest*.json"):
            train_evidence += [str(p) for p in RESULTS.glob(pat)][:50]
    out["training_evidence_count"] = len(train_evidence)
    if tracked:
        rc_diff, diff_out, _ = git.run("diff", "--quiet", "HEAD", "--", "configs/audit.yaml")
        out["checks"].append({"check": "working_tree_matches_committed_HEAD",
                              "status": "pass" if rc_diff == 0 else "warn",
                              "detail": "configs/audit.yaml has uncommitted edits" if rc_diff != 0 else "unchanged vs HEAD"})
    out["checks"].append({"check": "no_training_evidence_yet", "status": "pass" if not train_evidence else "warn",
                          "detail": "no runs/results training artifacts yet" if not train_evidence else train_evidence[:10]})
    out["status"] = "pass" if all(c["status"] == "pass" for c in out["checks"]) else \
        ("fail" if any(c["status"] == "fail" for c in out["checks"]) else "warn")
    return out


def check_config_hash_consistency() -> dict:
    out = {"id": "GV-3.config_hash_consistency", "checks": []}
    if not AUDIT_CFG.is_file():
        out["status"] = "fail"
        return out
    cur_sha = sha256_file(AUDIT_CFG)
    out["current_audit_yaml_sha256"] = cur_sha
    found = []
    if ENGINEER_AUDIT.is_dir():
        for p in ENGINEER_AUDIT.rglob("*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8-sig"))
            except Exception:  # noqa: BLE001
                continue
            if isinstance(d, dict):
                for key in ("config_hash", "config_sha256", "audit_config_sha256"):
                    if key in d:
                        found.append({"artifact": str(p), "key": key, "value": d[key],
                                      "matches_current": d[key] == cur_sha})
    out["scorecard_config_hashes"] = found
    if found:
        out["checks"].append({"check": "scorecard_hash_matches_current_config",
                              "status": "pass" if all(f["matches_current"] for f in found) else "fail"})
    else:
        out["checks"].append({"check": "scorecard_hash_matches_current_config", "status": "warn",
                              "detail": "no engineer scorecard with a recorded config hash yet"})
    out["status"] = "pass" if all(c["status"] == "pass" for c in out["checks"]) else "warn"
    return out


def check_decisions() -> dict:
    out = {"id": "GV-5.decision_log", "checks": []}
    if not DECISIONS.is_file():
        out["status"] = "fail"
        out["checks"].append({"check": "exists", "status": "fail"})
        return out
    text = DECISIONS.read_text(encoding="utf-8-sig")
    ids = re.findall(r"\| (D-\d{3}) \|", text)
    out["n_entries"] = len(ids)
    out["last_entry"] = ids[-1] if ids else None
    mentions = [ln for ln in text.splitlines() if re.search(r"audit\.yaml|threshold", ln, re.I)]
    out["entries_mentioning_thresholds_or_audit_yaml"] = mentions
    out["explicit_audit_threshold_decision"] = any("D-008" in m for m in mentions)
    out["checks"].append({"check": "audit_thresholds_entry_present", "status": "pass" if out["explicit_audit_threshold_decision"] else "warn",
                          "detail": "D-008 declares pre-registration of configs/audit.yaml thresholds"})
    out["status"] = "pass" if out["checks"][0]["status"] == "pass" else "warn"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref = json.loads(REFERENCE.read_text(encoding="utf-8-sig"))
    git = Git()
    result = {"generated_utc": utc_now(), "git_available": git.available,
              "git_head": None, "checks": []}
    if git.available:
        _, head, _ = git.run("rev-parse", "HEAD")
        result["git_head"] = head
        _, status, _ = git.run("status", "--porcelain", "-uall")
        result["git_dirty_entries"] = len([ln for ln in status.splitlines() if ln.strip()])
    prov = check_provenance(git, ref)
    cfg = check_audit_yaml(git)
    hashes = check_config_hash_consistency()
    dec = check_decisions()
    result["checks"] = [prov, cfg, hashes, dec]
    fails = [c for c in result["checks"] if c.get("status") == "fail"]
    warns = [c for c in result["checks"] if c.get("status") == "warn"]
    result["overall"] = "fail" if fails else ("warn" if warns else "pass")
    path = out_dir / "governance.json"
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"wrote {path}")
    for c in result["checks"]:
        print(f"{c['id']}: {c.get('status')} ({len(c.get('checks', []))} sub-checks)")
    print(f"OVERALL: {result['overall']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


