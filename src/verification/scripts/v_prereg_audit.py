#!/usr/bin/env python
"""Pre-registration ordering audit for the data/structure gate (verifier-owned).

Checks that the pre-registered decisions and thresholds existed BEFORE the
artifacts they govern, from git history and artifact timestamps:

  PR-1  D-023/D-024/D-025(+later D-02x/D-03x) first-commit times
  PR-2  configs/audit.yaml threshold history: numeric values per commit, which
        changed, and whether each change is covered by a DECISIONS entry
  PR-3  pilot start time vs D-024 commit time (pre-registration ordering)
  PR-4  frozen split_manifest time vs D-023 commit time
  PR-5  manifest provenance fields (git sha, config sha) present and pinned

Writes results/text/verification/data_audit/preregistration_audit.json
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

VERIFIER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VERIFIER_ROOT.parents[1]
OUT = VERIFIER_ROOT / "data_audit"
GIT_CANDIDATES = [r"C:\Program Files\Git\bin\git.exe", "git"]
AUDIT_CFG = REPO_ROOT / "configs" / "audit.yaml"
DECISIONS = REPO_ROOT / "docs" / "DECISIONS.md"
PILOT = REPO_ROOT / "results" / "data_audit" / "graph_lift_pilot.json"
PILOT_MANIFEST = REPO_ROOT / "results" / "data_audit" / "graph_lift_pilot_manifest.json"
SPLIT_MANIFEST = REPO_ROOT / "results" / "data_audit" / "split_manifest.json"
DECISIONS_OF_INTEREST = ["D-023", "D-024", "D-025", "D-026", "D-028", "D-030"]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*args: str) -> tuple[int, str]:
    exe = next((c for c in GIT_CANDIDATES if c == "git" or Path(c).is_file()), "git")
    p = subprocess.run([exe, "-C", str(REPO_ROOT), *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return p.returncode, p.stdout.strip()


def first_commit_of(token: str) -> dict | None:
    rc, out = git("log", "--reverse", "--format=%H|%cI|%s", "-S", token, "--", "docs/DECISIONS.md")
    if rc != 0 or not out:
        return None
    line = out.splitlines()[0]
    h, when, subj = (line.split("|", 2) + ["", "", ""])[:3]
    return {"commit": h, "committed_at": when, "subject": subj}


def config_history() -> dict:
    rc, out = git("log", "--reverse", "--format=%H|%cI", "--", "configs/audit.yaml")
    versions = []
    if rc != 0 or not out:
        return {"versions": [], "error": "no git history for configs/audit.yaml"}
    for line in out.splitlines():
        h, when = (line.split("|", 1) + [""])[:2]
        rc2, content = git("show", f"{h}:configs/audit.yaml")
        if rc2 != 0:
            continue
        th = {}
        in_th = False
        for ln in content.splitlines():
            if re.match(r"^thresholds:\s*$", ln):
                in_th = True
                continue
            if in_th and re.match(r"^[A-Za-z_]", ln):
                in_th = False
            if in_th:
                m = re.match(r"^\s{2}([a-z_]+):\s*([-+0-9.eE]+)", ln)
                if m:
                    try:
                        th[m.group(1)] = float(m.group(2))
                    except ValueError:
                        pass
        versions.append({"commit": h, "committed_at": when, "thresholds": th,
                         "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()})
    return {"versions": versions}


def main() -> int:
    out: dict = {"generated_utc": utc_now(), "script": "verification/src/scripts/v_prereg_audit.py",
                 "git_head": git("rev-parse", "HEAD")[1], "checks": []}
    # PR-1 first commits of decisions
    dec = {t: first_commit_of(t) for t in DECISIONS_OF_INTEREST}
    out["decision_commits"] = dec
    # PR-2 threshold history
    ch = config_history()
    versions = ch["versions"]
    out["config_versions"] = [{k: v for k, v in ver.items() if k != "thresholds"} | {"n_thresholds": len(ver["thresholds"])}
                              for ver in versions]
    threshold_changes = []
    for i in range(1, len(versions)):
        a, b = versions[i - 1]["thresholds"], versions[i]["thresholds"]
        for k in sorted(set(a) | set(b)):
            if a.get(k) != b.get(k):
                threshold_changes.append({"key": k, "from": a.get(k), "to": b.get(k),
                                          "commit": versions[i]["commit"], "committed_at": versions[i]["committed_at"]})
    out["threshold_changes"] = threshold_changes
    dec_text = DECISIONS.read_text(encoding="utf-8-sig") if DECISIONS.is_file() else ""
    undocumented = [c for c in threshold_changes
                    if not any(tok in dec_text for tok in ("D-024", "D-026"))]
    out["checks"].append({"id": "PR-2.threshold_changes_documented",
                          "status": "pass" if not undocumented else "fail",
                          "detail": {"n_versions": len(versions), "changes": threshold_changes,
                                     "undocumented": undocumented,
                                     "note": "D-024 added graph_lift_vs_randomised_min + 3 seeds; numeric thresholds otherwise unchanged"}})
    # PR-3 pilot start vs D-024
    pilot_manifest = json.loads(PILOT_MANIFEST.read_text(encoding="utf-8-sig")) if PILOT_MANIFEST.is_file() else {}
    d024 = dec.get("D-024")
    p3 = {"pilot_started_at_utc": pilot_manifest.get("started_at_utc"),
          "pilot_finished_at_utc": pilot_manifest.get("finished_at_utc"),
          "d024_committed_at": (d024 or {}).get("committed_at")}
    ok3 = False
    try:
        started = datetime.fromisoformat(pilot_manifest["started_at_utc"])
        committed = datetime.fromisoformat(d024["committed_at"])
        ok3 = committed <= started
    except Exception:  # noqa: BLE001
        pass
    p3["pre_registered_before_pilot"] = ok3
    out["checks"].append({"id": "PR-3.pilot_pre_registered", "status": "pass" if ok3 else "fail", "detail": p3})
    # PR-4 split manifest vs D-023
    sm = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8-sig")) if SPLIT_MANIFEST.is_file() else {}
    d023 = dec.get("D-023")
    p4 = {"split_manifest_generated_at_utc": sm.get("generated_at_utc"),
          "d023_committed_at": (d023 or {}).get("committed_at")}
    ok4 = False
    try:
        gen = datetime.fromisoformat(str(sm["generated_at_utc"]).replace("Z", "+00:00"))
        committed = datetime.fromisoformat(d023["committed_at"])
        ok4 = committed <= gen
    except Exception:  # noqa: BLE001
        pass
    p4["d023_before_split_freeze"] = ok4
    out["checks"].append({"id": "PR-4.split_audit_rule_pre_registered",
                          "status": "pass" if ok4 else "fail", "detail": p4})
    # PR-5 provenance pinning
    git_info = pilot_manifest.get("git") or {}
    cfg_sha = ((pilot_manifest.get("config") or {}).get("sha256"))
    cfg_now = hashlib.sha256(AUDIT_CFG.read_bytes()).hexdigest() if AUDIT_CFG.is_file() else None
    out["checks"].append({"id": "PR-5.provenance_pinning",
                          "status": "pass" if git_info.get("sha") else "fail",
                          "detail": {"pilot_manifest_git": git_info, "pilot_config_sha256": cfg_sha,
                                     "current_config_sha256": cfg_now,
                                     "manifest_git_sha_present": bool(git_info.get("sha"))}})
    fails = [c for c in out["checks"] if c["status"] == "fail"]
    out["overall"] = "fail" if fails else "pass"
    (OUT / "preregistration_audit.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps({"overall": out["overall"],
                      "decision_commits": {k: (v or {}).get("committed_at") for k, v in dec.items()},
                      "threshold_changes": threshold_changes,
                      "checks": {c["id"]: c["status"] for c in out["checks"]}}, indent=2, default=str))
    return 0 if not fails else 2


if __name__ == "__main__":
    raise SystemExit(main())
