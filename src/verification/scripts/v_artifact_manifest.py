#!/usr/bin/env python
"""Artifact manifest signer/verifier (verifier-owned; reusable at every gate).

Modes:
  sign    hash every artifact matching the given globs, write a manifest JSON
          (path, bytes, sha256, mtime) under verification/manifests/
  verify  compare current files with a previously written manifest; report
          missing/changed/new files
  check-run-manifests
          scan run manifests (docs/07 schema) and verify the split/processed/
          config hashes they claim match the current frozen artifacts

Usage:
  python verification\\scripts\\v_artifact_manifest.py sign --glob "datasets/splits/**/*" --name splits_v1
  python verification\\scripts\\v_artifact_manifest.py verify --manifest verification/manifests/splits_v1.json
  python verification\\scripts\\v_artifact_manifest.py check-run-manifests --runs runs
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MANIFEST_DIR = REPO_ROOT / "verification" / "manifests"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def collect(globs: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for g in globs:
        for p in REPO_ROOT.glob(g):
            if p.is_file():
                rel = str(p.relative_to(REPO_ROOT))
                out[rel] = {"bytes": p.stat().st_size, "sha256": sha256_file(p),
                            "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    return out


def sign(globs: list[str], name: str) -> Path:
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    files = collect(globs)
    data = {"name": name, "created_utc": utc_now(), "globs": globs,
            "n_files": len(files), "files": files}
    path = MANIFEST_DIR / f"{name}.json"
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"signed {len(files)} files -> {path}")
    return path


def verify(path: Path) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    problems = {"changed": [], "missing": [], "new": []}
    current = collect(data["globs"])
    for rel, meta in data["files"].items():
        if rel not in current:
            problems["missing"].append(rel)
        elif current[rel]["sha256"] != meta["sha256"]:
            problems["changed"].append({"path": rel, "expected": meta["sha256"], "actual": current[rel]["sha256"]})
    for rel in current:
        if rel not in data["files"]:
            problems["new"].append(rel)
    n_problems = sum(len(v) for v in problems.values())
    out = {"manifest": str(path), "verified_utc": utc_now(), "n_files_recorded": data["n_files"],
           "n_files_now": len(current), "problems": problems, "status": "pass" if n_problems == 0 else "fail"}
    out_path = path.with_suffix(".verify.json")
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: len(v) for k, v in problems.items()}))
    print(f"status: {out['status']} -> {out_path}")
    return 0 if n_problems == 0 else 2


def check_run_manifests(runs: Path) -> int:
    good, bad = [], []
    for p in sorted(runs.rglob("*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8-sig"))
        except Exception:  # noqa: BLE001
            continue
        if not isinstance(d, dict) or "run_id" not in d and "stage" not in str(d)[:200]:
            continue
        issues = []
        for section in ("dataset", "code", "config_path"):
            if section in d and d[section] is None:
                issues.append(f"{section} missing")
        for key, val in (d.get("dataset") or {}).items():
            if isinstance(val, dict) and "sha256" in val and "path" in val:
                target = REPO_ROOT / val["path"]
                if not target.is_file():
                    issues.append(f"{key} artifact missing: {val['path']}")
                elif sha256_file(target) != val["sha256"]:
                    issues.append(f"{key} sha256 mismatch")
        (good if not issues else bad).append({"manifest": str(p), "issues": issues})
    out = {"generated_utc": utc_now(), "runs_dir": str(runs), "n_manifests": len(good) + len(bad),
           "valid": len(good), "invalid": len(bad), "problems": bad}
    path = MANIFEST_DIR / "run_manifest_check.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"run manifests: valid={len(good)} invalid={len(bad)} -> {path}")
    return 0 if not bad else 2


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("sign")
    sp.add_argument("--glob", action="append", required=True)
    sp.add_argument("--name", required=True)
    vp = sub.add_parser("verify")
    vp.add_argument("--manifest", required=True)
    rp = sub.add_parser("check-run-manifests")
    rp.add_argument("--runs", default="runs")
    args = ap.parse_args()
    if args.cmd == "sign":
        sign(args.glob, args.name)
        return 0
    if args.cmd == "verify":
        return verify(Path(args.manifest))
    return check_run_manifests(Path(args.runs))


if __name__ == "__main__":
    sys.exit(main())
