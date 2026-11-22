#!/usr/bin/env python
"""T25 — pin a commit tree, record hashes, run the suite from the pinned tree.

Usage:
  python verification/src/scripts/v_t25_pin.py [<commit>]

Writes results/text/verification/data_audit/t25_pin.json and extracts the tree to
verification/audit_<sha7>/tree (verification scope; no repo files touched).
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

REPO = Path(r"C:\Users\iusearchbtw\Desktop\XAI_proj")
GIT = r"C:\Program Files\Git\bin\git.exe"
OUT = REPO / "verification" / "data_audit"
PY = str(REPO / ".venv" / "Scripts" / "python.exe")


def main() -> int:
    commit = sys.argv[1] if len(sys.argv) > 1 else "HEAD"
    rc, sha = subprocess.run([GIT, "-C", str(REPO), "rev-parse", commit], capture_output=True, text=True).returncode, \
        subprocess.run([GIT, "-C", str(REPO), "rev-parse", commit], capture_output=True, text=True).stdout.strip()
    tree = subprocess.run([GIT, "-C", str(REPO), "rev-parse", f"{sha}^{{tree}}"], capture_output=True, text=True).stdout.strip()
    subtrees = {}
    for sub in ("src", "scripts", "configs", "tests", "analysis"):
        p = subprocess.run([GIT, "-C", str(REPO), "rev-parse", f"{sha}:{sub}"], capture_output=True, text=True)
        subtrees[sub] = p.stdout.strip() if p.returncode == 0 else None
    short = sha[:7]
    dest = REPO / "verification" / f"audit_{short}"
    dest.mkdir(parents=True, exist_ok=True)
    zpath = dest / f"audit_{short}.zip"
    subprocess.run([GIT, "-C", str(REPO), "archive", "--format=zip", "-o", str(zpath), sha], check=True)
    tree_dir = dest / "tree"
    with zipfile.ZipFile(zpath) as zf:
        zf.extractall(tree_dir)
    zip_sha = hashlib.sha256(zpath.read_bytes()).hexdigest()
    result = {"commit": sha, "tree": tree, "subtrees": subtrees, "short": short,
              "zip": str(zpath), "zip_sha256": zip_sha, "tree_dir": str(tree_dir),
              "suite": None}
    # suite from the pinned tree (imports resolve inside the extracted tree)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree_dir)
    proc = subprocess.run([PY, "-m", "pytest", "tests", "analysis", "-q", "--no-header"],
                          cwd=str(tree_dir), capture_output=True, text=True, timeout=1800, env=env)
    tail = [line for line in proc.stdout.strip().splitlines() if "passed" in line or "failed" in line or "error" in line]
    result["suite"] = {"returncode": proc.returncode, "tail": tail[-2:] if tail else proc.stdout.strip().splitlines()[-3:]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t25_pin.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str))
    return 0 if proc.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
