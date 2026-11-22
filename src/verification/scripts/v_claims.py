#!/usr/bin/env python
"""Claim -> evidence matrix runner (verifier-owned; reusable at G4/G5/G6).

Purpose: make every quantitative claim traceable to (a) an artifact file, (b) a
command, (c) a recorded value. The analyst/Lead provide a claims file; this script
verifies the evidence exists, hashes it, and (where the artifact is machine
readable) recomputes/compares the stated value.

Claims file format (JSON list; a template can be generated with --build-template):
[
  {
    "id": "C-001",
    "claim": "Plain fine-tuning shows higher drift than replay on period-1 anchors",
    "value": 0.42,
    "artifact": "results/text/tables/drift_rq1.csv",
    "selector": {"column": "drift_mean", "filter": {"strategy": "FT", "period": 1}},
    "command": ".venv\\Scripts\\python.exe src/scripts/run_stage8.py ...",
    "gate": "G5",
    "status": "unchecked"
  }
]

Usage:
  python verification\\scripts\\v_claims.py --build-template results/text/tables
  python verification\\scripts\\v_claims.py --claims verification/claims/claims.json
  python verification\\scripts\\v_claims.py --claims <file> --out verification/claims
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_rows(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with open(path, newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    if path.suffix.lower() in (".json",):
        d = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(d, list):
            return d
        if isinstance(d, dict):
            return [d]
    return []


def matches(row: dict, flt: dict) -> bool:
    for k, v in flt.items():
        if str(row.get(k)) != str(v):
            return False
    return True


def resolve_value(rows: list[dict], selector: dict):
    col = selector.get("column")
    flt = selector.get("filter") or {}
    vals = [r for r in rows if matches(r, flt)]
    if not vals:
        return None, "no rows matched selector"
    if col is None:
        return vals, None
    out = []
    for r in vals:
        try:
            out.append(float(r[col]))
        except (KeyError, TypeError, ValueError):
            return None, f"column {col} not numeric in matched rows"
    return (out[0] if len(out) == 1 else out), None


def check_claim(c: dict) -> dict:
    res = dict(c)
    art = (REPO_ROOT / c["artifact"]).resolve() if c.get("artifact") else None
    res["artifact_abs"] = str(art) if art else None
    res["evaluated_utc"] = utc_now()
    res["status"] = "UNSUPPORTED"
    res["notes"] = []
    if art is None or not art.is_file():
        res["notes"].append("artifact missing")
        return res
    res["artifact_sha256"] = sha256_file(art)
    res["artifact_bytes"] = art.stat().st_size
    try:
        rows = read_rows(art)
    except Exception as exc:  # noqa: BLE001
        res["notes"].append(f"artifact unreadable: {exc}")
        return res
    if not rows:
        res["notes"].append("artifact has no machine-readable rows")
        res["status"] = "ARTIFACT_PRESENT_UNVERIFIED"
        return res
    got, err = resolve_value(rows, c.get("selector") or {})
    if err:
        res["notes"].append(err)
        res["status"] = "ARTIFACT_PRESENT_UNVERIFIED"
        return res
    res["recomputed_value"] = got
    stated = c.get("value")
    if stated is None:
        res["status"] = "ARTIFACT_PRESENT_UNVERIFIED"
        return res
    try:
        ok = abs(float(got) - float(stated)) <= float(c.get("tolerance", 1e-9))
    except (TypeError, ValueError):
        ok = False
    res["status"] = "PASS" if ok else "MISMATCH"
    if not ok:
        res["notes"].append(f"stated {stated} != recomputed {got}")
    if not c.get("command"):
        res["notes"].append("no command recorded")
    return res


def build_template(results_dir: Path) -> list[dict]:
    claims = []
    n = 0
    for p in sorted(results_dir.rglob("*")):
        if p.suffix.lower() not in (".csv", ".json") or not p.is_file():
            continue
        n += 1
        claims.append({"id": f"C-{n:03d}", "claim": f"TODO describe claim from {p.relative_to(REPO_ROOT)}",
                       "value": None, "artifact": str(p.relative_to(REPO_ROOT)),
                       "selector": {"column": "TODO", "filter": {}},
                       "command": "TODO exact command", "gate": "TODO", "status": "unchecked"})
    return claims


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--claims")
    ap.add_argument("--build-template")
    ap.add_argument("--out", default="verification/claims")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.build_template:
        claims = build_template(Path(args.build_template))
        path = out / "claims_template.json"
        path.write_text(json.dumps(claims, indent=2), encoding="utf-8")
        print(f"wrote {path} ({len(claims)} candidate artifacts; fill claim/value/selector)")
        return 0
    if not args.claims:
        print("need --claims or --build-template")
        return 1
    claims = json.loads(Path(args.claims).read_text(encoding="utf-8-sig"))
    checked = [check_claim(c) for c in claims]
    summary = {"generated_utc": utc_now(), "claims_file": args.claims,
               "n": len(checked),
               "status_counts": {s: sum(1 for c in checked if c["status"] == s)
                                 for s in ("PASS", "MISMATCH", "ARTIFACT_PRESENT_UNVERIFIED", "UNSUPPORTED")},
               "claims": checked}
    path = out / "claim_evidence_matrix.json"
    path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    # markdown matrix
    lines = ["# Claim -> evidence matrix", "", f"Generated: {summary['generated_utc']}", "",
             "| id | gate | claim | stated | recomputed | artifact (sha256) | status |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for c in checked:
        lines.append(f"| {c.get('id')} | {c.get('gate')} | {str(c.get('claim'))[:110]} | {c.get('value')} | "
                     f"{c.get('recomputed_value')} | {c.get('artifact')} ({str(c.get('artifact_sha256'))[:12]}) | {c['status']} |")
    (out / "claim_evidence_matrix.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["status_counts"]))
    print(f"wrote {path}")
    return 0 if summary["status_counts"]["MISMATCH"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
