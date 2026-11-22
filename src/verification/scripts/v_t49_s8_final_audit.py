"""T51/final verifier audit: six-group S8/S9 all-groups output (`seed42-all`).

Reads the combined out-root produced by
`src/scripts/run_analysis.py --mode real --groups seed42-all`, verifies expected full counts
(38,532 real-drift rows; 5,000 model-null; 45,000 repeat; 38,532 rq1 anchor; 4 comparison rows;
partial_groups=false, s8_s9_partial scope), pair coverage, strict s8/s9 validators, and the
validation-root prerequisites. Writes results/text/verification/data_audit/t49_s8_final_audit.json.
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

REPO = Path(r"C:\Users\iusearchbtw\Desktop\XAI_proj")
PY = str(REPO / ".venv" / "Scripts" / "python.exe")
import os
ROOT = Path(os.environ.get("S8_ROOT", str(REPO / "incoming_rtx5060/kaggle_runs/t39-explain-full-649109a/unpacked/s8_combined")))
AUDIT_OUT = REPO / "results/text/verification/data_audit" / os.environ.get("S8_AUDIT_OUT", "t49_s8_final_audit.json")
OUT = REPO / "results/text/verification/data_audit"
GROUPS = 6
NS = {1: 229, 2: 205, 3: 221, 4: 230, 5: 123}
sys.path.insert(0, str(REPO))


def rd(p: Path):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def rows(p: Path):
    return list(csv.DictReader(p.open(encoding="utf-8-sig")))


def find_summary():
    for p in (ROOT / "analysis_summary_real.json", ROOT / "results/analysis_summary_real.json"):
        if p.is_file():
            return rd(p)
    return {}


def main() -> int:
    res = {"checks": []}
    chk = res["checks"]

    def add(i, s, d):
        chk.append({"id": i, "status": s, "detail": d})

    if not ROOT.exists() or not any(ROOT.rglob("*.csv")):
        add("OUTPUT.present", "warn", {"root": str(ROOT), "note": "S8 combined output not yet present"})
        res["overall"] = "pending"
        AUDIT_OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
        print("S8 output not present yet")
        return 2

    summary = find_summary()
    tables = ROOT / "results/text/tables"
    manifest = {}
    for name in ("analysis_wiring_manifest.json", "results/analysis_wiring_manifest.json"):
        if (ROOT / name).is_file():
            manifest = rd(ROOT / name)
            break
    groups = summary.get("groups") or manifest.get("extra", {}).get("groups") or []
    add("RUN.groups",
        "pass" if len(groups) == 6 and summary.get("partial_groups") is False else "fail",
        {"groups": groups, "n_groups": len(groups), "partial_groups": summary.get("partial_groups"),
         "analysis_scope": summary.get("analysis_scope"), "multi_group": summary.get("multi_group"),
         "per_group_real_drift_rows": summary.get("per_group_real_drift_rows")})

    drift_ig = tables / "drift_metrics_ig.csv"
    drift_gnn = tables / "drift_metrics_gnnexplainer.csv"
    counts = {}
    drift_problems = []
    for name, p in (("ig", drift_ig), ("gnn", drift_gnn)):
        if not p.is_file():
            drift_problems.append(f"missing {p.name}")
            continue
        r = rows(p)
        counts[name] = len(r)
        keys = set()
        jt_bad = 0
        for row in r:
            try:
                j = int(row.get("anchor_period", -1)); t = int(row.get("checkpoint_t", -1))
            except ValueError:
                jt_bad += 1; continue
            if t < j:
                jt_bad += 1
            keys.add((row.get("strategy"), row.get("model"), row.get("anchor_id"), j, t))
        if len(keys) != len(r):
            drift_problems.append(f"{name} duplicate/ambiguous pair rows {len(r)} vs {len(keys)}")
        if jt_bad:
            drift_problems.append(f"{name} t<j or bad periods: {jt_bad}")
    add("REAL_DRIFT",
        "pass" if counts == {"ig": 19266, "gnn": 19266} and not drift_problems else "fail",
        {"rows": counts, "total": sum(counts.values()), "expected": {"ig": 19266, "gnn": 19266,
         "total": 38532}, "problems": drift_problems[:10]})

    noise = summary.get("noise_inputs") or {}
    add("NOISE.counts",
        "pass" if (int(noise.get("model_null_frame_rows", 0)) == 5000
                   and int(noise.get("explainer_repeat_frame_rows", 0)) == 45000) else "fail",
        {"noise_inputs": noise, "expected": {"model_null": 5000, "explainer_repeat": 45000}})

    files = {
        "noise_floor": tables / "noise_floor.csv",
        "rq1_anchor_level": tables / "rq1_anchor_level.csv",
        "rq1_strategy_comparison": tables / "rq1_strategy_comparison.csv",
        "rq1_h1_tests": tables / "rq1_h1_tests.csv",
        "cl_metrics": tables / "cl_metrics.csv",
    }
    table_counts = {k: (len(rows(p)) if p.is_file() else None) for k, p in files.items()}
    table_ok = (table_counts["rq1_anchor_level"] == 38532
                and table_counts["rq1_strategy_comparison"] == 4
                and table_counts["cl_metrics"] is not None and table_counts["cl_metrics"] >= 2
                and table_counts["noise_floor"] is not None)
    add("TABLES.counts",
        "pass" if table_ok else "fail",
        {"counts": table_counts,
         "expected": {"rq1_anchor_level": 38532, "rq1_strategy_comparison": 4,
                      "noise_floor": "per (channel, explainer, class)", "cl_metrics": ">=2"}})

    validation_root = {
        "anchors": (ROOT / "results/text/anchors").exists() or (ROOT / "anchors").exists(),
        "explain_index": (ROOT / "results/text/tables/explain_index.csv").is_file(),
        "retention": (ROOT / "results/text/tables/anchor_retention.csv").is_file()
                       or (ROOT / "results/text/tables/retention.csv").is_file(),
        "noise": (ROOT / "results/text/explanations_noise").exists()
                 or (ROOT / "explanations_noise").exists(),
    }
    add("VALIDATION_ROOT", "pass" if all(validation_root.values()) else "warn",
        {"prerequisites": validation_root, "root": str(ROOT),
         "note": "all must be copied into the staged validation root for strict s8/s9"})

    def stage(name):
        p = subprocess.run([PY, "-m", "src.analysis.validators", "--stage", name, "--strict",
                            "--results-tables", str(tables), "--json"],
                           cwd=str(REPO), capture_output=True, text=True)
        try:
            payload = json.loads(p.stdout[p.stdout.find("{"):])
        except Exception:
            payload = {}
        return {"returncode": p.returncode, "overall": payload.get("overall"), "counts": payload.get("counts"),
                "findings": [f for f in (payload.get("findings") or []) if f.get("status") != "pass"][:10]}
    s8, s9 = stage("s8"), stage("s9")
    add("VALIDATORS",
        "pass" if (s8.get("overall") == "pass" and s9.get("overall") == "pass"
                   and s8.get("counts", {}).get("fail", 0) == 0 and s9.get("counts", {}).get("fail", 0) == 0) else "fail",
        {"s8": s8, "s9": s9})

    fails = [c for c in chk if c["status"] == "fail"]
    res["overall"] = "fail" if fails else "pass"
    res["signoff"] = {
        "s8_s9": "SIGNED" if res["overall"] == "pass" else "NOT SIGNED",
        "groups": groups, "partial_groups": summary.get("partial_groups"),
        "analysis_scope": summary.get("analysis_scope"),
        "counts": {"real_drift_rows": sum(counts.values()), "model_null": noise.get("model_null_frame_rows"),
                   "explainer_repeat": noise.get("explainer_repeat_frame_rows"), **table_counts},
        "caveat": "s8_s9_partial scope: combined S8/S9 tables only; S10-S12 not produced",
    }
    AUDIT_OUT.write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({c["id"]: c["status"] for c in chk}, indent=1))
    print("overall:", res["overall"], "S8/S9:", res["signoff"]["s8_s9"])
    for c in chk:
        if c["status"] != "pass":
            print(c["id"], json.dumps(c["detail"], default=str)[:1200])
    return 0 if not fails else 2


if __name__ == "__main__":
    raise SystemExit(main())
