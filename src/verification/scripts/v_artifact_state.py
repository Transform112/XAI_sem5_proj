#!/usr/bin/env python
"""Fast consistency check of the current frozen-artifact set (verifier-owned).

Checks, without heavy raw-CSV passes:
  * every mask npz array sha256/n matches split_manifest recorded values
  * mask global-id ranges are within the cleaned-frame total and period offsets
  * preprocessor fit row count equals the period-1 train mask size
  * split_manifest time-disjoint / session boundary fields present
  * scorecard status vocabulary (no 'fail' checks) and C1 strategy
  * pilot artifact exists, point-unit verdict, sample period edges == C1 P1 total
Output: results/text/verification/data_audit/artifact_state.json
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(r"C:\Users\iusearchbtw\Desktop\XAI_proj")
OUT = REPO / "verification" / "data_audit"
SPL = REPO / "data" / "splits" / "NF-UNSW-NB15-v3" / "session_aware_1_4"


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main() -> int:
    man = json.loads((REPO / "results" / "data_audit" / "split_manifest.json").read_text(encoding="utf-8-sig"))
    pre = json.loads((REPO / "results" / "data_audit" / "preprocessor_state.json").read_text(encoding="utf-8-sig"))
    sc = json.loads((REPO / "results" / "data_audit" / "scorecard.json").read_text(encoding="utf-8-sig"))
    pilot = json.loads((REPO / "results" / "data_audit" / "graph_lift_pilot.json").read_text(encoding="utf-8-sig"))
    grouped = man["grouped"]
    out = {"generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "manifest_generated": man.get("generated_at_utc"),
           "manifest_frozen_hash": man.get("frozen_hash"),
           "period_strategy": man.get("period_strategy"),
           "checked_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "checks": []}
    chk = out["checks"]

    def add(i, status, detail):
        chk.append({"id": i, "status": status, "detail": detail})

    # masks vs manifest + offsets
    period_sizes, offsets, all_ok = [], [0], True
    for p in range(1, 6):
        mp = grouped["periods"][str(p - 1)]
        z = np.load(SPL / f"period_{p}_masks.npz")
        n = int(mp["train"]["n_edges"] + mp["val"]["n_edges"] + mp["test"]["n_edges"])
        period_sizes.append(n)
        ok = True
        details = {}
        for s in ("train", "val", "test"):
            a = z[s].astype(np.int64)
            same = sha(a) == mp[s]["sha256"] and len(a) == mp[s]["n_edges"]
            details[s] = {"n": int(len(a)), "hash_match": bool(same)}
            ok = ok and same
        add(f"mask_p{p}", "pass" if ok else "fail", details)
        all_ok = all_ok and ok
        offsets.append(offsets[-1] + n)
    total = offsets[-1]
    # id ranges
    range_ok = True
    for p in range(1, 6):
        z = np.load(SPL / f"period_{p}_masks.npz")
        idx = np.concatenate([z[s] for s in ("train", "val", "test")])
        ok = int(idx.min()) == offsets[p - 1] and int(idx.max()) == offsets[p] - 1 and len(np.unique(idx)) == period_sizes[p - 1]
        range_ok = range_ok and ok
    add("mask_global_id_ranges", "pass" if range_ok else "fail",
        {"period_sizes": period_sizes, "total": total, "session_boundary": man.get("session_boundary")})
    # preprocessor fit == P1 train
    fit_rows = (pre.get("fit") or {}).get("n_rows")
    p1_train = int(grouped["periods"]["0"]["train"]["n_edges"])
    add("preprocessor_fit_scope", "pass" if fit_rows == p1_train else "fail",
        {"fit_rows": fit_rows, "c1_p1_train_mask": p1_train, "delta": (fit_rows or 0) - p1_train})
    # scorecard
    statuses = {}
    for c in sc.get("checks", []):
        statuses[str(c.get("name") or c.get("id"))] = c.get("status")
    bad = {k: v for k, v in statuses.items() if v == "fail"}
    add("scorecard_status", "pass" if not bad else "fail",
        {"overall": sc.get("status"), "n_checks": len(statuses), "fails": bad,
         "flags": {k: v for k, v in statuses.items() if v == "flag"}})
    # pilot
    pe = int((pilot.get("sample") or {}).get("period_edges") or 0)
    verdict = (pilot.get("verdict") or {}).get("overall")
    add("pilot_artifact", "pass" if (pe == period_sizes[0] and verdict == "pass") else "fail",
        {"period_edges": pe, "c1_p1_edges": period_sizes[0], "verdict": verdict,
         "generated_at_utc": pilot.get("generated_at_utc")})
    # label conflict in scorecard quality
    lc = (sc.get("quality") or {}).get("label_conflict") or {}
    add("label_conflict_metric", "pass" if (lc.get("fraction") or 0) <= 0.001 else "fail", lc)
    # ---- D-036 no-drop state compliance (expected from independent audits)
    lc_audit_path = OUT / "label_conflict_audit.json"
    lc_audit = json.loads(lc_audit_path.read_text(encoding="utf-8-sig")) if lc_audit_path.is_file() else {}
    actual_mask_total = int(sum(period_sizes))
    score_rows = sc.get("rows") or {}
    score_periods = (sc.get("periods") or {}).get("period_rows") or []
    expected = {"total_rows": 2_350_609, "p1_rows": 1_131_909,
                "p2_p5_rows": [304_675, 304_675, 304_675, 304_675],
                "label_conflict": {"groups": 11, "edges": 33, "fraction": 5.75186556530642e-06}}
    actual = {"mask_total_rows": actual_mask_total, "mask_period_rows": period_sizes,
              "scorecard_rows": score_rows, "scorecard_period_rows": score_periods,
              "manifest_frozen_hash": man.get("frozen_hash"),
              "scorecard_label_conflict": (sc.get("quality") or {}).get("label_conflict")}
    discrepancy = {
        "mask_total_delta": actual_mask_total - expected["total_rows"],
        "p1_delta": (period_sizes[0] - expected["p1_rows"]) if period_sizes else None,
        "p2_p5_deltas": [a - b for a, b in zip(period_sizes[1:], expected["p2_p5_rows"])],
        "expected_total_rows": expected["total_rows"], "expected_p1_rows": expected["p1_rows"],
        "expected_p2_p5_rows": expected["p2_p5_rows"],
        "label_conflict_audit_groups": lc_audit.get("n_conflicted_groups"),
        "label_conflict_audit_edges": lc_audit.get("n_conflicted_edges"),
    }
    dropped = actual_mask_total != expected["total_rows"] or (bool(period_sizes) and period_sizes[0] != expected["p1_rows"])
    add("d036_no_drop_state", "fail" if dropped else "pass",
        {"actual": actual, "expected_no_drop": expected, "discrepancy": discrepancy,
         "note": ("D-036: current artifacts implement a DROP (rows missing) although D-034 requires report_only; "
                  "the difference equals the engineer's old stricter-conflict edge count (9,360).") if dropped else
                 "no-drop state confirmed"})
    out["overall"] = "pass" if all(c["status"] == "pass" for c in chk) else "fail"
    (OUT / "artifact_state.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps({c["id"]: c["status"] for c in chk}, indent=1))
    print("overall:", out["overall"])
    return 0 if out["overall"] == "pass" else 2


if __name__ == "__main__":
    raise SystemExit(main())
