#!/usr/bin/env python
"""Reconcile verifier-measured values vs engineer-measured values (verifier-owned).

Both sides are measurements from raw data, not claims: this script does not
"decide" which is right - it checks whether the two independent passes agree. Any
disagreement is surfaced with both files and both values.

Inputs:
  results/text/verification/data_audit/identity_<ds>.json    (verifier measurement)
  results/text/data_audit/identity_<ds>.json          (engineer measurement, if present)
  results/text/verification/reference/published_counts.json   (primary reference = paper Tables 2-3)

Outputs:
  results/text/verification/data_audit/reconciliation_<ds>.json
  results/text/verification/data_audit/reconciliation_summary.md
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

VERIFIER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VERIFIER_ROOT.parents[1]
MY_DIR = VERIFIER_ROOT / "data_audit"
ENG_DIR = REPO_ROOT / "results" / "data_audit"
REF = VERIFIER_ROOT / "reference" / "published_counts.json"
OUT = VERIFIER_ROOT / "data_audit"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load(p: Path):
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8-sig"))
    except Exception as exc:  # noqa: BLE001
        return {"__error__": str(exc)}


def parse_inf_columns(mine: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    tops = (mine.get("missingness") or [{}])[0].get("per_column_top", {})
    for key, cnt in tops.items():
        m = re.match(r"^\d+:(?P<col>[^:]+):(?:neg_)?inf_token$", key) or re.match(r"^(?P<col>.+):(?:neg_)?inf$", key)
        if m:
            out[m.group("col")] = out.get(m.group("col"), 0) + int(cnt)
    return out


def parse_empty_columns(mine: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    tops = (mine.get("missingness") or [{}])[0].get("per_column_top", {})
    for key, cnt in tops.items():
        m = re.match(r"^\d+:(?P<col>[^:]+):empty$", key) or re.match(r"^(?P<col>.+):nan_or_empty$", key)
        if m:
            out[m.group("col")] = out.get(m.group("col"), 0) + int(cnt)
    return out


def reconcile(ds: str, ref: dict) -> dict:
    mine = load(MY_DIR / f"identity_{ds}.json")
    eng = load(ENG_DIR / f"identity_{ds}.json")
    out: dict = {"dataset": ds, "generated_utc": utc_now(),
                 "verifier_artifact": str(MY_DIR / f"identity_{ds}.json"),
                 "engineer_artifact": str(ENG_DIR / f"identity_{ds}.json"),
                 "engineer_artifact_present": eng is not None,
                 "comparisons": [], "disagreements": [], "engineer_reference_status": None}
    if mine is None:
        out["status"] = "blocked"
        out["detail"] = "verifier identity artifact missing"
        return out
    if eng is None:
        out["status"] = "pending"
        out["detail"] = "engineer identity artifact not produced yet"
        return out

    def cmp(name: str, mine_val, eng_val, tol: float = 0) -> None:
        ok = False
        if isinstance(mine_val, (int, float)) and isinstance(eng_val, (int, float)):
            ok = abs(mine_val - eng_val) <= tol
        else:
            ok = mine_val == eng_val
        out["comparisons"].append({"item": name, "verifier": mine_val, "engineer": eng_val, "agree": ok})
        if not ok:
            out["disagreements"].append({"item": name, "verifier": mine_val, "engineer": eng_val})

    scans = (mine.get("scans") or [{}])[0]
    mine_labels = (mine.get("label_counts_merged") or {}).get("Attack") or {}
    mine_binary = (mine.get("label_counts_merged") or {}).get("Label") or {}
    mine_inf = parse_inf_columns(mine)
    mine_empty = parse_empty_columns(mine)
    eng_labels = ((eng.get("labels") or {}).get("measured") or {})
    eng_binary = ((eng.get("binary_label") or {}).get("measured") or {})
    eng_inf = ((eng.get("inf") or {}).get("columns") or {})
    eng_missing = ((eng.get("missing") or {}).get("columns") or {})

    cmp("rows_measured", mine.get("total_data_rows"), (eng.get("rows") or {}).get("measured"))
    my_sha = None
    if isinstance(mine.get("sha256_csv"), list) and mine["sha256_csv"]:
        my_sha = mine["sha256_csv"][0].get("sha256")
    cmp("csv_sha256", my_sha, (eng.get("files") or [{}])[0].get("sha256"))
    my_dup = ((mine.get("duplicates") or {}).get("full_row") or {}).get("duplicate_row_count")
    if my_dup is None:
        out["comparisons"].append({"item": "duplicate_rows_exact", "verifier": "deferred (light scan)",
                                   "engineer": (eng.get("duplicates") or {}).get("exact"),
                                   "agree": None, "note": "engineer-reported duplicate count not independently recounted in light mode"})
    else:
        cmp("duplicate_rows_exact", my_dup, (eng.get("duplicates") or {}).get("exact"))
    cmp("hosts_union_unique", (scans.get("host_stats") or {}).get("unique_hosts_union"),
        (eng.get("hosts") or {}).get("union_unique"))
    cmp("class_counts", mine_labels, eng_labels)
    cmp("binary_label_counts", mine_binary, eng_binary)
    cmp("inf_counts_per_column", mine_inf, eng_inf)
    cmp("empty_counts_per_column", mine_empty, eng_missing)
    my_ts = scans.get("timestamp") or {}
    eng_ts = eng.get("timestamps") or {}
    cmp("timestamp_parse_non_numeric", my_ts.get("non_numeric"), 0 if (eng_ts.get("parse_fraction") == 1.0) else None)
    cmp("timestamps_monotonic", (my_ts.get("decreasing_steps", 0) == 0), eng_ts.get("monotonic"))

    # engineer's published reference vs the paper reference
    pub = (ref["datasets"].get(ds) or {})
    eng_pub = ((eng.get("labels") or {}).get("published_classes") or {})
    eng_pub_clean = {k: v for k, v in eng_pub.items() if v is not None}
    paper_clean = dict(pub.get("classes") or {})
    if eng_pub_clean:
        mismatch = {k: {"engineer_published": eng_pub_clean.get(k), "paper": paper_clean.get(k)}
                    for k in set(eng_pub_clean) | set(paper_clean)
                    if eng_pub_clean.get(k) != paper_clean.get(k)}
        out["engineer_reference_status"] = {
            "engineer_published_classes": eng_pub_clean,
            "paper_classes": paper_clean,
            "fields_differing_from_paper": mismatch,
            "note": "engineer must re-run identity after D-013 (paper Tables 2-3 are authoritative; UQ website has errata)",
        }
    eng_status = eng.get("status")
    out["engineer_status"] = eng_status
    out["verifier_status"] = mine.get("overall")
    out["status"] = "agree" if not out["disagreements"] else "disagree"
    out["deferred_items"] = [x["item"] for x in out["comparisons"] if x.get("agree") is None]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all")
    args = ap.parse_args()
    ref = json.loads(REF.read_text(encoding="utf-8-sig"))
    datasets = [d for d in ref["datasets"] if ref["datasets"][d].get("paper_total_flows") is not None] \
        if args.dataset == "all" else [args.dataset]
    results = []
    for ds in datasets:
        r = reconcile(ds, ref)
        results.append(r)
        (OUT / f"reconciliation_{ds}.json").write_text(json.dumps(r, indent=2, default=str), encoding="utf-8")
    lines = ["# Verifier vs engineer identity reconciliation", "", f"Generated: {utc_now()}", "",
             "| dataset | status | engineer artifact | disagreements | engineer status |",
             "| --- | --- | --- | --- | --- |"]
    for r in results:
        lines.append(f"| {r['dataset']} | {r['status']} | {r.get('engineer_artifact_present')} | "
                     f"{len(r.get('disagreements', []))} | {r.get('engineer_status')} |")
    detail = []
    for r in results:
        detail.append(f"\n## {r['dataset']}")
        detail.append(f"- verifier: {r.get('verifier_status')} / engineer: {r.get('engineer_status')} -> {r['status']}")
        for c in r.get("comparisons", []):
            detail.append(f"- {c['item']}: {'AGREE' if c['agree'] else 'DISAGREE'} (verifier={c['verifier']}, engineer={c['engineer']})")
        if r.get("engineer_reference_status"):
            rs = r["engineer_reference_status"]
            detail.append(f"- engineer reference differs from paper on: {list(rs['fields_differing_from_paper'])}")
    (OUT / "reconciliation_summary.md").write_text("\n".join(lines + detail) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())

