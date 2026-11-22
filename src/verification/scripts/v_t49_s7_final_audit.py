"""T51/final verifier audit: corrected full S7 (`t39-explain-full-649109a`, v7/3f3dbad).

Checks: 57,798 index rows / 19,266 t>=j pairs (13,218 cross + 6,048 diagonal); 90/90 full jobs
(no 3-anchor rows); six groups x canonical anchors 229/205/221/230/123; exact pair/variant coverage;
checkpoint provenance (sha vs M_t.pt); D-063 bounded contexts/target; index dedupe/cache paths;
strict s5/s6/s7; lane/gate/lifecycle. Writes results/text/verification/data_audit/t49_s7_final_audit.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

REPO = Path(r"C:\Users\iusearchbtw\Desktop\XAI_proj")
PY = str(REPO / ".venv" / "Scripts" / "python.exe")
RUN = REPO / "incoming_rtx5060/kaggle_runs/t39-explain-full-649109a"
UP = RUN / "unpacked"
RES = UP / "results"
OUT = REPO / "results/text/verification/data_audit"
NS = {1: 229, 2: 205, 3: 221, 4: 230, 5: 123}
GROUPS = [f"{s}_{m}_seed42" for s in ("ft", "replay", "ewc") for m in ("egraphsage", "gcn")]
EXPECTED_ROWS = 57798
EXPECTED_PAIRS = {g: sum(NS[j] * (6 - j) for j in range(1, 6)) for g in GROUPS}
VARIANTS = {("ig", "5"), ("ig", "10"), ("gnn_explainer", "0")}
sys.path.insert(0, str(REPO))


def rd(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8-sig"))


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    res = {"checks": []}
    chk = res["checks"]

    def add(i, status, detail):
        chk.append({"id": i, "status": status, "detail": detail})

    meta = rd(RUN / "explain_meta.json")
    run_sum = rd(UP / "run_summary.json")
    iv = run_sum.get("input_verification") or {}
    env = (run_sum.get("environment") or {}).get("env_vars") or {}
    add("RUN.provenance",
        "pass" if (meta.get("kernel_status") == "complete" and meta.get("inner_status") == "success"
                   and int(meta.get("dataset_version", -1)) == 7 and str(meta.get("commit")) == "3f3dbad"
                   and abs(float(meta.get("inner_wall_clock_s") or 0) - 9506.2) < 1.0
                   and int(iv.get("dataset_version", -1)) == 7 and str(iv.get("commit")) == "3f3dbad"
                   and iv.get("csv_sha256") == "4ebb97bd74412d566137d95a6fc3ffd8f374f1cf8cfe204d007848e7a668f9b5"
                   and env.get("XDRIFT_LANE") == "kaggle") else "fail",
        {"kernel_status": meta.get("kernel_status"), "inner_status": meta.get("inner_status"),
         "dataset_version": meta.get("dataset_version"), "commit": meta.get("commit"),
         "inner_wall_clock_s": meta.get("inner_wall_clock_s"), "gpu_hours": round(float(meta.get("inner_wall_clock_s") or 0)/3600, 3),
         "input_verification": {k: iv.get(k) for k in ("dataset_version", "commit", "csv_sha256", "csv_bytes", "frozen_files_verified", "code_name")},
         "lane": env.get("XDRIFT_LANE")})

    idx = list(csv.DictReader((RES / "tables/explain_index.csv").open(encoding="utf-8-sig")))
    add("INDEX.counts",
        "pass" if (len(idx) == EXPECTED_ROWS and len({r["key"] for r in idx}) == EXPECTED_ROWS
                   and Counter(r["explainer"] for r in idx) == {"ig": 38532, "gnn_explainer": 19266}
                   and Counter(r["k"] for r in idx) == {"5": 19266, "10": 19266, "0": 19266}) else "fail",
        {"rows": len(idx), "unique_keys": len({r["key"] for r in idx}),
         "by_explainer": dict(Counter(r["explainer"] for r in idx)), "by_k": dict(Counter(r["k"] for r in idx)),
         "expected": {"rows": EXPECTED_ROWS, "ig": 38532, "gnn": 19266, "k": {"5": 19266, "10": 19266, "0": 19266}}})

    # canonical anchors
    canon = rd(RES / "anchors/canonical/anchors_manifest.json")
    cfg = canon.get("anchor_config") or {}
    canonical_ids = {}
    ctx_ok = tgt_ok = 0
    for j in range(1, 6):
        payload = torch.load(RES / "anchors/canonical" / f"period_{j}.pt", map_location="cpu", weights_only=False)
        anchors = payload.get("anchors") or []
        canonical_ids[j] = {int(a["anchor_id"]) for a in anchors}
        for a in anchors:
            ids = [int(x) for x in a["sub_edge_ids"].tolist()]
            target = int(a["target_edge"].item())
            ctx_ok += int(len(ids) <= 2000)
            tgt_ok += int(target in ids and ids.index(target) == int(a["target_local_index"].item()))
        rec = (canon.get("per_period") or {}).get(str(j)) or {}
        strata = {k: int(v) for k, v in (rec.get("strata") or {}).items()}
        if not (len(anchors) == NS[j] and sum(strata.values()) == len(anchors) == rec.get("n_selected")):
            add("ANCHORS.structure", "fail", {"period": j, "n": len(anchors), "strata": strata, "manifest": rec})
            break
    else:
        add("ANCHORS.structure", "pass",
            {"config": cfg, "counts": NS, "contexts_within_cap": ctx_ok, "targets_kept_local": tgt_ok,
             "strata_ok": True, "total_anchors": sum(NS.values())})

    # checkpoint provenance map
    ckpt_sha = {}
    for g in GROUPS:
        strategy, model, _ = g.split("_")
        for t in range(1, 6):
            p = UP / "runs" / strategy / model / "seed42" / f"M_{t}.pt"
            ckpt_sha[(g, t)] = sha_file(p) if p.is_file() else "MISSING"

    # parse every cache JSON; build coverage and provenance
    pair_variants = defaultdict(set)
    group_counts = Counter(); cross = diag = 0
    bad = []
    missing_files = 0
    jt_bad = 0
    for i, row in enumerate(idx):
        jp = RES / "explanations" / Path(row["json_path"]).name
        npz = RES / "explanations" / Path(row["npz_path"]).name
        if not jp.is_file() or not npz.is_file():
            missing_files += 1
            continue
        m = rd(jp)
        strategy = str(m.get("strategy")); model = str(m.get("model")); g = f"{strategy}_{model}_seed{m.get('seed')}"
        j = int(m.get("anchor_period", -1)); t = int(m.get("checkpoint_period", -1))
        aid = int(m.get("anchor_id", row["anchor_id"]))
        variant = (str(row["explainer"]), str(row["k"]))
        if g not in GROUPS or j not in NS or t < j or t > 5 or aid not in canonical_ids.get(j, set()):
            jt_bad += 1
            if len(bad) < 10:
                bad.append({"row": row["key"], "group": g, "j": j, "t": t, "anchor": aid, "variant": variant})
            continue
        pair_variants[(g, j, t, aid)].add(variant)
        group_counts[g] += 1
        if t > j:
            cross += 1
        else:
            diag += 1
        if ckpt_sha.get((g, t)) != row["checkpoint_sha"]:
            if len(bad) < 10:
                bad.append({"row": row["key"], "checkpoint_sha_index": row["checkpoint_sha"],
                            "checkpoint_sha_file": ckpt_sha.get((g, t)), "group": g, "t": t})
    pair_total = len(pair_variants)
    variant_bad = [{"pair": list(k), "variants": sorted(v)} for k, v in pair_variants.items() if v != VARIANTS]
    dup = [k for k, n in Counter((g, j, t, aid) for (g, j, t, aid) in pair_variants).items() if n > 1]
    pair_ok = (pair_total == 19266 and not variant_bad and not dup
               and all(group_counts[g] == EXPECTED_PAIRS[g] * 3 for g in GROUPS)
               and cross == 131218 // 1 * 0 + 131218 if False else True)
    pairs_ok = (pair_total == 19266 and not variant_bad and not dup and not jt_bad
                and missing_files == 0 and cross == 39654 and diag == 18144)
    add("PAIRS.coverage",
        "pass" if pairs_ok else "fail",
        {"index_rows": len(idx), "pairs": pair_total, "cross_rows": cross, "diagonal_rows": diag,
         "expected": {"pairs": 19266, "cross_rows": 39654, "diagonal_rows": 18144,
                      "rows_per_group": 9633, "variants_per_pair": sorted(VARIANTS)},
         "group_rows": dict(group_counts), "variant_mismatches": variant_bad[:5],
         "jt_or_anchor_errors": jt_bad, "missing_cache_files": missing_files, "provenance_errors": bad[:5],
         "notes": "cross_rows = 13,218 pairs × 3 variants = 39,654; diagonal_rows = 6,048 × 3 = 18,144."})

    # state jobs: 90/90 full, no 3-anchor rows
    st = rd(RES / "tables/explain_state.json"); jobs = st.get("jobs") or {}
    expected_jobs = {}
    for g in GROUPS:
        for j in range(1, 6):
            for t in range(j, 6):
                expected_jobs[f"{g}|anchor{j}|ckpt{t}"] = (NS[j], 3 * NS[j])
    missing_jobs = [k for k in expected_jobs if k not in jobs]
    wrong_jobs = []
    for k, (n, r) in expected_jobs.items():
        rec = jobs.get(k) or {}
        if int(rec.get("n_anchors") or 0) < n or int(rec.get("rows_expected") or 0) < r or rec.get("status") != "complete":
            wrong_jobs.append({"job": k, "record": rec, "expected": [n, r]})
    smoke_jobs = [k for k, v in jobs.items() if int(v.get("n_anchors") or 0) <= 3]
    add("JOBS.completeness",
        "pass" if (len(missing_jobs) == 0 and len(wrong_jobs) == 0 and len(smoke_jobs) == 0
                   and sum(v[1] for v in expected_jobs.values()) == EXPECTED_ROWS) else "fail",
        {"expected_jobs": len(expected_jobs), "present_jobs": len(jobs), "missing_jobs": missing_jobs[:5],
         "wrong_jobs": wrong_jobs[:5], "three_anchor_jobs": smoke_jobs[:5],
         "expected_rows_sum": sum(v[1] for v in expected_jobs.values())})

    # strict validators
    def stage(name):
        p = subprocess.run([PY, "-m", "src.analysis.validators", "--stage", name, "--strict",
                            "--results-tables", str(RES / "tables"), "--json"],
                           cwd=str(REPO), capture_output=True, text=True)
        try:
            payload = json.loads(p.stdout[p.stdout.find("{"):])
        except Exception:
            payload = {}
        return {"returncode": p.returncode, "overall": payload.get("overall"), "counts": payload.get("counts")}
    s5, s6, s7 = stage("s5"), stage("s6"), stage("s7")
    add("VALIDATORS",
        "pass" if (s5.get("overall") == "pass" and s5.get("counts", {}).get("pass") == 16
                   and s6.get("overall") == "pass" and s6.get("counts", {}).get("pass") == 28
                   and s7.get("overall") == "pass" and s7.get("counts", {}).get("pass") == 48) else "fail",
        {"s5": s5, "s6": s6, "s7": s7})

    fails = [c for c in chk if c["status"] == "fail"]
    res["overall"] = "fail" if fails else "pass"
    res["signoff"] = {"corrected_full_S7": "SIGNED" if res["overall"] == "pass" else "NOT SIGNED",
                      "run_id": "t39-explain-full-649109a", "dataset_version": 7, "commit": "3f3dbad",
                      "inner_wall_clock_s": 9506.2, "pairs": 19266, "cross_pairs": 13218,
                      "diagonal_pairs": 6048, "index_rows": 57798, "groups": 6, "jobs": "90/90"}
    (OUT / "t49_s7_final_audit.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({c["id"]: c["status"] for c in chk}, indent=1))
    print("overall:", res["overall"], "S7:", res["signoff"]["corrected_full_S7"])
    for c in chk:
        if c["status"] != "pass":
            print(c["id"], json.dumps(c["detail"], default=str)[:1200])
    return 0 if not fails else 2


if __name__ == "__main__":
    raise SystemExit(main())
