"""T47 verifier spot-audit of the paper S5 section: every rendered number must match a
committed artifact, the six wording caveats + noise F4 status must be present, and no
'pending' audit wording may remain. Read-only; writes
results/text/verification/data_audit/t47_paper_stats_audit.json.
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

REPO = Path(r"C:\Users\iusearchbtw\Desktop\XAI_proj")
PAPER = REPO / "paper_writing/manuscript/04_results_s5.md"
HEAD = REPO / "incoming_rtx5060/kaggle_runs/t42_headline/headline_table_valid.csv"
T46 = REPO / "results/text/tables/s5_baseline"
OUT = REPO / "results/text/verification/data_audit"


def rd(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def csv_rows(p):
    return list(csv.DictReader(p.open(encoding="utf-8-sig")))


def norm(text):
    return str(text).replace("\u2212", "-").replace("\u2013", "-").replace("\u00b1", " ")


def nums(text):
    return [float(x) for x in re.findall(r"[-+]?(?:\d+\.\d+|\d+e[-+]?\d+)", norm(text))]


def main() -> int:
    text = PAPER.read_text(encoding="utf-8")
    lines = text.splitlines()
    head = {(r["strategy"], r["model"], r["metric"]): r for r in csv_rows(HEAD)}
    comp = {(r["model"], r["metric"], r["comparison"]): r
            for r in csv_rows(T46 / "baseline_comparisons.csv")}
    finalrow = {(r["model"], r["strategy"], r["matrix"], int(r["period"])): r
                for r in csv_rows(T46 / "per_period_finalrow.csv")}
    perclass = csv_rows(T46 / "per_class_f1.csv")
    lam = {(r["model"], r["metric"]): r for r in csv_rows(T46 / "ewc_lambda_sensitivity.csv")}
    checks = []
    problems = []

    def check(cid, ok, detail):
        checks.append({"id": cid, "status": "pass" if ok else "fail", "detail": detail})
        if not ok:
            problems.append(cid)

    # Table 5.1 (6 rows x 8 mean±sd = 16 numbers each)
    t51 = [ln for ln in lines if re.match(r"\|\s*(FT|Replay \(corrected\)|EWC-λ100)\s*\|", ln)]
    t51_problems = []
    mapping = {"FT": "ft", "Replay (corrected)": "replay", "EWC-λ100": "ewc"}
    for ln in t51:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        strat = mapping[cells[0]]
        model = "egraphsage" if "E-GS" in cells[1] or "E-GraphSAGE" in cells[1] else "gcn"
        vals = []
        for cell in cells[2:8]:
            m = re.findall(r"[-+]?\d+\.\d+", norm(cell))
            vals.extend(float(x) for x in m)
        if len(vals) != 12:
            t51_problems.append({"row": ln, "n_numbers": len(vals)})
            continue
        row_all = head[(strat, model, "all_classes")]
        row_sup = head[(strat, model, "supported_set")]
        expected = [float(row_all["final_mean"]), float(row_all["final_sd"]),
                    float(row_sup["final_mean"]), float(row_sup["final_sd"]),
                    float(row_all["forgetting_mean"]), float(row_all["forgetting_sd"]),
                    float(row_sup["forgetting_mean"]), float(row_sup["forgetting_sd"]),
                    float(row_all["bwt_mean"]), float(row_all["bwt_sd"]),
                    float(row_sup["bwt_mean"]), float(row_sup["bwt_sd"])]
        for got, exp in zip(vals, expected):
            if abs(got - exp) > 6e-5:
                t51_problems.append({"row": ln, "got": got, "expected": exp})
    check("TABLE5.1", not t51_problems, {"rows": len(t51), "problems": t51_problems[:10],
                                        "note": "12 values per row = final/forgetting/BWT mean±sd for all and supported; "
                                                "paper rounds to 4 dp, compared at 6e-5"})

    # Table 5.2 (6 rows x 10 numbers)
    t52 = [ln for ln in lines if re.match(r"\|\s*(E-GS|GCN)\s*\|", ln)]
    t52_problems = []
    for ln in t52:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        model = "egraphsage" if cells[0] == "E-GS" else "gcn"
        label = cells[1].replace("−", "-").replace(" ", "")
        a, b = label.lower().split("-")
        comp_label = f"{a}_minus_{b}"
        vals = nums(" ".join(cells[2:8]))
        if len(vals) != 10:
            t52_problems.append({"row": ln, "n_numbers": len(vals), "expected": 10})
            continue
        all_row = comp[(model, "final_all", comp_label)]
        sup_row = comp[(model, "final_supported", comp_label)]
        expected = [float(all_row["diff"]), float(all_row["ci_lo"]), float(all_row["ci_hi"]),
                    float(all_row["cohens_d"]), float(all_row["p_holm"]),
                    float(sup_row["diff"]), float(sup_row["ci_lo"]), float(sup_row["ci_hi"]),
                    float(sup_row["cohens_d"]), float(sup_row["p_holm"])]
        for idx, (got, exp) in enumerate(zip(vals, expected)):
            tol = 6e-3 if idx in (3, 8) else max(6e-4, abs(exp) * 1e-3)
            if abs(got - exp) > tol:
                t52_problems.append({"row": ln, "idx": idx, "got": got, "expected": exp, "comparison": comp_label})
    check("TABLE5.2", not t52_problems, {"rows": len(t52), "problems": t52_problems[:10],
                                        "note": "each row = Δ/CI/d/p for final all and final supported; paper rounds to "
                                                "4 dp (d 2 dp, p 3 dp), compared with a relative/absolute 6e-4 tolerance"})

    # forgetting paragraph values
    fp = next((ln for ln in lines if ln.startswith("**Forgetting/BWT")), "")
    fp_nums = nums(fp)
    expect_forget = [
        float(comp[("gcn", "forgetting_all", "ewc_minus_ft")]["diff"]),
        float(comp[("gcn", "forgetting_all", "ewc_minus_ft")]["ci_lo"]),
        float(comp[("gcn", "forgetting_all", "ewc_minus_ft")]["ci_hi"]),
        float(comp[("gcn", "forgetting_all", "ewc_minus_ft")]["cohens_d"]),
        float(comp[("gcn", "forgetting_all", "ewc_minus_ft")]["p_holm"]),
        float(comp[("gcn", "forgetting_supported", "ewc_minus_ft")]["diff"]),
        float(comp[("gcn", "forgetting_supported", "ewc_minus_ft")]["ci_lo"]),
        float(comp[("gcn", "forgetting_supported", "ewc_minus_ft")]["ci_hi"]),
        float(comp[("gcn", "forgetting_supported", "ewc_minus_ft")]["p_holm"]),
        float(comp[("gcn", "forgetting_supported", "ewc_minus_replay")]["diff"]),
        float(comp[("gcn", "forgetting_supported", "ewc_minus_replay")]["ci_lo"]),
        float(comp[("gcn", "forgetting_supported", "ewc_minus_replay")]["ci_hi"]),
        float(comp[("gcn", "forgetting_supported", "ewc_minus_replay")]["p_holm"]),
        float(comp[("egraphsage", "forgetting_all", "replay_minus_ft")]["diff"]),
        float(comp[("egraphsage", "forgetting_all", "replay_minus_ft")]["ci_lo"]),
        float(comp[("egraphsage", "forgetting_all", "replay_minus_ft")]["ci_hi"]),
        float(comp[("egraphsage", "forgetting_all", "ewc_minus_ft")]["diff"]),
        float(comp[("egraphsage", "forgetting_all", "ewc_minus_ft")]["ci_lo"]),
        float(comp[("egraphsage", "forgetting_all", "ewc_minus_ft")]["ci_hi"]),
        float(comp[("gcn", "forgetting_all", "replay_minus_ft")]["diff"]),
        float(comp[("gcn", "forgetting_all", "replay_minus_ft")]["ci_lo"]),
        float(comp[("gcn", "forgetting_all", "replay_minus_ft")]["ci_hi"]),
        float(comp[("gcn", "forgetting_all", "replay_minus_ft")]["p_holm"]),
    ]
    fp_numbers = nums(fp)
    forget_problems = []
    for e in expect_forget:
        tol = 5e-3 if (abs(e) > 0.2 and abs(e) < 10) else max(6e-4, abs(e) * 1e-3)
        if not any(abs(g - e) <= tol for g in fp_numbers):
            forget_problems.append({"expected": e})
    check("FORGETTING_PARAGRAPH", not forget_problems, {"n_numbers": len(fp_numbers),
                                                        "expected_numbers": len(expect_forget),
                                                        "missing": forget_problems[:10],
                                                        "note": "each expected Δ/CI/d/Holm value must appear in the "
                                                                "paragraph within 6e-4/relative tolerance (order-insensitive)"})

    # per-period table 30 values
    pp = [ln for ln in lines if re.match(r"\|\s*(E-GS|GCN)\s+(FT|replay|EWC)\s*\|", ln)]
    pp_problems = []
    for ln in pp:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        first = cells[0]
        model = "egraphsage" if first.startswith("E-GS") else "gcn"
        strat_label = first.split()[-1]
        strategy = {"FT": "ft", "replay": "replay", "EWC": "ewc"}[strat_label]
        vals = [float(x) for x in cells[1:6]]
        for period, got in zip(range(1, 6), vals):
            exp = float(finalrow[(model, strategy, "all_classes", period)]["mean"])
            if abs(got - exp) > 6e-5:
                pp_problems.append({"model": model, "strategy": strategy, "period": period, "got": got, "expected": exp})
    check("PER_PERIOD_TABLE", not pp_problems, {"rows": len(pp), "values": 30, "problems": pp_problems[:10]})

    # rare-stratum values at period 5
    rare = [r for r in perclass if r["class"] == "rare_stratum_pooled" and r["evaluated_period"] == "5"]
    rare_expect = {(r["model"], r["strategy"]): round(float(r["mean_f1"]), 3) for r in rare}
    rare_text = next((ln for ln in lines if "pooled rare stratum" in ln), "")
    rare_nums = [x for x in nums(rare_text) if abs(x) < 1.0]
    exp_rare = [rare_expect[("egraphsage", s)] for s in ("ft", "replay", "ewc")] + \
               [rare_expect[("gcn", s)] for s in ("ft", "replay", "ewc")]
    rare_ok = len(rare_nums) >= 6 and all(abs(a - b) <= 1e-3 for a, b in zip(rare_nums[:6], exp_rare))
    check("RARE_STRATUM", rare_ok, {"got": rare_nums[:6], "expected": exp_rare})

    # lambda values
    lam_text = next((ln for ln in lines if ln.startswith("**λ sensitivity")), "")
    lam_nums = nums(lam_text)
    exp_lam = [float(lam[("egraphsage", "final_all")]["diff_100_minus_1000"]),
               float(lam[("egraphsage", "final_all")]["ci_lo"]), float(lam[("egraphsage", "final_all")]["ci_hi"]),
               float(lam[("egraphsage", "forgetting_all")]["diff_100_minus_1000"]),
               float(lam[("egraphsage", "forgetting_all")]["ci_lo"]), float(lam[("egraphsage", "forgetting_all")]["ci_hi"]),
               float(lam[("gcn", "final_all")]["diff_100_minus_1000"]),
               float(lam[("gcn", "final_all")]["ci_lo"]), float(lam[("gcn", "final_all")]["ci_hi"]),
               float(lam[("gcn", "forgetting_all")]["diff_100_minus_1000"]),
               float(lam[("gcn", "forgetting_all")]["ci_lo"]), float(lam[("gcn", "forgetting_all")]["ci_hi"])]
    lam_numbers = nums(lam_text)
    lam_problems = []
    for e in exp_lam:
        if not any(abs(g - e) <= max(6e-4, abs(e) * 1e-3) for g in lam_numbers):
            lam_problems.append({"expected": e})
    check("LAMBDA_TEXT", not lam_problems, {"n_numbers": len(lam_numbers), "missing": lam_problems[:10],
                                            "note": "expected λ sensitivity values must appear within tolerance"})

    # caveats / status wording
    required = {
        "status_verifier_audited": "VERIFIER-AUDITED" in text,
        "no_pending_audit_wording": not re.search(r"stats audit (is )?pending|verifier stats audit pending", text, re.I),
        "snapshot_caveat": "snapshot-provenance caveat" in text.lower() or "snapshot-provenance caveat" in text,
        "snapshot_runs": all(x in text for x in ("90e7635", "9ca4681", "649109a")),
        "holm_qualifier": "Holm within" in text and "3-comparison family" in text,
        "non_equivalence": "non-significance is not proof of equivalence" in text.lower(),
        "no_population_claim": "no population-level or causal claim" in text.lower(),
        "rare_support_limits": "support-limited" in text or "tiny test support" in text,
        "lambda1000_sensitivity": "labelled sensitivity" in text or "sensitivity (`ewc_lambda_sensitivity.csv`)" in text,
        "baseline_context": "baseline context" in text,
        "noise_f4_not_signed": "F4 smoke NOT signed" in text,
        "noise_sampled_reason": "batching=sampled" in text and "graph_batch" in text,
        "noise_fix_rerun": "fix + smoke re-run required" in text,
        "s7_pending": "Corrected cross-checkpoint S7 explanations" in text and "PENDING" in text,
    }
    check("WORDING_CAVEATS", all(required.values()), {"checks": required,
                                                      "missing": [k for k, v in required.items() if not v],
                                                      "note": "six verifier caveats, snapshot provenance, noise-F4 status and "
                                                              "S7-pending wording present; no stale 'pending audit' wording."})

    # headline cross-check of the table values against the signed CSV is covered by TABLE5.1.
    res = {"schema_version": 1, "kind": "t47_paper_stats_audit", "paper": str(PAPER.relative_to(REPO)),
           "checks": checks, "problems": problems, "overall": "fail" if problems else "pass",
           "note": "independent read-only spot audit of the paper S5 section numbers and caveats; "
                   "paper files are scout-owned and only checked, never modified."}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t47_paper_stats_audit.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({c["id"]: c["status"] for c in checks}, indent=1))
    print("overall:", res["overall"])
    for c in checks:
        if c["status"] != "pass":
            print(c["id"], json.dumps(c["detail"], default=str)[:1200])
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
