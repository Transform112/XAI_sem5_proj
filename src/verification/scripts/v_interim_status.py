"""Interim presentation-value certification: recompute the presentation-safe S5/stats values
from the committed CSVs, cross-check them against the signed paper table, verify that the
T46/T47 verifier audits are overall pass, and emit
results/text/verification/data_audit/interim_presentation_values.json for the 2026-11-03 presentation.

Read-only over signed artifacts; writes only the certification JSON.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

REPO = Path(r"C:\Users\iusearchbtw\Desktop\XAI_proj")
HEAD = REPO / "incoming_rtx5060/kaggle_runs/t42_headline/headline_table_valid.csv"
T46 = REPO / "results/text/tables/s5_baseline"
OUT = REPO / "results/text/verification/data_audit/interim_presentation_values.json"


def rd(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def csv_rows(p):
    return list(csv.DictReader(p.open(encoding="utf-8-sig")))


def main() -> int:
    head = csv_rows(HEAD)
    comp = csv_rows(T46 / "baseline_comparisons.csv")
    means = csv_rows(T46 / "baseline_means.csv")
    t42_rerun = rd(REPO / "results/text/verification/data_audit/t42_rerun_audit.json")
    t42_verify = rd(REPO / "results/text/verification/data_audit/t42_headline_verify.json")
    audits = {
        "t46_stats": rd(REPO / "results/text/verification/data_audit/t46_stats_audit.json"),
        "t47_paper": rd(REPO / "results/text/verification/data_audit/t47_paper_stats_audit.json"),
        "t42_rerun": t42_rerun,
        "t42_headline_signoff": t42_verify.get("signoff", {}),
    }
    assert t42_rerun.get("overall") == "pass", t42_rerun.get("overall")
    assert t42_verify.get("signoff", {}).get("corrected_headline_table", "").startswith("PASS"), t42_verify.get("signoff")
    assert all(a.get("overall") == "pass" for a in audits.values() if isinstance(a, dict) and "overall" in a), \
        {k: a.get("overall") for k, a in audits.items()}

    s5_rows = []
    for row in head:
        s5_rows.append({
            "strategy": row["strategy"], "model": row["model"], "metric": row["metric"],
            "n_seeds": int(row["n_seeds"]), "valid": row["valid"],
            "final_mean": round(float(row["final_mean"]), 4), "final_sd": round(float(row["final_sd"]), 4),
            "forgetting_mean": round(float(row["forgetting_mean"]), 4),
            "forgetting_sd": round(float(row["forgetting_sd"]), 4),
            "bwt_mean": round(float(row["bwt_mean"]), 4), "bwt_sd": round(float(row["bwt_sd"]), 4),
            "source_run": row["source_run"],
        })
    stats_rows = []
    for row in comp:
        stats_rows.append({
            "model": row["model"], "metric": row["metric"], "comparison": row["comparison"],
            "diff": round(float(row["diff"]), 4), "ci_lo": round(float(row["ci_lo"]), 4),
            "ci_hi": round(float(row["ci_hi"]), 4), "cohens_d": round(float(row["cohens_d"]), 2),
            "rank_biserial": round(float(row["rank_biserial"]), 2),
            "p_raw": round(float(row["p_raw"]), 3), "p_holm": round(float(row["p_holm"]), 3),
            "family": row["family"], "n": int(row["n"]), "n_resamples": int(row["n_resamples"]),
        })
    # presentation-safe quick facts (rounded from the signed CSVs; no new numbers)
    def get(model, metric, comparison):
        return next(r for r in stats_rows if r["model"] == model and r["metric"] == metric and r["comparison"] == comparison)
    facts = {
        "replay_worse_than_ft_final_egs_all": get("egraphsage", "final_all", "replay_minus_ft"),
        "ewc_vs_ft_final_egs_all": get("egraphsage", "final_all", "ewc_minus_ft"),
        "ewc_vs_ft_forgetting_gcn_all": get("gcn", "forgetting_all", "ewc_minus_ft"),
        "ewc_vs_ft_forgetting_gcn_supported": get("gcn", "forgetting_supported", "ewc_minus_ft"),
        "ewc_vs_replay_final_egs_all": get("egraphsage", "final_all", "ewc_minus_replay"),
        "ewc_vs_replay_final_gcn_all": get("gcn", "final_all", "ewc_minus_replay"),
        "replay_vs_ft_forgetting_egs_all": get("egraphsage", "forgetting_all", "replay_minus_ft"),
        "replay_vs_ft_forgetting_gcn_all": get("gcn", "forgetting_all", "replay_minus_ft"),
    }
    # consistency: headline means match the T46 means rows exactly (unrounded)
    mean_by = {(r["strategy"], r["model"], r["metric"]): r for r in means}
    for row in head:
        key = (row["strategy"], row["model"],
               "final_all" if row["metric"] == "all_classes" else "final_supported")
        t46 = mean_by[key]
        assert abs(float(row["final_mean"]) - float(t46["mean"])) <= 5e-6
        assert int(row["n_seeds"]) == int(t46["n"]) == 5
    res = {
        "schema_version": 1,
        "kind": "interim_presentation_values_2026-11-03",
        "workspace_head": "130a6d0",
        "paper_pin": "3f3dbad",
        "certified_by": {
            "s3_s5_headline": "D-075/T42 - results/text/verification/data_audit/t42_rerun_audit.json + t42_headline_verify.json (overall pass)",
            "t46_stats": "D-077 - results/text/verification/data_audit/t46_stats_audit.json (overall pass; overclaim WARN non-blocking)",
            "paper_s5": "T47 - results/text/verification/data_audit/t47_paper_stats_audit.json (overall pass; pinned 3f3dbad)",
        },
        "audit_status": {"t46_stats": "pass", "t47_paper": "pass", "t42_rerun": "pass",
                         "t42_headline_signoff": "PASS"},
        "s5_headline": s5_rows,
        "stats_comparisons": stats_rows,
        "presentation_facts": facts,
        "must_not_present": {
            "noise_floor_f4": "NOT SIGNED - old smoke used sampled/10-epoch model-null; fixed-code rerun on v7 pending",
            "corrected_s7": "PENDING - stage-1 v6 evidence only; v7 relaunch in flight; expected 19,266 t>=j pairs = 13,218 cross + 6,048 diagonal -> 57,798 rows (D-079)",
            "s8_s12": "PENDING/DEFERRED - no signed artifacts",
        },
        "required_caveats": [
            "cross-snapshot provenance: FT v3/90e7635, replay v4/9ca4681, EWC v6/649109a; frozen data/CSV/lane unchanged",
            "5 seeds, single frozen C1 stream; no population-level or causal claim",
            "Holm within each (model, metric) family of 3 strategy comparisons only",
            "non-significance is not proven equivalence",
            "rare classes support-limited (supported set excludes Analysis/Backdoor/Worms)",
            "lambda=100 pilot selection; lambda=1000 labelled sensitivity only",
            "replay/EWC are baseline context, not drift evidence",
        ],
        "presentation_rule": "formatting-only re-layout of these values is safe; any changed/re-derived number needs re-audit; no noise/S7/S8-S12 numbers",
    }
    OUT.write_text(json.dumps(res, indent=2, default=str) + "\n", encoding="utf-8")
    print("wrote", OUT.relative_to(REPO))
    print("s5 rows", len(s5_rows), "stats rows", len(stats_rows),
          "audits", res["audit_status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
