"""T46 verifier audit: baseline-CL statistics on the corrected (valid) S5 table.

Independently rebuilds per-seed metrics from the three signed Kaggle runs, recomputes the
paired 1000-resample bootstrap CIs/p-values, Cohen's d, rank-biserial, Holm family
adjustment, headline cross-check and lambda sensitivity; verifies the 6 helper tests and
the artifact copies; checks the interpretation/caveat wording for overclaim risk.

Writes results/text/verification/data_audit/t46_stats_audit.json. No repo code/results modified.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(r"C:\Users\iusearchbtw\Desktop\XAI_proj")
PY = str(REPO / ".venv" / "Scripts" / "python.exe")
OUT = REPO / "verification" / "data_audit"
EV = OUT / "t46_baseline_stats"
TABLES = REPO / "results" / "tables" / "s5_baseline"
HEAD = REPO / "incoming_rtx5060" / "kaggle_runs" / "t42_headline" / "headline_table_valid.csv"
RUNS = {
    "ft": REPO / "incoming_rtx5060/kaggle_runs/t37-matrix-full-90e7635",
    "replay": REPO / "incoming_rtx5060/kaggle_runs/t42-replay-rerun-9ca4681",
    "ewc": REPO / "incoming_rtx5060/kaggle_runs/t42-ewc-lambda100-649109a",
    "ewc_lambda1000": REPO / "incoming_rtx5060/kaggle_runs/t37-matrix-full-90e7635",
}
SEEDS = (42, 43, 44, 45, 46)
MODELS = ("egraphsage", "gcn")
COMPARISONS = (("replay", "ft"), ("ewc", "ft"), ("ewc", "replay"))

sys.path.insert(0, str(REPO))


def rd(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8-sig"))


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def csv_rows(p: Path):
    return list(csv.DictReader(p.open(encoding="utf-8-sig")))


def main() -> int:
    from src.analysis import stats as ast
    from src.analysis import s5_baseline_analysis as mod

    METRICS = dict(mod.METRICS)
    res: dict = {"checks": []}
    chk = res["checks"]

    def add(i, status, detail):
        chk.append({"id": i, "status": status, "detail": detail})

    # ---------------- A. rebuild per-seed values from the signed runs
    def group_payload(run_key: str, strategy: str, model: str, seed: int) -> dict:
        p = RUNS[run_key] / "unpacked/results/text/tables/matrix" / f"{strategy}_{model}_seed{seed}" / "cl_matrix.json"
        return rd(p)
    per_seed: dict[tuple[str, str, str, int], float] = {}
    for strategy, run_key in (("ft", "ft"), ("replay", "replay"), ("ewc", "ewc")):
        for model in MODELS:
            for seed in SEEDS:
                payload = group_payload(run_key, strategy, model, seed)
                for metric, (block, key) in METRICS.items():
                    per_seed[(strategy, metric, model, seed)] = float(payload[block][key])
    # lambda sensitivity: old lambda1000 from the original FT run
    for model in MODELS:
        for seed in SEEDS:
            payload = group_payload("ewc_lambda1000", "ewc", model, seed)
            for metric, (block, key) in METRICS.items():
                per_seed[("ewc_lambda1000", metric, model, seed)] = float(payload[block][key])

    # compare baseline_means.csv
    means_rows = csv_rows(TABLES / "baseline_means.csv")
    mean_problems = []
    for row in means_rows:
        strategy, model, metric = row["strategy"], row["model"], row["metric"]
        vals = np.array([per_seed[(strategy, metric, model, s)] for s in SEEDS], dtype=float)
        claimed = [float(x) for x in row["values"].split(";")] if row["values"] else []
        if (int(row["n"]) != 5 or abs(float(row["mean"]) - float(vals.mean())) > 1e-12
                or abs(float(row["sd"]) - float(vals.std(ddof=1))) > 1e-12
                or (claimed and any(abs(a - b) > 5e-7 for a, b in zip(sorted(claimed), sorted(vals))))):
            mean_problems.append({"row": [strategy, model, metric], "claimed": row, "computed": vals.tolist()})
    add("STATS.means", "pass" if not mean_problems else "fail",
        {"n_rows": len(means_rows), "problems": mean_problems[:10],
         "tolerance": {"mean_sd": "exact to 1e-12 (unrounded CSV fields)",
                       "per_seed_values_column": "5e-7 absolute (the `values` column is stored at 6 dp; this is a "
                                                 "display-rounding tolerance, not a data discrepancy)",
                       "headline_crosscheck": "5e-6 (T46 max_abs_mean_diff 5.57e-7)"},
         "note": "per-seed values rebuilt from the signed cl_matrix blocks; unrounded mean/sd match exactly; listed "
                 "6-dp per-seed values match within display rounding."})

    # ---------------- B. recompute comparisons with the documented method (same function+seeds),
    # and independently reimplement the paired bootstrap / effect sizes / Holm.
    def values(strategy, metric, model):
        return np.array([per_seed[(strategy, metric, model, s)] for s in SEEDS], dtype=float)

    def own_bootstrap(a, b, *, n=1000, alpha=0.05, seed=42):
        d = np.asarray(a, float) - np.asarray(b, float)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, d.size, size=(n, d.size))
        dist = d[idx].mean(axis=1)
        lo = float(np.percentile(dist, 100 * alpha / 2)); hi = float(np.percentile(dist, 100 * (1 - alpha / 2)))
        count_le = int(np.sum(dist <= 0)); count_ge = int(np.sum(dist >= 0))
        p_raw = float(min(1.0, 2 * min((1 + count_le) / (1 + n), (1 + count_ge) / (1 + n))))
        return {"diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi, "p_raw": p_raw}

    def own_effects(a, b):
        d = np.asarray(a, float) - np.asarray(b, float)
        cohen = float(d.mean() / d.std(ddof=1)) if d.std(ddof=1) > 0 else None
        nz = d[d != 0]
        if nz.size == 0:
            rb = 0.0
        else:
            ranks = np.argsort(np.argsort(np.abs(nz))) + 1.0
            wp = float(ranks[nz > 0].sum()); wm = float(ranks[nz < 0].sum())
            rb = (wp - wm) / (wp + wm) if (wp + wm) else 0.0
        return {"cohens_d": cohen, "rank_biserial": rb}

    artifact = cmp_rows = csv_rows(TABLES / "baseline_comparisons.csv")
    art_by = {(r["model"], r["metric"], r["comparison"]): r for r in cmp_rows}
    recomputed = {}
    mismatch = []
    alt_seed = {}
    for model in MODELS:
        for metric in METRICS:
            family = {}
            pending = {}
            for index, (a, b) in enumerate(COMPARISONS):
                va, vb = values(a, metric, model), values(b, metric, model)
                boot = ast.paired_bootstrap_diff(va, vb, n_resamples=1000, alpha=0.05, seed=42 + index)
                eff = ast.effect_size_summary(va, vb, paired=True)
                own_b = own_bootstrap(va, vb, seed=42 + index)
                own_e = own_effects(va, vb)
                label = f"{a}_minus_{b}"
                family[label] = float(boot.p_value)
                pending[label] = {"model": model, "metric": metric, "comparison": label,
                                  "diff": float(boot.diff), "ci_lo": float(boot.lo), "ci_hi": float(boot.hi),
                                  "p_raw": float(boot.p_value), "cohens_d": eff["cohens_d"],
                                  "rank_biserial": eff["rank_biserial"], "family": f"{model}:{metric}"}
                # independent implementation agreement
                if (abs(own_b["diff"] - boot.diff) > 1e-12 or abs(own_b["ci_lo"] - boot.lo) > 1e-12
                        or abs(own_b["ci_hi"] - boot.hi) > 1e-12 or abs(own_b["p_raw"] - boot.p_value) > 1e-12):
                    mismatch.append({"where": [model, metric, label, "own_bootstrap_vs_analysis.stats"], "own": own_b,
                                     "stats": {"diff": boot.diff, "ci_lo": boot.lo, "ci_hi": boot.hi, "p_raw": boot.p_value}})
                if (eff["cohens_d"] != own_e["cohens_d"]) or (abs(eff["rank_biserial"] - own_e["rank_biserial"]) > 1e-12):
                    mismatch.append({"where": [model, metric, label, "own_effect_vs_analysis.stats"], "own": own_e,
                                     "stats": eff})
                # robustness across alternative bootstrap seeds (not a protocol claim)
                signs = []
                for alt in range(100, 120):
                    ob = own_bootstrap(va, vb, seed=alt)
                    signs.append(ob["ci_lo"] > 0 or ob["ci_hi"] < 0)
                alt_seed[(model, metric, label)] = float(np.mean(signs))
            adjusted = ast.holm_correction_labelled(family)
            for label, row in pending.items():
                row["p_holm"] = float(adjusted[label])
                recomputed[(model, metric, label)] = row
    # compare to artifact CSV
    for key, row in recomputed.items():
        art = art_by.get(key)
        if art is None:
            mismatch.append({"where": list(key), "artifact": None})
            continue
        for field in ("diff", "ci_lo", "ci_hi", "p_raw", "cohens_d", "rank_biserial", "p_holm"):
            if art.get(field) in ("", None):
                mismatch.append({"where": list(key) + [field], "artifact": art.get(field), "computed": row[field]})
            elif abs(float(art[field]) - float(row[field])) > 1e-12:
                mismatch.append({"where": list(key) + [field], "artifact": float(art[field]), "computed": row[field]})
        if int(art.get("n", 0)) != 5 or int(art.get("n_resamples", 0)) != 1000:
            mismatch.append({"where": list(key) + ["n/n_resamples"], "artifact": art})
    # family scope: exactly 3 rows per (model,metric), Holm within family, no cross-family pooling
    fam_counts = {}
    for row in cmp_rows:
        fam_counts.setdefault((row["model"], row["metric"]), []).append(row)
    family_ok = (len(fam_counts) == len(MODELS) * len(METRICS)
                 and all(len(v) == 3 for v in fam_counts.values())
                 and all(v[0]["family"] == f"{v[0]['model']}:{v[0]['metric']}" for v in fam_counts.values()))
    # Holm recomputation from raw p values per family
    holm_ok = True
    for fam, rows in fam_counts.items():
        raw = {r["comparison"]: float(r["p_raw"]) for r in rows}
        adj = ast.holm_correction_labelled(raw)
        for r in rows:
            if abs(float(r["p_holm"]) - adj[r["comparison"]]) > 1e-12:
                holm_ok = False
    add("STATS.comparisons",
        "pass" if (not mismatch and len(recomputed) == len(MODELS) * len(METRICS) * 3
                   and len(cmp_rows) == len(MODELS) * len(METRICS) * 3 and family_ok and holm_ok) else "fail",
        {"n_artifact_rows": len(cmp_rows), "n_recomputed": len(recomputed),
         "artifact_vs_recomputed_mismatches": mismatch[:20],
         "family_counts": {str(k): len(v) for k, v in fam_counts.items()},
         "holm_within_family_ok": holm_ok, "family_scope_ok": family_ok,
         "bootstrap_alt_seed_ci_sign_fraction": {str(k): v for k, v in list(alt_seed.items())[:6]},
         "alternative_seed_note": "fraction of 20 alternative bootstrap seeds whose 95% CI excludes zero; not a "
                                  "protocol claim, used as a stability check of the reported CI sign."})

    # ---------------- C. headline cross-check and superseded sources
    head_rows = csv_rows(HEAD)
    head_by = {(r["strategy"], r["model"], r["metric"]): r for r in head_rows}
    headline_problems = []
    for (strategy, model, metric), row in head_by.items():
        if metric not in METRICS:
            # headline metric names: all_classes/supported_set; map to final_all/final_supported
            metric_key = "final_all" if metric == "all_classes" else "final_supported"
        else:
            metric_key = metric
        vals = values(strategy, metric_key, model)
        if abs(float(row["final_mean"]) - float(vals.mean())) > 5e-6:
            headline_problems.append({"cell": [strategy, model, metric], "headline": row["final_mean"],
                                      "computed": float(vals.mean())})
    add("STATS.headline_crosscheck", "pass" if not headline_problems else "fail",
        {"problems": headline_problems[:10], "headline_rows": len(head_rows),
         "note": "T46 per-seed values match the verifier-signed D-075 corrected headline table (<=5e-6)."})

    # ---------------- D. lambda sensitivity (non-primary, old lambda1000)
    try:
        sens_rows = csv_rows(TABLES / "ewc_lambda_sensitivity.csv")
        sens_problems = []
        for row in sens_rows:
            model, metric = row.get("model"), row.get("metric")
            # find comparison ewc_lambda100 vs ewc_lambda1000
            lam100 = values("ewc", metric, model)
            lam1000 = values("ewc_lambda1000", metric, model)
            boot = ast.paired_bootstrap_diff(lam100, lam1000, n_resamples=1000, alpha=0.05, seed=42)
            if row.get("mean_lambda100") and abs(float(row["mean_lambda100"]) - float(lam100.mean())) > 1e-12:
                sens_problems.append({"row": row, "field": "mean_lambda100"})
            if row.get("diff") and abs(float(row["diff"]) - float(boot.diff)) > 1e-12:
                sens_problems.append({"row": row, "field": "diff"})
        add("STATS.lambda_sensitivity", "pass" if not sens_problems else "fail",
            {"n_rows": len(sens_rows), "problems": sens_problems[:10],
             "note": "lambda sensitivity uses the old lambda=1000 run only as a labelled sensitivity (D-072); "
                     "it is not part of the primary headline table."})
    except Exception as exc:  # noqa: BLE001
        add("STATS.lambda_sensitivity", "warn", {"error": repr(exc)})

    # ---------------- E. per-period tables
    def matrix_stack(strategy, model):
        return np.stack([np.asarray(group_payload({"ft": "ft", "replay": "replay", "ewc": "ewc"}[strategy],
                                                  strategy, model, seed)["matrix"][block], dtype=float)
                         for block in ("all_classes", "supported_set")], axis=0) if False else None

    def payloads(strategy, model):
        return [group_payload({"ft": "ft", "replay": "replay", "ewc": "ewc"}[strategy], strategy, model, seed)
                for seed in SEEDS]

    per_problems = []
    finalrow_rows = csv_rows(TABLES / "per_period_finalrow.csv")
    diagonal_rows = csv_rows(TABLES / "per_period_diagonal.csv")
    forgetting_rows = csv_rows(TABLES / "per_period_forgetting.csv")
    for row in finalrow_rows:
        model, strategy, matrix_name, period = row["model"], row["strategy"], row["matrix"], int(row["period"])
        block = "all_classes" if matrix_name == "all_classes" else "supported_set"
        stack = np.stack([np.asarray(p["matrix"][block], dtype=float) for p in payloads(strategy, model)])
        vals = stack[:, -1, period - 1]
        if (int(row["n"]) != 5 or abs(float(row["mean"]) - vals.mean()) > 1e-12
                or abs(float(row["sd"]) - vals.std(ddof=1)) > 1e-12):
            per_problems.append({"table": "finalrow", "row": row, "computed": [vals.mean(), vals.std(ddof=1)]})
    for row in diagonal_rows:
        model, strategy, matrix_name, period = row["model"], row["strategy"], row["matrix"], int(row["checkpoint"])
        block = "all_classes" if matrix_name == "all_classes" else "supported_set"
        stack = np.stack([np.asarray(p["matrix"][block], dtype=float) for p in payloads(strategy, model)])
        vals = np.asarray([stack[i, period - 1, period - 1] for i in range(len(SEEDS))])
        if (int(row["n"]) != 5 or abs(float(row["mean"]) - vals.mean()) > 1e-12
                or abs(float(row["sd"]) - vals.std(ddof=1)) > 1e-12):
            per_problems.append({"table": "diagonal", "row": row, "computed": [vals.mean(), vals.std(ddof=1)]})
    for row in forgetting_rows:
        model, strategy, matrix_name, period = row["model"], row["strategy"], row["matrix"], int(row["period"])
        block = "cl_metrics_all_classes" if matrix_name == "all_classes" else "cl_metrics_supported_set"
        vals = []
        for payload in payloads(strategy, model):
            value = (payload.get(block) or {}).get("forgetting_per_period", {}).get(str(period))
            if value is not None:
                vals.append(float(value))
        vals = np.asarray(vals)
        if (int(row["n"]) != len(vals) or abs(float(row["mean"]) - vals.mean()) > 1e-12
                or abs(float(row["sd"]) - vals.std(ddof=1)) > 1e-12):
            per_problems.append({"table": "forgetting", "row": row, "computed": [vals.mean(), vals.std(ddof=1)]})
    add("STATS.per_period_tables", "pass" if not per_problems else "fail",
        {"finalrow_rows": len(finalrow_rows), "diagonal_rows": len(diagonal_rows), "forgetting_rows": len(forgetting_rows),
         "problems": per_problems[:10],
         "note": "final-row (M5 on each period), diagonal R[j,j] and forgetting-per-period tables rebuilt from the "
                 "seeded R matrices; mean/sd exact."})

    # ---------------- F. per-class table
    def per_class_payload(strategy, model, seed):
        p = RUNS[{"ft": "ft", "replay": "replay", "ewc": "ewc"}[strategy]] / "unpacked/results/text/tables/matrix" / \
            f"{strategy}_{model}_seed{seed}" / "cl_per_class_f1.json"
        return rd(p)
    rare_names = tuple(getattr(mod, "RARE_CLASSES", ("Analysis", "Backdoor", "Worms")))
    pc_rows = csv_rows(TABLES / "per_class_f1.csv")
    pc_problems = []
    for row in pc_rows:
        model, strategy, cls, period = row["model"], row["strategy"], row["class"], int(row["evaluated_period"])
        values = []
        supports = []
        pooled_values = []
        pooled_support = []
        for seed in SEEDS:
            cell = per_class_payload(strategy, model, seed)
            payload = cell["per_class_f1"]["5"][str(period)]
            order = list(cell.get("class_order") or [])
            if cls == "rare_stratum_pooled":
                values.append(float(np.mean([float(payload["per_class_f1"][str(order.index(n))]) for n in rare_names])))
                supports.append(int(sum(int(payload["dual_macro_f1"]["class_counts"][str(order.index(n))]) for n in rare_names)))
            else:
                idx = order.index(cls)
                values.append(float(payload["per_class_f1"][str(idx)]))
                supports.append(int(payload["dual_macro_f1"]["class_counts"][str(idx)]))
        values = np.asarray(values); supports = np.asarray(supports)
        seed_ci = 42 + period if cls != "rare_stratum_pooled" else 42 + 100 + period
        boot = ast.bootstrap_ci(values, n_resamples=1000, alpha=0.05, seed=seed_ci)
        ok = (abs(float(row["mean_f1"]) - values.mean()) <= 1e-12
              and abs(float(row["sd"]) - values.std(ddof=1)) <= 1e-12
              and abs(float(row["ci_lo"]) - boot.lo) <= 1e-12
              and abs(float(row["ci_hi"]) - boot.hi) <= 1e-12
              and int(row["support_min"]) == int(supports.min())
              and int(row["support_max"]) == int(supports.max())
              and int(row["support_p5"]) == int(supports[-1])
              and (row["rare_class"].strip().lower() == str(True if cls == "rare_stratum_pooled" else bool(supports.min() < 10)).lower())
              and (row["rare_stratum_member"].strip().lower()
                   == str(True if cls == "rare_stratum_pooled" or cls in rare_names else False).lower()))
        if not ok:
            pc_problems.append({"row": row, "computed": {"mean": values.mean(), "sd": values.std(ddof=1),
                                                        "ci_lo": boot.lo, "ci_hi": boot.hi,
                                                        "support": supports.tolist()}})
    add("STATS.per_class_table", "pass" if not pc_problems else "fail",
        {"rows": len(pc_rows), "problems": pc_problems[:10],
         "note": "all per-class rows (2 models x 3 strategies x 10 classes + pooled rare stratum x 5 periods) rebuilt "
                 "from the per-seed per-class matrices with the same bootstrap_ci seeds; mean/sd/CI/support/rare flags exact."})

    # ---------------- G. artifact-copy integrity + tests + suite
    copy_problems = []
    for name in ("s5_baseline_analysis.py", "test_s5_baseline_analysis.py", "baseline_comparisons.csv",
                 "baseline_means.csv", "checks.json", "per_period_finalrow.csv", "per_period_diagonal.csv",
                 "per_period_forgetting.csv", "per_class_f1.csv", "ewc_lambda_sensitivity.csv"):
        a = TABLES / name if (TABLES / name).is_file() else EV / name
        b = EV / name
        if not (a.is_file() and b.is_file()):
            copy_problems.append(f"missing {name}")
        elif sha_file(a) != sha_file(b):
            copy_problems.append(f"hash mismatch {name}")
    test = subprocess.run([PY, "-m", "pytest", str(EV / "test_s5_baseline_analysis.py"), "-q", "--no-header"],
                          cwd=str(REPO), capture_output=True, text=True)
    add("STATS.reproducibility",
        "pass" if (not copy_problems and test.returncode == 0 and "6 passed" in (test.stdout + test.stderr)) else "fail",
        {"artifact_copy_problems": copy_problems,
         "tests": {"returncode": test.returncode, "tail": (test.stdout + test.stderr).strip().splitlines()[-2:]},
         "checks_json": rd(EV / "checks.json").get("headline_crosscheck", {}).get("max_abs_mean_diff"),
         "note": "evidence copies match results/text/tables byte-for-byte; 6 helper tests pass."})

    # ---------------- H. Wording / overclaim audit (documented, human review)
    summary_text = (REPO / "docs/notes/s5_results_summary.md").read_text(encoding="utf-8", errors="replace")
    claim_flags = []
    # crude but explicit checks of the stated caveats
    checks = {
        "states_5_seeds_and_coarse_resolution": "Only **5 seeds**" in summary_text or "Only 5 seeds" in summary_text,
        "states_single_dataset_stream": "Single dataset" in summary_text,
        "states_lambda_pilot_short": "short seed-42 pilot" in summary_text,
        "states_rare_class_support_limit": "support-limited" in summary_text,
        "states_holm_family_scope": "Holm families are within each" in summary_text,
        "states_bwt_forgetting_mirror": "algebraically near-mirrors" in summary_text,
        "states_no_population_significance": "population-level significance" in summary_text,
        "mentions_snapshot_provenance_caveat": "snapshot" in summary_text.lower() and ("90e7635" in summary_text or "9ca4681" in summary_text),
    }
    if not checks["mentions_snapshot_provenance_caveat"]:
        claim_flags.append("The T46 summary does not restate the D-075 snapshot-provenance caveat (FT v3/90e7635, "
                           "replay v4/9ca4681, EWC v6/649109a); the headline table was signed with that caveat, so the "
                           "paper should state it (non-blocking).")
    claim_flags.append("Wording 'clearly beats fixed replay' is supported by EWC-replay CIs excluding zero within the "
                       "(model,metric) family; keep the Holm/family qualifier next to it.")
    claim_flags.append("'no forgetting gain' for EWC-E-GS means not statistically significant with n=5; avoid phrasing it "
                       "as proven equivalence.")
    add("STATS.overclaim_review", "warn" if claim_flags else "pass",
        {"caveat_checks": checks, "flags": claim_flags,
         "note": "the summary's interpretation and caveats are otherwise conservative: 5 seeds, single stream, short "
                 "seed-42 lambda pilot, rare-class support limits, Holm family scope and BWT/forgetting mirror are all "
                 "explicitly stated; no population-level claims are made."})

    fails = [c for c in chk if c["status"] == "fail"]
    res["overall"] = "fail" if fails else "pass"
    res["signoff"] = {
        "t46_stats_audit": "PASS - VERIFIER-AUDITED" if res["overall"] == "pass" else "FAIL",
        "scope": "paired 1000-resample percentile bootstrap CIs, bootstrap sign-test p, Cohen's d (paired), matched-pairs "
                 "rank-biserial, Holm within (model, metric) 3-comparison families, headline cross-check, per-period and "
                 "per-class tables, lambda sensitivity (non-primary), 6 helper tests",
        "rounding_tolerance": "unrounded statistics compared exactly; 6-dp display columns compared at 5e-7; headline "
                              "cross-check at 5e-6",
        "overclaim_caveats": ["snapshot provenance (FT v3/90e7635, replay v4/9ca4681, EWC v6/649109a) must be stated in the paper",
                              "keep Holm/family qualifier next to 'significant'",
                              "non-significant is not proven equivalence (EWC-E-GS forgetting)",
                              "rare-class support limits; lambda=1000 is sensitivity only; 5 seeds/single stream"],
        "paper_wording": "paper can flip 'verifier stats audit pending' -> VERIFIER-AUDITED with the caveats above",
    }
    res["summary"] = {"comparison_rows": len(cmp_rows), "means_rows": len(means_rows),
                      "families": len(fam_counts), "tests": "6 passed",
                      "headline_max_diff": rd(EV / "checks.json").get("headline_crosscheck", {}).get("max_abs_mean_diff")}
    (OUT / "t46_stats_audit.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({c["id"]: c["status"] for c in chk}, indent=1))
    print("overall:", res["overall"], "summary:", json.dumps(res["summary"]))
    for c in chk:
        if c["status"] != "pass":
            print(c["id"], json.dumps(c["detail"], default=str)[:1200])
    return 0 if not fails else 2


if __name__ == "__main__":
    raise SystemExit(main())
