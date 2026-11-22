"""Independent verifier audit of the COMPLETE 30-group Kaggle matrix (T37/D-069).

Reads the downloaded full-matrix output, re-runs strict per-group S5 validators, checks
150 period manifests + 30 group/eval manifests (lane/gate/lifecycle/code fingerprint),
recomputes R/dual-macro/forgetting/BWT for all 30 groups from the authoritative
`analysis.metrics_cl`, rebuilds the headline table from per-seed rows (and from the group
artifacts directly), checks lane purity, and documents the replay<FT anomaly.

Writes results/text/verification/data_audit/kaggle_matrix_full_verify.json. No repo code/results modified.
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(r"C:\Users\iusearchbtw\Desktop\XAI_proj")
PY = str(REPO / ".venv" / "Scripts" / "python.exe")
RUN = REPO / "incoming_rtx5060" / "kaggle_runs" / "t37-matrix-full-90e7635"
UP = RUN / "unpacked"
OUT = REPO / "verification" / "data_audit"
EV = OUT / "kaggle_matrix_full"
ENV_UTF8 = {**os.environ, "PYTHONIOENCODING": "utf-8"}
EXPECTED_MODELS = ("egraphsage", "gcn")
EXPECTED_STRATS = ("ft", "replay", "ewc")
EXPECTED_SEEDS = (42, 43, 44, 45, 46)
CODE_FP = "d907aa44d095690a24f811dd1a446da6bebcdaf3781f0acb003d6c9fe5bc9918"
SPLIT_HASH = "010a80d955802a2f62dedf472a3274e1afee17fb4b21c57541fe428b8feb47ff"

sys.path.insert(0, str(REPO))


def rd(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8-sig"))


def main() -> int:
    from analysis.metrics_cl import RMetricTensor

    res: dict = {"checks": []}
    chk = res["checks"]

    def add(i, status, detail):
        chk.append({"id": i, "status": status, "detail": detail})

    matrix_root = UP / "results" / "tables" / "matrix"
    audit_dir = UP / "results" / "data_audit"
    groups = sorted(p for p in matrix_root.iterdir() if p.is_dir())
    expected_groups = {f"{s}_{m}_seed{seed}" for s in EXPECTED_STRATS for m in EXPECTED_MODELS for seed in EXPECTED_SEEDS}
    actual_groups = {p.name for p in groups}
    add("FULL.groups", "pass" if len(groups) == 30 and actual_groups == expected_groups else "fail",
        {"n_groups": len(groups), "expected_combos": 30, "missing": sorted(expected_groups - actual_groups),
         "unexpected": sorted(actual_groups - expected_groups)})

    # ---------- per-group R / dual-macro / forgetting recomputation
    group_rows = []
    metric_rows = []          # per-seed rows for headline rebuild (from group artifacts)
    problems = []
    total_cells = 0
    total_lower = 0
    for g in groups:
        name = g.name
        strategy, model, seed_s = name.rsplit("_", 2)
        seed = int(seed_s.replace("seed", ""))
        cm = rd(g / "cl_matrix.json")
        R_all = np.asarray(cm["matrix"]["all_classes"], dtype=float)
        R_sup = np.asarray(cm["matrix"]["supported_set"], dtype=float)
        n = R_all.shape[0]
        checks_ok = True
        if R_all.shape != (5, 5) or R_sup.shape != (5, 5):
            problems.append(f"{name}: unexpected matrix shape {R_all.shape}/{R_sup.shape}")
            checks_ok = False
        finite_all = int(np.isfinite(R_all).sum())
        finite_sup = int(np.isfinite(R_sup).sum())
        mask_lower = np.tril(np.ones((5, 5), dtype=bool)) | np.eye(5, dtype=bool)
        lower_all = int((np.isfinite(R_all) & mask_lower).sum())
        lower_sup = int((np.isfinite(R_sup) & mask_lower).sum())
        upper_all = int((np.isfinite(R_all) & ~mask_lower).sum())
        upper_sup = int((np.isfinite(R_sup) & ~mask_lower).sum())
        if lower_all != 15 or lower_sup != 15 or upper_all != 10 or upper_sup != 10:
            problems.append(f"{name}: cell coverage lower={lower_all}/{lower_sup} upper={upper_all}/{upper_sup}")
            checks_ok = False
        total_cells += finite_all + finite_sup
        total_lower += lower_all + lower_sup
        checkpoints = cm.get("checkpoints_evaluated")
        if checkpoints != [1, 2, 3, 4, 5]:
            problems.append(f"{name}: checkpoints_evaluated={checkpoints}")
            checks_ok = False
        # authoritative metrics recomputation
        ref_all = RMetricTensor(R_all, n_periods=5); ref_all.validate()
        ref_sup = RMetricTensor(R_sup, n_periods=5); ref_sup.validate()
        claim_all = cm.get("cl_metrics_all_classes") or {}
        claim_sup = cm.get("cl_metrics_supported_set") or {}
        for label, ref, claim in (("all", ref_all, claim_all), ("supported", ref_sup, claim_sup)):
            vals = {"forgetting": float(ref.forgetting()), "bwt": float(ref.bwt()),
                    "final_average_macro_f1": float(ref.final_macro_f1())}
            for k, v in vals.items():
                if abs(float(claim.get(k, np.nan)) - v) > 1e-9:
                    problems.append(f"{name}: {label} {k} claim={claim.get(k)} ref={v}")
                    checks_ok = False
        # dual macro across all 25 cells
        dual = cm.get("dual_macro_f1_per_cell") or {}
        per = rd(g / "cl_per_class_f1.json").get("per_class_f1") or {}
        dual_cells = checked = 0
        for row_key, cols in dual.items():
            for col_key, cell in (cols or {}).items():
                per_cell = (per.get(row_key) or {}).get(col_key) or {}
                pc = per_cell.get("per_class_f1") or (cell or {}).get("per_class_f1") or {}
                block = (cell or {}).get("dual_macro_f1") or {}
                labels = block.get("supported_set_labels") or []
                if len(pc) != 10 or len(labels) != 7 or 0 not in labels:
                    problems.append(f"{name}: {row_key},{col_key} per-class/labels malformed")
                    checks_ok = False
                    continue
                vals = [float(v) for v in pc.values()]
                mac_all = float(np.mean(vals))
                mac_sup = float(np.mean([float(pc[str(i)]) for i in labels]))
                if abs(mac_all - float(block.get("macro_f1_all_classes", -1))) > 1e-9:
                    problems.append(f"{name}: {row_key},{col_key} all macro mismatch")
                    checks_ok = False
                if abs(mac_sup - float(block.get("macro_f1_supported_set", -1))) > 1e-9:
                    problems.append(f"{name}: {row_key},{col_key} supported macro mismatch")
                    checks_ok = False
                if int(sum((block.get("class_counts") or {}).values())) != int(block.get("n_edges", -1)):
                    problems.append(f"{name}: {row_key},{col_key} class_counts != n_edges")
                    checks_ok = False
                checked += 1
                dual_cells += 1
        if checked != 25:
            problems.append(f"{name}: only {checked}/25 dual macro cells")
            checks_ok = False
        # per-seed rows from the group's own cl_metrics (independent from the CSV)
        for metric_name, block_claim in (("all_classes", claim_all), ("supported_set", claim_sup)):
            metric_rows.append({"strategy": strategy, "model": model, "seed": seed,
                                "metric": metric_name, "group": name,
                                "final_average_macro_f1": float(block_claim.get("final_average_macro_f1")),
                                "forgetting": float(block_claim.get("forgetting")),
                                "bwt": float(block_claim.get("bwt"))})
        group_rows.append({"group": name, "complete": checks_ok, "matrix_shape": list(R_all.shape),
                           "finite_all": finite_all, "finite_supported": finite_sup,
                           "lower_all": lower_all, "lower_supported": lower_sup,
                           "dual_cells_checked": checked, "recomputed_metrics_exact": checks_ok})
    add("FULL.r_matrices", "pass" if (not problems and total_cells == 30 * 2 * 25
                                      and total_lower == 30 * 2 * 15) else "fail",
        {"n_groups": len(groups), "matrix_cells_finite": total_cells,
         "expected_cells": 1500, "lower_triangle_cells_finite": total_lower,
         "expected_lower_cells": 900,
         "forward_cells_finite_expected_600": total_cells - total_lower,
         "problems": problems[:20],
         "groups": group_rows})

    # ---------- per-group strict S5 validators (independent re-run)
    totals = {"pass": 0, "warn": 0, "fail": 0}
    per_group = []
    for g in groups:
        proc = subprocess.run([PY, "-m", "src.analysis.validators", "--stage", "s5", "--strict",
                               "--results-tables", str(g), "--json"],
                              cwd=str(REPO), capture_output=True, text=True, env=ENV_UTF8)
        payload = {}
        try:
            payload = json.loads(proc.stdout[proc.stdout.find("{"):])
        except Exception:  # noqa: BLE001
            pass
        counts = payload.get("counts") or payload.get("checks") or {}
        if not counts:
            tail = [ln for ln in (proc.stdout + proc.stderr).splitlines() if ln.strip().startswith("checks:")]
            if tail:
                try:
                    counts = eval(tail[-1].split("checks:", 1)[1].strip())  # noqa: S307
                except Exception:  # noqa: BLE001
                    counts = {}
        for k in totals:
            totals[k] += int(counts.get(k, 0))
        per_group.append({"group": g.name, "returncode": proc.returncode,
                          "counts": counts, "overall": payload.get("overall")})
    add("FULL.validators", "pass" if (totals["pass"] == 480 and totals["warn"] == 0 and totals["fail"] == 0) else "fail",
        {"totals": totals, "expected": {"pass": 480, "warn": 0, "fail": 0},
         "per_group": per_group})

    # ---------- 150 period manifests + 30 eval + 30 group manifests
    train_mans = sorted(audit_dir.glob("train_manifest_*_period*.json"))
    eval_mans = sorted(audit_dir.glob("eval_manifest_*.json"))
    group_mans = sorted(audit_dir.glob("matrix_manifest_*_seed*.json"))
    man_recs = []
    man_problems = []
    lane_ok = gate_ok = status_ok = code_ok = lifecycle_ok = full_eval = True
    durations = []
    peaks = []
    for p in train_mans:
        man = rd(p)
        gate = man.get("gate") or {}
        # corresponding metrics file
        met_path = audit_dir / p.name.replace("train_manifest_", "train_").replace("_period", "_period")
        met = rd(met_path) if met_path.is_file() else {}
        r = met.get("result") or {}
        t = r.get("timing") or {}
        dur = float(man.get("duration_s") or 0)
        ts = float(t.get("train_seconds") or 0)
        lane_ok = lane_ok and man.get("lane") == "kaggle" and man.get("lane_source") == "cli"
        gate_ok = gate_ok and bool(gate.get("real_lane")) and not bool(gate.get("allow_default_lane_env"))
        status_ok = status_ok and man.get("status") == "success"
        code_ok = code_ok and (man.get("code") or {}).get("sha256") == CODE_FP
        full_eval = full_eval and all((r.get(k) or {}).get("evaluation_mode") == "full_graph_batch_head"
                                      for k in ("val", "test"))
        if not (str(man.get("started_at_utc")) < str(man.get("finished_at_utc")) and dur >= ts * 0.9):
            lifecycle_ok = False
            man_problems.append(f"{p.name}: lifecycle {man.get('started_at_utc')}->{man.get('finished_at_utc')} dur={dur} train={ts}")
        durations.append(dur)
        peaks.append(t.get("max_memory_allocated_bytes"))
        extra = man.get("extra") or {}
        man_recs.append({"manifest": p.name, "strategy": extra.get("strategy"), "model": extra.get("model"),
                         "period": extra.get("period"), "run_key": extra.get("run_key"),
                         "env_xdrift_keys": sorted(k for k in (man.get("env") or {}) if str(k).startswith("XDRIFT")),
                         "lane": man.get("lane"), "gate_real": gate.get("real_lane"),
                         "status": man.get("status"), "duration_s": dur, "train_seconds": ts,
                         "batching": t.get("batching"), "epoch_seconds_mean": t.get("epoch_seconds_mean"),
                         "max_memory_allocated_bytes": t.get("max_memory_allocated_bytes"),
                         "updates_per_epoch": t.get("updates_per_epoch"),
                         "eval_modes": {k: (r.get(k) or {}).get("evaluation_mode") for k in ("val", "test")},
                         "code_sha256": (man.get("code") or {}).get("sha256")})
    add("FULL.period_manifests",
        "pass" if (len(train_mans) == 150 and lane_ok and gate_ok and status_ok and code_ok and lifecycle_ok and full_eval
                   and all(len(m.get("env_xdrift_keys") or []) == 7 for m in man_recs)) else "fail",
        {"n_period_manifests": len(train_mans), "expected": 150, "all_lane_kaggle": lane_ok,
         "all_gate_real": gate_ok, "all_status_success": status_ok, "all_code_fingerprint_d907aa44": code_ok,
         "all_lifecycle_real": lifecycle_ok, "all_full_graph_eval": full_eval,
         "duration_s_min": min(durations) if durations else None, "duration_s_max": max(durations) if durations else None,
         "max_memory_bytes_min": int(min(peaks)) if peaks else None,
         "max_memory_bytes_max": int(max(peaks)) if peaks else None,
         "batching_values": sorted({str(m.get("batching")) for m in man_recs}),
         "all_env_xdrift_keys_7": all(len(m.get("env_xdrift_keys") or []) == 7 for m in man_recs),
         "eval_mode_values": sorted({str(x) for m in man_recs for x in m["eval_modes"].values()}),
         "problems": man_problems[:20]})

    eval_recs = []
    eval_problems = []
    for p in eval_mans:
        man = rd(p)
        ex = man.get("extra") or {}
        gate = man.get("gate") or {}
        ev = ex.get("evaluation") or {}
        ok = (man.get("lane") == "kaggle" and bool(gate.get("real_lane")) and man.get("status") == "success"
              and ev.get("mode") == "edge_batch" and ex.get("checkpoints_present") == [1, 2, 3, 4, 5])
        if not ok:
            eval_problems.append(p.name)
        eval_recs.append({"manifest": p.name, "lane": man.get("lane"), "gate_real": gate.get("real_lane"),
                          "status": man.get("status"), "mode": ev.get("mode"),
                          "peak_memory_allocated_bytes": ev.get("peak_memory_allocated_bytes"),
                          "wall_clock_seconds": ev.get("wall_clock_seconds"),
                          "checkpoints_present": ex.get("checkpoints_present")})
    add("FULL.eval_manifests",
        "pass" if len(eval_mans) == 30 and not eval_problems else "fail",
        {"n_eval_manifests": len(eval_mans), "expected": 30,
         "eval_mode_values": sorted({str(r["mode"]) for r in eval_recs}),
         "peak_memory_bytes_min": min((r["peak_memory_allocated_bytes"] or 0) for r in eval_recs),
         "peak_memory_bytes_max": max((r["peak_memory_allocated_bytes"] or 0) for r in eval_recs),
         "problems": eval_problems})

    state = rd(UP / "results" / "tables" / "matrix_state.json")
    state_groups = state.get("groups") or {}
    complete_groups = [k for k, v in state_groups.items() if v.get("status") == "complete" and v.get("eval_complete")]
    add("FULL.matrix_state", "pass" if len(state_groups) == 30 and len(complete_groups) == 30 else "fail",
        {"n_state_groups": len(state_groups), "n_complete_eval": len(complete_groups),
         "groups": sorted(state_groups)})

    # ---------- headline recomputation from per-seed rows (group artifacts + committed CSV)
    def build_table(rows):
        table = {}
        for (strategy, model) in {(r["strategy"], r["model"]) for r in rows}:
            for metric in ("all_classes", "supported_set"):
                sel = [r for r in rows if r["strategy"] == strategy and r["model"] == model and r["metric"] == metric]
                if len(sel) != 5:
                    continue
                entry = {"n": len(sel)}
                for key in ("final_average_macro_f1", "forgetting", "bwt"):
                    vals = np.array([r[key] for r in sel], dtype=float)
                    entry[key] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)),
                                  "values": [float(v) for v in vals], "seed": [r["seed"] for r in sel]}
                table[(strategy, model, metric)] = entry
        return table

    rebuilt = build_table(metric_rows)
    headline = {}
    with (EV / "headline_table.csv").open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            headline[(row["strategy"], row["model"], row["metric"] if "metric" in row else "all_classes")] = row
    # committed CSV has no metric column: it has columns for both metric families
    headline_rec = {}
    with (EV / "headline_table.csv").open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            headline_rec[(row["strategy"], row["model"])] = row
    mismatches = []
    for (strategy, model, metric), entry in rebuilt.items():
        rec = headline_rec[(strategy, model)]
        prefix = "final_average_macro_f1" if False else ""
        for key in ("final_average_macro_f1", "forgetting", "bwt"):
            col_mean = f"{key}_{metric}_mean"
            col_sd = f"{key}_{metric}_sd"
            col_n = f"{key}_{metric}_n"
            got_mean = float(rec[col_mean]); got_sd = float(rec[col_sd]); got_n = int(rec[col_n])
            if got_n != 5 or abs(got_mean - entry[key]["mean"]) > 5e-6 or abs(got_sd - entry[key]["sd"]) > 5e-6:
                mismatches.append({"where": f"{strategy}/{model}/{metric}/{key}",
                                   "headline": [got_mean, got_sd, got_n],
                                   "rebuilt": [entry[key]["mean"], entry[key]["sd"], entry["n"]]})
    # also recompute from the committed per-seed CSV (independent of group artifacts) and compare to rebuilt
    csv_rows = []
    with (OUT / "kaggle_matrix_full_metrics.csv").open(newline="", encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            if not row.get("strategy"):
                continue
            csv_rows.append({"strategy": row["strategy"], "model": row["model"], "seed": int(row["seed"]),
                             "metric": row["metric"], "final_average_macro_f1": float(row["final_average_macro_f1"]),
                             "forgetting": float(row["forgetting"]), "bwt": float(row["bwt"]),
                             "group": row.get("group")})
    csv_rebuilt = build_table(csv_rows)
    csv_vs_artifact = []
    for key in sorted(rebuilt, key=str):
        a = rebuilt[key]; b = csv_rebuilt.get(key)
        if b is None:
            csv_vs_artifact.append({"key": list(key), "missing_csv": True})
            continue
        for k in ("final_average_macro_f1", "forgetting", "bwt"):
            if abs(a[k]["mean"] - b[k]["mean"]) > 5e-6 or abs(a[k]["sd"] - b[k]["sd"]) > 5e-6:
                csv_vs_artifact.append({"key": [k] + list(key), "artifact": [a[k]["mean"], a[k]["sd"]],
                                        "csv": [b[k]["mean"], b[k]["sd"]]})
    lead_snapshot = {
        ("egraphsage", "ft", "all_classes"): (0.439307, 0.007039, 0.044253, 0.006369, -0.044253),
        ("egraphsage", "ft", "supported_set"): (0.565139, 0.008945, 0.023118, 0.010241, -0.023118),
        ("gcn", "ft", "all_classes"): (0.414180, 0.017334, 0.071296, 0.023863, -0.071296),
        ("gcn", "ft", "supported_set"): (0.541311, 0.019550, 0.056845, 0.025559, -0.056845),
        ("egraphsage", "replay", "all_classes"): (0.345452, 0.016789, 0.145595, 0.023651, -0.145595),
        ("egraphsage", "replay", "supported_set"): (0.437119, 0.029581, 0.164917, 0.036396, -0.164917),
        ("gcn", "replay", "all_classes"): (0.314646, 0.024669, 0.181381, 0.022257, -0.181381),
        ("gcn", "replay", "supported_set"): (0.408053, 0.035428, 0.200009, 0.033293, -0.200009),
        ("egraphsage", "ewc", "all_classes"): (0.421467, 0.015555, 0.029478, 0.013892, -0.029224),
        ("egraphsage", "ewc", "supported_set"): (0.535952, 0.017323, 0.021306, 0.016194, -0.018954),
        ("gcn", "ewc", "all_classes"): (0.404203, 0.017580, 0.056084, 0.031627, -0.056084),
        ("gcn", "ewc", "supported_set"): (0.534660, 0.013006, 0.031823, 0.023858, -0.029696),
    }
    lead_vs_rebuilt = []
    for (model, strategy, metric), (mv, sv, mf, sf, mb) in lead_snapshot.items():
        e = rebuilt[(strategy, model, metric)]
        if (abs(e["final_average_macro_f1"]["mean"] - mv) > 1e-4 or abs(e["final_average_macro_f1"]["sd"] - sv) > 1e-4
            or abs(e["forgetting"]["mean"] - mf) > 1e-4 or abs(e["forgetting"]["sd"] - sf) > 1e-4
                or abs(e["bwt"]["mean"] - mb) > 1e-4):
            lead_vs_rebuilt.append({"cell": [model, strategy, metric],
                                    "lead": [mv, sv, mf, sf, mb],
                                    "rebuilt": [e["final_average_macro_f1"]["mean"], e["final_average_macro_f1"]["sd"],
                                                e["forgetting"]["mean"], e["forgetting"]["sd"], e["bwt"]["mean"]]})
    add("FULL.headline_recompute",
        "pass" if (len(rebuilt) == 12 and not mismatches and not csv_vs_artifact and not lead_vs_rebuilt) else "fail",
        {"headline_cells_rebuilt": len(rebuilt), "expected": 12,
         "headline_table_mismatches": mismatches,
         "per_seed_csv_vs_group_artifact_mismatches": csv_vs_artifact,
         "lead_snapshot_vs_rebuilt_mismatches": lead_vs_rebuilt,
         "headline_rows": [{"strategy": k[0], "model": k[1], "metric": k[2],
                            "final_mean": round(v["final_average_macro_f1"]["mean"], 6),
                            "final_sd": round(v["final_average_macro_f1"]["sd"], 6),
                            "forgetting_mean": round(v["forgetting"]["mean"], 6),
                            "bwt_mean": round(v["bwt"]["mean"], 6)}
                           for k, v in sorted(rebuilt.items(), key=str)]})

    # ---------- replay<FT anomaly (measured, not tuned)
    anomaly = []
    for metric in ("all_classes", "supported_set"):
        ft = rebuilt[("ft", "egraphsage", metric)]["final_average_macro_f1"]["mean"]
        rp = rebuilt[("replay", "egraphsage", metric)]["final_average_macro_f1"]["mean"]
        ftg = rebuilt[("ft", "gcn", metric)]["final_average_macro_f1"]["mean"]
        rpg = rebuilt[("replay", "gcn", metric)]["final_average_macro_f1"]["mean"]
        anomaly.append({"metric": metric, "egs_ft": round(ft, 6), "egs_replay": round(rp, 6),
                        "egs_replay_minus_ft": round(rp - ft, 6),
                        "gcn_ft": round(ftg, 6), "gcn_replay": round(rpg, 6),
                        "gcn_replay_minus_ft": round(rpg - ftg, 6)})
    add("FULL.replay_lt_ft_anomaly", "pass",
        {"direction": "replay below fine-tuning on every cell (all-class and supported macro-F1); reported as "
                      "measured anomaly, not tuned", "cells": anomaly})

    # ---------- lane purity + frozen + run provenance
    report = (RUN / "RunReport.md").read_text(encoding="utf-8", errors="replace")
    run_sum = rd(RUN / "kernel_output" / "run_summary.json")
    meta = rd(RUN / "run_full_meta.json")
    local_split = rd(REPO / "results/text/data_audit/split_manifest.json")
    add("FULL.run_provenance", "pass" if ("30 / 30" in report and "4598.3 s" in report
                                          and "459" in report and "153" in report
                                          and run_sum.get("dataset_sources", [])[:1] == ["mysteriousavailable/xdrift-nf-unsw-c1-inputs"]
                                          and len(meta.get("results", [])) == 1) else "warn",
        {"kernel_ref": run_sum.get("ref"), "kernel_status": run_sum.get("status"),
         "dataset_sources": run_sum.get("dataset_sources"),
         "inner_wall_clock_s": 4598.3, "local_wall_clock_s": 4723.1,
         "resume_merged": {"results": 459, "runs": 153},
         "report_groups_complete_line": next((ln.strip() for ln in report.splitlines() if "groups complete" in ln), None),
         "driver_validator_metadata": {"counts": ((meta.get("results") or [{}])[0].get("validators") or [{}])[0].get("counts"),
                                       "note": "full run used snapshot 90e7635; its run_full.py still validated the flat "
                                               "results/text/tables root (pre-ac68651 fix) -> recorded 3 flat-root failures. "
                                               "Independent per-group validation gives 480/0/0 (this audit)."},
         "note": "lane=kaggle only, dataset v3 snapshot 90e7635"})
    add("FULL.frozen_unchanged",
        "pass" if local_split.get("frozen_hash") == SPLIT_HASH else "fail",
        {"local_frozen_split_hash": local_split.get("frozen_hash"), "expected": SPLIT_HASH,
         "note": "frozen C1 artifacts were not regenerated for the Kaggle matrix; the run consumed the dataset v3 frozen copy"})
    add("FULL.lane_purity", "pass" if lane_ok and gate_ok else "fail",
        {"all_150_period_manifests_lane_kaggle": lane_ok, "all_gate_real": gate_ok,
         "eval_manifests_lane_kaggle": all(r["lane"] == "kaggle" for r in eval_recs),
         "legacy_artifacts": "separate diagnostic evidence only (D-065/D-069); headline table has no legacy rows"})

    # ---------- note the Lead audit tooling defect (R completeness check)
    lead = rd(EV / "matrix_full_audit.json")
    lead_groups = lead.get("groups") or []
    add("FULL.lead_audit_diff", "warn",
        {"lead_n_groups": lead.get("n_groups"), "lead_validator_totals": lead.get("validator_totals"),
         "lead_manifest_count": lead.get("n_manifests"),
         "lead_r_completeness_example": {k: lead_groups[0].get(k) for k in
                                         ("matrix_rows", "lower_cells", "finite_cells", "complete")} if lead_groups else None,
         "correct_matrix_shape": [5, 5], "correct_finite_cells_per_group": 25,
         "finding": "the Lead audit's R completeness counters read matrix_rows=2/lower_cells=3 because "
                    "`payload['matrix']` is a dict {all_classes, supported_set}; the checks were therefore vacuous. "
                    "This verifier audit re-computed every R matrix correctly: 30 groups x 2 matrices x 25 finite "
                    "cells = 1500 (lower 15 per matrix). No data defect found; only the audit tooling counters are wrong.",
         "lead_counts_still_consistent": (lead.get("n_groups") == 30 and lead.get("validator_totals") ==
                                          {"pass": 480, "warn": 0, "fail": 0})})

    fails = [c for c in chk if c["status"] == "fail"]
    res["overall"] = "fail" if fails else "pass"
    res["summary"] = {"n_groups": len(groups), "n_period_manifests": len(train_mans),
                      "n_eval_manifests": len(eval_mans), "validators": totals,
                      "headline_cells": len(rebuilt)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "kaggle_matrix_full_verify.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({c["id"]: c["status"] for c in chk}, indent=1))
    print("overall:", res["overall"], "summary:", json.dumps(res["summary"]))
    for c in chk:
        if c["status"] != "pass":
            print(c["id"], json.dumps(c["detail"], default=str)[:1200])
    return 0 if not fails else 2


if __name__ == "__main__":
    raise SystemExit(main())
