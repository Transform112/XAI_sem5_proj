"""T42 verifier audit: fixed replay + EWC-lambda100 Kaggle reruns, corrected headline table.

Reads the downloaded rerun outputs, re-runs strict per-group S5 validators, checks lane/gate/
lifecycle/code fingerprint, recomputes R/dual-macro/forgetting for the rerun groups, verifies
resume integrity of the untouched groups, rebuilds the 12-cell corrected headline table from
per-seed rows, checks supersession/no-legacy-mixing, and writes
results/text/verification/data_audit/t42_rerun_audit.json. No repo code/results modified.
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
OUT = REPO / "verification" / "data_audit"
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}
RUNS = {
    "ft": REPO / "incoming_rtx5060/kaggle_runs/t37-matrix-full-90e7635",
    "replay": REPO / "incoming_rtx5060/kaggle_runs/t42-replay-rerun-9ca4681",
    "ewc": REPO / "incoming_rtx5060/kaggle_runs/t42-ewc-lambda100-649109a",
}
HEAD = REPO / "incoming_rtx5060/kaggle_runs/t42_headline"
FP_TO_COMMIT = {
    "d907aa44d095690a24f811dd1a446da6bebcdaf3781f0acb003d6c9fe5bc9918": "90e7635",
    "1878a4cd90c51f1d79dce385002cb7c03ddf63c0077e8d02f4faaec6db82df2c": "9ca4681",
    "0f13f560aa3e9a29a06406d051fd7fac78686e38234d11201f18b767ddb84e8b": "649109a",
}
LEAD = {
    ("ft", "egraphsage", "all_classes"): (0.4393, 0.0070, 0.0443, 0.0231),
    ("ft", "egraphsage", "supported_set"): (0.5651, 0.0089, 0.0231, 0.0102),
    ("ft", "gcn", "all_classes"): (0.4142, 0.0173, 0.0713, 0.0568),
    ("ft", "gcn", "supported_set"): (0.5413, 0.0196, 0.0568, 0.0256),
    ("replay", "egraphsage", "all_classes"): (0.4056, 0.0239, 0.0348, 0.0285),
    ("replay", "egraphsage", "supported_set"): (0.5233, 0.0392, 0.0285, 0.0446),
    ("replay", "gcn", "all_classes"): (0.3903, 0.0248, 0.0557, 0.0583),
    ("replay", "gcn", "supported_set"): (0.5005, 0.0285, 0.0583, 0.0328),
    ("ewc", "egraphsage", "all_classes"): (0.4234, 0.0195, 0.0395, 0.0290),
    ("ewc", "egraphsage", "supported_set"): (0.5423, 0.0264, 0.0290, 0.0231),
    ("ewc", "gcn", "all_classes"): (0.4253, 0.0078, 0.0392, 0.0177),
    ("ewc", "gcn", "supported_set"): (0.5519, 0.0107, 0.0177, 0.0169),
}
LEAD_FORGET_ALL = {("ft", "egraphsage"): 0.0443, ("ft", "gcn"): 0.0713,
                   ("replay", "egraphsage"): 0.0348, ("replay", "gcn"): 0.0557,
                   ("ewc", "egraphsage"): 0.0395, ("ewc", "gcn"): 0.0392}

sys.path.insert(0, str(REPO))


def rd(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8-sig"))


def git(*args):
    return subprocess.run([r"C:\Program Files\Git\bin\git.exe", "-C", str(REPO), *args],
                          capture_output=True, text=True)


def rows_from_csv(p: Path):
    return list(csv.DictReader(p.open(encoding="utf-8-sig")))


def main() -> int:
    from analysis.metrics_cl import RMetricTensor

    res: dict = {"checks": []}
    chk = res["checks"]

    def add(i, status, detail):
        chk.append({"id": i, "status": status, "detail": detail})

    states = {k: rd(v / "unpacked/results/text/tables/matrix_state.json") for k, v in RUNS.items()}
    metas = {k: rd(v / "matrix_meta.json") for k, v in RUNS.items()}
    audit_jsons = {
        "ft": rd(RUNS["ft"] / "matrix_full_audit.json") if (RUNS["ft"] / "matrix_full_audit.json").is_file() else {},
        "replay": rd(RUNS["replay"] / "matrix_run_audit.json"),
        "ewc": rd(RUNS["ewc"] / "matrix_run_audit.json"),
    }

    # ---------------- A. run provenance / snapshot fingerprints / dataset versions
    fingerprint_by_strategy = {}
    for run_key, root in RUNS.items():
        ad = root / "unpacked/results/text/data_audit"
        fps = {}
        for strategy in ("ft", "replay", "ewc"):
            vals = set()
            for p in ad.glob(f"train_manifest_{strategy}_*_period1.json"):
                vals.add((rd(p).get("code") or {}).get("sha256"))
            if vals:
                fps[strategy] = sorted(vals)
        fingerprint_by_strategy[run_key] = fps
    add("PROV.fingerprints",
        "pass" if (fingerprint_by_strategy["replay"].get("replay") == [list(FP_TO_COMMIT)[1]]
                   and fingerprint_by_strategy["ewc"].get("ewc") == [list(FP_TO_COMMIT)[2]]
                   and fingerprint_by_strategy["replay"].get("ft") == [list(FP_TO_COMMIT)[0]]
                   and fingerprint_by_strategy["ewc"].get("ft") == [list(FP_TO_COMMIT)[0]]
                   and fingerprint_by_strategy["ewc"].get("replay") == [list(FP_TO_COMMIT)[1]]) else "fail",
        {"fingerprints_by_run_and_strategy": fingerprint_by_strategy,
         "fingerprint_to_commit": FP_TO_COMMIT,
         "dataset_versions": {k: metas[k].get("dataset_version") for k in RUNS},
         "dataset_commits": {k: metas[k].get("commit") for k in RUNS},
         "run_status": {k: {"inner_status": metas[k].get("inner_status") or (audit_jsons[k].get("inner_status")),
                            "kernel_status": metas[k].get("kernel_status"),
                            "kernel_ref": metas[k].get("kernel_ref")} for k in RUNS},
         "note": "FT rows from the original v3/90e7635 run; replay rerun on v4/9ca4681; EWC-lambda100 rerun on v6/649109a. "
                 "Matrix-relevant runtime paths are identical between 9ca4681 and 649109a; the 90e7635->9ca4681 deltas are "
                 "replay/EWC-specific fixes (see PROV.cross_snapshot_deltas)."})

    # cross-snapshot deltas for matrix-relevant paths
    paths = ["src/cl", "src/train", "src/eval", "src/models", "src/scripts/run_matrix.py", "src/scripts/run_eval.py",
             "configs/train.yaml", "configs/train_ewc_lambda100.yaml", "configs/eval.yaml"]
    d1 = git("diff", "--name-status", "90e7635..9ca4681", "--", *paths).stdout.strip().splitlines()
    d2 = git("diff", "--name-status", "9ca4681..649109a", "--", *paths).stdout.strip().splitlines()
    add("PROV.cross_snapshot_deltas",
        "pass" if d2 == [] else "warn",
        {"90e7635..9ca4681": d1, "9ca4681..649109a": d2,
         "note": "no matrix-relevant changes between the replay (9ca4681) and EWC (649109a) snapshots; the only matrix "
                 "deltas 90e7635->9ca4681 are the D-072 replay strategy/trainer fix, the EWC lambda overlay config and a "
                 "train.yaml comment/representation addition. FT rows remain the earlier signed run (D-070/D-072)."})

    # frozen / CSV identity from each run's graphs manifest + dataset meta
    frozen = {}
    for k, root in RUNS.items():
        gm = root / "unpacked/graphs_manifest.json"
        graphs = rd(gm) if gm.is_file() else {}
        inner = (metas[k].get("inner") or {})
        frozen[k] = {"dataset_version": metas[k].get("dataset_version"),
                     "commit": metas[k].get("commit"),
                     "frozen_split_hash": graphs.get("frozen_split_hash"),
                     "split_hash_matches_frozen": graphs.get("split_hash_matches_frozen"),
                     "input_verification": {kk: inner.get("input_verification", {}).get(kk) for kk in
                                            ("csv_sha256", "csv_bytes", "frozen_delivery", "code_snapshot_sha")}
                     if isinstance(inner.get("input_verification"), dict) else None}
    add("PROV.frozen_csv",
        "pass" if all(v.get("frozen_split_hash") == "010a80d955802a2f62dedf472a3274e1afee17fb4b21c57541fe428b8feb47ff"
                      and v.get("split_hash_matches_frozen") in (True, None) for v in frozen.values()) else "fail",
        {"by_run": frozen,
         "note": "all runs use the frozen C1 split hash 010a80d9...; dataset versions differ only in the code snapshot "
                 "(v3/v4/v6), raw CSV/frozen payload are unchanged."})

    # ---------------- B. per-group R/dual/forgetting for rerun groups + resume integrity
    def group_metrics(root: Path, group: str):
        md = root / "unpacked/results/text/tables/matrix" / group
        cm = rd(md / "cl_matrix.json")
        rows = rows_from_csv(md / "cl_metrics.csv")
        return cm, rows

    rerun_groups = {
        "replay": [f"replay_{m}_seed{s}" for m in ("egraphsage", "gcn") for s in range(42, 47)],
        "ewc": [f"ewc_{m}_seed{s}" for m in ("egraphsage", "gcn") for s in range(42, 47)],
    }
    problems = []
    group_recs = []
    total_cells = 0
    for strategy, groups in rerun_groups.items():
        root = RUNS[strategy]
        for group in groups:
            cm, rows = group_metrics(root, group)
            R_all = np.asarray(cm["matrix"]["all_classes"], dtype=float)
            R_sup = np.asarray(cm["matrix"]["supported_set"], dtype=float)
            lower_mask = np.tril(np.ones((5, 5), dtype=bool)) | np.eye(5, dtype=bool)
            finite_all = int(np.isfinite(R_all).sum())
            finite_sup = int(np.isfinite(R_sup).sum())
            lower_all = int((np.isfinite(R_all) & lower_mask).sum())
            lower_sup = int((np.isfinite(R_sup) & lower_mask).sum())
            total_cells += finite_all + finite_sup
            ref_all = RMetricTensor(R_all, n_periods=5); ref_all.validate()
            ref_sup = RMetricTensor(R_sup, n_periods=5); ref_sup.validate()
            claim_all = cm.get("cl_metrics_all_classes") or {}
            claim_sup = cm.get("cl_metrics_supported_set") or {}
            for label, ref, claim in (("all", ref_all, claim_all), ("supported", ref_sup, claim_sup)):
                for key in ("forgetting", "bwt", "final_average_macro_f1"):
                    val = {"forgetting": float(ref.forgetting()), "bwt": float(ref.bwt()),
                           "final_average_macro_f1": float(ref.final_macro_f1())}[key]
                    if abs(float(claim.get(key, np.nan)) - val) > 1e-9:
                        problems.append(f"{group}: {label} {key} claim={claim.get(key)} ref={val}")
            # dual macro checks all 25 cells
            dual = cm.get("dual_macro_f1_per_cell") or {}
            per = rd(root / "unpacked/results/text/tables/matrix" / group / "cl_per_class_f1.json").get("per_class_f1") or {}
            checked = 0
            for rk, cols in dual.items():
                for ck, cell in (cols or {}).items():
                    pc = (((per.get(rk) or {}).get(ck) or {}).get("per_class_f1")
                          or (cell or {}).get("per_class_f1") or {})
                    block = (cell or {}).get("dual_macro_f1") or {}
                    labels = block.get("supported_set_labels") or []
                    if len(pc) != 10 or len(labels) != 7:
                        problems.append(f"{group}: {rk},{ck} malformed dual/per-class")
                        continue
                    if abs(float(np.mean(list(pc.values()))) - float(block.get("macro_f1_all_classes", -1))) > 1e-9:
                        problems.append(f"{group}: {rk},{ck} all-class macro mismatch")
                    if abs(float(np.mean([float(pc[str(i)]) for i in labels])) - float(block.get("macro_f1_supported_set", -1))) > 1e-9:
                        problems.append(f"{group}: {rk},{ck} supported macro mismatch")
                    if int(sum((block.get("class_counts") or {}).values())) != int(block.get("n_edges", -1)):
                        problems.append(f"{group}: {rk},{ck} class_counts != n_edges")
                    checked += 1
            if checked != 25:
                problems.append(f"{group}: {checked}/25 dual cells")
            # EWC overlay evidence
            if strategy == "ewc":
                met = rd(root / f"unpacked/results/text/data_audit/train_ewc_{group.split('_', 1)[1]}_period1.json")
                r0 = met.get("result") or {}
                if float(r0.get("lambda", -1)) != 100.0:
                    problems.append(f"{group}: lambda={r0.get('lambda')} != 100.0")
            group_recs.append({"group": group, "strategy": strategy, "finite_cells": finite_all + finite_sup,
                               "lower_cells": lower_all + lower_sup, "dual_cells": checked})
    add("RERUN.r_science", "pass" if not problems and total_cells == 20 * 2 * 25 else "fail",
        {"n_rerun_groups": 20, "matrix_cells_finite": total_cells, "expected": 1000,
         "problems": problems[:20], "groups": group_recs})

    # resume integrity: untouched groups must be bit-identical to their source run
    def all_rows(root: Path):
        base = root / "unpacked/results/text/tables/matrix"
        out = {}
        for p in sorted(base.iterdir()):
            if p.is_dir():
                out[p.name] = {(r["metric"], r["strategy"], r["model"], r["seed"]):
                               {k: r[k] for k in ("final_average_macro_f1", "forgetting", "bwt")}
                               for r in rows_from_csv(p / "cl_metrics.csv")}
        return out
    ft_all = all_rows(RUNS["ft"]); replay_all = all_rows(RUNS["replay"]); ewc_all = all_rows(RUNS["ewc"])
    resume_problems = []
    for group in [g for g in replay_all if g.startswith("ft_")]:
        if replay_all[group] != ft_all.get(group):
            resume_problems.append(f"replay-run {group} != t37 ft")
    for group in [g for g in ewc_all if g.startswith("ft_")]:
        if ewc_all[group] != ft_all.get(group):
            resume_problems.append(f"ewc-run {group} != t37 ft")
    for group in [g for g in ewc_all if g.startswith("replay_")]:
        if ewc_all[group] != replay_all.get(group):
            resume_problems.append(f"ewc-run {group} != replay-run replay")
    # R matrices for resumed groups
    for group in [g for g in replay_all if g.startswith("ft_")]:
        a = rd(RUNS["ft"] / "unpacked/results/text/tables/matrix" / group / "cl_matrix.json")
        b = rd(RUNS["replay"] / "unpacked/results/text/tables/matrix" / group / "cl_matrix.json")
        if not np.allclose(np.asarray(a["matrix"]["all_classes"], float), np.asarray(b["matrix"]["all_classes"], float), equal_nan=True):
            resume_problems.append(f"replay-run FT R mismatch {group}")
    add("RERUN.resume_integrity", "pass" if not resume_problems else "fail",
        {"problems": resume_problems,
         "note": "untouched FT groups are bit-identical between the t37 run and the replay/EWC reruns; replay groups are "
                 "bit-identical between the replay rerun and the EWC rerun (resume integrity, no silent retraining)."})

    # ---------------- C. strict per-group S5 validators (all complete groups in both rerun runs)
    val_totals = {}
    val_problems = []
    for key, root in (("replay", RUNS["replay"]), ("ewc", RUNS["ewc"])):
        base = root / "unpacked/results/text/tables/matrix"
        totals = {"pass": 0, "warn": 0, "fail": 0}
        groups = []
        for p in sorted(base.iterdir()):
            if not p.is_dir():
                continue
            proc = subprocess.run([PY, "-m", "src.analysis.validators", "--stage", "s5", "--strict",
                                   "--results-tables", str(p), "--json"],
                                  cwd=str(REPO), capture_output=True, text=True, env=ENV)
            payload = {}
            try:
                payload = json.loads(proc.stdout[proc.stdout.find("{"):])
            except Exception:
                pass
            counts = payload.get("counts") or {}
            if not counts:
                continue
            for k in totals:
                totals[k] += int(counts.get(k, 0))
            if proc.returncode != 0 or counts.get("fail", 0) or counts.get("warn", 0):
                val_problems.append({"run": key, "group": p.name, "counts": counts, "rc": proc.returncode})
            groups.append(p.name)
        val_totals[key] = {"groups": len(groups), "totals": totals}
    add("RERUN.validators",
        "pass" if (not val_problems
                   and val_totals["replay"]["totals"] == {"pass": 320, "warn": 0, "fail": 0}
                   and val_totals["ewc"]["totals"] == {"pass": 480, "warn": 0, "fail": 0}) else "fail",
        {"by_run": val_totals, "problems": val_problems,
         "lead_evidence_checks": {"replay_audit_overall": audit_jsons["replay"].get("overall"),
                                  "ewc_audit_overall": audit_jsons["ewc"].get("overall")}})

    # ---------------- D. rerun manifests: lane/gate/status/fingerprint/lifecycle + EWC lambda
    man_problems = []
    man_recs = {"replay": [], "ewc": []}
    for strategy, groups in rerun_groups.items():
        root = RUNS[strategy]; ad = root / "unpacked/results/text/data_audit"
        expected_fp = {"replay": list(FP_TO_COMMIT)[1], "ewc": list(FP_TO_COMMIT)[2]}[strategy]
        for group in groups:
            strategy_name, model, seed_part = group.split("_")
            seed = int(seed_part.replace("seed", ""))
            for period in range(1, 6):
                p = ad / f"train_manifest_{strategy_name}_{model}_seed{seed}_period{period}.json"
                man = rd(p)
                met = rd(ad / f"train_{strategy_name}_{model}_seed{seed}_period{period}.json")
                t = (met.get("result") or {}).get("timing") or {}
                gate = man.get("gate") or {}
                dur = float(man.get("duration_s") or 0); ts = float(t.get("train_seconds") or 0)
                ok = (man.get("lane") == "kaggle" and man.get("lane_source") == "cli"
                      and bool(gate.get("real_lane")) and not bool(gate.get("allow_default_lane_env"))
                      and man.get("status") == "success"
                      and (man.get("code") or {}).get("sha256") == expected_fp
                      and str(man.get("started_at_utc")) < str(man.get("finished_at_utc"))
                      and dur >= ts * 0.9 and t.get("batching") == "graph_batch"
                      and len([k for k in (man.get("env") or {}) if str(k).startswith("XDRIFT")]) == 7)
                if not ok:
                    man_problems.append(f"{strategy}/{group}/period{period}")
                man_recs[strategy].append({"group": group, "period": period, "duration_s": dur,
                                           "epoch_seconds_mean": t.get("epoch_seconds_mean"),
                                           "max_memory_allocated_bytes": t.get("max_memory_allocated_bytes")})
            em = rd(ad / f"eval_manifest_{strategy_name}_{model}_seed{seed}.json")
            ex = em.get("extra") or {}; ev = ex.get("evaluation") or {}
            if not (em.get("lane") == "kaggle" and (em.get("gate") or {}).get("real_lane")
                    and em.get("status") == "success" and ev.get("mode") == "edge_batch"
                    and ex.get("checkpoints_present") == [1, 2, 3, 4, 5]):
                man_problems.append(f"{strategy}/{group}/eval")
    add("RERUN.manifests", "pass" if not man_problems else "fail",
        {"n_train_manifests_checked": sum(len(v) for v in man_recs.values()),
         "n_eval_manifests_checked": 20, "problems": man_problems[:20],
         "replay_epoch_seconds_p1": [r["epoch_seconds_mean"] for r in man_recs["replay"] if r["period"] == 1],
         "ewc_epoch_seconds_p1": [r["epoch_seconds_mean"] for r in man_recs["ewc"] if r["period"] == 1],
         "replay_memory_bytes_range": [min(r["max_memory_allocated_bytes"] for r in man_recs["replay"]),
                                       max(r["max_memory_allocated_bytes"] for r in man_recs["replay"])],
         "ewc_memory_bytes_range": [min(r["max_memory_allocated_bytes"] for r in man_recs["ewc"]),
                                    max(r["max_memory_allocated_bytes"] for r in man_recs["ewc"])]})

    # ---------------- E. corrected headline rebuild
    def per_seed_rows(run_key, strategy):
        base = RUNS[run_key] / "unpacked/results/text/tables/matrix"
        out = []
        for p in sorted(base.iterdir()):
            if p.is_dir() and p.name.startswith(strategy + "_"):
                for r in rows_from_csv(p / "cl_metrics.csv"):
                    out.append({"strategy": strategy, "model": r["model"], "seed": int(r["seed"]),
                                "metric": r["metric"],
                                "final": float(r["final_average_macro_f1"]),
                                "forgetting": float(r["forgetting"]), "bwt": float(r["bwt"]),
                                "group": p.name})
        return out
    rows = per_seed_rows("ft", "ft") + per_seed_rows("replay", "replay") + per_seed_rows("ewc", "ewc")
    table = {}
    for strategy in ("ft", "replay", "ewc"):
        for model in ("egraphsage", "gcn"):
            for metric in ("all_classes", "supported_set"):
                sel = [r for r in rows if r["strategy"] == strategy and r["model"] == model and r["metric"] == metric]
                if len(sel) != 5:
                    continue
                entry = {"n": len(sel)}
                for key in ("final", "forgetting", "bwt"):
                    vals = np.array([r[key] for r in sel], dtype=float)
                    entry[key] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1)),
                                  "values": [float(v) for v in vals]}
                table[(strategy, model, metric)] = entry
    csv_rows = rows_from_csv(HEAD / "headline_table_valid.csv")
    header = {(r["strategy"], r["model"], r["metric"]): r for r in csv_rows}
    csv_mismatch = []
    lead_mismatch = []
    for key, e in table.items():
        row = header.get(key)
        if row is None:
            csv_mismatch.append({"cell": key, "missing": True})
            continue
        if int(row["n_seeds"]) != 5 or abs(float(row["final_mean"]) - e["final"]["mean"]) > 5e-6 \
                or abs(float(row["final_sd"]) - e["final"]["sd"]) > 5e-6 \
                or abs(float(row["forgetting_mean"]) - e["forgetting"]["mean"]) > 5e-6 \
                or abs(float(row["forgetting_sd"]) - e["forgetting"]["sd"]) > 5e-6:
            csv_mismatch.append({"cell": key, "csv": [row["final_mean"], row["final_sd"],
                                                      row["forgetting_mean"], row["forgetting_sd"]],
                                 "rebuilt": [e["final"]["mean"], e["final"]["sd"],
                                             e["forgetting"]["mean"], e["forgetting"]["sd"]]})
        if key in LEAD:
            lm, ls, lf, lfs = LEAD[key]
            if (abs(e["final"]["mean"] - lm) > 1e-4 or abs(e["final"]["sd"] - ls) > 1e-4
                    or abs(e["forgetting"]["mean"] - lf) > 1e-4):
                lead_mismatch.append({"cell": key, "lead": [lm, ls, lf], "rebuilt":
                                      [e["final"]["mean"], e["final"]["sd"], e["forgetting"]["mean"]]})
    prov = rd(HEAD / "headline_provenance.json")
    expected_source = {"ft": "t37-matrix-full-90e7635", "replay": "t42-replay-rerun-9ca4681",
                       "ewc": "t42-ewc-lambda100-649109a"}
    source_ok = all(expected_source[k] in str(prov[k]["run"]) for k in expected_source)
    valid_ok = all(r["valid"] == "valid" for r in csv_rows)
    add("HEADLINE.rebuild",
        "pass" if (len(table) == 12 and not csv_mismatch and not lead_mismatch and source_ok and valid_ok) else "fail",
        {"cells": len(table), "n_rows_per_cell": {str(k): v["n"] for k, v in table.items()},
         "csv_mismatches": csv_mismatch, "lead_message_mismatches": lead_mismatch,
         "provenance_sources": prov, "provenance_source_ok": source_ok, "all_rows_valid": valid_ok,
         "headline_table": [{"strategy": k[0], "model": k[1], "metric": k[2],
                             "final_mean": round(v["final"]["mean"], 6), "final_sd": round(v["final"]["sd"], 6),
                             "forgetting_mean": round(v["forgetting"]["mean"], 6),
                             "forgetting_sd": round(v["forgetting"]["sd"], 6)}
                            for k, v in sorted(table.items())]})

    # superseded rows / no legacy mixing
    superseded = []
    for strategy, model in (("replay", "egraphsage"), ("replay", "gcn"), ("ewc", "egraphsage"), ("ewc", "gcn")):
        base = RUNS["ft"] / "unpacked/results/text/tables/matrix"
        p = base / f"{strategy}_{model}_seed42"
        if p.is_dir():
            old = rows_from_csv(p / "cl_metrics.csv")
            new = rows_from_csv(RUNS["replay" if strategy == "replay" else "ewc"] / "unpacked/results/text/tables/matrix" /
                                f"{strategy}_{model}_seed42" / "cl_metrics.csv")
            old_all = next(r for r in old if r["metric"] == "all_classes")
            new_all = next(r for r in new if r["metric"] == "all_classes")
            superseded.append({"strategy": strategy, "model": model,
                               "old_final": float(old_all["final_average_macro_f1"]),
                               "new_final": float(new_all["final_average_macro_f1"]),
                               "old_ignored_in_headline": float(old_all["final_average_macro_f1"]) != float(new_all["final_average_macro_f1"])})
    source_paths = {str(prov[k]["run"]) for k in prov}
    add("HEADLINE.supersession",
        "pass" if all(s["old_ignored_in_headline"] for s in superseded)
        and not any("matrix_canary" in p or "legacy" in p.lower() for p in source_paths) else "fail",
        {"superseded_old_rows": superseded, "headline_source_paths": sorted(source_paths),
         "note": "the headline table sources only the Kaggle canonical runs; old replay (D-072) and old EWC lambda=1000 "
                 "rows are excluded and explicitly superseded in the provenance file."})

    # ---------------- F. cross-check audit evidence files
    add("AUDIT.evidence_crosscheck", "pass" if (audit_jsons["replay"].get("overall") == "pass"
                                                 and audit_jsons["ewc"].get("overall") == "pass") else "fail",
        {"replay_matrix_run_audit": {"overall": audit_jsons["replay"].get("overall"),
                                     "group_count": audit_jsons["replay"].get("group_count"),
                                     "problems": audit_jsons["replay"].get("problems")},
         "ewc_matrix_run_audit": {"overall": audit_jsons["ewc"].get("overall"),
                                  "group_count": audit_jsons["ewc"].get("group_count"),
                                  "problems": audit_jsons["ewc"].get("problems")},
         "note": "verifier re-ran the same strict validators independently (see RERUN.validators); the engineer/Lead "
                 "audit JSONs are consistent."})

    fails = [c for c in chk if c["status"] == "fail"]
    res["overall"] = "fail" if fails else "pass"
    res["summary"] = {"rerun_groups": {"replay": 10, "ewc": 10},
                      "validators": val_totals,
                      "headline_cells": len(table),
                      "overall": res["overall"]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t42_rerun_audit.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({c["id"]: c["status"] for c in chk}, indent=1))
    print("overall:", res["overall"], "summary:", json.dumps(res["summary"]))
    for c in chk:
        if c["status"] != "pass":
            print(c["id"], json.dumps(c["detail"], default=str)[:1200])
    return 0 if not fails else 2


if __name__ == "__main__":
    raise SystemExit(main())
