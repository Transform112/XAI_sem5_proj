"""T40 — verifier audit of the corrected-S7 cross-checkpoint Kaggle lane (D-071) and the
T39 smoke evidence, plus code/status audit of the noise stage and S8 readiness.

Pins HEAD 9ca4681; verifies canonical first-seed anchors (cap/strata/D-063 context),
per-group isolation, t>=j job plan, cache keys/dedupe/scope filter, pair counts,
explanation-vs-checkpoint provenance, strict s5/s6/s7, test suite, and the noise-stage
code path (smoke not yet run). Writes results/text/verification/data_audit/t40_corrected_s7_audit.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(r"C:\Users\iusearchbtw\Desktop\XAI_proj")
PY = str(REPO / ".venv" / "Scripts" / "python.exe")
ROOT = REPO / "incoming_rtx5060" / "kaggle_runs" / "t39-explain-cross-smoke3-90e7635"
UP = ROOT / "unpacked"
OUT = REPO / "verification" / "data_audit"
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}
HEAD = "9ca4681"
SNAP = "90e7635"
RUN_META = ROOT / "explain_meta.json"
EXPECTED_GROUPS = {"ft_egraphsage_seed42", "ewc_egraphsage_seed42"}

sys.path.insert(0, str(REPO))


def rd(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8-sig"))


def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([r"C:\Program Files\Git\bin\git.exe", "-C", str(REPO), *args],
                          capture_output=True, text=True)


def main() -> int:
    from src.explain.cache import cache_key

    res: dict = {"checks": []}
    chk = res["checks"]

    def add(i, status, detail):
        chk.append({"id": i, "status": status, "detail": detail})

    meta = rd(RUN_META)
    inner = meta.get("inner") or {}
    pair_evidence = rd(ROOT / "crosscheck_pairs.json")

    # ---------------- A. pinning / executed-code provenance
    s7_paths = ["src/anchors", "src/orchestration/explain_plan.py", "src/scripts/build_anchors.py",
                "src/scripts/run_explain.py", "src/scripts/run_anchors_explain.py", "src/explain",
                "src/analysis/metrics_cl.py"]
    s7_diff = git("diff", "--name-only", f"{SNAP}..HEAD", "--", *s7_paths).stdout.strip().splitlines()
    suite = {}
    for suite_name in ("t40_suite.json", "t40_pin.json"):
        if (OUT / suite_name).is_file():
            suite = rd(OUT / suite_name)
            break
    add("PIN.snapshot_and_head",
        "pass" if (not s7_diff and str(inner.get("input_verification", {}).get("commit")) == SNAP
                   and str(inner.get("input_verification", {}).get("dataset_version")) == "3") else "fail",
        {"head": HEAD, "executed_snapshot": SNAP,
         "run_metadata": {"kernel_ref": meta.get("kernel_ref"), "kernel_status": meta.get("kernel_status"),
                          "stage": meta.get("stage"), "dataset_version": meta.get("dataset_version"),
                          "commit": meta.get("commit"), "inner_wall_clock_s": meta.get("inner_wall_clock_s"),
                          "local_wall_clock_s": meta.get("local_wall_clock_s")},
         "s7_code_diff_snapshot_to_head": s7_diff,
         "why": "the corrected-S7 orchestration lives in the Kaggle kernel/template (committed at HEAD); the executed "
                "dataset code snapshot is 90e7635 and all S7-relevant runtime paths are byte-identical between the "
                "snapshot and HEAD 9ca4681 -> executed S7 code == committed S7 code",
         "suite": suite})

    # ---------------- B. canonical anchors: first seed, caps/strata, D-063 contexts, isolation
    can = UP / "results" / "anchors" / "canonical"
    can_man = rd(can / "anchors_manifest.json")
    cfg = can_man.get("anchor_config") or {}
    ckpt_map = can_man.get("checkpoints") or {}
    source_ok = all(("/ft/egraphsage/seed42/" in str(v.get("path"))) for v in ckpt_map.values())
    periods = []
    all_context_ok = True
    target_ok = target_total = 0
    host_ok = True
    anchor_ids: dict[int, list[int]] = {}
    ctx_sizes = []
    hashes = {}
    for period in range(1, 6):
        p = can / f"period_{period}.pt"
        payload = torch.load(p, map_location="cpu", weights_only=False)
        anchors = payload.get("anchors") or []
        rec = can_man.get("per_period", {}).get(str(period)) or {}
        strata = dict(rec.get("strata") or {})
        ids = [int(a["anchor_id"]) for a in anchors]
        anchor_ids[period] = ids
        n_edges = [int(a.get("n_sub_edges") or len(a.get("sub_edge_ids") or [])) for a in anchors]
        ctx_sizes.extend(n_edges)
        for a in anchors:
            ids_t = [int(x) for x in a["sub_edge_ids"].tolist()]
            tgt = int(a["target_edge"].item())
            target_total += 1
            if tgt in ids_t and ids_t.index(tgt) == int(a["target_local_index"].item()):
                target_ok += 1
            for field in ("host_src_global", "host_dst_global", "sub_node_global_ids"):
                if a.get(field) is None:
                    host_ok = False
        n_sel = int(rec.get("n_selected") or len(anchors))
        cap = int(cfg.get("max_per_period", 0))
        class_cap = int(cfg.get("max_per_class", 0))
        non_rare_ok = all(int(v) <= class_cap for k, v in strata.items() if k != "rare")
        strata_sum_ok = sum(int(v) for v in strata.values()) == n_sel
        rare_classes = set(int(x) for x in (rec.get("rare_classes") or []))
        cc = {int(k): int(v) for k, v in (rec.get("class_counts") or {}).items()}
        expected_rare = {cls for cls, n in cc.items() if n < int(cfg.get("rare_min_eligible", 10))}
        rare_rule_ok = expected_rare == rare_classes or (not expected_rare and not rare_classes)
        periods.append({"period": period, "n_selected": n_sel, "n_eligible": rec.get("n_eligible"),
                        "strata": strata, "rare_classes": sorted(rare_classes),
                        "per_class_cap_ok": non_rare_ok, "strata_sum_ok": strata_sum_ok,
                        "max_per_period_ok": n_sel <= cap, "rare_rule_ok": rare_rule_ok,
                        "context_edges": {"min": min(n_edges), "max": max(n_edges)},
                        "anchor_ids": ids})
        all_context_ok = all_context_ok and max(n_edges) <= int(cfg.get("max_subgraph_edges", 2000))
        hashes[period] = sha_file(p)
    anchors_ok = (source_ok and all_context_ok and target_ok == target_total and host_ok
                  and all(p["per_class_cap_ok"] and p["max_per_period_ok"] and p["rare_rule_ok"]
                          for p in periods))
    strata_defects = [{"period": p["period"], "reported_strata_sum": sum(int(v) for v in p["strata"].values()),
                       "n_selected": p["n_selected"], "reported_strata": p["strata"]} for p in periods
                      if not p["strata_sum_ok"]]
    add("S6.canonical_anchors", "pass" if anchors_ok else "fail",
        {"source_group_ft_egraphsage_seed42": source_ok,
         "anchor_config": cfg,
         "note_caps": f"smoke override: max_per_class={cfg.get('max_per_class')}, max_per_period={cfg.get('max_per_period')} "
                      f"(production D-071 cap 250/period must be supplied by the full-run runctl; enforced by config override)",
         "periods": periods, "context_edges_all": {"min": min(ctx_sizes), "max": max(ctx_sizes),
                                                   "cap": cfg.get("max_subgraph_edges")},
         "target_kept_and_local_correct": f"{target_ok}/{target_total}",
         "production_cap_note": "configs/eval.yaml still defaults max_per_period=1000; the D-071 250/period cap is applied only by the "
                                "runctl override in the Kaggle kernel. The full-run runctl must set max_per_class=100, "
                                "max_per_period=250 and the full manifests must be checked for n_selected<=250.",
         "host_id_fields_present_for_S8": host_ok,
         "canonical_anchor_sha256": {str(k): v for k, v in hashes.items()}})

    # manifests refreshed + group isolation
    def _anchors_group_dir(group: str) -> Path:
        for candidate in (UP / "results" / "anchors" / "groups" / group,
                          UP / "results" / "anchors" / group):
            if candidate.is_dir():
                return candidate
        return UP / "results" / "anchors" / "groups" / group

    def _flat_sha(path: Path) -> str | None:
        if not path.is_file():
            return None
        payload = rd(path)
        if "sha256" in payload:
            return payload["sha256"]
        entry = (payload.get("anchor_files") or {}).get(str(payload.get("period"))) or {}
        return entry.get("sha256")
    manifest_problems = []
    for period in range(1, 6):
        actual = hashes[period]
        agg = ((can_man.get("periods") or {}).get(str(period)) or {}).get("sha256")
        per = _flat_sha(can / f"anchors_manifest_period{period}.json")
        root = _flat_sha(UP / "results" / "anchors" / f"anchors_manifest_period{period}.json")
        if not (actual == agg == per == root):
            manifest_problems.append({"period": period, "actual": actual, "canonical_aggregate": agg,
                                      "canonical_per_period": per, "root_per_period": root})
    group_problems = []
    for group in sorted(EXPECTED_GROUPS):
        gdir = _anchors_group_dir(group)
        ref = rd(gdir / "anchor_ref.json")
        for period in range(1, 6):
            copy = gdir / f"period_{period}.pt"
            copy_hash = sha_file(copy) if copy.is_file() else None
            ref_hash = ((ref.get("periods") or {}).get(str(period)) or {}).get("sha256")
            if copy_hash != hashes[period] or ref_hash != hashes[period]:
                group_problems.append({"group": group, "period": period, "copy": copy_hash,
                                       "ref": ref_hash, "canonical": hashes[period]})
    add("S6.strata_reporting", "fail" if strata_defects else "pass",
        {"defects": strata_defects,
         "actual_selected_labels_by_period": {str(period): [int(a["label"]) for a in torch.load(can / f"period_{period}.pt", map_location="cpu", weights_only=False)["anchors"]]
                                              for period in range(1, 6)},
         "root_cause": "src/anchors/build.py: when len(selected) > max_per_period, the per-stratum cap is applied only to a "
                       "local `capped` list; the `strata` dict keeps pre-cap lists and the manifest reports those pre-cap lengths "
                       "(lines ~219-224 and ~270-271). With the smoke override max_per_period=3 the manifest reports 12 selected "
                       "across strata while only 3 anchors are stored.",
         "production_impact": "under the full D-071 cap 250/period and max_per_class=100 the reported strata counts will sum to the "
                              "pre-cap per-class totals (typically >250) even though the stored anchor set is capped at 250; D-033 "
                              "strata provenance in the S6 manifests is therefore wrong for the full run until fixed.",
         "fix": "track the post-cap per-stratum selection (e.g. rebuild strata from `selected`) before writing n_selected/strata; "
                "regenerate the smoke/full manifests and add a validator check sum(strata)==n_selected."})

    add("S6.manifests_and_isolation", "pass" if (not manifest_problems and not group_problems
                                                 and len(EXPECTED_GROUPS) == 2) else "fail",
        {"per_period_manifest_refreshed": not manifest_problems, "problems": manifest_problems,
         "group_isolation": not group_problems, "group_problems": group_problems,
         "group_payload_copies": all(rd(_anchors_group_dir(g) / "anchor_ref.json").get("payload_copies")
                                     for g in EXPECTED_GROUPS),
         "note": "canonical payloads are copied to root + per-group dirs; group dirs never overwrite canonical (D-071 fix)"})

    # ---------------- C. job plan / pair counts from kernel steps + index + cache metadata
    steps = inner.get("steps") or {}
    jobs = []
    for name in steps:
        m = re.match(r"explain_(.+)_j(\d+)_ckpt(\d+)$", name)
        if not m:
            continue
        group, j, t = m.group(1), int(m.group(2)), int(m.group(3))
        step = steps[name]
        cmd = [str(x) for x in (step.get("command") or [])]
        jobs.append({"step": name, "group": group, "anchor_period": j, "checkpoint_period": t,
                     "returncode": step.get("returncode"), "duration_s": step.get("duration_s"),
                     "cmd": cmd})
    expected_jobs = {(g, j, t) for g in EXPECTED_GROUPS for j in range(1, 6) for t in range(j, 6)}
    got_jobs = {(x["group"], x["anchor_period"], x["checkpoint_period"]) for x in jobs}
    cmd_ok = True
    for x in jobs:
        cmd = x["cmd"]
        def val(flag):
            return cmd[cmd.index(flag) + 1] if flag in cmd else None
        if val("--period") != str(x["anchor_period"]) or not str(val("--checkpoint") or "").endswith(f"M_{x['checkpoint_period']}.pt") \
                or not str(val("--anchors") or "").endswith(f"anchors/canonical/period_{x['anchor_period']}.pt"):
            cmd_ok = False
    forward = [x["step"] for x in jobs if x["checkpoint_period"] < x["anchor_period"]]
    add("S7.job_plan", "pass" if (got_jobs == expected_jobs and cmd_ok and not forward
                                 and all(x["returncode"] == 0 for x in jobs)) else "fail",
        {"n_jobs": len(jobs), "expected_jobs": len(expected_jobs),
         "job_triples_match": got_jobs == expected_jobs, "commands_correct": cmd_ok,
         "forward_jobs": forward,
         "jobs": [{k: x[k] for k in ("step", "anchor_period", "checkpoint_period", "returncode", "duration_s")}
                  for x in sorted(jobs, key=lambda y: y["step"])]})

    idx = list(csv.DictReader((UP / "results" / "tables" / "explain_index.csv").open(encoding="utf-8-sig")))
    op_idx = list(csv.DictReader((UP / "results" / "explanations" / "index.csv").open(encoding="utf-8-sig")))
    cache_meta = {}
    for row in idx:
        p = UP / "results" / "explanations" / f"{row['key']}.json"
        cache_meta[row["key"]] = rd(p) if p.is_file() else None
    sha_to_period = {}
    for gk, info in ((inner.get("explain") or {}).get("checkpoints") or {}).items():
        sha_to_period[str(info.get("sha256"))] = int(re.search(r"M_(\d+)\.pt", str(gk)).group(1))
    rows_detail = []
    pair_keys = {}
    bad_meta = []
    for row in idx:
        meta_j = cache_meta.get(row["key"]) or {}
        anchor_id = int(row["anchor_id"])
        period = next((j for j, ids in anchor_ids.items() if anchor_id in ids), None)
        cp = meta_j.get("checkpoint_period")
        cp_legacy = meta_j.get("checkpoint_period_legacy")
        group = meta_j.get("strategy"), meta_j.get("model")
        pair_key = (f"{meta_j.get('strategy')}_{meta_j.get('model')}_seed{meta_j.get('seed')}",
                    period, int(cp) if cp is not None else None)
        pair_keys.setdefault(pair_key, set()).add((row["explainer"], row["k"]))
        rows_detail.append({"key": row["key"], "strategy": meta_j.get("strategy"), "model": meta_j.get("model"),
                            "seed": meta_j.get("seed"), "anchor_id": anchor_id, "anchor_period": period,
                            "checkpoint_period": cp, "checkpoint_period_legacy": cp_legacy,
                            "explainer": row["explainer"], "k": row["k"],
                            "checkpoint_sha": row["checkpoint_sha"], "config_sha": row["config_sha"],
                            "anchor_period_meta": meta_j.get("anchor_period"),
                            "class": meta_j.get("class"), "label": meta_j.get("label"),
                            "runtime_s": float(row["runtime_s"])})
        expected_cp = sha_to_period.get(row["checkpoint_sha"])
        if (period is None or cp is None or expected_cp is None or int(cp) != expected_cp
                or int(cp) < period or meta_j.get("anchor_period") != period or group[0] is None):
            bad_meta.append({"key": row["key"], "expected_checkpoint_period": expected_cp,
                             "meta": {k: meta_j.get(k) for k in ("anchor_period", "checkpoint_period",
                                                                 "checkpoint_period_legacy", "strategy", "model", "seed")}})
    # expected row universe: group x j x canonical anchors(j) x t>=j x 3 explainer rows
    allowed_groups = {g for g in EXPECTED_GROUPS}
    rows_by_key = {(x["strategy"], x["model"], x["seed"], x["anchor_id"], x["anchor_period"], x["checkpoint_period"],
                    x["explainer"], x["k"]) for x in rows_detail}
    expected_rows = set()
    for group in allowed_groups:
        strategy, model = group.split("_")[:2]
        for j in range(1, 6):
            for aid in anchor_ids[j]:
                for t in range(j, 6):
                    for explainer, k in (("ig", "5"), ("ig", "10"), ("gnn_explainer", "0")):
                        expected_rows.add((strategy, model, 42, aid, j, t, explainer, k))
    extra = [k for k in rows_detail
             if f"{k['strategy']}_{k['model']}_seed{k['seed']}" not in allowed_groups]
    cross_pairs = {(f"{x['strategy']}_{x['model']}_seed{x['seed']}", x["anchor_id"], x["anchor_period"],
                    x["checkpoint_period"]) for x in rows_detail
                   if x["checkpoint_period"] > x["anchor_period"]}
    diagonal_pairs = {(f"{x['strategy']}_{x['model']}_seed{x['seed']}", x["anchor_id"], x["anchor_period"])
                      for x in rows_detail if x["checkpoint_period"] == x["anchor_period"]}
    cross_rows = sum(1 for x in rows_detail if x["checkpoint_period"] > x["anchor_period"])
    diag_rows = sum(1 for x in rows_detail if x["checkpoint_period"] == x["anchor_period"])
    forward_rows = sum(1 for x in rows_detail if x["checkpoint_period"] < x["anchor_period"])
    canonical_keys = {str(r["key"]) for r in idx}
    operational_keys = {str(r["key"]) for r in op_idx}
    add("S7.pairs_and_index", "pass" if (rows_by_key == expected_rows and not bad_meta and not extra
                                         and cross_rows == 180 and forward_rows == 0 and diag_rows == 90
                                         and len(cross_pairs) == 60 and len(diagonal_pairs) == 30
                                         and canonical_keys == operational_keys
                                         and len(canonical_keys) == 270 and len(operational_keys) == 270) else "fail",
        {"index_rows": len(idx), "operational_rows": len(op_idx),
         "unique_keys": len(canonical_keys), "operational_unique_keys": len(operational_keys),
         "canonical_equals_operational_keys": canonical_keys == operational_keys,
         "pair_evidence_claims": {"cross_checkpoint_pairs": pair_evidence.get("cross_checkpoint_pairs"),
                                  "forward_pairs": pair_evidence.get("forward_pairs"),
                                  "index_rows": pair_evidence.get("index_rows"),
                                  "job_count": pair_evidence.get("job_count"),
                                  "pairs_per_group": pair_evidence.get("pairs_per_group")},
         "row_counts": {"total": len(idx), "cross_rows": cross_rows, "diagonal_rows": diag_rows,
                        "forward_rows": forward_rows},
         "pair_counts_definition": {"pairs_are_group_anchor_checkpoint_triples": True,
                                    "cross_pairs_t_gt_j": len(cross_pairs),
                                    "diagonal_pairs_t_eq_j": len(diagonal_pairs),
                                    "total_pairs_t_ge_j": len(cross_pairs) + len(diagonal_pairs)},
         "expected_rows_universe": len(expected_rows), "got_rows_universe": len(rows_by_key),
         "rows_missing": len(expected_rows - rows_by_key), "rows_unexpected": len(rows_by_key - expected_rows),
         "metadata_problems": bad_meta[:10],
         "explainer_rows": {"ig": sum(1 for x in rows_detail if x["explainer"] == "ig"),
                            "gnn_explainer": sum(1 for x in rows_detail if x["explainer"] == "gnn_explainer")}})

    # cache keys, files, context consistency, fidelity/runtime
    key_ok = 0
    files_ok = 0
    gnn_meta_missing = 0
    ctx_by_anchor = {}
    ig_len = {}
    gnn_len = {}
    for row in rows_detail:
        meta_j = cache_meta.get(row["key"]) or {}
        if row["explainer"] == "gnn_explainer" and meta_j.get("config_sha") is None:
            gnn_meta_missing += 1
        recomputed = cache_key(checkpoint_sha=row["checkpoint_sha"], anchor_id=row["anchor_id"],
                               explainer=row["explainer"], seed=int(row["seed"]), k=int(row["k"]),
                               config_sha=row["config_sha"])
        if recomputed == row["key"]:
            key_ok += 1
        npz = UP / "results" / "explanations" / f'{row["key"]}.npz'
        js = UP / "results" / "explanations" / f'{row["key"]}.json'
        if npz.is_file() and js.is_file():
            files_ok += 1
            with np.load(npz) as z:
                arr = z["attributions"] if "attributions" in z else (z[z.files[0]] if z.files else None)
                if row["explainer"] == "ig":
                    ig_len.setdefault((row["anchor_id"], row["k"]), set()).add(int(np.asarray(arr).shape[-1]) if arr is not None else None)
                else:
                    gnn_len.setdefault((row["anchor_id"],), set()).add(int(np.asarray(arr).shape[-1]) if arr is not None else None)
        ctx_by_anchor.setdefault((row["anchor_id"], row["anchor_period"]), set()).add(row["checkpoint_period"])
    ig_context_ok = all(len(v) == 1 and 50 in v for v in ig_len.values())
    gnn_context_ok = all(len(v) == 1 for v in gnn_len.values())
    runtime = np.array([x["runtime_s"] for x in rows_detail], dtype=float)
    meta_summary = inner.get("explain") or {}
    runtime_stats = {"total": float(runtime.sum()), "median": float(np.median(runtime)),
                     "p95": float(np.percentile(runtime, 95)),
                     "ig": float(runtime[[i for i, x in enumerate(rows_detail) if x["explainer"] == "ig"]].sum()),
                     "gnn": float(runtime[[i for i, x in enumerate(rows_detail) if x["explainer"] == "gnn_explainer"]].sum())}
    exp = meta_summary.get("explain_index") or {}
    stats_ok = (abs(runtime_stats["total"] - exp.get("runtime_total_s", -1)) < 0.2
                and abs(runtime_stats["median"] - exp.get("runtime_median_s", -1)) < 0.01
                and abs(runtime_stats["p95"] - exp.get("runtime_p95_s", -1)) < 0.01)
    pair_rate = runtime_stats["total"] / max(len(cross_pairs) + len(diagonal_pairs), 1)
    add("S7.cache_and_context", "pass" if (key_ok == len(rows_detail) and files_ok == len(rows_detail)
                                           and ig_context_ok and gnn_context_ok and stats_ok) else "fail",
        {"keys_recomputed": f"{key_ok}/{len(rows_detail)}", "cache_files_present": f"{files_ok}/{len(rows_detail)}",
         "ig_attribution_len_50_all_anchors": ig_context_ok, "gnn_attribution_len_consistent_per_anchor": gnn_context_ok,
         "context_note": "explanations always use the stored canonical period-j anchor payload (context never re-sampled); "
                         "IG vectors are fixed at 50 edge features and GNN masks at the stored bounded-context edge count",
         "runtime_recomputed": runtime_stats, "run_report_runtime": {k: exp.get(k) for k in
                                                                    ("runtime_total_s", "runtime_median_s", "runtime_p95_s",
                                                                     "runtime_by_explainer_s")},
         "anchor_pairs_per_second_equivalent": 1.0 / pair_rate if pair_rate else None,
         "explanation_seconds_per_anchor_pair": pair_rate,
         "lead_reported_seconds_per_anchor_pair": 1.046,
         "fidelity_fields_valid": all(np.isfinite(float(r.get("fidelity_sufficiency", "nan"))) for r in idx)})

    # ---------------- D. strict validators re-run locally
    def stage(name):
        p = subprocess.run([PY, "-m", "src.analysis.validators", "--stage", name, "--strict",
                            "--results-tables", str(UP / "results" / "tables"), "--json"],
                           cwd=str(REPO), capture_output=True, text=True, env=ENV)
        payload = {}
        try:
            payload = json.loads(p.stdout[p.stdout.find("{"):])
        except Exception:
            pass
        return {"returncode": p.returncode, "overall": payload.get("overall"),
                "counts": payload.get("counts"), "results_tables": str(UP / "results" / "tables")}
    s5, s6, s7 = stage("s5"), stage("s6"), stage("s7")
    ok_valid = (s5.get("overall") == "pass" and (s5.get("counts") or {}).get("pass") == 16
                and s6.get("overall") == "pass" and (s6.get("counts") or {}).get("pass") == 28
                and s7.get("overall") == "pass" and (s7.get("counts") or {}).get("pass") == 48)
    add("VALIDATORS.s5_s6_s7", "pass" if ok_valid else "fail",
        {"s5": s5, "s6": s6, "s7": s7,
         "note": "local re-run on downloaded smoke artifacts; strict s5/s6/s7 expected 16/28/48 pass (matches RunReport)"})

    # ---------------- E. noise stage: evidence + code path audit
    noise_dirs = [p.relative_to(UP).as_posix() for p in (UP / "results").glob("*noise*")]
    kernel_src = (REPO / "kaggle_lane" / "kernels" / "_template.py").read_text(encoding="utf-8", errors="replace")
    noise_script = (REPO / "scripts" / "build_noise_floor.py").is_file()
    noise_tests = git("grep", "-l", "noise_floor", "HEAD", "--", "tests", "src/src/analysis/tests").stdout.strip().splitlines()
    noise_code = {
        "kernel_noise_stage_present": "def run_noise_floor_stage" in kernel_src,
        "noise_script_present": noise_script,
        "noise_tests": noise_tests,
        "noise_evidence_dirs_in_smoke": noise_dirs,
        "null_model_recipe_finding": "src/scripts/build_noise_floor.py produce_model_null uses TrainSettings(batching='full_batch') "
                                     "(diagnostic mode) with fixed max_epochs=10/patience=10, while canonical training is "
                                     "graph_batch (D-059); the R=5 null reseeds should mirror the FT recipe (graph_batch + early "
                                     "stopping) or the null distribution is not comparable. Fix/confirm before the noise stage; "
                                     "noise-specific tests currently pass 16/16.",
        "checks_expected": ["reference group only (ft/egraphsage/seed42)",
                            "class-stratified 100 anchors/period subset",
                            "R=5 null model reseeds (FT same-period) + R=10 explainer repeats",
                            "per-(explainer,class) q95 with <20-pair global fallback pooling",
                            "combined null manifest + per-class/period keys"],
    }
    add("S8.noise_stage", "warn" if not noise_dirs else "pass",
        {**noise_code, "note": "T39 smoke did NOT exercise the noise-floor stage (no explanations_noise artifacts); the "
                               "kernel/script code path exists, so a dedicated noise smoke (reference group only) is "
                               "required before S8/S9 noise-floor claims. T40 go/no-go is conditional on that smoke."})

    # ---------------- F. S8 readiness: build sample real-drift features from cross pairs
    sha_by_group_t = {}
    for gk, info in ((inner.get("explain") or {}).get("checkpoints") or {}).items():
        group_key = str(gk).rsplit("/M_", 1)[0]
        period = int(re.search(r"M_(\d+)\.pt", str(gk)).group(1))
        sha_by_group_t[(group_key, period)] = str(info.get("sha256"))
    cache_rows_by_key = {}
    for row in rows_detail:
        group_key = f"{row['strategy']}_{row['model']}_seed{row['seed']}"
        cache_rows_by_key.setdefault((group_key, int(row["anchor_id"]), row["checkpoint_period"], row["explainer"], row["k"]),
                                     row["key"])
    drift_rows = []
    for group in sorted(EXPECTED_GROUPS):
        strategy, model = group.split("_")[:2]
        for j in range(1, 6):
            for aid in anchor_ids[j][:1]:
                base_key = cache_rows_by_key.get((group, int(aid), j, "ig", "5"))
                if base_key is None:
                    continue
                with np.load(UP / "results" / "explanations" / f"{base_key}.npz") as z:
                    base = np.asarray(z["attributions"]).ravel()
                base_top = set(np.argsort(-np.abs(base))[:5].tolist())
                for t in range(j + 1, 6):
                    if sha_by_group_t.get((group, t)) is None:
                        continue
                    cur_key = None
                    for row in rows_detail:
                        if (row["strategy"] == strategy and row["model"] == model and row["seed"] == 42
                                and int(row["anchor_id"]) == int(aid) and row["checkpoint_period"] == t
                                and row["explainer"] == "ig" and row["k"] == "5"):
                            cur_key = row["key"]
                            break
                    if cur_key is None:
                        continue
                    with np.load(UP / "results" / "explanations" / f"{cur_key}.npz") as z:
                        cur = np.asarray(z["attributions"]).ravel()
                    cur_top = set(np.argsort(-np.abs(cur))[:5].tolist())
                    union = base_top | cur_top
                    jac = len(base_top & cur_top) / len(union) if union else None
                    class_value = next((r["class"] for r in rows_detail if int(r["anchor_id"]) == int(aid)), None)
                    drift_rows.append({"group": group, "anchor_id": int(aid), "anchor_period": j,
                                       "checkpoint_period": t, "top5_intersection": len(base_top & cur_top),
                                       "ig_drift_1_minus_jaccard": (1 - jac) if jac is not None else None,
                                       "class": class_value})
    drift_ready = len(drift_rows) > 0 and all(r["ig_drift_1_minus_jaccard"] is not None for r in drift_rows)
    add("S8.readiness", "pass" if drift_ready else "warn",
        {"n_cross_pairs_probed": len(drift_rows),
         "drift_frame_sample": drift_rows[:6],
         "drift_1_minus_jaccard_values": [round(r["ig_drift_1_minus_jaccard"], 4) for r in drift_rows[:16]],
         "ingredients": {"canonical period-j anchors": True,
                         "host_ids": host_ok,
                         "checkpoint_period labels normalized": not bad_meta,
                         "class_period_keys": sorted({str((r["group"], r["class"], r["anchor_period"], r["checkpoint_period"]))
                                                      for r in drift_rows})[:5],
                         "retention_table": (UP / "results" / "tables" / "anchor_retention.csv").is_file(),
                         "noise_floor_thresholds": "missing (noise stage not run in T39 smoke)"},
         "note": "real-drift frame is computable for the smoke scale (non-empty cross-checkpoint pairs with class/period "
                 "keys); noise-floor thresholds (R=5 null reseeds / R=10 explainer repeats) are the only missing S8 input"})

    # ---------------- G. T38 core-pipeline status (board task-37) / commit audit
    core = {
        "run_anchors_explain_flags": subprocess.run([PY, "src/scripts/run_anchors_explain.py", "--help"],
                                                    cwd=str(REPO), capture_output=True, text=True, env=ENV).stdout,
    }
    flags_ok = all(f in core["run_anchors_explain_flags"] for f in
                   ("--canonical-anchors", "--checkpoint-periods", "--groups", "--job-offset", "--dedupe-index"))
    add("T38.core_pipeline", "warn" if not flags_ok else "pass",
        {"core_cli_flags_present": flags_ok,
         "note": "the corrected-S7 Kaggle lane is implemented in the kernel template (and exercised by T39 smoke); the "
                 "core-pipeline T38 flags on src/scripts/run_anchors_explain.py are still pending (board task-37). This does "
                 "not block the Kaggle lane but is required for laptop-level reproducibility of the same plan."})

    fails = [c for c in chk if c["status"] == "fail"]
    res["overall"] = "fail" if fails else "pass"
    res["summary"] = {"run_id": meta.get("run_id"), "kernel": meta.get("kernel_ref"),
                      "n_jobs": len(jobs), "index_rows": len(idx), "cross_pairs": len(cross_pairs),
                      "diagonal_rows": diag_rows, "validators": {"s5": (s5.get("counts") or {}),
                                                                 "s6": (s6.get("counts") or {}),
                                                                 "s7": (s7.get("counts") or {})}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t40_corrected_s7_audit.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({c["id"]: c["status"] for c in chk}, indent=1))
    print("overall:", res["overall"], "summary:", json.dumps(res["summary"]))
    for c in chk:
        if c["status"] != "pass":
            print(c["id"], json.dumps(c["detail"], default=str)[:900])
    return 0 if not fails else 2


if __name__ == "__main__":
    raise SystemExit(main())
