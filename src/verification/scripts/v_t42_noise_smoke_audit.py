"""T44/F4 verifier audit: Kaggle noise-floor smoke (model_null R=5, explainer_repeat R=10,
combined null 15k rows) on dataset v6 / snapshot 649109a, reference ft/egraphsage/seed42.

Independent checks: run/kernel evidence, row counts/schema/keys, class+period metadata,
canonical post-cap anchor provenance (D-033/F1), D-063 contexts on noise subsets, IG-repeat
determinism, q95 pooling readiness through the authoritative analysis.noise_floor API,
strict s5/s6/s7 validators, and the F2 null-training recipe check (canonical graph_batch /
early stop vs the smoke config). Writes results/text/verification/data_audit/t42_noise_smoke_audit.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(r"C:\Users\iusearchbtw\Desktop\XAI_proj")
PY = str(REPO / ".venv" / "Scripts" / "python.exe")
RUN = REPO / "incoming_rtx5060/kaggle_runs/t42-noise-smoke-649109a"
UP = RUN / "unpacked"
RES = UP / "results"
OUT = REPO / "results/text/verification/data_audit"
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8"}
SNAPSHOT_COMMIT = "649109a"
EXPECTED_CANONICAL = {"1": 229, "2": 205, "3": 221, "4": 230, "5": 123}

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
    from src.analysis import noise_floor as nf

    res: dict = {"checks": []}
    chk = res["checks"]

    def add(i, status, detail):
        chk.append({"id": i, "status": status, "detail": detail})

    run_sum = rd(RUN / "kernel_output/run_summary.json")
    noise_meta = rd(RUN / "noise_meta.json")
    lead_val = {}
    _lv = OUT / "t42_noise_smoke/noise_smoke_validation.json"
    if _lv.is_file():
        try:
            lead_val = json.loads(_lv.read_text(encoding="utf-8-sig"), strict=False)
        except Exception:  # noqa: BLE001
            lead_val = {}
    noise = run_sum.get("noise_floor") or {}

    # ---------------- A. kernel / provenance / lane
    output_zip = RUN / "kernel_output/xdrift_run_output.zip"
    zip_sha = sha_file(output_zip) if output_zip.is_file() else None
    iv = run_sum.get("input_verification") or {}
    env_vars = ((run_sum.get("environment") or {}).get("env_vars") or {})
    add("NOISE.provenance",
        "pass" if (run_sum.get("status") == "success"
                   and run_sum.get("kernel_ref") == "mysteriousavailable/xdrift-noise"
                   and str(run_sum.get("runctl", {}).get("commit")) == SNAPSHOT_COMMIT
                   and int(run_sum.get("runctl", {}).get("dataset_version", -1)) == 6
                   and str(iv.get("commit")) == SNAPSHOT_COMMIT
                   and zip_sha == "575480ccda2f428cc1478c0ffc804cd0032398de07b97e1dafa8706bd8b0df11"
                   and env_vars.get("XDRIFT_LANE") == "kaggle") else "fail",
        {"run_id": run_sum.get("run_id"), "kernel_ref": run_sum.get("kernel_ref"),
         "status": run_sum.get("status"), "wall_clock_s": run_sum.get("wall_clock_s"),
         "gpu_hours": round(float(run_sum.get("wall_clock_s") or 0) / 3600, 2),
         "runctl": run_sum.get("runctl"), "input_verification": {k: iv.get(k) for k in
                                                                ("dataset_version", "code_snapshot_sha", "csv_sha256", "csv_bytes", "frozen_delivery")},
         "output_zip_sha256": zip_sha,
         "expected_zip_sha256": "575480ccda2f428cc1478c0ffc804cd0032398de07b97e1dafa8706bd8b0df11",
         "lane_env": env_vars.get("XDRIFT_LANE"),
         "stale_run_full_meta": {"kernel_status": noise_meta.get("kernel_status"),
                                 "inner_status": noise_meta.get("inner_status"),
                                 "note": "noise_meta.json was polled while the kernel still reported 'running'; the "
                                         "downloaded zip + kernel log record status=success (confirmed by the "
                                         "run_summary above). Non-blocking metadata staleness."}})

    # ---------------- B. counts / schema / keys
    def channel_rows(channel: str):
        return list(csv.DictReader((RES / "explanations_noise" / channel / "index.csv").open(encoding="utf-8-sig")))
    chans = {ch: channel_rows(ch) for ch in ("model_null", "explainer_repeat")}
    combined = list(csv.DictReader((RES / "explanations_noise/combined_null.csv").open(encoding="utf-8-sig")))
    expected = {"model_null": 5 * 100 * 2 * 5, "explainer_repeat": 5 * 100 * 2 * 10}
    col_ok = all(tuple(r.keys()) == tuple(("key", "checkpoint_sha", "anchor_id", "explainer", "seed", "k", "config_sha",
                                          "created_at_utc", "npz_path", "json_path", "fidelity_sufficiency",
                                          "fidelity_comprehensiveness", "fidelity_over_random", "runtime_s"))
                  for rows in chans.values() for r in rows[:1])
    keys = {ch: [r["key"] for r in rows] for ch, rows in chans.items()}
    combined_keys = [r["key"] for r in combined]
    combined_channels = {}
    for r in combined:
        combined_channels[r.get("channel")] = combined_channels.get(r.get("channel"), 0) + 1
    add("NOISE.counts_schema",
        "pass" if (len(chans["model_null"]) == expected["model_null"]
                   and len(chans["explainer_repeat"]) == expected["explainer_repeat"]
                   and len(combined) == expected["model_null"] + expected["explainer_repeat"]
                   and col_ok
                   and len(set(keys["model_null"])) == len(keys["model_null"])
                   and len(set(keys["explainer_repeat"])) == len(keys["explainer_repeat"])
                   and not (set(keys["model_null"]) & set(keys["explainer_repeat"]))
                   and combined_channels == {"model_null": expected["model_null"],
                                             "explainer_repeat": expected["explainer_repeat"]}) else "fail",
        {"rows": {"model_null": len(chans["model_null"]), "explainer_repeat": len(chans["explainer_repeat"]),
                  "combined": len(combined)}, "expected": expected | {"combined": 15000},
         "columns_ok": col_ok, "unique_keys": {ch: len(set(v)) for ch, v in keys.items()},
         "channel_key_disjoint": not (set(keys["model_null"]) & set(keys["explainer_repeat"])),
         "combined_channel_counts": combined_channels,
         "note": "500 reference anchors (100/period) x 2 explainers x R=10 repeats / R=5 null reseeds"})

    # ---------------- C. metadata / class / period provenance
    def meta_of(channel: str, row: dict) -> dict:
        p = RES / "explanations_noise" / channel / Path(row["json_path"]).name
        return rd(p)
    noise_anchors = {}
    for period in range(1, 6):
        payload = torch.load(RES / "anchors_noise" / f"period_{period}.pt", map_location="cpu", weights_only=False)
        noise_anchors[period] = payload.get("anchors") or []
    anchor_labels = {(period, int(a["anchor_id"])): int(a["label"]) for period, anchors in noise_anchors.items() for a in anchors}
    metadata_problems = []
    repeat_counts = {}
    null_seed_counts = {}
    for channel, rows in chans.items():
        for row in rows:
            m = meta_of(channel, row)
            aid = int(row["anchor_id"]); ap = int(m.get("anchor_period", -1))
            if (ap, aid) not in anchor_labels:
                metadata_problems.append({"channel": channel, "anchor": aid, "reason": "not in noise anchor subset"})
                continue
            if m.get("label") != anchor_labels[(ap, aid)] or str(m.get("class")) != str(anchor_labels[(ap, aid)]):
                metadata_problems.append({"channel": channel, "anchor": aid, "reason": "class/label mismatch"})
            if m.get("anchor_period") != ap or m.get("checkpoint_period") != ap:
                metadata_problems.append({"channel": channel, "anchor": aid, "reason": "period mismatch"})
            if m.get("strategy") != "ft" or m.get("model") != "egraphsage":
                metadata_problems.append({"channel": channel, "anchor": aid, "reason": "group mismatch"})
            if channel == "explainer_repeat":
                repeat_counts[int(m.get("repeat_index", -1))] = repeat_counts.get(int(m.get("repeat_index", -1)), 0) + 1
            else:
                null_seed_counts[int(m.get("null_seed", -1))] = null_seed_counts.get(int(m.get("null_seed", -1)), 0) + 1
    expected_repeats = {i: 5 * 100 * 2 for i in range(10)}
    expected_nulls = {142 + i: 5 * 100 * 2 for i in range(5)}
    add("NOISE.metadata",
        "pass" if (not metadata_problems and repeat_counts == expected_repeats and null_seed_counts == expected_nulls) else "fail",
        {"metadata_problems": metadata_problems[:20], "n_problems": len(metadata_problems),
         "repeat_index_counts": repeat_counts, "expected": expected_repeats,
         "null_seed_counts": null_seed_counts, "expected_nulls": expected_nulls,
         "note": "every entry carries class/label/anchor_period/checkpoint_period/group metadata consistent with the "
                 "canonical anchor labels; repeat seeds = 42000+repeat_index; null seeds = 142..146"})

    # ---------------- D. canonical anchors post-cap + D-063 contexts + noise subset
    canon_man = rd(RES / "anchors/canonical/anchors_manifest.json")
    cfg = canon_man.get("anchor_config") or {}
    canon_recs = []
    context_ok = target_ok = 0
    for period in range(1, 6):
        payload = torch.load(RES / "anchors/canonical" / f"period_{period}.pt", map_location="cpu", weights_only=False)
        anchors = payload.get("anchors") or []
        rec = (canon_man.get("per_period") or {}).get(str(period)) or {}
        strata = {k: int(v) for k, v in (rec.get("strata") or {}).items()}
        canon_recs.append({"period": period, "n_selected_manifest": rec.get("n_selected"),
                           "n_anchors_file": len(anchors), "strata": strata, "strata_sum": sum(strata.values()),
                           "strata_sum_ok": sum(strata.values()) == len(anchors) == rec.get("n_selected"),
                           "config_cap": cfg.get("max_per_period"), "n_within_cap": len(anchors) <= int(cfg.get("max_per_period", 0))})
        for a in anchors:
            ids = [int(x) for x in a["sub_edge_ids"].tolist()]
            tgt = int(a["target_edge"].item())
            context_ok += int(len(ids) <= int(cfg.get("max_subgraph_edges", 2000)))
            target_ok += int(tgt in ids and ids.index(tgt) == int(a["target_local_index"].item()))
        # noise subset provenance: same anchor ids + identical contexts
        noise_ids = {int(a["anchor_id"]) for a in noise_anchors[period]}
        canon_by_id = {int(a["anchor_id"]): a for a in anchors}
        missing = noise_ids - set(canon_by_id)
        ctx_equal = 0
        for nid in noise_ids:
            if nid not in canon_by_id:
                continue
            a = canon_by_id[nid]; b = next(x for x in noise_anchors[period] if int(x["anchor_id"]) == nid)
            if a["sub_edge_ids"].tolist() == b["sub_edge_ids"].tolist():
                ctx_equal += 1
        canon_recs[-1].update({"noise_subset_n": len(noise_ids), "noise_subset_expected": 100,
                               "noise_ids_missing_from_canonical": sorted(missing)[:5],
                               "noise_contexts_identical": ctx_equal, "noise_subset_ok":
                               len(noise_ids) == 100 and not missing and ctx_equal == len(noise_ids)})
    add("NOISE.canonical_anchors",
        "pass" if (all(r["strata_sum_ok"] and r["n_within_cap"] and r["noise_subset_ok"] for r in canon_recs)
                   and context_ok == sum(r["n_anchors_file"] for r in canon_recs)
                   and target_ok == context_ok
                   and cfg.get("max_per_period") == 250 and cfg.get("max_per_class") == 100) else "fail",
        {"canonical_config": cfg, "periods": canon_recs,
         "contexts_within_cap": context_ok, "targets_kept_and_local": target_ok,
         "expected_counts": EXPECTED_CANONICAL,
         "note": "canonical first-seed anchors built with the D-071 production caps (100/class, 250/period); post-cap strata "
                 "sums equal n_selected (F1); noise subsets are 100/period stratified subsets of the canonical anchors with "
                 "identical D-063 contexts (no re-sampling)."})

    # ---------------- E. IG-repeat determinism (all IG repeats bit-identical) + analysis canary
    _repeat_cache: dict[str, dict[tuple[int, int], list[np.ndarray]]] = {}

    def load_repeat_arrays(explainer: str):
        if explainer in _repeat_cache:
            return _repeat_cache[explainer]
        arrays: dict[tuple[int, int], list[np.ndarray]] = {}
        for row in chans["explainer_repeat"]:
            if row["explainer"] != explainer:
                continue
            m = meta_of("explainer_repeat", row)
            key = (int(m["anchor_period"]), int(row["anchor_id"]))
            with np.load(RES / "explanations_noise/explainer_repeat" / Path(row["npz_path"]).name) as z:
                arr = z["attributions"] if "attributions" in z.files else z["edge_mask"]
            arrays.setdefault(key, [None] * 10)[int(m["repeat_index"])] = np.asarray(arr)
        _repeat_cache[explainer] = {k: v for k, v in arrays.items() if all(x is not None for x in v)}
        return _repeat_cache[explainer]
    ig_arrays = load_repeat_arrays("ig")
    non_identical = 0
    max_diff = 0.0
    for key, seq in ig_arrays.items():
        base = seq[0]
        for cur in seq[1:]:
            if base.shape != cur.shape:
                non_identical += 1
                continue
            diff = float(np.max(np.abs(base - cur))) if base.size else 0.0
            max_diff = max(max_diff, diff)
            if diff != 0.0:
                non_identical += 1
    try:
        canary = nf.ig_repeat_canary({k: v for k, v in ig_arrays.items()}, metric_set="feature")
        canary_max = max(float(v) for v in canary.values()) if canary else None
    except Exception as exc:  # noqa: BLE001
        canary, canary_max = {"error": repr(exc)}, None
    add("NOISE.ig_repeat_canary",
        "pass" if non_identical == 0 and max_diff == 0.0 else "fail",
        {"ig_anchors_checked": len(ig_arrays), "repeats_per_anchor": 10,
         "non_identical_repeats": non_identical, "max_abs_diff": max_diff,
         "analysis_ig_repeat_canary": canary_max, "summary_canary_max": 2.220446049250313e-16,
         "note": "IG has no stochastic component: all R=10 repeats must be bit-identical. The summary canary reports "
                 "~2.2e-16 through the cosine metric; direct array comparison is exactly 0."})

    # ---------------- F. q95 pooling readiness via the authoritative analysis API
    q95_recs = []
    threshold_sources = {}
    edge_classes = []
    for explainer, metric_set in (("ig", "feature"), ("gnn_explainer", "edge")):
        per_period_sources = {}
        for period in range(1, 6):
            ex = {k: v for k, v in load_repeat_arrays(explainer).items() if k[0] == period}
            class_by = {(p, aid): str(anchor_labels[(p, aid)]) for (p, aid) in ex}
            if not ex:
                continue
            frame = nf.explainer_repeat_drift(ex, class_by_anchor=class_by, metric_set=metric_set, ks=(5,))
            frame["explainer"] = explainer
            table = nf.per_class_thresholds(frame, cfg=nf.NoiseFloorConfig())
            rows = table.as_rows()
            for src in {r["threshold_source"] for r in rows}:
                per_period_sources[src] = per_period_sources.get(src, 0) + 1
            if explainer == "gnn_explainer":
                edge_classes = [r["class"] for r in rows]
        threshold_sources[explainer] = per_period_sources
        q95_recs.append({"explainer": explainer, "metric_set": metric_set,
                         "threshold_sources": per_period_sources})
    add("NOISE.q95_readiness",
        "pass" if all(v for v in threshold_sources.values()) and "class" in threshold_sources["ig"] else "warn",
        {"per_explainer": q95_recs, "noise_floor_config": {"n_null_reseeds": 5, "n_explainer_repeats": 10,
                                                           "quantile": 0.95, "min_anchors_per_class": 20,
                                                           "pooling": "class"},
         "note": "the authoritative analysis.noise_floor pipeline (explainer_repeat_drift -> per_class_thresholds) runs "
                 "on the smoke repeat caches: per-(explainer,class) q95 thresholds are produced with class pooling and the "
                 "global fallback recorded in threshold_source. The primary model_null channel additionally needs the S7 "
                 "reference explanations (produced by the corrected full S7 run), so model_null q95 is not computable from "
                 "the noise stage alone; its per-entry metadata/schema is ready."})

    # ---------------- G. F2 recipe check: smoke null models used sampled batching / 10 epochs
    log_text = (UP / "logs/noise_floor_p1.log").read_text(encoding="utf-8", errors="replace")
    sampled = log_text.count("batching=sampled")
    graph_batch = log_text.count("batching=graph_batch")
    null_epochs = noise.get("null_epochs")
    try:
        from src.utils.config import load_config
        from src.train.trainer import TrainSettings
        cfg_d = load_config("configs/drift.yaml", project_root=REPO).to_dict()
        settings = TrainSettings.from_config(cfg_d)
        null_cfg_train = cfg_d.get("train")
        resolved_null_epochs = (int((null_cfg_train or {}).get("max_epochs")) if null_cfg_train else 10)
        recipe = {"drift_config_has_train_section": null_cfg_train is not None,
                  "from_config_batching": settings.batching,
                  "main_fallback_null_epochs": resolved_null_epochs}
    except Exception as exc:  # noqa: BLE001
        recipe = {"error": repr(exc)}
    period1_sum = rd(RES / "data_audit/noise_floor_summary_ft_egraphsage_seed42_period1.json")
    null_checkpoints = ((period1_sum.get("model_null") or {}).get("null_checkpoints") or [])
    recipe.update({"noise_floor_null_epochs_from_runctl": null_epochs,
                   "log_sampled_lines": sampled, "log_graph_batch_lines": graph_batch,
                   "null_checkpoint_epochs": [c.get("epochs_run") for c in null_checkpoints]})
    f2_ok = (sampled == 0 and graph_batch > 0 and all(e == 30 for e in recipe["null_checkpoint_epochs"] if e is not None))
    add("NOISE.f2_null_recipe", "pass" if f2_ok else "fail",
        {**recipe,
         "root_cause": "src/scripts/build_noise_floor.py loads configs/drift.yaml (no `train:` section) and builds "
                       "TrainSettings.from_config(...), whose default batching is `sampled`; null_epochs then falls back to "
                       "10 because drift.yaml has no train.max_epochs. The F2 commit changed produce_model_null's default "
                       "settings but the CLI path still passes the drift-config defaults, so the smoke null models trained "
                       "with batching=sampled for 10 epochs (confirmed verbatim in noise_floor_p1.log).",
         "impact": "the model-null noise distribution is not the canonical FT recipe (D-059 graph_batch / max_epochs=30 / "
                   "patience=5) and cannot be used as the S8 noise floor until fixed and the noise smoke rerun.",
         "fix": "load the train section from configs/train.yaml (merge under drift config or pass explicit TrainSettings), "
                "default null_epochs to the canonical 30 (only an explicit, documented cap may shorten it), and assert "
                "batching=graph_batch in the noise stage summary."})

    # ---------------- H. per-period summaries/manifests
    sum_problems = []
    sum_recs = []
    for period in range(1, 6):
        p = RES / "data_audit" / f"noise_floor_summary_ft_egraphsage_seed42_period{period}.json"
        s = rd(p)
        mn = s.get("model_null") or {}; er = s.get("explainer_repeat") or {}
        combined_block = s.get("combined") or {}
        expected_combined_rows = 3000 * period  # combined_null.csv is cumulative across periods
        ok = (int(mn.get("entries", 0)) == 1000 and int(er.get("entries", 0)) == 2000
              and int(combined_block.get("rows", 0)) == expected_combined_rows
              and int(combined_block.get("channel_counts", {}).get("model_null", 0)) == 1000 * period
              and int(combined_block.get("channel_counts", {}).get("explainer_repeat", 0)) == 2000 * period)
        if not ok:
            sum_problems.append(str(p.name))
        sum_recs.append({"period": period, "model_null_entries": mn.get("entries"),
                         "explainer_repeat_entries": er.get("entries"),
                         "combined_rows": (s.get("combined") or {}).get("rows"),
                         "combined_channel_counts": (s.get("combined") or {}).get("channel_counts"),
                         "null_checkpoints": mn.get("null_checkpoints"),
                         "ig_canary": er.get("ig_repeat_canary_drift_max")})
    add("NOISE.period_summaries", "pass" if not sum_problems else "fail",
        {"periods": sum_recs, "problems": sum_problems,
         "note": "per-period summaries record 1,000 model-null + 2,000 explainer-repeat entries and 3,000 combined rows; "
                 "null seeds 142..146; IG canary ~2.2e-16 (array-exact check passes above)."})

    # ---------------- I. strict validators re-run
    def stage(name):
        p = subprocess.run([PY, "-m", "src.analysis.validators", "--stage", name, "--strict",
                            "--results-tables", str(RES / "tables"), "--json"],
                           cwd=str(REPO), capture_output=True, text=True, env=ENV)
        payload = {}
        try:
            payload = json.loads(p.stdout[p.stdout.find("{"):])
        except Exception:
            pass
        return {"returncode": p.returncode, "overall": payload.get("overall"), "counts": payload.get("counts")}
    s5, s6, s7 = stage("s5"), stage("s6"), stage("s7")
    add("NOISE.validators",
        "pass" if (s5.get("counts", {}).get("pass") == 16 and s5.get("overall") == "pass"
                   and s6.get("counts", {}).get("pass") == 28 and s6.get("overall") == "pass"
                   and s7.get("counts", {}).get("pass") == 48 and s7.get("overall") == "pass") else "fail",
        {"s5": s5, "s6": s6, "s7": s7,
         "note": "noise smoke carries the resumed canonical S5/S6/S7 artifacts; strict local re-run reproduces "
                 "16/0/0, 28/0/0, 48/0/0."})

    # ---------------- J. full corrected S7 status
    full = REPO / "incoming_rtx5060/kaggle_runs/t39-explain-full-649109a"
    n_full = len([p for p in full.rglob("*") if p.is_file()]) if full.is_dir() else 0
    add("S7.full_status", "warn" if n_full == 0 else "pass",
        {"dir": str(full), "downloaded_files": n_full,
         "expected": "6 seed-42 groups, actual canonical anchors 229/205/221/230/123 (cap 250 not binding); "
                     "19,266 t>=j pairs = 13,218 cross + 6,048 diagonal -> 57,798 explanation rows (D-079); audit plan "
                     "results/text/verification/data_audit/t49_s7_audit_plan.json"})

    # ---------------- overall
    fails = [c for c in chk if c["status"] == "fail"]
    res["overall"] = "fail" if fails else "pass"
    res["summary"] = {"run_id": run_sum.get("run_id"), "kernel": run_sum.get("kernel_ref"),
                      "status": run_sum.get("status"), "gpu_hours": round(float(run_sum.get("wall_clock_s") or 0) / 3600, 2),
                      "rows": {ch: len(rows) for ch, rows in chans.items()} | {"combined": len(combined)},
                      "f2_recipe": "FAIL (sampled/10 epochs)" if not f2_ok else "pass",
                      "validators": {"s5": s5.get("counts"), "s6": s6.get("counts"), "s7": s7.get("counts")}}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t42_noise_smoke_audit.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({c["id"]: c["status"] for c in chk}, indent=1))
    print("overall:", res["overall"], "summary:", json.dumps(res["summary"]))
    for c in chk:
        if c["status"] != "pass":
            print(c["id"], json.dumps(c["detail"], default=str)[:1200])
    return 0 if not fails else 2


if __name__ == "__main__":
    raise SystemExit(main())
