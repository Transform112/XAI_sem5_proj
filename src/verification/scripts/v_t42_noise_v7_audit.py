"""T50/F4 verifier audit: v7 noise-floor rerun t42-noise-v7-3f3dbad (canonical null recipe + R-aware noise caches).

Independent checks: run/provenance/zip, counts/schema/keys, canonical anchors post-cap, review/new
F2 train recipe (graph_batch / 30 / patience 5, early-stopped null epochs), IG-repeat determinism,
q95 pipeline readiness, strict validators, lane/gate. Writes
results/text/verification/data_audit/t42_noise_v7_audit.json. No repo code/results modified.
"""
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(r"C:\Users\iusearchbtw\Desktop\XAI_proj")
PY = str(REPO / ".venv" / "Scripts" / "python.exe")
RUN = REPO / "incoming_rtx5060/kaggle_runs/t42-noise-v7-3f3dbad"
UP = RUN / "unpacked"
RES = UP / "results"
OUT = REPO / "results/text/verification/data_audit"
EXPECTED_ZIP_SHA = "168f5ac4a4724576d2cc24cf642b37af15c2c76cb7b47c200b2a208b60210d74"
EXPECTED_ZIP_SIZE = 458498574
EXPECTED_TRAIN_SHA = "791bfcb2f8aa2becf2f39ac2cead934c65710e82e77cede52dc2813e15d5dd7c"
EXPECTED_ANCHORS = {1: 229, 2: 205, 3: 221, 4: 230, 5: 123}
SNAPSHOT = "3f3dbad"
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

    run_sum = rd(UP / "run_summary.json")
    noise_audit = rd(RUN / "noise_audit.json")
    iv = run_sum.get("input_verification") or {}
    env = run_sum.get("environment") or {}
    zip_path = RUN / "kernel_output/xdrift_run_output.zip"
    zip_sha = sha_file(zip_path)
    zip_size = zip_path.stat().st_size
    noise = run_sum.get("noise_floor") or {}
    add("PROV.run",
        "pass" if (run_sum.get("status") == "success" and run_sum.get("stage") == "noise"
                   and run_sum.get("kernel_ref") == "mysteriousavailable/xdrift-noise"
                   and int(iv.get("dataset_version", -1)) == 7 and str(iv.get("commit")) == SNAPSHOT
                   and iv.get("csv_sha256") == "4ebb97bd74412d566137d95a6fc3ffd8f374f1cf8cfe204d007848e7a668f9b5"
                   and iv.get("csv_bytes") == 577360958 and iv.get("frozen_files_verified") == 14
                   and ((env.get("env_vars") or {}).get("XDRIFT_LANE") == "kaggle")
                   and abs(float(run_sum.get("wall_clock_s") or 0) - 5807.7) < 1.0) else "fail",
        {"run_id": run_sum.get("runctl", {}).get("run_id"), "status": run_sum.get("status"),
         "stage": run_sum.get("stage"), "kernel_ref": run_sum.get("kernel_ref"),
         "wall_clock_s": run_sum.get("wall_clock_s"), "gpu_hours": round(float(run_sum.get("wall_clock_s") or 0)/3600, 3),
         "input_verification": {k: iv.get(k) for k in ("dataset_version", "commit", "csv_sha256", "csv_bytes", "frozen_files_verified", "code_name")},
         "lane": (env.get("env_vars") or {}).get("XDRIFT_LANE"),
         "output_zip": {"sha256": zip_sha, "expected_sha256": EXPECTED_ZIP_SHA, "bytes": zip_size,
                        "expected_bytes": EXPECTED_ZIP_SIZE,
                        "match": zip_sha == EXPECTED_ZIP_SHA and zip_size == EXPECTED_ZIP_SIZE}})
    add("PROV.zip",
        "pass" if zip_sha == EXPECTED_ZIP_SHA and zip_size == EXPECTED_ZIP_SIZE else "fail",
        {"sha256": zip_sha, "bytes": zip_size})
    add("PROV.noise_audit_json",
        "pass" if noise_audit.get("overall") == "pass" and not noise_audit.get("problems") else "fail",
        noise_audit)

    # ---------------- counts / schema
    def rows(channel):
        return list(csv.DictReader((RES / "explanations_noise" / channel / "index.csv").open(encoding="utf-8-sig")))
    null_rows = rows("model_null"); repeat_rows = rows("explainer_repeat")
    combined = list(csv.DictReader((RES / "explanations_noise/combined_null.csv").open(encoding="utf-8-sig")))
    expected_cols = ["key", "checkpoint_sha", "anchor_id", "explainer", "seed", "k", "config_sha",
                     "created_at_utc", "npz_path", "json_path", "fidelity_sufficiency",
                     "fidelity_comprehensiveness", "fidelity_over_random", "runtime_s"]
    from collections import Counter
    chan_counts = Counter(r.get("channel") for r in combined)
    type_counts = {ch: Counter(r["explainer"] for r in rows_) for ch, rows_ in (("model_null", null_rows), ("explainer_repeat", repeat_rows))}
    add("NOISE.counts_schema",
        "pass" if (len(null_rows) == 5000 and len(repeat_rows) == 10000 and len(combined) == 15000
                   and all(tuple(r.keys()) == tuple(expected_cols) for r in null_rows[:1] + repeat_rows[:1])
                   and len({r["key"] for r in null_rows}) == 5000 and len({r["key"] for r in repeat_rows}) == 10000
                   and chan_counts == {"model_null": 5000, "explainer_repeat": 10000}
                   and type_counts == {"model_null": {"ig": 2500, "gnn_explainer": 2500},
                                       "explainer_repeat": {"ig": 5000, "gnn_explainer": 5000}}) else "fail",
        {"rows": {"model_null": len(null_rows), "explainer_repeat": len(repeat_rows), "combined": len(combined)},
         "combined_channel_counts": dict(chan_counts), "rows_by_explainer": type_counts,
         "unique_keys": {"model_null": len({r["key"] for r in null_rows}), "explainer_repeat": len({r["key"] for r in repeat_rows})},
         "expected": {"model_null": 5000, "explainer_repeat": 10000, "combined": 15000,
                      "by_explainer": {"model_null": {"ig": 2500, "gnn_explainer": 2500},
                                       "explainer_repeat": {"ig": 5000, "gnn_explainer": 5000}}}})

    # per-period summaries + train recipe
    period_recs = []
    recipe_problems = []
    for period in range(1, 6):
        s = rd(RES / "data_audit" / f"noise_floor_summary_ft_egraphsage_seed42_period{period}.json")
        tc = s.get("train_config") or {}
        mn = s.get("model_null") or {}
        er = s.get("explainer_repeat") or {}
        ckpts = mn.get("null_checkpoints") or []
        epochs = [int(c.get("epochs_run", -999)) for c in ckpts]
        seeds = [int(c.get("null_seed", -1)) for c in ckpts]
        if (tc.get("batching") != "graph_batch" or int(tc.get("max_epochs", 0)) != 30
                or int(tc.get("null_epochs", 0)) != 30 or int(tc.get("patience", 0)) != 5
                or tc.get("path") != "configs/train.yaml" or tc.get("sha256") != EXPECTED_TRAIN_SHA
                or not epochs or max(epochs) > 30 or min(epochs) < 1 or seeds != [142, 143, 144, 145, 146]
                or int(mn.get("entries", 0)) != 1000 or int(er.get("entries", 0)) != 2000
                or float(er.get("ig_repeat_canary_drift_max", 1)) > 1e-12
                or (s.get("combined") or {}).get("rows") != 3000 * period):
            recipe_problems.append({"period": period, "train_config": tc, "epochs": epochs, "seeds": seeds})
        period_recs.append({"period": period, "train_config": tc, "epochs_run": epochs, "null_seeds": seeds,
                            "canary": er.get("ig_repeat_canary_drift_max"), "entries": [mn.get("entries"), er.get("entries")],
                            "combined_rows": (s.get("combined") or {}).get("rows")})
    add("F2.recipe", "pass" if not recipe_problems and len(period_recs) == 5 else "fail",
        {"train_config": period_recs[0]["train_config"],
         "all_periods_epochs": [r["epochs_run"] for r in period_recs],
         "all_periods_seeds": [r["null_seeds"] for r in period_recs],
         "canary": [r["canary"] for r in period_recs],
         "problems": recipe_problems[:5],
         "expected_train_sha": EXPECTED_TRAIN_SHA,
         "note": "canonical recipe: graph_batch, max_epochs=30, null_epochs default 30, patience=5; early stopping yields epochs_run<=30; seeds 142-146; all 5 summaries carry identical train_config provenance"})

    # logs: graph_batch present, sampled absent
    logs = []
    sampled = graph_batch = 0
    for period in range(1, 6):
        p = UP / "logs" / f"noise_floor_p{period}.log"
        if p.is_file():
            txt = p.read_text(encoding="utf-8", errors="replace")
            logs.append(p.name)
            sampled += txt.count("batching=sampled")
            graph_batch += txt.count("batching=graph_batch")
    add("F2.logs", "pass" if graph_batch > 0 and sampled == 0 and len(logs) == 5 else "fail",
        {"logs": logs, "batching_graph_batch": graph_batch, "batching_sampled": sampled})

    # ---------------- canonical anchors post-cap + noise subsets
    canon = rd(RES / "anchors/canonical/anchors_manifest.json")
    cfg = canon.get("anchor_config") or {}
    anchor_recs = []
    ctx_ok = tgt_ok = 0
    for period in range(1, 6):
        payload = torch.load(RES / "anchors/canonical" / f"period_{period}.pt", map_location="cpu", weights_only=False)
        anchors = payload.get("anchors") or []
        rec = (canon.get("per_period") or {}).get(str(period)) or {}
        strata = {k: int(v) for k, v in (rec.get("strata") or {}).items()}
        noise_payload = torch.load(RES / "anchors_noise" / f"period_{period}.pt", map_location="cpu", weights_only=False)
        noise_anchors = noise_payload.get("anchors") or []
        noise_ids = {int(a["anchor_id"]) for a in noise_anchors}
        canon_by = {int(a["anchor_id"]): a for a in anchors}
        ctx = sum(1 for a in noise_anchors if int(a["anchor_id"]) in canon_by
                  and canon_by[int(a["anchor_id"])]["sub_edge_ids"].tolist() == a["sub_edge_ids"].tolist())
        for a in anchors:
            ids = [int(x) for x in a["sub_edge_ids"].tolist()]
            tgt = int(a["target_edge"].item())
            ctx_ok += int(len(ids) <= 2000)
            tgt_ok += int(tgt in ids and ids.index(tgt) == int(a["target_local_index"].item()))
        anchor_recs.append({"period": period, "n_selected": rec.get("n_selected"), "n_file": len(anchors),
                            "strata": strata, "strata_sum": sum(strata.values()),
                            "strata_sum_ok": sum(strata.values()) == len(anchors) == rec.get("n_selected"),
                            "config_cap": cfg.get("max_per_period"), "within_cap": len(anchors) <= int(cfg.get("max_per_period", 0)),
                            "noise_subset_n": len(noise_ids), "noise_subset_expected": 100,
                            "noise_subset_in_canonical": not (noise_ids - set(canon_by)),
                            "noise_contexts_identical": ctx})
    add("ANCHORS.postcap",
        "pass" if (all(r["strata_sum_ok"] and r["within_cap"] and r["noise_subset_n"] == 100
                       and r["noise_subset_in_canonical"] and r["noise_contexts_identical"] == 100 for r in anchor_recs)
                   and ctx_ok == sum(r["n_file"] for r in anchor_recs) and tgt_ok == ctx_ok
                   and cfg.get("max_per_period") == 250 and cfg.get("max_per_class") == 100) else "fail",
        {"canonical_config": cfg, "periods": anchor_recs, "contexts_within_cap": ctx_ok,
         "targets_kept_local": tgt_ok, "expected_counts": EXPECTED_ANCHORS})

    # ---------------- IG-repeat determinism (all 10 repeats per anchor identical)
    def load_repeat_arrays(explainer):
        arrays = {}
        for row in repeat_rows:
            if row["explainer"] != explainer:
                continue
            jp = RES / "explanations_noise/explainer_repeat" / Path(row["json_path"]).name
            m = rd(jp)
            key = (int(m["anchor_period"]), int(row["anchor_id"]))
            with np.load(RES / "explanations_noise/explainer_repeat" / Path(row["npz_path"]).name) as z:
                arr = z["attributions"] if "attributions" in z.files else z["edge_mask"]
            arrays.setdefault(key, [None] * 10)[int(m["repeat_index"])] = np.asarray(arr)
        return {k: v for k, v in arrays.items() if all(a is not None for a in v)}
    ig = load_repeat_arrays("ig")
    non_identical = 0; max_diff = 0.0
    for seq in ig.values():
        base = seq[0]
        for cur in seq[1:]:
            diff = float(np.max(np.abs(base - cur))) if base.size else 0.0
            max_diff = max(max_diff, diff)
            non_identical += int(diff != 0.0)
    add("IG.canary", "pass" if len(ig) == 500 and non_identical == 0 and max_diff == 0.0 else "fail",
        {"ig_anchors": len(ig), "non_identical_repeats": non_identical, "max_abs_diff": max_diff,
         "summary_canary": 2.220446049250313e-16})

    # q95 readiness (repeat channel) via authoritative API
    q95_ok = True; q95_detail = {}
    for explainer, metric_set in (("ig", "feature"), ("gnn_explainer", "edge")):
        ex = load_repeat_arrays(explainer)
        class_by = {(1, 1): "0"}  # not used; load classes from metadata below
        # load class per anchor
        class_by = {}
        for row in repeat_rows:
            if row["explainer"] != explainer:
                continue
            m = rd(RES / "explanations_noise/explainer_repeat" / Path(row["json_path"]).name)
            class_by[(int(m["anchor_period"]), int(row["anchor_id"]))] = str(m.get("class"))
        frame = nf.explainer_repeat_drift(ex, class_by_anchor=class_by, metric_set=metric_set, ks=(5,))
        frame["explainer"] = explainer
        table = nf.per_class_thresholds(frame, cfg=nf.NoiseFloorConfig())
        rows_ = table.as_rows()
        q95_detail[explainer] = {"pairs": len(frame), "threshold_rows": len(rows_),
                                 "sources": sorted({r["threshold_source"] for r in rows_}),
                                 "finite_q95": all(np.isfinite(float(r["q95"])) for r in rows_)}
        q95_ok = q95_ok and bool(rows_) and all(np.isfinite(float(r["q95"])) for r in rows_)
    add("Q95.readiness", "pass" if q95_ok else "fail",
        {"detail": q95_detail, "note": "repeat-channel per-(explainer,class) q95 thresholds computed by the "
         "authoritative analysis.noise_floor pipeline; model-null q95 still needs the S7 real diagonal reference"})

    # ---------------- strict validators
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
    add("VALIDATORS", "pass" if (s5.get("overall") == "pass" and s5.get("counts", {}).get("pass") == 16
                                  and s6.get("overall") == "pass" and s6.get("counts", {}).get("pass") == 28
                                  and s7.get("overall") == "pass" and s7.get("counts", {}).get("pass") == 48) else "fail",
        {"s5": s5, "s6": s6, "s7": s7})

    fails = [c for c in chk if c["status"] == "fail"]
    res["overall"] = "fail" if fails else "pass"
    res["f4_signoff"] = {
        "noise_gate": "SIGNED" if res["overall"] == "pass" else "NOT SIGNED",
        "run_id": "t42-noise-v7-3f3dbad", "dataset_version": 7, "commit": SNAPSHOT,
        "canonical_recipe": "graph_batch / max_epochs=30 / patience=5", "reference": "ft/egraphsage/seed42",
        "counts": {"model_null": 5000, "explainer_repeat": 10000, "combined_null": 15000},
        "strict": {"s5": s5.get("counts"), "s6": s6.get("counts"), "s7": s7.get("counts")},
        "next": "unblocks S8/S9 once full S7 lands",
    }
    (OUT / "t42_noise_v7_audit.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    print(json.dumps({c["id"]: c["status"] for c in chk}, indent=1))
    print("overall:", res["overall"], "F4:", res["f4_signoff"]["noise_gate"])
    for c in chk:
        if c["status"] != "pass":
            print(c["id"], json.dumps(c["detail"], default=str)[:1500])
    return 0 if not fails else 2


if __name__ == "__main__":
    raise SystemExit(main())
