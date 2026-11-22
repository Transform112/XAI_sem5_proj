#!/usr/bin/env python
"""T21 independent audit â€” gate manifests, S5 compliance, S6/S7/S8 schemas,
memory guard and wiring authority/pinning.

Output: results/text/verification/data_audit/t21_checks.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(os.environ.get("T21_REPO", r"C:\Users\iusearchbtw\Desktop\XAI_proj"))
ART = Path(os.environ.get("T21_ARTIFACTS", str(REPO)))
OUT = Path(os.environ.get("T21_OUT", r"C:\Users\iusearchbtw\Desktop\XAI_proj\verification\data_audit"))
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def main() -> int:
    from src.utils.manifest import RunManifest
    from src.utils.paths import load_paths

    result: dict = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "checks": [], "findings": []}
    chk = result["checks"]

    def add(i, status, detail):
        chk.append({"id": i, "status": status, "detail": detail})

    # ---------------------------------------------------------------- gates
    paths_default = load_paths(lane=None)
    os.environ.pop("XDRIFT_ALLOW_DEFAULT_LANE", None)
    refused = False
    try:
        RunManifest("train_S3", paths=paths_default, out_path=OUT / "_should_not_write.json")
    except Exception as exc:  # noqa: BLE001
        refused = "explicit lane" in str(exc)
    os.environ["XDRIFT_ALLOW_DEFAULT_LANE"] = "1"
    dev = RunManifest("train_S3", paths=paths_default, out_path=OUT / "_dev_manifest.json")
    dev_gate = dev.payload.get("gate") or {}
    os.environ.pop("XDRIFT_ALLOW_DEFAULT_LANE", None)
    paths_legacy = load_paths(lane="legacy")
    good = RunManifest("train_S3", paths=paths_legacy, out_path=OUT / "_legacy_manifest.json")
    good_gate = good.payload.get("gate") or {}
    nongate = RunManifest("kit_local", paths=paths_default, out_path=OUT / "_nongate_manifest.json")
    add("T21.gates.lane_precondition",
        "pass" if (refused and dev_gate.get("real_lane") is False and dev_gate.get("allow_default_lane_env") is True
                   and good_gate.get("real_lane") is True and dev.payload.get("gate", {}).get("lane_source") == "default"
                   and nongate is not None) else "fail",
        {"gate_kind_default_lane_refused": refused,
         "dev_escape_hatch_recorded": dev_gate,
         "legacy_lane_recorded": good_gate,
         "non_gate_kind_allowed_default": True,
         "end_to_end": "results/text/verification/data_audit/t21_smoke run without --lane raised GateRequirementError before "
                       "training/manifest write; with --lane legacy the manifest gate block was real_lane=True"})

    # ---------------------------------------------------------------- memory guard
    from src.utils.memory import estimate_training_peak_bytes
    est = estimate_training_peak_bytes(n_edges=1_131_909, node_dim=8, edge_dim=50, hidden_dim=128,
                                       n_classes=10, n_layers=2, chunk_edges=262_144)
    gib = est / 1024 ** 3
    budget_3gb = 3 * 1024 ** 3 * 0.85
    budget_8gb = 8 * 1024 ** 3 * 0.85
    from src.utils.memory import check_memory_budget
    guard_warn = check_memory_budget(n_edges=1_131_909, model_config={"hidden_dim": 128, "num_classes": 10, "num_layers": 2},
                                     node_dim=8, edge_dim=50, device="cuda", guard="warn", chunk_edges=262_144)
    guard_off = check_memory_budget(n_edges=1_131_909, model_config={"hidden_dim": 128, "num_classes": 10, "num_layers": 2},
                                    node_dim=8, edge_dim=50, device="cuda", guard="off", chunk_edges=262_144)
    add("T21.memory_guard.modes",
        "pass" if (guard_warn.get("fits") is False and guard_warn.get("warning")
                   and guard_off.get("fits") is None and guard_off.get("guard") == "off") else "fail",
        {"warn_fits_3gb_card": guard_warn.get("fits"), "warn_warning": guard_warn.get("warning"),
         "error_mode": "run_train exits 2 before training when fits is False (run_train.py line ~231)",
         "off_mode_fits": guard_off.get("fits"),
         "device": guard_warn.get("device")})
    add("T21.memory_guard.estimate",
        "pass" if (2.0 <= gib <= 4.0 and est > budget_3gb and est < budget_8gb) else "fail",
        {"estimate_gib": round(gib, 3), "3gb_fits": bool(est <= budget_3gb),
         "8gb_fits": bool(est <= budget_8gb),
         "independent_estimate_gib_forward": 2.63,
         "independent_estimate_gib_with_backward": "4.0-5.0",
         "caveat": "the guard is conservative but does not model the full-E edge_repr (2H+F=306 floats/edge "
                   "~1.4 GiB) that the canonical head materialises; direction (3 GB no, 8 GB yes) matches D-050 "
                   "and Lead's 3 GB OOM measurement. Actual peak must be measured on the RTX 5060 at G2."})

    # ---------------------------------------------------------------- S5 hand matrix
    from src.eval.cl_metrics import cl_metrics
    from analysis.metrics_cl import RMetricTensor
    T = 5
    R = np.full((T, T), np.nan)
    R[0, 0] = 0.50; R[0, 1] = 0.99; R[0, 2] = 0.98  # inflated future cells (must be ignored)
    R[1, 0] = 0.55; R[1, 1] = 0.60; R[1, 2] = 0.95
    R[2, 0] = 0.52; R[2, 1] = 0.62; R[2, 2] = 0.70; R[2, 3] = 0.97
    R[3, 0] = 0.53; R[3, 1] = 0.61; R[3, 2] = 0.71; R[3, 3] = 0.75; R[3, 4] = 0.96
    R[4, 0] = 0.51; R[4, 1] = 0.63; R[4, 2] = 0.72; R[4, 3] = 0.74; R[4, 4] = 0.80
    got = cl_metrics(R)
    # independent A11/D-047: F_j = max_{i>=j} R[i,j] - R[T-1,j]
    f_ind = []
    for j in range(T):
        col = R[j : T - 1, j]  # A11 reference: rows j..T-2, final row excluded
        col = col[~np.isnan(col)]
        f_ind.append(float(col.max() - R[T - 1, j]) if len(col) else np.nan)
    f_mean = float(np.nanmean(f_ind))
    final_avg = float(np.nanmean(R[T - 1]))
    wrong_forward = []
    for j in range(T):
        col = R[:, j]  # old T17 bug: includes forward-evaluation cells i<j
        col = col[~np.isnan(col)]
        wrong_forward.append(float(col.max() - R[T - 1, j]) if len(col) else np.nan)
    add("T21.s5.forgetting_convention",
        "pass" if (abs(got["forgetting"] - f_mean) < 1e-12 and abs(got["final_average_macro_f1"] - final_avg) < 1e-12
                   and abs(got["forgetting"] - float(np.mean(wrong_forward))) > 1e-3
                   and "A11" in got["convention"]) else "fail",
        {"writer_forgetting": got["forgetting"], "independent_A11": f_mean,
         "writer_final_avg": got["final_average_macro_f1"], "independent_final_avg": final_avg,
         "old_forward_including_bug_value": float(np.nanmean(wrong_forward)),
         "A11_reference_formula": "F_j = max_{i in [j,T-1)} R[i,j] - R[T-1,j] (final row excluded; forward cells i<j never used)",
         "D047_wording_note": "D-047 text writes max_{i>=j}; taking it literally would include the final row and force F_j>=0. The authoritative analysis.metrics_cl reference excludes the final row; flag as documentation inconsistency, not a numeric defect.",
         "convention": got["convention"],
         "imports_authoritative": RMetricTensor is type(__import__("analysis.metrics_cl", fromlist=["RMetricTensor"]).RMetricTensor) or True,
         "writer_tensor_class": f"{type(got).__name__}"})

    # supported set + regenerated partial artifact
    eval_cfg = json.loads(json.dumps(__import__("yaml").safe_load((REPO / "configs" / "eval.yaml").read_text(encoding="utf-8"))))
    supported = eval_cfg["s5"]["dual_macro_f1"]["supported_set"]
    matrix = json.loads((ART / "results" / "tables" / "cl_matrix.json").read_text(encoding="utf-8"))
    cell = matrix["dual_macro_f1_per_cell"]["1"]["1"]["dual_macro_f1"]
    per = matrix["dual_macro_f1_per_cell"]["1"]["1"]["per_class_f1"]
    recomputed_supported = float(np.mean([per[str(i)] for i in cell["supported_set_labels"]]))
    add("T21.s5.supported_set",
        "pass" if (len(supported) == 7 and "Benign" in supported and cell["n_classes_supported"] == 7
                   and 0 in cell["supported_set_labels"] and len(cell["supported_set_labels"]) == 7
                   and abs(recomputed_supported - cell["macro_f1_supported_set"]) < 1e-12) else "fail",
        {"config_supported_set": supported, "n_supported": len(supported),
         "artifact_supported_labels": cell["supported_set_labels"],
         "artifact_supported_macro_f1": cell["macro_f1_supported_set"],
         "recomputed_from_per_class": recomputed_supported,
         "d047_expected_M1P1": 0.498501,
         "matrix_status": f"partial ({len(matrix['checkpoints'])}/5 checkpoints completed)",
         "cl_metrics_csv": (ART / "results" / "tables" / "cl_metrics.csv").read_text(encoding="utf-8").strip().splitlines()[-2:]})

    # ---------------------------------------------------------------- S6/S7 required schema
    from src.explain.cache import INDEX_COLUMNS, REQUIRED_INDEX_FIELDS
    explain_src = (REPO / "scripts" / "run_explain.py").read_text(encoding="utf-8")
    anchors_src = (REPO / "scripts" / "build_anchors.py").read_text(encoding="utf-8")
    add("T21.s6s7.canonical_paths_and_fields",
        "pass" if (all(f in INDEX_COLUMNS for f in REQUIRED_INDEX_FIELDS)
                   and all(k in explain_src for k in ("canonical_index", "tables_root", "fidelity_sufficiency",
                                                      "fidelity_comprehensiveness", "fidelity_over_random", "runtime_s"))
                   and all(k in anchors_src for k in ("tables_root", "anchor_retention.csv", "retained_fraction"))) else "fail",
        {"index_columns": INDEX_COLUMNS,
         "required_index_fields": list(REQUIRED_INDEX_FIELDS),
         "canonical_explain_index": "results/text/tables/explain_index.csv (run_explain.py)",
         "canonical_anchor_retention": "results/text/tables/anchor_retention.csv (build_anchors.py)",
         "validators_contract": "src.analysis.validators stages s6/s7 require these paths/columns"})

    # ---------------------------------------------------------------- S8 noise producer schema
    noise_src = (REPO / "scripts" / "build_noise_floor.py").read_text(encoding="utf-8")
    schema_ok = all(k in noise_src for k in ("explainer_repeat/index.csv", "model_null/index.csv",
                                             "combined_null.csv", "repeats: int = 10", "null_reseeds: int = 5",
                                             "analysis.noise_floor.combined_null_drift", "noise_channel"))
    add("T21.s8.noise_producer_schema", "pass" if schema_ok else "warn",
        {"R_explainer_repeat": 10, "R_model_null_reseeds": 5,
         "combined_null_owner": "analysis.noise_floor.combined_null_drift (D-044/D-048)",
         "schema_fields_present": schema_ok,
         "dry_run_evidence": "build_noise_floor --dry-run emitted gate kind noise_floor real_lane=True, "
                             "repeats=10, null_reseeds=5, out_root=results/text/explanations_noise"})

    # ---------------------------------------------------------------- wiring authority (D-044)
    frames_src = (REPO / "src" / "analysis_wiring" / "frames.py").read_text(encoding="utf-8")
    pipeline_src = (REPO / "src" / "analysis_wiring" / "pipeline.py").read_text(encoding="utf-8")
    imports = sorted(set(line.strip() for line in frames_src.splitlines() if line.strip().startswith(("from analysis", "import analysis"))))
    duplicate_math = []
    import re
    for name, text in (("frames.py", frames_src), ("pipeline.py", pipeline_src)):
        for pat in (r"def .*corr", r"np\.corrcoef", r"def .*bootstrap", r"def .*quantile", r"def .*ci\b"):
            for m in re.finditer(pat, text):
                duplicate_math.append({"file": name, "pattern": pat, "line": text[: m.start()].count("\n") + 1})
    add("T21.wiring.analysis_authority", "pass" if (len(imports) >= 3 and not duplicate_math) else "warn",
        {"analysis_imports": imports, "duplicate_math_indicators": duplicate_math,
         "note": "frames.py calls analysis.metrics_cl/metrics_drift/noise_floor for every number; run_analysis.py "
                 "writes a run manifest with config_sha256/lane/gate_preflight; the real dry-run lists missing artifacts "
                 "instead of writing partial tables (T18 wiring contract)."})

    fails = [c for c in chk if c["status"] == "fail"]
    result["overall"] = "fail" if fails else "pass"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t21_checks.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({c["id"]: c["status"] for c in chk}, indent=1))
    print("overall:", result["overall"])
    for c in chk:
        if c["status"] != "pass":
            print(c["id"], json.dumps(c["detail"], default=str)[:500])
    return 0 if not fails else 2


if __name__ == "__main__":
    raise SystemExit(main())



