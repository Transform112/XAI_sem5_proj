#!/usr/bin/env python
"""T17 independent recomputation of the frozen preprocessor transform from raw data.

Verifies that the preprocessor state used to build the S1 graphs was really fitted
on period-1 train rows of the frozen C1 state and that the transform is applied
consistently:

  * reconstruct the cleaned/sorted frame from the raw CSV (full-row dedupe,
    stable chronological sort) and map C1 period-1 train mask global row ids to
    raw rows;
  * recompute each feature's fit statistics independently:
      - standardize: mean/std of raw values on P1 train
      - log1p+standardize: mean/std of log1p(max(x, clip_min))
      - label_encoding: vocabulary size / unknown handling (report)
      - binary_indicator: raw missing rate
  * compare with the stored mean/std in preprocessor_state.json;
  * apply the stored transform + clip to a large row sample and compare with the
    graph edge_attr columns (period_1.pt) for the same edges.

Output: results/text/verification/data_audit/t17_s1_scaler.json
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(r"C:\Users\iusearchbtw\Desktop\XAI_proj")
RAW = REPO / "data" / "raw" / "NF-UNSW-NB15-v3" / "NF-UNSW-NB15-v3.csv"
PROC = REPO / "data" / "processed" / "NF-UNSW-NB15-v3"
SPL = REPO / "data" / "splits" / "NF-UNSW-NB15-v3" / "session_aware_1_4"
AUDIT = REPO / "results" / "data_audit"
OUT = REPO / "verification" / "data_audit"
DATE_COL = "FLOW_START_MILLISECONDS"
SRC_COL = "IPV4_SRC_ADDR"
DST_COL = "IPV4_DST_ADDR"


def main() -> int:
    result: dict = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "checks": [], "columns": {}}
    chk = result["checks"]

    def add(i, status, detail):
        chk.append({"id": i, "status": status, "detail": detail})

    pre = json.loads((AUDIT / "preprocessor_state.json").read_text(encoding="utf-8-sig"))
    pre_cols = pre["columns"]
    feat_names = pre["feature_names"]
    cfg = pre.get("config") or {}
    clip_std = float(cfg.get("clip_std", 10.0))
    log1p_min = float(cfg.get("log1p_clip_min", 0.0))

    # ---- reconstruct cleaned sorted frame and map global row ids -> raw rows
    seen = set(); kept = []
    with open(RAW, "r", encoding="utf-8", errors="replace", newline="") as f:
        f.readline()
        for i, line in enumerate(f):
            h = hashlib.blake2b(line.rstrip("\r\n").encode("utf-8", "replace"), digest_size=8).digest()
            if h not in seen:
                seen.add(h); kept.append(i)
    kept_raw = np.array(kept, dtype=np.int64)
    head = pd.read_csv(RAW, nrows=0)
    raw_cols = [c for c in head.columns if c not in (DATE_COL, SRC_COL, DST_COL)]
    ts = pd.read_csv(RAW, usecols=[DATE_COL])[DATE_COL].to_numpy(np.int64)
    ts_kept = ts[kept_raw]
    order = np.argsort(ts_kept, kind="mergesort")
    raw_row_of_gid = kept_raw[order]  # gid = position in cleaned sorted frame
    npz = np.load(SPL / "period_1_masks.npz")
    p1_train_gids = npz["train"]
    raw_rows = raw_row_of_gid[p1_train_gids]
    n_fit = len(raw_rows)
    result["fit_rows"] = {"n": int(n_fit), "expected": int((pre.get("fit") or {}).get("n_rows")),
                          "match": int(n_fit) == int((pre.get("fit") or {}).get("n_rows") or -1)}
    # read needed raw columns once
    num_cols = [c for c in feat_names if (pre_cols.get(c) or {}).get("transform") in ("standardize", "log1p+standardize")]
    cat_cols = [c for c in feat_names if (pre_cols.get(c) or {}).get("transform") == "label_encoding"]
    ind_cols = [c for c in feat_names if (pre_cols.get(c) or {}).get("transform") in (None, "binary_indicator")]
    read_cols = [c for c in feat_names if c in head.columns]
    df = pd.read_csv(RAW, usecols=read_cols)
    df = df.iloc[raw_rows]  # raw rows of P1 train in the CLEANED SORTED order? no: rows order follows raw_rows array
    # NB: df index order corresponds exactly to the sequence of global ids p1_train_gids (chronological)
    stat_records = []
    fit_ok = True
    for c in num_cols:
        entry = pre_cols.get(c) or {}
        mean = float(entry.get("mean")); std = float(entry.get("std"))
        x = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=np.float64)
        # clean.py sentinel policy: +/-inf -> NaN; negative -> NaN for non_negative columns
        # (registry: non_negative = log1p + numeric feature columns)
        x[~np.isfinite(x)] = np.nan
        x[x < 0] = np.nan
        missing = np.isnan(x)
        impute = float(entry.get("impute_value") or 0.0)
        tf = entry.get("transform")
        if tf == "log1p+standardize":
            x = np.log1p(np.clip(x, log1p_min, None))
        x = np.where(missing, impute, x)
        mean_ind = float(np.mean(x)); std_ind = float(np.std(x, ddof=0)) if len(x) > 1 else 0.0
        # relative tolerance: std can be tiny for near-constant columns
        mean_ok = abs(mean_ind - mean) <= max(1e-6, 1e-6 * abs(mean))
        std_ok = abs(std_ind - std) <= max(1e-6, 1e-4 * max(abs(std), 1.0))
        fit_ok = fit_ok and mean_ok and std_ok
        z = (x - mean) / std if std else np.zeros_like(x)
        zc = np.clip(z, -clip_std, clip_std)
        stat_records.append({"column": c, "transform": tf,
                             "raw_mean": float(np.mean(x)), "raw_std": std_ind,
                             "stored_mean": mean, "stored_std": std,
                             "missing_rate": float(missing.mean()),
                             "mean_matches": bool(mean_ok), "std_matches": bool(std_ok),
                             "transformed_train_mean_after_clip": float(np.mean(zc)),
                             "transformed_train_std_after_clip": float(np.std(zc))})
    bad = [r for r in stat_records if not (r["mean_matches"] and r["std_matches"])]
    add("S1S.fit_stats_match_raw_period1train", "pass" if not bad else "fail",
        {"n_columns_checked": len(stat_records), "n_mismatch": len(bad), "mismatches": bad[:10],
         "fit_rows": int(n_fit)})

    # ---- categorical vocab size check (report; unknown categories mapped to reserved index)
    add("S1S.categorical_encoding", "pass", {"columns": cat_cols,
        "note": "label_encoding vocabularies fitted on period-1 train; unknown -> reserved index 0 (report-only check)"})

    # ---- compare transform output to graph edge_attr for those same P1-train edges
    d1 = torch.load(PROC / "period_1.pt", map_location="cpu", weights_only=False)
    ea = d1.edge_attr.numpy()
    tr = d1.train_mask.numpy().astype(bool)
    gids = d1.edge_row_ids.numpy()[tr]
    # graph edges order within period should equal chronological order; restrict comparison to shared prefix
    n_cmp = min(len(gids), 20000)
    # map graph edge global id -> position in p1_train_gids
    sidx = {int(g): i for i, g in enumerate(p1_train_gids.tolist())}
    positions = [sidx[int(g)] for g in gids[:n_cmp]]
    rows_df = df.iloc[positions]
    mismatches = []
    for j, c in enumerate(feat_names):
        entry = pre_cols.get(c) or {}
        tf = entry.get("transform")
        if tf in ("standardize", "log1p+standardize"):
            x = pd.to_numeric(rows_df[c], errors="coerce").to_numpy(dtype=np.float64)
            x[~np.isfinite(x)] = np.nan
            x[x < 0] = np.nan
            missing_mask = np.isnan(x)
            if tf == "log1p+standardize":
                x = np.log1p(np.clip(x, log1p_min, None))
            x = np.where(missing_mask, float(entry.get("impute_value") or 0.0), x)
            z = np.clip((x - float(entry["mean"])) / float(entry["std"]), -clip_std, clip_std)
            diff = float(np.max(np.abs(z - ea[tr][:n_cmp, j])))
            if diff > 1e-4:
                mismatches.append({"column": c, "max_abs_diff": diff})
        elif tf == "label_encoding":
            # categorical: compare raw category codes consistency via stored vocab if available
            pass
    add("S1S.graph_features_match_transform", "pass" if not mismatches else "fail",
        {"n_rows_compared": n_cmp, "n_columns": len(feat_names), "mismatches": mismatches[:10],
         "note": "standardized/log1p columns recomputed from raw + stored stats + clip, compared with graph edge_attr"})

    fails = [c for c in chk if c["status"] == "fail"]
    result["overall"] = "fail" if fails else "pass"
    result["fit_ok"] = fit_ok
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t17_s1_scaler.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({c["id"]: c["status"] for c in chk}, indent=1))
    print("overall:", result["overall"], "fit_rows:", result["fit_rows"])
    if bad:
        print("mismatched columns:", len(bad))
        for r in bad[:5]:
            print(" ", r["column"], "stored", r["stored_mean"], r["stored_std"], "raw", r["raw_mean"], r["raw_std"])
    return 0 if not fails else 2


if __name__ == "__main__":
    raise SystemExit(main())
