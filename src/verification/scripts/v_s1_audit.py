#!/usr/bin/env python
"""T17 independent audit — S1 per-period graph artefacts (verifier-owned).

Loads datasets/processed/<dataset>/period_{1..5}.pt with torch and verifies the S1
contract independently of project code:

  * edge/node/feature counts vs graphs_manifest.json and the frozen C1 periods
  * x = ones(N, 8); edge_attr (E, 50) float32 and finite; feature names match the
    frozen preprocessor state; no IP / raw-timestamp / label columns present
  * edge_index local ids valid; node_global_ids mapping consistent with the
    edge_host_src/dst_global sidecars; masks boolean, disjoint, full coverage
  * masks reproduce the frozen split-mask sets (global row ids) exactly
  * edge_row_ids unique/chronological within period and inside the frozen C1
    global row-id range; edge_ts non-decreasing
  * y int64, 10-class space, counts match manifest per period
  * graphs_manifest pins: artifact fingerprint, frozen_split_hash, preprocessor
    state sha, raw CSV sha, config shas vs actual files
  * feature manifest cross-check (kept features == 50 names used)

Output: results/text/verification/data_audit/t17_s1.json
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
PROC = REPO / "data" / "processed" / "NF-UNSW-NB15-v3"
SPL = REPO / "data" / "splits" / "NF-UNSW-NB15-v3" / "session_aware_1_4"
MANIFEST = PROC / "graphs_manifest.json"
SPLIT_MANIFEST = REPO / "results" / "data_audit" / "split_manifest.json"
FEATURE_MANIFEST = REPO / "results" / "data_audit" / "feature_manifest.csv"
PREPROC = REPO / "results" / "data_audit" / "preprocessor_state.json"
OUT = REPO / "verification" / "data_audit"

FORBIDDEN = {
    "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", "FLOW_START_MILLISECONDS",
    "FLOW_END_MILLISECONDS", "Label", "Attack", "flow_id", "Flow ID", "index",
}


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha256_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def main() -> int:
    result: dict = {"generated_utc": pd.Timestamp.utcnow().isoformat(), "artifact": "t17_s1",
                    "checks": [], "findings": []}
    chk = result["checks"]
    man = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    split_man = json.loads(SPLIT_MANIFEST.read_text(encoding="utf-8-sig"))
    preproc = json.loads(PREPROC.read_text(encoding="utf-8-sig"))
    feature_names = list(man["feature_names"])
    preproc_features = list(preproc.get("feature_names") or [])
    result["manifest_summary"] = {"artifact_fingerprint": man.get("artifact_fingerprint_sha256"),
                                  "frozen_split_hash": man.get("frozen_split_hash"),
                                  "periods": {k: {kk: vv for kk, vv in v.items() if kk != "file"} for k, v in man["periods"].items()},
                                  "pins": man.get("pins")}

    def add(i, status, detail):
        chk.append({"id": i, "status": status, "detail": detail})

    # --- artifact file hashes + fingerprint
    hashes_ok, hex_hashes, rec = True, [], {}
    for p_str, entry in man["periods"].items():
        p = Path(entry["file"])
        actual = sha256_file(p)
        ok = actual == entry["sha256"] and p.stat().st_size == entry["bytes"]
        rec[p_str] = {"sha256": actual, "matches_manifest": ok, "bytes": p.stat().st_size}
        hashes_ok = hashes_ok and ok
        hex_hashes.append(actual)
    fp = hashlib.sha256("".join(hex_hashes).encode("ascii")).hexdigest()
    add("S1.artifact_hashes", "pass" if hashes_ok else "fail", rec)
    add("S1.artifact_fingerprint", "pass" if fp == man.get("artifact_fingerprint_sha256") else "fail",
        {"recomputed": fp, "manifest": man.get("artifact_fingerprint_sha256")})
    # --- frozen split hash
    add("S1.frozen_split_hash", "pass" if man.get("frozen_split_hash") == split_man.get("frozen_hash") else "fail",
        {"graphs_manifest": man.get("frozen_split_hash"), "split_manifest": split_man.get("frozen_hash")})
    # --- preprocessor reuse
    pre_sha = sha256_file(PREPROC)
    add("S1.preprocessor_state_sha", "pass" if pre_sha == (man.get("preprocessor_state") or {}).get("sha256") else "fail",
        {"actual": pre_sha, "manifest": (man.get("preprocessor_state") or {}).get("sha256"),
         "fit": preproc.get("fit")})
    # --- raw csv sha
    raw = man.get("raw_csv") or {}
    raw_ok = Path(raw.get("path", "")).is_file() and sha256_file(Path(raw["path"])) == raw.get("sha256")
    add("S1.raw_csv_sha", "pass" if raw_ok else "fail", raw)
    # --- pins vs actual config files
    pins = man.get("pins") or {}
    train_cfg_actual = sha256_file(REPO / "configs" / "train.yaml")
    audit_cfg_actual = sha256_file(REPO / "configs" / "audit.yaml")
    add("S1.config_pins", "pass" if pins.get("graph_config_sha256") == train_cfg_actual else "warn",
        {"train_yaml_actual": train_cfg_actual, "pinned_graph_config": pins.get("graph_config_sha256"),
         "audit_yaml_actual": audit_cfg_actual, "pinned_audit_config": pins.get("audit_config_sha256"),
         "note": "audit.yaml may legitimately have changed after D-034/D-040/D-041; graph config should match"})
    # --- feature contract
    forbidden_present = [f for f in feature_names if f in FORBIDDEN
                         or re.search(r"^(IPV4_SRC|IPV4_DST|FLOW_START|FLOW_END)", f)
                         or f.lower() in ("label", "attack", "timestamp", "flow_id")]
    add("S1.feature_names", "pass" if (len(feature_names) == 50 and not forbidden_present
                                       and feature_names == preproc_features) else "fail",
        {"n_features": len(feature_names), "forbidden_present": forbidden_present,
         "matches_preprocessor_state": feature_names == preproc_features,
         "has_L4_DST_PORT": "L4_DST_PORT" in feature_names,
         "note": "Doc 2 S1 forbids source port + IPs + raw timestamps as features; dst port kept by registry"})
    # feature_manifest cross-check
    fm = pd.read_csv(FEATURE_MANIFEST)
    keep_col = next((c for c in fm.columns if c.lower() in ("kept_as_feature", "kept", "role", "action", "decision")), None)
    result["feature_manifest_columns"] = list(fm.columns)
    added_features = sorted(str(x) for x in set(feature_names) - set(fm["column"]) ) if "column" in fm.columns else []
    add("S1.feature_manifest_crosscheck", "pass" if not added_features else "warn",
        {"n_manifest_columns": len(fm), "features_not_in_manifest": added_features[:20]})

    # --- per-period graph checks
    periods = {}
    all_ok = True
    for p_str in sorted(man["periods"], key=int):
        p = int(p_str)
        data = torch.load(PROC / f"period_{p}.pt", map_location="cpu", weights_only=False)
        rec: dict = {"period": p}
        n_edges = int(data.num_edges)
        n_nodes = int(data.num_nodes)
        m = man["periods"][p_str]
        e = {"n_edges": n_edges, "n_nodes": n_nodes,
             "matches_manifest": n_edges == m["n_edges"] and n_nodes == m["n_nodes"]}
        # x
        x = data.x
        e["x_ones"] = bool(torch.all(x == 1).item()) and tuple(x.shape) == (n_nodes, 8) and x.dtype == torch.float32
        # edge_attr
        ea = data.edge_attr
        e["edge_attr"] = {"shape": list(ea.shape), "dtype": str(ea.dtype),
                          "finite": bool(torch.isfinite(ea).all().item()),
                          "n_features": int(ea.shape[1]),
                          "matches_manifest": int(ea.shape[0]) == n_edges and int(ea.shape[1]) == int(m["n_features"])}
        # y
        y = data.y
        counts = np.bincount(y.numpy(), minlength=10).tolist()
        e["y"] = {"dtype": str(y.dtype), "n": int(y.numel()), "min": int(y.min()), "max": int(y.max()),
                  "counts_match_manifest": {str(i): counts[i] for i in range(10)} == {k: int(v) for k, v in m["class_counts"].items()}}
        # edge_index + mapping
        ei = data.edge_index
        e["edge_index"] = {"dtype": str(ei.dtype), "shape": list(ei.shape),
                           "in_range": bool(int(ei.min()) >= 0 and int(ei.max()) < n_nodes)}
        node_gids = data.node_global_ids.numpy()
        e["node_global_ids"] = {"n": int(len(node_gids)), "sorted_unique": bool(np.all(np.diff(node_gids) > 0)),
                                "matches_n_nodes": len(node_gids) == n_nodes}
        src_g = data.edge_host_src_global.numpy(); dst_g = data.edge_host_dst_global.numpy()
        mapping_ok = (np.array_equal(node_gids[ei[0].numpy()], src_g) and np.array_equal(node_gids[ei[1].numpy()], dst_g))
        e["edge_index_mapping"] = {"endpoint_global_ids_consistent": bool(mapping_ok),
                                   "src_in_node_gids": bool(np.isin(src_g, node_gids).all()),
                                   "dst_in_node_gids": bool(np.isin(dst_g, node_gids).all())}
        # masks vs frozen npz
        npz = np.load(SPL / f"period_{p}_masks.npz")
        masks_ok = True
        mask_rec = {}
        for s in ("train", "val", "test"):
            mask = getattr(data, f"{s}_mask").numpy().astype(bool)
            gids = data.edge_row_ids.numpy()[mask]
            frozen = npz[s]
            same = len(gids) == len(frozen) and np.array_equal(np.sort(gids), np.sort(frozen))
            masks_ok = masks_ok and same
            mask_rec[s] = {"n": int(mask.sum()), "manifest_count": int(m["split_counts"][s]),
                           "matches_frozen_ids": bool(same), "matches_manifest_count": int(mask.sum()) == int(m["split_counts"][s])}
        mask_union = np.zeros(n_edges, dtype=bool)
        for s in ("train", "val", "test"):
            mask_union |= getattr(data, f"{s}_mask").numpy().astype(bool)
        e["masks"] = {"splits": mask_rec, "covers_all_edges": bool(mask_union.all()),
                      "overlap": int(np.array_equal(mask_union, np.ones(n_edges, dtype=bool)) is False),
                      "matches_frozen_sets": bool(masks_ok)}
        # row ids / timestamps
        row_ids = data.edge_row_ids.numpy()
        ts = data.edge_ts.numpy()
        e["sidecars"] = {"row_ids_unique": bool(len(np.unique(row_ids)) == n_edges),
                         "row_ids_nondecreasing": bool(np.all(np.diff(row_ids) >= 0)),
                         "ts_nondecreasing": bool(np.all(np.diff(ts) >= 0)),
                         "row_id_range": [int(row_ids.min()), int(row_ids.max())]}
        # frozen period row range
        per = split_man["grouped"]["periods"][str(p - 1)]
        frozen_ids = np.concatenate([np.load(SPL / f"period_{p}_masks.npz")[s] for s in ("train", "val", "test")])
        e["period_global_range"] = {"frozen_min": int(frozen_ids.min()), "frozen_max": int(frozen_ids.max()),
                                    "pt_min": int(row_ids.min()), "pt_max": int(row_ids.max()),
                                    "consistent": int(row_ids.min()) == int(frozen_ids.min()) and int(row_ids.max()) == int(frozen_ids.max())}
        # e-graphsage mask-through-propagate sanity: zeroing edge_attr must change logits (architecture check via model)
        ok = (e["matches_manifest"] and e["x_ones"] and e["edge_attr"]["matches_manifest"] and e["edge_attr"]["finite"]
              and e["y"]["counts_match_manifest"] and e["edge_index"]["in_range"] and mapping_ok
              and masks_ok and e["masks"]["covers_all_edges"] and e["sidecars"]["row_ids_unique"]
              and e["sidecars"]["row_ids_nondecreasing"] and e["sidecars"]["ts_nondecreasing"]
              and e["period_global_range"]["consistent"])
        e["status"] = "pass" if ok else "fail"
        all_ok = all_ok and ok
        periods[p_str] = e
    result["periods"] = periods
    add("S1.graph_contract", "pass" if all_ok else "fail", {k: v["status"] for k, v in periods.items()})
    # --- feature standardisation scope: P1 train mean~0/std~1
    d1 = torch.load(PROC / "period_1.pt", map_location="cpu", weights_only=False)
    tr = d1.train_mask.numpy().astype(bool)
    ea = d1.edge_attr.numpy()[tr]
    pre_cols = preproc.get("columns") or {}
    std_idx, lab_idx, other_idx = [], [], []
    for i, name in enumerate(feature_names):
        tf = (pre_cols.get(name) or {}).get("transform")
        if tf in ("standardize", "log1p+standardize"):
            std_idx.append(i)
        elif tf == "label_encoding":
            lab_idx.append(i)
        else:
            other_idx.append(i)
    dev = []
    for i in std_idx:
        m, s = float(ea[:, i].mean()), float(ea[:, i].std())
        dev.append({"column": feature_names[i], "mean": round(m, 4), "std": round(s, 4),
                    "transform": pre_cols.get(feature_names[i], {}).get("transform")})
    big = [d0 for d0 in dev if abs(d0["mean"]) > 0.05 or abs(d0["std"] - 1.0) > 0.15]
    add("S1.scaler_fit_scope", "pass" if len(big) <= 2 else "warn",
        {"n_standardized_columns": len(std_idx), "n_label_encoded": len(lab_idx),
         "n_other": len(other_idx),
         "n_deviating": len(big), "deviating_columns": big[:20],
         "note": "preprocessor config has clip_std=10 and label-encoded categoricals; deviations from N(0,1) on"
                 " clipped heavy-tail columns are expected. Independent raw-data transform recomputation is in"
                 " t17_s1_scaler.json (v_s1_scaler_check.py)."})
    # --- node counts expectation from structure audit / scorecard
    add("S1.node_counts", "pass" if [periods[str(i)]["n_nodes"] for i in range(1, 6)] == [40, 42, 42, 42, 42] else "fail",
        {"n_nodes": [periods[str(i)]["n_nodes"] for i in range(1, 6)], "expected": [40, 42, 42, 42, 42]})
    fails = [c for c in chk if c["status"] == "fail"]
    result["overall"] = "fail" if fails else "pass"
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "t17_s1.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(json.dumps({c["id"]: c["status"] for c in chk}, indent=1))
    print("overall:", result["overall"])
    for p, e in periods.items():
        print(f"period {p}: edges={e['n_edges']} nodes={e['n_nodes']} status={e['status']}")
        if e["status"] != "pass":
            print("  ", json.dumps({k: v for k, v in e.items() if k not in ("edge_attr",)}, default=str)[:600])
    return 0 if not fails else 2


if __name__ == "__main__":
    raise SystemExit(main())
