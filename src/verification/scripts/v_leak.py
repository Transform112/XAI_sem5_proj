#!/usr/bin/env python
"""Independent leakage audit of the frozen split masks (verifier-owned).

Artifacts audited:
  results/text/data_audit/split_manifest.json
  datasets/splits/<dataset>/{period_t_masks.npz}
  results/text/data_audit/preprocessor_state.json

IMPORTANT INDEX CONVENTION (verified independently): the frozen mask arrays store
GLOBAL row ids of the cleaned, chronologically sorted frame (period 1 covers
0..470120, period 2 covers 470121..940242, ...). The script verifies this from the
manifest edge counts and the observed mask ranges before using it.

Checks (docs/09 section 7):
  LK-1  mask array sha256 vs recorded manifest sha256 + edge counts
  LK-2  per-period disjointness and full coverage in the global id range
  LK-3  period id ranges / timestamps ordered, label counts vs manifest
  LK-4  no test edge of period t in a training mask of a later period
  LK-5  near-duplicate canonical keys do not cross splits (config key definition)
  LK-6  host-pair overlap across splits (reported, not hard-failed)
  LK-7  preprocessor fit scope (period 1 train)
  LK-8  label distribution per split (chi-square train vs test)
  LK-9  adversarial validation AUC (raw numeric features, own classifier)

The cleaned frame is reconstructed from the raw CSV without project code:
full-row duplicates dropped (keep first in raw order), then stable chronological
sort by FLOW_START_MILLISECONDS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

VERIFIER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VERIFIER_ROOT.parents[1]
RAW = REPO_ROOT / "data" / "raw"
SPLITS = REPO_ROOT / "data" / "splits"
MANIFEST = REPO_ROOT / "results" / "data_audit" / "split_manifest.json"
PREPROC = REPO_ROOT / "results" / "data_audit" / "preprocessor_state.json"
ENG_IDENT = REPO_ROOT / "results" / "data_audit"
OUT = VERIFIER_ROOT / "data_audit"
DATE_COL = "FLOW_START_MILLISECONDS"
SRC_COL = "IPV4_SRC_ADDR"
DST_COL = "IPV4_DST_ADDR"
PORT_COLS = ["L4_SRC_PORT", "L4_DST_PORT"]
LABEL_COL = "Attack"
SPLITS_NAMES = ("train", "val", "test")
DROP_FEATURE = {DATE_COL, "FLOW_END_MILLISECONDS", SRC_COL, DST_COL, "L4_SRC_PORT", "L4_DST_PORT",
                "Label", "Attack"}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_arr(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def load_mask_files(dataset: str, subdir: str = "") -> dict[int, dict[str, np.ndarray]]:
    d = (SPLITS / dataset / subdir) if subdir else (SPLITS / dataset)
    out: dict[int, dict[str, np.ndarray]] = {}
    for p in sorted(d.glob("period_*_masks.npz")):
        idx = int(p.name.split("_")[1])
        z = np.load(p)
        out[idx] = {k: z[k].astype(np.int64) for k in z.files}
    return out


def scan_duplicates(csv_path: Path) -> tuple[int, np.ndarray, np.ndarray]:
    """Full-row duplicate detection (whole raw line). Returns (raw_rows, kept_raw_idx, ts_unsorted_flaginfo)."""
    seen: dict[int, int] = {}
    kept: list[int] = []
    raw_rows = 0
    with open(csv_path, "r", encoding="utf-8", errors="replace", newline="") as f:
        f.readline()
        for i, line in enumerate(f):
            raw_rows += 1
            h = int.from_bytes(hashlib.blake2b(line.rstrip("\r\n").encode("utf-8", "replace"), digest_size=8).digest(), "big")
            if h not in seen:
                seen[h] = i
                kept.append(i)
    return raw_rows, np.array(kept, dtype=np.int64), np.array(sorted(set(range(raw_rows)) - set(kept)), dtype=np.int64)


def reconstruct(csv_path: Path) -> tuple[pd.DataFrame, dict, np.ndarray, np.ndarray]:
    raw_rows, kept_raw, dup_rows = scan_duplicates(csv_path)
    usecols = [DATE_COL, SRC_COL, DST_COL, *PORT_COLS, "PROTOCOL", LABEL_COL]
    df = pd.read_csv(csv_path, usecols=usecols, dtype={SRC_COL: "str", DST_COL: "str", LABEL_COL: "str"})
    keep = np.ones(len(df), dtype=bool)
    if len(dup_rows):
        keep[dup_rows] = False
    df = df.loc[keep].reset_index(drop=True)
    df = df.sort_values(DATE_COL, kind="mergesort").reset_index(drop=True)
    info = {"raw_rows": int(raw_rows), "duplicate_rows_removed": int(len(dup_rows)),
            "rows_after_dedupe": int(len(df)), "sort": "stable mergesort by " + DATE_COL,
            "monotonic_nondecreasing": bool((df[DATE_COL].diff().dropna() >= 0).all())}
    return df, info, kept_raw, dup_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="NF-UNSW-NB15-v3")
    ap.add_argument("--adv-sample", type=int, default=25000)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--variant", default="grouped")
    ap.add_argument("--masks-dir", default="", help="subdirectory under datasets/splits/<dataset> (e.g. session_aware_1_4)")
    args = ap.parse_args()
    dataset = args.dataset
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    top = manifest[args.variant] if args.variant != "grouped" else manifest["grouped"]
    masks = load_mask_files(dataset, args.masks_dir)
    result: dict = {"dataset": dataset, "variant": args.variant,
                    "generated_utc": utc_now(), "script": "verification/src/scripts/v_leak.py",
                    "manifest": str(MANIFEST.relative_to(REPO_ROOT)),
                    "manifest_seed": top.get("seed"),
                    "manifest_frozen_hash": top.get("frozen_hash"),
                    "mask_files": {str(k): str((((SPLITS / dataset / args.masks_dir) if args.masks_dir else (SPLITS / dataset)) / f"period_{k}_masks.npz").relative_to(REPO_ROOT)) for k in masks},
                    "masks_dir": args.masks_dir or ".",
                    "checks": []}
    if not masks:
        result["status"] = "blocked"
        result["detail"] = "no mask npz files found"
        (out_dir / f"leakage_{dataset}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 2
    periods_map = top["periods"]

    # ---------------- LK-1 hashes + period sizes
    lk1 = {"id": "LK-1.mask_hashes", "periods": {}, "status": "pass"}
    period_sizes: dict[int, int] = {}
    for p, m in sorted(masks.items()):
        manp = periods_map[str(p - 1)]
        entry = {}
        for s in SPLITS_NAMES:
            a = m[s]
            ok = sha256_arr(a) == manp[s]["sha256"] and int(len(a)) == int(manp[s]["n_edges"])
            entry[s] = {"n": int(len(a)), "min": int(a.min()), "max": int(a.max()),
                        "sha256": sha256_arr(a), "matches_manifest": bool(ok)}
            if not ok:
                lk1["status"] = "fail"
        period_sizes[p] = int(sum(int(manp[s]["n_edges"]) for s in SPLITS_NAMES))
        lk1["periods"][str(p)] = entry
    result["checks"].append(lk1)
    offsets = np.cumsum([0] + [period_sizes[p] for p in sorted(period_sizes)]).astype(np.int64)
    total = int(offsets[-1])
    result["index_convention"] = {
        "note": "mask arrays store global row ids of the cleaned sorted frame; verified below",
        "period_offsets": offsets.tolist(), "total_rows": total}

    # ---------------- LK-2 coverage / disjointness in global range
    lk2 = {"id": "LK-2.disjointness_coverage", "periods": {}, "status": "pass"}
    for p, m in sorted(masks.items()):
        off = int(offsets[p - 1]); n = period_sizes[p]
        allidx = np.concatenate([m[s] for s in SPLITS_NAMES])
        overlaps = (len(np.intersect1d(m["train"], m["val"])) + len(np.intersect1d(m["train"], m["test"]))
                    + len(np.intersect1d(m["val"], m["test"])))
        uniq = np.unique(allidx)
        covered = bool(len(uniq) == n and int(uniq[0]) == off and int(uniq[-1]) == off + n - 1)
        entry = {"global_range": [off, off + n - 1], "n_edges": n, "unique": int(len(uniq)),
                 "covers_own_global_range": covered, "pairwise_overlap": int(overlaps),
                 "min_lt_offset": int(allidx.min()) < off, "max_ge_end": int(allidx.max()) >= off + n}
        entry["status"] = "pass" if covered and overlaps == 0 and not entry["min_lt_offset"] and not entry["max_ge_end"] else "fail"
        if entry["status"] != "pass":
            lk2["status"] = "fail"
        lk2["periods"][str(p)] = entry
    result["checks"].append(lk2)

    # ---------------- reconstruction from raw
    csv_path = list((RAW / dataset).rglob("*.csv"))[0]
    df, recon, kept_raw, dup_rows = reconstruct(csv_path)
    result["reconstruction"] = recon
    result["checks"].append({"id": "LK-3a.reconstruction_rows",
                             "status": "pass" if len(df) == total and recon["rows_after_dedupe"] == total else "fail",
                             "detail": {"raw_rows": recon["raw_rows"], "removed_duplicates": recon["duplicate_rows_removed"],
                                        "cleaned_rows": len(df), "manifest_total": total}})
    class_order = ["Benign", "Analysis", "Backdoor", "DoS", "Exploits", "Fuzzers", "Generic",
                   "Reconnaissance", "Shellcode", "Worms"]
    eng_ident = ENG_IDENT / f"identity_{dataset}.json"
    if eng_ident.is_file():
        try:
            class_order = (json.loads(eng_ident.read_text(encoding="utf-8-sig")).get("labels") or {}).get("class_order_expected") or class_order
        except Exception:  # noqa: BLE001
            pass
    cls2id = {c: i for i, c in enumerate(class_order)}
    mapped = df[LABEL_COL].map(cls2id)
    unmapped = int(mapped.isna().sum())
    labels = mapped.fillna(-1).to_numpy(dtype=np.int64)
    ts = df[DATE_COL].to_numpy(np.int64)
    if unmapped:
        result["checks"].append({"id": "LK-3a.label_mapping", "status": "warn", "detail": {"unmapped_rows": unmapped}})

    # ---------------- LK-3 period ranges / label counts
    lk3 = {"id": "LK-3.period_ranges_labels", "periods": {}, "status": "pass"}
    for p, m in sorted(masks.items()):
        off = int(offsets[p - 1]); end = off + period_sizes[p]
        manp = periods_map[str(p - 1)]
        ts_p = ts[off:end]
        entry = {"global_range": [off, end - 1],
                 "ts_min_iso": pd.Timestamp(int(ts_p.min()), unit="ms", tz="UTC").isoformat(),
                 "ts_max_iso": pd.Timestamp(int(ts_p.max()), unit="ms", tz="UTC").isoformat()}
        entry["manifest_first_last_ok"] = all(
            int(m[s].min()) == int(manp[s]["first_row_id"]) and int(m[s].max()) == int(manp[s]["last_row_id"])
            for s in SPLITS_NAMES)
        for s in SPLITS_NAMES:
            cnt = np.bincount(labels[m[s]], minlength=len(class_order))
            recorded = manp[s]["class_counts"]
            rec = {str(i): int(v) for i, v in recorded.items()}
            mine = {str(i): int(cnt[i]) for i in range(len(class_order)) if cnt[i]}
            entry[f"{s}_class_counts_match"] = (mine == rec)
            if not entry[f"{s}_class_counts_match"]:
                diff = {k: {"manifest": rec.get(k), "verifier": mine.get(k)} for k in set(rec) | set(mine)
                        if rec.get(k) != mine.get(k)}
                entry[f"{s}_class_count_diffs"] = diff
        entry["status"] = "pass" if (entry["manifest_first_last_ok"] and all(
            entry[f"{s}_class_counts_match"] for s in SPLITS_NAMES)) else "fail"
        if entry["status"] != "pass":
            lk3["status"] = "fail"
        lk3["periods"][str(p)] = entry
    ordered = True
    for p in range(2, len(masks) + 1):
        if lk3["periods"][str(p)]["ts_min_iso"] < lk3["periods"][str(p - 1)]["ts_max_iso"]:
            ordered = False
    lk3["periods_time_ordered"] = ordered
    if not ordered:
        lk3["status"] = "fail"
    result["checks"].append(lk3)

    # ---------------- LK-4 test never in later training
    test_sets = {p: set(masks[p]["test"].tolist()) for p in masks}
    train_sets = {p: set(masks[p]["train"].tolist()) for p in masks}
    lk4 = {"id": "LK-4.test_in_later_training", "checks": [], "status": "pass"}
    for p in sorted(masks):
        for p2 in sorted(masks):
            if p2 > p:
                ov = len(test_sets[p] & train_sets[p2])
                lk4["checks"].append({"test_period": p, "train_period": p2, "overlap": ov})
                if ov:
                    lk4["status"] = "fail"
    result["checks"].append(lk4)

    # ---------------- LK-5 near-duplicate groups crossing splits (global ids)
    key = (df[SRC_COL].astype(str) + "|" + df[DST_COL].astype(str) + "|"
           + df[PORT_COLS[0]].astype(str) + "|" + df[PORT_COLS[1]].astype(str) + "|"
           + df["PROTOCOL"].astype(str) + "|" + (ts // 1000).astype(str))
    key_codes = pd.factorize(key)[0]
    split_code = np.full(total, -1, dtype=np.int64)
    for p, m in sorted(masks.items()):
        for k, s in enumerate(SPLITS_NAMES):
            split_code[m[s]] = k
    lk5 = {"id": "LK-5.near_duplicate_cross_split", "status": "pass", "periods": {}}
    order = np.argsort(key_codes, kind="mergesort")
    cd = key_codes[order]; sc = split_code[order]
    starts = np.r_[True, cd[1:] != cd[:-1]]
    gid_arr = np.cumsum(starts) - 1
    n_groups = int(gid_arr[-1]) + 1
    first = np.full(n_groups, -1, dtype=np.int64)
    bad = np.zeros(n_groups, dtype=bool)
    for i in range(len(cd)):
        g = int(gid_arr[i]); sp = int(sc[i])
        if first[g] == -1:
            first[g] = sp
        elif sp != first[g]:
            bad[g] = True
    lk5["groups_total"] = n_groups
    lk5["groups_crossing_splits"] = int(bad.sum())
    lk5["status"] = "pass" if int(bad.sum()) == 0 else "fail"
    if bad.any():
        sizes_all = np.bincount(gid_arr)
        lk5["sample_crossing_group_sizes"] = sizes_all[np.nonzero(bad)[0]][:10].tolist()
    else:
        lk5["sample_crossing_group_sizes"] = []
    result["checks"].append(lk5)

    # ---------------- LK-6 host-pair overlap (reported)
    pair_codes = pd.factorize(df[SRC_COL].astype(str) + "|" + df[DST_COL].astype(str))[0]
    lk6 = {"id": "LK-6.host_pair_overlap_across_splits", "periods": {}, "status": "info"}
    for p, m in sorted(masks.items()):
        sets = {s: set(pair_codes[m[s]].tolist()) for s in SPLITS_NAMES}
        tr, va, te = sets["train"], sets["val"], sets["test"]
        lk6["periods"][str(p)] = {
            "n_pairs_train": len(tr), "n_pairs_val": len(va), "n_pairs_test": len(te),
            "train_val_overlap": len(tr & va), "train_test_overlap": len(tr & te), "val_test_overlap": len(va & te),
            "test_pairs_share_in_train": round(len(tr & te) / len(te), 6) if te else None,
        }
    result["checks"].append(lk6)

    # ---------------- LK-7 preprocessor fit scope
    lk7 = {"id": "LK-7.preprocessor_fit_scope", "status": "warn"}
    if PREPROC.is_file():
        pre = json.loads(PREPROC.read_text(encoding="utf-8-sig"))
        fit = pre.get("fit") or {}
        fitted = fit.get("n_rows") if isinstance(fit, dict) else None
        expected = int(periods_map["0"]["train"]["n_edges"])
        lk7["detail"] = {"fit_block": fit, "expected_period1_train_rows": expected,
                         "dataset": pre.get("dataset"),
                         "manifest_rule": manifest.get("preprocessing_rule")}
        lk7["status"] = "pass" if (fitted == expected and "period-1 train" in str(fit.get("note"))) else "warn"
    result["checks"].append(lk7)

    # ---------------- LK-8 label distribution chi-square
    from scipy.stats import chi2 as chi2dist
    lk8 = {"id": "LK-8.label_distribution", "periods": {}, "status": "pass"}
    for p, m in sorted(masks.items()):
        obs = np.array([[int((labels[m["train"]] == c).sum()), int((labels[m["test"]] == c).sum())]
                        for c in range(len(class_order))], dtype=float)
        obs = obs[obs.sum(axis=1) > 0]
        exp = obs.sum(axis=1, keepdims=True) @ obs.sum(axis=0, keepdims=True) / obs.sum()
        chi2 = float(((obs - exp) ** 2 / np.where(exp == 0, 1, exp)).sum())
        dof = max(1, len(obs) - 1)
        pval = float(chi2dist.sf(chi2, dof))
        zero_test = [class_order[i] for i in range(len(class_order)) if int((labels[m["test"]] == i).sum()) == 0]
        lk8["periods"][str(p)] = {"chi2": chi2, "dof": dof, "p_value": pval, "classes_absent_from_test": zero_test,
                                  "status": "pass" if pval >= 0.01 else "fail"}
        if pval < 0.01:
            lk8["status"] = "fail"
    result["checks"].append(lk8)

    # ---------------- LK-9 adversarial validation on raw numeric features
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import roc_auc_score
        head = pd.read_csv(csv_path, nrows=0)
        feat_cols = [c for c in head.columns if c not in DROP_FEATURE and not c.startswith("Unnamed")]
        order_all = np.argsort(ts, kind="mergesort")  # only valid if reconstruction order == sorted order (it is)
        inv = np.full(recon["raw_rows"], -1, dtype=np.int64)
        inv[kept_raw[order_all]] = np.arange(len(order_all), dtype=np.int64)
        rng = np.random.default_rng(20261101)
        lk9 = {"id": "LK-9.adversarial_auc", "periods": {}, "status": "pass", "features_used": len(feat_cols)}
        for p, m in sorted(masks.items()):
            tin = rng.choice(m["train"], size=min(args.adv_sample, len(m["train"])), replace=False)
            tout = rng.choice(m["test"], size=min(args.adv_sample, len(m["test"])), replace=False)
            want = np.concatenate([tin, tout])
            raw_rows = kept_raw[order_all[want]]
            raw_set = {int(r): i for i, r in enumerate(raw_rows)}
            rows: list[list[float]] = [None] * len(want)  # type: ignore[list-item]
            for chunk in pd.read_csv(csv_path, usecols=feat_cols, chunksize=500_000):
                start = int(chunk.index[0])
                hits = [r for r in range(start, start + len(chunk)) if r in raw_set]
                if not hits:
                    continue
                local = np.array([r - start for r in hits], dtype=np.int64)
                arr = chunk.iloc[local].to_numpy(dtype=np.float64)
                for j, r in enumerate(hits):
                    rows[raw_set[r]] = arr[j]
            X = np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
            yb = np.r_[np.zeros(len(tin)), np.ones(len(tout))]
            clf = HistGradientBoostingClassifier(max_iter=60, random_state=42)
            proba = cross_val_predict(clf, X, yb, cv=3, method="predict_proba")[:, 1]
            auc = float(roc_auc_score(yb, proba))
            lk9["periods"][str(p)] = {"auc": auc, "n_train": int(len(tin)), "n_test": int(len(tout)),
                                      "threshold": 0.65, "status": "pass" if auc < 0.65 else "fail"}
            if auc >= 0.65:
                lk9["status"] = "fail"
        result["checks"].append(lk9)
    except Exception as exc:  # noqa: BLE001
        result["checks"].append({"id": "LK-9.adversarial_auc", "status": "blocked", "detail": str(exc)})

    fails = [c for c in result["checks"] if c.get("status") == "fail"]
    result["overall"] = "fail" if fails else ("pass" if all(c.get("status") in ("pass", "info") for c in result["checks"]) else "pass_with_warnings")
    suffix = f"_{args.masks_dir.replace('/', '_')}" if args.masks_dir else ""
    out_path = out_dir / f"leakage_{dataset}{suffix}.json"
    out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"wrote {out_path}")
    for c in result["checks"]:
        print(f"  {c['id']}: {c.get('status')}")
    print("overall:", result["overall"])
    return 0 if not fails else 2


if __name__ == "__main__":
    raise SystemExit(main())
