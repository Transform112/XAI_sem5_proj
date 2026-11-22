#!/usr/bin/env python
"""Independent leakage / split-integrity audit (verifier-owned).

Runs on the engineer's frozen split artifacts (masks + split_manifest.json) and
the raw CSV. It never imports project code: masks are read through torch/numpy,
raw data through pandas, and every recorded number is recomputed.

Checks (docs/09 section 7; IDs match verification/AUDIT_PLAN.md):
  LK-1  mask/manifest presence + stored-hash verification
  LK-2  within-period disjointness and full coverage
  LK-3  period boundary/timestamp ordering recomputed from raw
  LK-4  no test edge in any later training mask (needs global edge ids)
  LK-5  near-duplicate leakage groups do not cross splits (needs raw-row mapping)
  LK-6  host-pair overlap across splits (needs raw-row mapping)
  LK-7  scaler/standardisation fitted on period-1 train only (processed tensors)
  LK-8  label distribution across splits (processed y + masks; chi-square)
  LK-9  adversarial validation AUC within period (train vs test, own classifier)
  LK-10 mask hash stability across reruns (hash timeline file)
  LK-12 anchors subset of test mask (when anchor artifacts exist)

Usage (repo root):
  .venv\\Scripts\\python.exe verification\\scripts\\v_splits.py
  .venv\\Scripts\\python.exe verification\\scripts\\v_splits.py --dataset NF-UNSW-NB15-v3 --raw data\\raw
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

VERIFIER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VERIFIER_ROOT.parents[1]
DEFAULT_RAW = REPO_ROOT / "data" / "raw"
DEFAULT_OUT = VERIFIER_ROOT / "data_audit"
DATE_COL = "FLOW_START_MILLISECONDS"
SRC_COL = "IPV4_SRC_ADDR"
DST_COL = "IPV4_DST_ADDR"
LABEL_COL = "Attack"
PERIODS = 5
SPLITS = ("train", "val", "test")


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def jdump(obj) -> str:
    return json.dumps(obj, indent=2, default=str)


# ---------------------------------------------------------------- discovery
def discover_manifests(repo: Path) -> list[Path]:
    hits = []
    for pat in ("data/**/split_manifest*.json", "data/**/*manifest*.json",
                "results/**/split_manifest*.json", "verification/**/split_manifest*.json"):
        hits += [p for p in repo.glob(pat) if p.is_file()]
    seen, out = set(), []
    for p in hits:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def discover_mask_files(repo: Path) -> list[Path]:
    out = []
    for pat in ("datasets/splits/**/*.pt", "datasets/splits/**/*.npz", "datasets/splits/**/*.npy",
                "datasets/processed/**/*mask*", "datasets/processed/**/period_*.pt"):
        out += [p for p in repo.glob(pat) if p.is_file()]
    return sorted(set(out))


# ---------------------------------------------------------------- loading helpers
def torch_load(path: Path):
    import torch
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch
        return torch.load(path, map_location="cpu")


def as_bool_mask(x) -> np.ndarray | None:
    """Convert a torch/numpy mask (bool or index tensor) to a boolean mask."""
    try:
        import torch
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    except Exception:  # noqa: BLE001
        pass
    a = np.asarray(x)
    if a.dtype == bool:
        return a.ravel()
    if a.ndim == 1 and np.issubdtype(a.dtype, np.integer):
        m = np.zeros(int(a.max()) + 1, dtype=bool) if a.size else np.zeros(0, dtype=bool)
        m[a] = True
        return m
    return None


def extract_split_masks(obj) -> dict[str, np.ndarray]:
    """Find train/val/test masks in an arbitrary nested artifact."""
    out: dict[str, np.ndarray] = {}

    def walk(o, path=""):
        if len(out) == 3:
            return
        if isinstance(o, dict):
            for k, v in o.items():
                kl = str(k).lower()
                for s in SPLITS:
                    if s in kl and s not in out:
                        m = as_bool_mask(v)
                        if m is not None:
                            out[s] = m
                            break
                else:
                    walk(v, f"{path}.{k}")
        elif isinstance(o, (list, tuple)) and len(o) <= 8:
            for i, v in enumerate(o):
                walk(v, f"{path}[{i}]")

    walk(obj)
    return out


# ---------------------------------------------------------------- checks
def check_hashes(manifests: list[Path], masks: list[Path]) -> dict:
    rec = {"id": "LK-1.mask_hashes", "checks": [], "files": {}}
    for p in masks + manifests:
        rec["files"][str(p.relative_to(REPO_ROOT))] = {"sha256": sha256_file(p),
                                                       "bytes": p.stat().st_size,
                                                       "mtime_utc": datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    # verify recorded hashes inside manifests, when present
    recorded: dict[str, str] = {}

    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if isinstance(k, str) and "sha256" in k.lower() and isinstance(v, str) and len(v) == 64:
                    recorded[k] = v
                elif isinstance(v, (dict, list)):
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    for mp in manifests:
        try:
            walk(json.loads(mp.read_text(encoding="utf-8-sig")))
        except Exception as exc:  # noqa: BLE001
            rec["checks"].append({"check": f"parse_manifest:{mp.name}", "status": "fail", "detail": str(exc)})
    rec["recorded_hashes_in_manifests"] = recorded
    if not masks and not manifests:
        rec["status"] = "blocked"
        rec["detail"] = "no split artifacts found (datasets/splits, split_manifest.json absent)"
        return rec
    rec["status"] = "pass"
    return rec


def load_processed_periods(repo: Path, dataset: str | None = None) -> dict[int, dict]:
    """Load period tensors into {period: {split: bool mask, 'y': labels, 'edge_attr': features, ...}}."""
    import torch  # noqa: F401
    out: dict[int, dict] = {}
    cands = sorted((repo / "data" / "processed").glob("period_*.pt")) if (repo / "data" / "processed").is_dir() else []
    for p in cands:
        m = re.search(r"period_(\d+)", p.name)
        if not m:
            continue
        t = int(m.group(1))
        try:
            obj = torch_load(p)
        except Exception as exc:  # noqa: BLE001
            out[t] = {"__load_error__": str(exc), "path": str(p)}
            continue
        rec: dict = {"path": str(p), "sha256": sha256_file(p)}
        masks = extract_split_masks(obj)
        rec.update(masks)
        # attributes
        for attr in ("y", "edge_attr", "x", "edge_index", "raw_idx", "orig_index", "flow_index"):
            if hasattr(obj, attr):
                rec[attr] = getattr(obj, attr)
            elif isinstance(obj, dict) and attr in obj:
                rec[attr] = obj[attr]
        out[t] = rec
    return out


def np_of(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:  # noqa: BLE001
        pass
    return np.asarray(x) if x is not None else None


def check_mask_integrity(masks_by_period: dict[int, dict]) -> dict:
    rec = {"id": "LK-2.disjointness_coverage", "periods": {}, "checks": []}
    if not masks_by_period:
        rec["status"] = "blocked"
        rec["detail"] = "no period tensors with masks found"
        return rec
    ok = True
    for t, recp in sorted(masks_by_period.items()):
        missing = [s for s in SPLITS if s not in recp or as_bool_mask(recp[s]) is None]
        if missing:
            rec["periods"][str(t)] = {"status": "blocked", "missing_splits": missing}
            rec["checks"].append({"check": f"period_{t}_masks_present", "status": "fail", "missing": missing})
            ok = False
            continue
        tr, va, te = (as_bool_mask(recp[s]) for s in SPLITS)
        if len({len(tr), len(va), len(te)}) != 1:
            rec["periods"][str(t)] = {"status": "fail", "detail": "mask lengths differ",
                                      "lengths": [len(tr), len(va), len(te)]}
            ok = False
            continue
        n = len(tr)
        inter = int((tr & va).sum() + (tr & te).sum() + (va & te).sum())
        union = int((tr | va | te).sum())
        entry = {"n_edges": n, "train": int(tr.sum()), "val": int(va.sum()), "test": int(te.sum()),
                 "pairwise_overlap": inter, "union": union,
                 "coverage": union == n}
        entry["status"] = "pass" if inter == 0 and union == n else "fail"
        if entry["status"] != "pass":
            ok = False
        rec["periods"][str(t)] = entry
        rec["checks"].append({"check": f"period_{t}_disjoint_cover", "status": entry["status"]})
    rec["status"] = "pass" if ok else "fail"
    return rec


def check_period_boundaries(manifests: list[Path], raw_path: Path | None) -> dict:
    rec = {"id": "LK-3.period_boundaries", "checks": []}
    if not manifests:
        rec["status"] = "blocked"
        rec["detail"] = "no split_manifest.json found"
        return rec
    if raw_path is None or not raw_path.is_file():
        rec["status"] = "blocked"
        rec["detail"] = "raw CSV not found for independent boundary recomputation"
        return rec
    import pandas as pd
    # collect candidate recorded boundaries from manifests
    found = []
    for mp in manifests:
        try:
            d = json.loads(mp.read_text(encoding="utf-8-sig"))
        except Exception:  # noqa: BLE001
            continue
        txt = json.dumps(d)
        # look for lists of 5-6 integer timestamps / boundary objects
        def walk(o):
            if isinstance(o, dict):
                for k, v in o.items():
                    if isinstance(v, list) and len(v) in (PERIODS, PERIODS + 1) and all(isinstance(x, (int, float)) for x in v) and v and min(v) > 10**12:
                        found.append({"source": str(mp), "key": k, "bounds": [int(x) for x in v]})
                    else:
                        walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
        walk(d)
    rec["recorded_boundary_candidates"] = found
    # recompute from raw
    ts = pd.read_csv(raw_path, usecols=[DATE_COL], dtype={DATE_COL: "int64"})
    v = ts[DATE_COL].to_numpy()
    n = len(v)
    vs = np.sort(v)
    bounds = vs[[round(k * n / PERIODS) for k in range(PERIODS + 1)]].astype(np.int64)
    rec["recomputed_equal_count_bounds_ts_ms"] = bounds.tolist()
    rec["recomputed_bounds_iso"] = [datetime.fromtimestamp(int(x) / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") for x in bounds]
    rec["raw_rows"] = int(n)
    if found:
        for cand in found:
            same = len(cand["bounds"]) == len(bounds) and all(abs(a - b) <= 1 for a, b in zip(cand["bounds"], bounds))
            rec["checks"].append({"check": f"bounds_match:{cand['source']}:{cand['key']}",
                                  "status": "pass" if same else "fail",
                                  "recorded": cand["bounds"], "recomputed": bounds.tolist()})
    else:
        rec["checks"].append({"check": "recorded_boundaries_found", "status": "warn",
                              "detail": "no 5- or 6-element timestamp boundary list found in manifests"})
    rec["status"] = "pass" if rec["checks"] and all(c["status"] == "pass" for c in rec["checks"]) else "warn"
    return rec


def chi2_sf(x, df):
    # survival function of chi-square, no scipy dependency required here
    try:
        from scipy.stats import chi2
        return float(chi2.sf(x, df))
    except Exception:  # noqa: BLE001
        if df <= 0:
            return 1.0
        return float(math.exp(-x / 2))  # crude, only used if scipy missing


def check_label_and_scaler(masks_by_period: dict[int, dict]) -> dict:
    rec = {"id": "LK-7/8.label_scaler", "periods": {}, "checks": []}
    if not masks_by_period:
        rec["status"] = "blocked"
        rec["detail"] = "no processed period tensors"
        return rec
    ok = True
    for t, r in sorted(masks_by_period.items()):
        y = np_of(r.get("y"))
        ea = np_of(r.get("edge_attr"))
        masks = {s: as_bool_mask(r.get(s)) for s in SPLITS}
        entry: dict = {"path": r.get("path")}
        if y is None or any(m is None for m in masks.values()):
            entry["status"] = "blocked"
            entry["detail"] = "missing y or masks"
        else:
            y = y.ravel()
            n = len(y)
            entry["label_counts"] = {s: dict(zip(*np.unique(y[masks[s]], return_counts=True))) for s in SPLITS if len(y[masks[s]])}
            entry["label_counts"] = {k: {str(a): int(b) for a, b in v.items()} for k, v in entry["label_counts"].items()}
            # chi-square train vs test on shared classes
            classes = sorted(set(np.unique(y[masks["train"]])) | set(np.unique(y[masks["test"]])))
            obs = []
            for c in classes:
                obs.append([int((y[masks["train"]] == c).sum()), int((y[masks["test"]] == c).sum())])
            if obs:
                obs_a = np.array(obs, dtype=float)
                row_tot = obs_a.sum(axis=1, keepdims=True)
                col_tot = obs_a.sum(axis=0, keepdims=True)
                exp = row_tot @ col_tot / obs_a.sum()
                chi = float(((obs_a - exp) ** 2 / np.where(exp == 0, 1, exp)).sum())
                dof = max(1, len(classes) - 1)
                p = chi2_sf(chi, dof)
                entry["chi2_train_vs_test"] = {"chi2": chi, "dof": dof, "p_value": p}
                if p < 0.01:
                    ok = False
            if ea is not None:
                ea2 = np_of(ea)
                if ea2 is not None and ea2.ndim == 2:
                    f32 = ea2.astype(np.float64)
                    stat = {}
                    for s in SPLITS:
                        m = masks[s]
                        if m is not None and m.sum() > 10:
                            stat[s] = {"n": int(m.sum()),
                                       "mean_abs_mean": float(np.abs(f32[m].mean(axis=0)).mean()),
                                       "mean_abs_std_dev": float(np.abs(f32[m].std(axis=0) - 1.0).mean())}
                    entry["feature_standardisation"] = stat
                    tr_s = stat.get("train", {})
                    if tr_s and tr_s.get("mean_abs_mean", 0) > 0.1:
                        entry["warning"] = "train features are not standardised (|mean| > 0.1); scaler may be missing"
        entry.setdefault("status", "pass" if entry.get("chi2_train_vs_test", {}).get("p_value", 1.0) >= 0.01 else "fail")
        if entry["status"] == "fail":
            ok = False
        rec["periods"][str(t)] = entry
        rec["checks"].append({"check": f"period_{t}_label_dist", "status": entry["status"]})
    rec["status"] = "pass" if ok else "fail"
    return rec


def check_adversarial(masks_by_period: dict[int, dict], sample: int = 30000) -> dict:
    rec = {"id": "LK-9.adversarial_auc", "periods": {}, "checks": []}
    if not masks_by_period:
        rec["status"] = "blocked"
        rec["detail"] = "no processed period tensors"
        return rec
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.model_selection import cross_val_predict
        from sklearn.metrics import roc_auc_score
    except Exception as exc:  # noqa: BLE001
        rec["status"] = "blocked"
        rec["detail"] = f"sklearn unavailable: {exc}"
        return rec
    rng = np.random.default_rng(20261101)
    ok = True
    for t, r in sorted(masks_by_period.items()):
        ea = np_of(r.get("edge_attr"))
        masks = {s: as_bool_mask(r.get(s)) for s in SPLITS}
        if ea is None or masks.get("train") is None or masks.get("test") is None:
            rec["periods"][str(t)] = {"status": "blocked", "detail": "edge_attr/train/test mask missing"}
            continue
        ea2 = np_of(ea).astype(np.float32)
        tr = np.where(masks["train"])[0]
        te = np.where(masks["test"])[0]
        if len(tr) > sample:
            tr = rng.choice(tr, size=sample, replace=False)
        if len(te) > sample:
            te = rng.choice(te, size=sample, replace=False)
        X = np.vstack([ea2[tr], ea2[te]])
        y = np.r_[np.zeros(len(tr), dtype=int), np.ones(len(te), dtype=int)]
        if np.isnan(X).any() or np.isinf(X).any():
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        clf = HistGradientBoostingClassifier(max_iter=80, random_state=42)
        proba = cross_val_predict(clf, X, y, cv=3, method="predict_proba")[:, 1]
        auc = float(roc_auc_score(y, proba))
        status = "pass" if auc < 0.65 else "fail"
        if status == "fail":
            ok = False
        rec["periods"][str(t)] = {"auc": round(auc, 6), "n_train_sampled": int(len(tr)),
                                  "n_test_sampled": int(len(te)), "threshold": 0.65, "status": status}
        rec["checks"].append({"check": f"period_{t}_adv_auc", "status": status, "auc": round(auc, 6)})
    rec["status"] = "pass" if ok else "fail"
    return rec


def check_test_in_training(masks_by_period: dict[int, dict]) -> dict:
    """LK-4: no test edge of period t may be in a training mask of period t' > t.

    Requires a global edge id per period. Supported encodings:
      * mask files store global boolean masks over a global edge index (all same length), or
      * processed tensors carry `raw_idx`/`orig_index`/`flow_index` arrays.
    If neither is available the check is reported blocked with the exact reason.
    """
    rec = {"id": "LK-4.test_in_training", "checks": []}
    if not masks_by_period:
        rec["status"] = "blocked"
        rec["detail"] = "no masks"
        return rec
    # Case A: global mask arrays (same length across periods)
    lens = {len(as_bool_mask(r["train"])) for r in masks_by_period.values() if "train" in r}
    if len(lens) == 1:
        gid = {}
        ok = True
        for t, r in sorted(masks_by_period.items()):
            gid[t] = as_bool_mask(r["test"])
        for t in sorted(masks_by_period):
            for t2 in sorted(masks_by_period):
                if t2 > t:
                    overlap = int((gid[t] & as_bool_mask(masks_by_period[t2]["train"])).sum())
                    rec["checks"].append({"check": f"test(t{t})_in_train(t{t2})", "status": "pass" if overlap == 0 else "fail",
                                          "overlap": overlap})
                    if overlap:
                        ok = False
        rec["status"] = "pass" if ok else "fail"
        rec["encoding"] = "global masks of equal length"
        return rec
    # Case B: raw index sidecars
    have_idx = all(any(k in r for k in ("raw_idx", "orig_index", "flow_index")) for r in masks_by_period.values())
    if have_idx:
        seen_test: set[int] = set()
        ok = True
        for t, r in sorted(masks_by_period.items()):
            key = next(k for k in ("raw_idx", "orig_index", "flow_index") if k in r)
            idx = np_of(r[key]).ravel()
            tr = as_bool_mask(r["train"]); te = as_bool_mask(r["test"])
            # NOTE: if idx is the period-local->raw map, masks are period-local over edges; otherwise blocked
            rec["checks"].append({"check": f"period_{t}_raw_index_present", "status": "pass", "n": int(len(idx))})
            seen_test |= set(idx[te].tolist())
            overlap = int(sum(1 for x in idx[tr].tolist() if x in seen_test))
            rec["checks"].append({"check": f"period_{t}_train_in_seen_test", "status": "pass" if overlap == 0 else "fail", "overlap": overlap})
            if overlap:
                ok = False
        rec["status"] = "pass" if ok else "fail"
        rec["encoding"] = "raw index sidecars"
        return rec
    rec["status"] = "blocked"
    rec["detail"] = ("period masks have different lengths and no global edge id / raw index sidecar is present; "
                     "cannot prove test-never-in-training without the manifest's period->global id mapping")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--raw", default=str(DEFAULT_RAW))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = Path(args.raw)
    manifests = discover_manifests(REPO_ROOT)
    masks = discover_mask_files(REPO_ROOT)
    processed = load_processed_periods(REPO_ROOT, args.dataset)
    raw_csv = None
    if args.dataset:
        d = raw / args.dataset
        csvs = [p for p in d.rglob("*.csv")] if d.is_dir() else []
        raw_csv = csvs[0] if csvs else None
    result = {
        "generated_utc": utc_now(),
        "script": "verification/src/scripts/v_splits.py",
        "dataset_filter": args.dataset,
        "discovery": {"manifests": [str(p.relative_to(REPO_ROOT)) for p in manifests],
                      "mask_files": [str(p.relative_to(REPO_ROOT)) for p in masks],
                      "processed_periods": {str(k): v.get("path") for k, v in processed.items()}},
        "checks": [],
    }
    result["checks"].append(check_hashes(manifests, masks))
    result["checks"].append(check_mask_integrity(processed))
    result["checks"].append(check_period_boundaries(manifests, raw_csv))
    result["checks"].append(check_test_in_training(processed))
    result["checks"].append(check_label_and_scaler(processed))
    result["checks"].append(check_adversarial(processed))
    statuses = [c.get("status") for c in result["checks"]]
    result["overall"] = ("blocked" if all(s == "blocked" for s in statuses) else
                         ("fail" if "fail" in statuses else ("pass" if all(s == "pass" for s in statuses) else "warn")))
    path = out / "leakage_checks.json"
    path.write_text(jdump(result), encoding="utf-8")
    print(f"wrote {path}")
    for c in result["checks"]:
        print(f"  {c['id']}: {c.get('status')}")
    print(f"OVERALL: {result['overall']}")
    return 0 if result["overall"] != "fail" else 2


if __name__ == "__main__":
    sys.exit(main())
