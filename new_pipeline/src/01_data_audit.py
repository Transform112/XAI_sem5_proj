# %% [markdown]
# # 01 - Data audit (v2, reviewed)
#
# Reads the full raw CSVs once, in chunks, and prints what it checked:
#
# - file size, header (BOM-safe), raw physical line count vs parsed rows
#   (detects rows silently dropped by the CSV parser);
# - exact duplicate rows, feature-duplicate rows over the 48 source features,
#   and label-conflict groups (all computed on a dtype-stable parsed frame, so
#   a column that is int in one chunk and float in another cannot hide
#   duplicates);
# - class counts (`Attack`), binary counts (`Label`) and their consistency;
# - per-column empty / non-numeric / infinity / negative / zero counts,
#   unique count (capped), min / max / finite mean;
# - exact host cardinality (src, dst, union) - needed for graph sizing;
# - timestamp range, backwards steps, largest backwards jump, per-class span;
# - evidence for the two missing-indicator columns.
#
# The audit never cleans, splits or modifies the raw CSVs.

# %%
from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

print("Notebook: 01 - data audit (v2)")
print("pandas:", pd.__version__)
print("numpy :", np.__version__)

# %%
def find_project_root(start: Path | None = None) -> Path:
    env = os.environ.get("XDRIFT_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    here = start or Path.cwd()
    for candidate in [here, *here.parents]:
        if (candidate / "datasets").is_dir() and (candidate / "new_pipeline").is_dir():
            return candidate
    return here.resolve()


PROJECT_ROOT = find_project_root()
ON_KAGGLE = Path("/kaggle/input").exists()
OUTPUT_ROOT = (
    Path("/kaggle/working/xdrift_pipeline")
    if ON_KAGGLE
    else PROJECT_ROOT / "new_pipeline" / "results"
)
AUDIT_ROOT = OUTPUT_ROOT / "data_audit"
AUDIT_ROOT.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "NF-UNSW-NB15-v3": "NF-UNSW-NB15-v3.csv",
    "NF-CSE-CIC-IDS2018-v3": "NF-CICIDS2018-v3.csv",
}
LABEL_COLUMNS = ["Label", "Attack"]
DROP_COLUMNS = [
    "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS",
    "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", *LABEL_COLUMNS,
]
EXPECTED_SOURCE_FEATURES = 48
MISSING_INDICATORS = [
    "SRC_TO_DST_SECOND_BYTES_missing",
    "DST_TO_SRC_SECOND_BYTES_missing",
]
INDICATOR_THRESHOLD_PCT = 0.1   # plan D7.7: indicator only if rate > 0.1%
TEXT_COLUMNS = {"IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Attack"}
UNIQUE_CAP = 20000

MAX_ROWS = int(os.environ.get("XDRIFT_AUDIT_MAX_ROWS", "0")) or None
CHUNK_ROWS = int(os.environ.get("XDRIFT_AUDIT_CHUNK_ROWS", "250000"))
DEDUPE = os.environ.get("XDRIFT_AUDIT_DEDUPE", "1") != "0"
FEATURE_DUPES = os.environ.get("XDRIFT_AUDIT_FEATURE_DUPES", "1") != "0"
_selected = os.environ.get("XDRIFT_AUDIT_DATASETS")
SELECTED = {s.strip() for s in _selected.split(",")} if _selected else set(DATASETS)
unknown = SELECTED - set(DATASETS)
if unknown:
    raise ValueError(f"unknown dataset(s) in XDRIFT_AUDIT_DATASETS: {sorted(unknown)}; "
                     f"valid: {sorted(DATASETS)}")


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                             capture_output=True, text=True, check=False)
        return out.stdout.strip() or None
    except Exception:
        return None


print("project root:", PROJECT_ROOT)
print("kaggle      :", ON_KAGGLE)
print("audit root  :", AUDIT_ROOT)
print("max rows    :", MAX_ROWS if MAX_ROWS is not None else "all")
print("chunk rows  :", CHUNK_ROWS)
print("datasets    :", sorted(SELECTED))
if MAX_ROWS is not None:
    print("[NOTE] MAX_ROWS reads the FIRST rows in file order; class/period counts "
          "from a smoke run are not representative.")

# %%
def find_dataset_path(name: str) -> Path | None:
    file_name = DATASETS[name]
    roots = [Path("/kaggle/input/nf-v3-combined-unsw-cicids")]
    if Path("/kaggle/input").is_dir():
        roots.extend(Path("/kaggle/input").glob("*"))
    for root in roots:
        candidate = root / file_name
        if candidate.is_file():
            return candidate
    candidate = PROJECT_ROOT / "datasets" / "raw" / name / file_name
    return candidate if candidate.is_file() else None


DATASET_FILES = {name: find_dataset_path(name) for name in DATASETS}
for name, path in DATASET_FILES.items():
    print(f"{name:26s} -> {path}")

# %%
def count_data_lines(path: Path) -> int:
    """Non-blank physical data lines (header excluded), counted from raw bytes.

    Blank lines are skipped by the CSV parser by design, so they must not be
    reported as lost rows.
    """
    lines = 0
    with path.open("rb") as handle:
        for line in handle:
            if line.strip():
                lines += 1
    return max(lines - 1, 0)


def new_column_state() -> dict:
    return {"empty": 0, "nonnumeric": 0, "inf": 0, "negative": 0, "zero": 0,
            "numeric": 0, "finite": 0, "unique": set(), "unique_capped": False,
            "min": None, "max": None, "sum": 0.0}


def parse_numeric(raw: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Return (parsed float64 series, empty mask). Empty cells -> NaN."""
    stripped = raw.str.strip()
    empty = stripped == ""
    parsed = pd.to_numeric(stripped.where(~empty), errors="coerce").astype("float64")
    return parsed, empty


def update_text(state: dict, raw: pd.Series) -> None:
    stripped = raw.str.strip()
    empty = stripped == ""
    state["empty"] += int(empty.sum())
    if not state["unique_capped"]:
        state["unique"].update(stripped[~empty].unique().tolist())
        if len(state["unique"]) > UNIQUE_CAP:
            state["unique"] = set(sorted(state["unique"])[:UNIQUE_CAP])
            state["unique_capped"] = True


def update_numeric(state: dict, parsed: pd.Series, empty: pd.Series) -> None:
    state["empty"] += int(empty.sum())
    nonnumeric = ~empty & parsed.isna()
    state["nonnumeric"] += int(nonnumeric.sum())
    valid = parsed[~empty & ~nonnumeric].to_numpy()
    if valid.size == 0:
        return
    state["numeric"] += int(valid.size)
    finite = valid[np.isfinite(valid)]
    state["inf"] += int(valid.size - finite.size)
    if finite.size == 0:
        return
    state["finite"] += int(finite.size)
    state["negative"] += int((finite < 0).sum())
    state["zero"] += int((finite == 0).sum())
    lo, hi = float(finite.min()), float(finite.max())
    state["min"] = lo if state["min"] is None else min(state["min"], lo)
    state["max"] = hi if state["max"] is None else max(state["max"], hi)
    state["sum"] += float(finite.sum())
    if not state["unique_capped"]:
        state["unique"].update(np.unique(finite).tolist())
        if len(state["unique"]) > UNIQUE_CAP:
            state["unique"] = set(sorted(state["unique"])[:UNIQUE_CAP])
            state["unique_capped"] = True


def ms_to_iso(value: float | None) -> str | None:
    if value is None:
        return None
    return pd.to_datetime(value, unit="ms", utc=True).isoformat()

# %%
print("cells ready; starting full audit")
results: dict[str, dict] = {}
for dataset in sorted(SELECTED):
    path = DATASET_FILES[dataset]
    print("\n" + "#" * 78)
    print("# DATASET:", dataset)
    print("#" * 78)
    if path is None:
        print("[FAIL] raw CSV not found")
        results[dataset] = {"status": "missing", "path": None}
        continue

    print("file:", path)
    print("size:", f"{path.stat().st_size:,} bytes")
    with path.open("r", encoding="utf-8-sig") as handle:      # BOM-safe
        header = handle.readline().rstrip("\r\n").split(",")
    print("columns:", len(header))
    feature_columns = [c for c in header if c not in DROP_COLUMNS]
    print("feature columns:", len(feature_columns))
    contract_ok = len(feature_columns) == EXPECTED_SOURCE_FEATURES
    if not contract_ok:
        print(f"[WARN] expected {EXPECTED_SOURCE_FEATURES} source features; got {len(feature_columns)}")

    raw_lines = None if MAX_ROWS is not None else count_data_lines(path)
    print("physical data lines:", f"{raw_lines:,}" if raw_lines is not None else "skipped (MAX_ROWS)")

    per_column = {name: new_column_state() for name in header}
    class_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    label_by_attack: dict[str, Counter[str]] = defaultdict(Counter)
    attack_code: dict[str, int] = {}
    class_span: dict[str, dict] = {}
    exact_hashes: list[np.ndarray] = []
    feature_hashes: list[np.ndarray] = []
    featlabel_hashes: list[np.ndarray] = []
    attack_codes: list[np.ndarray] = []
    src_hosts: list[np.ndarray] = []
    dst_hosts: list[np.ndarray] = []
    chunks = rows = 0
    ts_min = ts_max = ts_prev = None
    ts_backwards = 0
    max_backward_ms = 0.0

    # dtype=str + na_filter=False: every chunk is parsed by OUR code into
    # float64, so a column that looks int in one chunk and float in another
    # cannot change its hash. Empty cells stay "" and are counted exactly.
    reader = pd.read_csv(path, chunksize=CHUNK_ROWS, dtype=str, na_filter=False,
                         on_bad_lines="warn", encoding="utf-8-sig")
    for raw in reader:
        if MAX_ROWS is not None:
            remaining = MAX_ROWS - rows
            if remaining <= 0:
                break
            if len(raw) > remaining:
                raw = raw.iloc[:remaining].copy()
        chunks += 1
        rows += len(raw)
        if chunks == 1:
            print("first chunk columns:", len(raw.columns))

        parsed = pd.DataFrame(index=raw.index)
        for column in header:
            if column not in raw.columns:
                continue
            if column in TEXT_COLUMNS:
                update_text(per_column[column], raw[column])
                parsed[column] = raw[column].str.strip()
            else:
                values, empty = parse_numeric(raw[column])
                update_numeric(per_column[column], values, empty)
                parsed[column] = values

        if DEDUPE:
            exact_hashes.append(pd.util.hash_pandas_object(parsed, index=False).to_numpy("uint64"))
        if FEATURE_DUPES and set(feature_columns).issubset(parsed.columns) and "Attack" in parsed.columns:
            feature_hashes.append(pd.util.hash_pandas_object(parsed[feature_columns], index=False).to_numpy("uint64"))
            featlabel_hashes.append(pd.util.hash_pandas_object(parsed[[*feature_columns, *LABEL_COLUMNS]], index=False).to_numpy("uint64"))
            attacks = raw["Attack"].str.strip()
            for name in attacks.unique():
                attack_code.setdefault(name, len(attack_code))
            attack_codes.append(attacks.map(attack_code).to_numpy("int32"))

        if "Attack" in raw.columns:
            attacks = raw["Attack"].str.strip()
            class_counts.update(attacks.value_counts().to_dict())
            if "Label" in raw.columns:
                pairs = pd.DataFrame({"a": attacks, "l": raw["Label"].str.strip()})
                for (a, l), n in pairs.groupby(["a", "l"]).size().items():
                    label_by_attack[a][l] += int(n)
        if "Label" in raw.columns:
            label_counts.update(raw["Label"].str.strip().value_counts().to_dict())

        for col, store in (("IPV4_SRC_ADDR", src_hosts), ("IPV4_DST_ADDR", dst_hosts)):
            if col in parsed.columns:
                uniq = parsed[col].unique()
                store.append(np.unique(pd.util.hash_array(np.asarray(uniq, dtype=object))))

        if "FLOW_START_MILLISECONDS" in parsed.columns:
            ts = parsed["FLOW_START_MILLISECONDS"]
            valid = ts.notna().to_numpy()
            stamps = ts.to_numpy()[valid]
            if stamps.size:
                if ts_prev is not None and stamps[0] < ts_prev:
                    ts_backwards += 1
                    max_backward_ms = max(max_backward_ms, float(ts_prev - stamps[0]))
                diffs = np.diff(stamps)
                neg = diffs < 0
                if neg.any():
                    ts_backwards += int(neg.sum())
                    max_backward_ms = max(max_backward_ms, float(-diffs.min()))
                ts_prev = float(stamps[-1])
                # FIX: use the chunk's true min/max, not its first/last row.
                lo, hi = float(stamps.min()), float(stamps.max())
                ts_min = lo if ts_min is None else min(ts_min, lo)
                ts_max = hi if ts_max is None else max(ts_max, hi)
                if "Attack" in raw.columns:
                    span = pd.DataFrame({"a": raw["Attack"].str.strip().to_numpy()[valid], "t": stamps}
                                        ).groupby("a")["t"].agg(["min", "max", "count"])
                    for a, r in span.iterrows():
                        s = class_span.setdefault(a, {"min": float(r["min"]), "max": float(r["max"]), "count": 0})
                        s["min"] = min(s["min"], float(r["min"]))
                        s["max"] = max(s["max"], float(r["max"]))
                        s["count"] += int(r["count"])

        if MAX_ROWS is not None and rows >= MAX_ROWS:
            print("stopping at MAX_ROWS =", MAX_ROWS)
            break
        if chunks % 10 == 0:
            print(f"  chunk {chunks:4d}, rows {rows:12,d}")

    # ---- finalize ---------------------------------------------------------
    rows_lost = None if raw_lines is None else raw_lines - rows
    if rows_lost:
        print(f"[WARN] parser dropped {rows_lost:,} line(s): physical lines {raw_lines:,} vs parsed rows {rows:,}")

    def n_dup(parts: list[np.ndarray]) -> int:
        if not parts:
            return 0
        allh = np.concatenate(parts)
        return int(allh.size - np.unique(allh).size)

    duplicate_rows = n_dup(exact_hashes)
    feature_dup_rows = n_dup(feature_hashes)
    feature_label_dup_rows = n_dup(featlabel_hashes)

    label_conflict_groups = label_conflict_rows = 0
    label_conflict_examples: list[dict] = []
    if feature_hashes:
        fh = np.concatenate(feature_hashes)
        codes = np.concatenate(attack_codes)
        nlab = pd.DataFrame({"h": fh, "c": codes}).groupby("h")["c"].nunique()
        conflict_h = nlab.index[nlab.to_numpy() > 1].to_numpy(dtype="uint64")
        label_conflict_groups = int(conflict_h.size)
        label_conflict_rows = int(np.isin(fh, conflict_h).sum())      # no second CSV pass
        names = {v: k for k, v in attack_code.items()}
        if conflict_h.size:
            sample = pd.DataFrame({"h": fh, "c": codes})
            sample = sample[sample["h"].isin(conflict_h[:10])].groupby("h")["c"].unique()
            label_conflict_examples = [{"labels": sorted(names[int(c)] for c in cs)} for cs in sample]
    del exact_hashes, feature_hashes, featlabel_hashes, attack_codes

    def n_unique(parts: list[np.ndarray]) -> int:
        return int(np.unique(np.concatenate(parts)).size) if parts else 0

    host_src, host_dst = n_unique(src_hosts), n_unique(dst_hosts)
    host_union = n_unique([*src_hosts, *dst_hosts])

    inconsistent_attacks = {a: dict(d) for a, d in label_by_attack.items() if len(d) > 1}
    benign_ok = all(set(d) <= {"0"} for a, d in label_by_attack.items() if a.lower() == "benign") and \
                all(set(d) <= {"1"} for a, d in label_by_attack.items() if a.lower() != "benign")

    column_rows = []
    for column in header:
        s = per_column[column]
        column_rows.append({
            "column": column, "empty": s["empty"],
            "empty_pct": round(100.0 * s["empty"] / rows, 6) if rows else None,
            "nonnumeric": s["nonnumeric"], "inf": s["inf"],
            "negative": s["negative"],
            "negative_pct": round(100.0 * s["negative"] / rows, 6) if rows else None,
            "zero": s["zero"], "numeric": s["numeric"],
            "unique": len(s["unique"]), "unique_capped": s["unique_capped"],
            "min": s["min"], "max": s["max"],
            "mean": (s["sum"] / s["finite"]) if s["finite"] else None,   # FIX: finite denominator
        })
    by_col = {r["column"]: r for r in column_rows}
    indicator_evidence = {}
    for ind in MISSING_INDICATORS:
        base = ind.removesuffix("_missing")
        r = by_col.get(base)
        if r:
            sentinel_pct = (r["empty_pct"] or 0) + (r["negative_pct"] or 0)
            indicator_evidence[ind] = {
                "base_column": base, "empty_pct": r["empty_pct"],
                "negative_pct": r["negative_pct"],
                "justified_by_threshold": sentinel_pct > INDICATOR_THRESHOLD_PCT,
            }

    result = {
        "status": "ok", "dataset": dataset, "path": str(path),
        "audit_version": "2.0", "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(), "parse_policy": "dtype=str, parsed to float64 per chunk",
        "size_bytes": path.stat().st_size, "header": header,
        "rows": rows, "chunks": chunks,
        "physical_data_lines": raw_lines, "rows_lost_by_parser": rows_lost,
        "duplicate_rows": duplicate_rows,
        "duplicate_pct": round(100.0 * duplicate_rows / rows, 6) if rows else None,
        "rows_after_exact_dedup": rows - duplicate_rows,
        "feature_columns": feature_columns, "feature_columns_count": len(feature_columns),
        "feature_contract_ok": contract_ok,
        "missing_indicators": MISSING_INDICATORS, "missing_indicator_evidence": indicator_evidence,
        "feature_duplicate_rows": feature_dup_rows,
        "feature_duplicate_pct": round(100.0 * feature_dup_rows / rows, 6) if rows else None,
        "feature_duplicate_with_label_rows": feature_label_dup_rows,
        "feature_duplicate_with_label_pct": round(100.0 * feature_label_dup_rows / rows, 6) if rows else None,
        "label_conflict_groups": label_conflict_groups,
        "label_conflict_rows": label_conflict_rows,
        "label_conflict_examples": label_conflict_examples,
        "feature_duplicate_note": ("Preliminary raw-value hash over the 48 source features. "
                                   "Stage 02 recomputes with quantised values; groups are used "
                                   "for split grouping, not row removal."),
        "hosts": {"src_unique": host_src, "dst_unique": host_dst, "union_unique": host_union},
        "class_counts": dict(class_counts.most_common()),
        "label_counts": dict(label_counts),
        "label_by_attack": {k: dict(v) for k, v in label_by_attack.items()},
        "label_attack_consistent": (not inconsistent_attacks) and benign_ok,
        "label_attack_inconsistent": inconsistent_attacks,
        "class_time_span_ms": class_span,
        "timestamp_min_ms": ts_min, "timestamp_max_ms": ts_max,
        "timestamp_min_utc": ms_to_iso(ts_min), "timestamp_max_utc": ms_to_iso(ts_max),
        "timestamp_backwards_steps": ts_backwards,
        "timestamp_max_backward_jump_ms": max_backward_ms,
        "columns": column_rows, "max_rows_limit": MAX_ROWS,
    }
    results[dataset] = result

    print("\nSUMMARY:", dataset)
    print(f"  rows                : {rows:,}  (physical lines {raw_lines if raw_lines is None else format(raw_lines, ',')})")
    print(f"  exact duplicate rows: {duplicate_rows:,} ({result['duplicate_pct']}%)  -> {rows - duplicate_rows:,} after dedup")
    print(f"  feature duplicates  : {feature_dup_rows:,} ({result['feature_duplicate_pct']}%)")
    print(f"  feature+label dups  : {feature_label_dup_rows:,}")
    print(f"  label-conflict      : {label_conflict_groups:,} groups, {label_conflict_rows:,} rows")
    print(f"  hosts (src/dst/union): {host_src:,} / {host_dst:,} / {host_union:,}")
    print(f"  time range          : {result['timestamp_min_utc']} .. {result['timestamp_max_utc']}")
    print(f"  backwards steps     : {ts_backwards:,} (largest jump {max_backward_ms/1000:.1f} s)")
    print(f"  Label/Attack consistent: {result['label_attack_consistent']}")
    for label, count in class_counts.most_common():
        print(f"    {label:28s} {count:12,d}")
    for ind, ev in indicator_evidence.items():
        print(f"  indicator {ind}: empty {ev['empty_pct']}%  negative {ev['negative_pct']}%  justified={ev['justified_by_threshold']}")

# %%
for dataset, result in results.items():
    if result.get("status") != "ok":
        continue
    print("\n" + "=" * 110)
    print("PER-COLUMN AUDIT:", dataset)
    print("=" * 110)
    print(f"{'column':30s} {'empty':>10s} {'nonnum':>9s} {'inf':>7s} {'neg':>9s} "
          f"{'zero':>10s} {'unique':>8s} {'min':>14s} {'max':>14s}")

    def fmt(value):
        if value is None:
            return "-"
        return f"{value:.6g}" if isinstance(value, float) else f"{value:,}"

    for row in result["columns"]:
        print(f"{row['column']:30s} {row['empty']:>10,d} {row['nonnumeric']:>9,d} "
              f"{row['inf']:>7,d} {row['negative']:>9,d} {row['zero']:>10,d} "
              f"{row['unique']:>8,d} {fmt(row['min']):>14s} {fmt(row['max']):>14s}")

# %%
for dataset, result in results.items():
    if result.get("status") != "ok":
        continue
    safe = dataset.replace(" ", "_")
    json_path = AUDIT_ROOT / f"{safe}.json"
    csv_path = AUDIT_ROOT / f"{safe}_columns.csv"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(result["columns"]).to_csv(csv_path, index=False)
    print("saved:", json_path)
    print("saved:", csv_path)

# %%
print("=" * 78)
print("DATA AUDIT COMPLETE")
print("=" * 78)
failed = False
for dataset, result in results.items():
    if result.get("status") != "ok":
        print(f"  [FAIL] {dataset}: {result.get('status')}")
        failed = True
        continue
    flags = []
    if result["rows_lost_by_parser"]:
        flags.append(f"parser dropped {result['rows_lost_by_parser']} lines")
    if not result["feature_contract_ok"]:
        flags.append("feature contract mismatch")
    if not result["label_attack_consistent"]:
        flags.append("Label/Attack inconsistent")
    tag = "WARN" if flags else " OK "
    print(f"  [{tag}] {dataset}: rows={result['rows']:,}, duplicates={result['duplicate_rows']:,}, "
          f"classes={len(result['class_counts'])}, columns={len(result['columns'])} {'; '.join(flags)}")
print("output directory:", AUDIT_ROOT)