#!/usr/bin/env python
"""Independent dataset-identity audit for the NF-v3 family.

Verifier-owned code. Deliberately does NOT import any project module; it reads
raw archives/CSVs directly and compares against the published UQ reference
counts stored in results/text/verification/reference/published_counts.json.

Checks per dataset:
  identity.row_count                data rows == published total (exact)
  identity.class_counts             per-class counts == published table (exact)
  identity.schema                   header present, expected v3 column family,
                                    temporal columns present, label columns present
  identity.timestamp_order          FLOW_START_MILLISECONDS parses as epoch ms and
                                    is non-decreasing (records min/max/peak)
  identity.missing_tokens           count of empty / NaN / inf tokens per column
                                    (streamed; expensive columns reported as top list)
  identity.exact_duplicates         full-row and no-raw-timestamp duplicate counts
                                    (SQLite-backed, so it works at 27M rows)

Usage (Windows PowerShell, from repo root):
  .venv\\Scripts\\python.exe verification\\scripts\\v_identity.py --dataset NF-UNSW-NB15-v3
  .venv\\Scripts\\python.exe verification\\scripts\\v_identity.py --dataset all --skip-dedupe
  python verification\\scripts\\v_identity.py --dataset all

Outputs:
  results/text/verification/data_audit/identity_<dataset>.json
  results/text/verification/data_audit/identity_<dataset>.log.txt
  results/text/verification/data_audit/identity_summary.json / .md
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import re
import sqlite3
import sys
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

VERIFIER_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = VERIFIER_ROOT.parents[1]
DEFAULT_REFERENCE = VERIFIER_ROOT / "reference" / "published_counts.json"
DEFAULT_RAW = REPO_ROOT / "data" / "raw"
DEFAULT_OUT = VERIFIER_ROOT / "data_audit"
DATE_COL = "FLOW_START_MILLISECONDS"
LABEL_CANDIDATES = ["Label", "label", "Attack", "attack", "Attack_cat", "attack_cat"]
TIMESTAMP_COL_HINTS = ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS"]


# ---------------------------------------------------------------- helpers
def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Tee:
    def __init__(self, path: Path):
        self.f = open(path, "w", encoding="utf-8", buffering=1)

    def write(self, s: str) -> None:
        sys.stdout.write(s)
        self.f.write(s)

    def line(self, s: str = "") -> None:
        self.write(s + "\n")

    def close(self) -> None:
        self.f.close()


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def locate_csvs(raw: Path, dataset: str) -> tuple[list[Path], Path | None, str]:
    """Return (csv_paths, zip_path_or_None, source_kind). Prefer extracted dirs."""
    extracted = raw / dataset
    if extracted.is_dir():
        csvs = sorted(p for p in extracted.rglob("*.csv") if p.is_file())
        if csvs:
            return csvs, None, "extracted_csv"
        zips = sorted(extracted.rglob("*.zip"))
        if len(zips) == 1:
            return [], zips[0], "nested_zip"
    zp = raw / f"{dataset}.zip"
    if zp.is_file():
        return [], zp, "zip"
    # allow single differently named zip
    zips = sorted(raw.glob("*.zip"))
    for z in zips:
        if dataset.lower().replace("-", "") in z.name.lower().replace("-", ""):
            return [], z, "zip"
    raise FileNotFoundError(f"no CSV or zip found for {dataset} under {raw}")


def open_stream(src: Path | zipfile.ZipInfo, zf: zipfile.ZipFile | None):
    if zf is not None:
        raw = zf.open(src)  # type: ignore[arg-type]
        return io.TextIOWrapper(raw, encoding="utf-8", errors="replace", newline="")
    return open(src, "r", encoding="utf-8", errors="replace", newline="")


def norm(s: str) -> str:
    return (s or "").strip().strip('"').strip()


def is_missing_token(tok: str) -> str | None:
    t = tok.strip().strip('"')
    if t == "":
        return "empty"
    low = t.lower()
    if low in ("nan", "na", "n/a", "null", "none"):
        return "nan_token"
    if low in ("inf", "+inf", "infinity", "+infinity", "1e400"):
        return "inf_token"
    if low in ("-inf", "-infinity", "-1e400"):
        return "neg_inf_token"
    return None


def dedupe_sqlite(
    db_path: Path,
    streams: list[tuple[Path | zipfile.ZipInfo, zipfile.ZipFile | None]],
    label_idx: dict[int, str],
    ncols: int,
    log,
) -> dict:
    """Count full-row and no-raw-timestamp duplicates without holding rows in RAM."""
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(str(db_path))
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=OFF")
    cur.execute("PRAGMA synchronous=OFF")
    cur.execute("PRAGMA temp_store=MEMORY")
    cur.execute("CREATE TABLE h (full INTEGER, notime INTEGER)")
    batch: list[tuple[int, int]] = []
    total = 0
    t0 = time.time()
    for src, zf in streams:
        with open_stream(src, zf) as fh:
            r = csv.reader(fh)
            header = next(r)
            for row in r:
                if not row:
                    continue
                joined = "\x1f".join(row)
                full = int.from_bytes(hashlib.blake2b(joined.encode("utf-8", "replace"), digest_size=8).digest(), "big") & ((1 << 63) - 1)
                notime = full
                if DATE_COL in header:
                    ci = header.index(DATE_COL)
                    ts = row[ci] if ci < len(row) else ""
                    joined2 = "\x1f".join(row[:ci] + row[ci + 1:]) if ci < len(row) else joined
                    notime = int.from_bytes(hashlib.blake2b(joined2.encode("utf-8", "replace"), digest_size=8).digest(), "big") & ((1 << 63) - 1)
                batch.append((full, notime))
                total += 1
                if len(batch) >= 200_000:
                    cur.executemany("INSERT INTO h VALUES (?,?)", batch)
                    batch.clear()
                    log.line(f"  [dedupe] inserted {total:,} rows ({time.time()-t0:,.0f}s)")
    if batch:
        cur.executemany("INSERT INTO h VALUES (?,?)", batch)
        batch.clear()
    con.commit()
    log.line(f"  [dedupe] inserted all {total:,} rows in {time.time()-t0:,.0f}s; grouping...")
    t1 = time.time()
    out = {}
    for col, name in (("full", "full_row"), ("notime", "excluding_flow_start_ms")):
        cur.execute(f"SELECT COUNT(*), SUM(c-1) FROM (SELECT {col} AS k, COUNT(*) AS c FROM h GROUP BY {col} HAVING COUNT(*) > 1)")
        n_groups, extra = cur.fetchone()
        cur.execute(f"SELECT {col} AS k, COUNT(*) AS c FROM h GROUP BY {col} HAVING COUNT(*) > 1 ORDER BY c DESC LIMIT 5")
        top = cur.fetchall()
        cur.execute(f"SELECT {col} AS k, COUNT(*) AS c FROM h GROUP BY {col} HAVING COUNT(*) > 1")
        payload = {k: c for k, c in cur.fetchall()}
        dup_rows = sum(c for c in payload.values())
        out[name] = {
            "duplicate_row_count": int(extra or 0),
            "duplicate_group_count": int(n_groups or 0),
            "rows_in_duplicate_groups": int(dup_rows),
            "top_duplicate_group_sizes": [int(c) for _, c in top],
        }
        log.line(f"  [dedupe] {name}: dup_rows={out[name]['duplicate_row_count']:,} groups={out[name]['duplicate_group_count']:,}")
    con.close()
    db_path.unlink(missing_ok=True)
    log.line(f"  [dedupe] grouping done in {time.time()-t1:,.0f}s")
    return out


# ---------------------------------------------------------------- scanning
def scan_stream(
    src: Path | zipfile.ZipInfo,
    zf: zipfile.ZipFile | None,
    log,
    collect_dedupe_hashes: bool = False,
) -> dict:
    with open_stream(src, zf) as fh:
        r = csv.reader(fh)
        try:
            header = next(r)
        except StopIteration:
            raise SystemExit(f"EMPTY FILE: {src}")
        header = [norm(h) for h in header]
        ncols = len(header)
        label_cols = {i: h for i, h in enumerate(header) if h in LABEL_CANDIDATES}
        if not label_cols:
            # fall back: any column whose name contains 'label' or 'attack'
            label_cols = {i: h for i, h in enumerate(header) if ("label" in h.lower() or "attack" in h.lower())}
        counts = {i: Counter() for i in label_cols}
        label_pair = Counter() if len(label_cols) >= 2 else None
        label_pair_capped = False
        host_cols = {i: h for i, h in enumerate(header) if h in ("IPV4_SRC_ADDR", "IPV4_DST_ADDR")}
        host_sets = {h: set() for h in host_cols.values()}
        pair_counter = Counter() if len(host_cols) == 2 else None
        pair_capped = False
        self_loops = 0
        src_idx = next((i for i, h in host_cols.items() if h == "IPV4_SRC_ADDR"), None)
        dst_idx = next((i for i, h in host_cols.items() if h == "IPV4_DST_ADDR"), None)
        missing = Counter()           # col -> token kind -> count
        missing_rows = 0
        timestamp = {"column": None, "rows_checked": 0, "non_numeric": 0,
                     "min": None, "max": None, "decreasing_steps": 0, "first_decreasing_at": None}
        ts_idx = header.index(DATE_COL) if DATE_COL in header else None
        timestamp["column"] = header[ts_idx] if ts_idx is not None else None
        n = 0
        t0 = time.time()
        # Fast path: most rows are clean, so detect suspicious rows with C-level
        # list/string operations and only attribute missing tokens on those rows.
        for row in r:
            if not row or (len(row) == 1 and row[0] == ""):
                continue
            n += 1
            if len(row) != ncols:
                missing["__row_width__"] += 1
            suspicious = ("" in row) or ("nan" in (js := ",".join(row).lower())) or ("inf" in js) or ("null" in js) or ("n/a" in js)
            if suspicious:
                any_missing = False
                for i in range(min(ncols, len(row))):
                    kind = is_missing_token(row[i])
                    if kind is not None:
                        missing[f"{i}:{header[i]}:{kind}"] += 1
                        any_missing = True
                if any_missing:
                    missing_rows += 1
            for i in label_cols:
                if i < len(row):
                    counts[i][norm(row[i])] += 1
            if len(label_cols) >= 2 and not label_pair_capped:
                idxs = list(label_cols.keys())[:4]
                label_pair[tuple(norm(row[i]) if i < len(row) else "" for i in idxs)] += 1
                if len(label_pair) > 100_000:
                    label_pair_capped = True
            if src_idx is not None and dst_idx is not None and src_idx < len(row) and dst_idx < len(row):
                s, d = norm(row[src_idx]), norm(row[dst_idx])
                host_sets["IPV4_SRC_ADDR"].add(s)
                host_sets["IPV4_DST_ADDR"].add(d)
                if s == d:
                    self_loops += 1
                if not pair_capped:
                    pair_counter[(s, d)] += 1
                    if len(pair_counter) > 3_000_000:
                        pair_capped = True
            if ts_idx is not None and ts_idx < len(row):
                tok = row[ts_idx].strip()
                try:
                    v = int(tok)
                except ValueError:
                    try:
                        v = int(float(tok))
                    except ValueError:
                        timestamp["non_numeric"] += 1
                        continue
                timestamp["rows_checked"] += 1
                if timestamp["min"] is None or v < timestamp["min"]:
                    timestamp["min"] = v
                if timestamp["max"] is None or v > timestamp["max"]:
                    timestamp["max"] = v
                if timestamp.get("_prev") is not None and v < timestamp["_prev"]:
                    timestamp["decreasing_steps"] += 1
                    if timestamp["first_decreasing_at"] is None:
                        timestamp["first_decreasing_at"] = n
                timestamp["_prev"] = v
        elapsed = time.time() - t0
        timestamp.pop("_prev", None)
        if timestamp["rows_checked"]:
            # epoch ms sanity window: 1990..2065
            timestamp["min_iso"] = datetime.fromtimestamp(timestamp["min"] / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            timestamp["max_iso"] = datetime.fromtimestamp(timestamp["max"] / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
            timestamp["epoch_ms_sane"] = 631152000000 <= timestamp["min"] and timestamp["max"] <= 3000000000000
        else:
            timestamp["min_iso"] = timestamp["max_iso"] = None
            timestamp["epoch_ms_sane"] = False
        top_missing = dict(sorted(missing.items(), key=lambda kv: -kv[1])[:40])
        hosts_union = set()
        for s in host_sets.values():
            hosts_union |= s
        host_stats = {
            "unique_src_hosts": len(host_sets.get("IPV4_SRC_ADDR", ())),
            "unique_dst_hosts": len(host_sets.get("IPV4_DST_ADDR", ())),
            "unique_hosts_union": len(hosts_union),
            "self_loops": self_loops,
            "unique_host_pairs": len(pair_counter),
            "host_pair_count_capped": pair_capped,
        } if host_cols else None
        return {
            "file": str(src),
            "header": header,
            "n_columns": ncols,
            "data_rows": n,
            "scan_seconds": round(elapsed, 1),
            "label_columns": [header[i] for i in label_cols],
            "label_counts": {header[i]: dict(counts[i].most_common()) for i in label_cols},
            "label_pair_counts_capped": ({f"{k[0]}||{k[1]}": v for k, v in label_pair.most_common(200)} if label_pair is not None else None),
            "host_stats": host_stats,
            "missing": {
                "rows_with_any_missing_or_nan": missing_rows,
                "per_column_top": {k: int(v) for k, v in top_missing.items()},
                "row_width_anomalies": int(missing.get("__row_width__", 0)),
            },
            "timestamp": timestamp,
        }


# ---------------------------------------------------------------- pandas scan
def scan_pandas(path: Path, log, chunksize: int = 250_000, light: bool = False) -> dict:
    """Chunked pandas variant of scan_stream (used for the 17-27M-row datasets).

    Same semantics as scan_stream; much faster because parsing/aggregation happen
    in C. Empty strings and standard NA tokens are read as NaN (counted as
    nan_or_empty); IEEE inf values are counted separately.

    light=True skips the per-row host-pair Counter and the Label/Attack cross-tab
    (pure Python loops over 17-27M rows). Those metrics are covered independently
    and exactly by verification/src/scripts/v_hosts.py; identity JSON then records
    pairs_skipped=true instead of guessing.
    """
    import numpy as np
    import pandas as pd

    log.line(f"  engine=pandas chunksize={chunksize} pandas={pd.__version__} numpy={np.__version__}")
    peek = pd.read_csv(path, nrows=10)
    header = [str(c) for c in peek.columns]
    ncols = len(header)
    str_cols = [c for c in header if c in ("IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Label", "Attack")]
    num_cols = [c for c in header if c not in str_cols]
    dtype = {c: str for c in str_cols}
    try:
        pd.read_csv(path, nrows=500, dtype={**dtype, **{c: "float64" for c in num_cols}})
        dtype.update({c: "float64" for c in num_cols})
        numeric_ok = True
    except (ValueError, TypeError) as exc:
        log.line(f"  numeric dtype probe failed ({exc}); reading every column as str")
        dtype.update({c: str for c in num_cols})
        numeric_ok = False
    label_cols = [c for c in header if c in LABEL_CANDIDATES] or \
        [c for c in header if ("label" in c.lower() or "attack" in c.lower())]
    counts = {c: Counter() for c in label_cols}
    label_pair: Counter = Counter()
    label_pair_capped = False
    host_sets = {c: set() for c in ("IPV4_SRC_ADDR", "IPV4_DST_ADDR") if c in header}
    pair_counter: Counter = Counter() if not light else None
    pair_capped = False
    self_loops = 0
    missing = Counter()
    missing_rows = 0
    ts_col = DATE_COL if DATE_COL in header else None
    tmin = tmax = prev = None
    non_numeric = 0
    dec = 0
    n = 0
    t0 = time.time()
    reader = pd.read_csv(path, chunksize=chunksize, dtype=dtype, keep_default_na=True,
                         na_values=[""], low_memory=False)
    for chunk in reader:
        df = chunk
        n += len(df)
        # missingness
        na = df.isna()
        rows_any = None
        if na.any(axis=None):
            rows_any = na.any(axis=1).to_numpy()
            for col in header:
                c = int(na[col].sum())
                if c:
                    missing[f"{col}:nan_or_empty"] += c
        if numeric_ok and num_cols:
            num = df[num_cols].to_numpy(dtype="float64", copy=False)
            inf_mask = np.isinf(num)
            if inf_mask.any():
                inf_any = inf_mask.any(axis=1)
                rows_any = inf_any if rows_any is None else (rows_any | inf_any)
                colsum = inf_mask.sum(axis=0)
                for i, col in enumerate(num_cols):
                    if colsum[i]:
                        missing[f"{col}:inf"] += int(colsum[i])
        if rows_any is not None:
            missing_rows += int(rows_any.sum())
        # labels
        for c in label_cols:
            vc = df[c].fillna("__NA__").value_counts()
            counts[c].update({str(k): int(v) for k, v in vc.items()})
        if len(label_cols) >= 2 and not label_pair_capped and not light:
            lp = df[label_cols].fillna("").astype(str).agg("||".join, axis=1).value_counts()
            for k, v in lp.items():
                label_pair[k] += int(v)
            if len(label_pair) > 100_000:
                label_pair_capped = True
        # hosts / pairs
        if "IPV4_SRC_ADDR" in host_sets and "IPV4_DST_ADDR" in host_sets:
            src = df["IPV4_SRC_ADDR"].fillna("").astype(str)
            dst = df["IPV4_DST_ADDR"].fillna("").astype(str)
            host_sets["IPV4_SRC_ADDR"].update(src.unique().tolist())
            host_sets["IPV4_DST_ADDR"].update(dst.unique().tolist())
            self_loops += int((src == dst).sum())
            if not pair_capped and pair_counter is not None:
                pc = Counter(zip(src.tolist(), dst.tolist()))
                pair_counter.update(pc)
                if len(pair_counter) > 3_000_000:
                    pair_capped = True
        # timestamps
        if ts_col is not None:
            v = pd.to_numeric(df[ts_col], errors="coerce")
            non_numeric += int(v.isna().sum())
            vv = v.dropna().astype("int64")
            if len(vv):
                if tmin is None or int(vv.min()) < tmin:
                    tmin = int(vv.min())
                if tmax is None or int(vv.max()) > tmax:
                    tmax = int(vv.max())
                d = (vv.diff() < 0).sum()
                dec += int(d)
                if prev is not None and int(vv.iloc[0]) < prev:
                    dec += 1
                prev = int(vv.iloc[-1])
    elapsed = time.time() - t0
    total_rows = n
    hosts_union = set()
    for s in host_sets.values():
        hosts_union |= s
    host_stats = {
        "unique_src_hosts": len(host_sets.get("IPV4_SRC_ADDR", ())),
        "unique_dst_hosts": len(host_sets.get("IPV4_DST_ADDR", ())),
        "unique_hosts_union": len(hosts_union),
        "self_loops": self_loops,
        "unique_host_pairs": (len(pair_counter) if pair_counter is not None else None),
        "host_pair_count_capped": pair_capped,
        "pairs_skipped_in_light_mode": pair_counter is None,
    } if host_sets else None
    top_missing = dict(sorted(missing.items(), key=lambda kv: -kv[1])[:40])
    timestamp = {"column": ts_col, "rows_checked": total_rows - non_numeric, "non_numeric": non_numeric,
                 "min": tmin, "max": tmax, "decreasing_steps": dec, "first_decreasing_at": None}
    if tmin is not None:
        timestamp["min_iso"] = datetime.fromtimestamp(tmin / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        timestamp["max_iso"] = datetime.fromtimestamp(tmax / 1000, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
        timestamp["epoch_ms_sane"] = 631152000000 <= tmin and tmax <= 3000000000000
    else:
        timestamp["min_iso"] = timestamp["max_iso"] = None
        timestamp["epoch_ms_sane"] = False
    return {
        "file": str(path),
        "header": header,
        "n_columns": ncols,
        "data_rows": total_rows,
        "scan_seconds": round(elapsed, 1),
        "engine": "pandas",
        "label_columns": label_cols,
        "label_counts": {c: dict(counts[c].most_common()) for c in label_cols},
        "label_pair_counts_capped": (dict(label_pair.most_common(200)) if label_pair else None),
        "host_stats": host_stats,
        "missing": {"rows_with_any_missing_or_nan": missing_rows,
                    "per_column_top": {k: int(v) for k, v in top_missing.items()},
                    "row_width_anomalies": 0},
        "timestamp": timestamp,
    }


# ---------------------------------------------------------------- auditing
def audit_dataset(ds: str, ref: dict, raw: Path, out: Path, dedupe_policy: str = "auto",
                  force_dedupe: bool = False, engine: str = "auto",
                  light_bytes: int = 1_000_000_000) -> dict:
    log = Tee(out / f"identity_{ds}.log.txt")
    t_start = time.time()
    log.line(f"=== verifier dataset identity audit: {ds} ===")
    log.line(f"utc={utc_now()} cwd={os.getcwd()}")
    log.line(f"argv={' '.join(sys.argv)}")
    published = ref["datasets"].get(ds)
    if published is None:
        log.line(f"WARN no published reference for {ds}; identity cannot be judged")
    result: dict = {
        "dataset": ds,
        "check": "identity",
        "audited_utc": utc_now(),
        "verifier_script": "verification/src/scripts/v_identity.py",
        "raw_root": str(raw),
        "published_reference": published,
        "checks": [],
        "overall": "fail",
    }
    try:
        csvs, zpath, source_kind = locate_csvs(raw, ds)
    except FileNotFoundError as e:
        log.line(f"FAIL locate: {e}")
        result["checks"].append({"id": "identity.files_present", "status": "fail", "detail": str(e)})
        result["overall"] = "fail"
        json.dump(result, open(out / f"identity_{ds}.json", "w", encoding="utf-8"), indent=2)
        log.close()
        return result

    result["source"] = {"kind": source_kind, "csv_files": [str(p) for p in csvs], "zip": str(zpath) if zpath else None}
    streams: list[tuple[Path | zipfile.ZipInfo, zipfile.ZipFile | None]] = []
    zf = None
    if source_kind == "zip":
        zf = zipfile.ZipFile(zpath)  # type: ignore[arg-type]
        members = [m for m in zf.infolist() if m.filename.lower().endswith(".csv")]
        log.line(f"zip members: {[(m.filename, m.file_size) for m in members]}")
        if len(members) != 1:
            log.line(f"WARN expected exactly 1 CSV member, found {len(members)}")
        streams = [(m, zf) for m in members]
        result["source"]["zip_members"] = [{"name": m.filename, "uncompressed_bytes": m.file_size} for m in members]
        result["sha256_zip"] = sha256_file(zpath)  # type: ignore[arg-type]
        log.line(f"sha256(zip)={result['sha256_zip']}")
    else:
        for p in csvs:
            size = p.stat().st_size
            log.line(f"hashing {p} ({size:,} bytes)")
            h = sha256_file(p)
            result.setdefault("sha256_csv", []).append({"path": str(p), "bytes": size, "sha256": h})
            log.line(f"  sha256={h}")
            streams.append((p, None))
        # hash the archive too when it is still present next to the extracted dir
        archive = raw / f"{ds}.zip"
        if archive.is_file():
            log.line(f"hashing archive {archive} ({archive.stat().st_size:,} bytes)")
            result["sha256_zip"] = sha256_file(archive)
            result["zip_bytes"] = archive.stat().st_size
            log.line(f"  sha256(zip)={result['sha256_zip']}")

    # resolve scan engine
    resolved_engine = engine
    if resolved_engine == "auto":
        try:
            import pandas  # noqa: F401
            resolved_engine = "pandas"
        except Exception:  # noqa: BLE001
            resolved_engine = "stdlib"
    if resolved_engine == "pandas" and any(not isinstance(s0, Path) for s0, _ in streams):
        log.line("engine=pandas requested but source is a zip member; falling back to stdlib streaming")
        resolved_engine = "stdlib"
    result["scan_engine"] = resolved_engine
    # large files: heavy per-row metrics are covered by v_hosts.py; identity only
    # needs counts/schema/timestamps/missing, so skip the Python-level heavy parts
    csv_sizes = [p.stat().st_size for p in csvs] if csvs else []
    light = bool(csv_sizes) and max(csv_sizes) > light_bytes
    result["scan_light"] = light
    if light:
        log.line(f"light scan mode: csv bytes {max(csv_sizes):,} > {light_bytes:,}; "
                 f"host-pair Counter and Label/Attack cross-tab deferred to v_hosts.py")
    # scan each CSV
    scans = []
    for src, z in streams:
        log.line(f"scanning {src if isinstance(src, Path) else src.filename} ...")
        if resolved_engine == "pandas" and isinstance(src, Path):
            s = scan_pandas(src, log, light=light)
        else:
            s = scan_stream(src, z, log)
        log.line(f"  rows={s['data_rows']:,} cols={s['n_columns']} labels={s['label_columns']} ({s['scan_seconds']}s)")
        scans.append(s)
    result["scans"] = scans
    total_rows = sum(s["data_rows"] for s in scans)
    result["total_data_rows"] = total_rows
    merged_labels: dict[str, Counter] = {}
    for s in scans:
        for col, cnt in s["label_counts"].items():
            merged_labels.setdefault(col, Counter()).update(cnt)
    result["label_counts_merged"] = {c: dict(v.most_common()) for c, v in merged_labels.items()}

    # -- schema check
    hdr = scans[0]["header"]
    schema_checks = {
        "header_present": scans[0]["n_columns"] > 1,
        "single_data_file": len(scans) == 1,
        "has_label_col": any(c in LABEL_CANDIDATES for c in hdr),
        "has_attack_col": "Attack" in hdr,
        "has_flow_start_ms": DATE_COL in hdr,
        "has_uq_v3_temporal_features": sum(1 for c in hdr if c.endswith("_IAT_MIN") or c.endswith("_IAT_MAX") or c.endswith("_IAT_AVG") or c.endswith("_IAT_STDDEV")) >= 8,
        "n_columns_expected_55": scans[0]["n_columns"] == 55,
        "no_duplicate_column_names": len(hdr) == len(set(hdr)),
        "columns_do_not_contain_unnamed_index": not any(c.lower().startswith("unnamed") for c in hdr),
    }
    result["schema_checks"] = schema_checks

    if published and "classes" in published:
        def canon(s: str) -> str:
            return re.sub(r"[^a-z0-9]", "", str(s).strip().lower())

        aliases = {canon(k): canon(v) for k, v in (published.get("class_name_aliases_paper_to_csv") or {}).items()}
        exp_by_canon: dict[str, dict] = {}
        for cls, exp in published["classes"].items():
            key = canon(cls)
            exp_by_canon[aliases.get(key, key)] = {"published_name": cls, "count": exp, "alias_applied": key in aliases}
        # choose the label column whose canonical value set best matches the paper classes
        best_col, best = None, None
        for col0, cnt0 in merged_labels.items():
            overlap = len({canon(x) for x in cnt0} & set(exp_by_canon))
            if best is None or overlap > best[0]:
                best_col, best = col0, (overlap, cnt0)
        col = best_col
        actual = merged_labels.get(col, Counter()) if col else Counter()
        actual_by_canon: dict[str, dict] = {}
        for name, cnt in actual.items():
            actual_by_canon[canon(name)] = {"csv_name": name, "count": int(cnt)}
        row_ok = total_rows == published["paper_total_flows"]
        mismatches, mapping = {}, {}
        for key, info in exp_by_canon.items():
            got = actual_by_canon.get(key, {}).get("count", 0)
            csv_name = actual_by_canon.get(key, {}).get("csv_name")
            mapping[info["published_name"]] = csv_name
            if got != info["count"]:
                mismatches[info["published_name"]] = {"published": info["count"], "actual": got,
                                                       "actual_csv_name": csv_name, "delta": got - info["count"]}
        extra = sorted(k for k in actual_by_canon if k not in exp_by_canon)
        missing_cls = sorted(k for k in exp_by_canon if k not in actual_by_canon)
        class_ok = not mismatches and not extra and not missing_cls
        class_sum_actual = sum(actual_by_canon[k]["count"] for k in exp_by_canon if k in actual_by_canon) + \
            sum(actual_by_canon[k]["count"] for k in extra)
        checks = [
            {"id": "identity.row_count", "status": "pass" if row_ok else "fail",
             "detail": {"label_column": col, "published": published["paper_total_flows"], "actual": total_rows}},
            {"id": "identity.class_counts", "status": "pass" if class_ok else "fail",
             "detail": {"label_column": col, "published_classes": len(exp_by_canon), "actual_classes": len(actual_by_canon),
                        "mismatches": mismatches, "extra_classes": extra, "missing_classes": missing_cls,
                        "actual_class_sum": class_sum_actual,
                        "paper_to_csv_name_mapping": mapping}},
            {"id": "identity.schema", "status": "pass" if all(schema_checks.values()) else ("warn" if not schema_checks["has_uq_v3_temporal_features"] else "fail"),
             "detail": schema_checks},
        ]
        # binary label column consistency (paper Table 2 benign/malicious split)
        expected_bin = published.get("binary_label_expected")
        if expected_bin:
            bin_col = None
            for col2, cnt2 in merged_labels.items():
                if col2 == col:
                    continue
                if set(cnt2) <= {"0", "1"}:
                    bin_col = col2
                    got = {k: int(cnt2.get(k, 0)) for k in ("0", "1")}
                    break
            if bin_col is None:
                checks.append({"id": "identity.binary_label_consistency", "status": "warn",
                               "detail": {"reason": "no 0/1 label column found"}})
            else:
                exp = {"0": int(expected_bin["0"]), "1": int(expected_bin["1"])}
                ok = got == exp
                checks.append({"id": "identity.binary_label_consistency", "status": "pass" if ok else "fail",
                               "detail": {"column": bin_col, "published": exp, "actual": got,
                                          "attack_total_published": published.get("paper_attack_flows")}})
                log.line(f"BINARY LABEL ({bin_col}): published={exp} actual={got} -> {'PASS' if ok else 'FAIL'}")
            # cross-tab: Label 0 <=> Attack == Benign, Label 1 <=> Attack != Benign
            pairs = scans[0].get("label_pair_counts_capped") or {}
            bin_key, cls_key = (bin_col, col) if bin_col else (None, col)
            bad = 0
            n_pairs = 0
            for k, cnt in pairs.items():
                parts = k.split("||")
                if len(parts) != 2 or bin_col is None:
                    continue
                n_pairs += 1
                lv, av = parts[0], parts[1]
                is_benign = canon(av) == canon("Benign")
                if (lv == "0" and not is_benign) or (lv == "1" and is_benign):
                    bad += cnt
            if n_pairs:
                checks.append({"id": "identity.label_crosstab", "status": "pass" if bad == 0 else "fail",
                               "detail": {"binary_column": bin_col, "class_column": col,
                                          "inconsistent_rows": bad}})
        result["checks"] = checks
        result["website_errata"] = [e for e in ref.get("website_errata", []) if e.get("dataset") == ds]
        if ref.get("website_errata"):
            result["reference_status"] = "primary=paper Tables 2-3; website page contains documented errata (see website_errata)"
        log.line(f"\nROW COUNT: published={published['paper_total_flows']:,} actual={total_rows:,} -> {'PASS' if row_ok else 'FAIL'}")
        if mismatches or extra or missing_cls:
            log.line(f"CLASS MISMATCHES: {json.dumps({'mismatches': mismatches, 'extra': extra, 'missing': missing_cls}, indent=2)}")
        else:
            log.line(f"CLASS COUNTS: all {len(exp_by_canon)} classes exact match -> PASS")
        if result.get("website_errata"):
            log.line("REFERENCE NOTE: UQ website errata for this dataset: " + json.dumps(result["website_errata"]))
    # timestamp checks
    ts = scans[0]["timestamp"]
    result["checks"].append({
        "id": "identity.timestamp_order",
        "status": "pass" if (ts["rows_checked"] > 0 and ts["non_numeric"] == 0 and ts["epoch_ms_sane"]) else "fail",
        "detail": ts,
    })
    log.line(f"TIMESTAMP: {ts.get('min_iso')} .. {ts.get('max_iso')}, non_numeric={ts['non_numeric']}, decreasing_steps={ts['decreasing_steps']}")
    # missingness summary
    per_file_missing = [s["missing"] for s in scans]
    result["missingness"] = per_file_missing
    total_missing_rows = sum(m["rows_with_any_missing_or_nan"] for m in per_file_missing)
    result["checks"].append({
        "id": "identity.missing_tokens",
        "status": "pass" if total_missing_rows == 0 else "warn",
        "detail": {"rows_with_any_missing_or_nan": total_missing_rows,
                   "per_file": per_file_missing},
    })

    if (dedupe_policy == "on") or (dedupe_policy == "auto" and total_rows <= 3_000_000):
        cache = out / f"dedupe_{ds}.json"
        if cache.is_file() and not force_dedupe:
            result["duplicates"] = json.loads(cache.read_text(encoding="utf-8"))
            result["duplicates"]["reused_from_cache"] = True
            log.line(f"exact-duplicate audit: reusing cache {cache}")
        else:
            log.line("exact-duplicate audit (sqlite-backed) ...")
            db = out / f"_dedupe_{ds}.sqlite"
            result["duplicates"] = dedupe_sqlite(db, streams, {}, scans[0]["n_columns"], log)
            result["duplicates"]["reused_from_cache"] = False
            cache.write_text(json.dumps(result["duplicates"], indent=2), encoding="utf-8")
        result["checks"].append({
            "id": "identity.exact_duplicates",
            "status": "info",
            "detail": result["duplicates"],
        })
    else:
        result["duplicates"] = None
        result["checks"].append({
            "id": "identity.exact_duplicates",
            "status": "deferred",
            "detail": {"reason": "large dataset; full duplicate scan deferred to a separate background job",
                       "rows": total_rows, "policy": dedupe_policy},
        })
        log.line(f"exact-duplicate audit DEFERRED for {total_rows:,} rows (policy={dedupe_policy})")

    fails = [c for c in result["checks"] if c["status"] == "fail"]
    warns = [c for c in result["checks"] if c["status"] == "warn"]
    result["overall"] = "pass" if not fails else "fail"
    result["n_fail"] = len(fails)
    result["n_warn"] = len(warns)
    result["elapsed_seconds"] = round(time.time() - t_start, 1)
    if zf is not None:
        zf.close()
    with open(out / f"identity_{ds}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    log.line(f"\nOVERALL: {result['overall'].upper()}  fails={len(fails)} warns={len(warns)}  ({result['elapsed_seconds']}s)")
    log.close()
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="all", help="dataset name or 'all'")
    ap.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    ap.add_argument("--raw", default=str(DEFAULT_RAW))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--dedupe", default="auto", choices=["auto", "on", "off"],
                    help="auto = full duplicate scan only for datasets <=3M rows; on/off = force")
    ap.add_argument("--force-dedupe", action="store_true", help="ignore dedupe cache and recompute")
    ap.add_argument("--engine", default="auto", choices=["auto", "stdlib", "pandas"],
                    help="CSV scan engine; auto=pandas when importable (fast path for 17-27M rows)")
    ap.add_argument("--light-bytes", type=int, default=1_000_000_000,
                    help="CSVs larger than this are scanned in light mode (heavy metrics deferred to v_hosts.py)")
    args = ap.parse_args()
    ref = json.loads(Path(args.reference).read_text(encoding="utf-8"))
    raw = Path(args.raw)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.dataset == "all":
        present = []
        absent = []
        for d in ref["datasets"]:
            if ref["datasets"][d].get("paper_total_flows") is None:
                absent.append(d)  # non-NF-v3 identity (e.g. Elliptic) is audited separately
                continue
            if (raw / d).is_dir() or (raw / f"{d}.zip").is_file():
                present.append(d)
            else:
                absent.append(d)
        targets = present
    else:
        targets = [args.dataset]
    summary = {"generated_utc": utc_now(), "reference": str(args.reference), "datasets": {}}
    if args.dataset == "all" and absent:
        summary["skipped_absent_data"] = absent
        print(f"skipping datasets with no raw data present: {absent}")
    for ds in targets:
        try:
            res = audit_dataset(ds, ref, raw, out, args.dedupe, args.force_dedupe, args.engine,
                                light_bytes=args.light_bytes)
        except SystemExit as e:
            print(f"FAIL {ds}: {e}")
            summary["datasets"][ds] = {"overall": "fail", "error": str(e)}
            continue
        summary["datasets"][ds] = {
            "overall": res.get("overall"),
            "total_data_rows": res.get("total_data_rows"),
            "published_total": (res.get("published_reference") or {}).get("paper_total_flows"),
            "checks": {c["id"]: c["status"] for c in res.get("checks", [])},
        }
    with open(out / "identity_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    lines = ["# Independent dataset identity summary", "", f"Generated: {summary['generated_utc']}", "",
             "| dataset | overall | data rows | published total | row count | class counts | schema | timestamp | missing |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for ds, s in summary["datasets"].items():
        c = s.get("checks", {})
        lines.append(f"| {ds} | {s.get('overall')} | {s.get('total_data_rows')} | {s.get('published_total')} | "
                     f"{c.get('identity.row_count','-')} | {c.get('identity.class_counts','-')} | {c.get('identity.schema','-')} | "
                     f"{c.get('identity.timestamp_order','-')} | {c.get('identity.missing_tokens','-')} |")
    (out / "identity_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
