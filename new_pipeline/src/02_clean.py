# %% [markdown]
# # 02 - Cleaning, labels, periods and split groups
#
# This stage turns a raw NF-v3 CSV into one analysis table. It does **not**
# split, resample, fit a scaler, build graphs, or train anything.
#
# What it does:
#
# - validates the 55-column header and the 48-source-feature contract;
# - reads the CSV as strings and parses every number explicitly, so a value
#   cannot silently change dtype between chunks;
# - fails if the CSV parser drops a row (the audit reports `rows_lost_by_parser`);
# - removes exact-row duplicates and reports rows after exact dedupe;
# - builds quantised feature-duplicate groups (used for grouped splitting, not
#   for row removal);
# - flags label conflicts: identical features with different `Attack`;
# - applies negative/infinity sentinel rules and writes two missing indicators;
# - builds the fine, CIC-13, Shared-6 and Shared-core label views;
# - assigns the primary periods (UNSW chronological; CIC stratified-chronological);
# - builds the leak-aware split group (burst group merged with feature-duplicate
#   groups);
# - writes the cleaned table, label maps, class-period counts and a JSON report.
#
# No timestamps, IPs, source ports or labels are used as model features.

# %%
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

print("Notebook: 02 - clean")
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
    raise RuntimeError("Could not locate repository (datasets/ + new_pipeline/). Set XDRIFT_ROOT.")


PROJECT_ROOT = find_project_root()
ON_KAGGLE = Path("/kaggle/input").exists()
OUTPUT_ROOT = (
    Path("/kaggle/working/xdrift_pipeline")
    if ON_KAGGLE
    else PROJECT_ROOT / "new_pipeline" / "results"
)
REPORT_ROOT = OUTPUT_ROOT / "clean"
CLEAN_DATA_ROOT = (
    Path("/kaggle/working/xdrift_clean")
    if ON_KAGGLE
    else PROJECT_ROOT / "datasets" / "processed"
)
REPORT_ROOT.mkdir(parents=True, exist_ok=True)
print("project root:", PROJECT_ROOT)
print("kaggle      :", ON_KAGGLE)
print("report root :", REPORT_ROOT)
print("data root   :", CLEAN_DATA_ROOT)

# %%
DATASETS = {
    "NF-UNSW-NB15-v3": "NF-UNSW-NB15-v3.csv",
    "NF-CSE-CIC-IDS2018-v3": "NF-CICIDS2018-v3.csv",
}
EXPECTED_COLUMNS = [
    "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "IPV4_SRC_ADDR",
    "L4_SRC_PORT", "IPV4_DST_ADDR", "L4_DST_PORT", "PROTOCOL", "L7_PROTO",
    "IN_BYTES", "IN_PKTS", "OUT_BYTES", "OUT_PKTS", "TCP_FLAGS",
    "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS", "FLOW_DURATION_MILLISECONDS",
    "DURATION_IN", "DURATION_OUT", "MIN_TTL", "MAX_TTL", "LONGEST_FLOW_PKT",
    "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN",
    "SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES",
    "RETRANSMITTED_IN_BYTES", "RETRANSMITTED_IN_PKTS",
    "RETRANSMITTED_OUT_BYTES", "RETRANSMITTED_OUT_PKTS",
    "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT",
    "NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES",
    "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES",
    "NUM_PKTS_1024_TO_1514_BYTES", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT",
    "ICMP_TYPE", "ICMP_IPV4_TYPE", "DNS_QUERY_ID", "DNS_QUERY_TYPE",
    "DNS_TTL_ANSWER", "FTP_COMMAND_RET_CODE", "SRC_TO_DST_IAT_MIN",
    "SRC_TO_DST_IAT_MAX", "SRC_TO_DST_IAT_AVG", "SRC_TO_DST_IAT_STDDEV",
    "DST_TO_SRC_IAT_MIN", "DST_TO_SRC_IAT_MAX", "DST_TO_SRC_IAT_AVG",
    "DST_TO_SRC_IAT_STDDEV", "Label", "Attack",
]
LABEL_COLUMNS = ["Label", "Attack"]
DROP_COLUMNS = [
    "FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS",
    "IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_SRC_PORT", *LABEL_COLUMNS,
]
SOURCE_FEATURES = [c for c in EXPECTED_COLUMNS if c not in DROP_COLUMNS]
INDICATOR_BASES = ["SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES"]
MISSING_INDICATORS = [f"{c}_missing" for c in INDICATOR_BASES]
INDICATOR_THRESHOLD_PCT = 0.1
TEXT_COLUMNS = {"IPV4_SRC_ADDR", "IPV4_DST_ADDR", "Attack"}
# Columns that must parse to a finite number on every row; anything else is a
# corrupt file, not a "missing value".
REQUIRED_NUMERIC = ["FLOW_START_MILLISECONDS", "FLOW_END_MILLISECONDS", "Label"]

CHUNK_ROWS = int(os.environ.get("XDRIFT_CLEAN_CHUNK_ROWS", "250000"))
MAX_ROWS = int(os.environ.get("XDRIFT_CLEAN_MAX_ROWS", "0")) or None
# A row-limited run is a smoke test. It must never overwrite the full-run
# artifacts (guardrail 5: no silent replacement), so it writes elsewhere.
SMOKE = MAX_ROWS is not None
REPORT_DIR = REPORT_ROOT / "smoke" if SMOKE else REPORT_ROOT
REPORT_DIR.mkdir(parents=True, exist_ok=True)
_selected = os.environ.get("XDRIFT_CLEAN_DATASETS")
SELECTED = {s.strip() for s in _selected.split(",")} if _selected else set(DATASETS)
unknown = SELECTED - set(DATASETS)
if unknown:
    raise ValueError(f"unknown XDRIFT_CLEAN_DATASETS: {sorted(unknown)}; valid: {sorted(DATASETS)}")

assert len(EXPECTED_COLUMNS) == 55 and len(SOURCE_FEATURES) == 48
print("header      :", len(EXPECTED_COLUMNS), "columns")
print("source feats:", len(SOURCE_FEATURES))
print("indicators  :", MISSING_INDICATORS)

# %%
CIC13_MAP = {
    "SQL_Injection": "Web_attack",
    "Brute_Force_-Web": "Web_attack",
    "Brute_Force_-XSS": "Web_attack",
}

SHARED6_MAP = {
    "NF-UNSW-NB15-v3": {
        "Benign": "Benign",
        "DoS": "DoS/DDoS",
        "Backdoor": "Backdoor/brute force",
        "Exploits": "Web/exploit",
        "Reconnaissance": "Recon/infiltration",
        "Analysis": "Other",
        "Fuzzers": "Other",
        "Generic": "Other",
        "Shellcode": "Other",
        "Worms": "Other",
    },
    "NF-CSE-CIC-IDS2018-v3": {
        "Benign": "Benign",
        "DoS_attacks-Hulk": "DoS/DDoS",
        "DoS_attacks-GoldenEye": "DoS/DDoS",
        "DoS_attacks-Slowloris": "DoS/DDoS",
        "DoS_attacks-SlowHTTPTest": "DoS/DDoS",
        "DDOS_attack-HOIC": "DoS/DDoS",
        "DDOS_attacks-LOIC-HTTP": "DoS/DDoS",
        "DDOS_attack-LOIC-UDP": "DoS/DDoS",
        "FTP-BruteForce": "Backdoor/brute force",
        "SSH-Bruteforce": "Backdoor/brute force",
        "SQL_Injection": "Web/exploit",
        "Brute_Force_-Web": "Web/exploit",
        "Brute_Force_-XSS": "Web/exploit",
        "Infilteration": "Recon/infiltration",
        "Bot": "Other",
    },
}
SHARED_CORE = {"Benign", "DoS/DDoS", "Web/exploit", "Recon/infiltration"}

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
    local = PROJECT_ROOT / "datasets" / "raw" / name / file_name
    return local if local.is_file() else None


def count_data_lines(path: Path) -> int:
    lines, last = 0, b""
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            lines += block.count(b"\n")
            last = block[-1:]
    if last and last != b"\n":
        lines += 1
    return max(lines - 1, 0)


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return handle.readline().rstrip("\r\n").split(",")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


UNPARSEABLE: dict[str, int] = {}


def parse_numeric(raw: pd.Series, column: str = "") -> pd.Series:
    stripped = raw.str.strip()
    empty = stripped == ""
    parsed = pd.to_numeric(stripped.where(~empty), errors="coerce").astype("float64")
    # A non-empty token that did not parse is garbage, not a missing value.
    # errors="coerce" would hide it, so count it and report it.
    bad = int((parsed.isna() & ~empty).sum())
    if bad:
        UNPARSEABLE[column] = UNPARSEABLE.get(column, 0) + bad
    return parsed


def load_audit(dataset: str) -> dict:
    path = OUTPUT_ROOT / "data_audit" / f"{dataset}.json"
    if not path.is_file():
        print(f"[WARN] audit JSON not found: {path}")
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    print(f"audit      : {path} (v{data.get('audit_version', '?')})")
    return data


def map_hosts(series: pd.Series, mapping: dict[str, int]) -> np.ndarray:
    values = series.str.strip()
    for value in values.unique():
        if value not in mapping:
            mapping[value] = len(mapping)
    return values.map(mapping).to_numpy("int32")


def view_lookup(class_names: list[str], mapping: dict[str, str]) -> tuple[np.ndarray, list[str]]:
    """Class-level lookup table for a label view.

    Index it with the per-row fine code (`lookup[attack_code]`). Mapping at the
    class level (10-15 entries) instead of per row avoids building several
    20M-element Python lists on CIC.
    """
    mapped = [mapping.get(name, name) for name in class_names]
    classes = sorted(set(mapped))
    index = {name: i for i, name in enumerate(classes)}
    return np.array([index[name] for name in mapped], dtype="int16"), classes

# %%
results: dict[str, dict] = {}

for dataset in sorted(SELECTED):
    path = find_dataset_path(dataset)
    print("\n" + "#" * 78)
    print("# DATASET:", dataset)
    print("#" * 78)
    if path is None:
        print("[FAIL] raw CSV not found")
        results[dataset] = {"status": "missing", "path": None}
        continue

    header = read_header(path)
    if header != EXPECTED_COLUMNS:
        raise RuntimeError(f"{dataset}: header contract mismatch\n"
                           f"missing={[c for c in EXPECTED_COLUMNS if c not in header]}\n"
                           f"extra={[c for c in header if c not in EXPECTED_COLUMNS]}")
    audit = load_audit(dataset)
    physical_rows = count_data_lines(path)
    if audit.get("rows_lost_by_parser"):
        raise RuntimeError(f"{dataset}: audit reports {audit['rows_lost_by_parser']} rows lost "
                           "by the CSV parser; fix the raw file/parser before cleaning.")
    n_rows = min(physical_rows, MAX_ROWS) if MAX_ROWS else physical_rows
    print("file        :", path)
    print("size        :", f"{path.stat().st_size:,} bytes")
    print("raw rows    :", f"{physical_rows:,}", "| reading:", f"{n_rows:,}")

    features = np.full((n_rows, len(SOURCE_FEATURES)), np.nan, dtype="float32")
    ts_start = np.full(n_rows, -1, dtype="int64")
    ts_end = np.full(n_rows, -1, dtype="int64")
    host_src = np.full(n_rows, -1, dtype="int32")
    host_dst = np.full(n_rows, -1, dtype="int32")
    label = np.full(n_rows, -1, dtype="int8")
    attack_code = np.full(n_rows, -1, dtype="int16")
    seen_exact: set[int] = set()
    host_map: dict[str, int] = {}
    attack_code_map: dict[str, int] = {}
    attack_names: list[str] = []
    kept = 0
    chunks = 0
    rows_read = 0
    raw_row = np.full(n_rows, -1, dtype="int64")  # raw CSV data-row index of each kept row
    sentinel_neg = np.zeros(len(SOURCE_FEATURES), dtype="int64")
    sentinel_inf = np.zeros(len(SOURCE_FEATURES), dtype="int64")
    run_warnings: list[str] = []
    UNPARSEABLE.clear()
    if SMOKE:
        run_warnings.append(f"SMOKE RUN: first {n_rows:,} rows only; not for analysis")

    reader = pd.read_csv(
        path, dtype=str, na_filter=False, on_bad_lines="error",
        chunksize=CHUNK_ROWS, encoding="utf-8-sig", nrows=n_rows,
    )
    for raw in reader:
        if rows_read + len(raw) > n_rows:
            raise RuntimeError(f"{dataset}: parser returned more rows than the physical count {n_rows}")
        chunk_first_row = rows_read
        rows_read += len(raw)
        chunks += 1
        raw = raw.reindex(columns=EXPECTED_COLUMNS)
        parsed: dict[str, pd.Series] = {}
        for column in EXPECTED_COLUMNS:
            if column in TEXT_COLUMNS:
                parsed[column] = raw[column].str.strip()
            else:
                parsed[column] = parse_numeric(raw[column], column)

        # Hard validation: timestamps and Label drive sorting, periods and the
        # binary label; NaN there would be cast to garbage int64/int8 below.
        for column in REQUIRED_NUMERIC:
            if parsed[column].isna().any():
                raise RuntimeError(f"{dataset}: {int(parsed[column].isna().sum())} empty/unparseable "
                                   f"values in required column {column} (chunk {chunks})")
        for column in sorted(TEXT_COLUMNS):
            if (parsed[column] == "").any():
                raise RuntimeError(f"{dataset}: empty values in required text column {column} (chunk {chunks})")

        src_ids = map_hosts(parsed["IPV4_SRC_ADDR"], host_map)
        dst_ids = map_hosts(parsed["IPV4_DST_ADDR"], host_map)
        attack = parsed["Attack"]
        for name in attack.unique():
            if name not in attack_code_map:
                attack_code_map[name] = len(attack_names)
                attack_names.append(name)
        attack_ids = attack.map(attack_code_map).to_numpy("int16")

        # Exact duplicate key = every physical column. IPs are represented by
        # their one-to-one host ids; NaN is kept as NaN, not imputed.
        key = pd.DataFrame({
            c: parsed[c].to_numpy("float64") for c in SOURCE_FEATURES
        })
        key["start"] = parsed["FLOW_START_MILLISECONDS"].to_numpy("float64")
        key["end"] = parsed["FLOW_END_MILLISECONDS"].to_numpy("float64")
        key["src"] = src_ids.astype("float64")
        key["dst"] = dst_ids.astype("float64")
        key["src_port"] = parsed["L4_SRC_PORT"].to_numpy("float64")
        key["label"] = parsed["Label"].to_numpy("float64")
        key["attack"] = attack_ids.astype("float64")
        exact_hash = pd.util.hash_pandas_object(key, index=False).to_numpy("uint64")
        # Deduplicate within this chunk and against all previous chunks. The
        # first occurrence in file order is kept.
        duplicate_in_chunk = pd.Series(exact_hash).duplicated(keep="first").to_numpy()
        hash_list = exact_hash.tolist()  # python ints: no per-element int() in the loop
        duplicate_earlier = np.fromiter(
            (h in seen_exact for h in hash_list), dtype=bool, count=len(hash_list)
        )
        keep = ~(duplicate_in_chunk | duplicate_earlier)
        seen_exact.update(exact_hash[keep].tolist())
        if not keep.any():
            continue

        # Sentinel rules are applied to KEPT rows only, and every conversion is
        # counted per column so it can be checked against the audit.
        source = pd.DataFrame({c: parsed[c] for c in SOURCE_FEATURES})[keep]
        values64 = source.to_numpy("float64")
        sentinel_inf += np.isinf(values64).sum(axis=0)
        sentinel_neg += ((values64 < 0) & np.isfinite(values64)).sum(axis=0)
        source = source.replace([np.inf, -np.inf], np.nan).mask(source < 0)

        k = int(keep.sum())
        raw_row[kept:kept + k] = chunk_first_row + np.flatnonzero(keep)
        features[kept:kept + k] = source.to_numpy("float32")
        ts_start[kept:kept + k] = parsed["FLOW_START_MILLISECONDS"].to_numpy("float64")[keep].astype("int64")
        ts_end[kept:kept + k] = parsed["FLOW_END_MILLISECONDS"].to_numpy("float64")[keep].astype("int64")
        host_src[kept:kept + k] = src_ids[keep]
        host_dst[kept:kept + k] = dst_ids[keep]
        label[kept:kept + k] = parsed["Label"].to_numpy("float64")[keep].astype("int8")
        attack_code[kept:kept + k] = attack_ids[keep]
        kept += k
        if chunks % 10 == 0:
            print(f"  chunk {chunks:4d}  raw {rows_read:12,d}  kept {kept:12,d}")

    # The parser must deliver exactly the physical row count. Without this check
    # a silently dropped row would be counted as an "exact duplicate" below.
    if rows_read != n_rows:
        raise RuntimeError(f"{dataset}: parser read {rows_read:,} rows but the file has {n_rows:,} "
                           "data lines; fix the raw file/parser before cleaning.")
    features, ts_start, ts_end = features[:kept], ts_start[:kept], ts_end[:kept]
    host_src, host_dst = host_src[:kept], host_dst[:kept]
    label, attack_code = label[:kept], attack_code[:kept]
    row_index = raw_row[:kept]  # raw CSV data-row index (0-based), not a post-dedupe position
    exact_duplicate_rows = rows_read - kept
    print("rows kept   :", f"{kept:,}", "| exact duplicates:", f"{exact_duplicate_rows:,}")

    if kept == 0:
        raise RuntimeError(f"{dataset}: no rows kept")

    # Stable class codes: sort the names so codes do not depend on which class
    # happens to appear first in the file (smoke slice vs full file).
    sorted_names = sorted(attack_names)
    remap = np.array([sorted_names.index(n) for n in attack_names], dtype="int16")
    attack_code = remap[attack_code]
    attack_names = sorted_names

    # Label contract checks (fail loudly instead of passing unknown classes on).
    if "Benign" not in attack_names:
        raise RuntimeError(f"{dataset}: no 'Benign' class found in Attack column: {attack_names}")
    unmapped = sorted(set(attack_names) - set(SHARED6_MAP[dataset]))
    if unmapped:
        raise RuntimeError(f"{dataset}: classes missing from SHARED6_MAP: {unmapped}")
    absent = sorted(set(SHARED6_MAP[dataset]) - set(attack_names))
    if absent:
        run_warnings.append(f"classes in SHARED6_MAP never seen in the data: {absent}")
    if not set(np.unique(label).tolist()) <= {0, 1}:
        raise RuntimeError(f"{dataset}: Label must be binary 0/1, got {np.unique(label).tolist()}")
    benign_code = attack_names.index("Benign")
    label_attack_inconsistent = int(((label == 0) != (attack_code == benign_code)).sum())
    if label_attack_inconsistent:
        run_warnings.append(f"{label_attack_inconsistent:,} rows where Label disagrees with Attack=='Benign'")
    if UNPARSEABLE:
        run_warnings.append(f"unparseable non-empty tokens coerced to NaN: {dict(UNPARSEABLE)}")

    # Feature-duplicate groups use float32 values rounded to six decimals.
    # This is deliberately conservative for near-duplicates, not a row filter.
    feature_hash = np.empty(kept, dtype="uint64")
    for start in range(0, kept, CHUNK_ROWS):
        stop = min(start + CHUNK_ROWS, kept)
        block = pd.DataFrame(
            np.round(features[start:stop].astype("float64"), 6) + 0.0,  # +0.0 folds -0.0 into 0.0
            columns=SOURCE_FEATURES,
        )
        feature_hash[start:stop] = pd.util.hash_pandas_object(block, index=False).to_numpy("uint64")
    feature_group_raw, _ = pd.factorize(feature_hash)
    feature_group = feature_group_raw.astype("int32")
    feature_duplicate_rows = kept - int(feature_group.max()) - 1
    feature_group_count = int(feature_group.max()) + 1
    print("feature dups:", f"{feature_duplicate_rows:,}", "| groups:", f"{feature_group_count:,}")

    pair = pd.DataFrame({"group": feature_group, "attack": attack_code})
    n_labels = pair.drop_duplicates().groupby("group")["attack"].size()
    conflict_groups = n_labels.index[n_labels > 1]
    label_conflict = np.isin(feature_group, conflict_groups.to_numpy())
    label_conflict_groups = int(len(conflict_groups))
    label_conflict_rows = int(label_conflict.sum())
    print("label conflicts:", f"{label_conflict_groups:,} groups / {label_conflict_rows:,} rows")

    # Missing rate per source feature AFTER sentinel handling, checked against the
    # 0.1% indicator rule (D7.7) and the 50-input width contract.
    nan_counts = np.zeros(len(SOURCE_FEATURES), dtype="int64")
    for start in range(0, kept, CHUNK_ROWS):
        nan_counts += np.isnan(features[start:start + CHUNK_ROWS]).sum(axis=0)
    nan_pct = 100.0 * nan_counts / kept
    over_threshold = [c for c, p in zip(SOURCE_FEATURES, nan_pct) if p > INDICATOR_THRESHOLD_PCT]
    unexpected_over = sorted(set(over_threshold) - set(INDICATOR_BASES))
    below_threshold = [c for c in INDICATOR_BASES
                       if nan_pct[SOURCE_FEATURES.index(c)] <= INDICATOR_THRESHOLD_PCT]
    if unexpected_over:
        run_warnings.append(f"columns above {INDICATOR_THRESHOLD_PCT}% missing but without an "
                            f"indicator (width contract is 50): {unexpected_over}")
    if below_threshold:
        run_warnings.append(f"indicator columns at or below the {INDICATOR_THRESHOLD_PCT}% threshold: "
                            f"{below_threshold}")
    print("missing >0.1%:", over_threshold)
    for w in run_warnings:
        print("[WARN]", w)

    # Stable chronological sort. UNSW has 335k backwards steps; CIC has one.
    order = np.argsort(ts_start, kind="stable")
    features, ts_start, ts_end = features[order], ts_start[order], ts_end[order]
    host_src, host_dst = host_src[order], host_dst[order]
    label, attack_code = label[order], attack_code[order]
    feature_group = feature_group[order]
    label_conflict = label_conflict[order]
    row_index = row_index[order]
    print("sorted      : yes (stable by FLOW_START_MILLISECONDS)")

    # Period assignment.
    period = np.zeros(kept, dtype="int8")
    if dataset == "NF-UNSW-NB15-v3":
        gaps = np.diff(ts_start)
        boundary = int(np.argmax(gaps)) + 1 if gaps.size else 0
        period[:boundary] = 1
        remaining = kept - boundary
        if remaining:
            period[boundary:] = (2 + (np.arange(remaining) * 4 // remaining)).astype("int8")
        print("periods     : UNSW chronological; session boundary at row",
              f"{boundary:,}", "gap", f"{gaps[boundary - 1] / 1000:.1f} s" if gaps.size else "n/a")
    else:
        for code in range(len(attack_names)):
            positions = np.flatnonzero(attack_code == code)
            for p, block in enumerate(np.array_split(positions, 5), start=1):
                period[block] = p
        print("periods     : CIC stratified-chronological K=5 (per-class time blocks)")
    if (period < 1).any() or (period > 5).any():
        raise RuntimeError(f"{dataset}: {int(((period < 1) | (period > 5)).sum())} rows without a valid period")
    print("period rows :", np.bincount(period, minlength=6)[1:].tolist())

    # OCI is a separate secondary schedule; stage 02 only records a placeholder.
    period_oci = np.full(kept, -1, dtype="int8")

    # Split group = feature-duplicate group merged with burst group
    # (src host + fine attack + 5-minute bucket).
    bucket = ts_start // 300_000
    burst_hash = pd.util.hash_pandas_object(
        pd.DataFrame({"src": host_src, "attack": attack_code, "bucket": bucket}),
        index=False,
    ).to_numpy("uint64")
    burst_group = pd.factorize(burst_hash)[0].astype("int32")
    n_burst = int(burst_group.max()) + 1
    n_feature = feature_group_count
    graph = coo_matrix(
        # int32, not int8: duplicate (burst, feature) pairs are summed on conversion
        # and 256 duplicates would wrap an int8 weight back to 0.
        (np.ones(kept, dtype="int32"), (burst_group, n_burst + feature_group)),
        shape=(n_burst + n_feature, n_burst + n_feature),
    )
    _, component = connected_components(graph + graph.T, directed=False)
    leak_group = component[burst_group].astype("int32")
    leak_sizes = np.bincount(leak_group)
    largest_leak = float(leak_sizes.max() / kept)
    # Splits are made inside each period, so also report the worst component share per period.
    largest_leak_by_period = {}
    for p in range(1, 6):
        in_p = period == p
        if in_p.any():
            largest_leak_by_period[str(p)] = round(
                100.0 * float(np.bincount(leak_group[in_p]).max()) / int(in_p.sum()), 4)
    split_group = leak_group.copy()
    split_strategy = "burst+feature-transitive"
    if largest_leak > 0.5:
        # A transitive feature-duplicate merge can collapse a large fraction of
        # this repetitive stream into one component. That is a useful warning,
        # but an unusable 70/10/20 split group. Keep the leak component in the
        # table, use burst groups for the actual split, and report the warning.
        split_group = burst_group.copy()
        split_strategy = "burst-only-fallback"
        print(f"[WARN] transitive feature merge largest component = {largest_leak:.1%} of rows; "
              "using burst groups for _split_group and keeping _leak_group for diagnosis")
    print("split groups:", f"{int(split_group.max()) + 1:,}", f"({split_strategy}); "
          f"leak components={int(leak_group.max()) + 1:,}, largest={largest_leak:.1%}")

    # Label views.
    # Views are built from class-level lookup tables, then indexed per row.
    if dataset.startswith("NF-CSE"):
        cic13_lookup, cic13_names = view_lookup(attack_names, CIC13_MAP)
        cic13_code = cic13_lookup[attack_code]
    else:
        cic13_code, cic13_names = attack_code.copy(), attack_names.copy()
    shared_lookup, shared_names = view_lookup(attack_names, SHARED6_MAP[dataset])
    shared_code = shared_lookup[attack_code]
    core_lookup = np.array([SHARED6_MAP[dataset][n] in SHARED_CORE for n in attack_names], dtype="int8")
    shared_core_code = core_lookup[attack_code]
    print("label views :", {"fine": len(attack_names), "cic13": len(cic13_names),
                            "shared6": len(shared_names), "shared_core": len(SHARED_CORE)})

    # Assemble output table.
    out = pd.DataFrame({name: features[:, i] for i, name in enumerate(SOURCE_FEATURES)})
    missing = np.column_stack([
        (pd.isna(features[:, SOURCE_FEATURES.index(col)])).astype("uint8")
        for col in INDICATOR_BASES
    ])
    out[MISSING_INDICATORS[0]] = missing[:, 0]
    out[MISSING_INDICATORS[1]] = missing[:, 1]
    out["Label"] = label
    out["Attack"] = pd.Categorical.from_codes(attack_code, categories=attack_names)
    out["Attack_cic13"] = pd.Categorical.from_codes(cic13_code, categories=cic13_names)
    out["Attack_shared6"] = pd.Categorical.from_codes(shared_code, categories=shared_names)
    out["Attack_shared_core"] = shared_core_code
    out["_period"] = period
    out["_period_oci"] = period_oci
    out["_split_group"] = split_group
    out["_burst_group"] = burst_group
    out["_leak_group"] = leak_group
    out["_feature_group"] = feature_group
    out["_label_conflict"] = label_conflict
    out["_row_index"] = row_index
    out["_host_src_id"] = host_src
    out["_host_dst_id"] = host_dst
    out["_timestamp_ms"] = ts_start
    out["_flow_end_ms"] = ts_end

    out_dir = CLEAN_DATA_ROOT / dataset / ("clean_smoke" if SMOKE else "clean")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        out_path = out_dir / "cleaned.parquet"
        out.to_parquet(out_path, index=False)
    except ImportError:
        out_path = out_dir / "cleaned.pkl"
        out.to_pickle(out_path)
        print("[WARN] pyarrow missing; wrote pickle fallback:", out_path)

    class_counts_arr = np.bincount(attack_code, minlength=len(attack_names))
    label_vals, label_cnts = np.unique(label, return_counts=True)
    audit_exact = audit.get("duplicate_rows", audit.get("exact_duplicate_rows"))
    report = {
        "status": "ok", "dataset": dataset, "version": "1.1",
        "smoke_run": SMOKE, "max_rows": MAX_ROWS,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_path": str(path), "source_size_bytes": path.stat().st_size,
        "source_sha256": sha256_file(path),
        "audit_rows_lost_by_parser": audit.get("rows_lost_by_parser"),
        "physical_rows": physical_rows, "rows_read": rows_read, "chunks": chunks,
        "exact_duplicate_rows": exact_duplicate_rows, "rows_after_exact_dedupe": kept,
        "feature_duplicate_rows": feature_duplicate_rows,
        "feature_duplicate_pct": round(100.0 * feature_duplicate_rows / kept, 6),
        "feature_groups": feature_group_count,
        "label_conflict_groups": label_conflict_groups,
        "label_conflict_rows": label_conflict_rows,
        "split_group_strategy": split_strategy,
        "split_groups": int(split_group.max()) + 1,
        "leak_components": int(leak_group.max()) + 1,
        "largest_leak_component_pct": round(100.0 * largest_leak, 6),
        "largest_leak_component_pct_by_period": largest_leak_by_period,
        "features_48": SOURCE_FEATURES, "missing_indicators": MISSING_INDICATORS,
        "model_inputs_50": [*SOURCE_FEATURES, *MISSING_INDICATORS],
        "class_counts": {name: int(class_counts_arr[i]) for i, name in enumerate(attack_names)},
        "label_counts": {str(int(v)): int(c) for v, c in zip(label_vals, label_cnts)},
        "unparseable_tokens": dict(UNPARSEABLE),
        "sentinel_counts_kept_rows": {
            c: {"negative_to_nan": int(sentinel_neg[i]), "inf_to_nan": int(sentinel_inf[i])}
            for i, c in enumerate(SOURCE_FEATURES) if sentinel_neg[i] or sentinel_inf[i]
        },
        "missing_pct_after_clean": {c: round(float(nan_pct[i]), 6)
                                    for i, c in enumerate(SOURCE_FEATURES) if nan_pct[i] > 0},
        "row_index_meaning": "0-based data-row index in the raw CSV (header excluded)",
        "period_rows": {str(i): int((period == i).sum()) for i in range(1, 6)},
        "periods": "UNSW chronological (session boundary + 4 equal blocks)" if dataset == "NF-UNSW-NB15-v3"
                   else "CIC stratified-chronological K=5",
        "oci_periods": "not materialized in this stage",
        "hosts": {"src_unique": int(np.unique(host_src).size), "dst_unique": int(np.unique(host_dst).size),
                  "all_unique": len(host_map)},
        "output_path": str(out_path), "output_sha256": sha256_file(out_path),
        "warnings": list(run_warnings),
    }
    if (audit and not SMOKE and isinstance(audit_exact, (int, float))
            and int(audit_exact) != exact_duplicate_rows):
        report["warnings"].append(
            f"exact duplicate count {exact_duplicate_rows:,} != audit {int(audit_exact):,}"
        )
    if split_strategy != "burst+feature-transitive":
        report["warnings"].append(
            f"transitive feature merge collapsed {largest_leak:.1%} of rows; "
            f"_split_group uses {split_strategy}"
        )
    if not audit:
        report["warnings"].append("audit JSON missing; parser loss check not available")
    report_path = REPORT_DIR / f"{dataset}.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "label_maps.json").write_text(json.dumps({
        "fine": attack_names,
        "cic13": cic13_names,
        "shared6": shared_names,
        "shared_core": sorted(SHARED_CORE),
    }, indent=2), encoding="utf-8")

    # bincount keeps zero cells (a class absent from a period must show as 0,
    # because the support rule masks it); crosstab would drop them.
    cell = attack_code.astype("int64") * 5 + (period.astype("int64") - 1)
    counts = pd.DataFrame(
        np.bincount(cell, minlength=len(attack_names) * 5).reshape(len(attack_names), 5),
        index=attack_names, columns=[1, 2, 3, 4, 5],
    )
    counts.index.name = "Attack"
    counts.to_csv(REPORT_DIR / f"{dataset}_class_period_counts.csv")
    taxonomy = pd.DataFrame({
        "fine": attack_names,
        "count": class_counts_arr.tolist(),
    })
    taxonomy["cic13"] = [CIC13_MAP.get(name, name) for name in attack_names]
    taxonomy["shared6"] = [SHARED6_MAP[dataset][name] for name in attack_names]
    taxonomy["shared_core"] = [SHARED6_MAP[dataset][name] in SHARED_CORE for name in attack_names]
    taxonomy.to_csv(REPORT_DIR / f"{dataset}_label_taxonomy.csv", index=False)
    results[dataset] = report
    print("saved       :", out_path)
    print("report      :", report_path)

# %%
for dataset, report in results.items():
    if report.get("status") != "ok":
        print(f"[FAIL] {dataset}: {report.get('status')}")
        continue
    print("\n" + "=" * 78)
    print("CLEAN SUMMARY:", dataset)
    print("=" * 78)
    print(f"  rows after exact dedupe : {report['rows_after_exact_dedupe']:,}")
    print(f"  feature duplicates      : {report['feature_duplicate_rows']:,} "
          f"({report['feature_duplicate_pct']}%) in {report['feature_groups']:,} groups")
    print(f"  label conflicts         : {report['label_conflict_groups']:,} groups / "
          f"{report['label_conflict_rows']:,} rows")
    print(f"  classes                 : {len(report['class_counts'])}")
    print(f"  periods                 : {report['period_rows']}")
    print(f"  outputs                 : {report['output_path']}")
    print(f"  report                  : {REPORT_DIR / (dataset + '.json')}")
    for w in report["warnings"]:
        print("  [WARN]", w)
print("\nNext: 03_eda. It must never split, resample or fit a scaler.")