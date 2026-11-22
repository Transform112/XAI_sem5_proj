"""Dataset identity verification (stage T3).

Why this module exists
----------------------
Before any modelling, the raw files must be proven to be the datasets we claim
they are. A wrong mirror, a truncated download or a silently re-labelled copy
would invalidate every later result. This module produces direct evidence from
the raw CSVs, without ever modifying them:

* file inventory + SHA-256 of each CSV and archive,
* exact row count compared with the UQ published count,
* header schema compared with the registry contract (55 NF-v3 columns),
* multiclass label distribution compared with published class counts,
* binary-vs-multiclass label consistency, missing labels,
* NaN / infinity counts,
* exact duplicate flow count (full-row hash),
* timestamp parseability, range, monotonicity,
* host cardinality (source, destination, union),
* line-count consistency (catches truncated files / embedded newlines),
* provenance archive size and SHA-256 agreement with ``provenance.json``.

Design constraints
------------------
* **Streaming**: the largest dataset has 27.5 M rows, so every statistic is
  accumulated chunk by chunk; nothing loads a full dataset into memory. Only
  the 8-byte row hashes are retained (220 MB at 27.5 M rows).
* **Never alters raw data**: files are opened read-only; no cache is written
  next to them.
* **Config-driven**: chunk sizes and comparison tolerances come from
  ``configs/audit.yaml`` (``identity`` section), with documented fallbacks for
  direct library use in tests.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.registry import DatasetSpec
from src.data.taxonomy import LabelTaxonomy, TaxonomyError
from src.utils.logging import get_logger
from src.utils.manifest import sha256_file, utc_now, write_json

LOGGER = get_logger("xdrift.identity")

#: Which reference table the published-counts comparison uses. Bumped whenever
#: the authoritative reference changes so stale artifacts are unambiguous (D-013):
#: the UQ website is known to permute Analysis/Backdoor/Shellcode, so the paper
#: (arXiv:2503.04404v3 Tables 2-3) is authoritative.
REFERENCE_VERSION = "paper_tables_2_3_arxiv_2503.04404v3"

#: Fallbacks used only when a key is absent from the config file.
DEFAULT_OPTIONS: dict[str, Any] = {
    "chunksize": 250_000,
    "encoding": "utf-8",
    "encoding_errors": "replace",
    "large_dataset_rows": 5_000_000,
    "large_dataset_sample_rows": 1_000_000,
    "hash_large_files": True,
    "hash_archives": False,
    "published_tolerance": 0,
    "duplicate_method": "full_row_hash",
    "host_cardinality": "exact",
    "timestamp_parse_fraction_min": 1.0,
    "check_binary_label_consistency": True,
    "progress_every_chunks": 10,
}

_MISSING_LABEL = "<missing>"


# ---------------------------------------------------------------------------
# Option handling
# ---------------------------------------------------------------------------
def resolve_options(options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Merge caller options over :data:`DEFAULT_OPTIONS` (config wins)."""
    resolved = dict(DEFAULT_OPTIONS)
    if options:
        for key, value in dict(options).items():
            if value is not None:
                resolved[key] = value
    return resolved


def _check(
    name: str,
    passed: bool,
    *,
    measured: Any = None,
    expected: Any = None,
    severity: str = "critical",
    detail: str = "",
) -> dict[str, Any]:
    return {
        "name": name,
        "passed": bool(passed),
        "severity": severity,
        "measured": measured,
        "expected": expected,
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# Low-level file helpers
# ---------------------------------------------------------------------------
def count_newlines(path: str | Path, chunk_size: int = 1 << 22) -> int:
    """Count ``\\n`` bytes without decoding the file (fast truncation check)."""
    total = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            total += block.count(b"\n")
    return total


def read_header(path: str | Path, *, encoding: str = "utf-8", delimiter: str = ",") -> list[str]:
    """Read only the header row of a CSV."""
    with open(path, "r", encoding=encoding, errors="replace", newline="") as handle:
        try:
            return next(csv.reader(handle, delimiter=delimiter))
        except StopIteration:
            return []


def _string_columns(spec: DatasetSpec) -> set[str]:
    columns = {
        spec.column("host_src"),
        spec.column("host_dst"),
        spec.column("label_multiclass"),
    }
    binary = spec.optional_column("label_binary")
    if binary:
        columns.add(binary)
    return {column for column in columns if column}


def _numeric_columns(spec: DatasetSpec) -> set[str]:
    return set(spec.log1p) | set(spec.numeric)


def _dtype_map(header: Sequence[str], spec: DatasetSpec) -> dict[str, Any]:
    """Typed read map: floats for numeric columns, objects for text columns.

    Unknown columns are read as objects; they are flagged by the schema check.
    """
    numeric = _numeric_columns(spec)
    strings = _string_columns(spec)
    dtype: dict[str, Any] = {}
    for column in header:
        if column in strings:
            dtype[column] = "object"
        elif column in numeric:
            dtype[column] = "float64"
        else:
            dtype[column] = "object"
    return dtype


def _normalise_for_hash(chunk: pd.DataFrame) -> pd.DataFrame:
    """Replace unhashable/odd cells so row hashing is deterministic."""
    normalised = chunk.copy()
    for column in normalised.columns:
        if normalised[column].dtype == object:
            normalised[column] = normalised[column].astype("string").fillna(_MISSING_LABEL)
    return normalised


# ---------------------------------------------------------------------------
# Streaming scan
# ---------------------------------------------------------------------------
def _new_state() -> dict[str, Any]:
    return {
        "rows": 0,
        "header": [],
        "label_counts": Counter(),
        "binary_counts": Counter(),
        "missing": Counter(),
        "inf": Counter(),
        "unparseable": Counter(),
        "label_missing": 0,
        "binary_missing": 0,
        "binary_violations": 0,
        "hosts_src": set(),
        "hosts_dst": set(),
        "host_missing": Counter(),
        "dup_hashes": [],
        "timestamp": {
            "column": None,
            "count": 0,
            "parse_fail": 0,
            "inversions": 0,
            "min": None,
            "max": None,
            "last": None,
        },
        "chunks": 0,
        "fallback": None,
    }


def _iterate_chunks(path: Path, header: Sequence[str], dtype: Mapping[str, Any], options: Mapping[str, Any]):
    return pd.read_csv(
        path,
        chunksize=int(options["chunksize"]),
        dtype=dict(dtype),
        encoding=str(options["encoding"]),
        encoding_errors=str(options["encoding_errors"]),
        low_memory=False,
        on_bad_lines="warn",
    )


def _accumulate_chunk(
    chunk: pd.DataFrame,
    state: dict[str, Any],
    spec: DatasetSpec,
    options: Mapping[str, Any],
) -> None:
    """Fold one chunk into the running aggregates."""
    if not state["header"]:
        state["header"] = [str(column) for column in chunk.columns]

    rows = len(chunk)
    state["rows"] += rows
    state["chunks"] += 1

    label_column = spec.column("label_multiclass")
    binary_column = spec.optional_column("label_binary")
    timestamp_column = spec.optional_column("timestamp_start")
    src_column = spec.column("host_src")
    dst_column = spec.column("host_dst")

    raw_missing = chunk.isna()

    # Coerce numeric columns defensively (the typed reader normally already did).
    numeric_columns = [column for column in _numeric_columns(spec) if column in chunk.columns]
    for column in numeric_columns:
        if chunk[column].dtype != np.float64:
            coerced = pd.to_numeric(chunk[column], errors="coerce")
            newly_missing = coerced.isna() & ~raw_missing[column]
            count = int(newly_missing.sum())
            if count:
                state["unparseable"][column] += count
            chunk[column] = coerced

    # Missing / infinity counts.
    missing = chunk.isna()
    for column in chunk.columns:
        count = int(missing[column].sum())
        if count:
            state["missing"][column] += count
    for column in numeric_columns:
        values = chunk[column].to_numpy(dtype="float64", na_value=np.nan)
        infinity = int(np.isinf(values).sum())
        if infinity:
            state["inf"][column] += infinity

    # Multiclass and binary label distributions.
    if label_column in chunk.columns:
        values = chunk[label_column].astype("string")
        for value, count in values.value_counts(dropna=False).items():
            key = _MISSING_LABEL if pd.isna(value) else str(value)
            state["label_counts"][key] += int(count)
        state["label_missing"] += int(values.isna().sum())

    if binary_column and binary_column in chunk.columns:
        values = chunk[binary_column].astype("string")
        for value, count in values.value_counts(dropna=False).items():
            key = _MISSING_LABEL if pd.isna(value) else str(value)
            state["binary_counts"][key] += int(count)
        state["binary_missing"] += int(values.isna().sum())

        if label_column in chunk.columns and options.get("check_binary_label_consistency", True):
            attack = chunk[label_column].astype("string")
            binary = values
            benign = attack.eq("Benign")
            comparable = attack.notna() & binary.notna()
            violations = int((comparable & benign & binary.ne("0")).sum()) + int(
                (comparable & ~benign & binary.ne("1")).sum()
            )
            state["binary_violations"] += violations

    # Timestamps: parseability, range and monotonicity.
    timestamps = state["timestamp"]
    timestamps["column"] = timestamp_column
    if timestamp_column and timestamp_column in chunk.columns:
        parsed = pd.to_numeric(chunk[timestamp_column], errors="coerce").to_numpy(dtype="float64")
        finite = np.isfinite(parsed)
        count = int(finite.sum())
        timestamps["count"] += count
        timestamps["parse_fail"] += int(parsed.size - count)
        values = parsed[finite]
        if values.size:
            if timestamps["min"] is None or float(values.min()) < timestamps["min"]:
                timestamps["min"] = float(values.min())
            if timestamps["max"] is None or float(values.max()) > timestamps["max"]:
                timestamps["max"] = float(values.max())
            timestamps["inversions"] += int((np.diff(values) < 0).sum())
            if timestamps["last"] is not None and float(values[0]) < float(timestamps["last"]):
                timestamps["inversions"] += 1
            timestamps["last"] = float(values[-1])

    # Host cardinality.
    for column, bucket in ((src_column, "hosts_src"), (dst_column, "hosts_dst")):
        if column in chunk.columns:
            series = chunk[column]
            state["host_missing"][column] += int(series.isna().sum())
            state[bucket].update(str(value) for value in series.dropna().unique())

    # Exact duplicates: preserve one uint64 hash per row.
    state["dup_hashes"].append(
        pd.util.hash_pandas_object(_normalise_for_hash(chunk), index=False).to_numpy(dtype="uint64")
    )

    progress_every = int(options.get("progress_every_chunks", 10))
    if progress_every > 0 and state["chunks"] % progress_every == 0:
        LOGGER.info("scanned %s rows (%s chunks)", f"{state['rows']:,}", state["chunks"])


def _run_pass(path: Path, header: Sequence[str], spec: DatasetSpec, options: Mapping[str, Any], *, typed: bool):
    """One complete streaming pass; returns (state, error)."""
    state = _new_state()
    dtype = _dtype_map(header, spec) if typed else {column: "object" for column in header}
    try:
        for chunk in _iterate_chunks(path, header, dtype, options):
            _accumulate_chunk(chunk, state, spec, options)
    except (ValueError, TypeError) as exc:  # malformed numeric text in typed mode
        return state, exc
    return state, None


def scan_csv(path: str | Path, spec: DatasetSpec, options: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Stream one CSV and return identity aggregates (no data retained)."""
    resolved = resolve_options(options)
    file_path = Path(path)
    header = read_header(file_path, encoding=str(resolved["encoding"]))
    state, error = _run_pass(file_path, header, spec, resolved, typed=True)
    if error is not None:
        LOGGER.warning(
            "typed CSV read failed for %s (%s); retrying with per-chunk numeric coercion",
            file_path,
            error,
        )
        state, second_error = _run_pass(file_path, header, spec, resolved, typed=False)
        if second_error is not None:
            raise ValueError(f"could not parse {file_path}: {second_error}") from second_error
        state["fallback"] = {
            "used": True,
            "reason": f"{type(error).__name__}: {error}",
            "method": "object dtype + pd.to_numeric(errors=coerce) per chunk",
        }
    else:
        state["fallback"] = {"used": False}

    state["header"] = header
    state["line_count"] = count_newlines(file_path)
    if state["dup_hashes"]:
        hashes = np.concatenate(state["dup_hashes"]) if len(state["dup_hashes"]) > 1 else state["dup_hashes"][0]
        unique = int(np.unique(hashes).size)
        state["duplicate_rows"] = int(state["rows"] - unique)
        del state["dup_hashes"]
    else:
        state["duplicate_rows"] = 0
    return state


# ---------------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------------
def _iso_from_epoch_millis(value: float | None) -> str | None:
    if value is None:
        return None
    try:
        return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc).isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):  # pragma: no cover - absurd timestamps
        return None


def _archive_for(raw_dir: Path, name: str) -> Path | None:
    """Locate the downloaded archive.

    ``src/scripts/download_data.ps1`` keeps archives at ``datasets/raw/<dataset>.zip``
    (one level above the extracted ``datasets/raw/<dataset>/`` directory); older
    layouts may have put the archive next to the CSV. Both are accepted.
    """
    for candidate in (raw_dir / f"{name}.zip", raw_dir.parent / f"{name}.zip"):
        if candidate.is_file():
            return candidate
    return None


def _provenance_zip_bytes(provenance: Mapping[str, Any]) -> tuple[int | None, str | None]:
    """Read the archive size, tolerating the legacy field-name defect (D-020).

    ``provenance.json`` used to record the Kaggle metadata "total bytes" under
    ``kaggle_total_bytes_uncompressed``; for ToN/BoT/CIC that value is actually
    the zip size (the Kaggle metadata was wrong), so the field was renamed to
    ``kaggle_metadata_total_bytes`` and an explicit ``zip_bytes`` was added.
    """
    if provenance.get("zip_bytes") is not None:
        return int(provenance["zip_bytes"]), "zip_bytes"
    for legacy in ("kaggle_metadata_total_bytes", "kaggle_total_bytes_uncompressed"):
        if provenance.get(legacy) is not None:
            return int(provenance[legacy]), legacy
    return None, None


def _provenance_checks(
    spec: DatasetSpec,
    raw_dir: Path,
    measured: Mapping[str, Any],
    files: Sequence[Mapping[str, Any]],
    options: Mapping[str, Any],
    provenance: Mapping[str, Any] | None,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if not provenance:
        return checks
    tolerance = int(options.get("published_tolerance", 0))

    if provenance.get("published_flow_count") is not None:
        published = int(provenance["published_flow_count"])
        checks.append(
            _check(
                "provenance_published_flow_count",
                abs(int(measured["rows"]) - published) <= tolerance,
                measured=measured["rows"],
                expected=published,
                severity="warn",
                detail="provenance.json flow total vs measured rows",
            )
        )

    archive = _archive_for(raw_dir, spec.name)
    expected_zip_bytes, zip_bytes_source = _provenance_zip_bytes(provenance)
    if archive is not None and expected_zip_bytes is not None:
        actual = archive.stat().st_size
        checks.append(
            _check(
                "provenance_zip_bytes",
                int(actual) == int(expected_zip_bytes),
                measured=int(actual),
                expected=int(expected_zip_bytes),
                detail=f"archive={archive.name}; provenance field={zip_bytes_source}",
            )
        )
        expected_sha = provenance.get("sha256")
        if expected_sha and options.get("hash_archives"):
            actual_sha = sha256_file(archive)
            checks.append(
                _check(
                    "provenance_zip_sha256",
                    str(actual_sha) == str(expected_sha),
                    measured=actual_sha,
                    expected=expected_sha,
                )
            )
    elif expected_zip_bytes is not None:
        checks.append(
            _check(
                "provenance_zip_present",
                False,
                measured=None,
                expected=str(raw_dir.parent / f"{spec.name}.zip"),
                severity="warn",
                detail="provenance.json lists an archive that is absent (it may have been removed after extraction)",
            )
        )

    extracted = provenance.get("extracted_files") or []
    by_name = {str(Path(item["relative_path"]).name): item for item in files if item.get("relative_path")}
    for item in extracted:
        name = str(item.get("name"))
        expected_bytes = item.get("bytes")
        record = by_name.get(name)
        if record is None or expected_bytes is None:
            continue
        checks.append(
            _check(
                f"provenance_extracted_bytes[{name}]",
                int(record["bytes"]) == int(expected_bytes),
                measured=record["bytes"],
                expected=int(expected_bytes),
            )
        )
    return checks


def verify_dataset(
    spec: DatasetSpec,
    raw_dir: str | Path,
    *,
    options: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Verify one dataset directory and return a JSON-ready identity report."""
    resolved = resolve_options(options)
    raw_path = Path(raw_dir)
    if not raw_path.is_dir():
        raise FileNotFoundError(f"raw dataset directory not found: {raw_path}")

    csv_files = spec.csv_files(raw_path)
    tolerance = int(resolved.get("published_tolerance", 0))
    hash_files = bool(resolved.get("hash_large_files", True))

    files: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    for csv_path in csv_files:
        stats = csv_path.stat()
        header = read_header(csv_path, encoding=str(resolved["encoding"]))
        digest = sha256_file(csv_path) if hash_files else None
        relative = csv_path.relative_to(raw_path)
        files.append(
            {
                "relative_path": str(relative).replace("\\", "/"),
                "bytes": stats.st_size,
                "sha256": digest,
                "header": header,
                "header_sha256": None,
            }
        )
        states.append(scan_csv(csv_path, spec, resolved))

    if not states:
        raise FileNotFoundError(
            f"no CSV matching {spec.csv_glob!r} found under {raw_path}"
        )

    # Combine multiple CSVs if present (the NF-v3 mirrors have exactly one).
    rows = sum(state["rows"] for state in states)
    label_counts: Counter = Counter()
    binary_counts: Counter = Counter()
    missing: Counter = Counter()
    inf: Counter = Counter()
    unparseable: Counter = Counter()
    host_missing: Counter = Counter()
    hosts_src: set[str] = set()
    hosts_dst: set[str] = set()
    duplicate_rows = 0
    label_missing = binary_missing = binary_violations = 0
    line_count = 0
    ts_count = ts_fail = ts_inversions = 0
    ts_min: float | None = None
    ts_max: float | None = None
    fallbacks: list[dict[str, Any]] = []
    for state in states:
        label_counts.update(state["label_counts"])
        binary_counts.update(state["binary_counts"])
        missing.update(state["missing"])
        inf.update(state["inf"])
        unparseable.update(state["unparseable"])
        host_missing.update(state["host_missing"])
        hosts_src.update(state["hosts_src"])
        hosts_dst.update(state["hosts_dst"])
        duplicate_rows += state["duplicate_rows"]
        label_missing += state["label_missing"]
        binary_missing += state["binary_missing"]
        binary_violations += state["binary_violations"]
        line_count += state["line_count"]
        ts = state["timestamp"]
        ts_count += ts["count"]
        ts_fail += ts["parse_fail"]
        ts_inversions += ts["inversions"]
        if ts["min"] is not None:
            ts_min = ts["min"] if ts_min is None else min(ts_min, ts["min"])
        if ts["max"] is not None:
            ts_max = ts["max"] if ts_max is None else max(ts_max, ts["max"])
        if state.get("fallback", {}).get("used"):
            fallbacks.append(state["fallback"])

    header = states[0]["header"]
    expected_columns = list(spec.all_columns())
    measured_set = set(header)
    expected_set = set(expected_columns)
    missing_columns = sorted(expected_set - measured_set)
    extra_columns = sorted(measured_set - expected_set)
    matches_spec = not missing_columns and not extra_columns

    # Label distribution.
    measured_classes = {name: int(count) for name, count in label_counts.items() if name != _MISSING_LABEL}
    label_total = int(sum(measured_classes.values()))
    # Documented class-name aliases (paper name <-> raw CSV spelling), e.g. ToN
    # lower-case labels; the canonical distribution is what the paper tables use.
    aliases = spec.canonical_class_map()
    alias_applied: dict[str, str] = {}
    measured_canonical: dict[str, int] = {}
    for class_name, count in measured_classes.items():
        canonical = aliases.get(class_name, class_name)
        measured_canonical[canonical] = measured_canonical.get(canonical, 0) + int(count)
        if canonical != class_name:
            alias_applied[canonical] = class_name
    # D-031: documented fine->coarse label taxonomy (e.g. CIC-IDS2018 15 -> 7).
    taxonomy = LabelTaxonomy.from_spec(spec)
    taxonomy_error: str | None = None
    aggregated_classes: dict[str, int] = {}
    taxonomy_mismatches: list[dict[str, Any]] = []
    if taxonomy is not None:
        try:
            aggregated_classes = taxonomy.aggregate_counts(measured_canonical)
        except TaxonomyError as exc:
            taxonomy_error = str(exc)
        if taxonomy_error is None and taxonomy.published_aggregated_counts:
            for class_name in sorted(taxonomy.published_aggregated_counts):
                published = int(taxonomy.published_aggregated_counts[class_name])
                measured_value = int(aggregated_classes.get(class_name, 0))
                if published != measured_value:
                    taxonomy_mismatches.append(
                        {"class": class_name, "published": published, "measured": measured_value}
                    )
    comparison_classes = (
        aggregated_classes
        if (taxonomy is not None and taxonomy_error is None)
        else measured_canonical
    )
    published_flows = spec.published_flows
    published_classes = spec.published_classes
    tolerance = int(resolved.get("published_tolerance", 0))
    rows_match = published_flows is None or abs(rows - published_flows) <= tolerance
    row_delta = None if published_flows is None else rows - published_flows
    label_mismatches = [
        {"class": class_name, "published": published, "measured": comparison_classes.get(class_name, 0)}
        for class_name, published in published_classes.items()
        if published is not None and comparison_classes.get(class_name, 0) != published
    ]

    # Benign/attack split and attack total (paper Table 3 comparison).
    classes_expected = list(spec.class_order) if spec.class_order else []
    benign_class = "Benign" if "Benign" in comparison_classes else (classes_expected[0] if classes_expected else None)
    measured_attack_total = (
        int(sum(count for name, count in comparison_classes.items() if name != benign_class))
        if benign_class is not None
        else None
    )
    published_attack_total = spec.published.get("attack_total") if isinstance(spec.published, Mapping) else None
    published_binary = spec.published.get("binary_label") if isinstance(spec.published, Mapping) else None
    measured_binary = {name: int(count) for name, count in binary_counts.items() if name != _MISSING_LABEL}
    binary_mismatches = []
    if isinstance(published_binary, Mapping):
        for key, published_count in published_binary.items():
            measured_count = measured_binary.get(str(key))
            if measured_count is not None and int(published_count) != int(measured_count):
                binary_mismatches.append(
                    {"label": str(key), "published": int(published_count), "measured": int(measured_count)}
                )

    # UQ website values are known to be wrong (D-013): reported, never enforced.
    website_classes = spec.published.get("website_classes") if isinstance(spec.published, Mapping) else None
    website_mismatches: list[dict[str, Any]] = []
    if isinstance(website_classes, Mapping):
        for class_name, website_count in website_classes.items():
            if website_count is None:
                continue
            measured_count = comparison_classes.get(str(class_name), 0)
            if int(website_count) != int(measured_count):
                website_mismatches.append(
                    {"class": str(class_name), "website": int(website_count), "measured": int(measured_count)}
                )
    alternate_flow_counts: dict[str, Any] = {}
    alternate = spec.published.get("alternate_flow_counts") if isinstance(spec.published, Mapping) else None
    if isinstance(alternate, Mapping):
        for source_name, alternate_value in alternate.items():
            if alternate_value is None:
                continue
            value = int(alternate_value)
            alternate_flow_counts[str(source_name)] = {
                "published": value,
                "delta_measured_minus_published": rows - value,
                "match": rows == value,
            }

    class_table_sum = spec.published.get("class_table_sum") if isinstance(spec.published, Mapping) else None
    class_table_delta = None
    if class_table_sum is not None:
        class_table_delta = label_total - int(class_table_sum)

    unknown_classes = [
        name for name in comparison_classes if classes_expected and name not in classes_expected
    ]
    return_code = None

    # Checks.
    checks: list[dict[str, Any]] = []
    checks.append(
        _check(
            "csv_files_present",
            len(csv_files) == spec.expected_csv_count,
            measured=len(csv_files),
            expected=spec.expected_csv_count,
        )
    )
    checks.append(
        _check(
            "schema_columns",
            matches_spec,
            measured={"n_columns": len(header), "missing": missing_columns, "extra": extra_columns},
            expected={"n_columns": len(expected_columns)},
            detail="header compared to the 55-column NF-v3 registry contract",
        )
    )
    if published_flows is not None:
        checks.append(
            _check(
                "row_count_vs_published",
                rows_match,
                measured=rows,
                expected=published_flows,
                detail=f"delta = {row_delta}" if row_delta is not None else "",
            )
        )
    if taxonomy is not None:
        checks.append(
            _check(
                "label_taxonomy_validation",
                taxonomy_error is None,
                measured={
                    "fine_class_count": len(measured_canonical),
                    "taxonomy": taxonomy.name,
                    "aggregated_class_count": len(aggregated_classes),
                },
                expected={"aggregated_class_count": len(taxonomy.target_classes())},
                detail=taxonomy_error
                or "every raw label mapped and aggregated counts match the published table (D-031)",
            )
        )
        if taxonomy.published_aggregated_counts:
            checks.append(
                _check(
                    "aggregated_class_counts_vs_published",
                    not taxonomy_mismatches,
                    measured=aggregated_classes,
                    expected=dict(taxonomy.published_aggregated_counts),
                    detail=f"mismatches: {taxonomy_mismatches}",
                )
            )
    if aliases:
        checks.append(
            _check(
                "class_aliases_documented",
                True,
                measured={"n_aliases": len(aliases), "applied": alias_applied},
                expected={"n_aliases": len(aliases)},
                detail="raw CSV class spellings mapped to canonical paper names (never silent)",
            )
        )
    if published_classes:
        checks.append(
            _check(
                "label_counts_vs_published",
                not label_mismatches,
                measured=comparison_classes,
                expected=published_classes,
                detail=f"mismatches: {label_mismatches}",
            )
        )
    if classes_expected:
        checks.append(
            _check(
                "label_classes_expected",
                not unknown_classes,
                measured=sorted(comparison_classes),
                expected=classes_expected,
                detail=f"unknown classes: {unknown_classes}",
            )
        )
    published_classes_count = spec.published.get("classes_count") if isinstance(spec.published, Mapping) else None
    if published_classes_count is not None:
        checks.append(
            _check(
                "label_classes_count_vs_published",
                len(comparison_classes) == int(published_classes_count),
                measured=len(comparison_classes),
                expected=int(published_classes_count),
                detail=f"measured classes: {sorted(comparison_classes)}",
            )
        )
    if class_table_sum is not None:
        checks.append(
            _check(
                "label_total_vs_published_class_table",
                class_table_delta == 0,
                measured=label_total,
                expected=int(class_table_sum),
                severity="warn",
                detail=(
                    "published class table is internally inconsistent with the published total; "
                    f"delta = {class_table_delta} (reported, not forced)"
                ),
            )
        )
    if published_attack_total is not None and measured_attack_total is not None:
        checks.append(
            _check(
                "attack_total_vs_published",
                abs(int(measured_attack_total) - int(published_attack_total)) <= tolerance,
                measured=measured_attack_total,
                expected=int(published_attack_total),
                detail="sum of non-Benign classes vs paper Table 3 attack total",
            )
        )
    if isinstance(published_binary, Mapping):
        checks.append(
            _check(
                "binary_label_vs_published",
                not binary_mismatches,
                measured=measured_binary,
                expected={str(key): int(value) for key, value in published_binary.items()},
                detail=f"mismatches: {binary_mismatches}",
            )
        )
    checks.append(
        _check(
            "label_missing_values",
            label_missing == 0,
            measured=label_missing,
            expected=0,
            detail="every flow must carry a multiclass label",
        )
    )
    if resolved.get("check_binary_label_consistency", True):
        checks.append(
            _check(
                "binary_label_consistency",
                binary_violations == 0,
                measured=binary_violations,
                expected=0,
                detail="Label must be 0 for Benign and 1 for every attack class",
            )
        )
    parse_fraction = (ts_count / rows) if rows else 0.0
    checks.append(
        _check(
            "timestamp_parse_fraction",
            parse_fraction >= float(resolved.get("timestamp_parse_fraction_min", 1.0)),
            measured=round(parse_fraction, 6),
            expected=float(resolved.get("timestamp_parse_fraction_min", 1.0)),
            detail="fraction of rows with a parseable start timestamp",
        )
    )
    # A CSV can be truncated mid-row or contain embedded newlines; compare the
    # physical newline count against the number of parsed data rows.
    if line_count > 0:
        physical_rows = max(line_count - 1, 0)
        checks.append(
            _check(
                "line_count_vs_parsed_rows",
                physical_rows == rows,
                measured={"newline_count": line_count, "physical_rows": physical_rows, "parsed_rows": rows},
                expected=rows,
                severity="warn",
                detail="mismatch indicates truncated rows or embedded newlines (quoted fields)",
            )
        )
    checks.extend(
        _provenance_checks(spec, raw_path, {"rows": rows}, files, resolved, provenance)
    )

    status = "pass"
    if any(not check["passed"] and check["severity"] == "critical" for check in checks):
        status = "fail"
    elif any(not check["passed"] for check in checks):
        status = "warn"

    report: dict[str, Any] = {
        "schema_version": 1,
        "reference_version": REFERENCE_VERSION,
        "dataset": spec.name,
        "role": spec.role,
        "generated_at_utc": utc_now(),
        "raw_dir": str(raw_path),
        "options": resolved,
        "files": files,
        "schema": {
            "n_columns": len(header),
            "columns": header,
            "expected_n_columns": len(expected_columns),
            "missing_columns": missing_columns,
            "extra_columns": extra_columns,
            "matches_spec": matches_spec,
        },
        "rows": {
            "measured": rows,
            "published": published_flows,
            "delta": row_delta,
            "tolerance": tolerance,
            "match": rows_match,
            "line_count": line_count,
            "lines_minus_header": max(line_count - 1, 0),
            "published_count_source": (
                "paper arXiv:2503.04404v3 Tables 2-3 (D-013 authoritative)"
                if spec.schema == "nf_v3"
                else None
            ),
            "alternate_flow_counts": alternate_flow_counts,
        },
        "labels": {
            "column": spec.column("label_multiclass"),
            "measured": measured_classes,
            "measured_canonical": measured_canonical,
            "class_aliases": {"applied": alias_applied, "n_aliases": len(aliases)},
            "measured_fine": measured_classes,
            "measured_aggregated": aggregated_classes if taxonomy is not None else None,
            "comparison": comparison_classes,
            "taxonomy": (
                {
                    "name": taxonomy.name,
                    "source": taxonomy.source,
                    "fine_class_count_expected": taxonomy.fine_class_count,
                    "fine_class_count_measured": len(measured_classes),
                    "aggregated_class_count": len(aggregated_classes),
                    "published_aggregated_counts": dict(taxonomy.published_aggregated_counts),
                    "mismatches": taxonomy_mismatches,
                    "error": taxonomy_error,
                }
                if taxonomy is not None
                else None
            ),
            "total": label_total,
            "published_classes": published_classes,
            "published_class_table_sum": class_table_sum,
            "class_table_delta": class_table_delta,
            "mismatches": label_mismatches,
            "class_order_expected": classes_expected,
            "unknown_classes": unknown_classes,
            "missing": label_missing,
            "attack_total": measured_attack_total,
            "published_attack_total": None if published_attack_total is None else int(published_attack_total),
            "published_binary_label": (
                {str(key): int(value) for key, value in published_binary.items()}
                if isinstance(published_binary, Mapping)
                else None
            ),
            "binary_mismatches": binary_mismatches,
            "website_classes": (
                {str(key): int(value) for key, value in website_classes.items() if value is not None}
                if isinstance(website_classes, Mapping)
                else {}
            ),
            "website_mismatches": website_mismatches,
            "published_reference": dict(spec.published),
        },
        "binary_label": {
            "column": spec.optional_column("label_binary"),
            "measured": {name: int(count) for name, count in binary_counts.items()},
            "missing": binary_missing,
            "violations": binary_violations,
        },
        "missing": {
            "total": int(sum(missing.values())),
            "columns": {column: int(count) for column, count in sorted(missing.items())},
        },
        "unparseable_numeric": {
            "total": int(sum(unparseable.values())),
            "columns": {column: int(count) for column, count in sorted(unparseable.items())},
        },
        "inf": {
            "total": int(sum(inf.values())),
            "columns": {column: int(count) for column, count in sorted(inf.items())},
        },
        "duplicates": {
            "exact": int(duplicate_rows),
            "unique_rows": rows - int(duplicate_rows),
            "fraction": (duplicate_rows / rows) if rows else 0.0,
            "method": "full_row_hash",
        },
        "timestamps": {
            "column": states[0]["timestamp"]["column"],
            "parse_fraction": round(parse_fraction, 6),
            "min_utc": _iso_from_epoch_millis(ts_min),
            "max_utc": _iso_from_epoch_millis(ts_max),
            "non_monotonic_fraction": (ts_inversions / ts_count) if ts_count else 0.0,
            "monotonic": ts_inversions == 0,
        },
        "hosts": {
            "src_unique": len(hosts_src),
            "dst_unique": len(hosts_dst),
            "union_unique": len(hosts_src | hosts_dst),
            "method": "exact",
            "missing": {column: int(count) for column, count in sorted(host_missing.items())},
        },
        "parse_fallbacks": fallbacks,
        "checks": checks,
        "status": status,
    }
    report["summary"] = _summary(report)
    return report


def _summary(report: Mapping[str, Any]) -> dict[str, Any]:
    failed = [check["name"] for check in report["checks"] if not check["passed"]]
    return {
        "status": report["status"],
        "checks_total": len(report["checks"]),
        "checks_failed": failed,
        "rows_measured": report["rows"]["measured"],
        "rows_published": report["rows"]["published"],
        "n_classes": len(report["labels"]["measured"]),
        "duplicates": report["duplicates"]["exact"],
        "hosts_union": report["hosts"]["union_unique"],
    }


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------
def _format_number(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.6g}"
    if value is None:
        return "n/a"
    if isinstance(value, (list, dict)):
        text = json.dumps(value, sort_keys=True, default=str)
        return text if len(text) <= 120 else text[:117] + "..."
    return str(value)


_STATUS_ICON = {"pass": "PASS", "warn": "WARN", "fail": "FAIL"}


def identity_markdown(report: Mapping[str, Any]) -> str:
    """Render one identity report as a human-readable markdown page."""
    status = str(report["status"]).upper()
    lines = [
        f"# Dataset identity: {report['dataset']}",
        "",
        f"- **Status**: {_STATUS_ICON.get(report['status'], status)}",
        f"- **Role**: {report['role']}",
        f"- **Generated (UTC)**: {report['generated_at_utc']}",
        f"- **Raw directory**: `{report['raw_dir']}`",
        "",
        "## Files",
        "",
        "| File | Bytes | SHA-256 |",
        "| --- | ---: | --- |",
    ]
    for record in report["files"]:
        digest = record.get("sha256") or "n/a"
        lines.append(
            f"| `{record['relative_path']}` | {_format_number(record['bytes'])} | `{digest}` |"
        )
    lines += [
        "",
        "## Counts",
        "",
        "| Quantity | Measured | Published | Match |",
        "| --- | ---: | ---: | --- |",
        f"| Rows | {_format_number(report['rows']['measured'])} | "
        f"{_format_number(report['rows']['published'])} | {_format_number(report['rows']['match'])} |",
        f"| Label total | {_format_number(report['labels']['total'])} | "
        f"{_format_number(report['labels']['published_class_table_sum'])} | "
        f"{_format_number(report['labels']['class_table_delta'] in (None, 0) if report['labels']['class_table_delta'] is not None else None)} |",
        f"| Columns | {_format_number(report['schema']['n_columns'])} | "
        f"{_format_number(report['schema']['expected_n_columns'])} | {_format_number(report['schema']['matches_spec'])} |",
        f"| Classes | {_format_number(len(report['labels']['measured']))} | "
        f"{_format_number(len(report['labels']['published_classes']))} | - |",
        f"| Exact duplicate rows | {_format_number(report['duplicates']['exact'])} | 0 | - |",
        f"| NaN cells | {_format_number(report['missing']['total'])} | 0 | - |",
        f"| Infinity cells | {_format_number(report['inf']['total'])} | 0 | - |",
        f"| Hosts (union) | {_format_number(report['hosts']['union_unique'])} | - | - |",
        f"| Timestamp parse fraction | {_format_number(report['timestamps']['parse_fraction'])} | 1.0 | - |",
        f"| Timestamp monotonic | {_format_number(report['timestamps']['monotonic'])} | - | - |",
        "",
    ]
    if report["timestamps"]["min_utc"]:
        lines += [
            f"Timestamp range: **{report['timestamps']['min_utc']}** to **{report['timestamps']['max_utc']}**; "
            f"non-monotonic fraction {_format_number(report['timestamps']['non_monotonic_fraction'])}.",
            "",
        ]

    lines += ["## Label distribution", "", "| Class | Measured | Published |", "| --- | ---: | ---: |"]
    label_distribution = (
        report["labels"].get("measured_aggregated")
        or report["labels"].get("measured_canonical")
        or report["labels"]["measured"]
    )
    for class_name, count in sorted(label_distribution.items(), key=lambda item: -item[1]):
        published = report["labels"]["published_classes"].get(class_name)
        lines.append(f"| {class_name} | {_format_number(count)} | {_format_number(published)} |")
    lines.append("")
    taxonomy = report["labels"].get("taxonomy")
    if taxonomy:
        lines += [
            f"Documented label taxonomy **{taxonomy.get('name')}**: "
            f"{taxonomy.get('fine_class_count_measured')} fine labels aggregated to "
            f"{taxonomy.get('aggregated_class_count')} classes "
            f"(validation: {'PASS' if taxonomy.get('error') is None else 'FAIL: ' + str(taxonomy.get('error'))}).",
            "",
        ]
    if report["labels"]["mismatches"]:
        lines += ["**Paper-reference mismatches:**", ""]
        for item in report["labels"]["mismatches"]:
            lines.append(
                f"- `{item['class']}`: measured {_format_number(item['measured'])}, "
                f"published {_format_number(item['published'])}"
            )
        lines.append("")

    reference_conflicts = []
    if report["labels"].get("website_mismatches"):
        reference_conflicts.append(
            "UQ website values differ from the measured CSV (D-013: website is not authoritative): "
            + "; ".join(
                f"{item['class']} website {_format_number(item['website'])} vs measured {_format_number(item['measured'])}"
                for item in report["labels"]["website_mismatches"]
            )
        )
    for source_name, info in report["rows"].get("alternate_flow_counts", {}).items():
        delta = info["delta_measured_minus_published"]
        delta_text = f"{delta:+,}" if isinstance(delta, int) else str(delta)
        reference_conflicts.append(
            f"{source_name} reports {_format_number(info['published'])} rows; measured delta {delta_text}"
        )
    if report["labels"].get("class_table_delta") not in (None, 0):
        delta = report["labels"]["class_table_delta"]
        delta_text = f"{delta:+,}" if isinstance(delta, int) else str(delta)
        reference_conflicts.append(
            f"scout class-table sum {_format_number(report['labels']['published_class_table_sum'])}; "
            f"measured delta {delta_text}"
        )
    if report["labels"].get("published_attack_total") is not None:
        reference_conflicts.append(
            f"attack total measured {_format_number(report['labels']['attack_total'])} vs paper "
            f"{_format_number(report['labels']['published_attack_total'])}"
        )
    if reference_conflicts:
        lines += ["## Reference conflicts (reported, not enforced)", ""]
        lines += [f"- {item}" for item in reference_conflicts]
        lines.append("")

    lines += ["## Checks", "", "| Check | Severity | Result | Measured | Expected |", "| --- | --- | --- | --- | --- |"]
    for check in report["checks"]:
        result = "PASS" if check["passed"] else "FAIL"
        lines.append(
            f"| {check['name']} | {check['severity']} | **{result}** | "
            f"{_format_number(check['measured'])} | {_format_number(check['expected'])} |"
        )
    lines.append("")
    if report.get("parse_fallbacks"):
        lines += ["## Parse fallbacks", "", "```json", json.dumps(report["parse_fallbacks"], indent=2), "```", ""]
    lines += [
        "## Reproduction",
        "",
        "```",
        "python src/scripts/verify_datasets.py --dataset "
        f"{report['dataset']} --config configs/audit.yaml",
        "```",
        "",
        "_Raw files are never modified: this report is produced from read-only streaming passes._",
        "",
    ]
    return "\n".join(lines)


def write_identity_report(report: Mapping[str, Any], out_dir: str | Path) -> dict[str, str]:
    """Write ``identity_<dataset>.json`` and ``identity_<dataset>.md``."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stem = f"identity_{report['dataset']}"
    json_path = directory / f"{stem}.json"
    md_path = directory / f"{stem}.md"
    write_json(json_path, dict(report))
    md_path.write_text(identity_markdown(report), encoding="utf-8")
    return {"json": str(json_path), "md": str(md_path)}


def summarize_identity_reports(reports: Sequence[Mapping[str, Any]]) -> str:
    """Combined markdown summary across datasets (identity_summary.md)."""
    lines = [
        "# Dataset identity summary",
        "",
        f"Generated (UTC): {utc_now()}",
        "",
        "| Dataset | Rows measured | Rows published | Match | Classes | NaN | inf | Dup rows | Hosts | Status |",
        "| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for report in reports:
        rows = report["rows"]
        lines.append(
            f"| {report['dataset']} | {_format_number(rows['measured'])} | "
            f"{_format_number(rows['published'])} | {_format_number(rows['match'])} | "
            f"{len(report['labels']['measured'])} | {_format_number(report['missing']['total'])} | "
            f"{_format_number(report['inf']['total'])} | {_format_number(report['duplicates']['exact'])} | "
            f"{_format_number(report['hosts']['union_unique'])} | "
            f"**{_STATUS_ICON.get(report['status'], report['status'].upper())}** |"
        )
    lines += ["", "## Failed checks", ""]
    any_failed = False
    for report in reports:
        failed = [check for check in report["checks"] if not check["passed"]]
        if failed:
            any_failed = True
            lines.append(f"### {report['dataset']}")
            for check in failed:
                lines.append(
                    f"- `{check['name']}` ({check['severity']}): measured {_format_number(check['measured'])}, "
                    f"expected {_format_number(check['expected'])}"
                    + (f" - {check['detail']}" if check.get("detail") else "")
                )
            lines.append("")
    if not any_failed:
        lines.append("None - every dataset passed every identity check.")
        lines.append("")
    lines += [
        "## Notes",
        "",
        "- Row/label counts are direct measurements from the raw CSV (read-only streaming pass).",
        "- Published counts are the UQ NetFlow-v3 values pinned in `configs/datasets/<dataset>.yaml`; "
        "a mismatch is reported with the exact delta and is never forced.",
        "- Per-dataset JSON + markdown reports live next to this file as `identity_<dataset>.{json,md}`.",
        "",
    ]
    return "\n".join(lines)


__all__ = [
    "DEFAULT_OPTIONS",
    "count_newlines",
    "identity_markdown",
    "read_header",
    "resolve_options",
    "scan_csv",
    "summarize_identity_reports",
    "verify_dataset",
    "write_identity_report",
]
