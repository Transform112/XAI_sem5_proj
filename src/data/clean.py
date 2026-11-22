"""Raw loading and cleaning for the S-D1 stage.

Steps (ordered and idempotent):
1. load the raw CSV with explicit dtypes (numeric -> float64, hosts/labels ->
   objects, timestamps -> float64 ms) so parsing is deterministic;
2. drop exact duplicate rows (full physical-row hash, keep first);
3. apply the sentinel / NaN / inf policy: negative values in columns declared
   ``non_negative`` become NaN (NetFlow "not available"), ``+/-inf`` becomes NaN;
4. sort chronologically (mergesort, stable) and assign ``_row_id``;
5. build the global host-id map over the whole dataset (sorted, deterministic)
   and the global class map (``class_order`` from the registry when pinned);
6. assign near-duplicate leakage groups: canonical key = source/destination
   host ids + source/destination port + protocol + start timestamp binned to
   ``near_duplicate.time_bucket_ms`` (hashed to an int64 group id).

The raw file is never modified; the returned frame adds private columns
prefixed with ``_`` so downstream code cannot confuse them with raw features.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.data.registry import DatasetSpec
from src.data.taxonomy import LabelTaxonomy, TaxonomyError
from src.utils.logging import get_logger

LOGGER = get_logger("xdrift.clean")

DEFAULT_SENTINEL_POLICY: dict[str, Any] = {
    "negative_to_nan": True,
    "inf_to_nan": True,
    "explicit_values": [],
}

DEFAULT_NEAR_DUPLICATE: dict[str, Any] = {
    "time_bucket_ms": 1000,
    "keys": ["host_src", "host_dst", "port_src", "port_dst", "protocol"],
}


@dataclass
class CleanResult:
    """Cleaned frame plus the global maps and audit statistics."""

    frame: pd.DataFrame
    stats: dict[str, Any]
    host_map: dict[str, int] = field(repr=False)
    class_map: dict[str, int] = field(repr=False)

    @property
    def n_rows(self) -> int:
        return len(self.frame)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def read_dtype_map(spec: DatasetSpec, columns: list[str]) -> dict[str, Any]:
    """Explicit read dtypes: float64 numeric, object text, float64 timestamps."""
    numeric = set(spec.log1p) | set(spec.numeric)
    timestamps = {column for column in (spec.optional_column("timestamp_start"), spec.optional_column("timestamp_end")) if column}
    text = {
        spec.column("host_src"),
        spec.column("host_dst"),
        spec.column("label_multiclass"),
    }
    binary = spec.optional_column("label_binary")
    if binary:
        text.add(binary)
    dtype: dict[str, Any] = {}
    for column in columns:
        if column in text:
            dtype[column] = "object"
        else:  # numeric features and timestamps (and anything unknown -> number)
            dtype[column] = "float64"
    del numeric, timestamps
    return dtype


def load_raw_frame(spec: DatasetSpec, csv_path: str | Path) -> pd.DataFrame:
    """Load one raw NF-v3 CSV with fixed dtypes and no row filtering."""
    path = Path(csv_path)
    if not path.is_file():
        raise FileNotFoundError(f"raw CSV not found: {path}")
    header = pd.read_csv(path, nrows=0).columns.tolist()
    dtype = read_dtype_map(spec, header)
    LOGGER.info("loading %s (%s columns)", path.name, len(header))
    frame = pd.read_csv(
        path,
        dtype=dtype,
        low_memory=False,
        keep_default_na=True,
        na_values=["", "NA", "N/A", "nan", "NaN", "null", "NULL", "?"],
    )
    if frame.empty:
        raise ValueError(f"raw CSV is empty: {path}")
    return frame


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalise_for_hash(frame: pd.DataFrame) -> pd.DataFrame:
    normalised = frame.copy()
    for column in normalised.columns:
        if normalised[column].dtype == object:
            normalised[column] = normalised[column].astype("string").fillna("<missing>")
    return normalised


def _hash_rows(frame: pd.DataFrame, columns: list[str]) -> np.ndarray:
    """Stable uint64 hash per row over the selected columns."""
    return pd.util.hash_pandas_object(
        _normalise_for_hash(frame[columns]), index=False
    ).to_numpy(dtype="uint64")


def _resolve_key_columns(spec: DatasetSpec, keys: list[str]) -> list[str]:
    """Map canonical-key names (roles or raw column names) to physical columns."""
    columns: list[str] = []
    for key in keys:
        try:
            columns.append(spec.column(str(key)))
        except Exception:  # noqa: BLE001 - fall back to a literal column name
            available = set(spec.all_columns())
            if str(key) not in available:
                raise ValueError(
                    f"near-duplicate key {key!r} is neither a column role nor a known column"
                ) from None
            columns.append(str(key))
    return columns


def _iso_from_millis(value: float) -> str | None:
    try:
        return datetime.fromtimestamp(float(value) / 1000.0, tz=timezone.utc).isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):
        return None


def _label_conflict_examples(
    frame: pd.DataFrame,
    spec: DatasetSpec,
    conflicted_group_ids: np.ndarray,
    *,
    limit: int = 8,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Capture concrete examples of conflicting near-duplicate groups (D-032).

    Each example lists the canonical-key fields, the raw multiclass/binary labels
    and the first few rows so the decision whether this is genuine within-key
    label ambiguity (vs a canonical-key bug) is auditable.
    """
    if len(conflicted_group_ids) == 0:
        return []
    rng = np.random.default_rng(seed)
    if len(conflicted_group_ids) > limit:
        picked = np.sort(rng.choice(len(conflicted_group_ids), size=limit, replace=False))
        group_ids = [int(conflicted_group_ids[index]) for index in picked]
    else:
        group_ids = [int(value) for value in conflicted_group_ids]
    key_columns = [
        spec.column("host_src"),
        spec.column("host_dst"),
        spec.column("port_src"),
        spec.column("port_dst"),
        spec.optional_column("protocol"),
    ]
    key_columns = [column for column in key_columns if column]
    label_column = spec.column("label_multiclass")
    binary_column = spec.optional_column("label_binary")
    examples: list[dict[str, Any]] = []
    for group_id in group_ids:
        rows = frame[frame["_group_id"] == group_id]
        label_values = [str(value) for value in rows[label_column].astype("string").unique() if pd.notna(value)]
        example: dict[str, Any] = {
            "group_id": int(group_id),
            "n_rows": int(len(rows)),
            "labels": sorted(label_values),
            "key": {column: _jsonable(rows[column].iloc[0]) for column in key_columns},
            "rows": [],
        }
        if "TS_COLUMN" not in example:  # documentation anchor, no-op
            pass
        for _, row in rows.head(6).iterrows():
            record = {
                "row_id": int(row["_row_id"]),
                "timestamp_utc": _iso_from_millis(float(row["_ts"])),
                "class_id": int(row["_class_id"]),
                "label": str(row[label_column]),
            }
            if binary_column and binary_column in rows.columns:
                record["binary_label"] = _jsonable(row[binary_column])
            example["rows"].append(record)
        examples.append(example)
    return examples


def _jsonable(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
def clean_frame(
    raw: pd.DataFrame,
    spec: DatasetSpec,
    *,
    sentinel_policy: Mapping[str, Any] | None = None,
    near_duplicate: Mapping[str, Any] | None = None,
    dedupe: bool = True,
    label_conflict_policy: str = "report_only",
) -> CleanResult:
    """Apply the ordered cleaning pipeline and return the cleaned frame.

    ``label_conflict_policy`` (D-032):

    * ``report_only`` (default): measure near-duplicate groups whose rows carry
      different class labels and keep them (the caller decides what to do);
    * ``drop_groups``: drop ALL rows of label-conflicting near-duplicate groups
      before period/split assignment. This is the documented cleaning rule: it
      never relabels a flow and the before/after row + per-class counts are
      recorded in ``stats["label_conflict"]``.
    """
    if label_conflict_policy not in ("report_only", "drop_groups"):
        raise ValueError(
            f"unknown label_conflict_policy {label_conflict_policy!r}; "
            "expected 'report_only' or 'drop_groups'"
        )
    sentinel = {**DEFAULT_SENTINEL_POLICY, **dict(sentinel_policy or {})}
    group_config = {**DEFAULT_NEAR_DUPLICATE, **dict(near_duplicate or {})}

    missing_columns = set(spec.all_columns()) - set(raw.columns)
    if missing_columns:
        raise ValueError(f"raw frame is missing columns required by the spec: {sorted(missing_columns)}")

    stats: dict[str, Any] = {
        "rows_in": int(len(raw)),
        "exact_duplicates_removed": 0,
        "rows_after_dedupe": int(len(raw)),
        "negative_to_nan": {},
        "inf_to_nan": {},
        "explicit_sentinel_to_nan": {},
        "time_sorted": False,
        "non_monotonic_fraction_raw": 0.0,
        "hosts": 0,
        "missing_host_values": {},
        "classes_seen": [],
        "binary_label_violations": 0,
        "near_duplicate_groups": 0,
        "near_duplicate_rows_in_multi_row_groups": 0,
        "near_duplicate_max_group_size": 0,
    }

    timestamp_column = spec.column("timestamp_start")
    parsed_ts = pd.to_numeric(raw[timestamp_column], errors="coerce")
    if parsed_ts.isna().any():
        raise ValueError(
            f"{int(parsed_ts.isna().sum())} rows have an unparseable {timestamp_column}; "
            "run the identity stage first"
        )

    # 0) documented label taxonomy (D-031): aggregate fine labels on the RAW rows
    #    (published totals are raw totals), fail loudly on any mismatch/unknown
    #    label, then relabel the deduped frame explicitly.
    taxonomy = LabelTaxonomy.from_spec(spec)
    label_column = spec.column("label_multiclass")
    taxonomy_stats: dict[str, Any] | None = None
    if taxonomy is not None:
        raw_labels = raw[label_column].astype("string")
        fine_counts = {
            str(key): int(value)
            for key, value in raw_labels.value_counts(dropna=False).items()
            if pd.notna(key)
        }
        missing_labels = int(raw_labels.isna().sum())
        if missing_labels:
            raise TaxonomyError(
                f"{missing_labels} raw rows have a missing {label_column}; taxonomy validation requires labels"
            )
        aggregated = taxonomy.aggregate_counts(fine_counts)
        taxonomy_stats = {
            "name": taxonomy.name,
            "source": taxonomy.source,
            "fine_class_count": len(fine_counts),
            "fine_label_counts": dict(sorted(fine_counts.items())),
            "aggregated_class_counts": dict(sorted(aggregated.items())),
            "published_aggregated_counts": dict(sorted(taxonomy.published_aggregated_counts.items())),
            "strict_aggregate_check": taxonomy.strict_aggregate_check,
            "validation": "passed",
        }
        LOGGER.info(
            "label taxonomy %s applied: %s fine labels -> %s aggregated classes (published counts match)",
            taxonomy.name,
            len(fine_counts),
            len(aggregated),
        )
    # 1) exact duplicates ---------------------------------------------------
    frame = raw
    if dedupe:
        hashes = _hash_rows(frame, list(frame.columns))
        duplicated = pd.Series(hashes).duplicated(keep="first").to_numpy()
        keep = ~duplicated
        stats["exact_duplicates_removed"] = int((~keep).sum())
        frame = frame.loc[keep]
        # Verify the dedupe actually removed every duplicate (D-023: the 0.1%
        # threshold applies to duplicates remaining after dedupe).
        remaining_hashes = hashes[keep]
        stats["exact_duplicates_remaining"] = int(
            len(remaining_hashes) - np.unique(remaining_hashes).size
        )
    else:
        stats["exact_duplicates_remaining"] = None
    frame = frame.copy()
    stats["rows_after_dedupe"] = int(len(frame))

    # 2) sentinel / NaN / inf policy ---------------------------------------
    numeric_columns = [column for column in (*spec.log1p, *spec.numeric) if column in frame.columns]
    non_negative = [column for column in spec.non_negative if column in numeric_columns]
    if sentinel.get("negative_to_nan", True):
        for column in non_negative:
            values = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = values
            negative = values < 0
            count = int(negative.sum())
            if count:
                stats["negative_to_nan"][column] = count
                frame.loc[negative, column] = np.nan
    if sentinel.get("inf_to_nan", True):
        for column in numeric_columns:
            values = pd.to_numeric(frame[column], errors="coerce")
            frame[column] = values
            infinite = np.isinf(values.to_numpy(dtype="float64", na_value=np.nan))
            count = int(infinite.sum())
            if count:
                stats["inf_to_nan"][column] = count
                frame.loc[infinite, column] = np.nan
    explicit = sentinel.get("explicit_values") or []
    if explicit:
        for column in numeric_columns:
            marker = pd.to_numeric(frame[column], errors="coerce")
            replacement = marker.isin([float(v) for v in explicit])
            count = int(replacement.sum())
            if count:
                stats["explicit_sentinel_to_nan"][column] = count
                frame.loc[replacement, column] = np.nan

    # 3) chronological sort -------------------------------------------------
    raw_ts = pd.to_numeric(frame[timestamp_column], errors="coerce").to_numpy(dtype="float64")
    if raw_ts.size > 1:
        stats["non_monotonic_fraction_raw"] = float((np.diff(raw_ts) < 0).sum() / (raw_ts.size - 1))
    order = np.argsort(raw_ts, kind="mergesort")
    frame = frame.iloc[order].reset_index(drop=True)
    frame["_ts"] = pd.to_numeric(frame[timestamp_column], errors="coerce").astype("int64")
    frame["_row_id"] = np.arange(len(frame), dtype="int64")
    stats["time_sorted"] = bool(frame["_ts"].is_monotonic_increasing)
    stats["non_monotonic_fraction"] = 0.0

    # 4) global host map ----------------------------------------------------
    src_column = spec.column("host_src")
    dst_column = spec.column("host_dst")
    host_values = pd.concat([frame[src_column], frame[dst_column]], ignore_index=True)
    for column in (src_column, dst_column):
        missing = int(frame[column].isna().sum())
        if missing:
            stats["missing_host_values"][column] = missing
            frame[column] = frame[column].fillna("<missing>")
    hosts = sorted({str(value) for value in host_values.dropna().tolist()} | {"<missing>" if stats["missing_host_values"] else ""} - {""})
    host_map = {host: index for index, host in enumerate(hosts)}
    frame["_host_src_id"] = frame[src_column].map(host_map).astype("int64")
    frame["_host_dst_id"] = frame[dst_column].map(host_map).astype("int64")
    stats["hosts"] = len(host_map)

    # 5) global class map ---------------------------------------------------
    classes = frame[label_column].astype("string")
    if taxonomy is not None:
        classes = taxonomy.apply_series(classes)
        frame[label_column] = classes
        stats["label_taxonomy"] = taxonomy_stats
    # Documented class-name aliases (paper name <-> raw CSV spelling), e.g. ToN
    # lower-case labels. Never silent: unmapped names still raise below.
    alias_map = spec.canonical_class_map()  # raw CSV spelling -> canonical paper name
    if alias_map:
        raw_classes = classes.copy()
        mapped = classes.map(alias_map)
        classes = mapped.where(mapped.notna(), classes)
        frame[label_column] = classes
        applied = {
            canonical: raw_name
            for raw_name, canonical in alias_map.items()
            if bool((raw_classes == raw_name).any())
        }
        stats["class_aliases"] = {
            "n_aliases": len(alias_map),
            "applied": applied,
            "note": "raw CSV class values mapped to canonical paper names (D-020/D-031)",
        }
    if classes.isna().any():
        raise ValueError(f"{int(classes.isna().sum())} rows have a missing {label_column}")
    unique_classes = sorted({str(value) for value in classes.unique().tolist()})
    if spec.class_order:
        unknown = [name for name in unique_classes if name not in spec.class_order]
        if unknown:
            raise ValueError(
                f"dataset {spec.name!r} contains classes outside class_order: {unknown}"
            )
        class_names = list(spec.class_order)
        # Only classes present in the data get an id; the mapping order is frozen.
        present = [name for name in class_names if name in set(unique_classes)]
    else:
        present = unique_classes
    class_map = {name: index for index, name in enumerate(present)}
    frame["_class_id"] = classes.map(class_map)
    if frame["_class_id"].isna().any():  # pragma: no cover - defensive
        raise ValueError("class mapping produced missing class ids")
    frame["_class_id"] = frame["_class_id"].astype("int64")
    stats["classes_seen"] = present

    binary_column = spec.optional_column("label_binary")
    benign_class = "Benign" if "Benign" in class_map else present[0]
    derived_binary = (frame["_class_id"] != class_map[benign_class]).astype("int8")
    if binary_column and binary_column in frame.columns:
        raw_binary = pd.to_numeric(frame[binary_column], errors="coerce").fillna(-1).astype("int64")
        stats["binary_label_violations"] = int((raw_binary != derived_binary.astype("int64")).sum())
    frame["_label_binary"] = derived_binary

    # 6) near-duplicate leakage groups --------------------------------------
    # Key columns are resolved through the registry roles. HOST keys must use the
    # global integer host ids (`_host_src_id`/`_host_dst_id`), never the raw IP
    # strings: coercing IP strings to numeric would collapse every host to a
    # sentinel and make the canonical key far coarser than configured (D-034).
    group_key_names = [str(key) for key in group_config["keys"]]
    key_columns = _resolve_key_columns(spec, group_key_names)
    key_arrays: dict[str, np.ndarray] = {}
    for position, (key_name, column) in enumerate(zip(group_key_names, key_columns)):
        if key_name == "host_src":
            values = frame["_host_src_id"].to_numpy(dtype="int64")
        elif key_name == "host_dst":
            values = frame["_host_dst_id"].to_numpy(dtype="int64")
        else:
            values = (
                pd.to_numeric(frame[column], errors="coerce")
                .fillna(-1)
                .astype("int64")
                .to_numpy()
            )
        key_arrays[f"k_{position}"] = values
    bucket_ms = int(group_config["time_bucket_ms"])
    key_frame = pd.DataFrame(
        {
            **key_arrays,
            "ts_bucket": (frame["_ts"].to_numpy() // max(bucket_ms, 1)).astype("int64"),
        },
        index=frame.index,
    )
    key_hashes = pd.util.hash_pandas_object(key_frame, index=False).to_numpy(dtype="uint64")
    _, group_ids = np.unique(key_hashes, return_inverse=True)
    frame["_group_id"] = group_ids.astype("int64")
    sizes = pd.Series(group_ids).value_counts()
    stats["near_duplicate_groups"] = int(len(sizes))
    stats["near_duplicate_rows_in_multi_row_groups"] = int(sizes[sizes > 1].sum())
    stats["near_duplicate_max_group_size"] = int(sizes.max()) if len(sizes) else 0
    stats["near_duplicate_key_columns"] = key_columns
    stats["near_duplicate_key_roles"] = group_key_names
    stats["near_duplicate_time_bucket_ms"] = bucket_ms

    # 7) documented label-conflict handling (D-032) --------------------------
    conflict_stats: dict[str, Any] = {
        "policy": label_conflict_policy,
        "key": "near_duplicate_group",
        "n_conflicted_groups": 0,
        "rows_dropped": 0,
    }
    label_collisions = frame[["_group_id", "_class_id"]].drop_duplicates()
    group_class_counts = label_collisions.groupby("_group_id").size()
    conflicted_group_ids = group_class_counts[group_class_counts > 1].index.to_numpy()
    conflict_stats["n_conflicted_groups"] = int(len(conflicted_group_ids))
    if len(conflicted_group_ids):
        conflict_stats["examples"] = _label_conflict_examples(
            frame, spec, conflicted_group_ids, limit=8, seed=int(getattr(spec, "seed", 42) or 42)
        )
        conflicted_mask = frame["_group_id"].isin(conflicted_group_ids).to_numpy()
        conflict_stats["before"] = {
            "rows": int(len(frame)),
            "class_counts": {
                str(int(key)): int(value)
                for key, value in frame["_class_id"].value_counts().sort_index().items()
            },
            "label_counts": {
                str(key): int(value)
                for key, value in frame[label_column].astype("string").value_counts().items()
            },
        }
        conflict_stats["rows_in_conflicted_groups"] = int(conflicted_mask.sum())
        if label_conflict_policy == "drop_groups":
            frame = frame.loc[~conflicted_mask].reset_index(drop=True)
            frame["_row_id"] = np.arange(len(frame), dtype="int64")
            conflict_stats["rows_dropped"] = int(conflicted_mask.sum())
            LOGGER.warning(
                "D-032: dropped %s rows from %s label-conflicting near-duplicate groups",
                f"{conflicted_mask.sum():,}",
                f"{len(conflicted_group_ids):,}",
            )
        conflict_stats["after"] = {
            "rows": int(len(frame)),
            "class_counts": {
                str(int(key)): int(value)
                for key, value in frame["_class_id"].value_counts().sort_index().items()
            },
            "label_counts": {
                str(key): int(value)
                for key, value in frame[label_column].astype("string").value_counts().items()
            },
        }
    stats["label_conflict"] = conflict_stats

    # Row-level sanity recorded for the scorecard.
    stats["rows_after_clean"] = int(len(frame))
    stats["time_range_utc"] = {
        "min": _iso_from_millis(float(frame["_ts"].min())),
        "max": _iso_from_millis(float(frame["_ts"].max())),
    }
    LOGGER.info(
        "cleaned rows %s -> %s (removed %s duplicates); hosts=%s classes=%s groups=%s",
        f"{stats['rows_in']:,}",
        f"{stats['rows_after_clean']:,}",
        f"{stats['exact_duplicates_removed']:,}",
        stats["hosts"],
        len(stats["classes_seen"]),
        stats["near_duplicate_groups"],
    )
    return CleanResult(frame=frame, stats=stats, host_map=host_map, class_map=class_map)


__all__ = ["CleanResult", "DEFAULT_NEAR_DUPLICATE", "DEFAULT_SENTINEL_POLICY", "clean_frame", "load_raw_frame", "read_dtype_map"]
