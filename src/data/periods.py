"""Temporal period assignment (Doc 2 S2).

The empirical stream is five consecutive chronological periods. Two rules are
supported and the chosen one is recorded in the period statistics:

* ``equal_counts`` (default): sort by start timestamp and cut into periods of
  equal flow count. Chosen because the NF-v3 captures have very uneven traffic
  volume over time; equal time windows would give periods spanning ~35 orders of
  magnitude in size.
* ``equal_time``: cut the observed [min, max] timestamp range into equal spans.

The input frame must already be chronologically sorted (``clean_frame``
guarantees this; the function verifies it rather than silently re-sorting, so a
sorting bug cannot hide).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

LOGGER = get_logger("xdrift.periods")

VALID_STRATEGIES = ("equal_counts", "equal_time", "session_aware_1_4")

#: Default minimum gap used to detect a capture-session boundary (seconds).
DEFAULT_SESSION_MIN_GAP_SECONDS = 3600.0


@dataclass
class PeriodResult:
    """Frame with a ``_period`` column plus the stream statistics."""

    frame: pd.DataFrame
    stats: dict[str, Any]


def _iso(millis: float) -> str | None:
    try:
        return datetime.fromtimestamp(millis / 1000.0, tz=timezone.utc).isoformat(timespec="seconds")
    except (OverflowError, OSError, ValueError):
        return None


def _detect_session_boundary(ts: np.ndarray, min_gap_seconds: float) -> tuple[int, int]:
    """Return (split_row_index, gap_ms) at the largest gap in the timestamp stream.

    Deterministic from the raw timestamps alone: the D-026 C1 stream is defined by
    the single 26-day capture-session gap in NF-UNSW-NB15-v3, so the boundary is
    found rather than hard-coded.
    """
    if ts.size < 2:
        raise ValueError("need at least two rows to detect a session boundary")
    differences = np.diff(ts.astype("int64"))
    index = int(np.argmax(differences))
    gap_ms = int(differences[index])
    if gap_ms < int(min_gap_seconds * 1000.0):
        raise ValueError(
            f"no session gap >= {min_gap_seconds} s detected (largest gap {gap_ms / 1000.0:.3f} s); "
            "use strategy='equal_counts' for a single-session capture"
        )
    return index + 1, gap_ms


def _split_equal_counts(n_rows: int, n_periods: int) -> np.ndarray:
    edges = (np.arange(n_periods + 1, dtype="int64") * n_rows) // n_periods
    edges[-1] = n_rows
    return edges


def _session_aware_boundaries(
    ts: np.ndarray, n_periods: int, min_gap_seconds: float
) -> tuple[np.ndarray, dict[str, Any]]:
    """D-026 C1 stream: period 1 = first capture session, periods 2..T = equal
    counts inside the second session."""
    split_row, gap_ms = _detect_session_boundary(ts, min_gap_seconds)
    session1 = split_row
    session2 = int(ts.size) - session1
    if n_periods < 2:
        raise ValueError("session_aware_1_4 requires n_periods >= 2")
    if session2 < (n_periods - 1):
        raise ValueError(
            f"second session has {session2} rows, cannot form {n_periods - 1} periods"
        )
    period = np.zeros(ts.size, dtype="int64")
    remainder_edges = _split_equal_counts(session2, n_periods - 1)
    for index in range(n_periods - 1):
        start = session1 + int(remainder_edges[index])
        stop = session1 + int(remainder_edges[index + 1])
        period[start:stop] = index + 1
    boundaries = {
        "row_boundaries": [0, session1]
        + [session1 + int(value) for value in remainder_edges[1:].tolist()],
        "session_boundary_row": int(session1),
        "session_gap_seconds": round(gap_ms / 1000.0, 3),
        "session_boundary_utc": _iso(float(ts[session1])),
        "session_row_counts": [int(session1), int(session2)],
        "rule": (
            "largest timestamp gap >= min_session_gap_seconds defines the session boundary; "
            "period 1 = first session, periods 2..T = equal flow counts inside the second session"
        ),
    }
    return period, boundaries


def _period_boundaries(ts: np.ndarray, n_periods: int, strategy: str, min_session_gap_seconds: float = DEFAULT_SESSION_MIN_GAP_SECONDS) -> tuple[np.ndarray, dict[str, Any]]:
    """Return a 0-based period index per row and boundary metadata."""
    n = ts.size
    if strategy == "equal_counts":
        edges = (np.arange(n_periods + 1, dtype="int64") * n) // n_periods
        edges[-1] = n
        period = np.zeros(n, dtype="int64")
        for index in range(n_periods):
            period[edges[index] : edges[index + 1]] = index
        boundaries = {
            "row_boundaries": edges.tolist(),
            "time_boundaries_utc": [_iso(float(ts[min(edge, n - 1)])) for edge in edges],
        }
        return period, boundaries
    if strategy == "session_aware_1_4":
        return _session_aware_boundaries(ts, n_periods, min_session_gap_seconds)
    if strategy == "equal_time":
        minimum = float(ts.min())
        maximum = float(ts.max())
        span = max(maximum - minimum, 1.0)
        thresholds = minimum + span * np.arange(1, n_periods, dtype="float64") / n_periods
        period = np.searchsorted(thresholds, ts, side="right").astype("int64")
        empty = [index for index in range(n_periods) if not np.any(period == index)]
        if empty:
            raise ValueError(
                f"equal_time periods {empty} are empty; use strategy='equal_counts' "
                "or increase data volume"
            )
        boundaries = {
            "time_thresholds_utc": [_iso(float(value)) for value in thresholds],
            "row_boundaries": [int(np.searchsorted(period, index)) for index in range(n_periods + 1)],
        }
        return period, boundaries
    raise ValueError(f"unknown period strategy {strategy!r}; expected one of {VALID_STRATEGIES}")


def assign_periods(
    frame: pd.DataFrame,
    *,
    n_periods: int = 5,
    strategy: str = "equal_counts",
    ts_col: str = "_ts",
    period_col: str = "_period",
    label_column: str | None = None,
    min_flows_per_period: int = 1,
    min_session_gap_seconds: float = DEFAULT_SESSION_MIN_GAP_SECONDS,
) -> PeriodResult:
    """Assign each row to one of *n_periods* chronological periods."""
    if n_periods < 2:
        raise ValueError("n_periods must be >= 2")
    if ts_col not in frame.columns:
        raise KeyError(f"timestamp column {ts_col!r} missing from frame; run clean_frame first")
    if len(frame) < n_periods * min_flows_per_period:
        raise ValueError(
            f"{len(frame)} rows cannot be split into {n_periods} periods of at least "
            f"{min_flows_per_period} flows"
        )
    ts = frame[ts_col].to_numpy(dtype="int64")
    if ts.size > 1 and np.any(np.diff(ts) < 0):
        raise ValueError(
            "frame is not chronologically sorted; clean_frame must sort by timestamp first"
        )

    period, boundaries = _period_boundaries(ts, n_periods, strategy, min_session_gap_seconds)
    out = frame.copy()
    out[period_col] = period

    # Auto-detect a label column for the per-period histogram: the cleaned frame
    # carries the numeric ``_class_id``; otherwise fall back to the raw class
    # column name used by the NF-v3 family.
    if label_column is None:
        if "_class_id" in out.columns:
            label_column = "_class_id"
        elif "Attack" in out.columns:
            label_column = "Attack"

    counts = np.bincount(period, minlength=n_periods)
    per_period: dict[int, Any] = {}
    label_counts: dict[int, dict[str, int]] = {}
    for index in range(n_periods):
        subset = out[out[period_col] == index]
        label_counts[index] = {}
        if label_column and label_column in subset.columns:
            values = subset[label_column]
            if label_column == "_class_id":
                counts_series = values.astype("int64").value_counts()
                label_counts[index] = {
                    str(int(key)): int(value) for key, value in counts_series.items()
                }
            else:
                counts_series = values.astype("string").value_counts()
                label_counts[index] = {str(key): int(value) for key, value in counts_series.items()}
        per_period[index] = {
            "rows": int(counts[index]),
            "class_counts": label_counts[index],
            "ts_min_utc": _iso(float(subset[ts_col].min())) if len(subset) else None,
            "ts_max_utc": _iso(float(subset[ts_col].max())) if len(subset) else None,
        }

    time_disjoint = True
    for index in range(n_periods - 1):
        left_max = int(out.loc[out[period_col] == index, ts_col].max())
        right_min = int(out.loc[out[period_col] == index + 1, ts_col].min())
        if right_min < left_max:
            time_disjoint = False
            break

    empty = [index for index, count in enumerate(counts) if count < min_flows_per_period]
    if empty:
        raise ValueError(f"periods {empty} have fewer than {min_flows_per_period} flows")

    stats = {
        "n_periods": n_periods,
        "strategy": strategy,
        "rows": int(len(frame)),
        "period_rows": counts.tolist(),
        "label_counts": label_counts,
        "class_presence": {
            index: sorted(set(label_counts[index])) for index in range(n_periods)
        },
        "time_disjoint": bool(time_disjoint),
        "boundaries": boundaries,
        "per_period": per_period,
        "session": {
            key: boundaries[key]
            for key in (
                "session_boundary_row",
                "session_gap_seconds",
                "session_boundary_utc",
                "session_row_counts",
                "rule",
            )
            if key in boundaries
        }
        or None,
    }
    LOGGER.info(
        "assigned %s rows to %s periods (%s): %s",
        f"{len(frame):,}",
        n_periods,
        strategy,
        counts.tolist(),
    )
    return PeriodResult(frame=out, stats=stats)


__all__ = ["PeriodResult", "VALID_STRATEGIES", "assign_periods"]
