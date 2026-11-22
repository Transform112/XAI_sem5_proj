"""Leakage-safe split protocol (Doc 2 S2 + T4).

Inside each period the edges are split 70/10/20 into train/validation/test.
Because several flows can be near-duplicates of each other, splitting rows
independently would leak: a near-copy of a test flow could sit in train. The
protocol therefore shuffles *groups*, never rows:

* ``grouped`` (default): groups are the near-duplicate canonical keys computed
  by :func:`src.data.clean.clean_frame` (5-tuple + time bucket).
* ``host_pair_disjoint`` (robustness variant): groups are unordered host pairs,
  so no host pair appears in more than one split.

Group assignment is deterministic given the seed: groups are shuffled with
``numpy.random.default_rng(seed + period)`` and each group is given to the split
with the largest remaining deficit, which keeps edge fractions close to target
while guaranteeing group-disjointness. Masks are frozen: the split manifest
stores a SHA-256 per (period, split) mask and one combined ``frozen_hash``, so a
later stage can prove it used the same split.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.utils.logging import get_logger

LOGGER = get_logger("xdrift.splits")

SPLITS: tuple[str, ...] = ("train", "val", "test")
VALID_MODES = ("grouped", "host_pair_disjoint")


@dataclass
class SplitResult:
    """Per-period index masks (positions into the frame) plus statistics."""

    masks: dict[int, dict[str, np.ndarray]] = field(repr=False)
    stats: dict[str, Any] = field(default_factory=dict)
    fractions: dict[str, float] = field(default_factory=dict)
    seed: int = 42
    mode: str = "grouped"
    period_col: str = "_period"
    group_col: str = "_group_id"

    def flat_mask(self, split_name: str) -> np.ndarray:
        """Concatenate one split across periods (period order preserved)."""
        return np.concatenate([self.masks[period][split_name] for period in sorted(self.masks)])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _validate_fractions(fractions: Mapping[str, float]) -> dict[str, float]:
    resolved = {name: float(fractions[name]) for name in SPLITS}
    if any(value <= 0 for value in resolved.values()):
        raise ValueError(f"all split fractions must be positive: {resolved}")
    total = sum(resolved.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"split fractions must sum to 1.0, got {total}")
    return resolved


def _host_pair_keys(frame: pd.DataFrame) -> np.ndarray:
    """Unordered host-pair ids: hash of (min(host), max(host))."""
    src = frame["_host_src_id"].to_numpy(dtype="int64")
    dst = frame["_host_dst_id"].to_numpy(dtype="int64")
    low = np.minimum(src, dst)
    high = np.maximum(src, dst)
    pair = pd.util.hash_pandas_object(
        pd.DataFrame({"low": low, "high": high}), index=False
    ).to_numpy(dtype="uint64")
    return pair


def _assign_groups(
    group_ids: np.ndarray,
    group_sizes: np.ndarray,
    *,
    fractions: Mapping[str, float],
    seed: int,
) -> dict[str, np.ndarray]:
    """Assign whole groups to splits to approximate the requested fractions."""
    total = int(group_sizes.sum())
    targets = {name: fractions[name] * total for name in SPLITS}
    assigned = {name: 0.0 for name in SPLITS}
    split_of_group = np.zeros(len(group_ids), dtype=object)
    order = np.random.default_rng(seed).permutation(len(group_ids))
    for group in order:
        deficits = {name: targets[name] - assigned[name] for name in SPLITS}
        # Pick the split with the largest remaining deficit; ties break in the
        # frozen SPLITS order (train, val, test) for determinism.
        chosen = max(SPLITS, key=lambda name: (deficits[name], -SPLITS.index(name)))
        split_of_group[group] = chosen
        assigned[chosen] += float(group_sizes[group])
    result: dict[str, np.ndarray] = {}
    for name in SPLITS:
        result[name] = np.flatnonzero(split_of_group == name)
    return result


# ---------------------------------------------------------------------------
# Building splits
# ---------------------------------------------------------------------------
def build_splits(
    frame: pd.DataFrame,
    *,
    fractions: Mapping[str, float] = {"train": 0.7, "val": 0.1, "test": 0.2},
    seed: int = 42,
    mode: str = "grouped",
    group_col: str = "_group_id",
    period_col: str = "_period",
    min_groups_per_split: int = 10,
    min_share: float = 0.05,
) -> SplitResult:
    """Split every period into train/val/test without leaking groups."""
    resolved_fractions = _validate_fractions(fractions)
    if mode not in VALID_MODES:
        raise ValueError(f"unknown split mode {mode!r}; expected one of {VALID_MODES}")
    for column in (period_col, group_col, "_host_src_id", "_host_dst_id", "_class_id"):
        if column not in frame.columns:
            raise KeyError(f"column {column!r} missing; run clean_frame + assign_periods first")

    group_array = _host_pair_keys(frame) if mode == "host_pair_disjoint" else frame[group_col].to_numpy(dtype="int64")
    periods = np.unique(frame[period_col].to_numpy(dtype="int64"))
    masks: dict[int, dict[str, np.ndarray]] = {}
    per_period: dict[int, Any] = {}
    feasible = True
    infeasible_reasons: list[str] = []
    groups_crossing = 0

    for period in periods:
        period = int(period)
        positions = np.flatnonzero(frame[period_col].to_numpy(dtype="int64") == period)
        period_groups = group_array[positions]
        unique_groups, inverse, sizes = np.unique(period_groups, return_inverse=True, return_counts=True)
        assignment = _assign_groups(unique_groups, sizes, fractions=resolved_fractions, seed=seed + period)
        period_masks: dict[str, np.ndarray] = {}
        for name in SPLITS:
            group_positions = assignment[name]
            member = np.isin(inverse, group_positions)
            period_masks[name] = np.sort(positions[member]).astype("int64")
        # Group-disjointness must hold by construction; verify and count violations.
        seen: dict[int, str] = {}
        for name in SPLITS:
            for group in np.unique(group_array[period_masks[name]]):
                if int(group) in seen and seen[int(group)] != name:
                    groups_crossing += 1
                seen[int(group)] = name
        masks[period] = period_masks

        total = int(len(positions))
        n_groups = int(len(unique_groups))
        split_info = {}
        for name in SPLITS:
            count = int(len(period_masks[name]))
            class_ids = frame["_class_id"].to_numpy(dtype="int64")[period_masks[name]]
            class_counts = {int(key): int(value) for key, value in zip(*np.unique(class_ids, return_counts=True))}
            group_count = int(len(np.unique(group_array[period_masks[name]])))
            split_info[name] = {
                "n_edges": count,
                "fraction": count / total if total else 0.0,
                "n_groups": group_count,
                "class_counts": class_counts,
            }
            if group_count < min_groups_per_split:
                feasible = False
                infeasible_reasons.append(
                    f"period {period} split {name}: {group_count} groups < min_groups_per_split={min_groups_per_split}"
                )
            if total and count / total < min_share:
                feasible = False
                infeasible_reasons.append(
                    f"period {period} split {name}: share {count / total:.3f} < min_share={min_share}"
                )
        # Host-pair leakage check for the robustness variant.
        pair_overlap = 0
        if mode == "host_pair_disjoint":
            pair_sets = {
                name: set(np.unique(group_array[period_masks[name]]).tolist()) for name in SPLITS
            }
            pair_overlap = len(
                (pair_sets["train"] & pair_sets["test"])
                | (pair_sets["train"] & pair_sets["val"])
                | (pair_sets["val"] & pair_sets["test"])
            )
            if pair_overlap:
                feasible = False
                infeasible_reasons.append(f"period {period}: {pair_overlap} host pairs cross splits")
        per_period[period] = {
            "n_edges": total,
            "n_groups": n_groups,
            "max_group_size": int(sizes.max()) if len(sizes) else 0,
            "splits": split_info,
            "host_pair_overlap": pair_overlap,
        }

    stats = {
        "mode": mode,
        "seed": seed,
        "fractions": resolved_fractions,
        "n_groups_total": int(len(np.unique(group_array))),
        "periods": per_period,
        "groups_crossing_splits": groups_crossing,
        "feasible": bool(feasible),
        "infeasible_reasons": infeasible_reasons,
        "min_groups_per_split": min_groups_per_split,
        "min_share": min_share,
    }
    if groups_crossing:
        raise RuntimeError(f"split construction leaked {groups_crossing} groups across splits")
    LOGGER.info(
        "built %s splits for %s periods (seed=%s, feasible=%s)",
        mode,
        len(periods),
        seed,
        feasible,
    )
    return SplitResult(
        masks=masks,
        stats=stats,
        fractions=resolved_fractions,
        seed=seed,
        mode=mode,
        period_col=period_col,
        group_col=group_col,
    )


# ---------------------------------------------------------------------------
# Freezing and persistence
# ---------------------------------------------------------------------------
def mask_sha256(indices: np.ndarray) -> str:
    """SHA-256 of a sorted int64 index mask (stable across platforms)."""
    array = np.sort(np.asarray(indices, dtype="<i8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def freeze_split_masks(
    splits: SplitResult,
    frame: pd.DataFrame | None = None,
    *,
    dataset: str | None = None,
) -> dict[str, Any]:
    """Build the frozen split manifest (hashes + counts + class support)."""
    periods: dict[str, Any] = {}
    for period in sorted(splits.masks):
        block: dict[str, Any] = {}
        for name in SPLITS:
            indices = splits.masks[period][name]
            entry = {
                "n_edges": int(len(indices)),
                "sha256": mask_sha256(indices),
                "first_row_id": int(indices[0]) if len(indices) else None,
                "last_row_id": int(indices[-1]) if len(indices) else None,
            }
            if frame is not None:
                class_ids = frame["_class_id"].to_numpy(dtype="int64")[indices]
                entry["class_counts"] = {
                    int(key): int(value)
                    for key, value in zip(*np.unique(class_ids, return_counts=True))
                }
            block[name] = entry
        periods[str(period)] = block

    payload: dict[str, Any] = {
        "schema_version": 1,
        "dataset": dataset,
        "protocol": "S-D1 leakage-safe split v1",
        "mode": splits.mode,
        "seed": int(splits.seed),
        "fractions": dict(splits.fractions),
        "group_unit": "near_duplicate" if splits.mode == "grouped" else "host_pair",
        "frozen": True,
        "periods": periods,
        "stats": {
            "groups_crossing_splits": splits.stats.get("groups_crossing_splits"),
            "feasible": splits.stats.get("feasible"),
            "infeasible_reasons": splits.stats.get("infeasible_reasons", []),
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["frozen_hash"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def write_split_masks(splits: SplitResult, out_dir: str | Path, *, dataset: str | None = None) -> dict[str, Any]:
    """Persist per-period masks as ``period_<t>_masks.npz`` and return the file map."""
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    for period in sorted(splits.masks):
        path = directory / f"period_{period + 1}_masks.npz"
        np.savez_compressed(
            path,
            train=splits.masks[period]["train"],
            val=splits.masks[period]["val"],
            test=splits.masks[period]["test"],
        )
        files[str(period + 1)] = str(path)
    return {"dataset": dataset, "mask_files": files}


def load_split_masks(directory: str | Path, period: int) -> dict[str, np.ndarray]:
    """Load one period's frozen masks (1-based period number)."""
    path = Path(directory) / f"period_{period}_masks.npz"
    with np.load(path) as data:
        return {name: data[name] for name in ("train", "val", "test")}


__all__ = [
    "SPLITS",
    "SplitResult",
    "VALID_MODES",
    "build_splits",
    "freeze_split_masks",
    "load_split_masks",
    "mask_sha256",
    "write_split_masks",
]
