#!/usr/bin/env python
"""Step 02 - clean the four NF-v3 datasets and fit the preprocessing recipe.

This file has one job: turn a raw NF-v3 CSV into a reusable, leakage-safe
cleaned table plus a fitted preprocessing state.  It does not train a model,
does not split the data into train/validation/test files, and does not run
SMOTE.  Those belong to the next stage.

The order below is deliberate:

    raw CSV
      -> clean rows
      -> assign five chronological periods
      -> fit the scaler/encoder on period-1 TRAIN rows only
      -> save cleaned table + preprocessor state + JSON audit summary

Why fit on period-1 train only?
    The scaler, category vocabulary and missing-value statistics must never see
    validation, test or future-period data.  The split stage will freeze the
    same masks later; this stage only reuses the period-1 train rows to fit the
    recipe.

Why is SMOTE not here?
    SMOTE must run after the split and only on the training rows.  Resampling
    before the split would leak synthetic rows into validation/test.  The
    summary therefore records a read-only ``balance`` block (class support per
    period and for the fitting split) instead of changing any row counts.

All four NF-v3 datasets use the same 55-column schema and the same feature
contract, so the code path is shared.  A dataset with no raw CSV is skipped;
use --strict to turn that skip into a failure.

What the fitted recipe does (50 model inputs, from the 55 physical columns):
    - drop 7 identity/target/time columns (timestamps, IPs, source port,
      Label, Attack);
    - 5 categorical columns -> label encoding fitted on train only;
    - 34 heavy-tailed columns -> log1p -> median impute -> standardise -> clip;
    - 9 integer/code columns -> median impute -> standardise -> clip;
    - add a missing indicator only where train missingness > 0.1%;
    - the two second-bytes columns produce the expected two indicators.

Outputs per cleaned dataset:
    datasets/processed/<dataset>/clean/cleaned.parquet  (or .pkl fallback)
    datasets/processed/<dataset>/clean/preprocessor_state.json
    results/text/data_audit/clean/<dataset>.json
    results/images/preprocessing/<dataset>/fig01..fig10.png

Usage (from the repository root):
    python src/main/02_clean_preprocess.py
    python src/main/02_clean_preprocess.py --dataset NF-UNSW-NB15-v3
    python src/main/02_clean_preprocess.py --dataset all --strict
    python src/main/02_clean_preprocess.py --format pickle
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Mapping

# The scientific stack is required for this stage (unlike step 01, which can
# report missing packages).  Keep the error readable instead of a bare traceback.
try:
    import numpy as np
    import pandas as pd

    import matplotlib

    matplotlib.use("Agg")  # Kaggle/SSH has no display
    import matplotlib.pyplot as plt

    from src.data.clean import clean_frame, load_raw_frame
    from src.data.periods import assign_periods
    from src.data.preprocess import Preprocessor
    from src.data.registry import load_dataset_spec, resolve_dataset_names
    from src.data.splits import build_splits, load_split_masks
    from src.utils.config import load_config
    from src.utils.manifest import sha256_file, write_json
    from src.utils.paths import load_paths
except ImportError as exc:  # pragma: no cover - only on a bare interpreter
    sys.exit(
        f"Missing dependency: {exc}\n"
        "Install the project requirements with:\n"
        "  python -m pip install -r platforms/requirements/common.txt"
    )


# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/audit.yaml"
FOUR_NF_V3_DATASETS = (
    "NF-UNSW-NB15-v3",
    "NF-ToN-IoT-v3",
    "NF-BoT-IoT-v3",
    "NF-CSE-CIC-IDS2018-v3",
)


# ---------------------------------------------------------------------------
# Small output helpers - notebook-style but deliberately simple
# ---------------------------------------------------------------------------
def cell(number: int, total: int, title: str) -> None:
    """Print a visible section banner."""
    print()
    print("=" * 78)
    print(f"CELL {number:02d}/{total:02d} | {title}")
    print("=" * 78)


def line(status: str, name: str, detail: str = "") -> None:
    """Print one status line: [ OK ], [WARN], [SKIP], [FAIL] or [INFO]."""
    print(f"[{status:<4}] {name}" + (f" {detail}" if detail else ""), flush=True)


def info(name: str, detail: str = "") -> None:
    line("INFO", name, detail)


def ok(name: str, detail: str = "") -> None:
    line("OK", name, detail)


def warn(name: str, detail: str = "") -> None:
    line("WARN", name, detail)


def skip(name: str, detail: str = "") -> None:
    line("SKIP", name, detail)


def fail(name: str, detail: str = "") -> None:
    line("FAIL", name, detail)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="02_clean_preprocess",
        description="Clean NF-v3 CSVs and fit the period-1 train preprocessor.",
    )
    parser.add_argument(
        "--dataset",
        default="all",
        help="dataset name, comma-separated names, or 'all' (four NF-v3 datasets)",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="audit/cleaning YAML")
    parser.add_argument("--root", default=None, help="repository root override")
    parser.add_argument("--lane", default=None, help="path lane (cpu, kaggle, ...)")
    parser.add_argument(
        "--format",
        default="auto",
        choices=("auto", "parquet", "pickle"),
        help="cleaned-table format; auto falls back to pickle if parquet is unavailable",
    )
    parser.add_argument("--strict", action="store_true", help="missing datasets fail the run")
    parser.add_argument("--no-hash", action="store_true", help="skip raw CSV SHA-256")
    return parser.parse_args(argv)


def find_root(cli_root: str | None) -> Path:
    """Find the repository root from --root, XDRIFT_ROOT, or this file's parents."""
    candidates: list[Path] = []
    if cli_root:
        candidates.append(Path(cli_root))
    import os

    if os.environ.get("XDRIFT_ROOT"):
        candidates.append(Path(os.environ["XDRIFT_ROOT"]))
    candidates.extend([ROOT, Path.cwd(), *Path.cwd().parents])
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "src").is_dir() and (candidate / "configs").is_dir():
            return candidate
    return ROOT


# ---------------------------------------------------------------------------
# Dataset selection
# ---------------------------------------------------------------------------
def choose_datasets(requested: str, config: Mapping[str, Any], root: Path) -> list[str]:
    """Expand 'all' to the four datasets declared in configs/audit.yaml."""
    if requested == "all":
        names = config.get("datasets")
        if not isinstance(names, list) or not names:
            raise ValueError("configs/audit.yaml must declare a non-empty 'datasets' list")
        names = [str(name) for name in names]
    else:
        names = [name.strip() for name in requested.split(",") if name.strip()]
    # Validate against the registry so a typo fails before any cleaning work.
    return resolve_dataset_names(names, project_root=root)


# ---------------------------------------------------------------------------
# Class-balance snapshot (metadata only; no sampling happens here)
# ---------------------------------------------------------------------------
def summarize_balance(
    frame: pd.DataFrame,
    spec: Any,
    class_map: Mapping[str, int],
    train_positions: np.ndarray,
) -> dict[str, Any]:
    """Describe class support for the full data, each period and the fitting split.

    This is deliberately read-only metadata.  Cleaning must not undersample or
    oversample rows: that happens after the train/validation/test split, in the
    balance stage.  This block gives the next stages and the EDA script one
    consistent set of class counts to reason about.
    """
    class_order = [str(name) for name in spec.class_order]
    class_ids = {
        name: int(class_map[name]) for name in class_order if name in class_map
    }
    if not class_ids:
        raise ValueError(f"{spec.dataset}: no configured class names found in cleaned frame")

    benign_name = "Benign" if "Benign" in class_ids else class_order[0]
    benign_id = class_ids[benign_name]

    def counts_for(subset: pd.DataFrame) -> dict[str, int]:
        return {
            name: int((subset["_class_id"] == class_id).sum())
            for name, class_id in class_ids.items()
        }

    def row_summary(subset: pd.DataFrame) -> dict[str, Any]:
        counts = counts_for(subset)
        total = int(sum(counts.values()))
        attack = int(total - counts.get(benign_name, 0))
        return {
            "rows": total,
            "class_counts": counts,
            "attack_rows": attack,
            "benign_rows": int(counts.get(benign_name, 0)),
            "attack_share_pct": round(100.0 * attack / total, 4) if total else 0.0,
        }

    full = row_summary(frame)
    periods = []
    for period_id in sorted(int(value) for value in frame["_period"].unique()):
        subset = frame.loc[frame["_period"] == period_id]
        item = row_summary(subset)
        item["period"] = period_id + 1
        periods.append(item)

    train = row_summary(frame.iloc[train_positions]) if len(train_positions) else None
    rare_threshold = 1_000
    rare_classes = [
        {
            "class": name,
            "full_rows": full["class_counts"][name],
            "fit_train_rows": int(train["class_counts"][name]) if train else 0,
        }
        for name in class_order
        if full["class_counts"].get(name, 0) < rare_threshold
    ]

    return {
        "class_order": class_order,
        "class_ids": class_ids,
        "benign_class": benign_name,
        "rare_class_threshold": rare_threshold,
        "full": full,
        "periods": periods,
        "fit_train": train,
        "rare_classes": rare_classes,
    }


# ---------------------------------------------------------------------------
# Cleaning + period assignment
# ---------------------------------------------------------------------------
def clean_one_dataset(
    dataset: str,
    *,
    paths: Any,
    config: Mapping[str, Any],
    fmt: str,
    no_hash: bool,
) -> dict[str, Any]:
    """Clean one dataset and fit its period-1-train preprocessing recipe."""
    spec = load_dataset_spec(dataset, project_root=paths.project_root)
    if spec.schema != "nf_v3":
        raise ValueError(f"{dataset}: expected schema 'nf_v3', found {spec.schema!r}")

    # ---- 1. find the raw CSV ------------------------------------------------
    raw_dir = paths.raw_dataset(dataset)
    if not raw_dir.is_dir():
        raise FileNotFoundError(f"raw dataset directory not found: {raw_dir}")
    csv_files = spec.csv_files(raw_dir)
    if not csv_files:
        raise FileNotFoundError(f"no CSV matching {spec.csv_glob!r} under {raw_dir}")
    csv_path = csv_files[0]
    info("dataset", f"{dataset} ({spec.role})")
    info("raw CSV", f"{csv_path} ({csv_path.stat().st_size:,} bytes)")

    # ---- 2. load + clean ----------------------------------------------------
    raw = load_raw_frame(spec, csv_path)
    ok("loaded", f"{len(raw):,} rows x {len(raw.columns)} columns")

    cleaned = clean_frame(
        raw,
        spec,
        sentinel_policy=config.get("sentinels", {}),
        near_duplicate=config.get("near_duplicate", {}),
        dedupe=bool(config.get("dedupe", {}).get("exact", True)),
        label_conflict_policy=str(
            config.get("dedupe", {}).get("label_conflict_policy", "report_only")
        ),
    )
    del raw
    frame = cleaned.frame
    stats = cleaned.stats
    ok(
        "cleaned",
        f"{stats['rows_in']:,} -> {len(frame):,} rows; "
        f"{stats['exact_duplicates_removed']:,} exact duplicates removed",
    )
    ok(
        "hosts/classes/groups",
        f"hosts={stats['hosts']}; classes={len(stats['classes_seen'])}; "
        f"near-duplicate groups={stats['near_duplicate_groups']:,}",
    )

    warnings: list[str] = []
    if stats.get("binary_label_violations", 0):
        message = f"{stats['binary_label_violations']:,} rows disagree with the binary label"
        warn("binary labels", message)
        warnings.append(message)

    # Published counts are provenance, not a cleaning rule.  A secondary CSV
    # may differ legitimately; never drop or relabel rows just to match the YAML.
    observed = {
        str(name): int((frame["_class_id"] == int(class_id)).sum())
        for name, class_id in cleaned.class_map.items()
    }
    published = dict(spec.published_classes)
    mismatches = {
        name: {"observed": observed.get(name, 0), "published": published.get(name)}
        for name in sorted(set(observed) | set(published))
        if published.get(name) is not None and observed.get(name, 0) != int(published[name])
    }
    if mismatches:
        message = f"{len(mismatches)} class counts differ from the dataset YAML"
        warn("published class counts", f"{message} (reported, not enforced)")
        warnings.append(message)
    else:
        ok("published class counts", "all published totals match")

    # ---- 3. five chronological periods --------------------------------------
    period_cfg = config.get("periods", {})
    periods = assign_periods(
        frame,
        n_periods=int(period_cfg.get("count", 5)),
        strategy=str(period_cfg.get("strategy", "session_aware_1_4")),
        min_flows_per_period=int(period_cfg.get("min_flows_per_period", 1)),
        min_session_gap_seconds=float(period_cfg.get("min_session_gap_seconds", 3600)),
    )
    frame = periods.frame
    ok("periods", f"counts={periods.stats['period_rows']}")
    if not periods.stats.get("time_disjoint", True):
        warn("periods", "period timestamp ranges are not disjoint")
        warnings.append("period timestamp ranges are not disjoint")

    # ---- 4. fit only on period-1 training rows ------------------------------
    train_positions, fit_source = period_1_train_positions(frame, dataset, paths, config)
    ok("fit split", fit_source)
    if len(train_positions) == 0:
        raise RuntimeError("period-1 train mask is empty; cannot fit the preprocessor")

    preprocessor = Preprocessor(spec, config=config.get("preprocess", {})).fit(
        frame.iloc[train_positions]
    )
    ok(
        "preprocessor",
        f"fitted on {len(train_positions):,} training rows; "
        f"{len(preprocessor.feature_names)} model features",
    )
    balance = summarize_balance(frame, spec, cleaned.class_map, train_positions)
    ok(
        "balance snapshot",
        f"P1 train benign={balance['fit_train']['benign_rows']:,}; "
        f"rarest class={min(balance['fit_train']['class_counts'].values()):,} rows",
    )

    # ---- 5. save cleaned table, recipe and summary --------------------------
    out_dir = paths.processed_dataset(dataset) / "clean"
    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path, actual_format = write_clean_frame(frame, out_dir, fmt)
    state_path = out_dir / "preprocessor_state.json"
    preprocessor.save(state_path)
    ok("cleaned table", f"{cleaned_path} ({actual_format})")
    ok("preprocessor state", str(state_path))

    # ---- 6. diagnostic figures ---------------------------------------------
    figure_paths = make_preprocessing_figures(
        dataset=dataset,
        frame=frame,
        spec=spec,
        stats=stats,
        periods=periods,
        train_positions=train_positions,
        preprocessor=preprocessor,
        class_map=cleaned.class_map,
        paths=paths,
    )
    ok("figures", f"{len(figure_paths)} saved under {figure_paths[0].parent if figure_paths else 'n/a'}")

    # ---- 7. write the JSON summary -----------------------------------------
    summary = {
        "schema_version": 1,
        "step": "02_clean_preprocess",
        "dataset": dataset,
        "role": spec.role,
        "status": "ok",
        "raw": {
            "csv": str(csv_path),
            "bytes": int(csv_path.stat().st_size),
            "sha256": None if no_hash else sha256_file(csv_path),
            "rows": int(stats["rows_in"]),
            "columns": int(len(spec.all_columns())),
        },
        "cleaning": {
            "rows_in": int(stats["rows_in"]),
            "exact_duplicates_removed": int(stats["exact_duplicates_removed"]),
            "rows_after_clean": int(stats["rows_after_clean"]),
            "hosts": int(stats["hosts"]),
            "classes_seen": list(stats["classes_seen"]),
            "class_counts": observed,
            "published_class_counts": published,
            "published_mismatches": mismatches,
            "near_duplicate_groups": int(stats["near_duplicate_groups"]),
            "binary_label_violations": int(stats.get("binary_label_violations", 0)),
            "negative_to_nan": stats["negative_to_nan"],
            "inf_to_nan": stats["inf_to_nan"],
            "time_range_utc": stats.get("time_range_utc"),
        },
        "periods": periods.stats,
        "fit_split": fit_source,
        "balance": balance,
        "preprocess": {
            "fit_rows": int(len(train_positions)),
            "n_features": int(len(preprocessor.feature_names)),
            "feature_names": list(preprocessor.feature_names),
            "missing_indicators": dict(preprocessor.used_missing_indicators()),
            "state_path": str(state_path),
        },
        "outputs": {
            "cleaned_path": str(cleaned_path),
            "cleaned_format": actual_format,
            "cleaned_bytes": int(cleaned_path.stat().st_size),
            "cleaned_sha256": sha256_file(cleaned_path),
            "preprocessor_state": str(state_path),
            "preprocessor_state_sha256": sha256_file(state_path),
        },
        "figures": [str(path) for path in figure_paths],
        "warnings": warnings,
    }
    summary_dir = paths.audit_root / "clean"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{dataset}.json"
    write_json(summary_path, summary)
    summary["outputs"]["summary_path"] = str(summary_path)
    ok("summary", str(summary_path))
    return summary


# ---------------------------------------------------------------------------
# Split used only to fit the preprocessor
# ---------------------------------------------------------------------------
def period_1_train_positions(
    frame: pd.DataFrame,
    dataset: str,
    paths: Any,
    config: Mapping[str, Any],
) -> tuple[np.ndarray, str]:
    """Return period-1 training row positions and a one-line provenance note.

    Preference:
      1. the frozen mask from the split stage, if it exists;
      2. the same deterministic split rebuilt in memory, for fitting only.

    The in-memory fallback is NOT saved.  The split stage remains the only
    owner of the frozen mask files.
    """
    period_cfg = config.get("periods", {})
    strategy = str(period_cfg.get("strategy", "session_aware_1_4"))
    split_dir = paths.dataset_splits(dataset) / strategy
    mask_file = split_dir / "period_1_masks.npz"

    if mask_file.is_file():
        masks = load_split_masks(split_dir, 1)
        train = np.asarray(masks["train"], dtype="int64")
        if len(train) and int(train.max()) >= len(frame):
            raise RuntimeError(
                f"period-1 train mask index {int(train.max())} is outside the cleaned "
                f"frame ({len(frame)} rows); the dataset changed since the split was frozen"
            )
        return train, f"frozen mask {mask_file} ({len(train):,} rows)"

    split_cfg = config.get("splits", {})
    host_pair = split_cfg.get("host_pair_disjoint", {})
    splits = build_splits(
        frame,
        fractions=split_cfg.get("fractions", {"train": 0.70, "val": 0.10, "test": 0.20}),
        seed=int(split_cfg.get("seed", 42)),
        mode="grouped",
        min_groups_per_split=int(host_pair.get("min_groups_per_split", 10)),
        min_share=float(host_pair.get("min_share", 0.05)),
    )
    train = np.asarray(splits.masks[1]["train"], dtype="int64")
    return train, f"in-memory split for fitting only ({len(train):,} rows; masks not saved)"


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------
def write_clean_frame(frame: pd.DataFrame, out_dir: Path, fmt: str) -> tuple[Path, str]:
    """Write parquet by default; fall back to pickle when no parquet engine exists."""
    if fmt == "pickle":
        path = out_dir / "cleaned.pkl"
        frame.to_pickle(path)
        return path, "pickle"

    path = out_dir / "cleaned.parquet"
    try:
        frame.to_parquet(path, index=False)
        return path, "parquet"
    except (ImportError, ModuleNotFoundError) as exc:
        if fmt == "parquet":
            raise RuntimeError(f"parquet format requested but no parquet engine is installed: {exc}")
        print(f"[INFO] parquet engine unavailable ({exc}); falling back to pickle")
        path = out_dir / "cleaned.pkl"
        frame.to_pickle(path)
        return path, "pickle"


# ---------------------------------------------------------------------------
# Figures: ten quick views of the data and the preprocessing recipe
# ---------------------------------------------------------------------------
def make_preprocessing_figures(
    *,
    dataset: str,
    frame: pd.DataFrame,
    spec: Any,
    stats: Mapping[str, Any],
    periods: Any,
    train_positions: np.ndarray,
    preprocessor: Preprocessor,
    class_map: Mapping[str, int],
    paths: Any,
) -> list[Path]:
    """Save ~10 diagnostic figures under results/images/preprocessing/<dataset>/.

    The figures are intentionally simple.  They answer the questions a reader
    asks before modelling:

    1. how many rows were removed by deduplication?
    2. how imbalanced are the attack classes?
    3. how many rows are in each chronological period?
    4. how does the class mix change over time?
    5. which columns needed sentinel/missing handling?
    6. what does the 50-feature preprocessing recipe consist of?
    7. what does log1p actually do to a heavy-tailed flow feature?
    8. how much skew remains after log1p?
    9. how many categories does each categorical column have?
    10. is the period-1 training split representative of the full data?
    """
    out_dir = paths.figures_root / "preprocessing" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    def save(fig: Any, name: str) -> None:
        fig.tight_layout()
        path = out_dir / name
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    def style(ax: Any) -> None:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25)

    class_names = [name for name in (spec.class_order or ()) if name in class_map]
    if not class_names:
        class_names = sorted(class_map, key=lambda name: int(class_map[name]))
    class_ids = [int(class_map[name]) for name in class_names]
    class_colors = plt.cm.tab20(np.linspace(0, 1, max(len(class_names), 2)))

    # 1. raw rows -> cleaned rows -------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 4.5))
    values = [stats["rows_in"], stats["rows_after_clean"]]
    bars = ax.bar(["raw rows", "cleaned rows"], values, color=["#6b7c93", "#2f6f9f"])
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom")
    ax.set_title(
        f"{dataset}: exact-row deduplication "
        f"({stats['exact_duplicates_removed']:,} duplicates removed)"
    )
    ax.set_ylabel("rows")
    ax.set_ylim(0, max(values) * 1.12)
    style(ax)
    save(fig, "fig01_rows_before_after.png")

    # 2. class distribution --------------------------------------------------
    class_counts = [int((frame["_class_id"] == class_id).sum()) for class_id in class_ids]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    positions = np.arange(len(class_names))
    ax.barh(positions, class_counts, color=class_colors[: len(class_names)])
    ax.set_yticks(positions, class_names)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("flows (log scale)")
    ax.set_title(f"{dataset}: attack-class distribution after cleaning")
    for position, count in zip(positions, class_counts):
        ax.text(count, position, f" {count:,}", va="center", fontsize=8)
    style(ax)
    save(fig, "fig02_class_distribution.png")

    # 3. period sizes --------------------------------------------------------
    period_rows = [int(value) for value in periods.stats["period_rows"]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    positions = np.arange(1, len(period_rows) + 1)
    ax.bar(positions, period_rows, color="#3f7f5f")
    ax.set_xticks(positions, [f"P{index}" for index in positions])
    ax.set_ylabel("flows")
    ax.set_title(f"{dataset}: rows per chronological period")
    for position, value in zip(positions, period_rows):
        ax.text(position, value, f"{value:,}", ha="center", va="bottom", fontsize=8)
    style(ax)
    save(fig, "fig03_period_rows.png")

    # 4. class mix per period ------------------------------------------------
    table = (
        frame.groupby(["_period", "_class_id"]).size().unstack(fill_value=0)
        .reindex(columns=class_ids, fill_value=0)
    )
    shares = table.div(table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 100.0
    fig, ax = plt.subplots(figsize=(11, 5.5))
    x = np.arange(len(shares.index))
    bottom = np.zeros(len(shares.index))
    for name, class_id, color in zip(class_names, class_ids, class_colors):
        values = shares[class_id].to_numpy()
        ax.bar(x, values, bottom=bottom, label=name, color=color)
        bottom += values
    ax.set_xticks(x, [f"P{index}" for index in range(1, len(shares.index) + 1)])
    ax.set_ylabel("% of period flows")
    ax.set_title(f"{dataset}: class mix by period")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    style(ax)
    save(fig, "fig04_period_class_mix.png")

    # 5. sentinel / missing handling ----------------------------------------
    negative = dict(stats.get("negative_to_nan", {}))
    infinite = dict(stats.get("inf_to_nan", {}))
    totals = {
        column: int(negative.get(column, 0)) + int(infinite.get(column, 0))
        for column in set(negative) | set(infinite)
    }
    totals = {column: value for column, value in totals.items() if value > 0}
    top_columns = sorted(totals, key=totals.get, reverse=True)[:10]
    fig, ax = plt.subplots(figsize=(9.5, max(3.5, 0.42 * len(top_columns) + 2)))
    if top_columns:
        y = np.arange(len(top_columns))
        neg_values = [int(negative.get(column, 0)) for column in top_columns]
        inf_values = [int(infinite.get(column, 0)) for column in top_columns]
        ax.barh(y, neg_values, color="#b45f4d", label="negative -> NaN")
        ax.barh(y, inf_values, left=neg_values, color="#d9a441", label="inf -> NaN")
        ax.set_yticks(y)
        ax.set_yticklabels(top_columns, fontsize=8)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_xlabel("cells changed (log scale)")
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No negative or infinite values needed conversion", ha="center", va="center")
        ax.set_axis_off()
    ax.set_title(f"{dataset}: sentinel values converted to NaN")
    style(ax)
    save(fig, "fig05_sentinel_counts.png")

    # 6. what the preprocessing recipe contains ------------------------------
    recipe = {
        "categorical\nlabel encode": len(spec.categorical),
        "log1p +\nstandardise": len(spec.log1p),
        "standardise": len(spec.numeric),
        "missing\nindicators": len(preprocessor.used_missing_indicators()),
    }
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.bar(recipe.keys(), recipe.values(), color=["#2f6f9f", "#3f7f5f", "#d9a441", "#b45f4d"])
    ax.set_ylabel("input columns")
    ax.set_title(f"{dataset}: preprocessing recipe ({len(preprocessor.feature_names)} model inputs)")
    for bar, value in zip(bars, recipe.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom")
    style(ax)
    save(fig, "fig06_recipe_composition.png")

    # 7. log1p on one representative heavy-tailed column ---------------------
    example = "IN_BYTES" if "IN_BYTES" in spec.log1p else (spec.log1p[0] if spec.log1p else None)
    if example:
        values = frame[example].dropna().to_numpy(dtype=float)
        clipped = np.clip(values, 0, None)
        transformed = np.log1p(clipped)
        upper = float(np.percentile(clipped, 99.9)) if len(clipped) else 1.0
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
        axes[0].hist(np.clip(clipped, 0, upper), bins=80, color="#6b7c93")
        axes[0].set_title(f"{example}: raw values (clipped at p99.9)")
        axes[0].set_xlabel("raw value")
        axes[0].set_ylabel("flows")
        axes[1].hist(transformed, bins=80, color="#2f6f9f")
        axes[1].set_title(f"{example}: log1p values")
        axes[1].set_xlabel("log1p(value)")
        for ax in axes:
            ax.set_yscale("log")
            style(ax)
        fig.suptitle(f"{dataset}: effect of log1p on a heavy-tailed feature")
        save(fig, "fig07_log1p_example.png")

    # 8. skew before/after log1p --------------------------------------------
    if spec.log1p:
        raw_skew = frame[list(spec.log1p)].skew().abs().sort_values(ascending=False)
        selected = list(raw_skew.head(8).index)
        transformed_frame = np.log1p(frame[selected].clip(lower=0))
        before = frame[selected].skew().abs()
        after = transformed_frame.skew().abs()
        fig, ax = plt.subplots(figsize=(10, 4.8))
        x = np.arange(len(selected))
        ax.bar(x - 0.2, before.to_numpy(), width=0.4, label="raw", color="#6b7c93")
        ax.bar(x + 0.2, after.to_numpy(), width=0.4, label="log1p", color="#2f6f9f")
        ax.set_xticks(x, selected, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("absolute skew (symlog scale)")
        ax.set_yscale("symlog", linthresh=1)
        ax.set_title(f"{dataset}: skew before and after log1p (top 8 raw columns)")
        ax.legend(frameon=False)
        style(ax)
        save(fig, "fig08_skew_before_after.png")

    # 9. categorical cardinality --------------------------------------------
    if spec.categorical:
        cardinality = frame[list(spec.categorical)].nunique().sort_values()
        fig, ax = plt.subplots(figsize=(8.8, 4.2))
        positions = np.arange(len(cardinality))
        ax.barh(positions, cardinality.to_numpy(), color="#3f7f5f")
        ax.set_yticks(positions, cardinality.index)
        ax.set_xscale("log")
        ax.set_xlabel("distinct values (log scale)")
        ax.set_title(f"{dataset}: categorical cardinality")
        for position, value in zip(positions, cardinality.to_numpy()):
            ax.text(value, position, f" {value:,}", va="center", fontsize=8)
        style(ax)
        save(fig, "fig09_categorical_cardinality.png")

    # 10. full data vs period-1 training class counts ------------------------
    full_counts = frame["_class_id"].value_counts()
    train_counts = frame.iloc[train_positions]["_class_id"].value_counts()
    train_values = [int(train_counts.get(class_id, 0)) for class_id in class_ids]
    full_values = [int(full_counts.get(class_id, 0)) for class_id in class_ids]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    y = np.arange(len(class_names))
    ax.barh(y - 0.2, full_values, height=0.4, label="full cleaned data", color="#6b7c93")
    ax.barh(y + 0.2, train_values, height=0.4, label="period-1 train", color="#2f6f9f")
    ax.set_yticks(y, class_names)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("flows (log scale)")
    ax.set_title(f"{dataset}: class imbalance in the fitting split")
    ax.legend(frameon=False)
    style(ax)
    save(fig, "fig10_train_class_balance.png")

    return saved


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    started = time.perf_counter()
    args = parse_args(argv)

    cell(1, 3, "CONFIGURATION")
    root = find_root(args.root)
    info("project root", str(root))
    try:
        paths = load_paths(lane=args.lane, project_root=root)
        config = load_config(args.config, project_root=root, lane=args.lane)
    except Exception as exc:  # noqa: BLE001 - show a readable configuration error
        fail("configuration", f"{type(exc).__name__}: {exc}")
        return 1
    data_cfg = config.to_dict()
    ok("config", f"{args.config} (lane={paths.lane})")
    ok("paths", f"data={paths.data_root}; results={paths.results_root}")

    try:
        datasets = choose_datasets(args.dataset, data_cfg, root)
    except Exception as exc:  # noqa: BLE001
        fail("dataset list", f"{type(exc).__name__}: {exc}")
        return 1
    ok("datasets", ", ".join(datasets))

    # ---- clean every requested dataset --------------------------------------
    cell(2, 3, f"CLEAN + PREPROCESS ({len(datasets)} dataset(s))")
    completed: list[dict[str, Any]] = []
    missing = 0
    failures = 0
    for index, dataset in enumerate(datasets, start=1):
        print(f"\n--- DATASET {index}/{len(datasets)}: {dataset} " + "-" * 20)
        try:
            summary = clean_one_dataset(
                dataset,
                paths=paths,
                config=data_cfg,
                fmt=args.format,
                no_hash=args.no_hash,
            )
            completed.append(summary)
        except FileNotFoundError as exc:
            missing += 1
            skip(dataset, str(exc))
        except Exception as exc:  # noqa: BLE001 - keep going so other datasets are audited
            failures += 1
            fail(dataset, f"{type(exc).__name__}: {exc}")

    # ---- summary ------------------------------------------------------------
    cell(3, 3, "SUMMARY")
    if not completed:
        fail("datasets", "no requested dataset was cleaned")
        print("\nAdd the raw NF-v3 CSVs first; this stage never downloads data.")
        return 2

    for summary in completed:
        print(
            f"  {summary['dataset']:<28} "
            f"rows={summary['cleaning']['rows_after_clean']:>12,}  "
            f"features={summary['preprocess']['n_features']:>3}"
        )
    if missing:
        warn("missing datasets", f"{missing} requested dataset(s) have no raw CSV")
    if args.strict and missing:
        fail("strict mode", f"{missing} requested dataset(s) are missing")
        return 1
    if failures:
        fail("failed datasets", f"{failures} dataset(s) failed")
        return 1

    ok("status", f"cleaned {len(completed)} dataset(s) in {time.perf_counter() - started:.1f}s")
    print("\nNext: freeze the split masks, compute training-only class weights,")
    print("then use those weights (or optional benign undersampling) in training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
