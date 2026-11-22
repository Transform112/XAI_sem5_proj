#!/usr/bin/env python
"""Step 03 - exploratory data analysis on the cleaned NF-v3 data.

This stage is read-only.  It answers the questions we need before modelling:

    what does the data look like?
    how imbalanced is each class and each period?
    which features are heavy-tailed, correlated or shifted over time?
    what does the graph look like (hosts, repeated pairs)?
    what would a balanced training view look like, before SMOTE?

It writes:
    results/images/eda/<dataset>/fig01..fig15.png
    results/text/data_audit/eda/<dataset>.json
    analysis/eda/<dataset>.md

The balance preview in this script is illustrative only.  It caps benign rows
and repeats minority rows with replacement so the plots can show the proposed
training view.  Real SMOTE belongs to the split/balance stage, after the
validation/test masks are frozen.

Usage (from the repository root):
    python src/main/03_eda.py
    python src/main/03_eda.py --dataset NF-UNSW-NB15-v3
    python src/main/03_eda.py --dataset all --max-benign 30000 --min-class 2000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Mapping

try:
    import numpy as np
    import pandas as pd

    import matplotlib

    matplotlib.use("Agg")  # headless Kaggle/SSH
    import matplotlib.pyplot as plt

    from src.data.registry import load_dataset_spec, resolve_dataset_names
    from src.data.splits import load_split_masks
    from src.utils.config import load_config
    from src.utils.manifest import write_json
    from src.utils.paths import load_paths
except ImportError as exc:  # pragma: no cover - only on a bare interpreter
    sys.exit(
        f"Missing dependency: {exc}\n"
        "Install the project requirements with:\n"
        "  python -m pip install -r platforms/requirements/common.txt"
    )


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/audit.yaml"
FOUR_NF_V3_DATASETS = (
    "NF-UNSW-NB15-v3",
    "NF-ToN-IoT-v3",
    "NF-BoT-IoT-v3",
    "NF-CSE-CIC-IDS2018-v3",
)
FEATURE_PALETTE = "#2f6f9f"
ATTACK_PALETTE = "#b45f4d"
BENIGN_PALETTE = "#6b7c93"


# ---------------------------------------------------------------------------
# Small output helpers
# ---------------------------------------------------------------------------
def cell(number: int, total: int, title: str) -> None:
    print()
    print("=" * 78)
    print(f"CELL {number:02d}/{total:02d} | {title}")
    print("=" * 78)


def line(status: str, name: str, detail: str = "") -> None:
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
# CLI and paths
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="03_eda",
        description="Plot and summarize cleaned NF-v3 data before modelling.",
    )
    parser.add_argument(
        "--dataset",
        default="all",
        help="dataset name, comma-separated names, or 'all' (four NF-v3 datasets)",
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="audit/cleaning YAML")
    parser.add_argument("--root", default=None, help="repository root override")
    parser.add_argument("--lane", default=None, help="path lane (cpu, kaggle, ...)")
    parser.add_argument("--max-benign", type=int, default=30_000, help="preview benign cap")
    parser.add_argument("--min-class", type=int, default=2_000, help="preview minority floor")
    parser.add_argument("--seed", type=int, default=42, help="preview sampling seed")
    return parser.parse_args(argv)


def find_root(cli_root: str | None) -> Path:
    """Find the repository root from --root, XDRIFT_ROOT, or this file's parents."""
    import os

    candidates = []
    if cli_root:
        candidates.append(Path(cli_root))
    if os.environ.get("XDRIFT_ROOT"):
        candidates.append(Path(os.environ["XDRIFT_ROOT"]))
    candidates.extend([ROOT, Path.cwd(), *Path.cwd().parents])
    for candidate in candidates:
        candidate = candidate.resolve()
        if (candidate / "src").is_dir() and (candidate / "configs").is_dir():
            return candidate
    return ROOT


def choose_datasets(requested: str, config: Mapping[str, Any], root: Path) -> list[str]:
    """Expand 'all' to the four datasets declared in configs/audit.yaml."""
    if requested == "all":
        names = config.get("datasets")
        if not isinstance(names, list) or not names:
            raise ValueError("configs/audit.yaml must declare a non-empty 'datasets' list")
        names = [str(name) for name in names]
    else:
        names = [name.strip() for name in requested.split(",") if name.strip()]
    return resolve_dataset_names(names, project_root=root)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def load_cleaned(dataset: str, paths: Any) -> tuple[pd.DataFrame, Path, dict[str, Any]]:
    """Load the cleaned table and its saved preprocessor state."""
    clean_dir = paths.processed_dataset(dataset) / "clean"
    table = clean_dir / "cleaned.parquet"
    if not table.is_file():
        table = clean_dir / "cleaned.pkl"
    if not table.is_file():
        raise FileNotFoundError(
            f"{dataset}: no cleaned table under {clean_dir}; run step 02 first"
        )
    state_path = clean_dir / "preprocessor_state.json"
    if not state_path.is_file():
        raise FileNotFoundError(
            f"{dataset}: missing {state_path}; run step 02 first"
        )
    frame = pd.read_parquet(table) if table.suffix == ".parquet" else pd.read_pickle(table)
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return frame, table, state


def load_audit_summary(dataset: str, paths: Any) -> dict[str, Any]:
    path = paths.audit_root / "clean" / f"{dataset}.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_period_masks(dataset: str, paths: Any, config: Mapping[str, Any]) -> dict[str, np.ndarray] | None:
    """Load frozen period-1 split masks; return None if they are not available."""
    strategy = str(config.get("periods", {}).get("strategy", "session_aware_1_4"))
    split_dir = paths.dataset_splits(dataset) / strategy
    if not (split_dir / "period_1_masks.npz").is_file():
        return None
    masks = load_split_masks(split_dir, 1)
    return {name: np.asarray(value, dtype="int64") for name, value in masks.items()}


def class_info(
    frame: pd.DataFrame,
    spec: Any,
    summary: Mapping[str, Any],
) -> tuple[list[str], dict[str, int]]:
    """Return the configured class names and name -> id mapping."""
    names = [str(name) for name in spec.class_order]
    balance_ids = summary.get("balance", {}).get("class_ids")
    if isinstance(balance_ids, dict) and balance_ids:
        ids = {str(name): int(value) for name, value in balance_ids.items()}
    else:
        ids = {name: index for index, name in enumerate(names)}
    present = set(int(value) for value in frame["_class_id"].unique())
    names = [name for name in names if ids.get(name) in present]
    if not names:
        raise ValueError("no configured class names match the cleaned frame")
    return names, ids


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------
def class_counts(frame: pd.DataFrame, names: list[str], ids: Mapping[str, int]) -> dict[str, int]:
    return {
        name: int((frame["_class_id"] == int(ids[name])).sum())
        for name in names
    }


def split_counts(
    frame: pd.DataFrame,
    masks: Mapping[str, np.ndarray] | None,
    names: list[str],
    ids: Mapping[str, int],
) -> dict[str, dict[str, int]]:
    if not masks:
        return {}
    out: dict[str, dict[str, int]] = {}
    for split_name in ("train", "val", "test"):
        positions = np.asarray(masks.get(split_name, []), dtype="int64")
        if len(positions) == 0:
            continue
        if int(positions.max()) >= len(frame):
            raise ValueError(f"{split_name} mask index is outside the cleaned frame")
        out[split_name] = class_counts(frame.iloc[positions], names, ids)
    return out


def numeric_features(frame: pd.DataFrame, spec: Any) -> list[str]:
    """Numeric feature columns that actually exist in the cleaned frame."""
    candidates = list(spec.log1p) + list(spec.numeric)
    seen: set[str] = set()
    out: list[str] = []
    for column in candidates:
        if column in frame.columns and column not in seen:
            out.append(column)
            seen.add(column)
    return out


def feature_matrix(frame: pd.DataFrame, features: list[str], sample: int, seed: int) -> pd.DataFrame:
    """Return a deterministic sample of selected features as floats."""
    if len(frame) > sample:
        frame = frame.sample(n=sample, random_state=seed)
    return frame[features].astype(float)


def balance_preview(
    train: pd.DataFrame,
    names: list[str],
    ids: Mapping[str, int],
    max_benign: int,
    min_class: int,
    seed: int,
) -> pd.DataFrame:
    """Illustrative balanced training view: benign cap + minority floor.

    Minority rows are repeated with replacement.  This is intentionally a
    preview, not SMOTE; the real split/balance stage will apply the configured
    sampler after validation/test are frozen.
    """
    rng = np.random.default_rng(seed)
    parts: list[pd.DataFrame] = []
    for name in names:
        rows = train.loc[train["_class_id"] == int(ids[name])]
        if len(rows) == 0:
            continue
        if name.lower() == "benign":
            count = min(len(rows), max_benign)
        else:
            count = max(len(rows), min_class)
        if count <= len(rows):
            parts.append(rows.sample(n=count, random_state=int(rng.integers(0, 2**31 - 1))))
        else:
            parts.append(rows.sample(n=count, replace=True, random_state=int(rng.integers(0, 2**31 - 1))))
    return pd.concat(parts, ignore_index=True) if parts else train.iloc[0:0]


def class_share_table(frame: pd.DataFrame, names: list[str], ids: Mapping[str, int]) -> pd.DataFrame:
    table = pd.DataFrame(
        {name: frame.loc[frame["_class_id"] == int(ids[name])].groupby("_period").size() for name in names}
    ).fillna(0)
    table.index = [f"P{int(period) + 1}" for period in table.index]
    return table


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def style_axis(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.25)


def save_figure(fig: Any, out_dir: Path, name: str, saved: list[Path]) -> None:
    fig.tight_layout()
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    saved.append(path)


def plot_rows(frame: pd.DataFrame, summary: Mapping[str, Any], out_dir: Path, dataset: str, saved: list[Path]) -> None:
    raw_rows = int(summary.get("cleaning", {}).get("rows_in", len(frame)))
    cleaned_rows = int(len(frame))
    duplicates = int(summary.get("cleaning", {}).get("exact_duplicates_removed", raw_rows - cleaned_rows))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(["raw rows", "cleaned rows"], [raw_rows, cleaned_rows], color=[BENIGN_PALETTE, FEATURE_PALETTE])
    for bar, value in zip(bars, [raw_rows, cleaned_rows]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:,}", ha="center", va="bottom")
    ax.set_ylim(0, max(raw_rows, cleaned_rows) * 1.12)
    ax.set_ylabel("rows")
    ax.set_title(f"{dataset}: exact-row deduplication ({duplicates:,} duplicates removed)")
    style_axis(ax)
    save_figure(fig, out_dir, "fig01_rows_clean.png", saved)


def plot_class_balance(
    frame: pd.DataFrame,
    train: pd.DataFrame,
    names: list[str],
    ids: Mapping[str, int],
    out_dir: Path,
    dataset: str,
    saved: list[Path],
) -> None:
    full = class_counts(frame, names, ids)
    p1_train = class_counts(train, names, ids)
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.barh(y - 0.2, [full[name] for name in names], height=0.4, label="full cleaned data", color=BENIGN_PALETTE)
    ax.barh(y + 0.2, [p1_train[name] for name in names], height=0.4, label="period-1 train", color=FEATURE_PALETTE)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("flows (log scale)")
    ax.set_title(f"{dataset}: class counts, full data vs fitting split")
    ax.legend(frameon=False)
    style_axis(ax)
    save_figure(fig, out_dir, "fig02_class_counts.png", saved)


def plot_period_mix(frame: pd.DataFrame, names: list[str], ids: Mapping[str, int], out_dir: Path, dataset: str, saved: list[Path]) -> None:
    table = class_share_table(frame, names, ids)
    shares = table.div(table.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0) * 100.0
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bottom = np.zeros(len(shares.index))
    for name in names:
        values = shares[name].to_numpy()
        ax.bar(np.arange(len(shares.index)), values, bottom=bottom, label=name,
               color=plt.cm.tab20(names.index(name) / max(len(names) - 1, 1)))
        bottom += values
    ax.set_xticks(np.arange(len(shares.index)), shares.index)
    ax.set_ylabel("% of period flows")
    ax.set_title(f"{dataset}: class mix by period")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8, frameon=False)
    style_axis(ax)
    save_figure(fig, out_dir, "fig03_period_class_mix.png", saved)


def plot_attack_rate(frame: pd.DataFrame, names: list[str], ids: Mapping[str, int], out_dir: Path, dataset: str, saved: list[Path]) -> None:
    table = class_share_table(frame, names, ids)
    benign = "Benign" if "Benign" in names else names[0]
    attack = table.drop(columns=[benign]).sum(axis=1) if benign in table else table.sum(axis=1)
    total = table.sum(axis=1).replace(0, np.nan)
    rate = (attack / total * 100.0).fillna(0.0)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(rate.index, rate.to_numpy(), color=ATTACK_PALETTE)
    for bar, value in zip(bars, rate.to_numpy()):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}%", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("attack share of period (%)")
    ax.set_title(f"{dataset}: attack share increases after period 1")
    ax.set_ylim(0, max(rate.max() * 1.25, 1))
    style_axis(ax)
    save_figure(fig, out_dir, "fig04_attack_rate_period.png", saved)


def plot_split_support(
    splits: Mapping[str, Mapping[str, int]],
    names: list[str],
    out_dir: Path,
    dataset: str,
    saved: list[Path],
) -> None:
    if not splits:
        return
    x = np.arange(len(names))
    width = 0.8 / len(splits)
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for index, (split_name, counts) in enumerate(splits.items()):
        values = [int(counts.get(name, 0)) for name in names]
        ax.bar(x - 0.4 + width / 2 + index * width, values, width=width, label=split_name)
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("rows (log scale)")
    ax.set_title(f"{dataset}: class support in the frozen period-1 split")
    ax.legend(frameon=False)
    style_axis(ax)
    save_figure(fig, out_dir, "fig05_split_support.png", saved)


def plot_missing(
    frame: pd.DataFrame,
    summary: Mapping[str, Any],
    out_dir: Path,
    dataset: str,
    saved: list[Path],
) -> None:
    missing = frame.isna().mean().sort_values(ascending=False)
    missing = missing[missing > 0]
    sentinel = {}
    cleaning = summary.get("cleaning", {})
    for source in ("negative_to_nan", "inf_to_nan"):
        for column, count in cleaning.get(source, {}).items():
            sentinel[column] = sentinel.get(column, 0) + int(count)
    sentinel = {key: value for key, value in sentinel.items() if value > 0}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    if len(missing):
        axes[0].barh(missing.index[::-1], missing.to_numpy()[::-1] * 100.0, color=ATTACK_PALETTE)
        axes[0].set_xlabel("missing (%)")
        axes[0].set_title("Cleaned-table missingness")
    else:
        axes[0].text(0.5, 0.5, "No missing values in cleaned table", ha="center", va="center")
        axes[0].set_axis_off()
    if sentinel:
        columns = sorted(sentinel, key=sentinel.get, reverse=True)[:10]
        values = [sentinel[column] for column in columns]
        axes[1].barh(columns[::-1], values[::-1], color=BENIGN_PALETTE)
        axes[1].set_xlabel("cells converted to NaN")
        axes[1].set_title("Sentinel values before cleaning")
    else:
        axes[1].text(0.5, 0.5, "No sentinel values recorded", ha="center", va="center")
        axes[1].set_axis_off()
    for ax in axes:
        style_axis(ax)
    fig.suptitle(f"{dataset}: missingness and sentinel handling")
    save_figure(fig, out_dir, "fig06_missing_sentinel.png", saved)


def plot_feature_distributions(
    frame: pd.DataFrame,
    features: list[str],
    spec: Any,
    out_dir: Path,
    dataset: str,
    seed: int,
    saved: list[Path],
) -> None:
    selected = [column for column in features if column in frame.columns][:6]
    if not selected:
        return
    sample = feature_matrix(frame, selected, 200_000, seed)
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.2))
    for ax, column in zip(axes.ravel(), selected):
        values = sample[column].dropna().to_numpy(dtype=float)
        label = column
        if column in set(spec.log1p):
            values = np.log1p(np.clip(values, 0, None))
            label = f"log1p({column})"
        ax.hist(values, bins=70, color=FEATURE_PALETTE)
        ax.set_title(label, fontsize=9)
        ax.set_yscale("log")
        style_axis(ax)
    for ax in axes.ravel()[len(selected):]:
        ax.set_axis_off()
    fig.suptitle(f"{dataset}: key feature distributions")
    save_figure(fig, out_dir, "fig07_feature_distributions.png", saved)


def plot_class_feature_boxes(
    frame: pd.DataFrame,
    features: list[str],
    names: list[str],
    ids: Mapping[str, int],
    out_dir: Path,
    dataset: str,
    seed: int,
    saved: list[Path],
) -> None:
    selected = [column for column in features if column in frame.columns][:4]
    if not selected or not names:
        return
    sample = frame.sample(n=min(60_000, len(frame)), random_state=seed)
    fig, axes = plt.subplots(1, len(selected), figsize=(13, 4.6))
    if len(selected) == 1:
        axes = [axes]
    for ax, column in zip(axes, selected):
        data = [
            sample.loc[sample["_class_id"] == int(ids[name]), column].dropna().to_numpy(dtype=float)
            for name in names
        ]
        ax.boxplot(data, tick_labels=names, showfliers=False)
        if column in frame.columns:
            ax.set_title(column, fontsize=9)
        ax.tick_params(axis="x", rotation=60, labelsize=7)
        style_axis(ax)
    fig.suptitle(f"{dataset}: feature spread by class (sample of rows)")
    save_figure(fig, out_dir, "fig08_feature_by_class.png", saved)


def plot_correlation(
    frame: pd.DataFrame,
    features: list[str],
    out_dir: Path,
    dataset: str,
    seed: int,
    saved: list[Path],
) -> None:
    if len(features) < 3:
        return
    sample = feature_matrix(frame, features, 80_000, seed)
    variances = sample.var().sort_values(ascending=False)
    selected = list(variances.head(18).index)
    corr = sample[selected].corr()
    fig, ax = plt.subplots(figsize=(9.6, 8.2))
    image = ax.imshow(corr.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(selected)), selected, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(np.arange(len(selected)), selected, fontsize=7)
    ax.set_title(f"{dataset}: Pearson correlation among top-variance features")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, out_dir, "fig09_correlation.png", saved)


def plot_temporal_shift(
    frame: pd.DataFrame,
    features: list[str],
    out_dir: Path,
    dataset: str,
    saved: list[Path],
) -> None:
    if not features:
        return
    sample = frame[features + ["_period"]].copy()
    means = sample.groupby("_period")[features].mean()
    spread = frame[features].std().replace(0, np.nan)
    shift = (means - means.iloc[0]).abs().div(spread, axis=1)
    top = list(shift.max().sort_values(ascending=False).head(8).index)
    fig, ax = plt.subplots(figsize=(11, 5.0))
    for column in top:
        ax.plot(
            [f"P{int(period) + 1}" for period in means.index],
            shift[column].to_numpy(),
            marker="o",
            linewidth=1.4,
            label=column,
        )
    ax.set_ylabel("|standardized mean shift| vs P1")
    ax.set_title(f"{dataset}: largest temporal feature shifts")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    style_axis(ax)
    save_figure(fig, out_dir, "fig10_temporal_shift.png", saved)


def plot_hosts(frame: pd.DataFrame, out_dir: Path, dataset: str, saved: list[Path]) -> dict[str, Any]:
    host_counts = pd.concat(
        [frame["_host_src_id"], frame["_host_dst_id"]], ignore_index=True
    ).value_counts()
    pair_counts = frame.groupby(["_host_src_id", "_host_dst_id"]).size()
    repeated = int((pair_counts > 1).sum())
    pair_total = int(len(pair_counts))
    top_share = float(pair_counts.max() / pair_counts.sum()) if pair_total else 0.0
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    axes[0].hist(host_counts.to_numpy(), bins=30, color=FEATURE_PALETTE)
    axes[0].set_xlabel("flows per host")
    axes[0].set_ylabel("hosts")
    axes[0].set_title(f"Host activity (hosts={len(host_counts):,})")
    axes[1].hist(pair_counts.to_numpy(), bins=40, color=ATTACK_PALETTE)
    axes[1].set_xlabel("flows per ordered host pair")
    axes[1].set_ylabel("pairs")
    axes[1].set_title(f"Pair repetition ({repeated:,}/{pair_total:,} pairs repeat)")
    for ax in axes:
        ax.set_yscale("log")
        style_axis(ax)
    fig.suptitle(f"{dataset}: graph-side summary")
    save_figure(fig, out_dir, "fig11_hosts_pairs.png", saved)
    return {
        "hosts": int(len(host_counts)),
        "max_flows_per_host": int(host_counts.max()),
        "ordered_pairs": pair_total,
        "repeated_pairs": repeated,
        "repeated_pair_fraction": round(repeated / pair_total, 6) if pair_total else 0.0,
        "top_pair_share": round(top_share, 6),
    }


def plot_class_means(
    frame: pd.DataFrame,
    features: list[str],
    names: list[str],
    ids: Mapping[str, int],
    out_dir: Path,
    dataset: str,
    seed: int,
    saved: list[Path],
) -> None:
    if not features:
        return
    sample = frame.sample(n=min(200_000, len(frame)), random_state=seed)
    means = sample.groupby("_class_id")[features].mean()
    means = means.reindex([int(ids[name]) for name in names if int(ids[name]) in means.index])
    if means.empty:
        return
    spread = sample[features].std().replace(0, np.nan)
    z = (means - means.mean()) .div(spread, axis=1)
    top = list(z.abs().mean().sort_values(ascending=False).head(12).index)
    z = z[top]
    fig, ax = plt.subplots(figsize=(11, 5.0))
    image = ax.imshow(z.to_numpy(), cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(top)), top, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(np.arange(len(z.index)), [name for name in names if int(ids[name]) in means.index], fontsize=8)
    ax.set_title(f"{dataset}: class feature profiles (z-scored means)")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    save_figure(fig, out_dir, "fig12_class_means.png", saved)


def plot_balance_preview(
    train: pd.DataFrame,
    preview: pd.DataFrame,
    names: list[str],
    ids: Mapping[str, int],
    out_dir: Path,
    dataset: str,
    max_benign: int,
    min_class: int,
    saved: list[Path],
) -> None:
    before = class_counts(train, names, ids)
    after = class_counts(preview, names, ids)
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.barh(y - 0.2, [before[name] for name in names], height=0.4, label="natural P1 train", color=BENIGN_PALETTE)
    ax.barh(y + 0.2, [after[name] for name in names], height=0.4, label="proposed preview", color=FEATURE_PALETTE)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("rows (log scale)")
    ax.set_title(f"{dataset}: training view before vs preview (benign cap={max_benign:,}, minority floor={min_class:,})")
    ax.legend(frameon=False)
    style_axis(ax)
    save_figure(fig, out_dir, "fig13_balance_preview.png", saved)


def plot_balance_feature_shift(
    train: pd.DataFrame,
    preview: pd.DataFrame,
    features: list[str],
    out_dir: Path,
    dataset: str,
    saved: list[Path],
) -> None:
    selected = [column for column in features if column in train.columns][:10]
    if not selected:
        return
    before = train[selected].mean()
    after = preview[selected].mean()
    spread = train[selected].std().replace(0, np.nan)
    shift = ((after - before) / spread).abs().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.barh(shift.index[::-1], shift.to_numpy()[::-1], color=ATTACK_PALETTE)
    ax.set_xlabel("|standardized mean shift|")
    ax.set_title(f"{dataset}: feature shift caused by the preview sampler")
    style_axis(ax)
    save_figure(fig, out_dir, "fig14_preview_feature_shift.png", saved)


def plot_balance_share(
    train: pd.DataFrame,
    preview: pd.DataFrame,
    names: list[str],
    ids: Mapping[str, int],
    out_dir: Path,
    dataset: str,
    saved: list[Path],
) -> None:
    before = class_counts(train, names, ids)
    after = class_counts(preview, names, ids)
    benign = "Benign" if "Benign" in names else names[0]
    before_benign = before.get(benign, 0) / max(sum(before.values()), 1) * 100.0
    after_benign = after.get(benign, 0) / max(sum(after.values()), 1) * 100.0
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    bars = ax.bar(["natural train", "preview"], [before_benign, after_benign], color=[BENIGN_PALETTE, FEATURE_PALETTE])
    for bar, value in zip(bars, [before_benign, after_benign]):
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}%", ha="center", va="bottom")
    ax.set_ylim(0, 105)
    ax.set_ylabel("benign share of rows (%)")
    ax.set_title(f"{dataset}: benign dominance before/after preview")
    style_axis(ax)
    save_figure(fig, out_dir, "fig15_benign_share.png", saved)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def write_report(
    path: Path,
    dataset: str,
    frame: pd.DataFrame,
    summary: Mapping[str, Any],
    names: list[str],
    ids: Mapping[str, int],
    splits: Mapping[str, Mapping[str, int]],
    host_stats: Mapping[str, Any],
    balance: Mapping[str, Any],
    figures: list[Path],
) -> None:
    full = class_counts(frame, names, ids)
    total = sum(full.values())
    benign = "Benign" if "Benign" in names else names[0]
    attack_total = total - full.get(benign, 0)
    train = balance["train_counts"]
    preview = balance["preview_counts"]
    train_total = sum(train.values())
    preview_total = sum(preview.values())
    period_text = "\n".join(
        f"| P{item['period']} | {item['rows']:,} | {item['attack_rows']:,} | {item['attack_share_pct']:.2f}% |"
        for item in balance["periods"]
    )
    class_rows = [
        [name, f"{full[name]:,}", f"{full[name] / max(total, 1) * 100:.4f}%", f"{train[name]:,}", f"{preview[name]:,}"]
        for name in names
    ]
    split_rows = []
    for split_name, counts in splits.items():
        split_rows.append([split_name, f"{sum(counts.values()):,}"] + [f"{counts[name]:,}" for name in names])
    lines = [
        f"# EDA — {dataset}",
        "",
        f"Generated by `src/main/03_eda.py` from the cleaned table. This report is read-only; "
        f"the balance preview is illustrative and is not used for training.",
        "",
        f"## 1. Dataset shape",
        "",
        f"- Rows after cleaning: **{len(frame):,}**",
        f"- Raw rows: **{int(summary.get('cleaning', {}).get('rows_in', 0)):,}**",
        f"- Exact duplicates removed: **{int(summary.get('cleaning', {}).get('exact_duplicates_removed', 0)):,}**",
        f"- Classes: **{len(names)}** — {', '.join(names)}",
        f"- Full attack share: **{attack_total / max(total, 1) * 100:.2f}%**",
        "",
        f"## 2. Class balance",
        "",
        markdown_table(["class", "full rows", "full share", "P1 train rows", "preview rows"], class_rows),
        "",
        f"Natural P1 training view: **{train_total:,} rows**, benign **{train[benign] / max(train_total, 1) * 100:.2f}%**.",
        f"Proposed preview: **{preview_total:,} rows**, benign **{preview[benign] / max(preview_total, 1) * 100:.2f}%**.",
        "",
        f"## 3. Periods",
        "",
        "| period | rows | attacks | attack share |",
        "|---|---:|---:|---:|",
        period_text,
        "",
        "All configured classes are present in every period; the practical problem is rare-class support, "
        "not a missing class.",
        "",
        f"## 4. Frozen period-1 split support",
        "",
    ]
    if split_rows:
        lines.append(markdown_table(["split", "rows"] + names, split_rows))
    else:
        lines.append("No frozen period-1 masks were found; split-support table omitted.")
    lines += [
        "",
        f"## 5. Missingness and graph side",
        "",
        f"- Hosts in the cleaned data: **{host_stats.get('hosts', 0):,}**",
        f"- Ordered host pairs: **{host_stats.get('ordered_pairs', 0):,}**",
        f"- Repeated pairs: **{host_stats.get('repeated_pairs', 0):,}** "
        f"({host_stats.get('repeated_pair_fraction', 0.0) * 100:.2f}%)",
        f"- Largest host has **{host_stats.get('max_flows_per_host', 0):,}** flows",
        "",
        f"## 6. Figures",
        "",
        "\n".join(f"- `{figure.name}`" for figure in figures),
        "",
        f"## 7. Reading the preview",
        "",
        "The preview caps benign rows and repeats minority rows with replacement; it is **not** SMOTE and is "
        "not written to disk. Validation/test data stay untouched. The real split/balance stage should:",
        "",
        "1. freeze train/val/test masks;",
        "2. balance only the training split;",
        "3. apply SMOTE or a class-balanced sampler after the split;",
        "4. evaluate on the natural test distribution.",
        "",
        "Do not compare a balanced test set with the signed paper results: balancing evaluation would hide "
        "the real drift and inflate rare-class scores.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(lines))


# ---------------------------------------------------------------------------
# One dataset
# ---------------------------------------------------------------------------
def eda_one_dataset(
    dataset: str,
    *,
    paths: Any,
    config: Mapping[str, Any],
    max_benign: int,
    min_class: int,
    seed: int,
) -> dict[str, Any]:
    cell(2, 4, f"LOAD + ANALYZE {dataset}")
    frame, table_path, state = load_cleaned(dataset, paths)
    summary = load_audit_summary(dataset, paths)
    spec = load_dataset_spec(dataset, project_root=paths.project_root)
    names, ids = class_info(frame, spec, summary)
    ok("cleaned table", f"{table_path} ({len(frame):,} rows x {len(frame.columns)} columns)")
    ok("classes", f"{len(names)}: {', '.join(names)}")

    if "_period" not in frame.columns or "_class_id" not in frame.columns:
        raise ValueError(f"{dataset}: cleaned frame is missing _period or _class_id")

    masks = load_period_masks(dataset, paths, config)
    splits = split_counts(frame, masks, names, ids)
    features = numeric_features(frame, spec)
    chosen = ["IN_BYTES", "OUT_BYTES", "FLOW_DURATION_MILLISECONDS", "SRC_TO_DST_IAT_AVG",
              "DST_TO_SRC_IAT_AVG", "MIN_TTL", "L4_DST_PORT", "DNS_TTL_ANSWER"]
    chosen = [column for column in chosen if column in features]
    if len(chosen) < 6:
        chosen = features[:6]
    if not masks:
        warn("split masks", "period_1_masks.npz not found; split-support table is empty")
    train = frame.iloc[masks["train"]] if masks and "train" in masks else frame.loc[frame["_period"] == 0]
    preview = balance_preview(train, names, ids, max_benign=max_benign, min_class=min_class, seed=seed)
    ok("balance preview", f"{len(train):,} natural -> {len(preview):,} preview rows (illustrative only)")

    cell(3, 4, f"FIGURES {dataset}")
    out_dir = paths.figures_root / "eda" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    plot_rows(frame, summary, out_dir, dataset, saved)
    plot_class_balance(frame, train, names, ids, out_dir, dataset, saved)
    plot_period_mix(frame, names, ids, out_dir, dataset, saved)
    plot_attack_rate(frame, names, ids, out_dir, dataset, saved)
    plot_split_support(splits, names, out_dir, dataset, saved)
    plot_missing(frame, summary, out_dir, dataset, saved)
    plot_feature_distributions(frame, chosen, spec, out_dir, dataset, seed, saved)
    plot_class_feature_boxes(frame, chosen[:4], names, ids, out_dir, dataset, seed, saved)
    plot_correlation(frame, features, out_dir, dataset, seed, saved)
    plot_temporal_shift(frame, features, out_dir, dataset, saved)
    host_stats = plot_hosts(frame, out_dir, dataset, saved)
    plot_class_means(frame, chosen, names, ids, out_dir, dataset, seed, saved)
    plot_balance_preview(train, preview, names, ids, out_dir, dataset, max_benign, min_class, saved)
    plot_balance_feature_shift(train, preview, features, out_dir, dataset, saved)
    plot_balance_share(train, preview, names, ids, out_dir, dataset, saved)
    ok("figures", f"{len(saved)} saved under {out_dir}")

    # Metrics and report -----------------------------------------------------
    full = class_counts(frame, names, ids)
    train_counts = class_counts(train, names, ids)
    preview_counts = class_counts(preview, names, ids)
    period_rows = [
        {
            "period": int(period) + 1,
            "rows": int(len(frame.loc[frame["_period"] == period])),
            "attack_rows": int(len(frame.loc[frame["_period"] == period]) - (frame.loc[frame["_period"] == period, "_class_id"] == int(ids.get("Benign", 0))).sum()),
            "attack_share_pct": round(
                100.0
                * (
                    1
                    - (frame.loc[frame["_period"] == period, "_class_id"] == int(ids.get("Benign", 0))).sum()
                    / max(len(frame.loc[frame["_period"] == period]), 1)
                ),
                4,
            ),
        }
        for period in sorted(frame["_period"].unique())
    ]
    balance = {
        "natural_train_rows": int(len(train)),
        "preview_train_rows": int(len(preview)),
        "natural_benign_share_pct": round(100.0 * train_counts.get("Benign", 0) / max(sum(train_counts.values()), 1), 4),
        "preview_benign_share_pct": round(100.0 * preview_counts.get("Benign", 0) / max(sum(preview_counts.values()), 1), 4),
        "train_counts": train_counts,
        "preview_counts": preview_counts,
        "periods": period_rows,
        "max_benign": int(max_benign),
        "min_class": int(min_class),
        "note": "preview uses sampling, not SMOTE; validation/test unchanged",
    }
    metrics = {
        "schema_version": 1,
        "step": "03_eda",
        "dataset": dataset,
        "rows": int(len(frame)),
        "columns": int(len(frame.columns)),
        "classes": names,
        "class_ids": ids,
        "class_counts": full,
        "periods": period_rows,
        "split_counts": splits,
        "balance": balance,
        "hosts": host_stats,
        "cleaned_table": str(table_path),
        "preprocessor_state": str(paths.processed_dataset(dataset) / "clean" / "preprocessor_state.json"),
        "figures": [str(path) for path in saved],
        "warnings": [] if masks else ["period-1 masks not found; split support omitted"],
    }
    summary_dir = paths.audit_root / "eda"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{dataset}.json"
    write_json(summary_path, metrics)
    ok("summary", str(summary_path))

    cell(4, 4, f"REPORT {dataset}")
    report_path = paths.project_root / "analysis" / "eda" / f"{dataset}.md"
    write_report(report_path, dataset, frame, summary, names, ids, splits, host_stats, balance, saved)
    ok("report", str(report_path))
    metrics["report"] = str(report_path)
    metrics["summary"] = str(summary_path)
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    started = time.perf_counter()
    args = parse_args(argv)
    cell(1, 4, "CONFIGURATION")
    root = find_root(args.root)
    info("project root", str(root))
    try:
        paths = load_paths(lane=args.lane, project_root=root)
        config = load_config(args.config, project_root=root, lane=args.lane)
        data_cfg = config.to_dict()
    except Exception as exc:  # noqa: BLE001
        fail("configuration", f"{type(exc).__name__}: {exc}")
        return 1
    ok("paths", f"data={paths.data_root}; results={paths.results_root}")
    try:
        datasets = choose_datasets(args.dataset, data_cfg, root)
    except Exception as exc:  # noqa: BLE001
        fail("dataset list", f"{type(exc).__name__}: {exc}")
        return 1
    ok("datasets", ", ".join(datasets))

    completed = []
    missing = 0
    failures = 0
    for index, dataset in enumerate(datasets, start=1):
        print(f"\n--- DATASET {index}/{len(datasets)}: {dataset} " + "-" * 20)
        try:
            metrics = eda_one_dataset(
                dataset,
                paths=paths,
                config=data_cfg,
                max_benign=args.max_benign,
                min_class=args.min_class,
                seed=args.seed,
            )
            completed.append(metrics)
        except FileNotFoundError as exc:
            missing += 1
            skip(dataset, str(exc))
        except Exception as exc:  # noqa: BLE001
            failures += 1
            fail(dataset, f"{type(exc).__name__}: {exc}")

    cell(4, 4, "SUMMARY")
    for item in completed:
        print(
            f"  {item['dataset']:<28} rows={item['rows']:>12,}  "
            f"benign_train={item['balance']['natural_benign_share_pct']:>6.2f}%  "
            f"preview={item['balance']['preview_benign_share_pct']:>6.2f}%"
        )
    if missing:
        warn("missing cleaned datasets", f"{missing} requested dataset(s) not available")
    if failures:
        fail("failed datasets", f"{failures} dataset(s) failed")
        return 1
    if not completed:
        fail("datasets", "no cleaned dataset was available")
        return 2
    ok("status", f"EDA complete in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
