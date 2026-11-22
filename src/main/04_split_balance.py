#!/usr/bin/env python
"""Step 04 - freeze the temporal split and prepare graph-safe class balance.

This stage does three things, in order:

    1. build the grouped 70/10/20 split for every period;
    2. compare period 1 with the existing frozen mask (never silently replace it);
    3. save per-period training indices and class weights for the trainer.

It deliberately does NOT create synthetic rows.  SMOTE is unsafe for graph
edges because a synthetic row has no real source/destination host or context.
The graph-safe balance is class weights, plus an optional benign cap that
only removes benign rows from the training partition.

Outputs
-------
datasets/splits/<dataset>/<strategy>/period_{1..5}_masks.npz
datasets/splits/<dataset>/<strategy>/split_manifest.json
datasets/processed/<dataset>/balance/period_{1..5}_train_indices.npz
datasets/processed/<dataset>/balance/balance_summary.json
results/text/data_audit/split/<dataset>.json
results/images/split/<dataset>/fig01..fig04.png

Usage
-----
    python src/main/04_split_balance.py
    python src/main/04_split_balance.py --dataset NF-UNSW-NB15-v3
    python src/main/04_split_balance.py --benign-cap 150000
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
    from src.data.splits import (
        build_splits,
        freeze_split_masks,
        load_split_masks,
        write_split_masks,
    )
    from src.utils.config import load_config
    from src.utils.manifest import sha256_file, write_json
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
BLUE = "#2f6f9f"
ORANGE = "#b45f4d"
GREY = "#6b7c93"


# ---------------------------------------------------------------------------
# Small output helpers (same style as steps 01-03)
# ---------------------------------------------------------------------------
def cell(number: int, total: int, title: str) -> None:
    print()
    print("=" * 78)
    print(f"CELL {number:02d}/{total:02d} | {title}")
    print("=" * 78)


def info(text: str) -> None:
    print(f"[INFO] {text}")


def ok(text: str, detail: str = "") -> None:
    print(f"[OK  ] {text}" + (f" {detail}" if detail else ""))


def warn(text: str, detail: str = "") -> None:
    print(f"[WARN] {text}" + (f" {detail}" if detail else ""))


def fail(text: str, detail: str = "") -> None:
    print(f"[FAIL] {text}" + (f" {detail}" if detail else ""))


# ---------------------------------------------------------------------------
# CLI and dataset selection
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the split and save graph-safe class-balance metadata."
    )
    parser.add_argument("--dataset", default="all", help="comma-separated names or 'all'")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="audit/split config YAML")
    parser.add_argument("--root", default=None, help="project root (default: auto-detect)")
    parser.add_argument("--lane", default=None, help="path lane (cpu, kaggle, ...)")
    parser.add_argument(
        "--benign-cap",
        type=int,
        default=0,
        help="keep at most this many benign training rows per period (0 = no cap)",
    )
    parser.add_argument(
        "--balance-method",
        choices=("class_weight", "none"),
        default="class_weight",
        help="class_weight (default) or none (control)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild masks that already exist and differ from the new split",
    )
    parser.add_argument("--strict", action="store_true", help="missing cleaned data fails")
    return parser.parse_args(argv)


def find_root(given: str | None) -> Path:
    if given:
        return Path(given).expanduser().resolve()
    for candidate in (Path.cwd(), *Path.cwd().parents):
        if (candidate / "src").is_dir() and (candidate / "configs").is_dir():
            return candidate
    return ROOT


def choose_datasets(requested: str, root: Path) -> list[str]:
    names = resolve_dataset_names(requested, project_root=root)
    return list(names)


# ---------------------------------------------------------------------------
# Load inputs
# ---------------------------------------------------------------------------
def load_cleaned(dataset: str, paths: Any) -> tuple[pd.DataFrame, Path]:
    clean_dir = paths.processed_dataset(dataset) / "clean"
    for name in ("cleaned.parquet", "cleaned.pkl"):
        path = clean_dir / name
        if path.is_file():
            frame = pd.read_parquet(path) if name.endswith(".parquet") else pd.read_pickle(path)
            return frame, path
    raise FileNotFoundError(
        f"no cleaned table under {clean_dir}; run src/main/02_clean_preprocess.py first"
    )


def split_settings(config: Mapping[str, Any]) -> tuple[dict[str, float], int, dict[str, Any]]:
    split_cfg = config.get("splits", {})
    fractions = {
        "train": float(split_cfg.get("fractions", {}).get("train", 0.70)),
        "val": float(split_cfg.get("fractions", {}).get("val", 0.10)),
        "test": float(split_cfg.get("fractions", {}).get("test", 0.20)),
    }
    seed = int(split_cfg.get("seed", 42))
    pair_cfg = split_cfg.get("host_pair_disjoint", {})
    return fractions, seed, pair_cfg


# ---------------------------------------------------------------------------
# Split validation
# ---------------------------------------------------------------------------
def mask_hash(indices: np.ndarray) -> str:
    """Small local hash so the summary records exactly which rows were used."""
    import hashlib

    array = np.sort(np.asarray(indices, dtype="<i8"))
    return hashlib.sha256(array.tobytes()).hexdigest()


def validate_mask(
    *,
    dataset: str,
    period: int,
    masks: Mapping[str, np.ndarray],
    period_positions: np.ndarray,
) -> dict[str, Any]:
    """Check coverage, overlap and index bounds for one period."""
    union = np.sort(np.concatenate([masks["train"], masks["val"], masks["test"]]))
    expected = np.sort(np.asarray(period_positions, dtype="int64"))
    if len(union) != len(expected) or not np.array_equal(union, expected):
        raise ValueError(f"{dataset} period {period}: split masks do not cover every period row exactly once")
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if np.intersect1d(masks[left], masks[right]).size:
            raise ValueError(f"{dataset} period {period}: {left}/{right} masks overlap")
    return {
        "rows": int(len(period_positions)),
        "train": int(len(masks["train"])),
        "val": int(len(masks["val"])),
        "test": int(len(masks["test"])),
        "train_sha256": mask_hash(masks["train"]),
        "val_sha256": mask_hash(masks["val"]),
        "test_sha256": mask_hash(masks["test"]),
    }


def check_existing_masks(
    *,
    dataset: str,
    split_dir: Path,
    masks: Mapping[int, Mapping[str, np.ndarray]],
    force: bool,
) -> list[str]:
    """Compare existing frozen masks with the newly built ones.

    Period 1 is the signed reference.  A difference is a hard error unless the
    operator explicitly asks to rebuild; a silent replacement would invalidate
    the pre-registered split.
    """
    warnings: list[str] = []
    for period in sorted(masks):
        path = split_dir / f"period_{period + 1}_masks.npz"
        if not path.is_file():
            continue
        old = load_split_masks(split_dir, period + 1)
        same = all(
            np.array_equal(np.sort(old[name]), np.sort(masks[period][name]))
            for name in ("train", "val", "test")
        )
        if same:
            ok(f"period {period + 1} frozen mask matches", str(path))
            continue
        message = f"{dataset} period {period + 1}: existing frozen mask differs from rebuilt split"
        if period == 0 and not force:
            raise ValueError(
                message
                + "; refusing to replace the signed period-1 split (use --force only if intentional)"
            )
        if force:
            warnings.append(message + "; rebuilt because --force was given")
            warn(message, "rebuilt with --force")
        else:
            raise ValueError(message + "; use --force only if a rebuild is intentional")
    return warnings


# ---------------------------------------------------------------------------
# Split creation
# ---------------------------------------------------------------------------
def build_dataset_splits(
    *,
    dataset: str,
    frame: pd.DataFrame,
    config: Mapping[str, Any],
    paths: Any,
    force: bool,
) -> tuple[dict[int, Mapping[str, np.ndarray]], dict[str, Any], list[str]]:
    fractions, seed, pair_cfg = split_settings(config)
    strategy = str(config.get("periods", {}).get("strategy", "session_aware_1_4"))
    split_dir = paths.dataset_splits(dataset) / strategy
    split_dir.mkdir(parents=True, exist_ok=True)

    result = build_splits(
        frame,
        fractions=fractions,
        seed=seed,
        mode="grouped",
        min_groups_per_split=int(pair_cfg.get("min_groups_per_split", 10)),
        min_share=float(pair_cfg.get("min_share", 0.05)),
    )
    if not result.stats.get("feasible", True):
        reasons = "; ".join(result.stats.get("infeasible_reasons", []))
        raise ValueError(f"{dataset}: split is not feasible: {reasons}")

    masks = {int(period): result.masks[int(period)] for period in sorted(result.masks)}
    warnings = check_existing_masks(
        dataset=dataset, split_dir=split_dir, masks=masks, force=force
    )

    for period, period_masks in masks.items():
        period_positions = np.flatnonzero(frame["_period"].to_numpy(dtype="int64") == period)
        validate_mask(
            dataset=dataset,
            period=period + 1,
            masks=period_masks,
            period_positions=period_positions,
        )

    write_split_masks(result, split_dir, dataset=dataset)
    manifest = freeze_split_masks(result, frame, dataset=dataset)
    manifest_path = split_dir / "split_manifest.json"
    write_json(manifest_path, manifest)

    summary = {
        "strategy": strategy,
        "directory": str(split_dir),
        "manifest": str(manifest_path),
        "frozen_hash": manifest["frozen_hash"],
        "mode": result.mode,
        "seed": result.seed,
        "fractions": result.fractions,
        "periods": {
            str(period + 1): {
                "rows": int(len(frame)),
                "train": int(len(masks[period]["train"])),
                "val": int(len(masks[period]["val"])),
                "test": int(len(masks[period]["test"])),
                "train_sha256": mask_hash(masks[period]["train"]),
            }
            for period in sorted(masks)
        },
    }
    return masks, summary, warnings


# ---------------------------------------------------------------------------
# Balance metadata
# ---------------------------------------------------------------------------
def class_weights(
    train_ids: np.ndarray,
    class_names: list[str],
    *,
    method: str,
) -> tuple[dict[str, float], dict[str, int]]:
    """Balanced class weights, normalized so the mean training weight is 1."""
    counts_array = np.bincount(train_ids, minlength=len(class_names))
    counts = {name: int(counts_array[index]) for index, name in enumerate(class_names)}
    n_rows = int(len(train_ids))
    if method == "none":
        return {name: 1.0 for name in class_names}, counts

    raw = {}
    for index, name in enumerate(class_names):
        count = int(counts_array[index])
        raw[name] = (n_rows / (len(class_names) * count)) if count > 0 else 0.0
    mean_weight = sum(raw[name] * counts[name] for name in class_names) / max(n_rows, 1)
    weights = {
        name: round(raw[name] / mean_weight, 6) if mean_weight > 0 else 1.0
        for name in class_names
    }
    return weights, counts


def select_training_rows(
    *,
    dataset: str,
    period: int,
    train_positions: np.ndarray,
    class_ids: np.ndarray,
    benign_id: int | None,
    benign_cap: int,
    seed: int,
) -> tuple[np.ndarray, int, bool]:
    """Optionally cap benign rows; return selected full-frame positions."""
    if benign_cap <= 0 or benign_id is None:
        return train_positions, len(train_positions), False
    benign_positions = train_positions[class_ids[train_positions] == benign_id]
    if len(benign_positions) <= benign_cap:
        return train_positions, len(benign_positions), False
    other_positions = train_positions[class_ids[train_positions] != benign_id]
    rng = np.random.default_rng(seed + period + 10007)
    kept_benign = rng.choice(benign_positions, size=benign_cap, replace=False)
    selected = np.sort(np.concatenate([other_positions, kept_benign]))
    return selected, int(len(kept_benign)), True


def build_balance(
    *,
    dataset: str,
    frame: pd.DataFrame,
    masks: Mapping[int, Mapping[str, np.ndarray]],
    class_names: list[str],
    method: str,
    benign_cap: int,
    seed: int,
    out_dir: Path,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    class_ids = frame["_class_id"].to_numpy(dtype="int64")
    benign_id = class_names.index("Benign") if "Benign" in class_names else None

    periods: dict[str, Any] = {}
    for period in sorted(masks):
        train_positions = np.asarray(masks[period]["train"], dtype="int64")
        weights, counts = class_weights(
            class_ids[train_positions], class_names, method=method
        )
        selected, kept_benign, capped = select_training_rows(
            dataset=dataset,
            period=period,
            train_positions=train_positions,
            class_ids=class_ids,
            benign_id=benign_id,
            benign_cap=benign_cap,
            seed=seed,
        )
        selected_counts = {
            name: int((class_ids[selected] == index).sum())
            for index, name in enumerate(class_names)
        }
        index_path = out_dir / f"period_{period + 1}_train_indices.npz"
        np.savez_compressed(index_path, indices=selected.astype("int64"))
        periods[str(period + 1)] = {
            "train_rows": int(len(train_positions)),
            "selected_rows": int(len(selected)),
            "benign_rows_before": int(counts.get("Benign", 0)),
            "benign_rows_after": int(kept_benign if capped else counts.get("Benign", 0)),
            "benign_capped": bool(capped),
            "class_counts": counts,
            "selected_class_counts": selected_counts,
            "class_weights": weights,
            "indices_path": str(index_path),
            "indices_sha256": sha256_file(index_path),
        }
        ok(
            f"period {period + 1} balance",
            f"{len(train_positions):,} train rows -> {len(selected):,} selected",
        )

    summary = {
        "schema_version": 1,
        "dataset": dataset,
        "method": method,
        "benign_cap": int(benign_cap),
        "periods": periods,
    }
    path = out_dir / "balance_summary.json"
    write_json(path, summary)
    return {"summary": summary, "summary_path": str(path)}


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def _style(ax: Any) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d9dde3", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)


def make_figures(
    *,
    dataset: str,
    frame: pd.DataFrame,
    masks: Mapping[int, Mapping[str, np.ndarray]],
    class_names: list[str],
    balance: Mapping[str, Any],
    paths: Any,
) -> list[Path]:
    out_dir = paths.figures_root / "split" / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    periods = sorted(masks)
    class_ids = frame["_class_id"].to_numpy(dtype="int64")

    def save(fig: Any, name: str) -> None:
        path = out_dir / name
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved.append(path)

    # 1. class support in the first period's split
    first = periods[0]
    fig, ax = plt.subplots(figsize=(9.6, 5.2))
    x = np.arange(len(class_names))
    width = 0.27
    for offset, split, color in ((-1, "train", BLUE), (0, "val", ORANGE), (1, "test", GREY)):
        counts = [
            int((class_ids[masks[first][split]] == index).sum())
            for index in range(len(class_names))
        ]
        ax.bar(x + offset * width, counts, width=width, label=split, color=color)
    ax.set_xticks(x, class_names, rotation=35, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("rows (log scale)")
    ax.set_title(f"{dataset}: class support in period {first + 1} split")
    ax.legend(frameon=False)
    _style(ax)
    save(fig, "fig01_split_support.png")

    # 2. class weights by period and class (heatmap)
    weights = np.array(
        [
            [float(balance["periods"][str(period + 1)]["class_weights"][name]) for name in class_names]
            for period in periods
        ]
    )
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    im = ax.imshow(weights, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(class_names)), class_names, rotation=35, ha="right")
    ax.set_yticks(range(len(periods)), [f"P{period + 1}" for period in periods])
    ax.set_title(f"{dataset}: class weights per period (mean training weight = 1)")
    for row in range(weights.shape[0]):
        for col in range(weights.shape[1]):
            ax.text(col, row, f"{weights[row, col]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.8, label="weight")
    save(fig, "fig02_class_weights.png")

    # 3. rows kept after the benign cap (or after no cap)
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    before = [int(balance["periods"][str(period + 1)]["train_rows"]) for period in periods]
    after = [int(balance["periods"][str(period + 1)]["selected_rows"]) for period in periods]
    x = np.arange(len(periods))
    ax.bar(x - 0.2, before, width=0.4, label="natural train", color=GREY)
    ax.bar(x + 0.2, after, width=0.4, label="selected train", color=BLUE)
    for index in x:
        ax.text(index - 0.2, before[index], f"{before[index]:,}", ha="center", va="bottom", fontsize=7)
        ax.text(index + 0.2, after[index], f"{after[index]:,}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, [f"P{period + 1}" for period in periods])
    ax.set_ylabel("rows")
    ax.set_title(f"{dataset}: benign cap effect (cap={balance['benign_cap']:,})")
    ax.legend(frameon=False)
    _style(ax)
    save(fig, "fig03_benign_cap.png")

    # 4. split proportions by period
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    train_share = []
    val_share = []
    test_share = []
    for period in periods:
        total = sum(len(masks[period][name]) for name in ("train", "val", "test"))
        train_share.append(100.0 * len(masks[period]["train"]) / total)
        val_share.append(100.0 * len(masks[period]["val"]) / total)
        test_share.append(100.0 * len(masks[period]["test"]) / total)
    x = np.arange(len(periods))
    bottom = np.zeros(len(periods))
    for values, label, color in (
        (train_share, "train", BLUE),
        (val_share, "val", ORANGE),
        (test_share, "test", GREY),
    ):
        ax.bar(x, values, bottom=bottom, label=label, color=color)
        bottom += np.array(values)
    ax.set_xticks(x, [f"P{period + 1}" for period in periods])
    ax.set_ylabel("% of period rows")
    ax.set_ylim(0, 100)
    ax.set_title(f"{dataset}: split proportions by period")
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    _style(ax)
    save(fig, "fig04_split_proportions.png")
    return saved


# ---------------------------------------------------------------------------
# One dataset
# ---------------------------------------------------------------------------
def process_dataset(
    *,
    dataset: str,
    root: Path,
    config: Mapping[str, Any],
    paths: Any,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    info(f"dataset {dataset}")
    try:
        frame, cleaned_path = load_cleaned(dataset, paths)
    except FileNotFoundError as exc:
        if args.strict:
            raise
        warn(str(exc), "skipping")
        return None

    spec = load_dataset_spec(dataset, project_root=root)
    class_names = list(spec.class_order or ())
    if not class_names:
        raise ValueError(f"{dataset}: config has no class_order; cannot build class weights")
    if "_class_id" not in frame.columns or "_period" not in frame.columns:
        raise ValueError(f"{dataset}: cleaned table lacks _class_id/_period; rerun step 02")
    ok("cleaned table", f"{cleaned_path} ({len(frame):,} rows x {len(frame.columns)} columns)")
    ok("classes", ", ".join(class_names))

    masks, split_summary, warnings = build_dataset_splits(
        dataset=dataset,
        frame=frame,
        config=config,
        paths=paths,
        force=args.force,
    )
    balance_dir = paths.processed_dataset(dataset) / "balance"
    balance = build_balance(
        dataset=dataset,
        frame=frame,
        masks=masks,
        class_names=class_names,
        method=args.balance_method,
        benign_cap=args.benign_cap,
        seed=int(config.get("splits", {}).get("seed", 42)),
        out_dir=balance_dir,
    )
    figures = make_figures(
        dataset=dataset,
        frame=frame,
        masks=masks,
        class_names=class_names,
        balance=balance["summary"],
        paths=paths,
    )
    ok("figures", f"{len(figures)} saved under {figures[0].parent}")

    summary = {
        "schema_version": 1,
        "step": "04_split_balance",
        "dataset": dataset,
        "status": "ok",
        "input": {
            "cleaned_path": str(cleaned_path),
            "cleaned_sha256": sha256_file(cleaned_path),
            "rows": int(len(frame)),
        },
        "config": {
            "benign_cap": int(args.benign_cap),
            "balance_method": args.balance_method,
        },
        "split": split_summary,
        "balance": balance["summary"],
        "figures": [str(path) for path in figures],
        "warnings": list(warnings),
    }
    summary_dir = paths.audit_root / "split"
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{dataset}.json"
    write_json(summary_path, summary)
    ok("summary", str(summary_path))
    return {"dataset": dataset, "summary": summary}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    started = time.perf_counter()
    args = parse_args(argv)
    root = find_root(args.root)

    cell(1, 4, "CONFIGURATION")
    info(f"project root {root}")
    if args.benign_cap < 0:
        fail("--benign-cap must be >= 0")
        return 1
    config = load_config(args.config, project_root=root, lane=args.lane)
    paths = load_paths(lane=args.lane, project_root=root)
    names = choose_datasets(args.dataset, root)
    ok("datasets", ", ".join(names))
    ok("balance", f"{args.balance_method}; benign cap={args.benign_cap:,}")

    results = []
    for index, dataset in enumerate(names, start=1):
        cell(2, 4, f"SPLIT + BALANCE {index}/{len(names)}: {dataset}")
        try:
            result = process_dataset(
                dataset=dataset,
                root=root,
                config=config,
                paths=paths,
                args=args,
            )
            if result is not None:
                results.append(result)
        except Exception as exc:
            fail(dataset, str(exc))
            return 1

    cell(3, 4, "SUMMARY")
    if not results:
        warn("no dataset processed", "missing cleaned tables; run step 02")
        return 2 if args.strict else 0
    for result in results:
        periods = result["summary"]["balance"]["periods"]
        first = periods["1"]
        ok(
            result["dataset"],
            f"P1 train {first['train_rows']:,} -> selected {first['selected_rows']:,}; "
            f"frozen {result['summary']['split']['frozen_hash'][:12]}",
        )
    elapsed = time.perf_counter() - started
    print()
    print(f"[OK  ] status split+balance complete {len(results)} dataset(s) in {elapsed:.1f}s")
    print()
    print("Next: build per-period graph snapshots from the cleaned tables and use")
    print("the saved train indices + class weights in training.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
