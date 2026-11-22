"""S13 figure generators — write the ``results/images/s8_*`` … ``s12_*`` artifacts.

Authority
---------
* ``research/lit/claims_evidence_skeleton.md`` fixes the figure prefixes used by
  claims C1–C5 (``results/images/s8_*``, ``s9_*``, ``s10_*``, ``s11_*``);
  ``src/analysis/artifacts.py`` fixes the concrete filenames.
* ``docs/source_documents/Doc 2b`` §S13 lists the required figures: drift distributions
  (real vs null) per strategy; excess drift by period pair; scatter of
  neighbourhood change vs drift; per-class F1 trajectories with drift markers;
  EAR trade-off plot (drift vs macro-F1).

All figures are generated from the same frames as the tables (no hand edits),
use the non-interactive ``Agg`` backend, and are deterministic (no timestamps in
metadata) so a regenerated figure is reproducible from its inputs. Functions
raise :class:`FigureError` instead of silently writing an empty plot.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set before pyplot)

from src.analysis import artifacts  # noqa: E402

DPI = 150
FIG_SIZE = (8.0, 5.0)


class FigureError(ValueError):
    """Raised when a figure cannot be drawn from the given inputs."""


def _save(figure: plt.Figure, path: Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(target, dpi=DPI, metadata={"Software": None})
    plt.close(figure)
    if target.stat().st_size < 1024:
        raise FigureError(f"figure {target} is suspiciously small ({target.stat().st_size} bytes)")
    return target


def _require(frame: pd.DataFrame, columns: Sequence[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise FigureError(f"{name}: missing columns {missing}")


def plot_drift_real_vs_null(excess_frame: pd.DataFrame, path: str | Path) -> Path:
    """Drift distributions: real updates vs the null-update floor (S8/S9 figure)."""
    _require(excess_frame, ["strategy", "drift_primary", "null_mean"], "drift_real_vs_null")
    frame = excess_frame.dropna(subset=["drift_primary", "null_mean"])
    if frame.empty:
        raise FigureError("drift_real_vs_null: no rows with both real and null drift")
    strategies = list(dict.fromkeys(frame["strategy"].astype(str)))
    figure, axes = plt.subplots(1, len(strategies), figsize=(4.2 * len(strategies), 4.0), squeeze=False)
    bins = np.linspace(0.0, 1.0, 31)
    for axis, strategy in zip(axes[0], strategies):
        subset = frame[frame["strategy"].astype(str) == strategy]
        axis.hist(subset["drift_primary"].to_numpy(dtype=float), bins=bins, alpha=0.65, density=True, label="real drift", color="#1f77b4")
        axis.hist(subset["null_mean"].to_numpy(dtype=float), bins=bins, alpha=0.65, density=True, label="null-update mean", color="#ff7f0e")
        axis.set_title(str(strategy))
        axis.set_xlabel("drift (S8 primary metric)")
        axis.legend(fontsize=8)
    axes[0][0].set_ylabel("density")
    return _save(figure, Path(path))


def plot_noise_floor_q95(noise_floor_table: pd.DataFrame, path: str | Path, *, channel: str = "model_null") -> Path:
    """Per-class noise floor: null mean vs q95 threshold (S8 figure)."""
    _require(noise_floor_table, ["class", "mean", "q95", "channel"], "noise_floor_q95")
    frame = noise_floor_table[noise_floor_table["channel"] == channel].dropna(subset=["class", "mean", "q95"])
    if frame.empty:
        raise FigureError(f"noise_floor_q95: no rows for channel {channel!r}")
    frame = frame.groupby("class", as_index=False).agg(mean=("mean", "mean"), q95=("q95", "max")).sort_values("q95", ascending=False)
    figure, axis = plt.subplots(figsize=(max(8.0, 0.7 * len(frame)), 4.5))
    positions = np.arange(len(frame))
    axis.bar(positions - 0.2, frame["mean"], width=0.4, label="null mean", color="#9ecae1")
    axis.bar(positions + 0.2, frame["q95"], width=0.4, label="q95 threshold", color="#3182bd")
    axis.set_xticks(positions)
    axis.set_xticklabels(frame["class"], rotation=35, ha="right")
    axis.set_ylabel("drift")
    axis.set_title(f"Noise floor by class ({channel})")
    axis.legend()
    return _save(figure, Path(path))


def plot_excess_by_period_pair(rq1_aggregated: pd.DataFrame, path: str | Path, *, strategy: str = "ft", explainer: str = "IG") -> Path:
    """Mean excess drift over period pairs (Doc 2b §S13 "excess drift by period pair")."""
    _require(rq1_aggregated, ["strategy", "explainer", "checkpoint_j", "checkpoint_t", "excess_mean"], "excess_by_period_pair")
    frame = rq1_aggregated[(rq1_aggregated["strategy"] == strategy) & (rq1_aggregated["explainer"] == explainer)]
    if frame.empty:
        raise FigureError(f"excess_by_period_pair: no rows for strategy={strategy!r} explainer={explainer!r}")
    frame = frame.groupby(["checkpoint_j", "checkpoint_t"], as_index=False)["excess_mean"].mean()
    pivot = frame.pivot(index="checkpoint_j", columns="checkpoint_t", values="excess_mean")
    figure, axis = plt.subplots(figsize=(6.5, 5.0))
    image = axis.imshow(pivot.to_numpy(dtype=float), origin="lower", cmap="RdBu_r", vmin=-0.5, vmax=0.5)
    axis.set_xticks(range(pivot.shape[1]))
    axis.set_xticklabels([f"P{int(value)}" for value in pivot.columns])
    axis.set_yticks(range(pivot.shape[0]))
    axis.set_yticklabels([f"P{int(value)}" for value in pivot.index])
    axis.set_xlabel("update checkpoint $M_t$")
    axis.set_ylabel("anchor period $P_j$")
    axis.set_title(f"Mean excess drift, {strategy}/{explainer}")
    figure.colorbar(image, ax=axis, label="excess drift")
    return _save(figure, Path(path))


def plot_above_floor_fraction(h1_tests: pd.DataFrame, path: str | Path, *, expected: float = 0.05) -> Path:
    """Fraction of anchors above the null floor vs the 5 % expected (H1 figure)."""
    _require(h1_tests, ["fraction_above_floor", "fraction_ci_lo", "fraction_ci_hi"], "above_floor_fraction")
    if h1_tests.empty:
        raise FigureError("above_floor_fraction: empty H1 table")
    frame = h1_tests.copy()
    frame["label"] = frame.apply(lambda row: f"{row['strategy']}/{row['model']}/{row['explainer']}", axis=1)
    positions = np.arange(len(frame))
    values = frame["fraction_above_floor"].to_numpy(dtype=float)
    errors = np.vstack(
        [
            values - frame["fraction_ci_lo"].to_numpy(dtype=float),
            frame["fraction_ci_hi"].to_numpy(dtype=float) - values,
        ]
    )
    figure, axis = plt.subplots(figsize=(max(8.0, 0.8 * len(frame)), 4.5))
    axis.bar(positions, values, yerr=np.clip(errors, 0.0, None), capsize=3, color="#2ca02c")
    axis.axhline(expected, color="black", linestyle="--", linewidth=1.0, label=f"expected {expected:.0%}")
    axis.set_xticks(positions)
    axis.set_xticklabels(frame["label"], rotation=35, ha="right")
    axis.set_ylabel("fraction of anchors above q95")
    axis.set_title("H1: fraction above the null-update floor (95 % bootstrap CI)")
    axis.legend()
    return _save(figure, Path(path))


def plot_neighbourhood_vs_drift(rq2_anchor_level: pd.DataFrame, path: str | Path, *, explainer: str = "IG", max_points: int = 3000) -> Path:
    """Scatter of per-anchor neighbourhood change vs excess drift (RQ2 figure)."""
    _require(rq2_anchor_level, ["neighbourhood_change", "excess", "strategy"], "neighbourhood_vs_drift")
    frame = rq2_anchor_level.dropna(subset=["neighbourhood_change", "excess"])
    if "explainer" in frame.columns:
        frame = frame[frame["explainer"] == explainer]
    if frame.empty:
        raise FigureError("neighbourhood_vs_drift: no rows")
    if len(frame) > max_points:
        frame = frame.sample(n=max_points, random_state=42)
    figure, axis = plt.subplots(figsize=FIG_SIZE)
    for strategy, subset in frame.groupby("strategy"):
        axis.scatter(
            subset["neighbourhood_change"].to_numpy(dtype=float),
            subset["excess"].to_numpy(dtype=float),
            s=8,
            alpha=0.5,
            label=str(strategy),
        )
    x = frame["neighbourhood_change"].to_numpy(dtype=float)
    y = frame["excess"].to_numpy(dtype=float)
    if np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        line_x = np.linspace(x.min(), x.max(), 50)
        axis.plot(line_x, intercept + slope * line_x, color="black", linewidth=1.2, label="linear fit (visual guide)")
    axis.set_xlabel("neighbourhood change $1 - J(H_j, H_t)$")
    axis.set_ylabel("excess drift")
    axis.set_title("RQ2: structural change vs excess drift")
    axis.legend(fontsize=8)
    return _save(figure, Path(path))


def plot_f1_trajectories_with_drift(
    tensors: Mapping[Any, Any],
    path: str | Path,
    *,
    excess_by_checkpoint: pd.DataFrame | None = None,
) -> Path:
    """Per-strategy F1 trajectories with drift markers (Doc 2b §S13).

    Left panel: mean test macro-F1 at each checkpoint (mean over evaluated
    periods). Right panel: mean excess drift per checkpoint for the primary
    explainer (markers flag checkpoints whose above-floor fraction exceeds the
    5 % expectation). ``tensors`` maps run keys to
    :class:`analysis.metrics_cl.RMetricTensor`.
    """
    if not tensors:
        raise FigureError("f1_trajectories: no tensors supplied")
    means: dict[str, np.ndarray] = {}
    for key, tensor in tensors.items():
        strategy = tensor.strategy
        values = np.array([np.nanmean(tensor.R[index, : index + 1]) for index in range(tensor.n_periods)])
        means.setdefault(strategy, []).append(values)  # type: ignore[arg-type]
    figure, (left, right) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for strategy, arrays in sorted(means.items()):
        stacked = np.vstack(arrays)
        average = np.nanmean(stacked, axis=0)
        checkpoints = np.arange(1, average.size + 1)
        left.plot(checkpoints, average, marker="o", label=str(strategy))
    left.set_xlabel("checkpoint $M_i$")
    left.set_ylabel("mean macro-F1 over seen periods")
    left.set_title("F1 trajectories")
    left.legend(fontsize=8)
    if excess_by_checkpoint is not None and len(excess_by_checkpoint) > 0:
        frame = excess_by_checkpoint
        for strategy, subset in frame.groupby("strategy"):
            subset = subset.groupby("checkpoint_t", as_index=False)["excess_mean"].mean()
            right.plot(subset["checkpoint_t"], subset["excess_mean"], marker="s", label=str(strategy))
        right.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
        right.set_xlabel("checkpoint $M_t$")
        right.set_ylabel("mean excess drift")
        right.set_title("Drift markers (excess over null floor)")
        right.legend(fontsize=8)
    else:  # pragma: no cover - optional panel
        right.text(0.5, 0.5, "no drift table supplied", ha="center", va="center")
    return _save(figure, Path(path))


def plot_ear_tradeoff(rq4_comparison: pd.DataFrame, path: str | Path) -> Path:
    """EAR trade-off: drift vs macro-F1 per variant (RQ4 figure)."""
    _require(rq4_comparison, ["variant", "drift_mean", "macro_f1"], "ear_tradeoff")
    frame = rq4_comparison[rq4_comparison.get("row_kind", "run") == "run"].dropna(subset=["drift_mean", "macro_f1"])
    if frame.empty:
        raise FigureError("ear_tradeoff: no per-run rows")
    figure, axis = plt.subplots(figsize=FIG_SIZE)
    for variant, subset in frame.groupby("variant"):
        axis.scatter(subset["macro_f1"], subset["drift_mean"], s=28, alpha=0.75, label=str(variant))
    axis.set_xlabel("macro-F1")
    axis.set_ylabel("mean drift (primary metric)")
    axis.set_title("RQ4 / H4: EAR vs replay trade-off (lower drift is better)")
    axis.legend(fontsize=8)
    return _save(figure, Path(path))


def generate_all_figures(
    *,
    tensors: Mapping[Any, Any],
    excess_frame: pd.DataFrame,
    noise_floor_table: pd.DataFrame,
    rq1_aggregated: pd.DataFrame,
    h1_tests: pd.DataFrame,
    rq2_anchor_level: pd.DataFrame,
    rq4_comparison: pd.DataFrame,
    layout: artifacts.ArtifactLayout,
) -> dict[str, Path]:
    """Write every registered S13 figure from the assembled frames."""
    layout.ensure()
    paths: dict[str, Path] = {}
    paths["s8_drift_real_vs_null"] = plot_drift_real_vs_null(excess_frame, layout.figure("s8_drift_real_vs_null"))
    paths["s8_noise_floor_q95"] = plot_noise_floor_q95(noise_floor_table, layout.figure("s8_noise_floor_q95"))
    paths["s9_excess_by_period_pair"] = plot_excess_by_period_pair(rq1_aggregated, layout.figure("s9_excess_by_period_pair"))
    paths["s9_above_floor_fraction"] = plot_above_floor_fraction(h1_tests, layout.figure("s9_above_floor_fraction"))
    paths["s10_neighbourhood_vs_drift"] = plot_neighbourhood_vs_drift(rq2_anchor_level, layout.figure("s10_neighbourhood_vs_drift"))
    paths["s11_f1_trajectories_drift"] = plot_f1_trajectories_with_drift(
        tensors, layout.figure("s11_f1_trajectories_drift"), excess_by_checkpoint=rq1_aggregated
    )
    paths["s12_ear_drift_vs_f1"] = plot_ear_tradeoff(rq4_comparison, layout.figure("s12_ear_drift_vs_f1"))
    return paths


__all__ = [
    "FigureError",
    "generate_all_figures",
    "plot_above_floor_fraction",
    "plot_drift_real_vs_null",
    "plot_ear_tradeoff",
    "plot_excess_by_period_pair",
    "plot_f1_trajectories_with_drift",
    "plot_neighbourhood_vs_drift",
    "plot_noise_floor_q95",
]
