"""S13 figure generators — every registered figure is written from the table frames."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.analysis import artifacts
from src.analysis import figures as fig
from src.analysis import metrics_cl as mc


def _tensors() -> dict[tuple[str, str, int], mc.RMetricTensor]:
    tensors = {}
    for index, strategy in enumerate(("ft", "replay")):
        matrix = np.full((3, 3), np.nan)
        matrix[0, 0] = 0.8 - 0.05 * index
        matrix[1, 0] = 0.70 - 0.05 * index
        matrix[1, 1] = 0.9
        matrix[2, 0] = 0.60 + 0.02 * index
        matrix[2, 1] = 0.75
        matrix[2, 2] = 0.65
        tensors[(strategy, "m", 42)] = mc.RMetricTensor(matrix, strategy=strategy, model="m", seed=42)
    return tensors


def _frames() -> dict[str, pd.DataFrame]:
    excess = pd.DataFrame(
        {
            "strategy": ["ft", "ft", "replay", "replay"] * 5,
            "drift_primary": np.linspace(0.1, 0.7, 20),
            "null_mean": np.linspace(0.05, 0.2, 20),
        }
    )
    noise_floor = pd.DataFrame(
        {
            "channel": ["model_null", "model_null"],
            "class": ["A", "B"],
            "mean": [0.1, 0.2],
            "q95": [0.3, 0.4],
        }
    )
    aggregated = pd.DataFrame(
        {
            "strategy": ["ft", "ft", "ft", "ft"],
            "explainer": ["IG"] * 4,
            "checkpoint_j": [1, 1, 2, 2],
            "checkpoint_t": [2, 3, 3, 4],
            "excess_mean": [0.2, 0.3, -0.1, 0.05],
        }
    )
    h1 = pd.DataFrame(
        {
            "strategy": ["ft", "replay"],
            "model": ["m", "m"],
            "explainer": ["IG", "IG"],
            "fraction_above_floor": [0.45, 0.1],
            "fraction_ci_lo": [0.35, 0.05],
            "fraction_ci_hi": [0.55, 0.15],
        }
    )
    rq2 = pd.DataFrame(
        {
            "strategy": ["ft", "ft", "replay", "replay"],
            "explainer": ["IG"] * 4,
            "neighbourhood_change": [0.1, 0.4, 0.2, 0.8],
            "excess": [0.0, 0.3, 0.05, 0.5],
        }
    )
    rq4 = pd.DataFrame(
        {
            "variant": ["replay", "ear_a", "ear_b", "replay", "ear_a", "ear_b"],
            "drift_mean": [0.3, 0.22, 0.25, 0.31, 0.23, 0.24],
            "macro_f1": [0.62, 0.615, 0.613, 0.61, 0.605, 0.6],
            "row_kind": ["run"] * 6,
        }
    )
    return {
        "excess": excess,
        "noise_floor": noise_floor,
        "aggregated": aggregated,
        "h1": h1,
        "rq2": rq2,
        "rq4": rq4,
    }


def test_all_figures_are_written(tmp_path: Path) -> None:
    layout = artifacts.ArtifactLayout(tmp_path / "results")
    frames = _frames()
    paths = fig.generate_all_figures(
        tensors=_tensors(),
        excess_frame=frames["excess"],
        noise_floor_table=frames["noise_floor"],
        rq1_aggregated=frames["aggregated"],
        h1_tests=frames["h1"],
        rq2_anchor_level=frames["rq2"],
        rq4_comparison=frames["rq4"],
        layout=layout,
    )
    assert set(paths) == set(artifacts.FIGURE_ARTIFACTS)
    for name, path in paths.items():
        assert Path(path).is_file(), name
        assert Path(path).stat().st_size > 1024, name
        assert Path(path).suffix == ".png"


def test_figure_errors_are_explicit(tmp_path: Path) -> None:
    with pytest.raises(fig.FigureError):
        fig.plot_above_floor_fraction(pd.DataFrame({"wrong": [1]}), tmp_path / "bad.png")
    with pytest.raises(fig.FigureError):
        fig.plot_noise_floor_q95(pd.DataFrame({"channel": ["other"], "class": ["A"], "mean": [0.1], "q95": [0.2]}), tmp_path / "bad.png")
    with pytest.raises(fig.FigureError):
        fig.plot_f1_trajectories_with_drift({}, tmp_path / "bad.png")
