"""docs/05 §13 statistics — hand-computed checks of every primitive used in the paper."""

from __future__ import annotations

import numpy as np
import pytest

from src.analysis import stats as st


def test_holm_correction_hand_computed() -> None:
    # sorted: 0.005, 0.01, 0.03, 0.04 -> 0.02, 0.03, 0.06, 0.06 (monotone step-down)
    adjusted = st.holm_correction([0.01, 0.04, 0.03, 0.005])
    assert adjusted.tolist() == pytest.approx([0.03, 0.06, 0.06, 0.02])
    # single test and NaN handling
    assert st.holm_correction([0.02]).tolist() == pytest.approx([0.02])
    nan_adjusted = st.holm_correction([0.01, np.nan])
    assert nan_adjusted[0] == pytest.approx(0.01)  # NaN excluded from the family, m = 1
    assert np.isnan(nan_adjusted[1])


def test_holm_labelled_preserves_labels() -> None:
    adjusted = st.holm_correction_labelled({"a": 0.01, "b": 0.04})
    assert adjusted["a"] == pytest.approx(0.02)
    assert adjusted["b"] == pytest.approx(0.04)


def test_binomial_test_vs_expected_fraction() -> None:
    strong = st.binomial_test_proportion(20, 100, p0=0.05)
    assert strong["fraction"] == pytest.approx(0.20)
    assert strong["p_binomial"] < 1e-5
    assert strong["p_normal"] < 1e-5
    weak = st.binomial_test_proportion(5, 100, p0=0.05)
    assert weak["p_binomial"] > 0.05
    with pytest.raises(st.StatisticsError):
        st.binomial_test_proportion(3, 0)


def test_cohens_d_and_rank_biserial_hand_computed() -> None:
    # differences [1,2,3]: mean 2, sample sd 1 -> d = 2
    assert st.cohens_d_paired([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]) == pytest.approx(2.0)
    assert st.rank_biserial_from_wilcoxon([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]) == pytest.approx(1.0)
    assert st.wilcoxon_signed_rank([0.0, 0.0], [0.0, 0.0])["p_value"] == 1.0


def test_bootstrap_ci_degenerate_and_reproducible() -> None:
    constant = st.bootstrap_ci([2.0, 2.0, 2.0, 2.0], n_resamples=100, seed=7)
    assert constant.point == pytest.approx(2.0)
    assert constant.lo == pytest.approx(2.0)
    assert constant.hi == pytest.approx(2.0)
    first = st.bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], n_resamples=200, seed=42)
    second = st.bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0], n_resamples=200, seed=42)
    assert first == second
    assert first.lo <= first.point <= first.hi


def test_stratified_bootstrap_resamples_within_stratum() -> None:
    # strata 'a' all 1 -> any within-stratum resample stays 1
    result = st.stratified_bootstrap_ci([1.0, 1.0, 5.0, 5.0], ["a", "a", "b", "b"], n_resamples=100, seed=42)
    assert result.lo == pytest.approx(3.0)
    assert result.hi == pytest.approx(3.0)
    with pytest.raises(st.StatisticsError):
        st.stratified_bootstrap_ci([1.0, 2.0], ["a"])


def test_hierarchical_bootstrap_degenerate_and_ci_contains_point() -> None:
    degenerate = st.hierarchical_bootstrap_ci([[1.0, 1.0], [1.0, 1.0]], n_inner=20, n_outer=50, seed=42)
    assert degenerate.point == pytest.approx(1.0)
    assert degenerate.lo == pytest.approx(1.0)
    assert degenerate.hi == pytest.approx(1.0)
    result = st.hierarchical_bootstrap_ci([[0.50, 0.60], [0.45, 0.55], [0.55, 0.65]], n_inner=40, n_outer=200, seed=42)
    assert result.lo <= result.point <= result.hi


def test_paired_bootstrap_direction_and_p_value() -> None:
    result = st.paired_bootstrap_diff([0.6, 0.7, 0.8, 0.9], [0.5, 0.55, 0.6, 0.65], n_resamples=300, seed=42)
    assert result.diff == pytest.approx(0.175)  # mean of [0.10, 0.15, 0.20, 0.25]
    assert result.lo > 0.0
    assert result.p_value < 0.05
    # stratum argument must not change the point estimate
    stratified = st.paired_bootstrap_diff(
        [0.6, 0.7, 0.8, 0.9], [0.5, 0.55, 0.6, 0.65], stratum=["a", "a", "b", "b"], n_resamples=300, seed=42
    )
    assert stratified.diff == pytest.approx(result.diff)


def test_spearman_and_partial_spearman() -> None:
    assert st.spearman_correlation([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert st.spearman_correlation([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    # x tracks z strongly and y is driven by z: controlling for z should shrink the correlation
    rng = np.random.default_rng(0)
    z = np.arange(1, 61, dtype=float)
    x = z + rng.normal(0.0, 0.5, z.size)
    y = 3.0 * z + 0.1 * x + rng.normal(0.0, 0.5, z.size)
    raw = st.spearman_correlation(x, y)
    partial = st.partial_spearman(x, y, z.reshape(-1, 1))
    assert raw > 0.9
    assert abs(partial) < abs(raw)
    # controls=None == Spearman
    assert st.partial_spearman([1, 2, 3, 4, 5], [2, 1, 4, 3, 5], None) == pytest.approx(
        st.spearman_correlation([1, 2, 3, 4, 5], [2, 1, 4, 3, 5])
    )


def test_permutation_test_directions() -> None:
    x = np.arange(1, 12, dtype=float)
    positive = st.permutation_test(x, x, n_permutations=199, seed=42)
    assert positive.observed == pytest.approx(1.0)
    assert positive.p_value <= 0.01
    negative = st.permutation_test(x, x[::-1], alternative="greater", n_permutations=199, seed=42)
    assert negative.observed == pytest.approx(-1.0)
    assert negative.p_value >= 0.9
    with pytest.raises(ValueError):
        st.permutation_test(x, x, alternative="two_sided_typo")


def test_logistic_lopo_auc_separable() -> None:
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1])
    x = y.astype(float) * 2.0 + np.linspace(-0.1, 0.1, y.size)
    groups = np.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4])
    result = st.logistic_lopo_auc(x.reshape(-1, 1), y, groups)
    assert result.auc == pytest.approx(1.0)
    assert result.n == 12
    assert result.n_pos == 6 and result.n_neg == 6
    assert result.n_folds == 4
    with pytest.raises(st.StatisticsError):
        st.logistic_lopo_auc(x.reshape(-1, 1), np.zeros(12), groups)


def test_bootstrap_auc_and_delta() -> None:
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    perfect = np.array([0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9])
    interval = st.bootstrap_auc(y, perfect, n_resamples=100, seed=42)
    assert interval.point == pytest.approx(1.0)
    assert interval.lo == pytest.approx(1.0)
    delta = st.paired_auc_delta_bootstrap(y, perfect, -perfect, n_resamples=100, seed=42)
    assert delta.delta == pytest.approx(1.0)
    assert delta.lo == pytest.approx(1.0)
    assert delta.p_value < 0.05


def test_standardize_features() -> None:
    scaled, mean, std = st.standardize_features([[1.0], [2.0], [3.0]])
    assert scaled[:, 0].mean() == pytest.approx(0.0)
    assert scaled[:, 0].std(ddof=0) == pytest.approx(1.0)
    reused, _, _ = st.standardize_features([[4.0]], mean=mean, std=std)
    assert reused[0, 0] == pytest.approx((4.0 - 2.0) / np.std([1.0, 2.0, 3.0]))


def test_effect_size_summary_and_majority_rate() -> None:
    summary = st.effect_size_summary([0.6, 0.7, 0.8], [0.5, 0.5, 0.5], paired=True)
    assert summary["mean_diff"] == pytest.approx(0.2)
    assert "cohens_d" in summary and "rank_biserial" in summary
    assert st.majority_rate([0, 0, 1, 1, 1]) == pytest.approx(0.6)
