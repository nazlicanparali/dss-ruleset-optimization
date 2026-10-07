"""Tests for stats_utils against hand-computed values or scipy."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dss_ruleopt.stats_utils import (
    chi2_or_monte_carlo,
    cochran_armitage_trend,
    cramers_v,
    fisher_odds_ratio,
    fit_logistic_irls,
    lift_metrics,
    mantel_haenszel,
    wilson_ci,
)


def test_wilson_ci_matches_known_reference():
    # 8/20, computed by hand with z = 1.959964
    p_hat, lo, hi = wilson_ci(8, 20)
    assert p_hat == pytest.approx(0.4)
    assert lo == pytest.approx(0.21881, abs=1e-4)
    assert hi == pytest.approx(0.61342, abs=1e-4)


def test_wilson_ci_handles_zero_n():
    p_hat, lo, hi = wilson_ci(0, 0)
    assert np.isnan(p_hat)


def test_cramers_v_perfect_association_is_one():
    table = np.array([[100, 0], [0, 100]])
    assert cramers_v(table) == pytest.approx(1.0, abs=1e-9)


def test_cramers_v_independence_is_zero():
    table = np.array([[50, 50], [50, 50]])
    assert cramers_v(table) == pytest.approx(0.0, abs=1e-9)


def test_chi2_or_monte_carlo_falls_back_on_sparse_table():
    sparse_table = np.array([[2, 1], [1, 3]])
    result = chi2_or_monte_carlo(sparse_table, min_expected_count=5, n_simulations=2000, random_state=1)
    assert result["method"] == "monte_carlo"
    assert 0.0 <= result["p_value"] <= 1.0


def test_chi2_uses_asymptotic_when_cells_are_large_enough():
    table = np.array([[500, 400], [420, 480]])
    result = chi2_or_monte_carlo(table, min_expected_count=5)
    assert result["method"] == "chi2"


def test_cochran_armitage_detects_increasing_trend():
    # 5% -> 15% -> 30%
    counts_bad = [50, 150, 300]
    counts_total = [1000, 1000, 1000]
    result = cochran_armitage_trend(counts_bad, counts_total)
    assert result["direction"] == "increasing"
    assert result["p_value"] < 0.001


def test_cochran_armitage_detects_reversed_trend():
    # green is the riskiest -> wrongly ordered rule
    counts_bad = [300, 150, 50]
    counts_total = [1000, 1000, 1000]
    result = cochran_armitage_trend(counts_bad, counts_total)
    assert result["direction"] == "decreasing"


def test_logistic_irls_recovers_known_odds_ratio():
    # simulate two groups with OR = 3
    rng = np.random.default_rng(0)
    n_group0, n_group1 = 5000, 5000
    p0 = 0.10
    true_or = 3.0
    p1 = (true_or * p0 / (1 - p0)) / (1 + true_or * p0 / (1 - p0))

    y0 = rng.binomial(1, p0, n_group0)
    y1 = rng.binomial(1, p1, n_group1)

    x = np.concatenate([np.zeros(n_group0), np.ones(n_group1)]).reshape(-1, 1)
    y = np.concatenate([y0, y1])

    fit = fit_logistic_irls(x, y)
    fitted_or = fit["odds_ratios"][1]
    assert fitted_or == pytest.approx(true_or, rel=0.15)


def test_fisher_odds_ratio_matches_scipy_reference():
    from scipy.stats import fisher_exact

    ref_or, ref_p = fisher_exact([[10, 90], [5, 95]])
    result = fisher_odds_ratio(10, 90, 5, 95)
    assert result["odds_ratio"] == pytest.approx(ref_or)
    assert result["p_value"] == pytest.approx(ref_p)


def test_mantel_haenszel_pools_toward_true_common_effect():
    # two strata with OR 2.25 and 2.64 but different base rates
    strata = [
        np.array([[40, 160], [20, 180]]),
        np.array([[100, 100], [55, 145]]),
    ]
    result = mantel_haenszel(strata)
    assert 2.0 < result["or_mantel_haenszel"] < 3.0
    assert result["p_value"] < 0.05


def test_lift_metrics_basic_ratio():
    result = lift_metrics(bad_in_group=40, n_in_group=100, bad_in_baseline=10, n_in_baseline=100)
    assert result["lift"] == pytest.approx(4.0)
    assert result["ci_lower"] < 4.0 < result["ci_upper"]
