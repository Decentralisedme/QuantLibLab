"""Unit tests for quantliblab.math.statistics."""
import math

import numpy as np
import pytest

from quantliblab.math.statistics import (
    mean, variance, std, skewness, kurtosis,
    sharpe_ratio, max_drawdown, rolling_volatility,
    ols, beta_estimate,
)

RNG = np.random.default_rng(42)
TOL = 1e-10
LOOSE = 1e-6


# ---------------------------------------------------------------------------
# Moments
# ---------------------------------------------------------------------------

class TestMean:
    def test_simple(self):
        assert mean(np.array([1.0, 2.0, 3.0])) == pytest.approx(2.0)

    def test_single(self):
        assert mean(np.array([5.0])) == 5.0


class TestVariance:
    def test_known(self):
        # var([1,2,3], ddof=1) = 1.0
        assert variance(np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)

    def test_ddof0(self):
        assert variance(np.array([1.0, 2.0, 3.0]), ddof=0) == pytest.approx(2.0 / 3.0)

    def test_constant_series(self):
        assert variance(np.array([3.0, 3.0, 3.0])) == 0.0


class TestStd:
    def test_known(self):
        assert std(np.array([1.0, 2.0, 3.0])) == pytest.approx(1.0)


class TestSkewness:
    def test_symmetric_is_zero(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        assert abs(skewness(x)) < TOL

    def test_positive_skew(self):
        # Right-skewed: large positive outlier
        x = np.array([1.0, 1.0, 1.0, 1.0, 10.0])
        assert skewness(x) > 0

    def test_too_few_raises(self):
        with pytest.raises(ValueError):
            skewness(np.array([1.0, 2.0]))


class TestKurtosis:
    def test_normal_excess_near_zero(self):
        x = RNG.normal(0, 1, 10_000)
        assert abs(kurtosis(x)) < 0.2    # sample fluctuation

    def test_heavy_tails_positive(self):
        # t-distribution has positive excess kurtosis
        x = RNG.standard_t(df=4, size=10_000)
        assert kurtosis(x) > 0

    def test_raw_normal_near_three(self):
        x = RNG.normal(0, 1, 10_000)
        assert abs(kurtosis(x, excess=False) - 3.0) < 0.3

    def test_too_few_raises(self):
        with pytest.raises(ValueError):
            kurtosis(np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# Risk metrics
# ---------------------------------------------------------------------------

class TestSharpe:
    def test_zero_returns(self):
        r = np.zeros(252)
        assert sharpe_ratio(r) == 0.0

    def test_positive_drift(self):
        # Constant daily return of 0.001 => positive Sharpe
        r = np.full(252, 0.001)
        assert sharpe_ratio(r) > 0

    def test_scale(self):
        # Doubling returns doubles both mean and vol => Sharpe unchanged
        r = RNG.normal(0.001, 0.01, 252)
        s1 = sharpe_ratio(r)
        s2 = sharpe_ratio(2 * r)
        assert s2 == pytest.approx(s1, rel=1e-6)


class TestMaxDrawdown:
    def test_no_drawdown(self):
        prices = np.array([1.0, 1.1, 1.2, 1.3])
        assert max_drawdown(prices) == 0.0

    def test_known_drawdown(self):
        # Peak is 120, trough is 80 => drawdown = (80 - 120) / 120 = -1/3
        prices = np.array([100.0, 120.0, 80.0, 90.0])
        assert max_drawdown(prices) == pytest.approx(-40.0 / 120.0, rel=1e-6)

    def test_too_few_raises(self):
        with pytest.raises(ValueError):
            max_drawdown(np.array([1.0]))


class TestRollingVol:
    def test_shape(self):
        r = RNG.normal(0, 0.01, 100)
        rv = rolling_volatility(r, window=20)
        assert rv.shape == (100,)

    def test_nan_prefix(self):
        r = RNG.normal(0, 0.01, 50)
        rv = rolling_volatility(r, window=10)
        assert np.all(np.isnan(rv[:9]))
        assert not np.isnan(rv[9])


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

class TestOLS:
    def test_simple_known(self):
        # y = 2 + 3x, no noise
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = 2.0 + 3.0 * x
        result = ols(x, y)
        assert result.beta[0] == pytest.approx(2.0, abs=1e-8)  # intercept
        assert result.beta[1] == pytest.approx(3.0, abs=1e-8)  # slope
        assert result.r_squared == pytest.approx(1.0, abs=1e-8)

    def test_residuals_sum_to_zero(self):
        x = RNG.normal(0, 1, 50)
        y = 1.5 * x + RNG.normal(0, 0.1, 50)
        result = ols(x, y)
        assert abs(np.sum(result.residuals)) < 1e-8

    def test_no_intercept(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = 3.0 * x
        result = ols(x, y, intercept=False)
        assert result.beta[0] == pytest.approx(3.0, abs=1e-8)

    def test_multivariate(self):
        X = RNG.normal(0, 1, (100, 3))
        true_beta = np.array([1.0, 2.0, -1.5, 0.5])   # intercept + 3 slopes
        y = np.column_stack([np.ones(100), X]) @ true_beta + RNG.normal(0, 0.01, 100)
        result = ols(X, y)
        assert np.allclose(result.beta, true_beta, atol=0.05)

    def test_under_determined_raises(self):
        with pytest.raises(ValueError):
            ols(np.array([[1.0, 2.0]]), np.array([1.0]))

    def test_t_stats_significant(self):
        # Strong signal => |t| >> 2
        x = np.linspace(0, 10, 200)
        y = 5.0 * x + RNG.normal(0, 0.1, 200)
        result = ols(x, y)
        assert abs(result.t_stats[1]) > 100


class TestBetaEstimate:
    def test_beta_one_for_identical(self):
        r = RNG.normal(0, 0.01, 252)
        assert beta_estimate(r, r) == pytest.approx(1.0, abs=1e-8)

    def test_beta_two(self):
        benchmark = RNG.normal(0, 0.01, 500)
        asset = 2.0 * benchmark + RNG.normal(0, 0.0001, 500)
        assert beta_estimate(asset, benchmark) == pytest.approx(2.0, abs=0.01)
