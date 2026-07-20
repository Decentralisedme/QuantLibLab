"""Unit tests for the SVI smile, smile-adjusted digitals, and local vol."""
from __future__ import annotations

import math

import numpy as np
import pytest

from quantliblab.math.distributions.normal import cdf as N
from quantliblab.pricing.analytical.digital import (
    d2, digital_above, digital_below, one_touch_prob, polymarket_edge,
)
from quantliblab.volatility.smile.svi import SVIParams, fit_svi, fit_rmse
from quantliblab.volatility.surface.local_vol import LocalVolSurface, SVISlice

TRUE = SVIParams(a=0.004, b=0.35, rho=-0.3, m=0.02, s=0.20)
T1, T2 = 0.10, 0.30
F0 = 100_000.0


# ---------------------------------------------------------------------------
# SVI
# ---------------------------------------------------------------------------

class TestSVI:
    def test_fit_recovers_synthetic_smile(self):
        k = np.linspace(-0.6, 0.6, 25)
        iv = TRUE.implied_vol(k, T1)
        fit = fit_svi(k, iv, T1)
        assert fit_rmse(fit, k, iv, T1) < 5e-4          # < 0.05 vol pts

    def test_fitted_slice_butterfly_free(self):
        k = np.linspace(-0.6, 0.6, 25)
        fit = fit_svi(k, TRUE.implied_vol(k, T1), T1)
        assert fit.is_butterfly_free()

    def test_derivatives_match_finite_difference(self):
        k, h = 0.13, 1e-6
        wp_fd = (TRUE.w(k + h) - TRUE.w(k - h)) / (2 * h)
        wpp_fd = (TRUE.w(k + h) - 2 * TRUE.w(k) + TRUE.w(k - h)) / h**2
        assert abs(float(TRUE.dw_dk(k)) - float(wp_fd)) < 1e-6
        assert abs(float(TRUE.d2w_dk2(k)) - float(wpp_fd)) < 1e-4

    def test_fit_rejects_too_few_quotes(self):
        with pytest.raises(ValueError):
            fit_svi(np.array([0.0, 0.1]), np.array([0.5, 0.5]), T1)


# ---------------------------------------------------------------------------
# European digitals
# ---------------------------------------------------------------------------

class TestDigitals:
    def test_zero_slope_reduces_to_n_d2(self):
        p = digital_above(F0, 110_000.0, 0.25, 0.55, 0.0)
        assert abs(p - N(d2(F0, 110_000.0, 0.55, 0.25))) < 1e-12

    def test_above_below_sum_to_one(self):
        p_up = digital_above(F0, 90_000.0, 0.25, 0.60, -0.15)
        p_dn = digital_below(F0, 90_000.0, 0.25, 0.60, -0.15)
        assert abs(p_up + p_dn - 1.0) < 1e-12

    def test_negative_skew_raises_above_probability(self):
        """dsigma/dk < 0 adds -phi*sqrt(T)*slope > 0 to N(d2)."""
        base = digital_above(F0, 120_000.0, 0.25, 0.55, 0.0)
        skew = digital_above(F0, 120_000.0, 0.25, 0.55, -0.20)
        assert skew > base

    def test_expired(self):
        assert digital_above(F0, 90_000.0, 0.0, 0.5) == 1.0
        assert digital_above(F0, 110_000.0, 0.0, 0.5) == 0.0

    def test_edge_report_sides(self):
        assert polymarket_edge(0.60, 0.50)["side"] == "YES"
        assert polymarket_edge(0.40, 0.50)["side"] == "NO"


# ---------------------------------------------------------------------------
# One-touch
# ---------------------------------------------------------------------------

class TestOneTouch:
    def test_bounds_and_at_barrier(self):
        assert one_touch_prob(F0, F0, 0.5, 0.6) == 1.0
        p = one_touch_prob(F0, 130_000.0, 0.5, 0.6)
        assert 0.0 < p < 1.0

    def test_touch_dominates_terminal_digital(self):
        """P(touch B before T) >= P(F_T > B) for an upper barrier."""
        B, T, sig = 120_000.0, 0.25, 0.6
        assert one_touch_prob(F0, B, T, sig) >= digital_above(F0, B, T, sig) - 1e-12

    def test_monotone_in_vol(self):
        B, T = 130_000.0, 0.25
        assert one_touch_prob(F0, B, T, 0.8) > one_touch_prob(F0, B, T, 0.4)

    def test_up_down_symmetry_shape(self):
        """Lower-barrier touch is also a proper probability, monotone in vol."""
        B, T = 80_000.0, 0.25
        p = one_touch_prob(F0, B, T, 0.6)
        assert 0.0 < p < 1.0
        assert one_touch_prob(F0, B, T, 0.9) > p


# ---------------------------------------------------------------------------
# Local vol surface
# ---------------------------------------------------------------------------

def _surface() -> LocalVolSurface:
    s1 = SVISlice(T1, F0 * 1.004, SVIParams(0.002, 0.30, -0.3, 0.0, 0.15))
    s2 = SVISlice(T2, F0 * 1.012, SVIParams(0.008, 0.40, -0.25, 0.02, 0.25))
    return LocalVolSurface([s1, s2])


class TestLocalVolSurface:
    def test_needs_two_slices(self):
        with pytest.raises(ValueError):
            LocalVolSurface([SVISlice(T1, F0, TRUE)])

    def test_calendar_arbitrage_free(self):
        assert _surface().check_calendar_arbitrage() == []

    def test_smile_matches_slice_at_node(self):
        surf = _surface()
        sigma, _ = surf.smile(0.0, T1)
        assert abs(sigma - float(surf.slices[0].params.implied_vol(0.0, T1))) < 1e-10

    def test_local_variance_positive(self):
        surf = _surface()
        k = np.linspace(-1.0, 1.0, 41)
        for T in (0.05, 0.15, 0.25, 0.40):
            assert np.all(surf.local_variance(k, T) > 0.0)

    def test_mc_touch_brackets_below_closed_form(self):
        """With a skewed smile, local-vol MC is documented to sit BELOW the
        flat barrier-IV closed form (LV under-prices one-touch vs the
        upside-wing IV). Assert that ordering and a sane overall band —
        the two numbers are quoted as a bracket, not expected to agree."""
        surf = _surface()
        B, T = 115_000.0, 0.20
        sigma, _ = surf.smile(math.log(B / F0), T)
        p_cf = one_touch_prob(F0, B, T, sigma)
        p_mc, se = surf.one_touch_prob_mc(F0, B, T, n_paths=8_000,
                                          n_steps=120, seed=7)
        assert 0.0 < p_mc < 1.0 and 0.0 < se < 0.02
        assert p_mc <= p_cf + 3 * se          # LV side of the bracket
        assert abs(p_mc - p_cf) < 0.15        # band stays tradeably narrow
