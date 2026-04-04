"""
Unit tests for quantliblab.curves.

Tests cover:
  - Bootstrap: deposit and OIS swap calibration
  - RateCurve: discount_factor, zero_rate, forward_rate
  - OISCurve: SOFR, SONIA, ESTR named constructors
  - Arbitrage conditions: P(t) monotone decreasing, fwd rates positive
"""
from __future__ import annotations

import math
from datetime import date, timedelta

import pytest

from quantliblab.curves import OISCurve, CurveInstrument, InstrumentType
from quantliblab.curves.base.rate_curve import RateCurve, CurvePillar
from quantliblab.curves.bootstrapper import (
    bootstrap, _bootstrap_deposit, _bootstrap_ois_swap,
)
from quantliblab.conventions.day_count import DayCountBasis
from quantliblab.conventions.calendars import NewYorkCalendar, LondonCalendar, TARGETCalendar

# ---------------------------------------------------------------------------
# Sample instrument set — flat SOFR curve at 5.30%
# ---------------------------------------------------------------------------

FLAT_RATE = 0.0530
VALUATION = date(2025, 3, 24)

SOFR_INSTRUMENTS = [
    CurveInstrument("ON",  InstrumentType.DEPOSIT,  FLAT_RATE),
    CurveInstrument("1W",  InstrumentType.OIS_SWAP, FLAT_RATE),
    CurveInstrument("1M",  InstrumentType.OIS_SWAP, FLAT_RATE),
    CurveInstrument("3M",  InstrumentType.OIS_SWAP, FLAT_RATE),
    CurveInstrument("6M",  InstrumentType.OIS_SWAP, FLAT_RATE),
    CurveInstrument("12M", InstrumentType.OIS_SWAP, FLAT_RATE),
]

UPWARD_INSTRUMENTS = [
    CurveInstrument("ON",  InstrumentType.DEPOSIT,  0.0431),
    CurveInstrument("1W",  InstrumentType.OIS_SWAP, 0.0430),
    CurveInstrument("1M",  InstrumentType.OIS_SWAP, 0.0428),
    CurveInstrument("3M",  InstrumentType.OIS_SWAP, 0.0435),
    CurveInstrument("6M",  InstrumentType.OIS_SWAP, 0.0445),
    CurveInstrument("12M", InstrumentType.OIS_SWAP, 0.0465),
]


# ---------------------------------------------------------------------------
# Deposit bootstrap
# ---------------------------------------------------------------------------

class TestDepositBootstrap:
    def test_known_df(self):
        # P = 1 / (1 + 0.053 * 1/360)  for 1-day ON with ACT/360
        tau = 1.0 / 360.0
        df, zr = _bootstrap_deposit(0.0530, tau)
        expected_df = 1.0 / (1.0 + 0.0530 * tau)
        assert abs(df - expected_df) < 1e-12

    def test_zero_rate_roundtrip(self):
        tau = 0.25  # 3M
        df, zr = _bootstrap_deposit(0.0530, tau)
        # Reconstructing df from zero rate should match
        assert abs(math.exp(-zr * tau) - df) < 1e-12

    def test_positive_rate_positive_zr(self):
        _, zr = _bootstrap_deposit(0.0530, 0.5)
        assert zr > 0


# ---------------------------------------------------------------------------
# OIS swap bootstrap
# ---------------------------------------------------------------------------

class TestOISSwapBootstrap:
    def test_npv_is_zero(self):
        """Calibrated zero rate should make swap NPV exactly zero."""
        K, tau = 0.0530, 0.25
        df, zr = _bootstrap_ois_swap(K, tau)
        p = math.exp(-zr * tau)
        npv = (1 - p) - K * tau * p
        assert abs(npv) < 1e-10

    def test_consistent_with_deposit(self):
        """For single-period, OIS swap and deposit should give same df."""
        K, tau = 0.0530, 0.25
        df_dep, _ = _bootstrap_deposit(K, tau)
        df_swap, _ = _bootstrap_ois_swap(K, tau)
        assert abs(df_dep - df_swap) < 1e-8

    def test_higher_rate_lower_df(self):
        _, zr_low  = _bootstrap_ois_swap(0.03, 0.5)
        _, zr_high = _bootstrap_ois_swap(0.06, 0.5)
        assert zr_high > zr_low


# ---------------------------------------------------------------------------
# RateCurve
# ---------------------------------------------------------------------------

class TestRateCurve:
    def setup_method(self):
        self.curve = OISCurve.sofr(VALUATION, SOFR_INSTRUMENTS)

    def test_discount_factor_at_valuation_is_one(self):
        assert self.curve.discount_factor(VALUATION) == 1.0

    def test_discount_factors_decrease(self):
        """P(t) must be monotonically decreasing."""
        dfs = [self.curve.discount_factor(p.maturity_date) for p in self.curve.pillars]
        assert all(dfs[i] > dfs[i+1] for i in range(len(dfs)-1))

    def test_discount_factor_at_pillar(self):
        """df from curve should match stored pillar df."""
        for p in self.curve.pillars:
            assert abs(self.curve.discount_factor(p.maturity_date) - p.discount_factor) < 1e-8

    def test_zero_rate_at_pillar(self):
        for p in self.curve.pillars:
            assert abs(self.curve.zero_rate(p.maturity_date) - p.zero_rate) < 1e-6

    def test_zero_rates_positive_and_reasonable(self):
        """Zero rates should be positive and within 50bps of the quoted rate."""
        for p in self.curve.pillars:
            zr = self.curve.zero_rate(p.maturity_date)
            assert zr > 0
            assert abs(zr - FLAT_RATE) < 0.005   # within 50bps

    def test_forward_rate_positive(self):
        """Forward rates must be positive (no arbitrage)."""
        pillars = self.curve.pillars
        for i in range(len(pillars) - 1):
            fwd = self.curve.forward_rate(
                pillars[i].maturity_date,
                pillars[i+1].maturity_date,
            )
            assert fwd > 0

    def test_forward_rate_reasonable(self):
        """Forward rates should be in the right ballpark."""
        pillars = self.curve.pillars
        for i in range(len(pillars) - 1):
            fwd = self.curve.forward_rate(
                pillars[i].maturity_date,
                pillars[i+1].maturity_date,
            )
            assert abs(fwd - FLAT_RATE) < 0.005  # within 50bps

    def test_date_before_valuation_raises(self):
        with pytest.raises(ValueError):
            self.curve.discount_factor(VALUATION - timedelta(days=1))

    def test_forward_rate_reversed_dates_raises(self):
        p = self.curve.pillars
        with pytest.raises(ValueError):
            self.curve.forward_rate(p[1].maturity_date, p[0].maturity_date)


# ---------------------------------------------------------------------------
# OISCurve named constructors
# ---------------------------------------------------------------------------

class TestOISCurveConstructors:
    def test_sofr_curve_builds(self):
        curve = OISCurve.sofr(VALUATION, SOFR_INSTRUMENTS)
        assert curve.currency == "USD"
        assert curve.benchmark == "SOFR"
        assert curve.basis == DayCountBasis.ACT_360
        assert len(curve.pillars) == 6

    def test_sonia_curve_builds(self):
        curve = OISCurve.sonia(VALUATION, SOFR_INSTRUMENTS)
        assert curve.currency == "GBP"
        assert curve.benchmark == "SONIA"
        assert curve.basis == DayCountBasis.ACT_365

    def test_estr_curve_builds(self):
        curve = OISCurve.estr(VALUATION, SOFR_INSTRUMENTS)
        assert curve.currency == "EUR"
        assert curve.benchmark == "ESTR"
        assert curve.basis == DayCountBasis.ACT_360

    def test_to_dataframe_columns(self):
        curve = OISCurve.sofr(VALUATION, SOFR_INSTRUMENTS)
        df = curve.to_dataframe()
        expected = {"instrument", "tenor", "maturity_date",
                    "year_fraction", "zero_rate", "discount_factor"}
        assert set(df.columns) == expected
        assert len(df) == 6

    def test_to_dataframe_values_consistent(self):
        curve = OISCurve.sofr(VALUATION, SOFR_INSTRUMENTS)
        df = curve.to_dataframe()
        for _, row in df.iterrows():
            # df ≈ exp(-r * t)
            reconstructed = math.exp(-row["zero_rate"] * row["year_fraction"])
            assert abs(reconstructed - row["discount_factor"]) < 1e-6


# ---------------------------------------------------------------------------
# Upward sloping curve — more realistic
# ---------------------------------------------------------------------------

class TestUpwardSlopingCurve:
    def setup_method(self):
        self.curve = OISCurve.sofr(VALUATION, UPWARD_INSTRUMENTS)

    def test_pillars_count(self):
        assert len(self.curve.pillars) == 6

    def test_discount_factors_decrease(self):
        dfs = [self.curve.discount_factor(p.maturity_date) for p in self.curve.pillars]
        assert all(dfs[i] > dfs[i+1] for i in range(len(dfs)-1))

    def test_zero_rates_roughly_match_quotes(self):
        """Zero rates should be in the same ballpark as quoted OIS rates.
        Continuous zero rate vs simple par rate differs more at longer tenors — 20bps tolerance."""
        for p, instr in zip(self.curve.pillars, UPWARD_INSTRUMENTS):
            assert abs(p.zero_rate - instr.market_rate) < 0.002

    def test_repr(self):
        r = repr(self.curve)
        assert "USD" in r and "SOFR" in r
