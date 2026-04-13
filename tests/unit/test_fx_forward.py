"""Unit tests for FXForwardCurve (CIP-based FX forwards)."""
from __future__ import annotations

import math
from datetime import date

import pytest

from quantliblab.curves import (
    OISCurve, CurveInstrument, InstrumentType,
    FXForwardCurve, gbpusd_forward, eurusd_forward, eurgbp_forward,
)

VALUATION = date(2025, 3, 24)

# ── Flat curves for clean analytical tests ───────────────────────────────────

def _flat_sofr(rate: float = 0.0431) -> OISCurve:
    return OISCurve.sofr(VALUATION, [
        CurveInstrument("ON",  InstrumentType.DEPOSIT,  rate),
        CurveInstrument("3M",  InstrumentType.OIS_SWAP, rate),
        CurveInstrument("6M",  InstrumentType.OIS_SWAP, rate),
        CurveInstrument("12M", InstrumentType.OIS_SWAP, rate),
    ])

def _flat_sonia(rate: float = 0.04457) -> OISCurve:
    return OISCurve.sonia(VALUATION, [
        CurveInstrument("ON",  InstrumentType.DEPOSIT,  rate),
        CurveInstrument("3M",  InstrumentType.OIS_SWAP, rate),
        CurveInstrument("6M",  InstrumentType.OIS_SWAP, rate),
        CurveInstrument("12M", InstrumentType.OIS_SWAP, rate),
    ])

def _flat_estr(rate: float = 0.02418) -> OISCurve:
    return OISCurve.estr(VALUATION, [
        CurveInstrument("ON",  InstrumentType.DEPOSIT,  rate),
        CurveInstrument("3M",  InstrumentType.OIS_SWAP, rate),
        CurveInstrument("6M",  InstrumentType.OIS_SWAP, rate),
        CurveInstrument("12M", InstrumentType.OIS_SWAP, rate),
    ])


SPOT_GBPUSD = 1.2924
SPOT_EURUSD = 1.0836
SPOT_EURGBP = 0.8378


# ---------------------------------------------------------------------------
# CIP formula validation
# ---------------------------------------------------------------------------

class TestCIPFormula:
    def test_forward_at_valuation_equals_spot(self):
        """F(0) = S × P_dom(0) / P_for(0) = S × 1 / 1 = S"""
        curve = gbpusd_forward(SPOT_GBPUSD, _flat_sofr(), _flat_sonia())
        assert abs(curve.forward(VALUATION) - SPOT_GBPUSD) < 1e-10

    def test_equal_rates_same_basis_no_forward_premium(self):
        """Same rate, same basis => discount factors cancel exactly => F(T) = S."""
        # Use two SOFR-style curves (both ACT/360) to eliminate day count difference
        rate = 0.04
        sofr1 = OISCurve.sofr(VALUATION, [
            CurveInstrument("ON",  InstrumentType.DEPOSIT,  rate),
            CurveInstrument("12M", InstrumentType.OIS_SWAP, rate),
        ])
        sofr2 = OISCurve.sofr(VALUATION, [
            CurveInstrument("ON",  InstrumentType.DEPOSIT,  rate),
            CurveInstrument("12M", InstrumentType.OIS_SWAP, rate),
        ])
        curve = FXForwardCurve(sofr1, sofr2, SPOT_GBPUSD, "TEST")
        mat = date(2025, 9, 24)
        assert abs(curve.forward(mat) - SPOT_GBPUSD) < 1e-6

    def test_higher_dom_rate_lowers_forward(self):
        """Higher USD (dom) rate => P_dom drops more => F = S × P_dom/P_for falls."""
        sofr  = _flat_sofr(0.05)    # USD at 5% (high)
        sonia = _flat_sonia(0.04)   # GBP at 4%
        curve = gbpusd_forward(SPOT_GBPUSD, sofr, sonia)
        mat = date(2025, 9, 24)
        assert curve.forward(mat) < SPOT_GBPUSD

    def test_lower_dom_rate_raises_forward(self):
        """Lower USD (dom) rate => P_dom drops less => F = S × P_dom/P_for rises."""
        sofr  = _flat_sofr(0.03)    # USD at 3% (low)
        sonia = _flat_sonia(0.05)   # GBP at 5%
        curve = gbpusd_forward(SPOT_GBPUSD, sofr, sonia)
        mat = date(2025, 9, 24)
        assert curve.forward(mat) > SPOT_GBPUSD

    def test_swap_points_zero_same_basis(self):
        """Same rate, same basis => swap points ≈ 0."""
        rate = 0.04
        sofr1 = OISCurve.sofr(VALUATION, [
            CurveInstrument("ON",  InstrumentType.DEPOSIT,  rate),
            CurveInstrument("12M", InstrumentType.OIS_SWAP, rate),
        ])
        sofr2 = OISCurve.sofr(VALUATION, [
            CurveInstrument("ON",  InstrumentType.DEPOSIT,  rate),
            CurveInstrument("12M", InstrumentType.OIS_SWAP, rate),
        ])
        curve = FXForwardCurve(sofr1, sofr2, SPOT_GBPUSD, "TEST")
        mat = date(2025, 9, 24)
        assert abs(curve.swap_points(mat)) < 0.01   # < 0.01 pips


class TestFXForwardCurve:
    def setup_method(self):
        self.sofr  = _flat_sofr()
        self.sonia = _flat_sonia()
        self.curve = FXForwardCurve(self.sofr, self.sonia, SPOT_GBPUSD, "GBPUSD")

    def test_forward_curve_columns(self):
        df = self.curve.forward_curve()
        expected = {"tenor", "maturity_date", "spot", "forward",
                    "swap_points_pips", "fwd_pct_change"}
        assert set(df.columns) == expected

    def test_forward_curve_default_tenors(self):
        df = self.curve.forward_curve()
        assert list(df["tenor"]) == ["ON", "1W", "1M", "3M", "6M", "9M", "12M"]

    def test_forward_curve_spot_consistent(self):
        df = self.curve.forward_curve()
        assert all(df["spot"] == SPOT_GBPUSD)

    def test_swap_points_consistent_with_forward(self):
        df = self.curve.forward_curve()
        for _, row in df.iterrows():
            expected_swp = (row["forward"] - row["spot"]) * 10_000
            assert abs(row["swap_points_pips"] - expected_swp) < 1e-6

    def test_mismatched_valuation_dates_raises(self):
        from datetime import timedelta
        from quantliblab.curves.base.rate_curve import RateCurve, CurvePillar
        import math
        # Manually create a curve with a different valuation date
        other_date = VALUATION + timedelta(days=1)
        pillar = CurvePillar("Deposit", "ON", date(2025, 3, 24), date(2025, 3, 28),
                             1/360, 0.043, math.exp(-0.043/360))
        from quantliblab.conventions.day_count import DayCountBasis
        wrong_curve = RateCurve(other_date, [pillar], DayCountBasis.ACT_360)
        with pytest.raises(ValueError, match="valuation dates must match"):
            FXForwardCurve(self.sofr, wrong_curve, SPOT_GBPUSD, "GBPUSD")

    def test_negative_spot_raises(self):
        with pytest.raises(ValueError, match="positive"):
            FXForwardCurve(self.sofr, self.sonia, -1.0, "GBPUSD")

    def test_repr(self):
        r = repr(self.curve)
        assert "GBPUSD" in r and str(SPOT_GBPUSD) in r


# ---------------------------------------------------------------------------
# Convenience constructors
# ---------------------------------------------------------------------------

class TestConvenienceConstructors:
    def test_gbpusd(self):
        curve = gbpusd_forward(SPOT_GBPUSD, _flat_sofr(), _flat_sonia())
        assert curve.pair == "GBPUSD"
        assert curve.forward(date(2025, 6, 24)) > 0

    def test_eurusd(self):
        curve = eurusd_forward(SPOT_EURUSD, _flat_sofr(), _flat_estr())
        assert curve.pair == "EURUSD"
        df = curve.forward_curve()
        assert len(df) == 7

    def test_eurgbp(self):
        curve = eurgbp_forward(SPOT_EURGBP, _flat_sonia(), _flat_estr())
        assert curve.pair == "EURGBP"
        # EUR rates lower than GBP => EUR/GBP forward < spot (GBP appreciates)
        mat = date(2025, 9, 24)
        assert curve.forward(mat) < SPOT_EURGBP

    def test_triangular_consistency(self):
        """EURGBP forward ≈ EURUSD forward / GBPUSD forward (triangular arbitrage)."""
        sofr  = _flat_sofr()
        sonia = _flat_sonia()
        estr  = _flat_estr()
        mat = date(2025, 9, 24)

        eurusd = eurusd_forward(SPOT_EURUSD, sofr, estr).forward(mat)
        gbpusd = gbpusd_forward(SPOT_GBPUSD, sofr, sonia).forward(mat)
        eurgbp = eurgbp_forward(SPOT_EURGBP, sonia, estr).forward(mat)

        # EURGBP ≈ EURUSD / GBPUSD (triangular parity)
        # Small deviation (~10 pips) expected due to different day count bases
        # (SOFR/ESTR use ACT/360, SONIA uses ACT/365)
        implied_eurgbp = eurusd / gbpusd
        assert abs(eurgbp - implied_eurgbp) < 0.001
