"""
Unit tests for sofr_futures_helper.bootstrap_sofr_futures.

All tests are offline.  A deposit pillar anchors the short end;
futures contracts are synthetic (no network calls).
"""
import math
from datetime import date

import pytest

from quantliblab.curves.base.rate_curve import CurvePillar
from quantliblab.curves.sofr_futures_helper import bootstrap_sofr_futures
from quantliblab.conventions.day_count import DayCountBasis, year_fraction
from quantliblab.data.loaders.sofr_futures_loader import SOFRFuturesContract


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _deposit_pillar(val: date, maturity: date, rate: float) -> CurvePillar:
    """Build a synthetic deposit pillar (simple compounding)."""
    tau = year_fraction(val, maturity, DayCountBasis.ACT_360)
    df  = 1.0 / (1.0 + rate * tau)
    zr  = -math.log(df) / tau
    return CurvePillar("Deposit", "1M", val, maturity, tau, zr, df)


def _make_contract(
    ticker: str,
    contract_type: str,
    ref_start: date,
    ref_end: date,
    implied_rate: float,
) -> SOFRFuturesContract:
    price = 100.0 - implied_rate * 100.0
    return SOFRFuturesContract(
        ticker         = ticker,
        contract_type  = contract_type,
        delivery_year  = ref_end.year,
        delivery_month = ref_end.month,
        ref_start      = ref_start,
        ref_end        = ref_end,
        price          = price,
        implied_rate   = implied_rate,
    )


# ---------------------------------------------------------------------------
# Valuation date and base pillar for most tests
# ---------------------------------------------------------------------------

VAL        = date(2026, 4, 13)
# 1-month deposit maturing 2026-05-13
DEPOSIT_MAT = date(2026, 5, 13)
BASE_RATE   = 0.043      # 4.30% flat for simplicity

BASE_PILLAR = _deposit_pillar(VAL, DEPOSIT_MAT, BASE_RATE)


# ---------------------------------------------------------------------------
# Stub filtering
# ---------------------------------------------------------------------------

class TestStubFiltering:
    def test_skips_contract_with_past_ref_start(self):
        # ref_start before valuation_date → skipped
        past_contract = _make_contract(
            "SR3H26.CME", "SR3",
            date(2025, 12, 17), date(2026, 3, 18),
            0.043,
        )
        result = bootstrap_sofr_futures(VAL, [past_contract], [BASE_PILLAR])
        assert result == []

    def test_includes_contract_with_future_ref_start(self):
        future_contract = _make_contract(
            "SR1K26.CME", "SR1",
            date(2026, 5, 1), date(2026, 5, 31),
            0.043,
        )
        result = bootstrap_sofr_futures(VAL, [future_contract], [BASE_PILLAR])
        assert len(result) == 1

    def test_includes_contract_with_ref_start_on_valuation_date(self):
        # Edge: ref_start == valuation_date is treated as live
        contract = _make_contract(
            "SR1J26.CME", "SR1",
            VAL, date(2026, 4, 30),
            0.043,
        )
        result = bootstrap_sofr_futures(VAL, [contract], [BASE_PILLAR])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Pricing identity: P(ref_end) = P(ref_start) / (1 + K * tau)
# ---------------------------------------------------------------------------

class TestPricingIdentity:
    def test_single_sr1_discount_factor(self):
        """
        With only one SR1 contract, verify P(ref_end) satisfies
        the pricing identity exactly.
        """
        ref_start = date(2026, 5, 1)
        ref_end   = date(2026, 5, 31)
        K = 0.0430

        contract = _make_contract("SR1K26.CME", "SR1", ref_start, ref_end, K)
        pillars  = bootstrap_sofr_futures(VAL, [contract], [BASE_PILLAR])
        assert len(pillars) == 1

        p = pillars[0]

        # Recover P(ref_start) from the deposit pillar
        from quantliblab.curves.base.rate_curve import RateCurve
        curve  = RateCurve(VAL, [BASE_PILLAR], DayCountBasis.ACT_360)
        p_start = curve.discount_factor(ref_start)

        tau_ref = year_fraction(ref_start, ref_end, DayCountBasis.ACT_360)
        expected_df = p_start / (1.0 + K * tau_ref)

        assert abs(p.discount_factor - expected_df) < 1e-12

    def test_single_sr3_discount_factor(self):
        ref_start = date(2026, 6, 17)
        ref_end   = date(2026, 9, 16)
        K = 0.0367

        # Need deposits past ref_start; build a synthetic 6M pillar
        pillar_6m = _deposit_pillar(VAL, date(2026, 10, 13), 0.043)
        contract  = _make_contract("SR3U26.CME", "SR3", ref_start, ref_end, K)
        pillars   = bootstrap_sofr_futures(VAL, [contract], [pillar_6m])
        assert len(pillars) == 1

        p = pillars[0]

        from quantliblab.curves.base.rate_curve import RateCurve
        curve   = RateCurve(VAL, [pillar_6m], DayCountBasis.ACT_360)
        p_start = curve.discount_factor(ref_start)
        tau_ref = year_fraction(ref_start, ref_end, DayCountBasis.ACT_360)
        expected_df = p_start / (1.0 + K * tau_ref)

        assert abs(p.discount_factor - expected_df) < 1e-12

    def test_zero_rate_consistent_with_discount_factor(self):
        ref_start = date(2026, 5, 1)
        ref_end   = date(2026, 5, 31)
        contract  = _make_contract("SR1K26.CME", "SR1", ref_start, ref_end, 0.043)
        pillars   = bootstrap_sofr_futures(VAL, [contract], [BASE_PILLAR])
        p = pillars[0]

        tau2 = year_fraction(VAL, ref_end, DayCountBasis.ACT_360)
        expected_zr = -math.log(p.discount_factor) / tau2
        assert abs(p.zero_rate - expected_zr) < 1e-12


# ---------------------------------------------------------------------------
# Sequential chaining: P(T₂_end) is read from P(T₁_end)
# ---------------------------------------------------------------------------

class TestSequentialChaining:
    def test_two_sr1_chain(self):
        """
        Two consecutive SR1 contracts.  The second uses P(ref_start) which
        comes from the first contract's ref_end pillar.
        """
        c1 = _make_contract("SR1K26.CME", "SR1", date(2026, 5, 1),  date(2026, 5, 31), 0.043)
        c2 = _make_contract("SR1M26.CME", "SR1", date(2026, 6, 1),  date(2026, 6, 30), 0.043)

        pillars = bootstrap_sofr_futures(VAL, [c1, c2], [BASE_PILLAR])
        assert len(pillars) == 2
        assert pillars[0].maturity_date == date(2026, 5, 31)
        assert pillars[1].maturity_date == date(2026, 6, 30)

    def test_three_sr3_contiguous_chain(self):
        """
        Three consecutive SR3 contracts — ref_end of one is ref_start of next.
        Verify discount factors form a consistent monotone decreasing chain.
        """
        p0 = _deposit_pillar(VAL, date(2026, 12, 13), 0.043)

        contracts = [
            _make_contract("SR3M26.CME", "SR3", date(2026, 3, 18), date(2026, 6, 17), 0.043),
            _make_contract("SR3U26.CME", "SR3", date(2026, 6, 17), date(2026, 9, 16), 0.043),
            _make_contract("SR3Z26.CME", "SR3", date(2026, 9, 16), date(2026, 12, 16), 0.043),
        ]
        # First contract has ref_start = 2026-03-18 < VAL → skipped
        pillars = bootstrap_sofr_futures(VAL, contracts, [p0])
        assert len(pillars) == 2  # only U26 and Z26 survive

        dfs = [p.discount_factor for p in pillars]
        assert all(dfs[i] > dfs[i + 1] for i in range(len(dfs) - 1)), \
            "Discount factors should be strictly decreasing"

    def test_ordering_invariant(self):
        """Contracts passed in reverse order give the same pillars."""
        c1 = _make_contract("SR1K26.CME", "SR1", date(2026, 5, 1), date(2026, 5, 31), 0.043)
        c2 = _make_contract("SR1M26.CME", "SR1", date(2026, 6, 1), date(2026, 6, 30), 0.043)

        fwd = bootstrap_sofr_futures(VAL, [c1, c2], [BASE_PILLAR])
        rev = bootstrap_sofr_futures(VAL, [c2, c1], [BASE_PILLAR])

        for p1, p2 in zip(fwd, rev):
            assert p1.maturity_date == p2.maturity_date
            assert abs(p1.discount_factor - p2.discount_factor) < 1e-14


# ---------------------------------------------------------------------------
# Pillar metadata
# ---------------------------------------------------------------------------

class TestPillarMetadata:
    def test_instrument_label_sr1(self):
        c = _make_contract("SR1K26.CME", "SR1", date(2026, 5, 1), date(2026, 5, 31), 0.043)
        p = bootstrap_sofr_futures(VAL, [c], [BASE_PILLAR])[0]
        assert p.instrument == "SR1Future"

    def test_instrument_label_sr3(self):
        p0 = _deposit_pillar(VAL, date(2026, 10, 13), 0.043)
        c = _make_contract("SR3U26.CME", "SR3", date(2026, 6, 17), date(2026, 9, 16), 0.043)
        p = bootstrap_sofr_futures(VAL, [c], [p0])[0]
        assert p.instrument == "SR3Future"

    def test_tenor_is_ticker_without_exchange(self):
        c = _make_contract("SR1K26.CME", "SR1", date(2026, 5, 1), date(2026, 5, 31), 0.043)
        p = bootstrap_sofr_futures(VAL, [c], [BASE_PILLAR])[0]
        assert p.tenor == "SR1K26"

    def test_start_date_is_ref_start(self):
        c = _make_contract("SR1K26.CME", "SR1", date(2026, 5, 1), date(2026, 5, 31), 0.043)
        p = bootstrap_sofr_futures(VAL, [c], [BASE_PILLAR])[0]
        assert p.start_date == date(2026, 5, 1)

    def test_maturity_is_ref_end(self):
        c = _make_contract("SR1K26.CME", "SR1", date(2026, 5, 1), date(2026, 5, 31), 0.043)
        p = bootstrap_sofr_futures(VAL, [c], [BASE_PILLAR])[0]
        assert p.maturity_date == date(2026, 5, 31)


# ---------------------------------------------------------------------------
# Convexity adjustment
# ---------------------------------------------------------------------------

class TestConvexityAdjustment:
    def test_convexity_lowers_rate(self):
        """
        Applying a positive convexity_vol should lower the implied rate K,
        which raises P(ref_end) and lowers the zero rate.
        """
        c = _make_contract("SR1K26.CME", "SR1", date(2026, 5, 1), date(2026, 5, 31), 0.043)

        no_adj  = bootstrap_sofr_futures(VAL, [c], [BASE_PILLAR], convexity_vol=0.0)
        with_adj = bootstrap_sofr_futures(VAL, [c], [BASE_PILLAR], convexity_vol=0.01)

        assert with_adj[0].discount_factor > no_adj[0].discount_factor
        assert with_adj[0].zero_rate < no_adj[0].zero_rate

    def test_zero_vol_matches_no_adjustment(self):
        c = _make_contract("SR1K26.CME", "SR1", date(2026, 5, 1), date(2026, 5, 31), 0.043)
        p1 = bootstrap_sofr_futures(VAL, [c], [BASE_PILLAR], convexity_vol=0.0)[0]
        p2 = bootstrap_sofr_futures(VAL, [c], [BASE_PILLAR])[0]
        assert abs(p1.discount_factor - p2.discount_factor) < 1e-15
