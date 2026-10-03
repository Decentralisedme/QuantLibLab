"""
Multi-period OIS bootstrap tests (defect D-001).

The bootstrapper used to apply the single-period formula at every tenor,
which is wrong beyond 1Y (5Y zero ~28bp low on a flat 4% curve) and went
unnoticed because no test went past 12M. These tests are the closing
conditions for D-001:

  1. Flat-curve identity — flat zero r in, r out at every pillar (<0.1bp)
  2. Tenors to 30Y
  3. Round-trip — reprice every input on the built curve, recover par (<0.01bp)
  4. Known-good reference — hand-checkable textbook bootstrap

The par-rate pricer below is written independently of the bootstrapper:
it builds its own annual fixed-leg schedule and discounts on whatever
curve it is handed.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date, timedelta
from itertools import pairwise

import pytest
from dateutil.relativedelta import relativedelta

from quantliblab.conventions.business_day import BusinessDayConvention, adjust
from quantliblab.conventions.calendars import (
    BaseCalendar,
    LondonCalendar,
    NewYorkCalendar,
    TARGETCalendar,
)
from quantliblab.conventions.day_count import DayCountBasis, year_fraction
from quantliblab.conventions.tenor import Tenor
from quantliblab.curves import CurveInstrument, InstrumentType, OISCurve
from quantliblab.curves.bootstrapper import bootstrap

VALUATION = date(2025, 3, 24)

BP = 1e-4

TENORS_30Y = [
    "ON", "1W", "1M", "3M", "6M", "9M", "12M", "18M",
    "2Y", "3Y", "4Y", "5Y", "7Y", "10Y", "12Y", "15Y", "20Y", "25Y", "30Y",
]

# (constructor, basis, calendar, settlement_days)
CONVENTIONS = {
    "SOFR":  (OISCurve.sofr,  DayCountBasis.ACT_360, NewYorkCalendar(), 2),
    "SONIA": (OISCurve.sonia, DayCountBasis.ACT_365, LondonCalendar(),  0),
    "ESTR":  (OISCurve.estr,  DayCountBasis.ACT_360, TARGETCalendar(),  2),
}


# ---------------------------------------------------------------------------
# Independent par-rate pricer
# ---------------------------------------------------------------------------

def _next_bd(d: date, cal: BaseCalendar) -> date:
    d += timedelta(days=1)
    while not cal.is_business_day(d):
        d += timedelta(days=1)
    return d


def _schedule(tenor: str, cal: BaseCalendar, settle: int) -> list[date]:
    """Accrual dates [S, t_1, ..., T]: annual, rolled back from spot + tenor."""
    t1 = _next_bd(VALUATION, cal)
    if tenor == "ON":
        return [VALUATION, t1]
    spot = VALUATION
    for _ in range(settle):
        spot = _next_bd(spot, cal)
    if tenor == "TN":
        return [t1, spot]
    end = Tenor.from_string(tenor).add_to(spot)
    dates = [end]
    k = 1
    while end - relativedelta(years=k) > spot:
        dates.insert(0, end - relativedelta(years=k))
        k += 1
    mf = BusinessDayConvention.MODIFIED_FOLLOWING
    return [spot] + [adjust(d, mf, cal) for d in dates]


def _par_rate(
    schedule: list[date], basis: DayCountBasis, df: Callable[[date], float],
) -> float:
    """K such that K * sum tau_i P(t_i) = P(S) - P(T)."""
    annuity = sum(
        year_fraction(a, b, basis) * df(b) for a, b in pairwise(schedule)
    )
    return (df(schedule[0]) - df(schedule[-1])) / annuity


def _flat_quotes(r: float, basis: DayCountBasis, cal: BaseCalendar, settle: int):
    """Par quotes implied by a flat continuously-compounded zero curve r."""
    def df(d: date) -> float:
        t = 0.0 if d == VALUATION else year_fraction(VALUATION, d, basis)
        return math.exp(-r * t)

    return [
        CurveInstrument(
            t,
            InstrumentType.DEPOSIT if t == "ON" else InstrumentType.OIS_SWAP,
            _par_rate(_schedule(t, cal, settle), basis, df),
        )
        for t in TENORS_30Y
    ]


# Upward-sloping, humped-then-inverted long end — a realistic shape to 30Y
MARKET_QUOTES = dict(zip(TENORS_30Y, [
    0.0431, 0.0430, 0.0428, 0.0425, 0.0415, 0.0405, 0.0398, 0.0385,
    0.0378, 0.0372, 0.0371, 0.0373, 0.0380, 0.0391, 0.0396, 0.0401, 0.0404, 0.0399, 0.0392,
]))


def _market_instruments() -> list[CurveInstrument]:
    return [
        CurveInstrument(t, InstrumentType.DEPOSIT if t == "ON" else InstrumentType.OIS_SWAP, k)
        for t, k in MARKET_QUOTES.items()
    ]


# ---------------------------------------------------------------------------
# 1. Flat-curve identity
# ---------------------------------------------------------------------------

class TestFlatCurveIdentity:
    @pytest.mark.parametrize("name", list(CONVENTIONS))
    @pytest.mark.parametrize("r", [-0.005, 0.01, 0.04, 0.08])
    def test_zero_rate_recovered_at_every_pillar(self, name, r):
        ctor, basis, cal, settle = CONVENTIONS[name]
        curve = ctor(VALUATION, _flat_quotes(r, basis, cal, settle))
        for p in curve.pillars:
            assert abs(p.zero_rate - r) < 0.1 * BP, (
                f"{name} {p.tenor}: zero {p.zero_rate:.8f} vs flat {r}"
            )

    def test_flat_forwards_between_pillars(self):
        """On a flat curve every forward, not just every pillar, is r."""
        ctor, basis, cal, settle = CONVENTIONS["SOFR"]
        curve = ctor(VALUATION, _flat_quotes(0.04, basis, cal, settle))
        d = VALUATION + timedelta(days=30)
        while d < curve.pillars[-1].maturity_date:
            assert abs(curve.forward_rate(d, d + timedelta(days=91)) - 0.04) < 0.1 * BP
            d += timedelta(days=97)


# ---------------------------------------------------------------------------
# 2. Tenors to 30Y
# ---------------------------------------------------------------------------

class TestTenorsTo30Y:
    @pytest.mark.parametrize("name", list(CONVENTIONS))
    def test_full_curve_builds(self, name):
        curve = CONVENTIONS[name][0](VALUATION, _market_instruments())
        assert [p.tenor for p in curve.pillars] == TENORS_30Y
        assert curve.pillars[-1].year_frac > 29.9

    @pytest.mark.parametrize("name", list(CONVENTIONS))
    def test_no_arbitrage_to_30y(self, name):
        curve = CONVENTIONS[name][0](VALUATION, _market_instruments())
        dfs = [p.discount_factor for p in curve.pillars]
        assert all(a > b for a, b in pairwise(dfs))
        for a, b in pairwise(curve.pillars):
            assert curve.forward_rate(a.maturity_date, b.maturity_date) > 0

    def test_long_end_not_single_period(self):
        """
        The D-001 symptom: single-period P = 1/(1 + K*T) puts the 5Y zero
        ~28bp under a flat 4% par curve. The multi-period zero must sit
        near the annual-compounding answer log(1.04), not near that.
        """
        flat = [
            CurveInstrument(t, InstrumentType.DEPOSIT if t == "ON" else InstrumentType.OIS_SWAP, 0.04)
            for t in TENORS_30Y
        ]
        curve = OISCurve.sofr(VALUATION, flat)
        p5 = next(p for p in curve.pillars if p.tenor == "5Y")
        single_period = math.log(1 + 0.04 * p5.year_frac) / p5.year_frac
        assert p5.zero_rate - single_period > 20 * BP
        assert abs(p5.zero_rate - math.log(1.04)) < 1 * BP

    def test_broken_tenor_has_front_stub(self):
        """18M pays a 6M stub then a full year; the 18M pillar must still reprice."""
        sched = _schedule("18M", NewYorkCalendar(), 2)
        assert len(sched) == 3
        assert 170 < (sched[1] - sched[0]).days < 190


# ---------------------------------------------------------------------------
# 3. Round-trip
# ---------------------------------------------------------------------------

class TestRoundTrip:
    @pytest.mark.parametrize("name", list(CONVENTIONS))
    def test_reprice_inputs_to_par(self, name):
        ctor, basis, cal, settle = CONVENTIONS[name]
        curve = ctor(VALUATION, _market_instruments())
        for tenor, quote in MARKET_QUOTES.items():
            par = _par_rate(_schedule(tenor, cal, settle), basis, curve.discount_factor)
            assert abs(par - quote) < 0.01 * BP, (
                f"{name} {tenor}: repriced {par:.10f} vs quote {quote}"
            )

    def test_input_order_irrelevant(self):
        ins = _market_instruments()
        a = OISCurve.sofr(VALUATION, ins)
        b = OISCurve.sofr(VALUATION, list(reversed(ins)))
        assert [p.discount_factor for p in a.pillars] == [p.discount_factor for p in b.pillars]


# ---------------------------------------------------------------------------
# 4. Known-good reference
# ---------------------------------------------------------------------------

class _EveryDayCalendar(BaseCalendar):
    """No weekends, no holidays — strips calendar effects for textbook numbers."""

    def is_business_day(self, d: date) -> bool:
        return True


def _textbook_curve(quotes: dict[str, float]):
    """Annual par swaps, 30/360, spot = valuation: every accrual is exactly 1.0."""
    return bootstrap(
        date(2025, 1, 15),
        [CurveInstrument(t, InstrumentType.OIS_SWAP, k) for t, k in quotes.items()],
        DayCountBasis.THIRTY_360, _EveryDayCalendar(), settlement_days=0,
    )


class TestKnownReference:
    def test_textbook_stepped_par_curve(self):
        """
        Textbook par-swap bootstrap with tau = 1, solved forward one pillar
        at a time. Each line checkable by hand:

          P1 = 1 / 1.03                                  = 0.9708737864
          P2 = (1 - 0.035 * P1) / 1.035
             = (1 - 0.0339805825) / 1.035                = 0.9333520942
          P3 = (1 - 0.04 * (P1 + P2)) / 1.04
             = (1 - 0.04 * 1.9042258806) / 1.04          = 0.8882990046
        """
        pillars = _textbook_curve({"1Y": 0.03, "2Y": 0.035, "3Y": 0.04})
        assert [p.year_frac for p in pillars] == [1.0, 2.0, 3.0]
        expected = [0.9708737864, 0.9333520942, 0.8882990046]
        for p, e in zip(pillars, expected):
            assert abs(p.discount_factor - e) < 1e-10, p.tenor
        # Continuous zero: r_3 = -ln(P3) / 3 = 3.948229%
        assert abs(pillars[2].zero_rate - 0.03948229) < 1e-8

    def test_flat_annual_par_curve(self):
        """
        Flat par K with annual coupons and tau = 1 gives P(n) = (1 + K)^-n:
        every coupon bond prices at par when discounted at its own coupon.
        For K = 4%, the 5Y zero is ln(1.04) = 3.92207% — the D-001 case.
        """
        pillars = _textbook_curve({f"{n}Y": 0.04 for n in (1, 2, 3, 5, 10, 30)})
        for p in pillars:
            n = int(p.tenor[:-1])
            assert abs(p.discount_factor - 1.04 ** -n) < 1e-12, p.tenor
            assert abs(p.zero_rate - math.log(1.04)) < 1e-10, p.tenor
