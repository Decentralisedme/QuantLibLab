"""
OIS curve bootstrap algorithm.

Converts a list of market instruments (deposits, OIS swaps) into
calibrated CurvePillar objects that fully define the zero coupon curve.

Bootstrap logic
---------------
Pillars are solved sequentially from shortest to longest maturity.
At each step, all previously solved pillars are held fixed and the
new pillar's zero rate is solved so the instrument prices to par on
the curve built so far plus the new pillar.

Every instrument is priced the same way — as a fixed leg against a
compounded overnight floating leg, both running from start date S to
maturity T. Under single-curve OIS discounting the floating leg is
worth P(S) - P(T), so the par condition is:

    K * sum_i tau_i * P(t_i)  =  P(S) - P(T)

Deposits (ON, TN):
    One period [S, T]:  P(T) = P(S) / (1 + r * tau)   [simple compounding]

OIS Swaps:
    Annual fixed coupons, schedule generated backward from the
    unadjusted maturity (short front stub for broken tenors), each date
    adjusted Modified Following. Swaps of 1Y or less have one coupon.
    Coupon dates between the last solved pillar and T are discounted
    by flat-forward interpolation, so they depend on the unknown pillar —
    hence a 1-D Brent solve on the pillar's zero rate rather than a
    closed form.

All year fractions for pillars are from the valuation date; accruals
tau_i are over each coupon period, both in the curve's day count basis.
Zero rates are continuously compounded.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from enum import Enum

from dateutil.relativedelta import relativedelta

from quantliblab.conventions.business_day import BusinessDayConvention, adjust
from quantliblab.conventions.calendars import BaseCalendar
from quantliblab.conventions.day_count import DayCountBasis, year_fraction
from quantliblab.conventions.tenor import Tenor
from quantliblab.math.interpolation.flat_forward import FlatForwardInterpolator
from quantliblab.math.solvers.brent import solve as brent

from .base.rate_curve import CurvePillar


class InstrumentType(str, Enum):
    DEPOSIT  = "Deposit"
    OIS_SWAP = "OISSwap"


@dataclass
class CurveInstrument:
    """
    One market instrument used to calibrate the curve.

    Parameters
    ----------
    tenor           : e.g. "ON", "1M", "3M"
    instrument_type : Deposit or OISSwap
    market_rate     : quoted rate as decimal (e.g. 0.0530 for 5.30%)
    """
    tenor:           str
    instrument_type: InstrumentType
    market_rate:     float


def bootstrap(
    valuation_date: date,
    instruments:    list[CurveInstrument],
    basis:          DayCountBasis,
    calendar:       BaseCalendar,
    settlement_days: int = 2,
) -> list[CurvePillar]:
    """
    Bootstrap a zero coupon OIS curve from market instruments.

    Parameters
    ----------
    valuation_date  : curve reference / pricing date
    instruments     : market quotes, in any order (sorted by maturity here)
    basis           : day count convention for year fractions
    calendar        : holiday calendar for business day adjustment
    settlement_days : spot lag (0 for SONIA, 2 for SOFR/ESTR)

    Returns
    -------
    List of CurvePillar, one per instrument, sorted by maturity.
    """
    # Spot date = valuation + settlement lag (used for 1W and longer)
    next_business_day = _add_business_days(valuation_date, 1, calendar)
    spot_date = _add_business_days(valuation_date, settlement_days, calendar)

    # (instrument, accrual schedule [S, t_1, ..., T])
    priced: list[tuple[CurveInstrument, list[date]]] = []

    for instr in instruments:
        # O/N and T/N are exceptions to spot-starting:
        #   O/N: start = valuation date (T+0), end = next business day
        #   T/N: start = next business day (T+1), end = T+2 (spot)
        #   All others: start = spot date, end = spot + tenor
        if instr.tenor == "ON":
            schedule = [valuation_date, next_business_day]
        elif instr.tenor == "TN":
            schedule = [next_business_day, spot_date]
        else:
            unadjusted_end = Tenor.from_string(instr.tenor).add_to(spot_date)
            if instr.instrument_type == InstrumentType.OIS_SWAP:
                schedule = _fixed_leg_schedule(spot_date, unadjusted_end, calendar)
            else:
                schedule = [spot_date, adjust(
                    unadjusted_end, BusinessDayConvention.MODIFIED_FOLLOWING, calendar,
                )]
        priced.append((instr, schedule))

    priced.sort(key=lambda x: x[1][-1])

    ts:  list[float] = []
    dfs: list[float] = []
    pillars: list[CurvePillar] = []

    for instr, schedule in priced:
        start_date, maturity = schedule[0], schedule[-1]
        tau = year_fraction(valuation_date, maturity, basis)

        if tau <= 0:
            raise ValueError(
                f"Non-positive year fraction for tenor {instr.tenor}: "
                f"maturity={maturity}, valuation={valuation_date}"
            )
        if ts and tau <= ts[-1]:
            raise ValueError(
                f"Tenor {instr.tenor} matures on {maturity}, not after the previous "
                f"pillar ({pillars[-1].tenor}, {pillars[-1].maturity_date})"
            )

        times = [
            year_fraction(valuation_date, d, basis) if d > valuation_date else 0.0
            for d in schedule
        ]
        accruals = [
            year_fraction(schedule[i - 1], schedule[i], basis)
            for i in range(1, len(schedule))
        ]
        zr = _solve_pillar(instr.market_rate, times, accruals, ts, dfs)
        df = math.exp(-zr * tau)

        ts.append(tau)
        dfs.append(df)
        pillars.append(CurvePillar(
            instrument      = instr.instrument_type.value,
            tenor           = instr.tenor,
            start_date      = start_date,
            maturity_date   = maturity,
            year_frac       = tau,
            zero_rate       = zr,
            discount_factor = df,
        ))

    return pillars


# ---------------------------------------------------------------------------
# Instrument calibration
# ---------------------------------------------------------------------------

def _solve_pillar(
    rate:     float,
    times:    list[float],
    accruals: list[float],
    ts:       list[float],
    dfs:      list[float],
) -> float:
    """
    Solve the zero rate of the pillar at times[-1] so the instrument prices to par.

    times    : [S, t_1, ..., T] as year fractions from valuation
    accruals : tau_i for each coupon period (len(times) - 1)
    ts, dfs  : pillars already solved (held fixed)

    Residual (fixed leg vs floating leg, unit notional):
        f(r) = P(S) - P(T) - K * sum_i tau_i * P(t_i)
    where P is the flat-forward curve through the solved pillars plus
    (T, exp(-r * T)). f is increasing in r.
    """
    K, T = rate, times[-1]
    xs = [0.0] + ts + [T]

    def npv(r: float) -> float:
        interp = FlatForwardInterpolator(xs, [1.0] + dfs + [math.exp(-r * T)])

        def p(t: float) -> float:
            return 1.0 if t <= 0.0 else float(interp(t))

        fixed = K * sum(a * p(t) for a, t in zip(accruals, times[1:]))
        return p(times[0]) - p(T) - fixed

    # Initial guess: single-period simple rate over the whole life
    r0 = math.log(1.0 + K * T) / T if K * T > -1.0 else K
    lo, hi = r0 - 0.01, r0 + 0.01
    while npv(lo) > 0.0:
        lo -= 0.05
        if lo < -1.0:
            raise RuntimeError(f"Could not bracket zero rate for quote {K} at T={T}")
    while npv(hi) < 0.0:
        hi += 0.05
        if hi > 2.0:
            raise RuntimeError(f"Could not bracket zero rate for quote {K} at T={T}")

    return brent(npv, a=lo, b=hi, tol=1e-15)


def _fixed_leg_schedule(
    start:          date,
    unadjusted_end: date,
    calendar:       BaseCalendar,
) -> list[date]:
    """
    Annual fixed-leg accrual dates [S, t_1, ..., T] for an OIS swap.

    Generated backward from the unadjusted end date in 12M steps, so a
    broken tenor (e.g. 18M) gets a short front stub. Dates are adjusted
    Modified Following; the start date is already a business day.
    """
    unadjusted = [unadjusted_end]
    k = 1
    while True:
        d = unadjusted_end - relativedelta(years=k)
        if d <= start:
            break
        unadjusted.append(d)
        k += 1
    rolled = [
        adjust(d, BusinessDayConvention.MODIFIED_FOLLOWING, calendar)
        for d in reversed(unadjusted)
    ]
    return [start] + rolled


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _add_business_days(d: date, n: int, calendar: BaseCalendar) -> date:
    from datetime import timedelta
    for _ in range(n):
        d += timedelta(days=1)
        while not calendar.is_business_day(d):
            d += timedelta(days=1)
    return d
