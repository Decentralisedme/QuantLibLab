"""
OIS curve bootstrap algorithm.

Converts a list of market instruments (deposits, OIS swaps) into
calibrated CurvePillar objects that fully define the zero coupon curve.

Bootstrap logic
---------------
Pillars are solved sequentially from shortest to longest maturity.
At each step, all previously solved pillars are held fixed.

Deposits (ON, TN, 1W, 1M):
    Direct formula — no solver required.
    P(T) = 1 / (1 + r * tau)          [simple compounding, market convention]
    r_zero = log(1 + r * tau) / tau    [convert to continuous]

OIS Swaps (1W–12M, single-period):
    For swaps up to 1Y, there is only one coupon period so the
    floating leg NPV = 1 - P(T) and the fixed leg NPV = K * tau * P(T).
    Setting NPV = 0 gives the same direct formula:
    P(T) = 1 / (1 + K * tau)

    Newton-Raphson is used anyway so the code generalises cleanly
    to multi-period swaps when we extend beyond 1Y.
    In practice NR converges in 1–2 iterations for single-period.

Calibration objective (NPV = 0):
    f(r) = (1 - P(T; r)) - K * tau * P(T; r) = 0
    where P(T; r) = exp(-r * T) and T is the year fraction to maturity.
    f'(r) = T * P(T; r) * (1 + K * tau)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from enum import Enum

from quantliblab.conventions.day_count import DayCountBasis, year_fraction
from quantliblab.conventions.tenor import Tenor
from quantliblab.conventions.business_day import BusinessDayConvention, adjust
from quantliblab.conventions.calendars import BaseCalendar
from quantliblab.math.solvers.newton_raphson import solve as newton_raphson
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
    instruments     : market quotes sorted by tenor (or we sort here)
    basis           : day count convention for year fractions
    calendar        : holiday calendar for business day adjustment
    settlement_days : spot lag (0 for SONIA, 2 for SOFR/ESTR)

    Returns
    -------
    List of CurvePillar, one per instrument, sorted by maturity.
    """
    # Spot date = valuation + settlement lag
    spot_date = _add_business_days(valuation_date, settlement_days, calendar)

    pillars: list[CurvePillar] = []

    for instr in instruments:
        tenor = Tenor.from_string(instr.tenor)
        maturity = adjust(
            tenor.add_to(spot_date),
            BusinessDayConvention.MODIFIED_FOLLOWING,
            calendar,
        )
        tau = year_fraction(valuation_date, maturity, basis)

        if tau <= 0:
            raise ValueError(
                f"Non-positive year fraction for tenor {instr.tenor}: "
                f"maturity={maturity}, valuation={valuation_date}"
            )

        if instr.instrument_type == InstrumentType.DEPOSIT:
            df, zr = _bootstrap_deposit(instr.market_rate, tau)
        else:
            df, zr = _bootstrap_ois_swap(instr.market_rate, tau)

        pillars.append(CurvePillar(
            instrument      = instr.instrument_type.value,
            tenor           = instr.tenor,
            maturity_date   = maturity,
            year_frac       = tau,
            zero_rate       = zr,
            discount_factor = df,
        ))

    return sorted(pillars, key=lambda p: p.maturity_date)


# ---------------------------------------------------------------------------
# Instrument calibration
# ---------------------------------------------------------------------------

def _bootstrap_deposit(rate: float, tau: float) -> tuple[float, float]:
    """
    Bootstrap a deposit.
    Simple compounding: P = 1 / (1 + r * tau)
    Continuous zero:    r_zero = log(1 + r * tau) / tau
    """
    df = 1.0 / (1.0 + rate * tau)
    zr = math.log(1.0 + rate * tau) / tau
    return df, zr


def _bootstrap_ois_swap(rate: float, tau: float) -> tuple[float, float]:
    """
    Bootstrap a single-period OIS swap using Newton-Raphson.

    NPV(r) = (1 - exp(-r*tau)) - rate * tau * exp(-r*tau) = 0
    => P = 1 / (1 + rate * tau)  [same as deposit for single-period]

    NR is used for generality (multi-period extension).
    """
    K = rate

    def npv(r: float) -> float:
        p = math.exp(-r * tau)
        return (1.0 - p) - K * tau * p

    def dnpv(r: float) -> float:
        p = math.exp(-r * tau)
        return tau * p * (1.0 + K * tau)

    # Initial guess: simple rate approximation
    r0 = math.log(1.0 + K * tau) / tau

    try:
        zr = newton_raphson(npv, dnpv, x0=r0, tol=1e-12)
    except (ZeroDivisionError, RuntimeError):
        # Fallback to Brent with generous bracket
        zr = brent(npv, a=max(r0 - 0.05, -0.10), b=r0 + 0.05, tol=1e-12)

    df = math.exp(-zr * tau)
    return df, zr


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
