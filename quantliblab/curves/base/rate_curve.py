"""
Abstract base class for zero coupon rate curves.

All curves store a set of calibrated pillars and expose three methods:
  discount_factor(date)        -> P(t) = exp(-r * t)
  zero_rate(date)              -> r(t) = -log(P(t)) / t
  forward_rate(date1, date2)   -> f(t1,t2) = -log(P(t2)/P(t1)) / (t2-t1)

Interpolation between pillars is handled by a pluggable interpolator
(default: FlatForwardInterpolator — log-linear on discount factors).

All rates are continuously compounded.
All year fractions are computed from the curve's valuation date.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

from quantliblab.conventions.day_count import DayCountBasis, year_fraction
from quantliblab.math.interpolation.flat_forward import FlatForwardInterpolator
from quantliblab.math.interpolation.base import Extrapolation, Interpolator1D


@dataclass
class CurvePillar:
    """One calibrated point on the curve."""
    instrument:      str    # e.g. "Deposit", "OISSwap"
    tenor:           str    # e.g. "3M"
    maturity_date:   date
    year_frac:       float  # time in years from valuation date
    zero_rate:       float  # continuously compounded zero rate
    discount_factor: float  # exp(-zero_rate * year_frac)


class RateCurve:
    """
    Zero coupon rate curve.

    Parameters
    ----------
    valuation_date : curve reference date (today)
    pillars        : calibrated CurvePillar list, sorted by maturity
    basis          : day count convention used for year fractions
    interpolator   : optional pre-built interpolator (built from pillars if None)
    """

    def __init__(
        self,
        valuation_date: date,
        pillars: list[CurvePillar],
        basis: DayCountBasis = DayCountBasis.ACT_365,
        interpolator: Interpolator1D | None = None,
    ) -> None:
        if not pillars:
            raise ValueError("RateCurve requires at least one pillar")

        self.valuation_date = valuation_date
        self.pillars = sorted(pillars, key=lambda p: p.maturity_date)
        self.basis = basis

        # Build interpolator over (year_frac, discount_factor) pairs
        # Always include t=0 anchor: P(0) = 1.0
        xs = [0.0] + [p.year_frac for p in self.pillars]
        ys = [1.0]  + [p.discount_factor for p in self.pillars]

        self._interp: Interpolator1D = interpolator or FlatForwardInterpolator(
            xs, ys, extrapolation=Extrapolation.FLAT_FORWARD
        )

    # ------------------------------------------------------------------
    # Core curve API
    # ------------------------------------------------------------------

    def discount_factor(self, d: date) -> float:
        """
        Interpolated discount factor P(t).

        P(t) = exp(-r(t) * t)
        P(0) = 1.0 by definition.
        """
        t = self._year_frac(d)
        if t < 0:
            raise ValueError(f"Date {d} is before valuation date {self.valuation_date}")
        if t == 0.0:
            return 1.0
        return float(self._interp(t))

    def zero_rate(self, d: date) -> float:
        """
        Continuously compounded zero rate r(t).

        r(t) = -log(P(t)) / t
        """
        t = self._year_frac(d)
        if t <= 0:
            raise ValueError(f"zero_rate requires a date strictly after valuation date")
        p = self.discount_factor(d)
        if p <= 0:
            raise ValueError(f"Non-positive discount factor {p} at {d}")
        return -math.log(p) / t

    def forward_rate(self, d1: date, d2: date) -> float:
        """
        Continuously compounded forward rate between d1 and d2.

        f(t1, t2) = -log(P(t2) / P(t1)) / (t2 - t1)

        This is the rate locked in today for borrowing/lending
        between d1 and d2.
        """
        if d2 <= d1:
            raise ValueError(f"forward_rate requires d2 > d1, got {d1} >= {d2}")
        p1 = self.discount_factor(d1)
        p2 = self.discount_factor(d2)
        t1 = self._year_frac(d1)
        t2 = self._year_frac(d2)
        dt = t2 - t1
        if dt <= 0:
            raise ValueError(f"Year fraction between {d1} and {d2} is non-positive")
        return -math.log(p2 / p1) / dt

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _year_frac(self, d: date) -> float:
        """Year fraction from valuation date to d."""
        if d == self.valuation_date:
            return 0.0
        return year_fraction(self.valuation_date, d, self.basis)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"valuation={self.valuation_date}, "
            f"pillars={len(self.pillars)}, "
            f"basis={self.basis.value})"
        )
