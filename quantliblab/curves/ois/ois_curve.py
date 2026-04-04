"""
OIS curve — zero coupon curve bootstrapped from O/N deposits and OIS swaps.

Three pre-configured curves:
  OISCurve.sofr(valuation_date, quotes)   USD / SOFR / ACT360 / New York
  OISCurve.sonia(valuation_date, quotes)  GBP / SONIA / ACT365 / London
  OISCurve.estr(valuation_date, quotes)   EUR / ESTR  / ACT360 / TARGET2

Usage
-----
    from quantliblab.curves.ois import OISCurve
    from quantliblab.curves.bootstrapper import CurveInstrument, InstrumentType
    from datetime import date

    instruments = [
        CurveInstrument("ON",  InstrumentType.DEPOSIT,  0.0531),
        CurveInstrument("1W",  InstrumentType.OIS_SWAP, 0.0530),
        CurveInstrument("1M",  InstrumentType.OIS_SWAP, 0.0528),
        CurveInstrument("3M",  InstrumentType.OIS_SWAP, 0.0522),
        CurveInstrument("6M",  InstrumentType.OIS_SWAP, 0.0510),
        CurveInstrument("12M", InstrumentType.OIS_SWAP, 0.0490),
    ]

    curve = OISCurve.sofr(date(2025, 3, 24), instruments)
    print(curve.discount_factor(date(2025, 6, 24)))  # ~0.987
    print(curve.zero_rate(date(2025, 6, 24)))
    print(curve.forward_rate(date(2025, 6, 24), date(2025, 9, 24)))
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from quantliblab.conventions.day_count import DayCountBasis
from quantliblab.conventions.calendars import (
    NewYorkCalendar, LondonCalendar, TARGETCalendar,
)
from quantliblab.curves.base.rate_curve import RateCurve, CurvePillar
from quantliblab.curves.bootstrapper import CurveInstrument, bootstrap


class OISCurve(RateCurve):
    """
    Zero coupon OIS curve.

    Inherits discount_factor(), zero_rate(), forward_rate() from RateCurve.
    Adds currency metadata and a to_dataframe() display method.
    """

    def __init__(
        self,
        valuation_date:  date,
        pillars:         list[CurvePillar],
        basis:           DayCountBasis,
        currency:        str,
        benchmark:       str,
    ) -> None:
        super().__init__(valuation_date, pillars, basis)
        self.currency  = currency
        self.benchmark = benchmark

    # ------------------------------------------------------------------
    # Named constructors
    # ------------------------------------------------------------------

    @classmethod
    def sofr(
        cls,
        valuation_date: date,
        instruments:    list[CurveInstrument],
    ) -> "OISCurve":
        """USD SOFR OIS curve — ACT/360, New York calendar, T+2 settlement."""
        pillars = bootstrap(
            valuation_date,
            instruments,
            basis           = DayCountBasis.ACT_360,
            calendar        = NewYorkCalendar(),
            settlement_days = 2,
        )
        return cls(valuation_date, pillars, DayCountBasis.ACT_360, "USD", "SOFR")

    @classmethod
    def sonia(
        cls,
        valuation_date: date,
        instruments:    list[CurveInstrument],
    ) -> "OISCurve":
        """GBP SONIA OIS curve — ACT/365, London calendar, T+0 settlement."""
        pillars = bootstrap(
            valuation_date,
            instruments,
            basis           = DayCountBasis.ACT_365,
            calendar        = LondonCalendar(),
            settlement_days = 0,
        )
        return cls(valuation_date, pillars, DayCountBasis.ACT_365, "GBP", "SONIA")

    @classmethod
    def estr(
        cls,
        valuation_date: date,
        instruments:    list[CurveInstrument],
    ) -> "OISCurve":
        """EUR ESTR OIS curve — ACT/360, TARGET2 calendar, T+2 settlement."""
        pillars = bootstrap(
            valuation_date,
            instruments,
            basis           = DayCountBasis.ACT_360,
            calendar        = TARGETCalendar(),
            settlement_days = 2,
        )
        return cls(valuation_date, pillars, DayCountBasis.ACT_360, "EUR", "ESTR")

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Return pillar data as a tidy DataFrame."""
        rows = [
            {
                "instrument":       p.instrument,
                "tenor":            p.tenor,
                "maturity_date":    p.maturity_date.isoformat(),
                "year_fraction":    round(p.year_frac, 6),
                "zero_rate":        round(p.zero_rate, 8),
                "discount_factor":  round(p.discount_factor, 8),
            }
            for p in self.pillars
        ]
        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return (
            f"OISCurve({self.currency}/{self.benchmark}, "
            f"valuation={self.valuation_date}, "
            f"pillars={len(self.pillars)}, "
            f"basis={self.basis.value})"
        )
