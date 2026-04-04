"""
FX forward curve via Covered Interest Rate Parity (CIP).

The fair forward rate is derived from two OIS discount curves and the
spot FX rate — no separate market data required.

Formula:
    F(T) = S × P_dom(T) / P_for(T)

Where:
    S        : spot FX rate (units of domestic per one unit of foreign)
               e.g. GBPUSD = 1.2924 means 1 GBP costs 1.2924 USD
    P_dom(T) : domestic currency discount factor at maturity T
    P_for(T) : foreign currency discount factor at maturity T

Swap points (forward points):
    Swap pts = (F(T) - S) × 10_000

Sign convention:
    Positive swap points => domestic rates > foreign rates (dom depreciates)
    Negative swap points => domestic rates < foreign rates (dom appreciates)

CIP basis:
    Real market forwards deviate slightly from CIP (cross-currency basis).
    This effect is typically 5-30 pips post-2008. Not modelled here.

Supported pairs (domestic / foreign):
    GBPUSD  : USD dom, GBP for
    EURUSD  : USD dom, EUR for
    EURGBP  : GBP dom, EUR for
"""
from __future__ import annotations

from datetime import date

import pandas as pd

from quantliblab.conventions.tenor import Tenor
from quantliblab.conventions.business_day import BusinessDayConvention, adjust
from quantliblab.conventions.calendars import BaseCalendar, NewYorkCalendar
from quantliblab.curves.base.rate_curve import RateCurve


# Standard output tenors
_STANDARD_TENORS = ["ON", "1W", "1M", "3M", "6M", "9M", "12M"]


class FXForwardCurve:
    """
    FX forward curve derived from CIP using two OIS discount curves.

    Parameters
    ----------
    dom_curve   : domestic currency OIS curve (e.g. SOFR for USD)
    for_curve   : foreign currency OIS curve (e.g. SONIA for GBP)
    spot        : spot FX rate (dom per for, e.g. 1.2924 for GBPUSD)
    pair        : FX pair label, e.g. "GBPUSD"
    calendar    : business day calendar for date adjustment
    """

    def __init__(
        self,
        dom_curve:  RateCurve,
        for_curve:  RateCurve,
        spot:       float,
        pair:       str,
        calendar:   BaseCalendar | None = None,
    ) -> None:
        if spot <= 0:
            raise ValueError(f"Spot rate must be positive, got {spot}")

        self.spot      = spot
        self.dom_curve = dom_curve
        self.for_curve = for_curve
        self.pair      = pair.upper()
        self.calendar  = calendar or NewYorkCalendar()

        # Both curves must share the same valuation date
        if dom_curve.valuation_date != for_curve.valuation_date:
            raise ValueError(
                f"Curve valuation dates must match. "
                f"dom={dom_curve.valuation_date}, for={for_curve.valuation_date}"
            )
        self.valuation_date = dom_curve.valuation_date

    # ------------------------------------------------------------------
    # Core methods
    # ------------------------------------------------------------------

    def forward(self, d: date) -> float:
        """
        Fair forward FX rate at date d via CIP.

            F(T) = S × P_dom(T) / P_for(T)
        """
        p_dom = self.dom_curve.discount_factor(d)
        p_for = self.for_curve.discount_factor(d)
        return self.spot * p_dom / p_for

    def swap_points(self, d: date) -> float:
        """
        Forward points (swap points) at date d.

            Swap pts = (F(T) - S) × 10_000

        Expressed in pips (4th decimal place for most pairs).
        """
        return (self.forward(d) - self.spot) * 10_000

    def forward_curve(
        self,
        tenors: list[str] | None = None,
    ) -> pd.DataFrame:
        """
        Return forward rates and swap points at standard tenors.

        Parameters
        ----------
        tenors : list of tenor strings (default: ON/1W/1M/3M/6M/9M/12M)

        Returns
        -------
        DataFrame with columns:
            tenor, maturity_date, spot, forward, swap_points_pips, fwd_pct_change
        """
        tenors = tenors or _STANDARD_TENORS
        rows = []

        for t_str in tenors:
            tenor = Tenor.from_string(t_str)
            mat = adjust(
                tenor.add_to(self.valuation_date),
                BusinessDayConvention.MODIFIED_FOLLOWING,
                self.calendar,
            )
            fwd = self.forward(mat)
            swp = self.swap_points(mat)
            pct = (fwd / self.spot - 1.0) * 100.0

            rows.append({
                "tenor":            t_str,
                "maturity_date":    mat.isoformat(),
                "spot":             round(self.spot, 6),
                "forward":          round(fwd, 6),
                "swap_points_pips": round(swp, 2),
                "fwd_pct_change":   round(pct, 4),
            })

        return pd.DataFrame(rows)

    def __repr__(self) -> str:
        return (
            f"FXForwardCurve({self.pair}, spot={self.spot:.4f}, "
            f"valuation={self.valuation_date})"
        )


# ------------------------------------------------------------------
# Convenience constructors for the three main pairs
# ------------------------------------------------------------------

def gbpusd_forward(
    spot:        float,
    sofr_curve:  RateCurve,
    sonia_curve: RateCurve,
) -> FXForwardCurve:
    """GBPUSD forward curve. Domestic=USD(SOFR), Foreign=GBP(SONIA)."""
    from quantliblab.conventions.calendars import LondonCalendar
    return FXForwardCurve(sofr_curve, sonia_curve, spot, "GBPUSD", LondonCalendar())


def eurusd_forward(
    spot:       float,
    sofr_curve: RateCurve,
    estr_curve: RateCurve,
) -> FXForwardCurve:
    """EURUSD forward curve. Domestic=USD(SOFR), Foreign=EUR(ESTR)."""
    from quantliblab.conventions.calendars import TARGETCalendar
    return FXForwardCurve(sofr_curve, estr_curve, spot, "EURUSD", TARGETCalendar())


def eurgbp_forward(
    spot:        float,
    sonia_curve: RateCurve,
    estr_curve:  RateCurve,
) -> FXForwardCurve:
    """EURGBP forward curve. Domestic=GBP(SONIA), Foreign=EUR(ESTR)."""
    from quantliblab.conventions.calendars import LondonCalendar
    return FXForwardCurve(sonia_curve, estr_curve, spot, "EURGBP", LondonCalendar())
