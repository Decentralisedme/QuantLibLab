"""
SOFR futures loader — Yahoo Finance.

Fetches the active SR1 and SR3 strip and maps each contract to its
reference period dates and implied SOFR rate.

SR3 (3-Month SOFR Futures)
--------------------------
  Delivery months  : March (H), June (M), September (U), December (Z)
  Reference period : 3rd Wednesday of month-3 → 3rd Wednesday of delivery month
  Settlement rate  : compounded daily SOFR over reference period
  Quoted as        : 100 − R  (e.g. 96.33 → 3.67%)

SR1 (1-Month SOFR Futures)
--------------------------
  Delivery months  : all calendar months
  Reference period : 1st → last calendar day of delivery month
  Settlement rate  : arithmetic average daily SOFR over reference month
  Quoted as        : 100 − R

Ticker format: SR3{MonthCode}{2-digit-year}.CME
Month codes   : F=Jan G=Feb H=Mar J=Apr K=May M=Jun
                N=Jul Q=Aug U=Sep V=Oct X=Nov Z=Dec
"""
from __future__ import annotations

import calendar as _cal
from dataclasses import dataclass
from datetime import date
from typing import Optional

import yfinance as yf

from quantliblab.conventions.calendars import CMECalendar

# ---------------------------------------------------------------------------
# Month code lookup
# ---------------------------------------------------------------------------

_MONTH_CODE_TO_NUM: dict[str, int] = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}
_NUM_TO_MONTH_CODE: dict[int, str] = {v: k for k, v in _MONTH_CODE_TO_NUM.items()}

# SR3 uses only quarterly IMM months
_SR3_MONTHS = {3, 6, 9, 12}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SOFRFuturesContract:
    """
    One SOFR futures contract with reference period dates and implied rate.

    Parameters
    ----------
    ticker          : Yahoo Finance ticker (e.g. "SR3M26.CME")
    contract_type   : "SR1" or "SR3"
    delivery_year   : calendar year of delivery
    delivery_month  : calendar month of delivery (1–12)
    ref_start       : first day of the reference accrual period
    ref_end         : last day of the reference accrual period (exclusive = maturity)
    price           : quoted futures price (100 − R)
    implied_rate    : R as decimal (e.g. 0.0367 for 3.67%)
    """
    ticker:         str
    contract_type:  str
    delivery_year:  int
    delivery_month: int
    ref_start:      date
    ref_end:        date
    price:          float
    implied_rate:   float


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_sr3_strip(
    valuation_date: Optional[date] = None,
    n_contracts: int = 8,
) -> list[SOFRFuturesContract]:
    """
    Fetch the front n_contracts of the SR3 quarterly strip.

    Parameters
    ----------
    valuation_date : reference date (defaults to today)
    n_contracts    : how many quarterly contracts to load
    """
    val = valuation_date or date.today()
    cal = CMECalendar()
    contracts = []

    # Find the first SR3 delivery month whose IMM date >= valuation_date
    y, m = _first_active_sr3(val)

    count = 0
    while count < n_contracts:
        ticker = f"SR3{_NUM_TO_MONTH_CODE[m]}{str(y)[-2:]}.CME"
        price = _fetch_price(ticker)
        if price is not None:
            ref_start, ref_end = _sr3_reference_period(y, m, cal)
            contracts.append(SOFRFuturesContract(
                ticker        = ticker,
                contract_type = "SR3",
                delivery_year = y,
                delivery_month= m,
                ref_start     = ref_start,
                ref_end       = ref_end,
                price         = price,
                implied_rate  = (100.0 - price) / 100.0,
            ))
            count += 1

        # Advance to next quarterly month
        m += 3
        if m > 12:
            m -= 12
            y += 1

    return contracts


def fetch_sr1_strip(
    valuation_date: Optional[date] = None,
    n_contracts: int = 12,
) -> list[SOFRFuturesContract]:
    """
    Fetch the front n_contracts of the SR1 monthly strip.

    Parameters
    ----------
    valuation_date : reference date (defaults to today)
    n_contracts    : how many monthly contracts to load
    """
    val = valuation_date or date.today()
    contracts = []

    y, m = val.year, val.month
    # Skip current month if we're past its last day (already settled)
    last_day = _cal.monthrange(y, m)[1]
    if val >= date(y, m, last_day):
        m += 1
        if m > 12:
            m = 1
            y += 1

    count = 0
    while count < n_contracts:
        ticker = f"SR1{_NUM_TO_MONTH_CODE[m]}{str(y)[-2:]}.CME"
        price = _fetch_price(ticker)
        if price is not None:
            ref_start = date(y, m, 1)
            ref_end   = date(y, m, _cal.monthrange(y, m)[1])
            contracts.append(SOFRFuturesContract(
                ticker        = ticker,
                contract_type = "SR1",
                delivery_year = y,
                delivery_month= m,
                ref_start     = ref_start,
                ref_end       = ref_end,
                price         = price,
                implied_rate  = (100.0 - price) / 100.0,
            ))
            count += 1

        m += 1
        if m > 12:
            m = 1
            y += 1

    return contracts


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fetch_price(ticker: str) -> Optional[float]:
    """Return last price for a Yahoo Finance ticker, or None if unavailable."""
    try:
        price = yf.Ticker(ticker).fast_info.last_price
        if price and price > 0:
            return float(price)
    except Exception:
        pass
    return None


def _imm_date(year: int, month: int) -> date:
    """3rd Wednesday of the given month (raw, not holiday-adjusted)."""
    count = 0
    for day in range(1, _cal.monthrange(year, month)[1] + 1):
        d = date(year, month, day)
        if d.weekday() == 2:  # Wednesday
            count += 1
            if count == 3:
                return d
    raise ValueError(f"No 3rd Wednesday in {year}-{month:02d}")


def _sr3_reference_period(
    delivery_year: int,
    delivery_month: int,
    calendar: CMECalendar,
) -> tuple[date, date]:
    """
    Return (ref_start, ref_end) for an SR3 contract.

    ref_end   = 3rd Wednesday of delivery month
    ref_start = 3rd Wednesday of delivery month − 3 months

    Both dates are raw IMM Wednesdays (the reference period definition
    does not adjust for CME holidays; the futures settle against realised
    SOFR which the NY Fed publishes on every Fed business day).
    """
    ref_end = _imm_date(delivery_year, delivery_month)

    # Month 3 quarters back
    start_month = delivery_month - 3
    start_year  = delivery_year
    if start_month <= 0:
        start_month += 12
        start_year  -= 1
    ref_start = _imm_date(start_year, start_month)

    return ref_start, ref_end


def _first_active_sr3(valuation_date: date) -> tuple[int, int]:
    """
    Return (year, month) of the first SR3 contract whose IMM date >= valuation_date.
    """
    y, m = valuation_date.year, valuation_date.month
    for _ in range(16):  # at most 16 months ahead to find a quarterly month
        if m in _SR3_MONTHS:
            if _imm_date(y, m) >= valuation_date:
                return y, m
        m += 1
        if m > 12:
            m = 1
            y += 1
    raise RuntimeError("Could not find active SR3 contract")
