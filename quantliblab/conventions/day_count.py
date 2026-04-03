"""
Day count conventions.

Supported bases:
  ACT/360      — money market standard (USD, EUR)
  ACT/365      — GBP, crypto
  ACT/ACT ISDA — bonds, swaps
  30/360       — US corporate bonds
  30E/360      — Eurobond
"""
from __future__ import annotations

import calendar
from datetime import date
from enum import Enum


class DayCountBasis(str, Enum):
    ACT_360 = "ACT/360"
    ACT_365 = "ACT/365"
    ACT_ACT_ISDA = "ACT/ACT ISDA"
    THIRTY_360 = "30/360"
    THIRTY_E_360 = "30E/360"


def day_count(start: date, end: date, basis: DayCountBasis) -> int:
    """Raw day count numerator for the given basis."""
    _check_order(start, end)
    match basis:
        case DayCountBasis.ACT_360 | DayCountBasis.ACT_365 | DayCountBasis.ACT_ACT_ISDA:
            return (end - start).days
        case DayCountBasis.THIRTY_360:
            return _thirty_360_days(start, end)
        case DayCountBasis.THIRTY_E_360:
            return _thirty_e_360_days(start, end)


def year_fraction(start: date, end: date, basis: DayCountBasis) -> float:
    """Year fraction between start and end under the given day count basis."""
    _check_order(start, end)
    match basis:
        case DayCountBasis.ACT_360:
            return (end - start).days / 360.0
        case DayCountBasis.ACT_365:
            return (end - start).days / 365.0
        case DayCountBasis.ACT_ACT_ISDA:
            return _act_act_isda(start, end)
        case DayCountBasis.THIRTY_360:
            return _thirty_360_days(start, end) / 360.0
        case DayCountBasis.THIRTY_E_360:
            return _thirty_e_360_days(start, end) / 360.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_order(start: date, end: date) -> None:
    if start > end:
        raise ValueError(f"start {start} must be <= end {end}")


def _act_act_isda(start: date, end: date) -> float:
    if start.year == end.year:
        diy = 366 if calendar.isleap(start.year) else 365
        return (end - start).days / diy

    from datetime import timedelta

    eoy = date(start.year + 1, 1, 1)
    days_first = (eoy - start).days
    diy_first = 366 if calendar.isleap(start.year) else 365

    boy = date(end.year, 1, 1)
    days_last = (end - boy).days
    diy_last = 366 if calendar.isleap(end.year) else 365

    full_years = end.year - start.year - 1
    return days_first / diy_first + full_years + days_last / diy_last


def _thirty_360_days(start: date, end: date) -> int:
    d1, m1, y1 = start.day, start.month, start.year
    d2, m2, y2 = end.day, end.month, end.year
    d1 = min(d1, 30)
    if d1 == 30:
        d2 = min(d2, 30)
    return 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)


def _thirty_e_360_days(start: date, end: date) -> int:
    d1, m1, y1 = min(start.day, 30), start.month, start.year
    d2, m2, y2 = min(end.day, 30), end.month, end.year
    return 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
