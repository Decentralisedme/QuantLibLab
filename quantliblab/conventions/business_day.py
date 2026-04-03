"""
Business day conventions and date adjustment.
"""
from __future__ import annotations

from datetime import date, timedelta
from enum import Enum
from typing import Protocol


class BusinessDayConvention(str, Enum):
    UNADJUSTED = "UNADJUSTED"
    FOLLOWING = "FOLLOWING"
    MODIFIED_FOLLOWING = "MODIFIED_FOLLOWING"
    PRECEDING = "PRECEDING"
    MODIFIED_PRECEDING = "MODIFIED_PRECEDING"


class Calendar(Protocol):
    """Minimal interface expected by adjustment functions."""

    def is_business_day(self, d: date) -> bool: ...


def adjust(d: date, convention: BusinessDayConvention, cal: Calendar) -> date:
    """Return d adjusted to a business day under the given convention."""
    if convention == BusinessDayConvention.UNADJUSTED:
        return d
    if convention == BusinessDayConvention.FOLLOWING:
        return _following(d, cal)
    if convention == BusinessDayConvention.PRECEDING:
        return _preceding(d, cal)
    if convention == BusinessDayConvention.MODIFIED_FOLLOWING:
        adj = _following(d, cal)
        return adj if adj.month == d.month else _preceding(d, cal)
    if convention == BusinessDayConvention.MODIFIED_PRECEDING:
        adj = _preceding(d, cal)
        return adj if adj.month == d.month else _following(d, cal)
    raise ValueError(f"Unsupported convention: {convention}")


def add_business_days(d: date, n: int, cal: Calendar) -> date:
    """Shift d by n business days (negative n moves backwards)."""
    step = timedelta(days=1 if n >= 0 else -1)
    remaining = abs(n)
    while remaining > 0:
        d += step
        if cal.is_business_day(d):
            remaining -= 1
    return d


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _following(d: date, cal: Calendar) -> date:
    while not cal.is_business_day(d):
        d += timedelta(days=1)
    return d


def _preceding(d: date, cal: Calendar) -> date:
    while not cal.is_business_day(d):
        d -= timedelta(days=1)
    return d
