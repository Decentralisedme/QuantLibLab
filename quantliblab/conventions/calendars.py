"""
Financial holiday calendars.

Calendars provided:
  LondonCalendar  — GBP / SONIA
  NewYorkCalendar — USD / SOFR (NY Fed / Federal Reserve holidays)
  TARGETCalendar  — EUR / ESTR (TARGET2 settlement days)
  CMECalendar     — CME exchange holidays (used for SOFR futures date generation)

NewYorkCalendar vs CMECalendar:
  The NY Fed calendar (used for SOFR fixing and OIS swaps) and the CME exchange
  calendar (used for SOFR futures reference quarter dates) differ on one point:
    - Good Friday:  CME is CLOSED,  NY Fed is OPEN
  All other holidays are identical.

Holiday lists cover 2020-2035 by default.  Pass a custom `years` range
to extend or restrict coverage.
"""
from __future__ import annotations

import calendar as _cal
from datetime import date, timedelta


class BaseCalendar:
    """Weekends are non-business days; subclasses add holiday sets."""

    _holidays: frozenset[date] = frozenset()

    def is_weekend(self, d: date) -> bool:
        return d.weekday() >= 5

    def is_holiday(self, d: date) -> bool:
        return d in self._holidays

    def is_business_day(self, d: date) -> bool:
        return not self.is_weekend(d) and not self.is_holiday(d)


class LondonCalendar(BaseCalendar):
    """UK bank holiday calendar (GBP, SONIA)."""

    def __init__(self, years: range | None = None) -> None:
        years = years or range(2020, 2036)
        self._holidays = frozenset(_uk_holidays(years))


class NewYorkCalendar(BaseCalendar):
    """US Federal Reserve holiday calendar (USD, SOFR)."""

    def __init__(self, years: range | None = None) -> None:
        years = years or range(2020, 2036)
        self._holidays = frozenset(_us_holidays(years))


class TARGETCalendar(BaseCalendar):
    """TARGET2 calendar (EUR, ESTR)."""

    def __init__(self, years: range | None = None) -> None:
        years = years or range(2020, 2036)
        self._holidays = frozenset(_target_holidays(years))


class CMECalendar(BaseCalendar):
    """
    CME exchange holiday calendar.

    Used for SOFR futures reference quarter date generation.
    Differs from NewYorkCalendar (NY Fed) in one way:
      - Good Friday:  CME CLOSED,  NY Fed OPEN
    """

    def __init__(self, years: range | None = None) -> None:
        years = years or range(2020, 2036)
        self._holidays = frozenset(_cme_holidays(years))


# ---------------------------------------------------------------------------
# Holiday generators
# ---------------------------------------------------------------------------

def _uk_holidays(years: range) -> list[date]:
    holidays: list[date] = []
    for y in years:
        easter = _easter(y)
        holidays += [
            _nearest_monday(date(y, 1, 1)),    # New Year's Day (observed)
            easter - timedelta(days=2),         # Good Friday
            easter + timedelta(days=1),         # Easter Monday
            _nth_weekday(y, 5, 0, 1),           # Early May bank holiday
            _last_weekday(y, 5, 0),             # Spring bank holiday
            _last_weekday(y, 8, 0),             # Summer bank holiday
            _nearest_monday(date(y, 12, 25)),   # Christmas (observed)
            _nearest_monday(date(y, 12, 26)),   # Boxing Day (observed)
        ]
    return holidays


def _us_holidays(years: range) -> list[date]:
    holidays: list[date] = []
    for y in years:
        holidays += [
            _observed(date(y, 1, 1)),           # New Year's Day
            _nth_weekday(y, 1, 0, 3),           # MLK Day (3rd Mon Jan)
            _nth_weekday(y, 2, 0, 3),           # Presidents' Day (3rd Mon Feb)
            _last_weekday(y, 5, 0),             # Memorial Day (last Mon May)
            _observed(date(y, 6, 19)),           # Juneteenth
            _observed(date(y, 7, 4)),            # Independence Day
            _nth_weekday(y, 9, 0, 1),           # Labor Day (1st Mon Sep)
            _nth_weekday(y, 11, 3, 4),          # Thanksgiving (4th Thu Nov)
            _observed(date(y, 12, 25)),          # Christmas
        ]
    return holidays


def _cme_holidays(years: range) -> list[date]:
    """
    CME exchange holidays.

    Same as US Federal Reserve holidays except:
      + Good Friday (CME closed, NY Fed open)
      - Columbus Day (CME open, NY Fed closed)
    """
    holidays: list[date] = []
    for y in years:
        easter = _easter(y)
        holidays += [
            _observed(date(y, 1, 1)),           # New Year's Day
            _nth_weekday(y, 1, 0, 3),           # MLK Day (3rd Mon Jan)
            _nth_weekday(y, 2, 0, 3),           # Presidents' Day (3rd Mon Feb)
            easter - timedelta(days=2),          # Good Friday — CME closed
            _last_weekday(y, 5, 0),             # Memorial Day (last Mon May)
            _observed(date(y, 6, 19)),           # Juneteenth
            _observed(date(y, 7, 4)),            # Independence Day
            _nth_weekday(y, 9, 0, 1),           # Labor Day (1st Mon Sep)
            _nth_weekday(y, 11, 3, 4),          # Thanksgiving (4th Thu Nov)
            _observed(date(y, 12, 25)),          # Christmas
            # Note: Columbus Day (2nd Mon Oct) is NOT included — CME is open
        ]
    return holidays


def _target_holidays(years: range) -> list[date]:
    holidays: list[date] = []
    for y in years:
        easter = _easter(y)
        holidays += [
            date(y, 1, 1),
            easter - timedelta(days=2),   # Good Friday
            easter + timedelta(days=1),   # Easter Monday
            date(y, 5, 1),                # Labour Day
            date(y, 12, 25),
            date(y, 12, 26),
        ]
    return holidays


# ---------------------------------------------------------------------------
# Date arithmetic utilities
# ---------------------------------------------------------------------------

def _easter(year: int) -> date:
    """Gauss/Anonymous Gregorian algorithm for Easter Sunday."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    month, day = divmod(h + ll - 7 * m + 114, 31)
    return date(year, month, day + 1)


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """n-th occurrence (1-based) of weekday (0=Mon…6=Sun) in month."""
    count = 0
    for day in range(1, _cal.monthrange(year, month)[1] + 1):
        d = date(year, month, day)
        if d.weekday() == weekday:
            count += 1
            if count == n:
                return d
    raise ValueError(f"No {n}th weekday {weekday} in {year}-{month:02d}")


def _last_weekday(year: int, month: int, weekday: int) -> date:
    """Last occurrence of weekday in month."""
    last_day = _cal.monthrange(year, month)[1]
    d = date(year, month, last_day)
    while d.weekday() != weekday:
        d -= timedelta(days=1)
    return d


def _observed(d: date) -> date:
    """Saturday holidays observed Friday; Sunday holidays observed Monday."""
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def _nearest_monday(d: date) -> date:
    """Move Saturday/Sunday holidays to the nearest Monday."""
    if d.weekday() == 5:
        return d + timedelta(days=2)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d
