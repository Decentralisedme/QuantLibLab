"""Unit tests for quantliblab.conventions."""
from datetime import date

import pytest

from quantliblab.conventions import (
    DayCountBasis,
    day_count,
    year_fraction,
    BusinessDayConvention,
    adjust,
    LondonCalendar,
    NewYorkCalendar,
    TARGETCalendar,
    Tenor,
    TenorUnit,
)


# ---------------------------------------------------------------------------
# Day count
# ---------------------------------------------------------------------------

class TestActual360:
    def test_basic(self):
        yf = year_fraction(date(2024, 1, 1), date(2024, 7, 1), DayCountBasis.ACT_360)
        assert abs(yf - 182 / 360) < 1e-10

    def test_zero(self):
        assert year_fraction(date(2024, 1, 1), date(2024, 1, 1), DayCountBasis.ACT_360) == 0.0

    def test_order_error(self):
        with pytest.raises(ValueError):
            year_fraction(date(2024, 7, 1), date(2024, 1, 1), DayCountBasis.ACT_360)


class TestActual365:
    def test_one_year(self):
        yf = year_fraction(date(2023, 1, 1), date(2023, 12, 31), DayCountBasis.ACT_365)
        assert abs(yf - 364 / 365) < 1e-10


class TestActActISDA:
    def test_single_year_non_leap(self):
        yf = year_fraction(date(2023, 1, 1), date(2023, 12, 31), DayCountBasis.ACT_ACT_ISDA)
        assert abs(yf - 364 / 365) < 1e-10

    def test_single_year_leap(self):
        yf = year_fraction(date(2024, 1, 1), date(2024, 12, 31), DayCountBasis.ACT_ACT_ISDA)
        assert abs(yf - 365 / 366) < 1e-10

    def test_cross_year(self):
        # 2023-07-01 to 2024-07-01: straddles a non-leap and a leap year
        yf = year_fraction(date(2023, 7, 1), date(2024, 7, 1), DayCountBasis.ACT_ACT_ISDA)
        assert 0.99 < yf < 1.01


class TestThirty360:
    def test_known_value(self):
        # 2024-01-31 to 2024-04-30 => 30/360: d1=31->30, d2=30->30, 3 full months = 90
        dc = day_count(date(2024, 1, 31), date(2024, 4, 30), DayCountBasis.THIRTY_360)
        assert dc == 90

    def test_full_year(self):
        yf = year_fraction(date(2024, 1, 1), date(2025, 1, 1), DayCountBasis.THIRTY_360)
        assert abs(yf - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Business day adjustment
# ---------------------------------------------------------------------------

class TestAdjust:
    cal = NewYorkCalendar()

    def test_unadjusted(self):
        saturday = date(2024, 1, 6)
        assert adjust(saturday, BusinessDayConvention.UNADJUSTED, self.cal) == saturday

    def test_following(self):
        saturday = date(2024, 1, 6)
        result = adjust(saturday, BusinessDayConvention.FOLLOWING, self.cal)
        assert result == date(2024, 1, 8)  # Monday

    def test_modified_following_stays_in_month(self):
        # 2024-03-30 is a Saturday; Following = Mon 2024-04-01 (different month)
        # Modified Following should go back to Friday 2024-03-29
        d = date(2024, 3, 30)
        result = adjust(d, BusinessDayConvention.MODIFIED_FOLLOWING, self.cal)
        assert result.month == 3

    def test_preceding(self):
        saturday = date(2024, 1, 6)
        result = adjust(saturday, BusinessDayConvention.PRECEDING, self.cal)
        assert result == date(2024, 1, 5)  # Friday


# ---------------------------------------------------------------------------
# Calendars
# ---------------------------------------------------------------------------

class TestCalendars:
    def test_weekends_not_business_days(self):
        cal = LondonCalendar()
        assert not cal.is_business_day(date(2024, 1, 6))   # Saturday
        assert not cal.is_business_day(date(2024, 1, 7))   # Sunday

    def test_weekday_is_business_day(self):
        cal = TARGETCalendar()
        assert cal.is_business_day(date(2024, 1, 2))

    def test_uk_christmas(self):
        cal = LondonCalendar()
        assert not cal.is_business_day(date(2024, 12, 25))

    def test_target_labour_day(self):
        cal = TARGETCalendar()
        assert not cal.is_business_day(date(2024, 5, 1))

    def test_us_independence_day_observed(self):
        # July 4 2026 is a Saturday -> observed Friday July 3
        cal = NewYorkCalendar()
        assert not cal.is_business_day(date(2026, 7, 3))


# ---------------------------------------------------------------------------
# Tenor
# ---------------------------------------------------------------------------

class TestTenor:
    def test_parse_month(self):
        t = Tenor.from_string("3M")
        assert t.quantity == 3
        assert t.unit == TenorUnit.MONTH

    def test_parse_overnight(self):
        t = Tenor.from_string("ON")
        assert t.quantity == 1
        assert t.unit == TenorUnit.DAY

    def test_invalid(self):
        with pytest.raises(ValueError):
            Tenor.from_string("3X")

    def test_add_month_end(self):
        # Jan 31 + 1M = Feb 29 (2024 is a leap year)
        t = Tenor.from_string("1M")
        result = t.add_to(date(2024, 1, 31))
        assert result == date(2024, 2, 29)

    def test_add_year(self):
        t = Tenor.from_string("1Y")
        assert t.add_to(date(2024, 3, 15)) == date(2025, 3, 15)

    def test_ordering(self):
        tenors = [Tenor.from_string(s) for s in ["1Y", "3M", "1W", "6M"]]
        assert sorted(tenors) == [
            Tenor.from_string("1W"),
            Tenor.from_string("3M"),
            Tenor.from_string("6M"),
            Tenor.from_string("1Y"),
        ]

    def test_str(self):
        assert str(Tenor.from_string("6M")) == "6M"
