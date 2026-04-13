"""
Unit tests for sofr_futures_loader.

All tests are offline — they test the date arithmetic helpers directly
without making any network calls.
"""
from datetime import date
from unittest.mock import patch

import pytest

from quantliblab.data.loaders.sofr_futures_loader import (
    _imm_date,
    _sr3_reference_period,
    _first_active_sr3,
    fetch_sr3_strip,
    fetch_sr1_strip,
    SOFRFuturesContract,
    _MONTH_CODE_TO_NUM,
    _NUM_TO_MONTH_CODE,
)
from quantliblab.conventions.calendars import CMECalendar


# ---------------------------------------------------------------------------
# IMM date
# ---------------------------------------------------------------------------

class TestIMMDate:
    def test_known_mar_2026(self):
        # 3rd Wednesday of March 2026
        d = _imm_date(2026, 3)
        assert d == date(2026, 3, 18)
        assert d.weekday() == 2  # Wednesday

    def test_known_jun_2026(self):
        assert _imm_date(2026, 6) == date(2026, 6, 17)

    def test_known_sep_2026(self):
        assert _imm_date(2026, 9) == date(2026, 9, 16)

    def test_known_dec_2026(self):
        assert _imm_date(2026, 12) == date(2026, 12, 16)

    def test_known_mar_2027(self):
        assert _imm_date(2027, 3) == date(2027, 3, 17)


# ---------------------------------------------------------------------------
# SR3 reference period
# ---------------------------------------------------------------------------

class TestSR3ReferencePeriod:
    def setup_method(self):
        self.cal = CMECalendar()

    def test_sr3h26(self):
        # SR3H26: delivery March 2026
        start, end = _sr3_reference_period(2026, 3, self.cal)
        assert start == date(2025, 12, 17)  # 3rd Wed Dec 2025
        assert end   == date(2026, 3, 18)   # 3rd Wed Mar 2026

    def test_sr3m26(self):
        # SR3M26: delivery June 2026 — start = end of SR3H26
        start, end = _sr3_reference_period(2026, 6, self.cal)
        assert start == date(2026, 3, 18)
        assert end   == date(2026, 6, 17)

    def test_contiguous(self):
        # Consecutive contracts share a boundary
        _, end_h = _sr3_reference_period(2026, 3, self.cal)
        start_m, _ = _sr3_reference_period(2026, 6, self.cal)
        assert end_h == start_m

    def test_year_boundary(self):
        # SR3H27: delivery March 2027, start = 3rd Wed Dec 2026
        start, end = _sr3_reference_period(2027, 3, self.cal)
        assert start == date(2026, 12, 16)
        assert end   == date(2027, 3, 17)


# ---------------------------------------------------------------------------
# First active SR3
# ---------------------------------------------------------------------------

class TestFirstActiveSR3:
    def test_before_imm(self):
        # 2026-04-05 is before the Jun 2026 IMM date (2026-06-17)
        y, m = _first_active_sr3(date(2026, 4, 5))
        assert (y, m) == (2026, 6)

    def test_on_imm_date(self):
        # On the IMM date itself, that contract is still active
        y, m = _first_active_sr3(date(2026, 3, 18))
        assert (y, m) == (2026, 3)

    def test_after_imm_date(self):
        # Day after Mar 2026 IMM → first active is Jun 2026
        y, m = _first_active_sr3(date(2026, 3, 19))
        assert (y, m) == (2026, 6)


# ---------------------------------------------------------------------------
# Month code mapping
# ---------------------------------------------------------------------------

class TestMonthCodes:
    def test_all_months_covered(self):
        assert set(_MONTH_CODE_TO_NUM.keys()) == {"F","G","H","J","K","M","N","Q","U","V","X","Z"}
        assert set(_MONTH_CODE_TO_NUM.values()) == set(range(1, 13))

    def test_round_trip(self):
        for code, num in _MONTH_CODE_TO_NUM.items():
            assert _NUM_TO_MONTH_CODE[num] == code


# ---------------------------------------------------------------------------
# fetch_sr3_strip / fetch_sr1_strip — mocked network
# ---------------------------------------------------------------------------

def _mock_price(ticker: str):
    """Return a synthetic price for any SR1/SR3 ticker."""
    return 96.25


class TestFetchSR3Strip:
    @patch("quantliblab.data.loaders.sofr_futures_loader._fetch_price", side_effect=_mock_price)
    def test_returns_n_contracts(self, mock_fetch):
        contracts = fetch_sr3_strip(valuation_date=date(2026, 4, 5), n_contracts=4)
        assert len(contracts) == 4

    @patch("quantliblab.data.loaders.sofr_futures_loader._fetch_price", side_effect=_mock_price)
    def test_first_contract_is_jun2026(self, mock_fetch):
        contracts = fetch_sr3_strip(valuation_date=date(2026, 4, 5), n_contracts=1)
        assert contracts[0].ticker == "SR3M26.CME"
        assert contracts[0].delivery_month == 6
        assert contracts[0].delivery_year  == 2026

    @patch("quantliblab.data.loaders.sofr_futures_loader._fetch_price", side_effect=_mock_price)
    def test_implied_rate(self, mock_fetch):
        contracts = fetch_sr3_strip(valuation_date=date(2026, 4, 5), n_contracts=1)
        c = contracts[0]
        assert abs(c.implied_rate - (100.0 - 96.25) / 100.0) < 1e-10

    @patch("quantliblab.data.loaders.sofr_futures_loader._fetch_price", side_effect=_mock_price)
    def test_quarterly_sequence(self, mock_fetch):
        contracts = fetch_sr3_strip(valuation_date=date(2026, 4, 5), n_contracts=4)
        months = [c.delivery_month for c in contracts]
        assert months == [6, 9, 12, 3]  # Jun, Sep, Dec, Mar

    @patch("quantliblab.data.loaders.sofr_futures_loader._fetch_price", side_effect=_mock_price)
    def test_contract_type(self, mock_fetch):
        contracts = fetch_sr3_strip(valuation_date=date(2026, 4, 5), n_contracts=2)
        assert all(c.contract_type == "SR3" for c in contracts)


class TestFetchSR1Strip:
    @patch("quantliblab.data.loaders.sofr_futures_loader._fetch_price", side_effect=_mock_price)
    def test_returns_n_contracts(self, mock_fetch):
        contracts = fetch_sr1_strip(valuation_date=date(2026, 4, 5), n_contracts=6)
        assert len(contracts) == 6

    @patch("quantliblab.data.loaders.sofr_futures_loader._fetch_price", side_effect=_mock_price)
    def test_first_contract(self, mock_fetch):
        # April is still active on 2026-04-05 (not the last day)
        contracts = fetch_sr1_strip(valuation_date=date(2026, 4, 5), n_contracts=1)
        assert contracts[0].delivery_month == 4
        assert contracts[0].delivery_year  == 2026

    @patch("quantliblab.data.loaders.sofr_futures_loader._fetch_price", side_effect=_mock_price)
    def test_ref_period_is_full_month(self, mock_fetch):
        contracts = fetch_sr1_strip(valuation_date=date(2026, 4, 5), n_contracts=1)
        c = contracts[0]
        assert c.ref_start == date(2026, 4, 1)
        assert c.ref_end   == date(2026, 4, 30)

    @patch("quantliblab.data.loaders.sofr_futures_loader._fetch_price", side_effect=_mock_price)
    def test_consecutive_months(self, mock_fetch):
        contracts = fetch_sr1_strip(valuation_date=date(2026, 4, 5), n_contracts=4)
        months = [c.delivery_month for c in contracts]
        assert months == [4, 5, 6, 7]

    @patch("quantliblab.data.loaders.sofr_futures_loader._fetch_price", side_effect=_mock_price)
    def test_contract_type(self, mock_fetch):
        contracts = fetch_sr1_strip(valuation_date=date(2026, 4, 5), n_contracts=2)
        assert all(c.contract_type == "SR1" for c in contracts)
