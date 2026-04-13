"""
Unit tests for data/loaders/deribit_loader.py.

All tests mock the HTTP layer — no live API calls.
Fixtures are constructed from real Deribit API response shapes.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from quantliblab.data.loaders.deribit_loader import (
    _ms_to_date,
    _ms_to_datetime,
    _parse_expiry_from_name,
    fetch_index_price,
    fetch_futures,
    fetch_perpetual,
    fetch_funding_history,
    fetch_option_ticker,
)
from quantliblab.instruments.crypto import (
    BTCFuture,
    ETHFuture,
    BTCOption,
    ETHOption,
    BTCPerpetual,
    ETHPerpetual,
    OptionKind,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_response(result: object) -> MagicMock:
    """Build a mock requests.Response returning a Deribit-shaped JSON body."""
    mock_resp = MagicMock()
    mock_resp.raise_for_status.return_value = None
    mock_resp.json.return_value = {"jsonrpc": "2.0", "result": result}
    return mock_resp


# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------

class TestInternalUtils:
    def test_ms_to_date(self):
        # 2025-03-28 00:00:00 UTC = 1743120000000 ms
        assert _ms_to_date(1743120000000) == date(2025, 3, 28)

    def test_ms_to_datetime(self):
        dt = _ms_to_datetime(1743120000000)
        assert dt.year == 2025
        assert dt.month == 3
        assert dt.day == 28
        assert dt.tzinfo == timezone.utc

    def test_parse_expiry_btc_option(self):
        assert _parse_expiry_from_name("BTC-28MAR25-80000-C") == date(2025, 3, 28)

    def test_parse_expiry_btc_future(self):
        assert _parse_expiry_from_name("BTC-27JUN25") == date(2025, 6, 27)

    def test_parse_expiry_eth_option(self):
        assert _parse_expiry_from_name("ETH-26SEP25-2500-P") == date(2025, 9, 26)

    def test_parse_expiry_december(self):
        assert _parse_expiry_from_name("BTC-26DEC25") == date(2025, 12, 26)

    def test_parse_expiry_invalid_raises(self):
        with pytest.raises(ValueError):
            _parse_expiry_from_name("BTC-PERPETUAL")


# ---------------------------------------------------------------------------
# fetch_index_price
# ---------------------------------------------------------------------------

class TestFetchIndexPrice:
    def test_btc_index(self):
        result = {"index_price": 82000.0, "estimated_delivery_price": 82000.0}
        with patch("requests.get", return_value=_make_response(result)):
            price = fetch_index_price("BTC")
        assert price == pytest.approx(82000.0)

    def test_eth_index(self):
        result = {"index_price": 2100.0, "estimated_delivery_price": 2100.0}
        with patch("requests.get", return_value=_make_response(result)):
            price = fetch_index_price("ETH")
        assert price == pytest.approx(2100.0)

    def test_lowercase_underlying(self):
        result = {"index_price": 82000.0, "estimated_delivery_price": 82000.0}
        with patch("requests.get", return_value=_make_response(result)) as mock_get:
            fetch_index_price("btc")
        call_params = mock_get.call_args[1]["params"]
        assert call_params["index_name"] == "btc_usd"


# ---------------------------------------------------------------------------
# fetch_futures
# ---------------------------------------------------------------------------

_INSTRUMENTS_FUTURE = [
    {
        "instrument_name": "BTC-28MAR25",
        "expiration_timestamp": 1743120000000,  # 2025-03-28
        "is_active": True,
        "kind": "future",
    },
    {
        "instrument_name": "BTC-27JUN25",
        "expiration_timestamp": 1751030400000,  # 2025-06-27
        "is_active": True,
        "kind": "future",
    },
    {
        "instrument_name": "BTC-PERPETUAL",
        "expiration_timestamp": 32503680000000,
        "is_active": True,
        "kind": "future",
    },
]

_BOOK_SUMMARY_FUTURE = [
    {
        "instrument_name": "BTC-28MAR25",
        "mark_price": 82000.0,
        "last": 81950.0,
        "bid_price": 81980.0,
        "ask_price": 82020.0,
        "volume_usd": 500_000_000.0,
        "open_interest": 12000.0,
        "creation_timestamp": 1743000000000,
    },
    {
        "instrument_name": "BTC-27JUN25",
        "mark_price": 83000.0,
        "last": 82900.0,
        "bid_price": 82980.0,
        "ask_price": 83020.0,
        "volume_usd": 200_000_000.0,
        "open_interest": 8000.0,
        "creation_timestamp": 1743000000000,
    },
    {
        "instrument_name": "BTC-PERPETUAL",
        "mark_price": 82100.0,
        "last": 82050.0,
        "bid_price": 82080.0,
        "ask_price": 82120.0,
        "volume_usd": 2_000_000_000.0,
        "open_interest": 50000.0,
        "creation_timestamp": 1743000000000,
    },
]


class TestFetchFutures:
    def _mock_get(self, endpoint, params, **kwargs):
        if endpoint.endswith("get_instruments"):
            return _make_response(_INSTRUMENTS_FUTURE)
        if endpoint.endswith("get_book_summary_by_currency"):
            return _make_response(_BOOK_SUMMARY_FUTURE)
        raise AssertionError(f"Unexpected endpoint: {endpoint}")

    def test_returns_btc_futures(self):
        with patch("requests.get", side_effect=self._mock_get):
            futures = fetch_futures("BTC")
        assert len(futures) == 2  # perpetual excluded
        assert all(isinstance(f, BTCFuture) for f in futures)

    def test_perpetual_excluded(self):
        with patch("requests.get", side_effect=self._mock_get):
            futures = fetch_futures("BTC")
        names = [f.instrument_name for f in futures]
        assert "BTC-PERPETUAL" not in names

    def test_sorted_by_expiry(self):
        with patch("requests.get", side_effect=self._mock_get):
            futures = fetch_futures("BTC")
        assert futures[0].expiry_date < futures[1].expiry_date

    def test_fields_populated(self):
        with patch("requests.get", side_effect=self._mock_get):
            futures = fetch_futures("BTC")
        f = futures[0]  # BTC-28MAR25
        assert f.instrument_name == "BTC-28MAR25"
        assert f.mark_price == pytest.approx(82000.0)
        assert f.bid == pytest.approx(81980.0)
        assert f.ask == pytest.approx(82020.0)
        assert f.volume_usd == pytest.approx(500_000_000.0)

    def test_eth_returns_eth_futures(self):
        eth_instruments = [
            {
                "instrument_name": "ETH-28MAR25",
                "expiration_timestamp": 1743120000000,
                "is_active": True,
            }
        ]
        eth_summary = [
            {
                "instrument_name": "ETH-28MAR25",
                "mark_price": 2100.0,
                "last": 2095.0,
                "bid_price": 2098.0,
                "ask_price": 2102.0,
                "volume_usd": 50_000_000.0,
                "open_interest": 8000.0,
                "creation_timestamp": 1743000000000,
            }
        ]
        def mock_get(endpoint, params, **kwargs):
            if endpoint.endswith("get_instruments"):
                return _make_response(eth_instruments)
            return _make_response(eth_summary)

        with patch("requests.get", side_effect=mock_get):
            futures = fetch_futures("ETH")
        assert len(futures) == 1
        assert isinstance(futures[0], ETHFuture)


# ---------------------------------------------------------------------------
# fetch_perpetual
# ---------------------------------------------------------------------------

_PERP_TICKER = {
    "mark_price": 82050.0,
    "index_price": 82000.0,
    "last_price": 82040.0,
    "best_bid_price": 82030.0,
    "best_ask_price": 82070.0,
    "open_interest": 953_000_000.0,
    "current_funding": 0.0001,
    "funding_8h": 0.00012,
    "interest_value": 0.0,
    "stats": {"volume_usd": 2_000_000_000.0},
}


class TestFetchPerpetual:
    def test_btc_perpetual(self):
        with patch("requests.get", return_value=_make_response(_PERP_TICKER)):
            perp = fetch_perpetual("BTC")
        assert isinstance(perp, BTCPerpetual)
        assert perp.underlying == "BTC"

    def test_eth_perpetual(self):
        with patch("requests.get", return_value=_make_response(_PERP_TICKER)):
            perp = fetch_perpetual("ETH")
        assert isinstance(perp, ETHPerpetual)
        assert perp.underlying == "ETH"

    def test_fields(self):
        with patch("requests.get", return_value=_make_response(_PERP_TICKER)):
            perp = fetch_perpetual("BTC")
        assert perp.mark_price == pytest.approx(82050.0)
        assert perp.index_price == pytest.approx(82000.0)
        assert perp.current_funding == pytest.approx(0.0001)
        assert perp.funding_8h == pytest.approx(0.00012)

    def test_basis(self):
        with patch("requests.get", return_value=_make_response(_PERP_TICKER)):
            perp = fetch_perpetual("BTC")
        assert perp.basis == pytest.approx(50.0)

    def test_annualised_funding(self):
        with patch("requests.get", return_value=_make_response(_PERP_TICKER)):
            perp = fetch_perpetual("BTC")
        assert perp.annualised_funding == pytest.approx(0.0001 * 3 * 365)


# ---------------------------------------------------------------------------
# fetch_option_ticker
# ---------------------------------------------------------------------------

_OPTION_TICKER = {
    "mark_price": 0.0400,           # in BTC
    "index_price": 82000.0,
    "best_bid_price": 0.0390,
    "best_ask_price": 0.0410,
    "open_interest": 500.0,
    "mark_iv": 60.0,                # percent
    "bid_iv": 58.0,
    "ask_iv": 62.0,
    "greeks": {
        "delta": 0.55,
        "gamma": 0.00002,
        "vega": 150.0,
        "theta": -45.0,
        "rho": 12.0,
    },
    "stats": {"volume_usd": 10_000_000.0},
}


class TestFetchOptionTicker:
    def test_btc_call(self):
        with patch("requests.get", return_value=_make_response(_OPTION_TICKER)):
            opt = fetch_option_ticker("BTC-28MAR25-80000-C")
        assert isinstance(opt, BTCOption)
        assert opt.kind == OptionKind.CALL
        assert opt.strike == pytest.approx(80_000.0)

    def test_eth_put(self):
        with patch("requests.get", return_value=_make_response(_OPTION_TICKER)):
            opt = fetch_option_ticker("ETH-28MAR25-2500-P")
        assert isinstance(opt, ETHOption)
        assert opt.kind == OptionKind.PUT

    def test_price_converted_to_usd(self):
        # mark_price 0.04 BTC × 82000 USD = 3280 USD
        with patch("requests.get", return_value=_make_response(_OPTION_TICKER)):
            opt = fetch_option_ticker("BTC-28MAR25-80000-C")
        assert opt.mark_price == pytest.approx(0.04 * 82000.0)

    def test_iv_converted_to_decimal(self):
        with patch("requests.get", return_value=_make_response(_OPTION_TICKER)):
            opt = fetch_option_ticker("BTC-28MAR25-80000-C")
        assert opt.mark_iv == pytest.approx(0.60)
        assert opt.bid_iv == pytest.approx(0.58)
        assert opt.ask_iv == pytest.approx(0.62)

    def test_greeks_populated(self):
        with patch("requests.get", return_value=_make_response(_OPTION_TICKER)):
            opt = fetch_option_ticker("BTC-28MAR25-80000-C")
        assert opt.delta == pytest.approx(0.55)
        assert opt.gamma == pytest.approx(0.00002)
        assert opt.vega == pytest.approx(150.0)
        assert opt.theta == pytest.approx(-45.0)
        assert opt.rho == pytest.approx(12.0)


# ---------------------------------------------------------------------------
# fetch_funding_history
# ---------------------------------------------------------------------------

_FUNDING_HISTORY = [
    {"timestamp": 1743000000000, "index_price": 81500.0, "interest_8h": 0.0001, "interest_1h": 0.0, "prev_index_price": 81400.0},
    {"timestamp": 1743003600000, "index_price": 81600.0, "interest_8h": 0.00012, "interest_1h": 0.0, "prev_index_price": 81500.0},
    {"timestamp": 1743007200000, "index_price": 81700.0, "interest_8h": -0.00005, "interest_1h": 0.0, "prev_index_price": 81600.0},
]


class TestFetchFundingHistory:
    def test_returns_list_of_tuples(self):
        with patch("requests.get", return_value=_make_response(_FUNDING_HISTORY)):
            history = fetch_funding_history(
                "BTC",
                start=datetime(2025, 3, 26, tzinfo=timezone.utc),
                end=datetime(2025, 3, 27, tzinfo=timezone.utc),
            )
        assert len(history) == 3
        assert all(isinstance(dt, datetime) for dt, _ in history)
        assert all(isinstance(rate, float) for _, rate in history)

    def test_funding_rates_correct(self):
        with patch("requests.get", return_value=_make_response(_FUNDING_HISTORY)):
            history = fetch_funding_history(
                "BTC",
                start=datetime(2025, 3, 26, tzinfo=timezone.utc),
                end=datetime(2025, 3, 27, tzinfo=timezone.utc),
            )
        rates = [r for _, r in history]
        assert rates[0] == pytest.approx(0.0001)
        assert rates[1] == pytest.approx(0.00012)
        assert rates[2] == pytest.approx(-0.00005)

    def test_datetimes_are_utc(self):
        with patch("requests.get", return_value=_make_response(_FUNDING_HISTORY)):
            history = fetch_funding_history(
                "BTC",
                start=datetime(2025, 3, 26, tzinfo=timezone.utc),
                end=datetime(2025, 3, 27, tzinfo=timezone.utc),
            )
        for dt, _ in history:
            assert dt.tzinfo == timezone.utc
