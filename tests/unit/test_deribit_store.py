"""
Unit tests for data/loaders/deribit_store.py.

Tests verify that save_index() and save_perp() call the Deribit loader
and write the correct columns to the CSV store.  Both the loader and
the store are mocked — no live API calls, no disk I/O.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, call, patch

import pytest

from quantliblab.data.loaders.deribit_store import save_index, save_perp
from quantliblab.instruments.crypto import BTCPerpetual


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_perp(**kwargs) -> BTCPerpetual:
    defaults = dict(
        instrument_name="BTC-PERPETUAL",
        mark_price=82050.0,
        index_price=82000.0,
        last_price=82040.0,
        bid=82030.0,
        ask=82070.0,
        volume_usd=2_000_000_000.0,
        open_interest=953_000.0,
        current_funding=0.0001,
        funding_8h=0.00012,
    )
    defaults.update(kwargs)
    return BTCPerpetual(**defaults)


# ---------------------------------------------------------------------------
# save_index
# ---------------------------------------------------------------------------

class TestSaveIndex:
    def test_calls_fetch_index_price(self):
        with patch("quantliblab.data.loaders.deribit_store.fetch_index_price", return_value=82000.0) as mock_fetch, \
             patch("quantliblab.data.loaders.deribit_store.write") as mock_write:
            save_index("BTC", as_of=date(2025, 3, 28))
        mock_fetch.assert_called_once_with("BTC")

    def test_writes_to_btc_index(self):
        with patch("quantliblab.data.loaders.deribit_store.fetch_index_price", return_value=82000.0), \
             patch("quantliblab.data.loaders.deribit_store.write") as mock_write:
            save_index("BTC", as_of=date(2025, 3, 28))
        mock_write.assert_called_once_with(
            asset_class="crypto",
            dataset="btc_index",
            date_=date(2025, 3, 28),
            data={"index_price": 82000.0},
        )

    def test_writes_to_eth_index(self):
        with patch("quantliblab.data.loaders.deribit_store.fetch_index_price", return_value=2100.0), \
             patch("quantliblab.data.loaders.deribit_store.write") as mock_write:
            save_index("ETH", as_of=date(2025, 3, 28))
        mock_write.assert_called_once_with(
            asset_class="crypto",
            dataset="eth_index",
            date_=date(2025, 3, 28),
            data={"index_price": 2100.0},
        )

    def test_price_rounded_to_2dp(self):
        with patch("quantliblab.data.loaders.deribit_store.fetch_index_price", return_value=82000.12345), \
             patch("quantliblab.data.loaders.deribit_store.write") as mock_write:
            save_index("BTC", as_of=date(2025, 3, 28))
        written_data = mock_write.call_args[1]["data"]
        assert written_data["index_price"] == 82000.12


# ---------------------------------------------------------------------------
# save_perp
# ---------------------------------------------------------------------------

class TestSavePerp:
    def test_calls_fetch_perpetual(self):
        perp = _make_perp()
        with patch("quantliblab.data.loaders.deribit_store.fetch_perpetual", return_value=perp) as mock_fetch, \
             patch("quantliblab.data.loaders.deribit_store.write"):
            save_perp("BTC", as_of=date(2025, 3, 28))
        mock_fetch.assert_called_once_with("BTC")

    def test_writes_to_btc_perp(self):
        perp = _make_perp()
        with patch("quantliblab.data.loaders.deribit_store.fetch_perpetual", return_value=perp), \
             patch("quantliblab.data.loaders.deribit_store.write") as mock_write:
            save_perp("BTC", as_of=date(2025, 3, 28))
        call_kwargs = mock_write.call_args[1]
        assert call_kwargs["asset_class"] == "crypto"
        assert call_kwargs["dataset"] == "btc_perp"
        assert call_kwargs["date_"] == date(2025, 3, 28)

    def test_writes_to_eth_perp(self):
        from quantliblab.instruments.crypto import ETHPerpetual
        perp = ETHPerpetual(
            instrument_name="ETH-PERPETUAL",
            mark_price=2105.0,
            index_price=2100.0,
            last_price=2103.0,
            bid=2102.0,
            ask=2108.0,
            volume_usd=500_000_000.0,
            open_interest=20_000.0,
            current_funding=0.00005,
        )
        with patch("quantliblab.data.loaders.deribit_store.fetch_perpetual", return_value=perp), \
             patch("quantliblab.data.loaders.deribit_store.write") as mock_write:
            save_perp("ETH", as_of=date(2025, 3, 28))
        assert mock_write.call_args[1]["dataset"] == "eth_perp"

    def test_correct_columns_written(self):
        perp = _make_perp(mark_price=82050.0, index_price=82000.0,
                          current_funding=0.0001, funding_8h=0.00012,
                          open_interest=953_000.0)
        with patch("quantliblab.data.loaders.deribit_store.fetch_perpetual", return_value=perp), \
             patch("quantliblab.data.loaders.deribit_store.write") as mock_write:
            save_perp("BTC", as_of=date(2025, 3, 28))
        data = mock_write.call_args[1]["data"]
        assert set(data.keys()) == {
            "mark_price", "index_price", "basis_usd",
            "current_funding", "funding_8h", "open_interest",
        }

    def test_basis_computed(self):
        perp = _make_perp(mark_price=82050.0, index_price=82000.0)
        with patch("quantliblab.data.loaders.deribit_store.fetch_perpetual", return_value=perp), \
             patch("quantliblab.data.loaders.deribit_store.write") as mock_write:
            save_perp("BTC", as_of=date(2025, 3, 28))
        data = mock_write.call_args[1]["data"]
        assert data["basis_usd"] == pytest.approx(50.0, abs=0.01)

    def test_funding_8h_none_stored_as_empty_string(self):
        perp = _make_perp(funding_8h=None)
        with patch("quantliblab.data.loaders.deribit_store.fetch_perpetual", return_value=perp), \
             patch("quantliblab.data.loaders.deribit_store.write") as mock_write:
            save_perp("BTC", as_of=date(2025, 3, 28))
        data = mock_write.call_args[1]["data"]
        assert data["funding_8h"] == ""

    def test_funding_8h_present(self):
        perp = _make_perp(funding_8h=0.00012)
        with patch("quantliblab.data.loaders.deribit_store.fetch_perpetual", return_value=perp), \
             patch("quantliblab.data.loaders.deribit_store.write") as mock_write:
            save_perp("BTC", as_of=date(2025, 3, 28))
        data = mock_write.call_args[1]["data"]
        assert data["funding_8h"] == pytest.approx(0.00012)
