"""
Unit tests for data loaders.

All external HTTP calls are mocked — no API key or network required.
"""
from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantliblab.data.loaders.fred_loader import (
    fetch_sofr_on, fetch_sonia_on, fetch_estr_on, _parse_observations,
)
from quantliblab.data.loaders.nyfed_loader import fetch_sofr_averages
from quantliblab.data.loaders.fx_loader import fetch_fx_spot


# ---------------------------------------------------------------------------
# FRED loader
# ---------------------------------------------------------------------------

FRED_SOFR_OBSERVATIONS = [
    {"date": "2025-01-02", "value": "4.3100"},
    {"date": "2025-01-03", "value": "4.3200"},
    {"date": "2025-01-04", "value": "."},       # missing value
]


class TestFredLoader:
    def _mock_fetch(self, observations):
        """Patch _fetch_series to return given observations."""
        with patch(
            "quantliblab.data.loaders.fred_loader._fetch_series",
            return_value=observations,
        ) as mock:
            yield mock

    def test_sofr_on_parses_correctly(self):
        with patch(
            "quantliblab.data.loaders.fred_loader._fetch_series",
            return_value=FRED_SOFR_OBSERVATIONS,
        ):
            result = fetch_sofr_on(date(2025, 1, 2), date(2025, 1, 4))

        assert len(result) == 2   # missing value skipped
        assert result[0] == (date(2025, 1, 2), {"ON": 0.0431})
        assert result[1] == (date(2025, 1, 3), {"ON": 0.0432})

    def test_missing_values_skipped(self):
        obs = [{"date": "2025-01-02", "value": "."}]
        result = _parse_observations(obs, "ON")
        assert result == []

    def test_percentage_to_decimal_conversion(self):
        obs = [{"date": "2025-01-02", "value": "5.30"}]
        result = _parse_observations(obs, "ON")
        assert abs(result[0][1]["ON"] - 0.0530) < 1e-10

    def test_sonia_on(self):
        with patch(
            "quantliblab.data.loaders.fred_loader._fetch_series",
            return_value=[{"date": "2025-01-02", "value": "4.4570"}],
        ):
            result = fetch_sonia_on(date(2025, 1, 2), date(2025, 1, 2))
        assert abs(result[0][1]["ON"] - 0.044570) < 1e-8

    def test_estr_on(self):
        with patch(
            "quantliblab.data.loaders.fred_loader._fetch_series",
            return_value=[{"date": "2025-01-02", "value": "2.4180"}],
        ):
            result = fetch_estr_on(date(2025, 1, 2), date(2025, 1, 2))
        assert abs(result[0][1]["ON"] - 0.024180) < 1e-8

    def test_no_api_key_raises(self, monkeypatch):
        monkeypatch.delenv("FRED_API_KEY", raising=False)
        with patch("quantliblab.data.loaders.fred_loader.os.getenv", return_value=""):
            from quantliblab.data.loaders.fred_loader import _api_key
            with pytest.raises(EnvironmentError, match="FRED_API_KEY"):
                _api_key()


# ---------------------------------------------------------------------------
# SOFR averages (via FRED)
# ---------------------------------------------------------------------------

SOFR_AVG_OBS = [{"date": "2025-01-02", "value": "4.3500"}]


class TestSofrAverages:
    def test_returns_all_tenors(self):
        with patch(
            "quantliblab.data.loaders.nyfed_loader._fetch_series",
            return_value=SOFR_AVG_OBS,
        ):
            result = fetch_sofr_averages(date(2025, 1, 2), date(2025, 1, 2))

        assert len(result) == 1
        d, row = result[0]
        assert d == date(2025, 1, 2)
        assert set(row.keys()) == {"30D", "90D", "180D"}

    def test_decimal_conversion(self):
        with patch(
            "quantliblab.data.loaders.nyfed_loader._fetch_series",
            return_value=SOFR_AVG_OBS,
        ):
            result = fetch_sofr_averages(date(2025, 1, 2), date(2025, 1, 2))
        assert abs(result[0][1]["90D"] - 0.04350) < 1e-8


# ---------------------------------------------------------------------------
# FX loader
# ---------------------------------------------------------------------------

def _make_fx_df(ticker: str, dates: list[str], values: list[float]) -> pd.DataFrame:
    """Build a minimal yfinance-style MultiIndex DataFrame."""
    idx = pd.DatetimeIndex(dates, name="Date")
    cols = pd.MultiIndex.from_tuples(
        [("Close", ticker), ("High", ticker), ("Low", ticker),
         ("Open", ticker), ("Volume", ticker)],
        names=["Price", "Ticker"],
    )
    data = {
        ("Close", ticker): values,
        ("High",  ticker): values,
        ("Low",   ticker): values,
        ("Open",  ticker): values,
        ("Volume",ticker): [0] * len(values),
    }
    return pd.DataFrame(data, index=idx, columns=cols)


class TestFxLoader:
    def test_single_pair(self):
        df = _make_fx_df("EURUSD=X", ["2025-01-02", "2025-01-03"], [1.0850, 1.0860])
        with patch("quantliblab.data.loaders.fx_loader.yf.download", return_value=df):
            result = fetch_fx_spot(["EURUSD"], date(2025, 1, 2), date(2025, 1, 3))
        assert len(result["EURUSD"]) == 2
        assert abs(result["EURUSD"][0][1]["spot"] - 1.0850) < 1e-6

    def test_empty_response_returns_empty(self):
        empty = pd.DataFrame()
        with patch("quantliblab.data.loaders.fx_loader.yf.download", return_value=empty):
            result = fetch_fx_spot(["EURUSD"], date(2025, 1, 2), date(2025, 1, 3))
        assert result["EURUSD"] == []

    def test_unknown_pair_raises(self):
        with pytest.raises(ValueError, match="Unknown FX pairs"):
            fetch_fx_spot(["XYZABC"], date(2025, 1, 2), date(2025, 1, 3))

    def test_multiple_pairs(self):
        df_eur = _make_fx_df("EURUSD=X", ["2025-01-02"], [1.085])
        df_gbp = _make_fx_df("GBPUSD=X", ["2025-01-02"], [1.290])
        # Merge into one DataFrame as yfinance would return
        combined = pd.concat([df_eur, df_gbp], axis=1)
        with patch("quantliblab.data.loaders.fx_loader.yf.download", return_value=combined):
            result = fetch_fx_spot(
                ["EURUSD", "GBPUSD"], date(2025, 1, 2), date(2025, 1, 2)
            )
        assert len(result["EURUSD"]) == 1
        assert len(result["GBPUSD"]) == 1
