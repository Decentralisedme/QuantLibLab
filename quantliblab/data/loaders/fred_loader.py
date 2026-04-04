"""
FRED loader — overnight rates via the FRED REST API.

Fetches:
  SOFR O/N    : series SOFR
  SONIA O/N   : series IUDSOIA
  ESTR O/N    : series ECBESTRVOLWGTTRMDMNRT

API key is read from the environment variable FRED_API_KEY.
Load it via python-dotenv before calling this module.

FRED returns rates as percentages (e.g. 5.30 for 5.30%).
We store them as decimals (0.0530) to match quant finance convention.

Rate docs:
  https://fred.stlouisfed.org/series/SOFR
  https://fred.stlouisfed.org/series/IUDSOIA
  https://fred.stlouisfed.org/series/ECBESTRVOLWGTTRMDMNRT
"""
from __future__ import annotations

import os
import time
from datetime import date
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()

_FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"
_RETRY_ATTEMPTS = 3
_RETRY_DELAY = 2.0  # seconds

# FRED series IDs for each rate
SERIES = {
    "sofr_on":  "SOFR",
    "sonia_on": "IUDSOIA",
    "estr_on":  "ECBESTRVOLWGTTRMDMNRT",
}


def _api_key() -> str:
    key = os.getenv("FRED_API_KEY", "")
    if not key:
        raise EnvironmentError(
            "FRED_API_KEY is not set. Add it to your .env file."
        )
    return key


def _fetch_series(
    series_id: str,
    start: date,
    end: date,
) -> list[dict[str, Any]]:
    """
    Fetch observations for a single FRED series.
    Returns the raw list of {date, value} dicts from FRED.
    Retries on transient network errors.
    """
    params = {
        "series_id":         series_id,
        "observation_start": start.isoformat(),
        "observation_end":   end.isoformat(),
        "api_key":           _api_key(),
        "file_type":         "json",
        "frequency":         "d",
        "aggregation_method": "avg",
    }

    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(_FRED_BASE, params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()["observations"]
        except requests.RequestException as e:
            if attempt == _RETRY_ATTEMPTS:
                raise RuntimeError(
                    f"FRED request for {series_id} failed after "
                    f"{_RETRY_ATTEMPTS} attempts: {e}"
                ) from e
            time.sleep(_RETRY_DELAY)

    return []   # unreachable but satisfies type checker


def _parse_observations(
    observations: list[dict[str, Any]],
    field_name: str,
) -> list[tuple[date, dict[str, float]]]:
    """
    Convert FRED observations to (date, {field_name: value}) tuples.

    FRED uses "." for missing values — those are skipped.
    Rates are divided by 100 to convert percentage -> decimal.
    """
    result = []
    for obs in observations:
        raw = obs.get("value", ".")
        if raw == ".":
            continue
        try:
            value = float(raw) / 100.0
        except ValueError:
            continue
        obs_date = date.fromisoformat(obs["date"])
        result.append((obs_date, {field_name: value}))
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_sofr_on(start: date, end: date) -> list[tuple[date, dict[str, float]]]:
    """Fetch SOFR overnight rate from FRED."""
    obs = _fetch_series(SERIES["sofr_on"], start, end)
    return _parse_observations(obs, "ON")


def fetch_sonia_on(start: date, end: date) -> list[tuple[date, dict[str, float]]]:
    """Fetch SONIA overnight rate from FRED."""
    obs = _fetch_series(SERIES["sonia_on"], start, end)
    return _parse_observations(obs, "ON")


def fetch_estr_on(start: date, end: date) -> list[tuple[date, dict[str, float]]]:
    """Fetch ESTR overnight rate from FRED."""
    obs = _fetch_series(SERIES["estr_on"], start, end)
    return _parse_observations(obs, "ON")
