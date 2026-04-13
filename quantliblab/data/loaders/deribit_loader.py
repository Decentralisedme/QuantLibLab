"""
Deribit market data loader.

Fetches live snapshots from the Deribit REST API (v2).
No API key required — all endpoints used are public.

Data fetched:
  - BTC/ETH spot index price       get_index_price
  - Futures book summary           get_book_summary_by_currency (kind=future)
  - Options book summary + Greeks  get_book_summary_by_currency (kind=option)
                                   + per-instrument ticker for Greeks
  - Perpetual ticker               ticker (BTC-PERPETUAL / ETH-PERPETUAL)
  - Funding rate history           get_funding_rate_history

All prices returned by Deribit are in USD for futures/perps.
Options prices are in BTC/ETH (base currency) — converted to USD using the
index price at fetch time.

API docs: https://docs.deribit.com/
"""
from __future__ import annotations

import time
from datetime import date, datetime, timezone
from typing import Any

import requests

from quantliblab.instruments.crypto.futures import BTCFuture, ETHFuture, CryptoFuture
from quantliblab.instruments.crypto.options import BTCOption, ETHOption, CryptoOption, OptionKind
from quantliblab.instruments.crypto.perpetuals import BTCPerpetual, ETHPerpetual, Perpetual

_BASE = "https://www.deribit.com/api/v2/public"
_TIMEOUT = 15          # seconds per request
_RETRY_ATTEMPTS = 3
_RETRY_DELAY = 2.0     # seconds between retries


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(endpoint: str, params: dict[str, Any]) -> Any:
    """
    GET a Deribit public endpoint.  Returns the 'result' field of the JSON
    response.  Retries on transient network errors.
    """
    url = f"{_BASE}/{endpoint}"
    for attempt in range(1, _RETRY_ATTEMPTS + 1):
        try:
            resp = requests.get(url, params=params, timeout=_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
            if "error" in payload:
                raise RuntimeError(
                    f"Deribit API error on {endpoint}: {payload['error']}"
                )
            return payload["result"]
        except requests.RequestException as exc:
            if attempt == _RETRY_ATTEMPTS:
                raise RuntimeError(
                    f"Deribit request to {endpoint} failed after "
                    f"{_RETRY_ATTEMPTS} attempts: {exc}"
                ) from exc
            time.sleep(_RETRY_DELAY)
    return None  # unreachable


def _ms_to_date(ms: int) -> date:
    """Convert millisecond epoch timestamp to a UTC date."""
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc).date()


def _ms_to_datetime(ms: int) -> datetime:
    """Convert millisecond epoch timestamp to a UTC datetime."""
    return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)


# ---------------------------------------------------------------------------
# Instrument metadata (expiry dates)
# ---------------------------------------------------------------------------

def _fetch_instrument_expiries(currency: str, kind: str) -> dict[str, date]:
    """
    Return {instrument_name: expiry_date} for all active instruments of
    the given kind ('future' or 'option') for the given currency.
    Uses get_instruments which includes expiration_timestamp.
    """
    results = _get("get_instruments", {"currency": currency, "kind": kind})
    return {
        r["instrument_name"]: _ms_to_date(r["expiration_timestamp"])
        for r in results
        if r.get("is_active", True)
    }


# ---------------------------------------------------------------------------
# Spot index
# ---------------------------------------------------------------------------

def fetch_index_price(underlying: str) -> float:
    """
    Fetch the current composite index price for BTC or ETH.

    Parameters
    ----------
    underlying : "BTC" or "ETH"

    Returns
    -------
    Index price in USD.
    """
    index_name = f"{underlying.lower()}_usd"
    result = _get("get_index_price", {"index_name": index_name})
    return float(result["index_price"])


# ---------------------------------------------------------------------------
# Futures
# ---------------------------------------------------------------------------

def fetch_futures(underlying: str) -> list[CryptoFuture]:
    """
    Fetch all active futures contracts for BTC or ETH.

    Returns a list of BTCFuture or ETHFuture dataclass instances,
    one per active contract (including the perpetual is excluded here —
    use fetch_perpetual() for that).

    Parameters
    ----------
    underlying : "BTC" or "ETH"
    """
    currency = underlying.upper()
    expiries = _fetch_instrument_expiries(currency, "future")
    summaries = _get(
        "get_book_summary_by_currency",
        {"currency": currency, "kind": "future"},
    )
    now = datetime.now(tz=timezone.utc)
    cls = BTCFuture if currency == "BTC" else ETHFuture

    futures: list[CryptoFuture] = []
    for s in summaries:
        name = s["instrument_name"]
        if "PERPETUAL" in name:
            continue  # handled by fetch_perpetual()

        expiry = expiries.get(name)
        if expiry is None:
            continue

        futures.append(cls(
            instrument_name=name,
            expiry_date=expiry,
            mark_price=float(s["mark_price"]),
            last_price=float(s["last"]) if s.get("last") is not None else None,
            bid=float(s["bid_price"]) if s.get("bid_price") is not None else None,
            ask=float(s["ask_price"]) if s.get("ask_price") is not None else None,
            volume_usd=float(s.get("volume_usd", 0.0)),
            open_interest=float(s.get("open_interest", 0.0)),
            timestamp=now,
        ))

    futures.sort(key=lambda f: f.expiry_date)
    return futures


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------

def fetch_options(underlying: str) -> list[CryptoOption]:
    """
    Fetch all active options contracts for BTC or ETH.

    This calls get_book_summary_by_currency (kind=option) for the full
    snapshot, then fetch_option_greeks() individually for any contract
    you need detailed Greeks on.  Greeks are NOT included in the book
    summary — use fetch_option_ticker() for a single contract.

    Parameters
    ----------
    underlying : "BTC" or "ETH"

    Returns
    -------
    List of BTCOption or ETHOption instances (Greeks will be None).
    For Greeks, call fetch_option_ticker() on the specific instrument.
    """
    currency = underlying.upper()
    expiries = _fetch_instrument_expiries(currency, "option")
    summaries = _get(
        "get_book_summary_by_currency",
        {"currency": currency, "kind": "option"},
    )
    index_price = fetch_index_price(underlying)
    now = datetime.now(tz=timezone.utc)
    cls = BTCOption if currency == "BTC" else ETHOption

    options: list[CryptoOption] = []
    for s in summaries:
        name = s["instrument_name"]
        expiry = expiries.get(name)
        if expiry is None:
            continue

        # Parse strike and kind from name: "BTC-28MAR25-80000-C"
        parts = name.split("-")
        if len(parts) < 4:
            continue
        try:
            strike = float(parts[2])
        except ValueError:
            continue
        kind = OptionKind.CALL if parts[3] == "C" else OptionKind.PUT

        # Deribit options prices are in BTC/ETH — convert to USD
        mark_usd = float(s["mark_price"]) * index_price
        bid_usd = float(s["bid_price"]) * index_price if s.get("bid_price") else None
        ask_usd = float(s["ask_price"]) * index_price if s.get("ask_price") else None

        mark_iv_raw = s.get("mark_iv")
        mark_iv = float(mark_iv_raw) / 100.0 if mark_iv_raw is not None else 0.0

        options.append(cls(
            instrument_name=name,
            expiry_date=expiry,
            strike=strike,
            kind=kind,
            mark_price=mark_usd,
            mark_iv=mark_iv,
            bid=bid_usd,
            ask=ask_usd,
            volume_usd=float(s.get("volume_usd", 0.0)),
            open_interest=float(s.get("open_interest", 0.0)),
            timestamp=now,
        ))

    return options


def fetch_option_ticker(instrument_name: str) -> CryptoOption:
    """
    Fetch a single option contract including Greeks from the ticker endpoint.

    Use this when you need delta/gamma/vega/theta for a specific contract.

    Parameters
    ----------
    instrument_name : e.g. "BTC-28MAR25-80000-C"

    Returns
    -------
    BTCOption or ETHOption with Greeks populated.
    """
    result = _get("ticker", {"instrument_name": instrument_name})
    parts = instrument_name.split("-")
    underlying = parts[0]   # "BTC" or "ETH"
    strike = float(parts[2])
    kind = OptionKind.CALL if parts[3] == "C" else OptionKind.PUT

    index_price = float(result.get("index_price", 0.0))
    mark_usd = float(result["mark_price"]) * index_price

    greeks = result.get("greeks", {})

    expiry_ms = None
    # expiry not in ticker — parse from instrument name
    # format: BTC-28MAR25 → need get_instruments for the exact timestamp
    # Use a lightweight parse: not needed for greeks-only use case
    # Fall back to None — caller can enrich if needed
    expiry_date = _parse_expiry_from_name(instrument_name)

    cls = BTCOption if underlying == "BTC" else ETHOption
    now = datetime.now(tz=timezone.utc)

    return cls(
        instrument_name=instrument_name,
        expiry_date=expiry_date,
        strike=strike,
        kind=kind,
        mark_price=mark_usd,
        mark_iv=float(result.get("mark_iv", 0.0)) / 100.0,
        bid_iv=float(result["bid_iv"]) / 100.0 if result.get("bid_iv") else None,
        ask_iv=float(result["ask_iv"]) / 100.0 if result.get("ask_iv") else None,
        bid=float(result["best_bid_price"]) * index_price if result.get("best_bid_price") else None,
        ask=float(result["best_ask_price"]) * index_price if result.get("best_ask_price") else None,
        volume_usd=float(result.get("stats", {}).get("volume_usd", 0.0)),
        open_interest=float(result.get("open_interest", 0.0)),
        delta=greeks.get("delta"),
        gamma=greeks.get("gamma"),
        vega=greeks.get("vega"),
        theta=greeks.get("theta"),
        rho=greeks.get("rho"),
        timestamp=now,
    )


def _parse_expiry_from_name(instrument_name: str) -> date:
    """
    Parse expiry date from Deribit instrument name.

    Deribit format: BTC-28MAR25-80000-C  or  BTC-28MAR25 (futures)
    The date token is always the second segment: "28MAR25"
    Maps to 28 March 2025.
    """
    import re
    _MONTH = {
        "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
        "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
    }
    parts = instrument_name.split("-")
    if len(parts) < 2:
        raise ValueError(f"Cannot parse expiry from: {instrument_name}")
    token = parts[1]  # e.g. "28MAR25"
    m = re.fullmatch(r"(\d{1,2})([A-Z]{3})(\d{2})", token)
    if not m:
        raise ValueError(f"Unrecognised date token '{token}' in {instrument_name}")
    day = int(m.group(1))
    month = _MONTH[m.group(2)]
    year = 2000 + int(m.group(3))
    return date(year, month, day)


# ---------------------------------------------------------------------------
# Perpetuals
# ---------------------------------------------------------------------------

def fetch_perpetual(underlying: str) -> Perpetual:
    """
    Fetch the current perpetual swap snapshot for BTC or ETH.

    Parameters
    ----------
    underlying : "BTC" or "ETH"

    Returns
    -------
    BTCPerpetual or ETHPerpetual with mark price, index price, funding rates.
    """
    instrument_name = f"{underlying.upper()}-PERPETUAL"
    result = _get("ticker", {"instrument_name": instrument_name})
    stats = result.get("stats", {})
    now = datetime.now(tz=timezone.utc)

    cls = BTCPerpetual if underlying.upper() == "BTC" else ETHPerpetual

    return cls(
        instrument_name=instrument_name,
        mark_price=float(result["mark_price"]),
        index_price=float(result["index_price"]),
        last_price=float(result["last_price"]) if result.get("last_price") else None,
        bid=float(result["best_bid_price"]) if result.get("best_bid_price") else None,
        ask=float(result["best_ask_price"]) if result.get("best_ask_price") else None,
        volume_usd=float(stats.get("volume_usd", 0.0)),
        open_interest=float(result.get("open_interest", 0.0)),
        current_funding=float(result.get("current_funding", 0.0)),
        funding_8h=float(result["funding_8h"]) if result.get("funding_8h") else None,
        interest_rate=float(result.get("interest_value", 0.0)),
        timestamp=now,
    )


# ---------------------------------------------------------------------------
# Funding rate history
# ---------------------------------------------------------------------------

def fetch_funding_history(
    underlying: str,
    start: datetime,
    end: datetime,
) -> list[tuple[datetime, float]]:
    """
    Fetch hourly historical funding rates for the perpetual swap.

    Parameters
    ----------
    underlying : "BTC" or "ETH"
    start : Start datetime (UTC)
    end   : End datetime (UTC)

    Returns
    -------
    List of (datetime, 8h_funding_rate) tuples, oldest first.
    Funding rate is in decimal (e.g. 0.0001 = 0.01% per 8h).
    """
    instrument_name = f"{underlying.upper()}-PERPETUAL"
    start_ms = int(start.timestamp() * 1000)
    end_ms = int(end.timestamp() * 1000)

    results = _get(
        "get_funding_rate_history",
        {
            "instrument_name": instrument_name,
            "start_timestamp": start_ms,
            "end_timestamp": end_ms,
        },
    )

    return [
        (_ms_to_datetime(r["timestamp"]), float(r["interest_8h"]))
        for r in results
    ]
