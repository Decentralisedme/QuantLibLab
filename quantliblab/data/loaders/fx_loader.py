"""
FX spot loader via yfinance.

Fetches daily closing spot rates for major pairs:
  EURUSD, GBPUSD, USDJPY, USDCHF, AUDUSD, USDCAD

yfinance ticker format for FX: "EURUSD=X"

Returns closing price (mid-market proxy — bid/ask not available on
free feeds). Good enough for curve discounting and indicative pricing.

No API key required.
"""
from __future__ import annotations

from datetime import date, timedelta

import yfinance as yf

# Map from our label to yfinance ticker
FX_TICKERS: dict[str, str] = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "AUDUSD": "AUDUSD=X",
    "USDCAD": "USDCAD=X",
}


def fetch_fx_spot(
    pairs: list[str],
    start: date,
    end: date,
) -> dict[str, list[tuple[date, dict[str, float]]]]:
    """
    Fetch daily closing FX spot rates for the given pairs.

    Parameters
    ----------
    pairs : list of pair names, e.g. ["EURUSD", "GBPUSD"]
    start : first date (inclusive)
    end   : last date (inclusive)

    Returns
    -------
    dict of pair -> list of (date, {"spot": rate}) tuples

    Raises
    ------
    ValueError if an unknown pair is requested
    """
    unknown = [p for p in pairs if p not in FX_TICKERS]
    if unknown:
        raise ValueError(
            f"Unknown FX pairs: {unknown}. "
            f"Supported: {sorted(FX_TICKERS.keys())}"
        )

    tickers = [FX_TICKERS[p] for p in pairs]

    # yfinance end date is exclusive — add one day
    yf_end = end + timedelta(days=1)

    raw = yf.download(
        tickers=tickers,
        start=start.isoformat(),
        end=yf_end.isoformat(),
        auto_adjust=True,
        progress=False,
    )

    result: dict[str, list[tuple[date, dict[str, float]]]] = {p: [] for p in pairs}

    if raw.empty:
        return result

    # yfinance returns MultiIndex columns when multiple tickers requested
    # Single ticker returns flat columns
    for pair, ticker in zip(pairs, tickers):
        try:
            # yfinance always returns MultiIndex columns (Price, Ticker)
            close_series = raw["Close"][ticker]
        except KeyError:
            continue

        for ts, val in close_series.items():
            try:
                fval = float(val)
            except (TypeError, ValueError):
                continue
            if fval != fval:   # NaN check
                continue
            obs_date = ts.date() if hasattr(ts, "date") else ts
            result[pair].append((obs_date, {"spot": fval}))

    return result


def fetch_all_fx(
    start: date,
    end: date,
) -> dict[str, list[tuple[date, dict[str, float]]]]:
    """Fetch all supported FX pairs for the given date range."""
    return fetch_fx_spot(list(FX_TICKERS.keys()), start, end)
