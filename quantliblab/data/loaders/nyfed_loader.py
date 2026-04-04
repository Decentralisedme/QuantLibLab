"""
SOFR compounded averages loader — via FRED (NY Fed publishes, FRED mirrors).

Fetches:
  SOFR 30-day compounded average  : FRED series SOFR30DAYAVG
  SOFR 90-day compounded average  : FRED series SOFR90DAYAVG  (best 3M proxy)
  SOFR 180-day compounded average : FRED series SOFR180DAYAVG

These are backward-looking compounded averages published by the NY Fed
and available through FRED with a free API key.

Note: these are NOT the same as CME Term SOFR (forward-looking).
They are the correct rates for SOFR-linked loans and OIS discounting.

Rates returned as decimals (0.0530 for 5.30%).
"""
from __future__ import annotations

from datetime import date

from .fred_loader import _fetch_series, _parse_observations

_SERIES = {
    "30D":  "SOFR30DAYAVG",
    "90D":  "SOFR90DAYAVG",
    "180D": "SOFR180DAYAVG",
}


def fetch_sofr_averages(
    start: date,
    end: date,
) -> list[tuple[date, dict[str, float]]]:
    """
    Fetch SOFR 30/90/180-day compounded averages from FRED.

    Returns list of (date, {"30D": rate, "90D": rate, "180D": rate})
    where rates are decimals.
    """
    # Fetch all three series
    per_tenor: dict[str, dict[date, float]] = {}
    for label, series_id in _SERIES.items():
        records = _parse_observations(_fetch_series(series_id, start, end), label)
        per_tenor[label] = {d: row[label] for d, row in records}

    # Merge by date — only include dates present in all three series
    all_dates = set(per_tenor["30D"]) & set(per_tenor["90D"]) & set(per_tenor["180D"])

    result = []
    for d in sorted(all_dates):
        row = {
            "30D":  per_tenor["30D"][d],
            "90D":  per_tenor["90D"][d],
            "180D": per_tenor["180D"][d],
        }
        result.append((d, row))

    return result
