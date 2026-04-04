"""
Daily market data fetch script.

Pulls yesterday's rates and FX spots, validates them, and saves to the
CSV store. Designed to be run as a cron job each weekday morning.

Usage:
    python scripts/fetch_daily_data.py              # fetch yesterday
    python scripts/fetch_daily_data.py 2024-01-15   # fetch specific date
    python scripts/fetch_daily_data.py 2024-01-01 2024-01-31  # date range
"""
from __future__ import annotations

import sys
from datetime import date, timedelta

from dotenv import load_dotenv

load_dotenv()

from quantliblab.data.loaders.fred_loader import fetch_sofr_on, fetch_sonia_on, fetch_estr_on
from quantliblab.data.loaders.nyfed_loader import fetch_sofr_averages
from quantliblab.data.loaders.fx_loader import fetch_all_fx
from quantliblab.data.store import write_many
from quantliblab.data.validators import (
    validate_rate, validate_fx_spot, ValidationError,
)


def _parse_args() -> tuple[date, date]:
    args = sys.argv[1:]
    today = date.today()
    if len(args) == 0:
        d = today - timedelta(days=1)
        return d, d
    if len(args) == 1:
        d = date.fromisoformat(args[0])
        return d, d
    return date.fromisoformat(args[0]), date.fromisoformat(args[1])


def fetch_rates(start: date, end: date) -> None:
    print(f"  Fetching SOFR O/N (FRED)...")
    records = fetch_sofr_on(start, end)
    validated = []
    for d, row in records:
        try:
            validate_rate(row["ON"], label="SOFR ON")
            validated.append((d, row))
        except ValidationError as e:
            print(f"    [SKIP] {d}: {e}")
    write_many("rates", "sofr_on", validated)
    print(f"    Saved {len(validated)} rows to rates/sofr_on.csv")

    print(f"  Fetching SOFR averages (NY Fed)...")
    records = fetch_sofr_averages(start, end)
    write_many("rates", "sofr_averages", records)
    print(f"    Saved {len(records)} rows to rates/sofr_averages.csv")

    print(f"  Fetching SONIA O/N (FRED)...")
    records = fetch_sonia_on(start, end)
    validated = []
    for d, row in records:
        try:
            validate_rate(row["ON"], label="SONIA ON")
            validated.append((d, row))
        except ValidationError as e:
            print(f"    [SKIP] {d}: {e}")
    write_many("rates", "sonia_on", validated)
    print(f"    Saved {len(validated)} rows to rates/sonia_on.csv")

    print(f"  Fetching ESTR O/N (FRED)...")
    records = fetch_estr_on(start, end)
    validated = []
    for d, row in records:
        try:
            validate_rate(row["ON"], label="ESTR ON", min_rate=-0.02)
            validated.append((d, row))
        except ValidationError as e:
            print(f"    [SKIP] {d}: {e}")
    write_many("rates", "estr_on", validated)
    print(f"    Saved {len(validated)} rows to rates/estr_on.csv")


def fetch_fx(start: date, end: date) -> None:
    print(f"  Fetching FX spots (yfinance)...")
    all_fx = fetch_all_fx(start, end)
    for pair, records in all_fx.items():
        validated = []
        for d, row in records:
            try:
                validate_fx_spot(row["spot"], pair=pair)
                validated.append((d, row))
            except ValidationError as e:
                print(f"    [SKIP] {pair} {d}: {e}")
        write_many("fx", pair.lower(), validated)
        print(f"    Saved {len(validated)} rows to fx/{pair.lower()}.csv")


def main() -> None:
    start, end = _parse_args()
    print(f"\nQuantLibLab daily data fetch: {start} -> {end}\n")

    print("[ Rates ]")
    try:
        fetch_rates(start, end)
    except Exception as e:
        print(f"  ERROR fetching rates: {e}")

    print("\n[ FX ]")
    try:
        fetch_fx(start, end)
    except Exception as e:
        print(f"  ERROR fetching FX: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
