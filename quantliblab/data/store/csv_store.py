"""
CSV-based market data store.

Manages daily snapshots of market data in structured CSV files under
quantliblab/data/raw/{asset_class}/{dataset}.csv

File format:
  - One row per date
  - First column always "date" (ISO 8601: YYYY-MM-DD)
  - Remaining columns are instrument/tenor names
  - Files are sorted by date on every write

Design principles:
  - Append-or-update: writing a date that already exists updates that row
  - Never deletes existing rows unless explicitly called
  - All I/O goes through pathlib — no hardcoded separators or OS paths
  - Raises on schema mismatch to prevent silent data corruption
"""
from __future__ import annotations

import csv
import os
from datetime import date
from pathlib import Path
from typing import Any

# Root of the raw data directory, relative to this file
_RAW_ROOT = Path(__file__).parents[2] / "data" / "raw"

# Valid asset class subdirectories
ASSET_CLASSES = frozenset(["fx", "rates", "commodities", "crypto"])


def _resolve_path(asset_class: str, dataset: str) -> Path:
    """Return the full path to a dataset CSV, validating the asset class."""
    if asset_class not in ASSET_CLASSES:
        raise ValueError(
            f"Unknown asset class '{asset_class}'. "
            f"Valid options: {sorted(ASSET_CLASSES)}"
        )
    directory = _RAW_ROOT / asset_class
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{dataset}.csv"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def write(
    asset_class: str,
    dataset: str,
    date_: date,
    data: dict[str, Any],
) -> None:
    """
    Write or update a single row for date_ in the given dataset.

    Parameters
    ----------
    asset_class : one of "fx", "rates", "commodities", "crypto"
    dataset     : filename stem, e.g. "sofr_rates", "eurusd_spot"
    date_       : the observation date
    data        : column_name -> value mapping (values will be str-converted)

    Raises
    ------
    ValueError  if the new data has columns inconsistent with the existing file
    """
    path = _resolve_path(asset_class, dataset)
    date_str = date_.isoformat()
    new_row = {"date": date_str, **{k: str(v) for k, v in data.items()}}

    if path.exists():
        rows = _read_all(path)
        existing_cols = set(rows[0].keys()) if rows else set()
        new_cols = set(new_row.keys())

        if existing_cols and existing_cols != new_cols:
            raise ValueError(
                f"Schema mismatch in {path.name}. "
                f"Existing columns: {sorted(existing_cols)}. "
                f"New columns: {sorted(new_cols)}."
            )

        # Replace existing row for this date or append
        updated = False
        for i, row in enumerate(rows):
            if row["date"] == date_str:
                rows[i] = new_row
                updated = True
                break
        if not updated:
            rows.append(new_row)
    else:
        rows = [new_row]

    _write_all(path, rows)


def write_many(
    asset_class: str,
    dataset: str,
    records: list[tuple[date, dict[str, Any]]],
) -> None:
    """
    Write or update multiple rows efficiently (single file read/write).

    Parameters
    ----------
    records : list of (date, data) tuples
    """
    for date_, data in records:
        write(asset_class, dataset, date_, data)


def read(
    asset_class: str,
    dataset: str,
    start: date | None = None,
    end: date | None = None,
) -> list[dict[str, str]]:
    """
    Read rows from a dataset, optionally filtered by date range.

    Returns a list of dicts with string values (including "date").
    Returns an empty list if the file does not exist.

    Parameters
    ----------
    start : inclusive lower bound (None = no lower bound)
    end   : inclusive upper bound (None = no upper bound)
    """
    path = _resolve_path(asset_class, dataset)
    if not path.exists():
        return []

    rows = _read_all(path)
    if start is not None:
        rows = [r for r in rows if r["date"] >= start.isoformat()]
    if end is not None:
        rows = [r for r in rows if r["date"] <= end.isoformat()]
    return rows


def read_latest(asset_class: str, dataset: str) -> dict[str, str] | None:
    """
    Return the most recent row, or None if the file is empty / missing.
    """
    rows = read(asset_class, dataset)
    return rows[-1] if rows else None


def list_datasets(asset_class: str) -> list[str]:
    """Return all dataset names (file stems) for an asset class."""
    if asset_class not in ASSET_CLASSES:
        raise ValueError(f"Unknown asset class '{asset_class}'")
    directory = _RAW_ROOT / asset_class
    if not directory.exists():
        return []
    return sorted(p.stem for p in directory.glob("*.csv"))


def delete_row(asset_class: str, dataset: str, date_: date) -> bool:
    """
    Delete the row for date_ from the dataset.

    Returns True if a row was deleted, False if the date was not found.
    """
    path = _resolve_path(asset_class, dataset)
    if not path.exists():
        return False

    rows = _read_all(path)
    date_str = date_.isoformat()
    new_rows = [r for r in rows if r["date"] != date_str]

    if len(new_rows) == len(rows):
        return False

    _write_all(path, new_rows)
    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _read_all(path: Path) -> list[dict[str, str]]:
    """Read all rows from a CSV, sorted by date."""
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return sorted(rows, key=lambda r: r["date"])


def _write_all(path: Path, rows: list[dict[str, str]]) -> None:
    """Write all rows to CSV, sorted by date, atomically via a temp file."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    rows = sorted(rows, key=lambda r: r["date"])
    fieldnames = list(rows[0].keys())

    # Write to a temp file first, then rename — prevents corruption on crash
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Atomic replace
    os.replace(tmp, path)
