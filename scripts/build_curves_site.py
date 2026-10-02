#!/usr/bin/env python3
"""
build_curves_site.py — freeze every curve, every available date, into
site/curves.json for the static (Netlify) mirror of dashboard/curve_viewer.py.

Same tables, same math (dashboard/curves.py is shared verbatim with the
Streamlit app) — but where the Streamlit app picks one snapshot at a
time via read_latest()/a sidebar dropdown, this script walks every
golden snapshot (quantliblab.data.golden.list_snapshots) and every date
in the CSV store (quantliblab.data.store.read) and freezes the lot.

Also attaches, per curve:
  * schedule / latest_captured_at — which systemd timer produces this
    curve's data, and the real wall-clock time it was last captured (the
    golden snapshot's own manifest.json for golden curves; the on-disk
    CSV's mtime for CSV-store curves, since the store has no per-row
    capture timestamp — see SCHEDULES / _store_mtime below). The page
    computes "next refresh" client-side from the schedule, since that's
    relative to viewing time, not build time.
  * interp_kind — which of the three interpolation methods (vol / rate /
    futures) the client-side interpolation tool should use on curves
    with a calendar-date maturity axis; None for curves that don't have
    one (tenor-ladder or O/N curves, smile-by-strike curves).

Curves and surfaces only — this never reads data/harness/snapshots.csv
(or anything with fair/market_yes/edge_* columns). Those are the harness's
paper-trading numbers, not public market-data curves, and the output of
this script is published to a public URL.

Run from project root:
    python scripts/build_curves_site.py
Writes:
    site/curves.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd

from dashboard.curves import (
    MATURITY, _pct, curve_atm_term_structure, curve_deribit_futures,
    curve_smile, curve_sofr_futures, surface_by_delta,
)
from quantliblab.data.golden import list_snapshots
from quantliblab.data.store import read

SITE_DIR = ROOT / "site"
RAW_STORE_ROOT = ROOT / "quantliblab" / "data" / "raw"

# dataset -> (label, ccy) for the "Overnight reference rates" curve
ON_SERIES = {
    "sofr_on": ("SOFR", "USD"),
    "sonia_on": ("SONIA", "GBP"),
    "estr_on": ("ESTR", "EUR"),
}

# Which systemd timer produces each curve's data, for the "captured /
# next refresh" display. Keep in sync with deploy/systemd/*.timer.
SCHEDULES = {
    "golden_snapshot": {"label": "golden snapshot capture",
                        "hour_utc": 6, "minute_utc": 0, "weekdays_only": False},
    "daily_data": {"label": "daily rates/FX fetch",
                   "hour_utc": 6, "minute_utc": 20, "weekdays_only": True},
}


# ---------------------------------------------------------------------------
# DataFrame -> JSON-able table
# ---------------------------------------------------------------------------

def _table_entry(result, asof: str) -> dict | None:
    if result is None:
        return None
    _, df, source = result
    return {
        "asof": asof,
        "columns": df.columns.tolist(),
        "rows": df.values.tolist(),
        "source": source,
    }


def _manifest_captured_at(snap: Path) -> str | None:
    try:
        return json.loads((snap / "manifest.json").read_text()).get("captured_at")
    except (OSError, ValueError):
        return None


def _store_mtime(asset_class: str, dataset: str) -> str | None:
    """ISO timestamp the dataset's CSV was last written, or None if it
    doesn't exist yet (fetch_daily_data.py has never run)."""
    path = RAW_STORE_ROOT / asset_class / f"{dataset}.csv"
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Golden-snapshot-keyed curves (every snapshot that has the feed)
# ---------------------------------------------------------------------------

def build_golden_curve(fn, snaps: list[Path], interp_kind: str | None) -> dict:
    dates = {}
    latest_captured_at = None
    for snap in snaps:
        entry = _table_entry(fn(snap), snap.name)
        if entry is not None:
            dates[snap.name] = entry
            captured_at = _manifest_captured_at(snap)
            if captured_at is not None:
                latest_captured_at = captured_at   # snaps are oldest-first
    return {"schedule": "golden_snapshot", "latest_captured_at": latest_captured_at,
            "interp_kind": interp_kind, "dates": dates}


def build_smile_curve(ccy: str, snaps: list[Path]) -> dict:
    dates = {}
    latest_captured_at = None
    for snap in snaps:
        result, expiries = curve_smile(snap, ccy, None)
        if result is None or not expiries:
            continue
        by_expiry = {}
        for expiry in expiries:
            r2, _ = curve_smile(snap, ccy, expiry)
            entry = _table_entry(r2, snap.name)
            if entry is not None:
                by_expiry[expiry] = entry
        if by_expiry:
            dates[snap.name] = {"asof": snap.name, "expiries": expiries,
                                "by_expiry": by_expiry}
            captured_at = _manifest_captured_at(snap)
            if captured_at is not None:
                latest_captured_at = captured_at
    # Strike ladders, not a maturity axis — no interpolation tool here.
    return {"schedule": "golden_snapshot", "latest_captured_at": latest_captured_at,
            "interp_kind": None, "dates": dates}


# ---------------------------------------------------------------------------
# CSV-store-keyed curves (every date on file, not just the latest)
# ---------------------------------------------------------------------------

def build_sofr_averages_all() -> dict:
    dates = {}
    for row in read("rates", "sofr_averages"):
        df = pd.DataFrame({
            MATURITY: ["30D", "90D", "180D"],
            "USD SOFR (%)": [_pct(row[t]) for t in ("30D", "90D", "180D")],
        })
        dates[row["date"]] = {
            "asof": row["date"],
            "columns": df.columns.tolist(),
            "rows": df.values.tolist(),
            "source": "Source: NY Fed compounded SOFR averages, via FRED "
                       "(SOFR30/90/180DAYAVG). Backward-looking averages.",
        }
    # Tenor ladder (30D/90D/180D), not a calendar-date maturity axis —
    # no interpolation tool here.
    return {"schedule": "daily_data",
            "latest_captured_at": _store_mtime("rates", "sofr_averages"),
            "interp_kind": None, "dates": dates}


def build_on_rates_all() -> dict:
    by_date: dict[str, dict[str, dict]] = defaultdict(dict)
    for dataset in ON_SERIES:
        for row in read("rates", dataset):
            by_date[row["date"]][dataset] = row

    dates = {}
    for d, rows_by_dataset in sorted(by_date.items()):
        recs = []
        for dataset, (label, ccy) in ON_SERIES.items():
            row = rows_by_dataset.get(dataset)
            if row:
                recs.append({MATURITY: f"O/N ({ccy} {label})",
                             "Rate (%)": _pct(row["ON"])})
        if not recs:
            continue
        df = pd.DataFrame(recs)
        dates[d] = {
            "asof": d,
            "columns": df.columns.tolist(),
            "rows": df.values.tolist(),
            "source": "Source: FRED (series SOFR, IUDSOIA, "
                       "ECBESTRVOLWGTTRMDMNRT) — official overnight fixings.",
        }
    latest_captured_at = max(
        (t for t in (_store_mtime("rates", ds) for ds in ON_SERIES) if t is not None),
        default=None)
    # Three currencies side by side, not a maturity axis — no interpolation tool.
    return {"schedule": "daily_data", "latest_captured_at": latest_captured_at,
            "interp_kind": None, "dates": dates}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    snaps = list_snapshots()

    curves = {
        "USD SOFR — futures strip (SR3)":
            build_golden_curve(curve_sofr_futures, snaps, interp_kind="rate"),
        "USD SOFR — compounded averages": build_sofr_averages_all(),
        "Overnight reference rates": build_on_rates_all(),
        "Deribit BTC futures":
            build_golden_curve(lambda s: curve_deribit_futures(s, "BTC"), snaps,
                                interp_kind="futures"),
        "Deribit ETH futures":
            build_golden_curve(lambda s: curve_deribit_futures(s, "ETH"), snaps,
                                interp_kind="futures"),
        "BTC ATM vol term structure":
            build_golden_curve(lambda s: curve_atm_term_structure(s, "BTC"), snaps,
                                interp_kind="vol"),
        "ETH ATM vol term structure":
            build_golden_curve(lambda s: curve_atm_term_structure(s, "ETH"), snaps,
                                interp_kind="vol"),
        "BTC vol surface (by delta)":
            build_golden_curve(lambda s: surface_by_delta(s, "BTC"), snaps,
                                interp_kind="vol"),
        "ETH vol surface (by delta)":
            build_golden_curve(lambda s: surface_by_delta(s, "ETH"), snaps,
                                interp_kind="vol"),
        "BTC smile (SVI fit, by strike)": build_smile_curve("BTC", snaps),
        "ETH smile (SVI fit, by strike)": build_smile_curve("ETH", snaps),
    }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "curve_names": list(curves.keys()),
        "schedules": SCHEDULES,
        "curves": curves,
    }

    SITE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SITE_DIR / "curves.json"
    out_path.write_text(json.dumps(payload, indent=1))

    n_dates = sum(len(v["dates"]) for v in curves.values())
    print(f"{len(snaps)} golden snapshot(s) on disk: "
          f"{', '.join(s.name for s in snaps) or '(none)'}")
    for name, v in curves.items():
        print(f"  {name}: {len(v['dates'])} date(s), "
              f"schedule={v['schedule']}, interp_kind={v['interp_kind']}")
    print(f"Wrote {out_path} ({n_dates} curve-dates total)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
