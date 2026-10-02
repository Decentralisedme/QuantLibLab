"""
Curve Viewer — deliberately simple.

One curve at a time:  date · curve name · two-column table
(Maturity | value, headers named properly) · one source line.

Data comes from (a) golden snapshots minted by
scripts/capture_golden_snapshot.py and (b) the CSV store filled by
scripts/fetch_daily_data.py. No live network calls — what you see is
what is on disk.

Run from project root:
    streamlit run dashboard/curve_viewer.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import streamlit as st

from dashboard.curves import (
    MATURITY, _pct, curve_atm_term_structure, curve_deribit_futures,
    curve_smile, curve_sofr_futures, surface_by_delta,
)
from quantliblab.data.golden import list_snapshots
from quantliblab.data.store import read_latest

st.set_page_config(page_title="Curve Viewer", layout="centered")


# ---------------------------------------------------------------------------
# Curve builders that read the live CSV store (not golden-snapshot-keyed) —
# see dashboard/curves.py for the golden-snapshot builders shared with
# scripts/build_curves_site.py.
# ---------------------------------------------------------------------------

def curve_sofr_averages():
    row = read_latest("rates", "sofr_averages")
    if not row:
        return None
    df = pd.DataFrame({
        MATURITY: ["30D", "90D", "180D"],
        "USD SOFR (%)": [_pct(row[t]) for t in ("30D", "90D", "180D")],
    })
    return (row["date"], df,
            "Source: NY Fed compounded SOFR averages, via FRED "
            "(SOFR30/90/180DAYAVG). Backward-looking averages.")


def curve_on_rates():
    out, latest_date = [], None
    for label, dataset, ccy in (("SOFR", "sofr_on", "USD"),
                                ("SONIA", "sonia_on", "GBP"),
                                ("ESTR", "estr_on", "EUR")):
        row = read_latest("rates", dataset)
        if row:
            out.append({MATURITY: f"O/N ({ccy} {label})",
                        "Rate (%)": _pct(row["ON"])})
            latest_date = max(latest_date or row["date"], row["date"])
    if not out:
        return None
    return (latest_date, pd.DataFrame(out),
            "Source: FRED (series SOFR, IUDSOIA, ECBESTRVOLWGTTRMDMNRT) — "
            "official overnight fixings.")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.title("Curve Viewer")

snaps = list_snapshots()
snap = None
if snaps:
    label = st.sidebar.selectbox("Snapshot date",
                                 [p.name for p in reversed(snaps)])
    snap = next(p for p in snaps if p.name == label)
else:
    st.sidebar.info("No golden snapshots found — run "
                    "`python scripts/capture_golden_snapshot.py` to enable "
                    "the Deribit / SOFR-futures curves.")

CURVES = [
    "USD SOFR — futures strip (SR3)",
    "USD SOFR — compounded averages",
    "Overnight reference rates",
    "Deribit BTC futures",
    "Deribit ETH futures",
    "BTC ATM vol term structure",
    "ETH ATM vol term structure",
    "BTC vol surface (by delta)",
    "ETH vol surface (by delta)",
    "BTC smile (SVI fit, by strike)",
    "ETH smile (SVI fit, by strike)",
]
choice = st.sidebar.selectbox("Curve", CURVES)

result, expiries = None, []
if choice == "USD SOFR — futures strip (SR3)":
    result = curve_sofr_futures(snap)
elif choice == "USD SOFR — compounded averages":
    result = curve_sofr_averages()
elif choice == "Overnight reference rates":
    result = curve_on_rates()
elif choice == "Deribit BTC futures":
    result = curve_deribit_futures(snap, "BTC")
elif choice == "Deribit ETH futures":
    result = curve_deribit_futures(snap, "ETH")
elif choice.endswith("ATM vol term structure"):
    result = curve_atm_term_structure(snap, choice[:3])
elif choice.endswith("vol surface (by delta)"):
    result = surface_by_delta(snap, choice[:3])
else:
    ccy = "BTC" if choice.startswith("BTC") else "ETH"
    result, expiries = curve_smile(snap, ccy, None)
    if expiries:
        expiry = st.sidebar.selectbox("Expiry", expiries)
        result, _ = curve_smile(snap, ccy, expiry)

if result is None:
    st.warning("No data on disk for this curve yet. "
               "Run `scripts/capture_golden_snapshot.py` (Deribit, futures, "
               "smiles) or `scripts/fetch_daily_data.py` (rates, FX).")
else:
    asof, df, source = result
    st.subheader(choice)
    st.markdown(f"**Date:** {asof}")
    st.dataframe(df, hide_index=True, use_container_width=True)
    st.caption(source)
