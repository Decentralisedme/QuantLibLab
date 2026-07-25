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

import csv
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import pandas as pd
import streamlit as st

from quantliblab.data.golden import list_snapshots
from quantliblab.data.store import read_latest

st.set_page_config(page_title="Curve Viewer", layout="centered")

MATURITY = "Maturity"


# ---------------------------------------------------------------------------
# Small readers
# ---------------------------------------------------------------------------

def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _pct(x: str | float) -> float:
    return round(float(x) * 100.0, 4)


# ---------------------------------------------------------------------------
# Curve builders — each returns (asof_date, DataFrame, source_line) or None
# ---------------------------------------------------------------------------

def curve_sofr_futures(snap: Path | None):
    if snap is None:
        return None
    rows = [r for r in _read_csv(snap / "normalized" / "sofr_futures.csv")
            if r["contract_type"] == "SR3"]
    if not rows:
        return None
    df = pd.DataFrame({
        MATURITY: [r["ref_end"] for r in rows],
        "USD SOFR (%)": [_pct(r["implied_rate"]) for r in rows],
    }).sort_values(MATURITY, ignore_index=True)
    return (snap.name, df,
            "Source: CME SR3 SOFR futures via Yahoo Finance "
            "(last price — settlement-file upgrade planned). "
            "Maturity = end of reference accrual period.")


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


def curve_deribit_futures(snap: Path | None, ccy: str):
    if snap is None:
        return None
    rows = _read_csv(snap / "normalized" / f"deribit_{ccy.lower()}_futures.csv")
    if not rows:
        return None
    df = pd.DataFrame({
        MATURITY: [r["expiry"] for r in rows],
        f"{ccy} future (USD)": [round(float(r["mark_price"]), 1) for r in rows],
    }).sort_values(MATURITY, ignore_index=True)
    return (snap.name, df,
            f"Source: Deribit {ccy} futures book summary (mark price).")


def curve_smile(snap: Path | None, ccy: str, expiry: str | None):
    """SVI-fitted smile for one expiry as Strike | IV table."""
    if snap is None:
        return None, []
    rows = [r for r in _read_csv(snap / "normalized" / f"svi_{ccy.lower()}.csv")
            if r["used"] == "True"]
    if not rows:
        return None, []
    expiries = [r["expiry"] for r in rows]
    r = next((x for x in rows if x["expiry"] == expiry), rows[0])
    from quantliblab.volatility.smile.svi import SVIParams
    p = SVIParams(float(r["a"]), float(r["b"]), float(r["rho"]),
                  float(r["m"]), float(r["s"]))
    F, T = float(r["F"]), float(r["T"])
    ks = [x / 10.0 for x in range(-5, 6)]                 # k in [-0.5, 0.5]
    df = pd.DataFrame({
        "Strike (USD)": [round(F * math.exp(k), -1) for k in ks],
        f"{ccy} IV (%)": [round(float(p.implied_vol(k, T)) * 100, 2)
                          for k in ks],
    })
    src = (f"Source: SVI fit of Deribit {ccy} options "
           f"(expiry {r['expiry']}, F={F:,.0f}, "
           f"RMSE {r['rmse_volpts']} vol pts).")
    return (snap.name, df, src), expiries


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
    "BTC smile (SVI fit)",
    "ETH smile (SVI fit)",
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
