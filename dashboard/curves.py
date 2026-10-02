"""
Golden-snapshot curve builders — pure data, no Streamlit.

Each function takes a `snap` (one golden snapshot directory, as returned
by quantliblab.data.golden.list_snapshots) and returns (asof_date,
DataFrame, source_line), or None if that snapshot doesn't have the feed.
curve_smile() additionally returns the list of expiries available in
that snapshot.

Shared by dashboard/curve_viewer.py (Streamlit, one snapshot at a time)
and scripts/build_curves_site.py (static JSON mirror, every snapshot) so
both render identical tables from identical math.
"""
from __future__ import annotations

import csv
import math
from pathlib import Path

import pandas as pd

MATURITY = "Maturity"


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


def _pct(x: str | float) -> float:
    return round(float(x) * 100.0, 4)


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


def _svi_rows(snap: Path | None, ccy: str) -> list[dict]:
    if snap is None:
        return []
    return [r for r in _read_csv(snap / "normalized" / f"svi_{ccy.lower()}.csv")
            if r["used"] == "True"]


def curve_atm_term_structure(snap: Path | None, ccy: str):
    """ATM-forward vol per expiry — the vol term structure."""
    from quantliblab.volatility.smile.svi import SVIParams
    rows = _svi_rows(snap, ccy)
    if not rows:
        return None
    recs = []
    for r in rows:
        prm = SVIParams(float(r["a"]), float(r["b"]), float(r["rho"]),
                        float(r["m"]), float(r["s"]))
        recs.append({MATURITY: r["expiry"],
                     f"{ccy} ATM vol (%)":
                         round(float(prm.implied_vol(0.0, float(r["T"]))) * 100, 2)})
    df = pd.DataFrame(recs).sort_values(MATURITY, ignore_index=True)
    return (snap.name, df,
            f"Source: SVI fits of Deribit {ccy} options; ATM = at-the-money "
            "forward (k = 0) per expiry.")


def surface_by_delta(snap: Path | None, ccy: str):
    """Smile matrix: rows = expiry, cols = 10dP..10dC (+ RR25 / BF25)."""
    from quantliblab.volatility.smile.delta_conventions import (
        risk_reversal_butterfly, smile_by_delta,
    )
    from quantliblab.volatility.smile.svi import SVIParams
    rows = _svi_rows(snap, ccy)
    if not rows:
        return None
    recs = []
    for r in rows:
        prm = SVIParams(float(r["a"]), float(r["b"]), float(r["rho"]),
                        float(r["m"]), float(r["s"]))
        T = float(r["T"])
        try:
            q = smile_by_delta(lambda k: float(prm.implied_vol(k, T)), T)
        except ValueError:
            continue                          # delta unreachable on this slice
        rrbf = risk_reversal_butterfly(q)
        recs.append({MATURITY: r["expiry"],
                     **{lbl: round(q[lbl]["sigma"] * 100, 2)
                        for lbl in ("10dP", "25dP", "ATM", "25dC", "10dC")},
                     "RR25": round(rrbf["rr25"] * 100, 2),
                     "BF25": round(rrbf["bf25"] * 100, 2)})
    if not recs:
        return None
    df = pd.DataFrame(recs).sort_values(MATURITY, ignore_index=True)
    return (snap.name, df,
            f"Source: SVI fits of Deribit {ccy} options, forward-delta "
            "convention. All values in vol %. RR25 = 25dC − 25dP (skew), "
            "BF25 = wing avg − ATM (convexity).")


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
