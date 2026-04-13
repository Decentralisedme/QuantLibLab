"""
OIS Curves Dashboard — Streamlit

Three side-by-side columns, one per curve (SOFR / SONIA / ESTR).
Each column has two charts stacked:
  1. Zero rate — the bootstrapped zero coupon rate (smooth, curved between pillars)
  2. Instantaneous forward rate — FLAT between pillars, jumps at each pillar
     This is the direct output of flat-forward interpolation and shows
     the market structure clearly.

Sliders in the sidebar let you shock individual tenors live.
Pillar tables below each column.

Run from project root:
    streamlit run dashboard/ois_curves.py
"""
from __future__ import annotations

import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from quantliblab.curves import OISCurve, CurveInstrument, InstrumentType
from quantliblab.data.store import read_latest

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="OIS Curves",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ─────────────────────────────────────────────────────────────────

VALUATION = date.today()
TENORS = ["ON", "1W", "1M", "3M", "6M", "12M"]

DEFAULTS = {
    "SOFR":  {"ON": 4.31, "1W": 4.30, "1M": 4.28, "3M": 4.35, "6M": 4.45, "12M": 4.65},
    "SONIA": {"ON": 4.46, "1W": 4.45, "1M": 4.43, "3M": 4.38, "6M": 4.25, "12M": 4.05},
    "ESTR":  {"ON": 2.42, "1W": 2.41, "1M": 2.39, "3M": 2.30, "6M": 2.15, "12M": 1.95},
}

COLORS = {"SOFR": "#1f77b4", "SONIA": "#d62728", "ESTR": "#2ca02c"}
LABELS = {"SOFR": "SOFR (USD)", "SONIA": "SONIA (GBP)", "ESTR": "ESTR (EUR)"}

CONSTRUCTORS = {
    "SOFR":  OISCurve.sofr,
    "SONIA": OISCurve.sonia,
    "ESTR":  OISCurve.estr,
}

# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.title("Rate inputs (%)")
st.sidebar.caption(f"Valuation: {VALUATION.isoformat()}")

rates: dict[str, dict[str, float]] = {}

for curve_name in ["SOFR", "SONIA", "ESTR"]:
    st.sidebar.markdown(f"**{LABELS[curve_name]}**")
    rates[curve_name] = {}
    for tenor in TENORS:
        default = DEFAULTS[curve_name][tenor]
        rates[curve_name][tenor] = st.sidebar.slider(
            label=f"{curve_name} {tenor}",
            min_value=max(0.01, round(default - 3.0, 2)),
            max_value=round(default + 3.0, 2),
            value=default,
            step=0.01,
            format="%.2f%%",
            key=f"{curve_name}_{tenor}",
        )
    st.sidebar.divider()

if st.sidebar.button("Reset to defaults"):
    for curve_name in ["SOFR", "SONIA", "ESTR"]:
        for tenor in TENORS:
            st.session_state[f"{curve_name}_{tenor}"] = DEFAULTS[curve_name][tenor]
    st.rerun()

# ── Build curves ──────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def build_curve(name: str, valuation: date, rate_tuple: tuple) -> OISCurve:
    instruments = []
    for tenor, rate_pct in zip(TENORS, rate_tuple):
        kind = InstrumentType.DEPOSIT if tenor == "ON" else InstrumentType.OIS_SWAP
        instruments.append(CurveInstrument(tenor, kind, rate_pct / 100.0))
    return CONSTRUCTORS[name](valuation, instruments)


curves: dict[str, OISCurve] = {}
for curve_name in ["SOFR", "SONIA", "ESTR"]:
    rate_tuple = tuple(rates[curve_name][t] for t in TENORS)
    curves[curve_name] = build_curve(curve_name, VALUATION, rate_tuple)


def zero_rate_grid(curve: OISCurve):
    """Daily zero rates (%) over 1Y horizon."""
    dates = [curve.valuation_date + timedelta(days=d) for d in range(1, 366)]
    zeros = [curve.zero_rate(d) * 100 for d in dates]
    return dates, zeros


def forward_rate_grid(curve: OISCurve):
    """
    Instantaneous forward rates (%) over 1Y horizon.

    With flat-forward interpolation the forward rate is CONSTANT between
    each pair of adjacent pillars — this produces the characteristic step
    function that traders use to read curve shape.

    Built by sampling at 1-day intervals: f(t, t+1day).
    """
    one_day = timedelta(days=1)
    dates = [curve.valuation_date + timedelta(days=d) for d in range(1, 366)]
    fwds = [curve.forward_rate(d, d + one_day) * 100 for d in dates]
    return dates, fwds


# ── Header ────────────────────────────────────────────────────────────────────

# Last date present in the rates CSV store (most recent successful data pull)
_sofr_latest = read_latest("rates", "sofr_on")
_storage_date = _sofr_latest["date"] if _sofr_latest else "N/A"

st.markdown(
    f"<span style='font-size:2rem; font-weight:600;'>Rate Curves</span><br>"
    f"<small style='color:#1f77b4;'>Observation date: <b>{VALUATION.isoformat()}</b><br>"
    f"Data Storage Date: <b>{_storage_date}</b> "
    f"(last upload from source — FRED / NY Fed)</small>",
    unsafe_allow_html=True,
)
st.caption("Top: zero rates  ·  Bottom: instantaneous forward rates (flat-forward step function)")

# ── Three side-by-side columns ────────────────────────────────────────────────

columns = st.columns(3)

for col, curve_name in zip(columns, ["SOFR", "SONIA", "ESTR"]):
    curve  = curves[curve_name]
    color  = COLORS[curve_name]
    label  = LABELS[curve_name]

    z_dates, z_zeros = zero_rate_grid(curve)
    f_dates, f_fwds  = forward_rate_grid(curve)

    pillar_dates  = [p.maturity_date for p in curve.pillars]
    pillar_zeros  = [p.zero_rate * 100 for p in curve.pillars]
    pillar_tenors = [p.tenor for p in curve.pillars]

    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=("Zero rate (%)", "Fwd rate — flat-forward (%)"),
        vertical_spacing=0.14,
        shared_xaxes=True,
    )

    # ── Zero rate ──────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=z_dates, y=z_zeros,
        mode="lines",
        line=dict(color=color, width=2),
        hovertemplate="%{x|%d %b %Y}<br><b>%{y:.3f}%</b><extra>Zero</extra>",
        showlegend=False,
    ), row=1, col=1)

    # Pillar dots with tenor labels
    fig.add_trace(go.Scatter(
        x=pillar_dates, y=pillar_zeros,
        mode="markers+text",
        marker=dict(color=color, size=8, line=dict(color="white", width=1.5)),
        text=pillar_tenors,
        textposition="top center",
        textfont=dict(size=9),
        hovertemplate="%{text}<br><b>%{y:.3f}%</b><extra>Pillar</extra>",
        showlegend=False,
    ), row=1, col=1)

    # ── Forward rate (step function) ──────────────────────────────────────
    # Use linear_open shape — Plotly's "hv" (horizontal-then-vertical) step
    # renders a true step function with flat horizontal segments and
    # vertical jumps, exactly matching flat-forward behaviour.
    fig.add_trace(go.Scatter(
        x=f_dates, y=f_fwds,
        mode="lines",
        line=dict(color=color, width=2, shape="hv"),
        hovertemplate="%{x|%d %b %Y}<br><b>%{y:.3f}%</b><extra>Fwd</extra>",
        showlegend=False,
    ), row=2, col=1)

    # Vertical markers at pillar boundaries
    for p in curve.pillars:
        fig.add_vline(
            x=p.maturity_date,
            line=dict(color=color, width=0.8, dash="dot"),
            row=2, col=1,
        )

    z_min = min(z_zeros) - 0.05
    z_max = max(z_zeros) + 0.15
    f_min = min(f_fwds) - 0.05
    f_max = max(f_fwds) + 0.15

    fig.update_layout(
        title=dict(text=f"<b>{label}</b>", font=dict(size=13, color=color), x=0.5),
        height=520,
        margin=dict(t=60, b=30, l=50, r=15),
        plot_bgcolor="white",
        paper_bgcolor="white",
        hovermode="x unified",
        showlegend=False,
    )
    fig.update_yaxes(
        tickformat=".2f", showgrid=True, gridcolor="#f0f0f0",
        range=[z_min, z_max], row=1, col=1,
    )
    fig.update_yaxes(
        tickformat=".2f", showgrid=True, gridcolor="#f0f0f0",
        range=[f_min, f_max], row=2, col=1,
    )
    fig.update_xaxes(tickformat="%b %Y", showgrid=False, row=2, col=1)

    with col:
        st.plotly_chart(fig, use_container_width=True)

# ── Pillar tables ─────────────────────────────────────────────────────────────

st.divider()
st.markdown(
    f"<span style='font-size:1.3rem; font-weight:600;'>Rate Curves</span><br>"
    f"<small style='color:#1f77b4;'>Observation date: <b>{VALUATION.isoformat()}</b><br>"
    f"Data Storage Date: <b>{_storage_date}</b></small>",
    unsafe_allow_html=True,
)

for col, curve_name in zip(st.columns(3), ["SOFR", "SONIA", "ESTR"]):
    curve = curves[curve_name]
    df = curve.to_dataframe()[
        ["instrument", "start_date", "maturity_date", "tenor", "zero_rate", "discount_factor"]
    ].copy()
    df["tenor"] = df["tenor"].str.replace("ON", "O/N", regex=False)
    df["zero_rate"] = (df["zero_rate"] * 100).map("{:.4f}%".format)
    df["discount_factor"] = df["discount_factor"].map("{:.6f}".format)
    df.columns = ["Instrument", "Start Date", "End Date", "Maturity", "Zero Rate", "Ds Factor"]
    with col:
        st.caption(LABELS[curve_name])
        st.dataframe(df, hide_index=True, use_container_width=True)
