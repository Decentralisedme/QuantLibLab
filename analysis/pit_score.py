"""
H-004 PIT (probability integral transform) scoring for Kalshi digital ladders.

For a qualifying Kalshi expiry (is_pit_observation=True), price_ladder()
prices every strike as fair(K) = P(S_T > K) against the Deribit-only
surface. The model's implied CDF is F(K) = 1 - fair(K), which should be
non-decreasing in K on a butterfly-free fit. Once the expiry settles at
S*, u = F(S*) is Uniform(0,1) under a correctly calibrated model -- the
standard PIT / probability-calibration test (Rosenblatt 1952; Diebold,
Gunther & Tay 1998). Clustering near 0/1 or a U-shaped histogram is
evidence the Deribit-implied smile is systematically mis-calibrated
against what Kalshi actually settles at.

There are zero settled Kalshi PIT expiries in data/harness/snapshots.csv
as of 2026-09-19 -- H-004 only just started polling. This module has to
be correct before that data exists, so it's tested against synthetic
ladders where the analytically correct u is known, and reports "0
observations" cleanly rather than crashing when nothing has settled yet.

Usage:
    python analysis/pit_score.py [path/to/snapshots.csv]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from quantliblab.harness.kalshi import fetch_settlement

log = logging.getLogger("analysis.pit_score")

REPO_ROOT = Path(__file__).resolve().parents[1]
SNAPSHOTS_CSV = REPO_ROOT / "data" / "harness" / "snapshots.csv"
OUT_DIR = REPO_ROOT / "analysis" / "plots"

INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"
BAR_COLOR = "#2a78d6"
REF_COLOR = "#eb6834"


# ---------------------------------------------------------------------------
# Core: invert one priced ladder into a single PIT observation
# ---------------------------------------------------------------------------

def invert_ladder_to_pit(strikes, fair, settlement: float) -> float:
    """
    u = F(S*), the model CDF evaluated at the settlement print.

    strikes, fair : parallel arrays for one qualifying ladder snapshot.
    fair[i] is P(S_T > strikes[i]) -- the "fair" column price_ladder()
    writes (the model survival function). F(K) = 1 - fair(K) is the CDF.

    Linear interpolation between the two strikes bracketing S*.
    Settlement outside the ladder's strike range has no bracketing pair
    to interpolate between, so it's clamped to the nearest end of the
    CDF (0 at the bottom of the ladder, 1 at the top) -- np.interp's
    default out-of-range behaviour, which is exactly this clamp.
    """
    strikes = np.asarray(strikes, dtype=float)
    fair = np.asarray(fair, dtype=float)
    if strikes.size == 0:
        raise ValueError("empty ladder")
    order = np.argsort(strikes)
    strikes_sorted = strikes[order]
    cdf_sorted = 1.0 - fair[order]
    return float(np.interp(settlement, strikes_sorted, cdf_sorted))


# ---------------------------------------------------------------------------
# Ladder-level orchestration
# ---------------------------------------------------------------------------

def _is_pit_row(value) -> bool:
    """Robust to the column coming back as real bool, the string "True",
    or NaN (legacy rows written before this field existed)."""
    return str(value).strip().lower() == "true"


def select_pit_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "is_pit_observation" not in df.columns:
        return df.iloc[0:0]
    mask = df["is_pit_observation"].map(_is_pit_row)
    return df[mask]


def resolve_settlements(pit_df: pd.DataFrame) -> pd.DataFrame:
    """Attach a 'settlement' column via kalshi.fetch_settlement, one call
    per event_ticker (settlement is shared across every strike in the
    ladder). Drops events that aren't settled yet or whose settlement
    can't be fetched (network down, market still open) -- never raises,
    so an all-open book still reports "0 observations" instead of
    crashing."""
    if pit_df.empty:
        return pit_df.assign(settlement=pd.Series(dtype=float))

    settled_groups = []
    for event_ticker, group in pit_df.groupby("event_ticker"):
        ticker = group["market_id"].iloc[0]
        try:
            _, settlement = fetch_settlement(ticker)
        except Exception as e:
            log.warning("settlement fetch failed for %s: %s", event_ticker, e)
            continue
        if settlement is None:
            continue
        g = group.copy()
        g["settlement"] = settlement
        settled_groups.append(g)

    if not settled_groups:
        return pit_df.iloc[0:0].assign(settlement=pd.Series(dtype=float))
    return pd.concat(settled_groups, ignore_index=True)


def score_pit(settled_df: pd.DataFrame) -> dict:
    """One u per event_ticker in settled_df (a 'settlement' column already
    attached), plus the KS test against Uniform(0,1). n=0 is a valid,
    non-crashing result."""
    u_values = []
    event_tickers = []
    for event_ticker, group in settled_df.groupby("event_ticker"):
        settlement = float(group["settlement"].iloc[0])
        u = invert_ladder_to_pit(group["strike"].to_numpy(), group["fair"].to_numpy(),
                                 settlement)
        u_values.append(u)
        event_tickers.append(event_ticker)

    u_values = np.array(u_values, dtype=float)
    n = len(u_values)
    if n == 0:
        return {"n": 0, "u_values": u_values, "event_tickers": [],
                "ks_stat": None, "ks_pvalue": None}

    ks_stat, ks_pvalue = stats.kstest(u_values, "uniform")
    return {"n": n, "u_values": u_values, "event_tickers": event_tickers,
            "ks_stat": float(ks_stat), "ks_pvalue": float(ks_pvalue)}


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_pit_histogram(result: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    if result["n"] == 0:
        ax.text(0.5, 0.5, "0 observations\n(no settled Kalshi PIT expiries yet)",
                ha="center", va="center", fontsize=12, color=INK_MUTED,
                transform=ax.transAxes)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("H-004 PIT histogram", fontsize=12, color=INK_PRIMARY)
    else:
        n_bins = max(5, min(20, result["n"] // 2 or 1))
        ax.hist(result["u_values"], bins=n_bins, range=(0, 1), density=True,
                color=BAR_COLOR, edgecolor=SURFACE, linewidth=0.5)
        ax.axhline(1.0, color=REF_COLOR, linewidth=2, linestyle="--",
                   label="Uniform(0,1) density")
        ax.set_xlim(0, 1)
        ax.set_xlabel("u = F(S*)")
        ax.set_ylabel("density")
        ax.legend(fontsize=9, frameon=False, labelcolor=INK_SECONDARY)
        ax.set_title(
            f"H-004 PIT histogram (n={result['n']})  "
            f"KS={result['ks_stat']:.3f}  p={result['ks_pvalue']:.3f}",
            fontsize=11, color=INK_PRIMARY)

    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.tick_params(colors=INK_SECONDARY)
    plt.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=str(SNAPSHOTS_CSV))
    args = ap.parse_args()

    df = pd.read_csv(args.path)
    pit_df = select_pit_rows(df)
    settled = resolve_settlements(pit_df)
    result = score_pit(settled)

    print(f"PIT-eligible rows: {len(pit_df)}  settled expiries: {result['n']}")
    if result["n"] == 0:
        print("0 observations -- no settled Kalshi PIT expiries yet.")
    else:
        print(f"KS stat={result['ks_stat']:.4f}  p-value={result['ks_pvalue']:.4f}")

    out_path = OUT_DIR / "pit_histogram.png"
    plot_pit_histogram(result, out_path)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
