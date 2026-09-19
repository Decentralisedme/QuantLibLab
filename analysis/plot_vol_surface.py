"""
BTC vol surface diagnostic — 3D sigma(k,T) surface + calendar-arbitrage check.

Reads the fitted SVI slices straight out of data/harness/surfaces.csv (no
live Deribit call). Picks the most recent asof with a usable BTC surface
(>=2 used=True slices — the minimum LocalVolSurface needs) and renders:

  1. sigma(k, T) as a 3D surface, built from LocalVolSurface — the same
     linear-in-total-variance interpolation the pricer uses between
     fitted expiries — over the interior [T_front, T_back] range (no
     extrapolation beyond the fitted slices). The fitted expiries
     themselves are drawn as solid black curves on top of the surface so
     calibrated slices are visually distinct from interpolated ones.
     Rejected expiries have no retained SVI params (surfaces.csv only
     stores a/b/rho/m/s for slices LocalVolSurface actually kept), so
     they can't be placed on the surface — they're listed in the caption.
  2. total variance w = sigma^2*T against T at a few fixed k, one line per
     k across the fitted expiries. A non-monotone line is a calendar-
     arbitrage violation (w must be non-decreasing in T at fixed k).
     Rejected expiries are marked as vertical dashed lines on this panel,
     since T is their x-position even without a fitted w(k).

Usage:
    python analysis/plot_vol_surface.py [--currency BTC]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3D projection

from quantliblab.volatility.smile.svi import SVIParams
from quantliblab.volatility.surface.local_vol import LocalVolSurface, SVISlice

# Palette (light mode) — quantliblab/analysis reference palette
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
SURFACE = "#fcfcfb"

# Sequential blue ramp, ordinal steps 250->700 (light->dark), light-mode floor
SEQ_STEPS = [
    "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6",
    "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]

# Categorical slots 1-5, fixed order, for the fixed-k lines in panel 2
CAT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]

FIXED_K = [-0.5, -0.25, 0.0, 0.25, 0.5]
K_GRID = np.linspace(-0.8, 0.8, 400)
N_K_CELLS = 60   # coarser k grid for the dw/dT heatmap — discrete, markable cells
DIV_CMAP = LinearSegmentedColormap.from_list("diverging_br", ["#e34948", "#f0efec", "#2a78d6"])

REPO_ROOT = Path(__file__).resolve().parents[1]
SURFACES_CSV = REPO_ROOT / "data" / "harness" / "surfaces.csv"
OUT_DIR = REPO_ROOT / "analysis" / "plots"


def pick_asof(df: pd.DataFrame, currency: str) -> str:
    """Most recent asof for `currency` with >=2 used=True slices."""
    sub = df[df["currency"] == currency]
    counts = sub.groupby("asof")["used"].sum().sort_index()
    good = counts[counts >= 2]
    if good.empty:
        raise SystemExit(f"no asof with a usable {currency} surface (>=2 used slices)")
    return good.index[-1]


def seq_colors(n: int) -> list[str]:
    idx = np.linspace(0, len(SEQ_STEPS) - 1, n).round().astype(int)
    return [SEQ_STEPS[i] for i in idx]


SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", SEQ_STEPS)


def plot_surface_3d(ax, used: pd.DataFrame) -> None:
    used = used.sort_values("T")
    slices = [
        SVISlice(T=row["T"], F=row["F"],
                 params=SVIParams(row["a"], row["b"], row["rho"], row["m"], row["s"]))
        for _, row in used.iterrows()
    ]
    surface = LocalVolSurface(slices)

    t_grid = np.linspace(used["T"].min(), used["T"].max(), 60)
    Z = np.array([surface.implied_vol(K_GRID, float(t)) * 100 for t in t_grid])
    K_mesh, T_mesh = np.meshgrid(K_GRID, t_grid)

    ax.plot_surface(K_mesh, T_mesh, Z, cmap=SEQ_CMAP, alpha=0.85,
                    linewidth=0, antialiased=True, rstride=2, cstride=4)

    # Calibrated slices, drawn on top so real fits stand out from the
    # linear-in-T interpolation filling the surface between them.
    for _, row in used.iterrows():
        params = SVIParams(row["a"], row["b"], row["rho"], row["m"], row["s"])
        sigma = params.implied_vol(K_GRID, row["T"]) * 100
        ax.plot(K_GRID, np.full_like(K_GRID, row["T"]), sigma,
                color=INK_PRIMARY, linewidth=1.3, zorder=5)

    ax.set_xlabel("k = ln(K/F)", color=INK_SECONDARY, labelpad=8)
    ax.set_ylabel("T (years)", color=INK_SECONDARY, labelpad=8)
    ax.set_zlabel("implied vol (%)", color=INK_SECONDARY, labelpad=8)
    ax.set_title("SVI surface  sigma(k, T) — black curves are calibrated expiries",
                fontsize=10.5, color=INK_PRIMARY)
    ax.tick_params(colors=INK_SECONDARY)
    ax.xaxis.pane.set_facecolor(SURFACE)
    ax.yaxis.pane.set_facecolor(SURFACE)
    ax.zaxis.pane.set_facecolor(SURFACE)
    ax.view_init(elev=22, azim=-60)


def plot_calendar(ax, used: pd.DataFrame, rejected: pd.DataFrame) -> None:
    used = used.sort_values("T")
    for color, k in zip(CAT_COLORS, FIXED_K):
        w = []
        for _, row in used.iterrows():
            params = SVIParams(row["a"], row["b"], row["rho"], row["m"], row["s"])
            w.append(float(params.w(k)))
        ax.plot(used["T"], w, color=color, linewidth=2, marker="o",
                markersize=5, label=f"k={k:+.2f}")

    for _, row in rejected.iterrows():
        ax.axvline(row["T"], color=INK_MUTED, linewidth=1.2, linestyle="--",
                   alpha=0.7, zorder=0)

    if not rejected.empty:
        ax.axvline(rejected["T"].iloc[0], color=INK_MUTED, linewidth=1.2,
                   linestyle="--", alpha=0.7, label="rejected (used=False)")

    ax.set_xlabel("T (years)")
    ax.set_ylabel(r"total variance  $w = \sigma^2 T$")
    ax.set_title("Calendar check: w(k) vs T at fixed k", fontsize=11, color=INK_PRIMARY)
    ax.grid(True, color=GRIDLINE, linewidth=0.8)
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=INK_SECONDARY)
    ax.legend(fontsize=8, frameon=False, labelcolor=INK_SECONDARY)


def plot_dwdt_heatmap(ax, used: pd.DataFrame) -> None:
    """dw/dT between each adjacent pair of FITTED slices, evaluated
    directly from their SVI params (not LocalVolSurface — that interpolates
    linearly in w between slices, which is monotone in T by construction
    and would hide exactly the calendar-arbitrage violations this panel
    exists to show)."""
    used = used.sort_values("T")
    Ts = used["T"].to_numpy()
    params_list = [SVIParams(r["a"], r["b"], r["rho"], r["m"], r["s"])
                  for _, r in used.iterrows()]

    k_edges = np.linspace(-0.8, 0.8, N_K_CELLS + 1)
    k_centers = 0.5 * (k_edges[:-1] + k_edges[1:])

    n_pairs = len(Ts) - 1
    Z = np.empty((n_pairs, N_K_CELLS))
    for i in range(n_pairs):
        w_lo = params_list[i].w(k_centers)
        w_hi = params_list[i + 1].w(k_centers)
        Z[i] = (w_hi - w_lo) / (Ts[i + 1] - Ts[i])

    violations = Z < -1e-10
    n_violations = int(violations.sum())

    vmin = min(float(Z.min()), -1e-9)
    vmax = max(float(Z.max()), 1e-9)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    mesh = ax.pcolormesh(k_edges, Ts, Z, cmap=DIV_CMAP, norm=norm,
                         shading="flat", edgecolors=GRIDLINE, linewidth=0.2)
    cbar = plt.colorbar(mesh, ax=ax, pad=0.02)
    cbar.set_label(r"$\partial w/\partial T$", color=INK_SECONDARY)
    cbar.ax.tick_params(colors=INK_SECONDARY)

    # Unmissable violation marking: black hatched cell outline, independent
    # of the red shading underneath.
    for i in range(n_pairs):
        for j in range(N_K_CELLS):
            if violations[i, j]:
                ax.add_patch(mpatches.Rectangle(
                    (k_edges[j], Ts[i]), k_edges[j + 1] - k_edges[j], Ts[i + 1] - Ts[i],
                    fill=False, hatch="////", edgecolor=INK_PRIMARY, linewidth=0.8,
                ))

    for T in Ts:
        ax.axhline(T, color=SURFACE, linewidth=0.4, alpha=0.6)

    status = (f"{n_violations} violating cell{'s' if n_violations != 1 else ''}"
             if n_violations else "0 violating cells (calendar-arbitrage-free)")
    ax.set_title(f"dw/dT between fitted expiries — {status}", fontsize=10.5,
                color=INK_PRIMARY)
    ax.set_xlabel("k = ln(K/F)")
    ax.set_ylabel("T (years)")
    ax.tick_params(colors=INK_SECONDARY)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--currency", default="BTC")
    args = ap.parse_args()

    df = pd.read_csv(SURFACES_CSV)
    asof = pick_asof(df, args.currency)
    day = df[(df["currency"] == args.currency) & (df["asof"] == asof)]
    used = day[day["used"]]
    rejected = day[~day["used"]]

    fig = plt.figure(figsize=(21, 6))
    fig.patch.set_facecolor(SURFACE)
    fig.suptitle(f"{args.currency} vol surface — {asof}", fontsize=13,
                fontweight="bold", color=INK_PRIMARY)
    ax1 = fig.add_subplot(1, 3, 1, projection="3d")
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    plot_surface_3d(ax1, used)
    plot_calendar(ax2, used, rejected)
    plot_dwdt_heatmap(ax3, used)

    if not rejected.empty:
        reasons = "; ".join(f"{r.expiry} ({r.reason})" for r in rejected.itertuples())
        fig.text(0.5, 0.01, f"Rejected: {reasons}", ha="center", fontsize=8,
                 color=INK_MUTED, wrap=True)

    plt.tight_layout(rect=(0, 0.04, 1, 1))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"vol_surface_{args.currency.lower()}_{asof[:10]}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=SURFACE)
    print(f"Saved: {out_path}")
    print(f"asof={asof}  used={len(used)}  rejected={len(rejected)}")


if __name__ == "__main__":
    main()
