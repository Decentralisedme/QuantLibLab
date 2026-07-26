"""
Deribit vol surface calibration — smiles in, fitted SVI slices out.

Pipeline per currency (BTC / ETH):
  1. fetch index, futures strip, options book (existing loaders — one
     book-summary call per currency, NO per-instrument ticker calls)
  2. forward F(T) per option expiry: futures-implied carry r(T) is
     interpolated linearly in T, F(T) = index * exp(r(T) * T).
     (Deribit lists futures for fewer expiries than options.)
  3. quote filtering (OTM side only, live two-sided IVs, sanity bounds)
  4. fit SVI per expiry (weights = 1 / IV bid-ask width)
  5. diagnostics: RMSE, butterfly check, calendar check across slices

Output: FittedSurface with LocalVolSurface + per-slice diagnostics,
plus smile_at(k, T) returning (sigma, dsigma_dk) for digital pricing.
"""
from __future__ import annotations

import logging
import math
from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import date, datetime, timezone

import numpy as np

from quantliblab.data.loaders.deribit_loader import (
    fetch_futures, fetch_index_price, fetch_options,
)
from quantliblab.pricing.analytical.black76 import vega as _b76_vega
from quantliblab.volatility.smile.svi import SVIParams, fit_svi, fit_rmse
from quantliblab.volatility.surface.local_vol import LocalVolSurface, SVISlice

log = logging.getLogger("harness.deribit")

# Quote-quality thresholds — tune after a week of live fits
MIN_QUOTES_PER_EXPIRY = 7
MIN_T_YEARS = 2.0 / 365.0        # skip expiries under 2 days (pin risk, junk IVs)
IV_MIN, IV_MAX = 0.01, 5.0
MIN_IV_SPREAD = 0.005            # floor for weight denominator
MAX_ABS_K = 1.5                  # drop far-wing quotes (|log-moneyness| > 1.5)


@dataclass
class SliceDiagnostics:
    expiry: date
    T: float
    F: float
    n_quotes: int
    rmse_volpts: float
    butterfly_free: bool
    used: bool
    reason: str = ""


@dataclass
class FittedSurface:
    currency: str
    asof: datetime
    index_price: float
    surface: LocalVolSurface | None
    diagnostics: list[SliceDiagnostics] = field(default_factory=list)

    # ------------------------------------------------------------------
    def smile_at(self, k: float, T: float) -> tuple[float, float]:
        """
        (sigma, dsigma_dk) at log-moneyness k and horizon T, interpolating
        total variance and its k-derivative linearly in T between slices
        (consistent with LocalVolSurface.total_variance).
        """
        if self.surface is None:
            raise RuntimeError(f"no usable surface for {self.currency}")
        return self.surface.smile(k, T)   # single implementation in LocalVolSurface

    def slice_rows(self) -> list[dict]:
        """One flat dict per expiry (fitted or rejected) — the nightly
        surface archive row format. Persisting these builds the
        historical vol-surface dataset (ATM/skew/RR time series)."""
        by_T = ({round(sl.T, 9): sl for sl in self.surface.slices}
                if self.surface is not None else {})
        rows = []
        for d in self.diagnostics:
            sl = by_T.get(round(d.T, 9))
            rows.append({
                "asof": self.asof.isoformat(),
                "currency": self.currency,
                "index_price": round(self.index_price, 2),
                "expiry": d.expiry.isoformat(),
                "T": round(d.T, 6), "F": round(d.F, 2),
                "a": round(sl.params.a, 8) if sl else "",
                "b": round(sl.params.b, 8) if sl else "",
                "rho": round(sl.params.rho, 6) if sl else "",
                "m": round(sl.params.m, 6) if sl else "",
                "s": round(sl.params.s, 6) if sl else "",
                "rmse_volpts": (round(d.rmse_volpts, 3)
                                if d.rmse_volpts == d.rmse_volpts else ""),
                "n_quotes": d.n_quotes, "used": d.used, "reason": d.reason,
            })
        return rows

    def forward_at(self, T: float) -> float:
        """F(T) from the fitted slices' forwards, carry-interpolated."""
        if self.surface is None:
            raise RuntimeError(f"no usable surface for {self.currency}")
        pts = [(s.T, s.F) for s in self.surface.slices]
        return _interp_forward(self.index_price, pts, T)


# ---------------------------------------------------------------------------
# Forward curve from the futures strip
# ---------------------------------------------------------------------------

def _interp_forward(index: float, pts: list[tuple[float, float]], T: float) -> float:
    """Interpolate implied carry r_i = ln(F_i/index)/T_i linearly in T."""
    if not pts:
        return index
    pts = sorted(pts)
    rs = [(t, math.log(f / index) / t) for t, f in pts if t > 1e-6]
    if not rs:
        return index
    ts = [t for t, _ in rs]
    rr = [r for _, r in rs]
    if T <= ts[0]:
        r = rr[0]
    elif T >= ts[-1]:
        r = rr[-1]
    else:
        i = bisect_left(ts, T)
        lam = (T - ts[i - 1]) / (ts[i] - ts[i - 1])
        r = (1 - lam) * rr[i - 1] + lam * rr[i]
    return index * math.exp(r * T)


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def _year_frac(asof: datetime, expiry: date) -> float:
    """ACT/365 in seconds. Deribit options expire 08:00 UTC on expiry date."""
    exp_dt = datetime(expiry.year, expiry.month, expiry.day, 8, 0,
                      tzinfo=timezone.utc)
    return max((exp_dt - asof).total_seconds(), 0.0) / (365.0 * 86400.0)


def calibrate_currency(currency: str) -> FittedSurface:
    """Fetch Deribit data and fit the full surface for BTC or ETH."""
    asof = datetime.now(timezone.utc)
    index = fetch_index_price(currency)
    futures = fetch_futures(currency)
    options = fetch_options(currency)

    fut_pts = [(_year_frac(asof, f.expiry_date), f.mark_price)
               for f in futures if f.mark_price and f.mark_price > 0]

    by_expiry: dict[date, list] = {}
    for o in options:
        by_expiry.setdefault(o.expiry_date, []).append(o)

    slices: list[SVISlice] = []
    diags: list[SliceDiagnostics] = []

    for expiry in sorted(by_expiry):
        T = _year_frac(asof, expiry)
        F = _interp_forward(index, fut_pts, T)
        d = SliceDiagnostics(expiry, T, F, 0, float("nan"), False, False)

        if T < MIN_T_YEARS:
            d.reason = "expiry too near"; diags.append(d); continue

        ks, ivs, wts = [], [], []
        for o in by_expiry[expiry]:
            # Two-sided market required. NOTE: the book-summary loader
            # (fetch_options) carries USD bid/ask + mark_iv but NOT
            # bid_iv/ask_iv (those only come from per-instrument tickers,
            # which we deliberately avoid — one call per currency).
            # Accept either form of a live two-sided quote.
            has_iv_pair = (o.bid_iv is not None and o.ask_iv is not None
                           and o.ask_iv > o.bid_iv)
            has_px_pair = (o.bid is not None and o.ask is not None
                           and 0.0 < o.bid < o.ask)
            if not (has_iv_pair or has_px_pair):
                continue
            k = math.log(o.strike / F)
            if abs(k) > MAX_ABS_K:
                continue
            # OTM side only: puts left of the forward, calls right
            if (k < 0) != (o.kind.value == "put"):
                continue
            iv = o.mid_iv if o.mid_iv is not None else o.mark_iv
            if iv is None or not (IV_MIN < iv < IV_MAX):
                continue
            if has_iv_pair:
                spread_iv = o.ask_iv - o.bid_iv
            else:
                # IV-spread proxy from the relative price spread; only needs
                # to be monotone in quote quality for weighting purposes
                spread_iv = iv * (o.ask - o.bid) / max(o.mark_price, 1e-12)
            ks.append(k); ivs.append(iv)
            # vega-over-spread weights: trust tight markets AND focus fit
            # accuracy where price sensitivity to vol lives (textbook
            # smile-calibration weighting). F-normalized vega keeps
            # magnitudes comparable across currencies; overall scale is
            # irrelevant (fit_svi normalizes weights).
            vga = _b76_vega(F, o.strike, T, iv) / F
            wts.append(vga / max(spread_iv, MIN_IV_SPREAD))

        d.n_quotes = len(ks)
        if len(ks) < MIN_QUOTES_PER_EXPIRY:
            d.reason = f"only {len(ks)} usable quotes"; diags.append(d); continue

        try:
            params = fit_svi(np.array(ks), np.array(ivs), T, np.array(wts))
        except Exception as e:                             # keep other slices alive
            d.reason = f"fit failed: {e}"; diags.append(d); continue

        d.rmse_volpts = fit_rmse(params, np.array(ks), np.array(ivs), T) * 100
        d.butterfly_free = params.is_butterfly_free()
        if not d.butterfly_free:
            d.reason = "butterfly arbitrage in fit"; diags.append(d); continue

        d.used = True
        diags.append(d)
        slices.append(SVISlice(T=T, F=F, params=params))

    surface = LocalVolSurface(slices) if len(slices) >= 2 else None
    fs = FittedSurface(currency, asof, index, surface, diags)

    if surface is not None:
        cal = surface.check_calendar_arbitrage()
        if cal:
            log.warning("%s calendar arbitrage between T pairs: %s — "
                        "digitals near these horizons are suspect", currency, cal)
    used = sum(1 for x in diags if x.used)
    log.info("%s: fitted %d/%d expiries, index=%.0f", currency, used, len(diags), index)
    return fs
