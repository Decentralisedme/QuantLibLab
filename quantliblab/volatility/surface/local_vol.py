"""
Local volatility surface — Dupire local vol extracted from SVI slices.

This is the production form of the Derman–Kani program ("The Local
Volatility Surface", Derman & Kani 1994/96): the market's implied vol
surface determines a unique state-dependent diffusion sigma_loc(S, t)
that reprices every vanilla. Derman built it with implied trees; the
numerically robust route (Dupire 1994 + Gatheral) works directly on the
total-variance surface w(k, T):

                              dw/dT
    sigma_loc^2(k, T) = -----------------
                              g(k)

where g(k) is the Gatheral–Jacquier butterfly function (see svi.py).
A butterfly-free surface (g > 0) with non-decreasing w in T (no calendar
arbitrage) guarantees a real, positive local vol — this is why we fit
SVI first instead of finite-differencing raw quotes, which amplifies
noise catastrophically in the second strike derivative.

Time interpolation: LINEAR IN TOTAL VARIANCE at fixed k. This preserves
calendar-arbitrage-freeness between slices and makes dw/dT piecewise
constant and trivially available.

Use cases in this stack
-----------------------
- One-touch / "does BTC trade through X before date T" Polymarket
  markets: path-dependent -> need local vol MC (simulate_paths below).
- Plain "BTC above X at date T" digitals do NOT need local vol — use
  pricing/analytical/digital.py (smile slope only).

Caveats (be honest with yourself before sizing trades)
------------------------------------------------------
- Local vol is the *minimal* diffusion consistent with vanillas. Real
  crypto has jumps and stochastic vol; LV is known to misprice barriers
  (typically UNDER-prices one-touch probability vs stochastic vol when
  the smile is steep). Treat LV one-touch as one bound and the flat-BS
  barrier-IV price (digital.py) as the other; quote inside the band.
- Deribit expiries are sparse (<= ~10 slices); dw/dT between the first
  two slices is your short-end extrapolation — do not trust local vol
  below the first listed expiry.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
import math

import numpy as np

from quantliblab.volatility.smile.svi import SVIParams


@dataclass
class SVISlice:
    T: float          # years to expiry (ACT/365)
    F: float          # forward for this expiry (Deribit futures mark)
    params: SVIParams


class LocalVolSurface:
    """
    Total-variance surface from SVI slices; Dupire local vol; path MC.

    Parameters
    ----------
    slices : SVI fits per expiry, any order (sorted internally).
             Fit each with fit_svi() and verify is_butterfly_free().
    """

    def __init__(self, slices: list[SVISlice]) -> None:
        if len(slices) < 2:
            raise ValueError("need >= 2 expiry slices for dw/dT")
        self.slices = sorted(slices, key=lambda s: s.T)
        self._Ts = np.array([s.T for s in self.slices])

    # ------------------------------------------------------------------
    # Surface queries
    # ------------------------------------------------------------------

    def total_variance(self, k, T: float):
        """w(k, T): linear interpolation in T at fixed k; flat-in-vol
        extrapolation before the first slice (w scaled by T/T1) and
        beyond the last (w scaled by T/Tn)."""
        k = np.asarray(k, dtype=float)
        if T <= self._Ts[0]:
            return self.slices[0].params.w(k) * (T / self._Ts[0])
        if T >= self._Ts[-1]:
            return self.slices[-1].params.w(k) * (T / self._Ts[-1])
        i = bisect_left(self._Ts, T)
        t0, t1 = self._Ts[i - 1], self._Ts[i]
        w0 = self.slices[i - 1].params.w(k)
        w1 = self.slices[i].params.w(k)
        lam = (T - t0) / (t1 - t0)
        return (1 - lam) * w0 + lam * w1

    def implied_vol(self, k, T: float):
        return np.sqrt(np.maximum(self.total_variance(k, T), 1e-12) / T)

    def smile(self, k: float, T: float) -> tuple[float, float]:
        """Return (sigma, dsigma_dk) at log-moneyness k and expiry T —
        exactly the two inputs the smile-adjusted digital formula needs.

        w and w' are interpolated linearly in T between bracketing SVI
        slices (consistent with total_variance), then
        sigma = sqrt(w/T),  dsigma/dk = w'(k) / (2*sqrt(w*T))."""
        if T <= self._Ts[0]:
            p_lo = p_hi = self.slices[0].params
            lam, scale = 0.0, T / self._Ts[0]
        elif T >= self._Ts[-1]:
            p_lo = p_hi = self.slices[-1].params
            lam, scale = 1.0, T / self._Ts[-1]
        else:
            i = bisect_left(self._Ts, T)
            p_lo, p_hi = self.slices[i - 1].params, self.slices[i].params
            lam = (T - self._Ts[i - 1]) / (self._Ts[i] - self._Ts[i - 1])
            scale = 1.0
        w = max(((1 - lam) * float(p_lo.w(k)) + lam * float(p_hi.w(k))) * scale, 1e-12)
        wp = ((1 - lam) * float(p_lo.dw_dk(k)) + lam * float(p_hi.dw_dk(k))) * scale
        sigma = math.sqrt(w / T)
        dsig = wp / (2.0 * math.sqrt(w * T))
        return sigma, dsig

    def check_calendar_arbitrage(self, k_min=-2.0, k_max=2.0, n=201) -> list[tuple]:
        """Return [(T_i, T_{i+1})] pairs where w decreases in T anywhere
        on the k grid — empty list means calendar-arbitrage-free."""
        grid = np.linspace(k_min, k_max, n)
        bad = []
        for s0, s1 in zip(self.slices, self.slices[1:]):
            if np.any(s1.params.w(grid) < s0.params.w(grid) - 1e-10):
                bad.append((s0.T, s1.T))
        return bad

    # ------------------------------------------------------------------
    # Dupire local vol
    # ------------------------------------------------------------------

    def local_variance(self, k, T: float):
        """sigma_loc^2(k, T) = (dw/dT) / g(k).

        dw/dT is the slope of the piecewise-linear-in-T total variance;
        g(k) is evaluated on the T-interpolated slice via SVI derivatives
        interpolated the same way."""
        k = np.asarray(k, dtype=float)

        # dw/dT: piecewise slope of the linear-in-T interpolation
        if T <= self._Ts[0]:
            dwdT = self.slices[0].params.w(k) / self._Ts[0]
            p_lo = p_hi = self.slices[0].params
            lam = 0.0
        elif T >= self._Ts[-1]:
            dwdT = self.slices[-1].params.w(k) / self._Ts[-1]
            p_lo = p_hi = self.slices[-1].params
            lam = 1.0
        else:
            i = bisect_left(self._Ts, T)
            t0, t1 = self._Ts[i - 1], self._Ts[i]
            p_lo, p_hi = self.slices[i - 1].params, self.slices[i].params
            dwdT = (p_hi.w(k) - p_lo.w(k)) / (t1 - t0)
            lam = (T - t0) / (t1 - t0)

        # interpolate w, w', w'' linearly in T, then form g(k)
        w = np.maximum((1 - lam) * p_lo.w(k) + lam * p_hi.w(k), 1e-12)
        wp = (1 - lam) * p_lo.dw_dk(k) + lam * p_hi.dw_dk(k)
        wpp = (1 - lam) * p_lo.d2w_dk2(k) + lam * p_hi.d2w_dk2(k)
        g = ((1.0 - k * wp / (2.0 * w)) ** 2
             - (wp ** 2 / 4.0) * (1.0 / w + 0.25)
             + wpp / 2.0)

        dwdT = np.maximum(dwdT, 1e-10)   # calendar-arb guard
        g = np.maximum(g, 1e-8)          # butterfly guard
        return dwdT / g

    def local_vol(self, k, T: float):
        return np.sqrt(self.local_variance(k, T))

    # ------------------------------------------------------------------
    # Monte Carlo under local vol (for one-touch / path-dependent payoffs)
    # ------------------------------------------------------------------

    def simulate_paths(
        self,
        F0: float,
        T: float,
        n_paths: int = 20_000,
        n_steps: int = 200,
        seed: int | None = None,
    ) -> np.ndarray:
        """
        Simulate forward paths dF = sigma_loc(k, t) * F * dW (martingale —
        pricing off the futures, so no drift; discounting is a separate,
        tiny effect at Deribit horizons).

        Returns array (n_paths, n_steps + 1) of forward levels, F0 first.
        Log-Euler scheme. For one-touch, apply a Brownian-bridge barrier
        correction or use n_steps >= ~50 per month of expiry to keep
        discretisation bias below the Polymarket tick.
        """
        rng = np.random.default_rng(seed)
        dt = T / n_steps
        sqdt = np.sqrt(dt)
        logF = np.full(n_paths, np.log(F0))
        out = np.empty((n_paths, n_steps + 1))
        out[:, 0] = F0
        logF0 = np.log(F0)
        for i in range(n_steps):
            t = (i + 0.5) * dt
            k = logF - logF0                       # log-moneyness vs F0
            sig = self.local_vol(k, max(t, 1e-6))
            z = rng.standard_normal(n_paths)
            logF += -0.5 * sig ** 2 * dt + sig * sqdt * z
            out[:, i + 1] = np.exp(logF)
        return out

    def one_touch_prob_mc(self, F0: float, barrier: float, T: float,
                          n_paths: int = 40_000, n_steps: int = 250,
                          seed: int | None = None) -> tuple[float, float]:
        """
        P(F touches barrier before T) under local vol, with standard error.
        Upper barrier if barrier > F0, lower otherwise.
        """
        paths = self.simulate_paths(F0, T, n_paths, n_steps, seed)
        hit = (paths.max(axis=1) >= barrier if barrier > F0
               else paths.min(axis=1) <= barrier)
        p = float(hit.mean())
        se = float(np.sqrt(p * (1 - p) / n_paths))
        return p, se
