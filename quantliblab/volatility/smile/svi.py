"""
SVI (Stochastic Volatility Inspired) smile parametrization.

Raw SVI (Gatheral 2004) expresses TOTAL implied variance w = sigma^2 * T
as a function of log-forward-moneyness k = ln(K / F):

    w(k) = a + b * ( rho * (k - m) + sqrt((k - m)^2 + s^2) )

Parameters
----------
a   : overall variance level          (w >= 0 requires a + b*s*sqrt(1-rho^2) >= 0)
b   : angle between left/right wings  (b >= 0)
rho : rotation / skew                 (-1 < rho < 1)
m   : horizontal translation
s   : ATM curvature ("sigma" in the literature; renamed to avoid clashing
      with implied vol)

Why SVI here
------------
This is the modern implementation of the program in Derman's local
volatility papers: fit an arbitrage-free implied variance surface, then
extract local vol analytically via Dupire (see volatility/surface/local_vol.py).
SVI's wings are linear in k, consistent with Lee's moment formula, and all
derivatives needed by Dupire are analytic — no numerical differentiation
of noisy market quotes.

Butterfly arbitrage
-------------------
A slice is butterfly-arbitrage-free iff (Gatheral–Jacquier 2014)

    g(k) = (1 - k*w'/(2w))^2 - (w'^2/4)*(1/w + 1/4) + w''/2  >=  0

g(k) is also exactly the denominator of the Dupire local-variance formula,
so a butterfly-free fit guarantees a real, positive local vol.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares


@dataclass(frozen=True)
class SVIParams:
    """Raw SVI parameters for one expiry slice."""
    a:   float
    b:   float
    rho: float
    m:   float
    s:   float

    # ------------------------------------------------------------------
    # Total variance and analytic derivatives in k
    # ------------------------------------------------------------------

    def w(self, k):
        """Total implied variance w(k) = sigma^2(k) * T."""
        k = np.asarray(k, dtype=float)
        return self.a + self.b * (self.rho * (k - self.m)
                                  + np.sqrt((k - self.m) ** 2 + self.s ** 2))

    def dw_dk(self, k):
        """First derivative w'(k)."""
        k = np.asarray(k, dtype=float)
        return self.b * (self.rho
                         + (k - self.m) / np.sqrt((k - self.m) ** 2 + self.s ** 2))

    def d2w_dk2(self, k):
        """Second derivative w''(k)."""
        k = np.asarray(k, dtype=float)
        return self.b * self.s ** 2 / ((k - self.m) ** 2 + self.s ** 2) ** 1.5

    # ------------------------------------------------------------------
    # Implied vol helpers
    # ------------------------------------------------------------------

    def implied_vol(self, k, T: float):
        """Implied volatility sigma(k) = sqrt(w(k) / T)."""
        return np.sqrt(np.maximum(self.w(k), 1e-12) / T)

    def dsigma_dk(self, k, T: float):
        """d(sigma)/dk — the smile slope in log-moneyness (used by the
        smile-adjusted digital formula)."""
        w = np.maximum(self.w(k), 1e-12)
        return self.dw_dk(k) / (2.0 * np.sqrt(w * T))

    # ------------------------------------------------------------------
    # No-arbitrage
    # ------------------------------------------------------------------

    def g(self, k):
        """Gatheral–Jacquier butterfly function. g(k) >= 0 for all k
        <=> the slice is free of butterfly arbitrage. Also the Dupire
        denominator."""
        k = np.asarray(k, dtype=float)
        w = np.maximum(self.w(k), 1e-12)
        wp = self.dw_dk(k)
        wpp = self.d2w_dk2(k)
        return ((1.0 - k * wp / (2.0 * w)) ** 2
                - (wp ** 2 / 4.0) * (1.0 / w + 0.25)
                + wpp / 2.0)

    def is_butterfly_free(self, k_min: float = -2.0, k_max: float = 2.0,
                          n: int = 401) -> bool:
        """Check g(k) >= 0 on a dense grid over the relevant moneyness range."""
        grid = np.linspace(k_min, k_max, n)
        return bool(np.all(self.g(grid) >= -1e-10))

    def min_variance_ok(self) -> bool:
        """Static constraint keeping w(k) >= 0 everywhere."""
        return self.a + self.b * self.s * np.sqrt(max(1.0 - self.rho ** 2, 0.0)) >= 0.0


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def fit_svi(
    k: np.ndarray,
    iv: np.ndarray,
    T: float,
    weights: np.ndarray | None = None,
    penalize_butterfly: float = 100.0,
) -> SVIParams:
    """
    Fit raw SVI to one expiry's implied vols by weighted least squares
    on TOTAL VARIANCE (fitting w, not sigma, gives better-behaved wings).

    Parameters
    ----------
    k       : log-forward-moneyness ln(K/F) per quote
    iv      : implied vols (decimal) per quote — use Deribit mid_iv;
              fall back to mark_iv only where bid/ask IV is missing
    T       : time to expiry in years (ACT/365)
    weights : per-quote weights. Recommended: vega weights, or
              1/(ask_iv - bid_iv) to trust tight markets more.
              None = equal weights.
    penalize_butterfly : soft-penalty coefficient added to the residuals
              when g(k) < 0 on the quote grid, steering the optimizer
              toward arbitrage-free fits. Set 0.0 to disable.

    Returns
    -------
    SVIParams. Check .is_butterfly_free() before feeding to Dupire;
    if it fails, refit with a larger penalty or drop the worst quotes.

    Notes
    -----
    Multi-start Levenberg–Marquardt via scipy least_squares with box
    bounds. Deribit smiles are well behaved; 3 starts are plenty.
    """
    k = np.asarray(k, dtype=float)
    iv = np.asarray(iv, dtype=float)
    if k.shape != iv.shape or k.size < 5:
        raise ValueError("need >= 5 (k, iv) quotes of equal length to fit 5 SVI params")
    if weights is None:
        weights = np.ones_like(k)
    weights = np.asarray(weights, dtype=float)
    sw = np.sqrt(weights / weights.sum())

    w_mkt = iv ** 2 * T
    w_atm = float(np.interp(0.0, np.sort(k), w_mkt[np.argsort(k)]))

    lo = np.array([-1.0, 1e-8, -0.999, -1.5, 1e-4])
    hi = np.array([np.maximum(w_mkt.max() * 2, 1.0), 10.0, 0.999, 1.5, 2.0])

    def residuals(p):
        params = SVIParams(*p)
        res = sw * (params.w(k) - w_mkt)
        if penalize_butterfly > 0.0:
            viol = np.minimum(params.g(k), 0.0)
            res = np.concatenate([res, penalize_butterfly * viol])
        return res

    starts = [
        np.array([w_atm * 0.5, 0.1, -0.4, 0.0, 0.1]),
        np.array([w_atm * 0.8, 0.3, 0.0, 0.0, 0.2]),
        np.array([w_atm * 0.2, 0.5, 0.4, -0.1, 0.3]),
    ]

    best, best_cost = None, np.inf
    for x0 in starts:
        x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)
        sol = least_squares(residuals, x0, bounds=(lo, hi), method="trf",
                            max_nfev=2000)
        if sol.cost < best_cost:
            best, best_cost = sol, sol.cost

    params = SVIParams(*best.x)
    if not params.min_variance_ok():
        # nudge a up to restore w >= 0 (rare with the bounds above)
        floor = -params.b * params.s * np.sqrt(1 - params.rho ** 2)
        params = SVIParams(floor + 1e-8, params.b, params.rho, params.m, params.s)
    return params


def fit_rmse(params: SVIParams, k: np.ndarray, iv: np.ndarray, T: float) -> float:
    """RMSE of the fit in vol points — report next to every calibration."""
    return float(np.sqrt(np.mean((params.implied_vol(k, T) - iv) ** 2)))
