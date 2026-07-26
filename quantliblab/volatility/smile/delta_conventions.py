"""
Delta conventions — mapping a smile from strike space to delta space.

Practitioner vol surfaces are quoted per (delta, expiry), not
(strike, expiry): 10dP | 25dP | ATM | 25dC | 10dC. Delta normalizes each
expiry by its own vol and maturity, so columns are comparable across
expiries, across days, and across assets — a strike grid is not.

Convention used: FORWARD delta (no discounting/premium adjustment),
consistent with pricing everything off the Deribit future:

    call:  delta =  N(d1),   put:  delta = N(d1) - 1
    d1 = (-k + w/2) / sqrt(w),   k = ln(K/F),  w = sigma(k)^2 * T

Because sigma depends on k, strike-from-delta is a root-solve, not a
formula. The smile enters as a callable sigma(k) so this works with SVI
params, a LocalVolSurface, or anything else.

Derived quotes:
    risk reversal RR25 = sigma(25dC) - sigma(25dP)      (skew)
    butterfly     BF25 = (sigma(25dC)+sigma(25dP))/2 - sigma(ATM)
"""
from __future__ import annotations

import math
from typing import Callable

from scipy.optimize import brentq

from quantliblab.math.distributions.normal import cdf as N

SmileFn = Callable[[float], float]        # k = ln(K/F)  ->  sigma(k)

K_MIN, K_MAX = -3.0, 3.0                  # search bracket in log-moneyness


def d1(k: float, sigma: float, T: float) -> float:
    st = sigma * math.sqrt(T)
    return (-k + 0.5 * sigma * sigma * T) / st


def forward_delta(k: float, sigma: float, T: float, call: bool) -> float:
    """Forward (driftless) Black delta at log-moneyness k."""
    nd1 = N(d1(k, sigma, T))
    return nd1 if call else nd1 - 1.0


def k_for_delta(smile: SmileFn, T: float, delta: float) -> float:
    """
    Log-moneyness k whose smile-consistent forward delta equals `delta`.

    delta > 0 -> call (e.g. 0.25 for 25dC); delta < 0 -> put (-0.25 for
    25dP). Root-solve because sigma varies with k; forward delta is
    strictly decreasing in k for any positive smile, so the root is
    unique when it exists in the bracket.
    """
    if not (-1.0 < delta < 1.0) or delta == 0.0:
        raise ValueError(f"delta must be in (-1,0)∪(0,1), got {delta}")
    call = delta > 0.0

    def f(k: float) -> float:
        return forward_delta(k, max(float(smile(k)), 1e-8), T, call) - delta

    lo, hi = f(K_MIN), f(K_MAX)
    if lo * hi > 0.0:                      # bracket failure — extreme smile/T
        raise ValueError(
            f"delta {delta} unreachable on k∈[{K_MIN},{K_MAX}] "
            f"(f({K_MIN})={lo:.4f}, f({K_MAX})={hi:.4f})")
    return float(brentq(f, K_MIN, K_MAX, xtol=1e-10))


def smile_by_delta(
    smile: SmileFn,
    T: float,
    deltas: tuple[float, ...] = (-0.10, -0.25, 0.50, 0.25, 0.10),
) -> dict[str, dict]:
    """
    Evaluate one expiry's smile on a delta grid.

    Default grid is the standard 10dP | 25dP | ATM | 25dC | 10dC quote
    set; 0.50 is treated as ATM-forward (k = 0), the anchor Deribit
    smiles are quoted around. Returns {label: {k, sigma}}.
    """
    out: dict[str, dict] = {}
    for d in deltas:
        if d == 0.50:
            label, k = "ATM", 0.0
        else:
            side = "C" if d > 0 else "P"
            label = f"{int(round(abs(d) * 100))}d{side}"
            k = k_for_delta(smile, T, d)
        out[label] = {"k": k, "sigma": float(smile(k))}
    return out


def risk_reversal_butterfly(quotes: dict[str, dict]) -> dict[str, float]:
    """RR25 / BF25 from a smile_by_delta() result (needs 25dP/ATM/25dC)."""
    c, p, atm = (quotes[x]["sigma"] for x in ("25dC", "25dP", "ATM"))
    return {"rr25": c - p, "bf25": 0.5 * (c + p) - atm}
