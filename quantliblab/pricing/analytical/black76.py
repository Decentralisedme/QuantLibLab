"""
Black-76 — vanilla option pricing and analytic Greeks on the FORWARD.

Everything in this stack is expressed on the forward F for expiry T
(Deribit futures mark absorbs rates/carry), so Black-76 — not
Black-Scholes-on-spot — is the native model. Discounting is a separate
multiplicative df; all Greeks below are undiscounted (df = 1), matching
digital.py's conventions. Multiply price/theta by df if needed.

    d1 = (ln(F/K) + w/2) / sqrt(w),   d2 = d1 - sqrt(w),   w = sigma^2 T

Uses in this stack
------------------
- vega(k) as SVI fit weights (deribit_surface): concentrates fit
  accuracy where price sensitivity to vol actually lives, the textbook
  weighting for smile calibration.
- delta/vega of digital positions (flat-vol level): what hedging a
  Polymarket binary would cost; feeds Taylor P&L attribution later.

Greek conventions
-----------------
delta : dV/dF (forward delta) — call in (0,1), put in (-1,0)
gamma : d2V/dF2                — same for call and put
vega  : dV/dsigma per 1.00 of vol (divide by 100 for per-vol-point)
theta : dV/dT sign-flipped to "per year of calendar decay" (negative
        for long options), at constant sigma, undiscounted
"""
from __future__ import annotations

import math

from quantliblab.math.distributions.normal import cdf as N, pdf as phi


def _d1d2(F: float, K: float, T: float, sigma: float) -> tuple[float, float]:
    st = sigma * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * st * st) / st
    return d1, d1 - st


# ---------------------------------------------------------------------------
# Price
# ---------------------------------------------------------------------------

def price(F: float, K: float, T: float, sigma: float, call: bool = True) -> float:
    """Undiscounted Black-76 forward price of a vanilla call/put."""
    if T <= 0.0:
        intrinsic = F - K if call else K - F
        return max(intrinsic, 0.0)
    d1, d2 = _d1d2(F, K, T, sigma)
    if call:
        return F * N(d1) - K * N(d2)
    return K * N(-d2) - F * N(-d1)


# ---------------------------------------------------------------------------
# Vanilla Greeks
# ---------------------------------------------------------------------------

def delta(F: float, K: float, T: float, sigma: float, call: bool = True) -> float:
    if T <= 0.0:
        itm = F > K if call else F < K
        return (1.0 if call else -1.0) if itm else 0.0
    d1, _ = _d1d2(F, K, T, sigma)
    return N(d1) if call else N(d1) - 1.0


def gamma(F: float, K: float, T: float, sigma: float) -> float:
    if T <= 0.0:
        return 0.0
    d1, _ = _d1d2(F, K, T, sigma)
    return phi(d1) / (F * sigma * math.sqrt(T))


def vega(F: float, K: float, T: float, sigma: float) -> float:
    """dV/dsigma per 1.00 vol — same for call and put."""
    if T <= 0.0:
        return 0.0
    d1, _ = _d1d2(F, K, T, sigma)
    return F * phi(d1) * math.sqrt(T)


def theta(F: float, K: float, T: float, sigma: float) -> float:
    """Calendar decay dV/dt = -dV/dT at constant sigma (undiscounted,
    forward measure — no rate carry terms). Negative for long options;
    same for call and put by put-call parity on the forward."""
    if T <= 0.0:
        return 0.0
    d1, _ = _d1d2(F, K, T, sigma)
    return -F * phi(d1) * sigma / (2.0 * math.sqrt(T))


def vanna(F: float, K: float, T: float, sigma: float) -> float:
    """d(delta)/dsigma = d(vega)/dF — skew-hedge sensitivity."""
    if T <= 0.0:
        return 0.0
    d1, d2 = _d1d2(F, K, T, sigma)
    return -phi(d1) * d2 / sigma


def volga(F: float, K: float, T: float, sigma: float) -> float:
    """d(vega)/dsigma — convexity in vol (smile-position sensitivity)."""
    if T <= 0.0:
        return 0.0
    d1, d2 = _d1d2(F, K, T, sigma)
    return F * phi(d1) * math.sqrt(T) * d1 * d2 / sigma


# ---------------------------------------------------------------------------
# Digital (cash-or-nothing) Greeks — flat-vol level
# ---------------------------------------------------------------------------
# NOTE: these are the Greeks of P = N(d2) at a FIXED sigma. They answer
# "what does hedging this binary cost" to first order. They do NOT
# include the smile-slope correction's own sensitivities (sticky-strike
# vs sticky-delta smile dynamics — a modelling choice deferred to the
# risk layer).

def digital_delta(F: float, K: float, T: float, sigma: float) -> float:
    """d/dF of P(F_T > K) = N(d2): phi(d2) / (F sigma sqrt(T))."""
    if T <= 0.0:
        return 0.0
    _, d2 = _d1d2(F, K, T, sigma)
    return phi(d2) / (F * sigma * math.sqrt(T))


def digital_vega(F: float, K: float, T: float, sigma: float) -> float:
    """d/dsigma of N(d2) = -phi(d2) * d1 / sigma.

    Sign flips at d1 = 0, i.e. K* = F * exp(sigma^2 T / 2) — slightly
    ABOVE the forward. For K < K* (including all strikes below F) more
    vol LOWERS P(F_T > K): the lognormal median F*exp(-w/2) falls faster
    than the right tail fattens. Only for strikes beyond K* does the
    fat-tail effect win and digital vega turn positive."""
    if T <= 0.0:
        return 0.0
    d1, d2 = _d1d2(F, K, T, sigma)
    return -phi(d2) * d1 / sigma
