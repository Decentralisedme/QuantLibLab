"""
Digital (binary) option pricing — the Polymarket fair-value engine.

A Polymarket market "BTC above K at date T" is a cash-or-nothing digital.
Its risk-neutral fair value is NOT N(d2): with a strike-dependent smile,

    P(F_T > K) = -dC/dK = N(d2) - phi(d2) * sqrt(T) * (dsigma/dk)

where k = ln(K/F) and dsigma/dk is the smile slope in log-moneyness.
With crypto's typically negative put skew (dsigma/dk < 0 below ATM,
often positive above for calls), the correction runs several vol points
of probability — routinely 2-6 cents on a Polymarket contract, i.e. the
entire edge. Anyone quoting N(d2) with ATM vol is the counterparty you
are looking for.

"Reaches $X before date T" markets are one-touch options: use
one_touch_prob (flat-vol closed form) as the fast bound and
LocalVolSurface.one_touch_prob_mc as the smile-consistent estimate.

Conventions: everything is expressed on the FORWARD F for expiry T
(use the Deribit futures mark for that expiry — this absorbs
crypto rates/carry so no separate r, q inputs are needed). Probabilities
are undiscounted, matching Polymarket's $1-payout quoting (USD rates
over these horizons shift fair value by well under one tick; add
df = P(0,T) from the SOFR curve if you want to be exact).
"""
from __future__ import annotations

import math

from quantliblab.math.distributions.normal import cdf as N, pdf as phi
# If the normal module exposes different names, alias accordingly:
#   from scipy.stats import norm; N, phi = norm.cdf, norm.pdf


# ---------------------------------------------------------------------------
# European digitals ("above/below K at T")
# ---------------------------------------------------------------------------

def d2(F: float, K: float, sigma: float, T: float) -> float:
    return (math.log(F / K) - 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))


def digital_above(
    F: float,
    K: float,
    T: float,
    sigma: float,
    dsigma_dk: float = 0.0,
) -> float:
    """
    Fair probability that F_T > K (cash-or-nothing call, undiscounted).

    Parameters
    ----------
    F         : forward for expiry T (Deribit futures mark)
    K         : Polymarket threshold
    T         : years to resolution (ACT/365, use exact resolution time UTC)
    sigma     : implied vol AT STRIKE K (interpolate the smile at k=ln(K/F))
    dsigma_dk : smile slope d(sigma)/dk at K in log-moneyness.
                From an SVI fit: params.dsigma_dk(k, T).
                0.0 reproduces plain Black N(d2).

    Returns
    -------
    Probability in [0, 1]. Clipped defensively — a violent smile slope
    pushing the raw formula outside [0,1] means the input smile has
    butterfly arbitrage at K; fix the fit, don't trade the number.
    """
    if T <= 0:
        return 1.0 if F > K else 0.0
    _d2 = d2(F, K, sigma, T)
    p = N(_d2) - phi(_d2) * math.sqrt(T) * dsigma_dk
    return min(max(p, 0.0), 1.0)


def digital_below(F: float, K: float, T: float, sigma: float,
                  dsigma_dk: float = 0.0) -> float:
    """Fair probability that F_T < K."""
    return 1.0 - digital_above(F, K, T, sigma, dsigma_dk)


def digital_above_from_svi(F: float, K: float, T: float, svi_params) -> float:
    """Convenience: smile-adjusted digital directly from an SVIParams fit."""
    k = math.log(K / F)
    sigma = float(svi_params.implied_vol(k, T))
    slope = float(svi_params.dsigma_dk(k, T))
    return digital_above(F, K, T, sigma, slope)


# ---------------------------------------------------------------------------
# One-touch ("trades through B before T") — flat-vol closed form
# ---------------------------------------------------------------------------

def one_touch_prob(F: float, B: float, T: float, sigma: float) -> float:
    """
    P(F touches barrier B before T) for driftless geometric Brownian
    motion (martingale forward), flat vol sigma.

    Reflection-principle closed form. With log-drift nu = -sigma^2/2:

      upper barrier (B > F), m = ln(B/F) > 0:
        P = N((nu*T - m)/(sigma*sqrt(T))) + e^{2*nu*m/sigma^2} * N((-m - nu*T)/(sigma*sqrt(T)))
        where e^{2*nu*m/sigma^2} = F/B.
      lower barrier (B < F): mirror image with F/B -> B/F... (implemented
      via symmetry on m < 0 below).

    Practitioner guidance: evaluate with sigma = implied vol AT THE
    BARRIER strike. Under a steep smile this flat-vol number and the
    local-vol MC (LocalVolSurface.one_touch_prob_mc) bracket the truth;
    quote inside the band, size to the band width.
    """
    if T <= 0 or F <= 0 or B <= 0:
        return 0.0
    if math.isclose(F, B, rel_tol=1e-12):
        return 1.0          # already at the barrier
    st = sigma * math.sqrt(T)
    nu = -0.5 * sigma * sigma
    m = math.log(B / F)
    if m > 0:      # upper barrier: exp(2*nu*m/sigma^2) = exp(-m) = F/B
        p = N((nu * T - m) / st) + (F / B) * N((-m - nu * T) / st)
    else:          # lower barrier, m < 0
        p = N((m - nu * T) / st) + math.exp(2.0 * nu * m / (sigma * sigma)) \
            * N((m + nu * T) / st)
    return min(max(p, 0.0), 1.0)


# ---------------------------------------------------------------------------
# Trading wrapper
# ---------------------------------------------------------------------------

def polymarket_edge(
    fair_prob: float,
    market_yes_price: float,
    fee: float = 0.0,
) -> dict:
    """
    Edge report for one Polymarket binary.

    Returns dict with fair value, market price, edge per side and the
    side (if any) with positive edge after fee. Deliberately dumb —
    sizing, UMA-resolution haircut, and adverse-selection checks
    (informed-wallet flow) belong in Spock, not here.
    """
    edge_yes = fair_prob - market_yes_price - fee
    edge_no = (1.0 - fair_prob) - (1.0 - market_yes_price) - fee
    side = "YES" if edge_yes > 0 else "NO" if edge_no > 0 else None
    return {
        "fair": round(fair_prob, 4),
        "market": market_yes_price,
        "edge_yes": round(edge_yes, 4),
        "edge_no": round(edge_no, 4),
        "side": side,
    }
