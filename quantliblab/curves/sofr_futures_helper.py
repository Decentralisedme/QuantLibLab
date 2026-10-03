"""
SOFR futures rate helper.

Converts a list of SOFRFuturesContract objects into CurvePillar objects
that can be merged with deposit/OIS pillars for a full SOFR curve.

Pricing identity
----------------
A SOFR futures contract with reference period [T₁, T₂] and implied rate K
satisfies (under the futures pricing measure, ignoring convexity):

    P(T₁) / P(T₂) = 1 + K * τ

where τ = ACT/360 year fraction from T₁ to T₂ (SOFR convention).

Rearranging:

    P(T₂) = P(T₁) / (1 + K * τ)
    r_zero(T₂) = -log(P(T₂)) / T₂_yf    [T₂_yf = year frac from valuation]

P(T₁) is read from the curve built from all pillars solved so far. If
T₁ lies beyond the last solved pillar (a gap — O/N to the first contract,
SR1 month-end to the next 1st, last SR1 to the SR3 handover), P(T₁) is
taken from the final flat-forward curve including the new pillar, so the
contract reprices exactly once the pillar is in place.

SR3 (3-Month) vs SR1 (1-Month)
-------------------------------
Both product types use the same pricing identity.  The difference is:
  SR3 — compounded daily SOFR over the reference quarter
  SR1 — arithmetic average daily SOFR over the reference month

For maturities ≤ 1Y the arithmetic/compounded difference is <0.5bp and
is ignored here.  Pass convexity_vol to apply an approximate futures
convexity adjustment (disabled by default).

Convexity adjustment
--------------------
Daily margining causes futures rates to be slightly higher than equivalent
OIS forward rates:

    CA ≈ σ² · τ₁ · τ₂ / 2

where σ is annual SOFR rate volatility and τ₁, τ₂ are year fractions from
the valuation date to T₁ and T₂.  For τ₁ < 1 and σ ≈ 1% the adjustment
is < 1bp and is often omitted at the short end.

Strip construction — SR1 front, SR3 beyond (D-002)
---------------------------------------------------
SR1 and SR3 reference periods overlap: one SR3 spans parts of three or
four SR1 months. Bootstrapping both states the same forward period more
than once with inconsistent rates, and the sequential bootstrap
oscillates to satisfy them all (2.47%–5.88% sawtooth on the 2026-07-25
strip, against 3.94%–4.23% from SR3 alone).

select_strip() removes the overlap by construction:
  - SR3 from the first live contract onward (the liquid, contiguous strip)
  - SR1 only for the front, i.e. months ending on or before the first
    live SR3's ref_start — the handover date
The short gap between the last SR1 month-end and the handover is filled
by flat-forward interpolation. With no live SR3, all live SR1s are used.

Stub handling
-------------
Contracts whose reference period has already started (ref_start < valuation_date)
are skipped — the realised stub SOFR compounding is not handled here.
"""
from __future__ import annotations

import math
from datetime import date
from itertools import pairwise

from quantliblab.conventions.day_count import DayCountBasis, year_fraction
from quantliblab.curves.base.rate_curve import CurvePillar, RateCurve
from quantliblab.data.loaders.sofr_futures_loader import SOFRFuturesContract


def select_strip(
    valuation_date: date,
    contracts:      list[SOFRFuturesContract],
) -> list[SOFRFuturesContract]:
    """
    Choose a non-overlapping strip: SR1 for the front, SR3 beyond.

    Returns the contracts to bootstrap, sorted by ref_end. Contracts that
    have already started, and SR1s overlapping the SR3 strip, are dropped.
    Raises ValueError on an unknown contract type, or if the result still
    overlaps (e.g. duplicate or malformed reference periods).
    """
    for c in contracts:
        if c.contract_type not in ("SR1", "SR3"):
            raise ValueError(f"Unknown contract type {c.contract_type!r} for {c.ticker}")

    live = [c for c in contracts if c.ref_start >= valuation_date]
    sr1 = [c for c in live if c.contract_type == "SR1"]
    sr3 = [c for c in live if c.contract_type == "SR3"]

    if sr3:
        handover = min(c.ref_start for c in sr3)
        strip = [c for c in sr1 if c.ref_end <= handover] + sr3
    else:
        strip = sr1

    strip.sort(key=lambda c: c.ref_end)
    for a, b in pairwise(strip):
        if b.ref_start < a.ref_end:
            raise ValueError(
                f"Overlapping reference periods: {a.ticker} [{a.ref_start}, {a.ref_end}] "
                f"and {b.ticker} [{b.ref_start}, {b.ref_end}]"
            )
    return strip


def bootstrap_sofr_futures(
    valuation_date:   date,
    contracts:        list[SOFRFuturesContract],
    existing_pillars: list[CurvePillar],
    basis:            DayCountBasis = DayCountBasis.ACT_360,
    convexity_vol:    float = 0.0,
) -> list[CurvePillar]:
    """
    Bootstrap SOFR futures contracts into CurvePillar objects.

    Parameters
    ----------
    valuation_date   : curve reference date
    contracts        : SR1 and/or SR3 contracts (any order); reduced to a
                       non-overlapping strip by select_strip()
    existing_pillars : pillars already on the curve (deposits, prior swaps);
                       used to read P(ref_start) for the first futures contract
    basis            : day count for zero rate year fractions (ACT/360 for SOFR)
    convexity_vol    : annualised SOFR rate vol σ for convexity adjustment;
                       0.0 disables the adjustment (default)

    Returns
    -------
    List of CurvePillar, one per contract in the selected strip, sorted by
    ref_end. Started contracts and SR1s overlapping the SR3 strip are skipped.
    """
    live = select_strip(valuation_date, contracts)

    # Running curve — starts from deposit pillars, grows as we add futures pillars
    all_pillars: list[CurvePillar] = list(existing_pillars)
    futures_pillars: list[CurvePillar] = []

    for contract in live:
        curve = RateCurve(valuation_date, all_pillars, basis)

        # Year fractions from valuation date (for zero rate computation)
        tau1_yf = year_fraction(valuation_date, contract.ref_start, basis)
        tau2_yf = year_fraction(valuation_date, contract.ref_end,   basis)

        # ACT/360 year fraction from ref_start to ref_end (for the pricing identity)
        tau_ref = year_fraction(contract.ref_start, contract.ref_end, DayCountBasis.ACT_360)

        # Implied rate with optional convexity adjustment
        K = contract.implied_rate
        if convexity_vol > 0.0:
            K -= 0.5 * convexity_vol ** 2 * tau1_yf * tau2_yf

        last = max(all_pillars, key=lambda p: p.maturity_date, default=None)
        if last is None or contract.ref_start <= last.maturity_date:
            # P(ref_start) is fixed by the running curve
            p_start = curve.discount_factor(contract.ref_start)
            p_end = p_start / (1.0 + K * tau_ref)
        else:
            # ref_start lies past the last pillar, so once this pillar is
            # added P(ref_start) is flat-forward interpolated between the
            # last knot (t0, P0) and the new one. Solving on the extrapolated
            # P(ref_start) would not reprice the contract. With
            # w = (T1 - t0) / (T2 - t0), the identity gives
            #     log P(T2) = log P0 - log(1 + K * tau) / (1 - w)
            # i.e. one flat forward across the gap and the reference period.
            t0, p0 = (last.year_frac, last.discount_factor)
            w = (tau1_yf - t0) / (tau2_yf - t0)
            p_end = p0 * math.exp(-math.log1p(K * tau_ref) / (1.0 - w))

        if p_end <= 0:
            raise ValueError(
                f"Non-positive discount factor for {contract.ticker}: "
                f"p_end={p_end:.6f}"
            )

        zero_rate = -math.log(p_end) / tau2_yf

        pillar = CurvePillar(
            instrument      = contract.contract_type + "Future",
            tenor           = contract.ticker.split(".")[0],  # e.g. "SR3M26"
            start_date      = contract.ref_start,
            maturity_date   = contract.ref_end,
            year_frac       = tau2_yf,
            zero_rate       = zero_rate,
            discount_factor = p_end,
        )

        futures_pillars.append(pillar)
        all_pillars.append(pillar)

    return futures_pillars
