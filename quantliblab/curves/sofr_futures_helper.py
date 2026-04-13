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

P(T₁) is read from the curve built from all pillars solved so far.

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

Stub handling
-------------
Contracts whose reference period has already started (ref_start < valuation_date)
are skipped — the realised stub SOFR compounding is not handled here.
"""
from __future__ import annotations

import math
from datetime import date

from quantliblab.conventions.day_count import DayCountBasis, year_fraction
from quantliblab.curves.base.rate_curve import CurvePillar, RateCurve
from quantliblab.data.loaders.sofr_futures_loader import SOFRFuturesContract


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
    contracts        : SR1 and/or SR3 contracts (any order — sorted internally)
    existing_pillars : pillars already on the curve (deposits, prior swaps);
                       used to read P(ref_start) for the first futures contract
    basis            : day count for zero rate year fractions (ACT/360 for SOFR)
    convexity_vol    : annualised SOFR rate vol σ for convexity adjustment;
                       0.0 disables the adjustment (default)

    Returns
    -------
    List of CurvePillar, one per usable futures contract, sorted by ref_end.
    Contracts with ref_start < valuation_date are skipped.
    """
    # Only use contracts whose reference period has not yet started
    live = [c for c in contracts if c.ref_start >= valuation_date]
    live.sort(key=lambda c: c.ref_end)

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

        # P(ref_start) from the running curve
        p_start = curve.discount_factor(contract.ref_start)

        # Solve P(ref_end) from the pricing identity
        p_end = p_start / (1.0 + K * tau_ref)

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
