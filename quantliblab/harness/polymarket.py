"""
Polymarket crypto-binary loader, parser, and pricer.

Fetches active markets from the public Gamma API, parses the question
text into a typed spec, and prices each against a FittedSurface.

Market taxonomy handled
-----------------------
  EUROPEAN : "Will Bitcoin be above $105,000 on July 8?"   -> P(F_T > K)
             "Bitcoin below $95k on August 1?"             -> P(F_T < K)
  TOUCH    : "Will Bitcoin reach/hit $150,000 by Dec 31?"  -> P(touch up)
             "Will ETH dip to/drop to $2,000 by ...?"      -> P(touch down)
  SKIPPED  : up-or-down dailies, price-bucket markets, anything ambiguous.
             Conservative by design: a skipped market costs nothing; a
             mis-parsed one costs money.

Honest caveats
--------------
* The Gamma API schema is not formally versioned; field names below
  (question, endDate, outcomes, outcomePrices, closed, conditionId,
  bestBid, bestAsk, liquidityNum) match its current shape but VERIFY on
  first live run. Everything is parsed defensively.
* Resolution-source basis: Polymarket crypto markets typically resolve
  on a Binance or Chainlink print at a stated time; Deribit's index is
  a different composite. Basis is small but nonzero — one more reason
  to require an edge threshold before trading, not price equality.
* Discounting: fair values are undiscounted probabilities, consistent
  with $1-payout quoting; USD rates over these horizons are < 1 tick.
"""
from __future__ import annotations

import json
import logging
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

import requests

from quantliblab.pricing.analytical.digital import (
    digital_above, digital_below, one_touch_prob, polymarket_edge,
)

log = logging.getLogger("harness.polymarket")

GAMMA_BASE = "https://gamma-api.polymarket.com"
_TIMEOUT = 20


class MarketType(str, Enum):
    EUROPEAN_ABOVE = "european_above"
    EUROPEAN_BELOW = "european_below"
    TOUCH = "touch"


@dataclass
class ParsedMarket:
    market_id: str
    question: str
    asset: str                 # "BTC" | "ETH"
    mtype: MarketType
    strike: float
    resolution: datetime       # UTC
    yes_price: float | None    # market mid for YES
    liquidity: float
    raw: dict


# ---------------------------------------------------------------------------
# Question parsing
# ---------------------------------------------------------------------------

_ASSET_RE = re.compile(r"\b(bitcoin|btc|ethereum|ether|eth)\b", re.I)
_MONEY_RE = re.compile(r"\$\s*([\d][\d,]*(?:\.\d+)?)\s*([kKmM])?")
_MONEY_NO_DOLLAR_RE = re.compile(r"\b([\d]+(?:\.\d+)?)\s*([kK])\b")
_TOUCH_RE = re.compile(r"\b(reach|hit|touch|dip\s+to|drop\s+to|fall\s+to)\b", re.I)
_ABOVE_RE = re.compile(r"\b(above|higher\s+than|greater\s+than|over|close\s+above)\b", re.I)
_BELOW_RE = re.compile(r"\b(below|lower\s+than|less\s+than|under|close\s+below)\b", re.I)
_SKIP_RE = re.compile(r"\b(up\s+or\s+down|between|what\s+price|range|flip|dominance|etf|all[-\s]?time\s+high)\b", re.I)

_MULT = {"k": 1e3, "m": 1e6}


def _parse_money(text: str) -> float | None:
    m = _MONEY_RE.search(text)
    if m:
        val = float(m.group(1).replace(",", ""))
        suf = (m.group(2) or "").lower()
        return val * _MULT.get(suf, 1.0)
    m = _MONEY_NO_DOLLAR_RE.search(text)
    if m:
        return float(m.group(1)) * 1e3
    return None


def parse_question(question: str) -> tuple[str, MarketType, float] | tuple[None, None, str]:
    """
    Returns (asset, market_type, strike) or (None, None, skip_reason).
    ATH markets, buckets, and up-or-down dailies are skipped explicitly.
    """
    if _SKIP_RE.search(question):
        return None, None, "skip-pattern (bucket/daily/ambiguous)"
    am = _ASSET_RE.search(question)
    if not am:
        return None, None, "no asset"
    asset = "BTC" if am.group(1).lower() in ("bitcoin", "btc") else "ETH"

    strike = _parse_money(question)
    if strike is None or strike <= 0:
        return None, None, "no strike"

    if _TOUCH_RE.search(question):
        return asset, MarketType.TOUCH, strike
    if _ABOVE_RE.search(question):
        return asset, MarketType.EUROPEAN_ABOVE, strike
    if _BELOW_RE.search(question):
        return asset, MarketType.EUROPEAN_BELOW, strike
    return None, None, "no direction keyword"


# ---------------------------------------------------------------------------
# Gamma API
# ---------------------------------------------------------------------------

def fetch_gamma_markets(max_pages: int = 5, page_size: int = 100) -> list[dict]:
    """Fetch active, unresolved markets (paged)."""
    out: list[dict] = []
    for page in range(max_pages):
        r = requests.get(
            f"{GAMMA_BASE}/markets",
            params={"active": "true", "closed": "false",
                    "limit": page_size, "offset": page * page_size},
            timeout=_TIMEOUT,
        )
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        out.extend(batch)
        if len(batch) < page_size:
            break
    return out


def _yes_mid(m: dict) -> float | None:
    """YES mid from bestBid/bestAsk, falling back to outcomePrices[0]."""
    bb, ba = m.get("bestBid"), m.get("bestAsk")
    try:
        if bb is not None and ba is not None and float(ba) > 0:
            return (float(bb) + float(ba)) / 2.0
    except (TypeError, ValueError):
        pass
    op = m.get("outcomePrices")
    try:
        prices = json.loads(op) if isinstance(op, str) else op
        return float(prices[0])
    except Exception:
        return None


def extract_crypto_binaries(raw_markets: list[dict],
                            min_liquidity: float = 1000.0) -> list[ParsedMarket]:
    parsed: list[ParsedMarket] = []
    skipped: dict[str, int] = {}
    for m in raw_markets:
        q = m.get("question") or ""
        asset, mtype, strike_or_reason = parse_question(q)
        if asset is None:
            skipped[strike_or_reason] = skipped.get(strike_or_reason, 0) + 1
            continue
        end = m.get("endDate") or m.get("endDateIso") or m.get("end_date_iso")
        try:
            resolution = datetime.fromisoformat(str(end).replace("Z", "+00:00"))
            if resolution.tzinfo is None:
                resolution = resolution.replace(tzinfo=timezone.utc)
        except Exception:
            skipped["bad endDate"] = skipped.get("bad endDate", 0) + 1
            continue
        try:
            liquidity = float(m.get("liquidityNum") or m.get("liquidity") or 0.0)
        except (TypeError, ValueError):
            liquidity = 0.0
        if liquidity < min_liquidity:
            skipped["thin liquidity"] = skipped.get("thin liquidity", 0) + 1
            continue
        parsed.append(ParsedMarket(
            market_id=str(m.get("conditionId") or m.get("id") or m.get("slug")),
            question=q, asset=asset, mtype=mtype, strike=strike_or_reason,
            resolution=resolution, yes_price=_yes_mid(m),
            liquidity=liquidity, raw=m,
        ))
    if skipped:
        log.info("parser skipped: %s", skipped)
    return parsed


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

def price_market(pm: ParsedMarket, surface, now: datetime | None = None) -> dict | None:
    """
    Fair-value one parsed market against a FittedSurface for its asset.
    Returns a flat dict (a snapshot row) or None if not priceable.
    """
    now = now or datetime.now(timezone.utc)
    T = (pm.resolution - now).total_seconds() / (365.0 * 86400.0)
    if T <= 0:
        return None
    F = surface.forward_at(T)
    k = math.log(pm.strike / F)
    sigma, slope = surface.smile_at(k, T)

    fair_lo = fair_hi = fair = None
    if pm.mtype == MarketType.EUROPEAN_ABOVE:
        fair = digital_above(F, pm.strike, T, sigma, slope)
    elif pm.mtype == MarketType.EUROPEAN_BELOW:
        fair = digital_below(F, pm.strike, T, sigma, slope)
    else:  # TOUCH — closed form at barrier IV and local-vol MC bracket the truth
        p_cf = one_touch_prob(F, pm.strike, T, sigma)
        p_mc, se = surface.surface.one_touch_prob_mc(
            F, pm.strike, T, n_paths=20_000, n_steps=max(int(T * 365 * 2), 60))
        fair_lo, fair_hi = min(p_cf, p_mc), max(p_cf, p_mc)
        fair = 0.5 * (fair_lo + fair_hi)

    row = {
        "asof": now.isoformat(),
        "market_id": pm.market_id,
        "question": pm.question[:120],
        "asset": pm.asset,
        "type": pm.mtype.value,
        "strike": pm.strike,
        "resolution": pm.resolution.isoformat(),
        "T_years": round(T, 6),
        "forward": round(F, 2),
        "sigma_at_k": round(sigma, 4),
        "smile_slope": round(slope, 4),
        "fair": round(fair, 4),
        "fair_lo": round(fair_lo, 4) if fair_lo is not None else "",
        "fair_hi": round(fair_hi, 4) if fair_hi is not None else "",
        "market_yes": pm.yes_price if pm.yes_price is not None else "",
        "liquidity": pm.liquidity,
    }
    if pm.yes_price is not None:
        row.update({f"edge_{k2}": v for k2, v in
                    polymarket_edge(fair, pm.yes_price).items() if k2.startswith("edge") or k2 == "side"})
    return row


# ---------------------------------------------------------------------------
# Resolution fetch (for Brier scoring)
# ---------------------------------------------------------------------------

def fetch_resolution(market_id: str) -> int | None:
    """
    Outcome of a resolved market: 1 if YES, 0 if NO, None if unresolved.
    Reads outcomePrices from Gamma once closed (["1","0"] or ["0","1"]).
    """
    r = requests.get(f"{GAMMA_BASE}/markets",
                     params={"condition_ids": market_id}, timeout=_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    if not data:
        return None
    m = data[0]
    if not m.get("closed"):
        return None
    try:
        prices = m.get("outcomePrices")
        prices = json.loads(prices) if isinstance(prices, str) else prices
        p_yes = float(prices[0])
        return 1 if p_yes > 0.5 else 0
    except Exception:
        return None
