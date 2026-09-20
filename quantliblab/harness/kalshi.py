"""
Kalshi crypto-digital loader, ladder builder, and PIT-design pricer.

Public API, no auth — https://external-api.kalshi.com/trade-api/v2.
KXBTCD / KXETHD list a fixed-strike ladder ($250 apart) per expiry, and
every strike in one event settles off a single underlying print
(`expiration_value`) — fifty-to-two-hundred strikes, one outcome. The
ladder is therefore qualified and scored as a unit, never as independent
per-contract observations. See H-004 / ADR-0010.

Schema notes (verified against the live API 2026-09-19; not formally
versioned upstream, so re-check on drift):
  * event.product_metadata.cadence is "hourly" | "daily" | "weekly" —
    purely descriptive. It is NOT used to filter what gets priced; see
    "Bracketing rule" below.
  * market.strike_type is "greater" on every KXBTCD/KXETHD strike observed
    (i.e. YES = price above floor_strike). Anything else is logged and
    skipped rather than guessed at.
  * market.expiration_value is the single settlement print S*, identical
    across every strike in an event. market.result ("yes"/"no") is
    Kalshi's own derived flag from S* vs floor_strike; expiration_value
    is what H-004's PIT test actually needs.
  * Kalshi only lists ~1 open event per cadence tier plus unopened
    hourlies ~1.5 days out — it does not pre-list a backlog of future
    weeklies. The harness must poll regularly to catch each new one as
    it appears (H-004 task 4: uptime, not batch backfill).

Bracketing rule (H-004, frozen 2026-09-18): price an expiry only when its
T falls strictly between two USED Deribit slices — never extrapolate
below the front slice (or above the back one, symmetrically: extrapolation
there is equally ungrounded even though H-004 doesn't hit it in practice).
There is deliberately no fixed T floor: ADR-0010 proposed T >= 0.002, but
that's a stale pre-registration since superseded by this adaptive rule —
Deribit's front usable slice slides forward daily, so a fixed floor would
drift out of sync with what's actually priceable.

Every event this module polls — open or unopened, any cadence — is run
through the qualifying-expiry test and logged qualify/exclude with a
reason, before settlement is known (H-004's pre-registration requirement:
excluding after seeing S* would be selection on the outcome).
"""
from __future__ import annotations

import logging
import math
from bisect import bisect_left
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import requests

from quantliblab.harness.deribit_surface import SliceDiagnostics
from quantliblab.pricing.analytical.digital import digital_above
from quantliblab.volatility.surface.local_vol import SVISlice

log = logging.getLogger("harness.kalshi")

KALSHI_BASE = "https://external-api.kalshi.com/trade-api/v2"
SERIES = {"KXBTCD": "BTC", "KXETHD": "ETH"}
_TIMEOUT = 20
_PAGE = 200
_MAX_PAGES = 20

MIN_TWO_SIDED_STRIKES = 20    # H-004 qualifying-expiry condition
# Monotonicity is only enforced inside this p-band, with a one-Kalshi-tick
# tolerance (quotes are in half-cent increments). Near 0 or 1 the true
# probability step between adjacent $250 strikes can be sub-tick, so
# quantization alone produces apparent non-monotonicity there — checking it
# would reject good ladders in the wings, not catch a crossed book. Fixed;
# do not widen.
MONOTONE_BAND = (0.02, 0.98)
MONOTONE_TOL = 0.005
_CAL_ARB_GRID = np.linspace(-2.0, 2.0, 201)   # matches LocalVolSurface.check_calendar_arbitrage defaults


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

@dataclass
class Strike:
    ticker: str
    floor_strike: float
    yes_bid: float | None
    yes_ask: float | None
    result: str                       # "", "yes", "no"


@dataclass
class Ladder:
    event_ticker: str
    asset: str
    close_time: datetime | None
    expiration_value: float | None    # S* — shared by every strike, set once settled
    strikes: list[Strike] = field(default_factory=list)


@dataclass
class BracketInfo:
    T_lower: float
    T_upper: float
    slice_lower: SVISlice
    slice_upper: SVISlice
    diag_lower: SliceDiagnostics | None
    diag_upper: SliceDiagnostics | None
    calendar_arb: bool     # True if check_calendar_arbitrage fires on this pair


@dataclass
class QualifyResult:
    ok: bool
    reason: str
    bracket: BracketInfo | None = None
    n_two_sided: int = 0


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch_events(series_ticker: str) -> list[dict]:
    """Every open + unopened event for a series, deduped. No cadence
    filtering — qualification is decided purely by the T-bracketing test
    once a Deribit surface is available."""
    out: dict[str, dict] = {}
    for status in ("open", "unopened"):
        cursor = ""
        for _ in range(_MAX_PAGES):
            params = {"series_ticker": series_ticker, "status": status, "limit": _PAGE}
            if cursor:
                params["cursor"] = cursor
            r = requests.get(f"{KALSHI_BASE}/events", params=params, timeout=_TIMEOUT)
            r.raise_for_status()
            data = r.json()
            for e in data.get("events", []):
                out[e["event_ticker"]] = e
            cursor = data.get("cursor") or ""
            if not cursor:
                break
    return list(out.values())


def fetch_ladder(event: dict) -> Ladder:
    """The full strike ladder for one event, paginated."""
    asset = SERIES[event["series_ticker"]]
    strikes: list[Strike] = []
    close_time: datetime | None = None
    expiration_value: float | None = None
    cursor = ""
    for _ in range(_MAX_PAGES):
        params = {"event_ticker": event["event_ticker"], "limit": _PAGE}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{KALSHI_BASE}/markets", params=params, timeout=_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        for m in data.get("markets", []):
            if m.get("strike_type") != "greater":
                log.warning("unexpected strike_type=%r on %s — skipped",
                            m.get("strike_type"), m.get("ticker"))
                continue
            if close_time is None and m.get("close_time"):
                close_time = _parse_dt(m["close_time"])
            ev = m.get("expiration_value")
            if ev not in (None, ""):
                expiration_value = float(ev)
            strikes.append(Strike(
                ticker=m["ticker"],
                floor_strike=float(m["floor_strike"]),
                yes_bid=_dollars(m.get("yes_bid_dollars")),
                yes_ask=_dollars(m.get("yes_ask_dollars")),
                result=m.get("result") or "",
            ))
        cursor = data.get("cursor") or ""
        if not cursor:
            break
    strikes.sort(key=lambda s: s.floor_strike)
    return Ladder(event["event_ticker"], asset, close_time, expiration_value, strikes)


def fetch_settlement(ticker: str) -> tuple[int | None, float | None]:
    """(outcome, S*) for one Kalshi market ticker. outcome is 1 if YES,
    0 if NO, None if not yet finalized. S* is expiration_value, shared by
    every strike in that market's event."""
    r = requests.get(f"{KALSHI_BASE}/markets/{ticker}", timeout=_TIMEOUT)
    r.raise_for_status()
    m = r.json().get("market", {})
    result = m.get("result")
    if result not in ("yes", "no"):
        return None, None
    ev = m.get("expiration_value")
    settlement = float(ev) if ev not in (None, "") else None
    return (1 if result == "yes" else 0), settlement


def _dollars(v) -> float | None:
    try:
        return float(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


def _parse_dt(s: str) -> datetime:
    dt = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Bracketing — H-004's frozen rule, no fixed T floor
# ---------------------------------------------------------------------------

def bracket(fs, T: float) -> BracketInfo | None:
    """
    The two Deribit slices bracketing T, or None if T does not fall
    strictly between two fitted slices. Mirrors LocalVolSurface's own
    interior-interpolation branch (same bisect_left, same slice pairing)
    so "bracketed" here means exactly what smile()/forward_at() will use
    — this function never extrapolates, where those two do at the ends.
    """
    if fs is None or fs.surface is None:
        return None
    slices = fs.surface.slices                 # sorted by T
    Ts = [s.T for s in slices]
    if T <= Ts[0] or T >= Ts[-1]:
        return None
    i = bisect_left(Ts, T)
    lo, hi = slices[i - 1], slices[i]
    by_T = {round(d.T, 9): d for d in fs.diagnostics}
    return BracketInfo(
        T_lower=lo.T, T_upper=hi.T,
        slice_lower=lo, slice_upper=hi,
        diag_lower=by_T.get(round(lo.T, 9)),
        diag_upper=by_T.get(round(hi.T, 9)),
        calendar_arb=_pair_has_calendar_arb(lo, hi),
    )


def _pair_has_calendar_arb(lo: SVISlice, hi: SVISlice) -> bool:
    """Same check as LocalVolSurface.check_calendar_arbitrage, applied to
    one specific adjacent pair rather than the whole surface — this is
    always exactly the pair bracket() just picked, since bracket() only
    ever pairs adjacent slices."""
    return bool(np.any(hi.params.w(_CAL_ARB_GRID) < lo.params.w(_CAL_ARB_GRID) - 1e-10))


# ---------------------------------------------------------------------------
# Qualifying expiry — H-004 "Qualifying expiry" section, all conditions
# checked pre-settlement so exclusions are pre-registered, not post-hoc.
# ---------------------------------------------------------------------------

def qualify_ladder(ladder: Ladder, fs, T: float) -> QualifyResult:
    two_sided = sum(
        1 for s in ladder.strikes
        if s.yes_bid is not None and s.yes_ask is not None
        and s.yes_bid > 0.0 and s.yes_ask < 1.0
    )
    if ladder.close_time is None:
        return QualifyResult(False, "no close_time on ladder", n_two_sided=two_sided)
    if not ladder.strikes:
        return QualifyResult(False, "empty ladder", n_two_sided=two_sided)
    if fs is None or fs.surface is None:
        return QualifyResult(False, f"no usable Deribit surface for {ladder.asset}",
                              n_two_sided=two_sided)

    br = bracket(fs, T)
    if br is None:
        return QualifyResult(False, "T not bracketed by two usable Deribit slices",
                              n_two_sided=two_sided)

    # fs.surface.slices only ever contains used=True fits (deribit_surface
    # appends a slice iff d.used), so this should never fire — kept as an
    # explicit, named check because H-004 lists it as its own condition.
    if not (br.diag_lower and br.diag_lower.used and br.diag_upper and br.diag_upper.used):
        return QualifyResult(False, "bracketing slice not used=True", br, n_two_sided=two_sided)

    if two_sided < MIN_TWO_SIDED_STRIKES:
        return QualifyResult(
            False, f"only {two_sided} two-sided strikes (need {MIN_TWO_SIDED_STRIKES})",
            br, n_two_sided=two_sided)

    mids = sorted(
        (s.floor_strike, (s.yes_bid + s.yes_ask) / 2.0)
        for s in ladder.strikes if s.yes_bid is not None and s.yes_ask is not None
    )
    lo, hi = MONOTONE_BAND
    for (k0, p0), (k1, p1) in zip(mids, mids[1:]):
        if not (lo < p0 < hi and lo < p1 < hi):
            continue  # tick quantization in the wings, not a crossed book
        if p1 > p0 + MONOTONE_TOL:
            return QualifyResult(
                False,
                f"ladder not monotone in Kalshi's own quotes: "
                f"K={k0:.2f}->{p0:.4f}, K={k1:.2f}->{p1:.4f}",
                br, n_two_sided=two_sided,
            )

    return QualifyResult(True, "qualifies", br, n_two_sided=two_sided)


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------

def price_ladder(ladder: Ladder, fs, T: float, br: BracketInfo,
                  now: datetime | None = None,
                  is_pit_observation: bool = True) -> list[dict]:
    """
    Fair-value every strike in a qualifying ladder against a Deribit-only
    FittedSurface. Each row carries the interpolation diagnostics
    (w_atm, T_lower/upper, sigma_lower/upper, rmse_lower/upper) so a
    U-shaped PIT histogram can be traced to interpolation vs. the
    underlying fit without re-running the surface, plus:

      * is_pit_observation — True only on the FIRST qualifying snapshot of
        this expiry (event_ticker). A weekly ladder stays open for days and
        gets re-priced on every poll; only the first qualifying snapshot is
        the PIT observation. Later ones are kept for sensitivity analysis
        but must be excluded from the KS test. Caller decides which poll is
        first (see fetch_and_price) — this function just stamps the row.
      * calendar_arb_in_bracket — True if LocalVolSurface.check_calendar_arbitrage
        fires on the (T_lower, T_upper) pair used to interpolate this expiry.
        Flag, don't exclude: not one of H-004's four pre-registered
        qualifying conditions. The final KS result is reported both over
        all qualifying expiries and over the unflagged subset (pre-committed).
    """
    assert ladder.close_time is not None, "price_ladder requires a qualified ladder"
    close_time = ladder.close_time
    now = now or datetime.now(timezone.utc)
    F = fs.forward_at(T)
    w_atm = float(fs.surface.total_variance(0.0, T))
    rmse_lo = br.diag_lower.rmse_volpts if br.diag_lower else float("nan")
    rmse_hi = br.diag_upper.rmse_volpts if br.diag_upper else float("nan")

    rows = []
    for s in ladder.strikes:
        k = math.log(s.floor_strike / F)
        sigma, slope = fs.surface.smile(k, T)
        fair = digital_above(F, s.floor_strike, T, sigma, slope)
        sigma_lo = float(br.slice_lower.params.implied_vol(k, br.T_lower))
        sigma_hi = float(br.slice_upper.params.implied_vol(k, br.T_upper))
        yes_mid = ((s.yes_bid + s.yes_ask) / 2.0
                   if s.yes_bid is not None and s.yes_ask is not None else "")
        rows.append({
            "asof": now.isoformat(),
            "venue": "kalshi",
            "market_id": s.ticker,
            "event_ticker": ladder.event_ticker,
            "question": f"{ladder.asset} > {s.floor_strike:.2f} @ {close_time.isoformat()}",
            "asset": ladder.asset,
            "type": "european_above",
            "strike": s.floor_strike,
            "resolution": close_time.isoformat(),
            "T_years": round(T, 6),
            "forward": round(F, 2),
            "sigma_at_k": round(sigma, 4),
            "smile_slope": round(slope, 4),
            "fair": round(fair, 4),
            "fair_lo": "", "fair_hi": "",
            "market_yes": yes_mid,
            "liquidity": "",
            "w_atm": round(w_atm, 8),
            "T_lower": round(br.T_lower, 6),
            "T_upper": round(br.T_upper, 6),
            "sigma_lower": round(sigma_lo, 4),
            "sigma_upper": round(sigma_hi, 4),
            "rmse_lower": round(rmse_lo, 3) if rmse_lo == rmse_lo else "",
            "rmse_upper": round(rmse_hi, 3) if rmse_hi == rmse_hi else "",
            "is_pit_observation": is_pit_observation,
            "calendar_arb_in_bracket": br.calendar_arb,
        })
    return rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def fetch_and_price(surfaces: dict, now: datetime | None = None,
                     known_pit_events: set[str] | None = None) -> tuple[list[dict], list[dict]]:
    """
    Poll every open+unopened KXBTCD/KXETHD event, run the qualifying-expiry
    test, and price the ones that pass. Returns (snapshot_rows, exclusion_log)
    — the exclusion log records every event polled, qualifying or not, with
    its reason, so the pre-registration in H-004 is auditable after the fact.

    known_pit_events: event_tickers that have ALREADY produced a qualifying
    snapshot in a previous poll (the caller derives this from snapshots.csv
    history). A weekly ladder stays open for days and gets re-priced on
    every poll; only the first-ever qualifying snapshot of an expiry is
    stamped is_pit_observation=True — every repoll of the same event_ticker,
    within this call or across runs, is False.
    """
    now = now or datetime.now(timezone.utc)
    rows: list[dict] = []
    exclusions: list[dict] = []
    seen = set(known_pit_events or ())

    for series_ticker, asset in SERIES.items():
        fs = surfaces.get(asset)
        try:
            events = fetch_events(series_ticker)
        except Exception as e:
            log.warning("%s: event fetch failed: %s", series_ticker, e)
            continue
        log.info("%s: %d open+unopened events", series_ticker, len(events))

        for event in events:
            try:
                ladder = fetch_ladder(event)
            except Exception as e:
                log.warning("ladder fetch failed for %s: %s", event["event_ticker"], e)
                continue
            cadence = (event.get("product_metadata") or {}).get("cadence", "")
            if ladder.close_time is None:
                T = float("nan")
            else:
                T = (ladder.close_time - now).total_seconds() / (365.0 * 86400.0)

            qr = qualify_ladder(ladder, fs, T)
            exclusions.append({
                "asof": now.isoformat(),
                "event_ticker": event["event_ticker"],
                "asset": asset,
                "cadence": cadence,
                "close_time": ladder.close_time.isoformat() if ladder.close_time else "",
                "T_years": round(T, 6) if T == T else "",
                "n_strikes": len(ladder.strikes),
                "n_two_sided": qr.n_two_sided,
                "qualifies": qr.ok,
                "reason": qr.reason,
            })
            log.info("%s %s T=%s cadence=%s -> %s (%s)",
                     asset, event["event_ticker"],
                     f"{T:.4f}" if T == T else "n/a", cadence,
                     "QUALIFIES" if qr.ok else "excluded", qr.reason)
            if not qr.ok or qr.bracket is None:
                continue
            is_first = event["event_ticker"] not in seen
            seen.add(event["event_ticker"])
            rows.extend(price_ladder(ladder, fs, T, qr.bracket, now,
                                      is_pit_observation=is_first))

    return rows, exclusions
