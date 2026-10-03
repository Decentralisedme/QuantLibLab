"""
H-004 PIT section for the static site — pure data, no Streamlit.

Builds the public PIT payload from the harness CSVs:
  * snapshots.csv      — is_pit_observation=True rows only (one ladder per
                         Kalshi expiry: its first qualifying poll)
  * resolutions.csv    — settlement print per market_id
  * kalshi_exclusions.csv — every polled event, qualifying or not, for the
                         qualification rate and the n_two_sided trend

u = F(S*) is computed with analysis/pit_score.invert_ladder_to_pit, the
same inversion the offline PIT script uses.

PUBLIC OUTPUT ONLY. The site is published to a public URL, and the
snapshots carry the harness's paper-trading numbers (fair, fair_lo/hi,
market_yes, edge_*). Those are read here to compute u but never emitted:
each observation is built from OBSERVATION_FIELDS and nothing else, and
the remaining sections are aggregates of the exclusion log.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

from analysis.pit_score import invert_ladder_to_pit
from quantliblab.harness.kalshi import MIN_TWO_SIDED_STRIKES

PIT_TARGET_N = 30
QUALIFICATION_WINDOW_DAYS = 14

# The only per-expiry fields that leave this module.
OBSERVATION_FIELDS = ("expiry", "asset", "T", "u", "calendar_arb")


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _is_true(v) -> bool:
    return str(v).strip().lower() == "true"


def _observations(snapshots: list[dict], resolutions: list[dict]) -> list[dict]:
    settlement = {r["market_id"]: float(r["settlement_value"])
                  for r in resolutions if r.get("settlement_value") not in ("", None)}

    ladders: dict[str, list[dict]] = defaultdict(list)
    for r in snapshots:
        if r.get("venue") == "kalshi" and _is_true(r.get("is_pit_observation")):
            ladders[r["event_ticker"]].append(r)

    out = []
    for rows in ladders.values():
        first = rows[0]
        s_star = next((settlement[r["market_id"]] for r in rows
                       if r["market_id"] in settlement), None)
        u = None
        if s_star is not None:
            u = invert_ladder_to_pit([float(r["strike"]) for r in rows],
                                     [float(r["fair"]) for r in rows], s_star)
        obs = {
            "expiry": first["resolution"],
            "asset": first["asset"],
            "T": round(float(first["T_years"]), 6),
            "u": None if u is None else round(u, 4),
            "calendar_arb": _is_true(first.get("calendar_arb_in_bracket")),
        }
        out.append({k: obs[k] for k in OBSERVATION_FIELDS})
    out.sort(key=lambda o: (o["expiry"], o["asset"]))
    return out


def _qualification(exclusions: list[dict], today: date) -> dict:
    start = today - timedelta(days=QUALIFICATION_WINDOW_DAYS - 1)
    window = [r for r in exclusions
              if start.isoformat() <= r["asof"][:10] <= today.isoformat()]

    polled: dict[str, set] = defaultdict(set)
    qualified: dict[str, set] = defaultdict(set)
    best: dict[str, dict[str, int]] = defaultdict(dict)   # asset -> day -> max n_two_sided
    for r in window:
        asset, day = r["asset"], r["asof"][:10]
        polled[asset].add(r["event_ticker"])
        if _is_true(r["qualifies"]):
            qualified[asset].add(r["event_ticker"])
        if r.get("n_two_sided") not in ("", None):
            n2 = int(float(r["n_two_sided"]))
            best[asset][day] = max(best[asset].get(day, 0), n2)

    assets = sorted(polled)
    days = sorted({d for a in best.values() for d in a})
    return {
        "window": {"start": start.isoformat(), "end": today.isoformat(),
                   "days": QUALIFICATION_WINDOW_DAYS},
        "by_asset": {a: {"polled": len(polled[a]), "qualified": len(qualified[a]),
                         "rate": len(qualified[a]) / len(polled[a])}
                     for a in assets},
        "n_two_sided": {
            "threshold": MIN_TWO_SIDED_STRIKES,
            "days": days,
            "series": {a: [best[a].get(d) for d in days] for a in assets},
        },
    }


def build_pit_section(harness_dir: Path, today: date) -> dict:
    snapshots = _read_csv(harness_dir / "snapshots.csv")
    resolutions = _read_csv(harness_dir / "resolutions.csv")
    exclusions = _read_csv(harness_dir / "kalshi_exclusions.csv")

    observations = _observations(snapshots, resolutions)
    latest_poll = max((r["asof"] for r in exclusions), default=None)
    return {
        "schedule": "harness",
        "latest_captured_at": latest_poll,
        "target_n": PIT_TARGET_N,
        "n": sum(o["u"] is not None for o in observations),
        "observations": observations,
        "qualification": _qualification(exclusions, today),
        "source": "Source: H-004 harness — Kalshi KXBTCD/KXETHD ladders priced off the "
                  "Deribit-only SVI surface at each expiry's first qualifying poll; "
                  "u = F(S*) at the Kalshi settlement print. Qualification and "
                  "n_two_sided from the harness exclusion log.",
    }
