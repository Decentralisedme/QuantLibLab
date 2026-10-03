"""
Tests for dashboard/pit.py — the public H-004 PIT section of the static site.

The site is public; snapshots.csv carries the harness's paper-trading
numbers. The leak tests here are the guard: per-expiry output must be
exactly {expiry, asset, T, u, calendar_arb}, and no fair / market / edge
value may appear anywhere in the serialized payload.
"""
from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

import pytest

from dashboard.pit import OBSERVATION_FIELDS, build_pit_section

REPO_HARNESS = Path(__file__).resolve().parents[2] / "data" / "harness"

SNAP_FIELDS = ["asof", "venue", "market_id", "event_ticker", "question", "asset", "type",
               "strike", "resolution", "T_years", "forward", "sigma_at_k", "smile_slope",
               "fair", "fair_lo", "fair_hi", "market_yes", "liquidity",
               "edge_edge_yes", "edge_edge_no", "edge_side", "is_pit_observation",
               "calendar_arb_in_bracket"]
EXCL_FIELDS = ["asof", "event_ticker", "asset", "cadence", "close_time", "T_years",
               "n_strikes", "qualifies", "reason", "n_two_sided"]

# Distinctive sentinel values so a leak is findable in the JSON text
FAIR_MARK = 0.123457
MARKET_MARK = 0.987653
EDGE_MARK = 0.555559


def _write(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(rows)


def _ladder(event: str, asset: str, expiry: str, strikes: list[float], cdf, pit=True,
            arb=False, asof="2026-09-20T07:00:00+00:00") -> list[dict]:
    """Kalshi ladder rows with fair = 1 - cdf(K)."""
    return [{
        "asof": asof, "venue": "kalshi", "market_id": f"{event}-T{k}",
        "event_ticker": event, "question": "q", "asset": asset, "type": "above",
        "strike": k, "resolution": expiry, "T_years": 0.0140, "fair": 1 - cdf(k),
        "fair_lo": FAIR_MARK, "fair_hi": FAIR_MARK, "market_yes": MARKET_MARK,
        "edge_edge_yes": EDGE_MARK, "edge_edge_no": EDGE_MARK, "edge_side": "YES",
        "is_pit_observation": pit, "calendar_arb_in_bracket": arb,
    } for k in strikes]


@pytest.fixture
def harness(tmp_path: Path) -> Path:
    strikes = [100.0, 200.0, 300.0, 400.0, 500.0]
    lin = lambda k: (k - 100.0) / 400.0                       # CDF 0 at 100, 1 at 500
    snaps = (
        _ladder("BTC-A", "BTC", "2026-09-25T21:00:00+00:00", strikes, lin)
        + _ladder("ETH-A", "ETH", "2026-09-26T21:00:00+00:00", strikes, lin, arb=True)
        + _ladder("BTC-B", "BTC", "2026-10-09T21:00:00+00:00", strikes, lin)
        # a repoll of BTC-A: not a PIT row, must not change u
        + _ladder("BTC-A", "BTC", "2026-09-25T21:00:00+00:00", strikes, lambda k: 0.5,
                  pit=False, asof="2026-09-21T07:00:00+00:00")
        # polymarket row with a fair value — never part of PIT
        + [{"asof": "2026-09-20T07:00:00+00:00", "market_id": "0xabc", "asset": "BTC",
            "strike": 1e6, "resolution": "2026-09-30T00:00:00+00:00", "T_years": 0.03,
            "fair": FAIR_MARK, "market_yes": MARKET_MARK, "edge_edge_yes": EDGE_MARK}]
    )
    _write(tmp_path / "snapshots.csv", SNAP_FIELDS, snaps)
    _write(tmp_path / "resolutions.csv",
           ["market_id", "resolved_at", "outcome", "settlement_value"],
           [{"market_id": "BTC-A-T300.0", "resolved_at": "x", "outcome": 1,
             "settlement_value": 350.0},                       # u = 0.625
            {"market_id": "ETH-A-T100.0", "resolved_at": "x", "outcome": 1,
             "settlement_value": 150.0},                       # u = 0.125
            {"market_id": "0xabc", "resolved_at": "x", "outcome": 0,
             "settlement_value": ""}])
    excl = []
    for day, q in [("2026-09-19", True), ("2026-09-20", True), ("2026-10-03", False)]:
        for i in range(4):
            excl.append({"asof": f"{day}T07:00:00+00:00", "event_ticker": f"BTC-{day}-{i}",
                         "asset": "BTC", "qualifies": q and i == 0, "reason": "r",
                         "n_two_sided": 10 * i})
    excl.append({"asof": "2026-10-03T07:00:00+00:00", "event_ticker": "ETH-x",
                 "asset": "ETH", "qualifies": False, "reason": "r", "n_two_sided": ""})
    _write(tmp_path / "kalshi_exclusions.csv", EXCL_FIELDS, excl)
    return tmp_path


def _keys(obj) -> set[str]:
    if isinstance(obj, dict):
        return set(obj) | set().union(*(_keys(v) for v in obj.values()))
    if isinstance(obj, list):
        return set().union(*(_keys(v) for v in obj)) if obj else set()
    return set()


class TestNoLeak:
    def test_observation_fields_exact(self, harness):
        pit = build_pit_section(harness, date(2026, 10, 3))
        for o in pit["observations"]:
            assert tuple(o) == OBSERVATION_FIELDS

    def test_no_fair_market_edge_anywhere(self, harness):
        pit = build_pit_section(harness, date(2026, 10, 3))
        text = json.dumps(pit)
        for mark in (FAIR_MARK, MARKET_MARK, EDGE_MARK):
            assert str(mark) not in text
        for k in _keys(pit):
            assert not any(bad in k.lower() for bad in ("fair", "market", "edge", "strike")), k

    def test_real_harness_data_does_not_leak(self):
        """The payload actually published from data/harness/."""
        if not (REPO_HARNESS / "snapshots.csv").exists():
            pytest.skip("no harness data on disk")
        pit = build_pit_section(REPO_HARNESS, date(2026, 10, 3))
        for o in pit["observations"]:
            assert tuple(o) == OBSERVATION_FIELDS
        for k in _keys(pit):
            assert not any(bad in k.lower() for bad in ("fair", "market", "edge", "strike")), k


class TestObservations:
    def test_n_counts_settled_only(self, harness):
        pit = build_pit_section(harness, date(2026, 10, 3))
        assert pit["target_n"] == 30
        assert pit["n"] == 2
        assert len(pit["observations"]) == 3

    def test_u_from_pit_ladder_only(self, harness):
        """u uses the first-qualifying (PIT) ladder, not the later repoll."""
        obs = {(o["asset"], o["expiry"][:10]): o
               for o in build_pit_section(harness, date(2026, 10, 3))["observations"]}
        assert obs[("BTC", "2026-09-25")]["u"] == pytest.approx(0.625)
        assert obs[("ETH", "2026-09-26")]["u"] == pytest.approx(0.125)
        assert obs[("BTC", "2026-10-09")]["u"] is None

    def test_calendar_arb_flag(self, harness):
        obs = build_pit_section(harness, date(2026, 10, 3))["observations"]
        assert [o["calendar_arb"] for o in obs] == [False, True, False]


class TestQualification:
    def test_window_is_last_14_days(self, harness):
        q = build_pit_section(harness, date(2026, 10, 3))["qualification"]
        assert q["window"] == {"start": "2026-09-20", "end": "2026-10-03", "days": 14}
        # 2026-09-19 is outside the window
        assert q["by_asset"]["BTC"] == {"polled": 8, "qualified": 1, "rate": 1 / 8}
        assert q["by_asset"]["ETH"] == {"polled": 1, "qualified": 0, "rate": 0.0}

    def test_n_two_sided_best_per_day(self, harness):
        tr = build_pit_section(harness, date(2026, 10, 3))["qualification"]["n_two_sided"]
        assert tr["threshold"] == 20
        assert tr["days"] == ["2026-09-20", "2026-10-03"]
        assert tr["series"]["BTC"] == [30, 30]
        assert tr["series"]["ETH"] == [None, None]     # blank n_two_sided (legacy rows)

    def test_empty_harness(self, tmp_path):
        pit = build_pit_section(tmp_path, date(2026, 10, 3))
        assert pit["n"] == 0 and pit["observations"] == []
        assert pit["qualification"]["by_asset"] == {}
