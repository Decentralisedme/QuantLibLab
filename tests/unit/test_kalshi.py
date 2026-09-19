"""Unit tests for the Kalshi bracketing, qualifying-expiry, and pricing
logic — pure functions, no network (fetch_events/fetch_ladder are the
network boundary and are exercised only via the live --selftest / harness
run, matching how polymarket.py's fetch_gamma_markets is untested here)."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from quantliblab.harness.deribit_surface import FittedSurface, SliceDiagnostics
from quantliblab.harness.kalshi import (
    Ladder, Strike, bracket, price_ladder, qualify_ladder,
)
from quantliblab.volatility.smile.svi import SVIParams
from quantliblab.volatility.surface.local_vol import LocalVolSurface, SVISlice

F0 = 110_000.0
T1, T2 = 0.05, 0.30
NOW = datetime(2026, 9, 19, tzinfo=timezone.utc)


def _surface(used=(True, True)) -> FittedSurface:
    s1 = SVISlice(T1, F0 * 1.002, SVIParams(0.0015, 0.30, -0.3, 0.0, 0.12))
    s2 = SVISlice(T2, F0 * 1.010, SVIParams(0.0090, 0.35, -0.25, 0.02, 0.20))
    diags = [
        SliceDiagnostics(NOW.date(), T1, s1.F, 20, 0.8, True, used[0]),
        SliceDiagnostics(NOW.date(), T2, s2.F, 25, 1.1, True, used[1]),
    ]
    return FittedSurface("BTC", NOW, F0, LocalVolSurface([s1, s2]), diags)


def _strike(floor_strike: float, bid: float | None, ask: float | None,
            result: str = "") -> Strike:
    return Strike(ticker=f"KXBTCD-TEST-T{floor_strike}", floor_strike=floor_strike,
                  yes_bid=bid, yes_ask=ask, result=result)


def _ladder(strikes: list[Strike], close_time=None) -> Ladder:
    return Ladder("KXBTCD-TEST", "BTC", close_time or NOW + timedelta(days=40),
                   expiration_value=None, strikes=strikes)


# ---------------------------------------------------------------------------
# Bracketing
# ---------------------------------------------------------------------------

class TestBracket:
    def test_interior_T_brackets_the_two_slices(self):
        fs = _surface()
        br = bracket(fs, 0.15)
        assert br is not None
        assert br.T_lower == T1
        assert br.T_upper == T2

    def test_below_front_slice_returns_none(self):
        """H-004: never extrapolate below the front slice."""
        fs = _surface()
        assert bracket(fs, T1 * 0.5) is None

    def test_at_or_above_back_slice_returns_none(self):
        fs = _surface()
        assert bracket(fs, T2) is None
        assert bracket(fs, T2 * 1.5) is None

    def test_no_surface_returns_none(self):
        fs = FittedSurface("BTC", NOW, F0, None, [])
        assert bracket(fs, 0.15) is None


# ---------------------------------------------------------------------------
# Qualifying expiry
# ---------------------------------------------------------------------------

class TestQualifyLadder:
    def _good_strikes(self, n=25, center=F0):
        # monotone non-increasing YES mid as floor_strike rises, all two-sided
        out = []
        for i in range(n):
            k = center * (0.85 + 0.012 * i)
            p = max(0.02, 0.9 - 0.03 * i)
            out.append(_strike(k, bid=round(p - 0.01, 4), ask=round(p + 0.01, 4)))
        return out

    def test_qualifies_when_all_conditions_met(self):
        fs = _surface()
        ladder = _ladder(self._good_strikes())
        qr = qualify_ladder(ladder, fs, 0.15)
        assert qr.ok, qr.reason
        assert qr.bracket is not None

    def test_excluded_when_T_not_bracketed(self):
        fs = _surface()
        ladder = _ladder(self._good_strikes())
        qr = qualify_ladder(ladder, fs, T1 * 0.5)
        assert not qr.ok
        assert "bracketed" in qr.reason

    def test_excluded_on_too_few_two_sided_strikes(self):
        fs = _surface()
        strikes = self._good_strikes(n=5)
        ladder = _ladder(strikes)
        qr = qualify_ladder(ladder, fs, 0.15)
        assert not qr.ok
        assert "two-sided" in qr.reason

    def test_excluded_on_non_monotone_ladder(self):
        """A mispricing in Kalshi's own quotes excludes the expiry —
        it does not get silently priced anyway."""
        fs = _surface()
        strikes = self._good_strikes()
        # inject a violation: raise the mid of a high strike above a lower one
        strikes[-1] = _strike(strikes[-1].floor_strike, bid=0.95, ask=0.99)
        ladder = _ladder(strikes)
        qr = qualify_ladder(ladder, fs, 0.15)
        assert not qr.ok
        assert "monotone" in qr.reason

    def test_excluded_on_empty_ladder(self):
        fs = _surface()
        qr = qualify_ladder(_ladder([]), fs, 0.15)
        assert not qr.ok
        assert "empty" in qr.reason

    def test_excluded_when_bracket_slice_not_used(self):
        fs = _surface(used=(True, False))
        ladder = _ladder(self._good_strikes())
        qr = qualify_ladder(ladder, fs, 0.15)
        assert not qr.ok
        assert "used" in qr.reason


# ---------------------------------------------------------------------------
# Pricing + diagnostics
# ---------------------------------------------------------------------------

class TestPriceLadder:
    def test_rows_carry_bracket_diagnostics(self):
        fs = _surface()
        T = 0.15
        br = bracket(fs, T)
        assert br is not None
        ladder = _ladder([
            _strike(F0 * 0.9, 0.80, 0.82),
            _strike(F0, 0.48, 0.52),
            _strike(F0 * 1.1, 0.15, 0.18),
        ])
        rows = price_ladder(ladder, fs, T, br, now=NOW)
        assert len(rows) == 3
        for row in rows:
            assert row["venue"] == "kalshi"
            assert 0.0 <= float(row["fair"]) <= 1.0
            assert row["T_lower"] == pytest.approx(T1)
            assert row["T_upper"] == pytest.approx(T2)
            assert float(row["sigma_lower"]) > 0.0
            assert float(row["sigma_upper"]) > 0.0
            assert float(row["rmse_lower"]) == pytest.approx(0.8)
            assert float(row["rmse_upper"]) == pytest.approx(1.1)
            assert row["w_atm"] != ""

    def test_fair_decreasing_in_strike(self):
        fs = _surface()
        T = 0.15
        br = bracket(fs, T)
        assert br is not None
        ladder = _ladder([
            _strike(F0 * 0.8, None, None),
            _strike(F0 * 1.0, None, None),
            _strike(F0 * 1.2, None, None),
        ])
        rows = price_ladder(ladder, fs, T, br, now=NOW)
        fairs = [float(r["fair"]) for r in rows]
        assert fairs[0] > fairs[1] > fairs[2]


# ---------------------------------------------------------------------------
# is_pit_observation — first qualifying snapshot per expiry only
# ---------------------------------------------------------------------------

class TestPitObservationFlag:
    def test_first_poll_is_pit_observation(self):
        fs = _surface()
        T = 0.15
        br = bracket(fs, T)
        assert br is not None
        ladder = _ladder([_strike(F0, 0.48, 0.52)])
        rows = price_ladder(ladder, fs, T, br, now=NOW)  # default: is_pit_observation=True
        assert all(r["is_pit_observation"] is True for r in rows)

    def test_repoll_of_known_event_is_not_pit_observation(self):
        fs = _surface()
        T = 0.15
        br = bracket(fs, T)
        assert br is not None
        ladder = _ladder([_strike(F0, 0.48, 0.52)])
        rows = price_ladder(ladder, fs, T, br, now=NOW, is_pit_observation=False)
        assert all(r["is_pit_observation"] is False for r in rows)

    def test_fetch_and_price_flags_only_the_first_poll(self, monkeypatch):
        import quantliblab.harness.kalshi as kalshi_mod

        fs = _surface()
        T = 0.15
        event = {"event_ticker": "KXBTCD-TEST", "series_ticker": "KXBTCD",
                 "product_metadata": {"cadence": "weekly"}}
        close_time = NOW + timedelta(seconds=T * 365 * 86400)
        ladder = _ladder(TestQualifyLadder()._good_strikes(), close_time=close_time)

        monkeypatch.setattr(kalshi_mod, "fetch_events", lambda series_ticker: [event])
        monkeypatch.setattr(kalshi_mod, "fetch_ladder", lambda ev: ladder)

        rows1, _ = kalshi_mod.fetch_and_price({"BTC": fs, "ETH": None}, now=NOW)
        assert rows1 and all(r["is_pit_observation"] is True for r in rows1)

        seen = {r["event_ticker"] for r in rows1}
        rows2, _ = kalshi_mod.fetch_and_price({"BTC": fs, "ETH": None}, now=NOW,
                                               known_pit_events=seen)
        assert rows2 and all(r["is_pit_observation"] is False for r in rows2)


# ---------------------------------------------------------------------------
# calendar_arb_in_bracket — flag, never exclude
# ---------------------------------------------------------------------------

class TestCalendarArbFlag:
    def test_false_on_arb_free_bracket(self):
        fs = _surface()
        br = bracket(fs, 0.15)
        assert br is not None
        assert br.calendar_arb is False

    def test_true_when_bracket_pair_violates_calendar_no_arb(self):
        s1 = SVISlice(T1, F0, SVIParams(0.02, 0.30, -0.3, 0.0, 0.12))
        s2 = SVISlice(T2, F0, SVIParams(0.001, 0.05, 0.0, 0.0, 0.30))
        diags = [
            SliceDiagnostics(NOW.date(), T1, F0, 20, 0.8, True, True),
            SliceDiagnostics(NOW.date(), T2, F0, 25, 1.1, True, True),
        ]
        fs = FittedSurface("BTC", NOW, F0, LocalVolSurface([s1, s2]), diags)
        br = bracket(fs, 0.15)
        assert br is not None
        assert br.calendar_arb is True

    def test_flag_does_not_exclude_the_expiry(self):
        """Not one of H-004's four pre-registered qualifying conditions —
        a flagged bracket still qualifies and gets priced."""
        s1 = SVISlice(T1, F0, SVIParams(0.02, 0.30, -0.3, 0.0, 0.12))
        s2 = SVISlice(T2, F0, SVIParams(0.001, 0.05, 0.0, 0.0, 0.30))
        diags = [
            SliceDiagnostics(NOW.date(), T1, F0, 20, 0.8, True, True),
            SliceDiagnostics(NOW.date(), T2, F0, 25, 1.1, True, True),
        ]
        fs = FittedSurface("BTC", NOW, F0, LocalVolSurface([s1, s2]), diags)
        ladder = _ladder(TestQualifyLadder()._good_strikes())
        qr = qualify_ladder(ladder, fs, 0.15)
        assert qr.ok and qr.bracket is not None
        assert qr.bracket.calendar_arb is True
        rows = price_ladder(ladder, fs, 0.15, qr.bracket, now=NOW)
        assert all(r["calendar_arb_in_bracket"] is True for r in rows)
