"""
Unit tests for analysis/pit_score.py -- H-004's PIT scoring machinery.

Tested against synthetic ladders where the analytically correct u = F(S*)
is known, since there are zero settled Kalshi PIT expiries in real data
yet (H-004 only just started polling) -- this has to be right before
that data exists.

analysis/ isn't a package, so the module is loaded by file path (same
approach as test_snapshots_atomic_write.py for scripts/).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "analysis" / "pit_score.py"


def _load_pit_score_module():
    spec = importlib.util.spec_from_file_location("pit_score", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pit_score"] = mod
    spec.loader.exec_module(mod)
    return mod


pit_score = _load_pit_score_module()


# ---------------------------------------------------------------------------
# invert_ladder_to_pit -- synthetic ladders, analytically known u
# ---------------------------------------------------------------------------

class TestInvertLadderToPit:
    def test_linear_cdf_recovered_exactly(self):
        """fair(K) = 1 - K/100 on [0, 100] -> CDF(K) = K/100 exactly. The
        ladder's piecewise-linear interpolation matches an already-linear
        underlying function exactly, anywhere between grid points."""
        strikes = np.arange(0.0, 101.0, 10.0)
        fair = 1.0 - strikes / 100.0
        u = pit_score.invert_ladder_to_pit(strikes, fair, settlement=37.0)
        assert u == pytest.approx(0.37, abs=1e-9)

    def test_two_point_midpoint_interpolation(self):
        strikes = [100.0, 200.0]
        fair = [0.8, 0.3]          # CDF = [0.2, 0.7]
        u = pit_score.invert_ladder_to_pit(strikes, fair, settlement=150.0)
        assert u == pytest.approx(0.45, abs=1e-12)

    def test_exact_at_grid_point_for_nonlinear_cdf(self):
        """No interpolation error possible exactly AT a strike, regardless
        of how curved the true underlying CDF is between strikes."""
        mu, sigma = 100_000.0, 8_000.0
        strikes = np.linspace(70_000.0, 130_000.0, 25)
        fair = 1.0 - norm.cdf(strikes, mu, sigma)
        target = strikes[10]
        u = pit_score.invert_ladder_to_pit(strikes, fair, settlement=target)
        assert u == pytest.approx(float(norm.cdf(target, mu, sigma)), abs=1e-9)

    def test_settlement_below_range_clamps_to_zero_end(self):
        strikes = [100.0, 200.0, 300.0]
        fair = [0.9, 0.5, 0.1]     # CDF = [0.1, 0.5, 0.9]
        u = pit_score.invert_ladder_to_pit(strikes, fair, settlement=50.0)
        assert u == pytest.approx(0.1, abs=1e-12)

    def test_settlement_above_range_clamps_to_one_end(self):
        strikes = [100.0, 200.0, 300.0]
        fair = [0.9, 0.5, 0.1]     # CDF = [0.1, 0.5, 0.9]
        u = pit_score.invert_ladder_to_pit(strikes, fair, settlement=500.0)
        assert u == pytest.approx(0.9, abs=1e-12)

    def test_unsorted_strikes_give_same_result_as_sorted(self):
        strikes_sorted = [100.0, 200.0, 300.0]
        fair_sorted = [0.9, 0.5, 0.1]
        u_sorted = pit_score.invert_ladder_to_pit(strikes_sorted, fair_sorted, 220.0)

        order = [2, 0, 1]
        strikes_shuffled = [strikes_sorted[i] for i in order]
        fair_shuffled = [fair_sorted[i] for i in order]
        u_shuffled = pit_score.invert_ladder_to_pit(strikes_shuffled, fair_shuffled, 220.0)

        assert u_shuffled == pytest.approx(u_sorted, abs=1e-12)

    def test_empty_ladder_raises(self):
        with pytest.raises(ValueError):
            pit_score.invert_ladder_to_pit([], [], settlement=100.0)


# ---------------------------------------------------------------------------
# select_pit_rows -- robust to real bool / string / NaN columns
# ---------------------------------------------------------------------------

class TestSelectPitRows:
    def test_selects_true_rows_only(self):
        df = pd.DataFrame({
            "event_ticker": ["A", "A", "B"],
            "is_pit_observation": [True, False, True],
        })
        out = pit_score.select_pit_rows(df)
        assert list(out["event_ticker"]) == ["A", "B"]

    def test_missing_column_returns_empty(self):
        df = pd.DataFrame({"event_ticker": ["A"]})
        out = pit_score.select_pit_rows(df)
        assert len(out) == 0

    def test_all_nan_column_returns_empty(self):
        """Matches the current state of snapshots.csv -- rows written
        before is_pit_observation existed."""
        df = pd.DataFrame({
            "event_ticker": ["A", "B"],
            "is_pit_observation": [float("nan"), float("nan")],
        })
        out = pit_score.select_pit_rows(df)
        assert len(out) == 0


# ---------------------------------------------------------------------------
# score_pit -- ladder grouping + KS test, and the "0 observations" path
# ---------------------------------------------------------------------------

def _settled_ladder(event_ticker: str, strikes, fair, settlement: float) -> pd.DataFrame:
    return pd.DataFrame({
        "event_ticker": event_ticker,
        "strike": strikes,
        "fair": fair,
        "settlement": settlement,
    })


class TestScorePit:
    def test_zero_observations_does_not_crash(self):
        empty = pd.DataFrame(columns=["event_ticker", "strike", "fair", "settlement"])
        result = pit_score.score_pit(empty)
        assert result["n"] == 0
        assert result["ks_stat"] is None
        assert result["ks_pvalue"] is None
        assert len(result["u_values"]) == 0

    def test_one_u_per_event_ticker(self):
        strikes = np.arange(0.0, 101.0, 10.0)
        fair = 1.0 - strikes / 100.0
        df = pd.concat([
            _settled_ladder("EVT-1", strikes, fair, settlement=37.0),
            _settled_ladder("EVT-2", strikes, fair, settlement=82.0),
        ], ignore_index=True)

        result = pit_score.score_pit(df)
        assert result["n"] == 2
        assert sorted(result["u_values"]) == pytest.approx([0.37, 0.82], abs=1e-9)
        assert result["ks_stat"] is not None
        assert 0.0 <= result["ks_pvalue"] <= 1.0

    def test_ks_does_not_reject_genuinely_uniform_observations(self):
        """Sanity check on the KS wiring, not a synthetic-ladder-inversion
        test: feed it u's drawn from Uniform(0,1) directly (bypassing the
        ladder inversion) and confirm the test doesn't spuriously reject."""
        rng = np.random.default_rng(0)
        strikes = np.array([0.0, 1.0])
        rows = []
        for i, u in enumerate(rng.uniform(size=200)):
            fair = [1.0, 1.0 - u]   # CDF = [0, u] -> settlement=1.0 recovers u
            rows.append(_settled_ladder(f"EVT-{i}", strikes, fair, settlement=1.0))
        df = pd.concat(rows, ignore_index=True)

        result = pit_score.score_pit(df)
        assert result["n"] == 200
        assert result["ks_pvalue"] > 0.01   # not a hard guarantee, but flags real breakage


# ---------------------------------------------------------------------------
# resolve_settlements -- network boundary, mocked
# ---------------------------------------------------------------------------

class TestResolveSettlements:
    def test_empty_input_short_circuits_without_network_call(self, monkeypatch):
        def _boom(ticker):
            raise AssertionError("fetch_settlement should not be called for an empty ladder")
        monkeypatch.setattr(pit_score, "fetch_settlement", _boom)

        empty = pd.DataFrame(columns=["event_ticker", "market_id", "strike", "fair"])
        out = pit_score.resolve_settlements(empty)
        assert out.empty

    def test_settled_event_gets_settlement_column(self, monkeypatch):
        monkeypatch.setattr(pit_score, "fetch_settlement", lambda ticker: (1, 55_000.0))

        pit_df = pd.DataFrame({
            "event_ticker": ["EVT-1", "EVT-1"],
            "market_id": ["EVT-1-T50000", "EVT-1-T60000"],
            "strike": [50_000.0, 60_000.0],
            "fair": [0.9, 0.2],
        })
        out = pit_score.resolve_settlements(pit_df)
        assert len(out) == 2
        assert (out["settlement"] == 55_000.0).all()

    def test_unsettled_event_is_dropped(self, monkeypatch):
        monkeypatch.setattr(pit_score, "fetch_settlement", lambda ticker: (None, None))

        pit_df = pd.DataFrame({
            "event_ticker": ["EVT-1"],
            "market_id": ["EVT-1-T50000"],
            "strike": [50_000.0],
            "fair": [0.9],
        })
        out = pit_score.resolve_settlements(pit_df)
        assert out.empty

    def test_fetch_failure_is_skipped_not_raised(self, monkeypatch):
        def _boom(ticker):
            raise ConnectionError("network down")
        monkeypatch.setattr(pit_score, "fetch_settlement", _boom)

        pit_df = pd.DataFrame({
            "event_ticker": ["EVT-1"],
            "market_id": ["EVT-1-T50000"],
            "strike": [50_000.0],
            "fair": [0.9],
        })
        out = pit_score.resolve_settlements(pit_df)   # must not raise
        assert out.empty
