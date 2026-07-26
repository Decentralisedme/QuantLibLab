"""Unit tests for delta-space smile conventions."""
from __future__ import annotations

import math

import pytest

from quantliblab.volatility.smile.delta_conventions import (
    forward_delta, k_for_delta, risk_reversal_butterfly, smile_by_delta,
)
from quantliblab.volatility.smile.svi import SVIParams

T = 0.25
SKEWED = SVIParams(a=0.004, b=0.35, rho=-0.35, m=0.0, s=0.20)   # put-skewed


def flat(_k: float) -> float:
    return 0.60


def svi_smile(k: float) -> float:
    return float(SKEWED.implied_vol(k, T))


class TestSolver:
    def test_roundtrip_flat_smile(self):
        """Solve k for a delta, then confirm that k reproduces the delta."""
        for d in (-0.10, -0.25, 0.25, 0.10):
            k = k_for_delta(flat, T, d)
            assert abs(forward_delta(k, flat(k), T, call=d > 0) - d) < 1e-9

    def test_roundtrip_svi_smile(self):
        for d in (-0.10, -0.25, 0.25, 0.10):
            k = k_for_delta(svi_smile, T, d)
            assert abs(forward_delta(k, svi_smile(k), T, call=d > 0) - d) < 1e-9

    def test_strike_ordering(self):
        """10dP < 25dP < 25dC < 10dC in strike (k) space."""
        ks = [k_for_delta(svi_smile, T, d) for d in (-0.10, -0.25, 0.25, 0.10)]
        assert ks[0] < ks[1] < ks[2] < ks[3]
        assert ks[0] < 0.0 < ks[3]          # puts below forward, calls above

    def test_invalid_delta_rejected(self):
        for bad in (0.0, 1.0, -1.0, 1.5):
            with pytest.raises(ValueError):
                k_for_delta(flat, T, bad)


class TestQuotes:
    def test_grid_labels_and_atm_anchor(self):
        q = smile_by_delta(svi_smile, T)
        assert set(q) == {"10dP", "25dP", "ATM", "25dC", "10dC"}
        assert q["ATM"]["k"] == 0.0
        assert abs(q["ATM"]["sigma"] - svi_smile(0.0)) < 1e-12

    def test_put_skew_shows_in_rr(self):
        """rho < 0 (put skew): puts richer than calls -> RR25 < 0."""
        q = smile_by_delta(svi_smile, T)
        rrbf = risk_reversal_butterfly(q)
        assert rrbf["rr25"] < 0.0
        assert rrbf["bf25"] > 0.0            # convex smile: wings above ATM

    def test_flat_smile_degenerates(self):
        q = smile_by_delta(flat, T)
        rrbf = risk_reversal_butterfly(q)
        assert abs(rrbf["rr25"]) < 1e-9
        assert abs(rrbf["bf25"]) < 1e-9
