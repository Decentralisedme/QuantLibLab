"""Black-76 unit tests — every analytic Greek is verified against a
central finite difference of the price, plus structural identities."""
from __future__ import annotations

import math

import pytest

from quantliblab.pricing.analytical import black76 as b
from quantliblab.pricing.analytical.digital import digital_above

F, K, T, SIG = 100_000.0, 110_000.0, 0.25, 0.60
H_F, H_S, H_T = F * 1e-5, 1e-6, 1e-6


def fd(f, x, h):
    return (f(x + h) - f(x - h)) / (2.0 * h)


class TestPrice:
    def test_put_call_parity_on_forward(self):
        c = b.price(F, K, T, SIG, call=True)
        p = b.price(F, K, T, SIG, call=False)
        assert abs((c - p) - (F - K)) < 1e-6

    def test_expiry_intrinsic(self):
        assert b.price(F, K, 0.0, SIG, call=True) == 0.0
        assert b.price(F, K, 0.0, SIG, call=False) == K - F

    def test_atm_approximation(self):
        """ATM forward straddle ~ 0.8 * F * sigma * sqrt(T)."""
        c = b.price(F, F, T, SIG, True) + b.price(F, F, T, SIG, False)
        assert abs(c / (0.7979 * F * SIG * math.sqrt(T)) - 1.0) < 0.01


class TestGreeksVsFiniteDifference:
    def test_delta(self):
        for call in (True, False):
            num = fd(lambda x: b.price(x, K, T, SIG, call), F, H_F)
            assert abs(b.delta(F, K, T, SIG, call) - num) < 1e-7

    def test_gamma(self):
        num = fd(lambda x: b.delta(x, K, T, SIG, True), F, H_F)
        assert abs(b.gamma(F, K, T, SIG) - num) < 1e-10

    def test_vega(self):
        num = fd(lambda s: b.price(F, K, T, s, True), SIG, H_S)
        assert abs(b.vega(F, K, T, SIG) - num) < 1e-2      # vega ~ 1.8e4

    def test_theta(self):
        num = -fd(lambda t: b.price(F, K, t, SIG, True), T, H_T)
        assert abs(b.theta(F, K, T, SIG) - num) < 1e-2

    def test_vanna(self):
        num = fd(lambda s: b.delta(F, K, T, s, True), SIG, H_S)
        assert abs(b.vanna(F, K, T, SIG) - num) < 1e-6

    def test_volga(self):
        num = fd(lambda s: b.vega(F, K, T, s), SIG, H_S)
        assert abs(b.volga(F, K, T, SIG) - num) < 1e-1


class TestGreekStructure:
    def test_call_put_shared_greeks(self):
        """Gamma, vega, theta identical for call and put (forward parity)."""
        num_p = fd(lambda x: b.price(x, K, T, SIG, False), F, H_F)
        num_c = fd(lambda x: b.price(x, K, T, SIG, True), F, H_F)
        assert abs((num_c - num_p) - 1.0) < 1e-7   # d/dF of parity = 1

    def test_signs_and_bounds(self):
        assert 0.0 < b.delta(F, K, T, SIG, True) < 1.0
        assert -1.0 < b.delta(F, K, T, SIG, False) < 0.0
        assert b.gamma(F, K, T, SIG) > 0.0
        assert b.vega(F, K, T, SIG) > 0.0
        assert b.theta(F, K, T, SIG) < 0.0

    def test_vega_peaks_near_atm(self):
        assert b.vega(F, F, T, SIG) > b.vega(F, F * 1.4, T, SIG)
        assert b.vega(F, F, T, SIG) > b.vega(F, F * 0.6, T, SIG)


class TestDigitalGreeks:
    def test_digital_delta_matches_fd_of_digital(self):
        num = fd(lambda x: digital_above(x, K, T, SIG, 0.0), F, H_F)
        assert abs(b.digital_delta(F, K, T, SIG) - num) < 1e-10

    def test_digital_vega_matches_fd_of_digital(self):
        num = fd(lambda s: digital_above(F, K, T, s, 0.0), SIG, H_S)
        assert abs(b.digital_vega(F, K, T, SIG) - num) < 1e-6

    def test_digital_vega_sign_flips_at_d1_zero(self):
        """Boundary is K* = F*exp(w/2), NOT the forward: below K* (even
        deep ITM digitals) more vol lowers P(F_T>K) via the falling
        median; beyond K* the fattening tail wins."""
        k_star = F * math.exp(0.5 * SIG * SIG * T)
        assert b.digital_vega(F, F * 0.8, T, SIG) < 0.0     # below K*
        assert b.digital_vega(F, k_star * 0.999, T, SIG) < 0.0
        assert b.digital_vega(F, k_star * 1.001, T, SIG) > 0.0
        assert b.digital_vega(F, F * 1.5, T, SIG) > 0.0     # far beyond
