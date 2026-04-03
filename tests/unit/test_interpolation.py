"""Unit tests for quantliblab.math.interpolation."""
import math

import pytest

from quantliblab.math.interpolation import (
    Extrapolation,
    LinearInterpolator,
    LogLinearInterpolator,
    CubicSplineInterpolator,
    FlatForwardInterpolator,
)

# Shared knots: year fractions and discount factors on a simple flat 5% curve
# P(t) = exp(-0.05 * t)
TIMES = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
DISC  = [math.exp(-0.05 * t) for t in TIMES]
RATES = [0.01, 0.02, 0.03, 0.04, 0.05]   # upward-sloping rate grid
RTIMES = [0.25, 0.5, 1.0, 2.0, 5.0]

TOL = 1e-6


# ---------------------------------------------------------------------------
# Input validation (shared behaviour)
# ---------------------------------------------------------------------------

class TestValidation:
    def test_length_mismatch(self):
        with pytest.raises(ValueError):
            LinearInterpolator([1.0, 2.0], [1.0])

    def test_too_few_points(self):
        with pytest.raises(ValueError):
            LinearInterpolator([1.0], [1.0])

    def test_not_strictly_increasing(self):
        with pytest.raises(ValueError):
            LinearInterpolator([1.0, 1.0, 2.0], [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Linear
# ---------------------------------------------------------------------------

class TestLinear:
    def setup_method(self):
        self.interp = LinearInterpolator(RTIMES, RATES)

    def test_at_knot(self):
        assert abs(self.interp(0.5) - 0.02) < TOL

    def test_midpoint(self):
        # Midpoint between 0.5 (0.02) and 1.0 (0.03) => 0.025
        assert abs(self.interp(0.75) - 0.025) < TOL

    def test_extrapolate_left_flat(self):
        assert self.interp(0.0) == RATES[0]

    def test_extrapolate_right_flat(self):
        assert self.interp(10.0) == RATES[-1]


# ---------------------------------------------------------------------------
# Log-linear
# ---------------------------------------------------------------------------

class TestLogLinear:
    def setup_method(self):
        self.interp = LogLinearInterpolator(TIMES, DISC)

    def test_at_knot(self):
        for t, p in zip(TIMES, DISC):
            assert abs(self.interp(t) - p) < TOL

    def test_midpoint_flat_curve(self):
        # On a flat 5% curve, P(1.5) = exp(-0.075)
        expected = math.exp(-0.05 * 1.5)
        assert abs(self.interp(1.5) - expected) < TOL

    def test_requires_positive_values(self):
        with pytest.raises(ValueError):
            LogLinearInterpolator([0.0, 1.0], [1.0, -0.5])

    def test_extrapolate_right_flat_forward(self):
        interp = LogLinearInterpolator(TIMES, DISC, extrapolation=Extrapolation.FLAT_FORWARD)
        # Beyond t=5: should extend the flat 5% forward
        expected = math.exp(-0.05 * 6.0)
        assert abs(interp(6.0) - expected) < 1e-4


# ---------------------------------------------------------------------------
# Cubic spline
# ---------------------------------------------------------------------------

class TestCubicSpline:
    def setup_method(self):
        self.interp = CubicSplineInterpolator(RTIMES, RATES)

    def test_at_knot(self):
        for t, r in zip(RTIMES, RATES):
            assert abs(self.interp(t) - r) < TOL

    def test_smoothness(self):
        # Spline should be smooth: nearby points should be close
        v1 = self.interp(1.0 - 1e-6)
        v2 = self.interp(1.0 + 1e-6)
        assert abs(v1 - v2) < 1e-4

    def test_extrapolate_flat(self):
        assert self.interp(0.0) == RATES[0]
        assert self.interp(10.0) == RATES[-1]

    def test_monotone_on_monotone_input(self):
        # All rates are increasing; spline should not dip below first or above last
        xs = [self.interp(t) for t in [0.3, 0.7, 1.5, 3.0, 4.0]]
        assert all(RATES[0] <= v <= RATES[-1] for v in xs)


# ---------------------------------------------------------------------------
# Flat-forward
# ---------------------------------------------------------------------------

class TestFlatForward:
    def setup_method(self):
        self.interp = FlatForwardInterpolator(TIMES, DISC)

    def test_at_knot(self):
        for t, p in zip(TIMES, DISC):
            assert abs(self.interp(t) - p) < TOL

    def test_flat_curve_exact(self):
        # Flat 5% curve: flat-forward should reproduce exp(-0.05*t) exactly
        for t in [0.25, 0.75, 1.5, 2.5, 4.0]:
            expected = math.exp(-0.05 * t)
            assert abs(self.interp(t) - expected) < TOL

    def test_extrapolation_beyond_last_pillar(self):
        expected = math.exp(-0.05 * 7.0)
        assert abs(self.interp(7.0) - expected) < 1e-4
