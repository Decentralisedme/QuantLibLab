"""Unit tests for quantliblab.math.solvers."""
import math

import pytest

from quantliblab.math.solvers import newton_raphson, bisection, brent


# Simple test functions reused across all solvers
def f_quadratic(x: float) -> float:
    """x^2 - 2 = 0  =>  root at sqrt(2) ~ 1.41421356"""
    return x * x - 2.0


def df_quadratic(x: float) -> float:
    return 2.0 * x


def f_cubic(x: float) -> float:
    """x^3 - x - 2 = 0  =>  root at ~1.52138"""
    return x**3 - x - 2.0


def df_cubic(x: float) -> float:
    return 3.0 * x**2 - 1.0


def f_trig(x: float) -> float:
    """cos(x) - x = 0  =>  Dottie number ~0.739085"""
    return math.cos(x) - x


def df_trig(x: float) -> float:
    return -math.sin(x) - 1.0


SQRT2 = math.sqrt(2)
TOL = 1e-8


# ---------------------------------------------------------------------------
# Newton-Raphson
# ---------------------------------------------------------------------------

class TestNewtonRaphson:
    def test_sqrt2(self):
        root = newton_raphson(f_quadratic, df_quadratic, x0=1.0)
        assert abs(root - SQRT2) < TOL

    def test_cubic(self):
        root = newton_raphson(f_cubic, df_cubic, x0=1.5)
        assert abs(f_cubic(root)) < TOL

    def test_trig(self):
        root = newton_raphson(f_trig, df_trig, x0=0.5)
        assert abs(f_trig(root)) < TOL

    def test_zero_derivative_raises(self):
        with pytest.raises(ZeroDivisionError):
            newton_raphson(lambda x: x**2, lambda x: 0.0, x0=1.0)

    def test_no_convergence_raises(self):
        with pytest.raises(RuntimeError):
            # Oscillating function — NR diverges
            newton_raphson(lambda x: x**3 - 2 * x + 2, lambda x: 3 * x**2 - 2,
                           x0=0.0, max_iter=5)


# ---------------------------------------------------------------------------
# Bisection
# ---------------------------------------------------------------------------

class TestBisection:
    def test_sqrt2(self):
        root = bisection(f_quadratic, a=1.0, b=2.0)
        assert abs(root - SQRT2) < TOL

    def test_cubic(self):
        root = bisection(f_cubic, a=1.0, b=2.0)
        assert abs(f_cubic(root)) < TOL

    def test_no_bracket_raises(self):
        with pytest.raises(ValueError):
            bisection(f_quadratic, a=2.0, b=3.0)  # both positive

    def test_exact_boundary(self):
        # f(a) = 0 exactly
        root = bisection(lambda x: x - 1.5, a=1.5, b=3.0)
        assert root == 1.5


# ---------------------------------------------------------------------------
# Brent
# ---------------------------------------------------------------------------

class TestBrent:
    def test_sqrt2(self):
        root = brent(f_quadratic, a=1.0, b=2.0)
        assert abs(root - SQRT2) < TOL

    def test_cubic(self):
        root = brent(f_cubic, a=1.0, b=2.0)
        assert abs(f_cubic(root)) < TOL

    def test_trig(self):
        root = brent(f_trig, a=0.0, b=1.0)
        assert abs(f_trig(root)) < TOL

    def test_no_bracket_raises(self):
        with pytest.raises(ValueError):
            brent(f_quadratic, a=2.0, b=3.0)

    def test_wide_bracket(self):
        # f(0) = -2, f(100) = 9998 — valid wide bracket, root at sqrt(2)
        root = brent(f_quadratic, a=0.0, b=100.0)
        assert abs(root - SQRT2) < TOL
