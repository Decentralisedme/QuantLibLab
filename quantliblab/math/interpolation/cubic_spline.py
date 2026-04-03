"""
Natural cubic spline interpolation.

Fits a piecewise cubic polynomial such that:
  - The curve passes through every knot
  - First and second derivatives are continuous everywhere
  - Second derivative is zero at the two endpoints (natural boundary)

Produces smooth curves but can oscillate with uneven knot spacing.
Typical use: smooth yield curves, vol surfaces.

Implementation uses the standard tridiagonal system solve (Thomas algorithm).
"""
from __future__ import annotations

import bisect

from .base import Extrapolation, Interpolator1D


class CubicSplineInterpolator(Interpolator1D):
    """Natural cubic spline (C2 continuous, zero second derivative at endpoints)."""

    _h: list[float]       # knot spacings
    _c: list[float]       # second-derivative coefficients (sigma)
    _b: list[float]
    _d: list[float]

    def __init__(
        self,
        xs: list[float],
        ys: list[float],
        extrapolation: Extrapolation = Extrapolation.FLAT,
    ) -> None:
        super().__init__(xs, ys, extrapolation)

    def _build(self) -> None:
        n = len(self.xs)
        h = [self.xs[i + 1] - self.xs[i] for i in range(n - 1)]

        # Build tridiagonal system for second derivatives (sigma)
        # Natural spline: sigma[0] = sigma[n-1] = 0
        alpha = [0.0] * n
        for i in range(1, n - 1):
            alpha[i] = (
                3.0 / h[i] * (self.ys[i + 1] - self.ys[i])
                - 3.0 / h[i - 1] * (self.ys[i] - self.ys[i - 1])
            )

        # Thomas algorithm (forward sweep)
        l = [1.0] * n
        mu = [0.0] * n
        z = [0.0] * n

        for i in range(1, n - 1):
            l[i] = 2.0 * (self.xs[i + 1] - self.xs[i - 1]) - h[i - 1] * mu[i - 1]
            mu[i] = h[i] / l[i]
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

        # Back substitution
        sigma = [0.0] * n
        b = [0.0] * (n - 1)
        d = [0.0] * (n - 1)

        for j in range(n - 2, -1, -1):
            sigma[j] = z[j] - mu[j] * sigma[j + 1]

        for i in range(n - 1):
            b[i] = (self.ys[i + 1] - self.ys[i]) / h[i] - h[i] * (sigma[i + 1] + 2.0 * sigma[i]) / 3.0
            d[i] = (sigma[i + 1] - sigma[i]) / (3.0 * h[i])

        self._h = h
        self._c = sigma   # second-derivative / 2
        self._b = b
        self._d = d

    def _interpolate(self, x: float) -> float:
        i = bisect.bisect_right(self.xs, x) - 1
        i = min(i, len(self.xs) - 2)
        dx = x - self.xs[i]
        return (
            self.ys[i]
            + self._b[i] * dx
            + self._c[i] * dx**2
            + self._d[i] * dx**3
        )
