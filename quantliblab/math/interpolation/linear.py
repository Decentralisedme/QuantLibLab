"""
Linear interpolation.

Connects consecutive knots with straight lines.
Simple and fast; discontinuous first derivative at knots.

Typical use: FX forward points, rough rate grids.
"""
from __future__ import annotations

import bisect

from .base import Extrapolation, Interpolator1D


class LinearInterpolator(Interpolator1D):
    """Piecewise linear interpolation."""

    def __init__(
        self,
        xs: list[float],
        ys: list[float],
        extrapolation: Extrapolation = Extrapolation.FLAT,
    ) -> None:
        super().__init__(xs, ys, extrapolation)

    def _interpolate(self, x: float) -> float:
        i = bisect.bisect_right(self.xs, x) - 1
        i = min(i, len(self.xs) - 2)
        x0, x1 = self.xs[i], self.xs[i + 1]
        y0, y1 = self.ys[i], self.ys[i + 1]
        t = (x - x0) / (x1 - x0)
        return y0 + t * (y1 - y0)
