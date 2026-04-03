"""
Log-linear interpolation.

Interpolates linearly on log(y), which is equivalent to assuming a
constant instantaneous forward rate between knots.

This is the standard method for interpolating discount factors:
    log(P(t)) interpolated linearly  =>  flat forward between pillars.

Requires all y-values to be strictly positive.
"""
from __future__ import annotations

import bisect
import math

from .base import Extrapolation, Interpolator1D


class LogLinearInterpolator(Interpolator1D):
    """Piecewise log-linear interpolation (constant forward between knots)."""

    _log_ys: list[float]

    def __init__(
        self,
        xs: list[float],
        ys: list[float],
        extrapolation: Extrapolation = Extrapolation.FLAT_FORWARD,
    ) -> None:
        if any(y <= 0 for y in ys):
            raise ValueError("LogLinear requires all y-values to be strictly positive")
        super().__init__(xs, ys, extrapolation)

    def _build(self) -> None:
        self._log_ys = [math.log(y) for y in self.ys]

    def _interpolate(self, x: float) -> float:
        i = bisect.bisect_right(self.xs, x) - 1
        i = min(i, len(self.xs) - 2)
        x0, x1 = self.xs[i], self.xs[i + 1]
        ly0, ly1 = self._log_ys[i], self._log_ys[i + 1]
        t = (x - x0) / (x1 - x0)
        return math.exp(ly0 + t * (ly1 - ly0))
