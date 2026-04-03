"""
Base class and extrapolation enum shared by all interpolators.

Every interpolator accepts an Extrapolation setting:
  FLAT          — repeat the boundary value beyond the known range
  FLAT_FORWARD  — hold the last instantaneous forward rate constant (log-linear extension)
                  Only meaningful when the y-values are discount factors.
                  For rate curves, use FLAT on the right end instead.
"""
from __future__ import annotations

from enum import Enum


class Extrapolation(str, Enum):
    FLAT = "FLAT"
    FLAT_FORWARD = "FLAT_FORWARD"


class Interpolator1D:
    """
    Abstract base for 1-D interpolators.

    Subclasses must implement _interpolate(x) for x strictly inside
    the knot range.  Extrapolation is handled here in __call__.
    """

    def __init__(
        self,
        xs: list[float],
        ys: list[float],
        extrapolation: Extrapolation = Extrapolation.FLAT,
    ) -> None:
        if len(xs) != len(ys):
            raise ValueError("xs and ys must have the same length")
        if len(xs) < 2:
            raise ValueError("At least two data points are required")
        if any(xs[i] >= xs[i + 1] for i in range(len(xs) - 1)):
            raise ValueError("xs must be strictly increasing")

        self.xs = list(xs)
        self.ys = list(ys)
        self.extrapolation = extrapolation
        self._build()

    def _build(self) -> None:
        """Pre-compute any coefficients.  Override in subclasses."""

    def _interpolate(self, x: float) -> float:
        raise NotImplementedError

    def __call__(self, x: float) -> float:
        if x < self.xs[0]:
            return self._extrapolate_left(x)
        if x > self.xs[-1]:
            return self._extrapolate_right(x)
        return self._interpolate(x)

    def _extrapolate_left(self, x: float) -> float:
        if self.extrapolation == Extrapolation.FLAT:
            return self.ys[0]
        # FLAT_FORWARD on the left: flat rate (same logic as flat)
        return self.ys[0]

    def _extrapolate_right(self, x: float) -> float:
        if self.extrapolation == Extrapolation.FLAT:
            return self.ys[-1]
        # FLAT_FORWARD: extend log-linearly using the last forward rate
        x1, x2 = self.xs[-2], self.xs[-1]
        y1, y2 = self.ys[-2], self.ys[-1]
        if y1 <= 0 or y2 <= 0:
            # Fall back to flat if values are not positive (not discount factors)
            return self.ys[-1]
        import math
        log_slope = (math.log(y2) - math.log(y1)) / (x2 - x1)
        return y2 * math.exp(log_slope * (x - x2))
