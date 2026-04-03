"""
Flat-forward interpolation.

Between two knots t1 and t2, the instantaneous forward rate f is held
constant at the value implied by the two discount factors P(t1) and P(t2):

    f = -[log P(t2) - log P(t1)] / (t2 - t1)

Equivalently, this is log-linear interpolation on discount factors — but
the class is provided separately because it is the standard method named
in term-sheet and ISDA documentation for OIS curve interpolation.

Extrapolation default: FLAT_FORWARD (hold the last forward rate beyond
the final pillar, and the first forward rate before the first pillar).
"""
from __future__ import annotations

from .base import Extrapolation
from .log_linear import LogLinearInterpolator


class FlatForwardInterpolator(LogLinearInterpolator):
    """
    Flat-forward interpolation on discount factors.

    Mathematically identical to log-linear; provided as a named class
    so curve builders can select it explicitly by convention.
    """

    def __init__(
        self,
        xs: list[float],
        ys: list[float],
        extrapolation: Extrapolation = Extrapolation.FLAT_FORWARD,
    ) -> None:
        super().__init__(xs, ys, extrapolation)
