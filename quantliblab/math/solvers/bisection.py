"""
Bisection root finder.

Finds x in [a, b] such that f(x) = 0 by repeatedly halving the interval.
Guaranteed to converge as long as f(a) and f(b) have opposite signs.
Slow (linear convergence) but completely reliable.

Typical use: fallback solver, or when only rough bounds are known.
"""
from __future__ import annotations

from collections.abc import Callable


def solve(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> float:
    """
    Find x in [a, b] such that f(x) == 0.

    Parameters
    ----------
    f        : continuous function with a root in [a, b]
    a, b     : bracketing interval — f(a) and f(b) must have opposite signs
    tol      : convergence tolerance on interval width |b - a|
    max_iter : maximum number of halvings

    Returns
    -------
    x such that |f(x)| < tol or |b - a| < tol

    Raises
    ------
    ValueError   if f(a) and f(b) have the same sign (no bracket)
    RuntimeError if max_iter is reached without convergence
    """
    fa, fb = f(a), f(b)

    if fa * fb > 0:
        raise ValueError(
            f"Bisection requires f(a) and f(b) to have opposite signs. "
            f"Got f({a})={fa:.6e}, f({b})={fb:.6e}"
        )

    if fa == 0.0:
        return a
    if fb == 0.0:
        return b

    for _ in range(max_iter):
        mid = (a + b) / 2.0
        fmid = f(mid)

        if abs(fmid) < tol or (b - a) / 2.0 < tol:
            return mid

        if fa * fmid < 0:
            b, fb = mid, fmid
        else:
            a, fa = mid, fmid

    raise RuntimeError(
        f"Bisection did not converge after {max_iter} iterations. "
        f"Interval [{a}, {b}], f(mid)={f((a+b)/2):.6e}"
    )
