"""
Brent's method root finder.

Combines bisection, secant, and inverse quadratic interpolation to get
the reliability of bisection with near-superlinear convergence speed.
No derivative required.

This is the recommended default solver for curve bootstrapping and
vol surface calibration where f may not be smooth everywhere.

Reference: Brent, R.P. (1973) "Algorithms for Minimization without Derivatives"
"""
from __future__ import annotations

from collections.abc import Callable


def solve(
    f: Callable[[float], float],
    a: float,
    b: float,
    tol: float = 1e-10,
    max_iter: int = 500,
) -> float:
    """
    Find x in [a, b] such that f(x) == 0.

    Parameters
    ----------
    f        : continuous function with a root in [a, b]
    a, b     : bracketing interval — f(a) and f(b) must have opposite signs
    tol      : convergence tolerance
    max_iter : maximum iterations

    Returns
    -------
    x such that |f(x)| < tol

    Raises
    ------
    ValueError   if f(a) and f(b) have the same sign
    RuntimeError if max_iter is reached without convergence
    """
    fa, fb = f(a), f(b)

    if fa * fb > 0:
        raise ValueError(
            f"Brent requires f(a) and f(b) to have opposite signs. "
            f"Got f({a})={fa:.6e}, f({b})={fb:.6e}"
        )

    if fa == 0.0:
        return a
    if fb == 0.0:
        return b

    # Ensure |f(b)| <= |f(a)| — b is always the better guess
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c, fc = a, fa
    mflag = True
    s = d = 0.0

    for _ in range(max_iter):
        if abs(fb) < tol or abs(b - a) < tol:
            return b

        if fa != fc and fb != fc:
            # Inverse quadratic interpolation
            s = (
                a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb))
            )
        else:
            # Secant method
            s = b - fb * (b - a) / (fb - fa)

        # Conditions to fall back to bisection
        cond1 = not ((3 * a + b) / 4 < s < b or b < s < (3 * a + b) / 4)
        cond2 = mflag and abs(s - b) >= abs(b - c) / 2
        cond3 = (not mflag) and abs(s - b) >= abs(c - d) / 2
        cond4 = mflag and abs(b - c) < tol
        cond5 = (not mflag) and abs(c - d) < tol

        if cond1 or cond2 or cond3 or cond4 or cond5:
            s = (a + b) / 2.0
            mflag = True
        else:
            mflag = False

        fs = f(s)
        d, c, fc = c, b, fb

        if fa * fs < 0:
            b, fb = s, fs
        else:
            a, fa = s, fs

        if abs(fa) < abs(fb):
            a, b = b, a
            fa, fb = fb, fa

    raise RuntimeError(
        f"Brent's method did not converge after {max_iter} iterations. "
        f"Last x={b}, f(x)={fb:.6e}"
    )
