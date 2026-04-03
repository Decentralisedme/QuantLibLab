"""
Newton-Raphson root finder.

Finds x such that f(x) = 0 using the recurrence:
    x_{n+1} = x_n - f(x_n) / f'(x_n)

Fast (quadratic convergence) when the derivative is available and the
initial guess is close to the root.  Falls back gracefully if the
derivative is zero at any step.

Typical use: implied volatility inversion, swap rate solving.
"""
from __future__ import annotations

from collections.abc import Callable


def solve(
    f: Callable[[float], float],
    df: Callable[[float], float],
    x0: float,
    tol: float = 1e-10,
    max_iter: int = 100,
) -> float:
    """
    Find x such that f(x) == 0 starting from initial guess x0.

    Parameters
    ----------
    f        : function whose root we seek
    df       : derivative of f
    x0       : initial guess
    tol      : convergence tolerance on |f(x)|
    max_iter : maximum number of iterations

    Returns
    -------
    x such that |f(x)| < tol

    Raises
    ------
    ZeroDivisionError  if the derivative is zero during iteration
    RuntimeError       if max_iter is reached without convergence
    """
    x = x0
    for i in range(max_iter):
        fx = f(x)
        if abs(fx) < tol:
            return x
        dfx = df(x)
        if dfx == 0.0:
            raise ZeroDivisionError(
                f"Newton-Raphson: derivative is zero at x={x} (iteration {i})"
            )
        x -= fx / dfx

    raise RuntimeError(
        f"Newton-Raphson did not converge after {max_iter} iterations. "
        f"Last x={x}, f(x)={f(x):.6e}"
    )
