"""
Ordinary Least Squares (OLS) linear regression.

Used by:
  - Beta estimation (asset vs benchmark)
  - Factor model fitting (Fama-French, macro factors)
  - Curve fitting residuals during bootstrapping
  - P&L attribution decomposition

Provides a lightweight OLS implementation plus a result object
with the standard regression statistics.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RegressionResult:
    """
    Output of an OLS regression  y = X @ beta + eps.

    Attributes
    ----------
    beta        : coefficient vector (shape: n_features [+ 1 if intercept])
    residuals   : y - y_hat
    r_squared   : coefficient of determination
    std_errors  : standard errors of beta estimates
    t_stats     : t-statistics for each coefficient
    n           : number of observations
    """
    beta: np.ndarray
    residuals: np.ndarray
    r_squared: float
    std_errors: np.ndarray
    t_stats: np.ndarray
    n: int

    @property
    def y_hat(self) -> np.ndarray:
        """In-sample fitted values (residuals must be stored separately)."""
        raise AttributeError(
            "y_hat is not stored. Compute as y - residuals from outside."
        )


def ols(
    x: np.ndarray,
    y: np.ndarray,
    intercept: bool = True,
) -> RegressionResult:
    """
    Fit y = X @ beta (+ intercept) via OLS.

    Parameters
    ----------
    x         : (n, k) feature matrix or (n,) vector for simple regression
    y         : (n,) response vector
    intercept : if True, prepend a column of ones to x

    Returns
    -------
    RegressionResult with beta, residuals, R², standard errors, t-stats
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim != 1:
        raise ValueError("y must be a 1-D array")
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"x and y must have the same number of rows. Got {x.shape[0]} vs {y.shape[0]}")

    n, k = x.shape

    if intercept:
        x = np.column_stack([np.ones(n), x])
        k += 1

    if n <= k:
        raise ValueError(
            f"Under-determined system: {n} observations, {k} parameters. Need n > k."
        )

    # Normal equations: beta = (X'X)^{-1} X'y
    XtX = x.T @ x
    Xty = x.T @ y

    try:
        beta = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"OLS failed — design matrix is singular: {e}") from e

    y_hat = x @ beta
    residuals = y - y_hat

    # R²
    ss_res = float(residuals @ residuals)
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Standard errors
    sigma2 = ss_res / (n - k)                      # unbiased variance estimate
    cov_beta = sigma2 * np.linalg.inv(XtX)
    std_errors = np.sqrt(np.diag(cov_beta))

    t_stats = beta / std_errors

    return RegressionResult(
        beta=beta,
        residuals=residuals,
        r_squared=r_squared,
        std_errors=std_errors,
        t_stats=t_stats,
        n=n,
    )


def beta_estimate(asset_returns: np.ndarray, benchmark_returns: np.ndarray) -> float:
    """
    Convenience wrapper: estimate the market beta of an asset.

        asset = alpha + beta * benchmark + eps

    Returns the slope coefficient (beta).
    """
    result = ols(benchmark_returns, asset_returns, intercept=True)
    return float(result.beta[1])   # beta[0] = intercept, beta[1] = slope
