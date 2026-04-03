"""
Descriptive statistics: moments and risk metrics.

Used by:
  - P&L attribution (mean return, vol, Sharpe)
  - Monte Carlo output analysis (moments of simulated paths)
  - Vol surface calibration (fitting to market moments)

All functions accept 1-D numpy arrays or list of floats.
ddof=1 (sample statistics) is the default throughout.
"""
from __future__ import annotations

import math

import numpy as np


# ---------------------------------------------------------------------------
# Central moments
# ---------------------------------------------------------------------------

def mean(x: np.ndarray) -> float:
    """Arithmetic mean."""
    return float(np.mean(x))


def variance(x: np.ndarray, ddof: int = 1) -> float:
    """
    Sample variance (ddof=1 by default).

    Use ddof=0 for population variance.
    """
    return float(np.var(x, ddof=ddof))


def std(x: np.ndarray, ddof: int = 1) -> float:
    """Sample standard deviation."""
    return float(np.std(x, ddof=ddof))


def skewness(x: np.ndarray) -> float:
    """
    Excess skewness (Fisher's definition).

    Positive => right tail heavier (common in commodity returns).
    Negative => left tail heavier (common in equity returns).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 3:
        raise ValueError("Skewness requires at least 3 observations")
    m = mean(x)
    s = std(x, ddof=1)
    if s == 0.0:
        return 0.0
    return float(np.mean(((x - m) / s) ** 3))


def kurtosis(x: np.ndarray, excess: bool = True) -> float:
    """
    Kurtosis of the sample.

    excess=True  (default) returns excess kurtosis (normal = 0).
    excess=False returns raw kurtosis (normal = 3).

    High excess kurtosis => fat tails (relevant for crypto, commodities).
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 4:
        raise ValueError("Kurtosis requires at least 4 observations")
    m = mean(x)
    s = std(x, ddof=1)
    if s == 0.0:
        return 0.0
    raw = float(np.mean(((x - m) / s) ** 4))
    return raw - 3.0 if excess else raw


# ---------------------------------------------------------------------------
# Risk metrics (finance-specific)
# ---------------------------------------------------------------------------

def sharpe_ratio(returns: np.ndarray, risk_free: float = 0.0, ddof: int = 1) -> float:
    """
    Annualised Sharpe ratio assuming daily returns.

        Sharpe = (mean(r) - rf) / std(r) * sqrt(252)

    Parameters
    ----------
    returns    : daily return series
    risk_free  : daily risk-free rate (default 0)
    ddof       : degrees of freedom for std (default 1)
    """
    returns = np.asarray(returns, dtype=float)
    excess = returns - risk_free
    s = std(excess, ddof=ddof)
    if s == 0.0:
        return 0.0
    return float(mean(excess) / s * math.sqrt(252))


def max_drawdown(prices: np.ndarray) -> float:
    """
    Maximum peak-to-trough drawdown as a fraction.

    Returns a value in [-1, 0]; e.g. -0.20 means a 20% drawdown.
    """
    prices = np.asarray(prices, dtype=float)
    if len(prices) < 2:
        raise ValueError("max_drawdown requires at least 2 prices")
    peak = np.maximum.accumulate(prices)
    drawdown = (prices - peak) / peak
    return float(np.min(drawdown))


def rolling_volatility(returns: np.ndarray, window: int) -> np.ndarray:
    """
    Rolling annualised volatility (standard deviation * sqrt(252)).

    Returns an array of the same length; first (window-1) values are NaN.
    """
    returns = np.asarray(returns, dtype=float)
    result = np.full(len(returns), np.nan)
    for i in range(window - 1, len(returns)):
        result[i] = std(returns[i - window + 1: i + 1]) * math.sqrt(252)
    return result
