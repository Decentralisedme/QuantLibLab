"""
Standard normal distribution: PDF, CDF, and inverse CDF (ppf).

Used by:
  - Black-Scholes formula (d1, d2 terms)
  - Monte Carlo path generation (sampling)
  - Vol surface calibration (probability transforms)

We delegate to scipy.special for numerical accuracy rather than
reimplementing Abramowitz & Stegun approximations.  scipy is already
a hard dependency of the project.

All functions operate on scalars and numpy arrays interchangeably.
"""
from __future__ import annotations

import math
from typing import Union

import numpy as np
from scipy import special, stats

# Type alias — accepts float or numpy array
Numeric = Union[float, np.ndarray]


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def pdf(x: Numeric) -> Numeric:
    """
    Standard normal probability density function.

        phi(x) = exp(-x^2 / 2) / sqrt(2*pi)
    """
    return np.exp(-0.5 * np.asarray(x, dtype=float) ** 2) / math.sqrt(2.0 * math.pi)


def cdf(x: Numeric) -> Numeric:
    """
    Standard normal cumulative distribution function.

        Phi(x) = P(Z <= x),  Z ~ N(0, 1)

    Uses scipy.special.ndtr for full machine-precision accuracy.
    """
    return special.ndtr(x)


def ppf(p: Numeric) -> Numeric:
    """
    Inverse CDF (percent-point function / quantile function).

        ppf(p) = x  such that  Phi(x) = p,  p in (0, 1)

    Used to convert uniform random samples to standard normal samples
    in quasi-Monte Carlo path generation.

    Raises ValueError if any p is outside (0, 1).
    """
    p = np.asarray(p, dtype=float)
    if np.any(p <= 0.0) or np.any(p >= 1.0):
        raise ValueError("ppf requires p in the open interval (0, 1)")
    return special.ndtri(p)


# ---------------------------------------------------------------------------
# Convenience: moments
# ---------------------------------------------------------------------------

def mean() -> float:
    """Mean of the standard normal distribution."""
    return 0.0


def variance() -> float:
    """Variance of the standard normal distribution."""
    return 1.0


def std() -> float:
    """Standard deviation of the standard normal distribution."""
    return 1.0


# ---------------------------------------------------------------------------
# Convenience: general normal N(mu, sigma)
# ---------------------------------------------------------------------------

def general_pdf(x: Numeric, mu: float, sigma: float) -> Numeric:
    """PDF of N(mu, sigma^2)."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    z = (np.asarray(x, dtype=float) - mu) / sigma
    return pdf(z) / sigma


def general_cdf(x: Numeric, mu: float, sigma: float) -> Numeric:
    """CDF of N(mu, sigma^2)."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    z = (np.asarray(x, dtype=float) - mu) / sigma
    return cdf(z)


def general_ppf(p: Numeric, mu: float, sigma: float) -> Numeric:
    """Inverse CDF of N(mu, sigma^2)."""
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    return mu + sigma * ppf(p)
