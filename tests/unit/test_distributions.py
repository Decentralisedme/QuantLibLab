"""Unit tests for quantliblab.math.distributions.normal."""
import math

import numpy as np
import pytest

from quantliblab.math.distributions import (
    pdf, cdf, ppf,
    general_pdf, general_cdf, general_ppf,
)

TOL = 1e-10
LOOSE = 1e-6   # for values cross-checked against known tables


class TestPDF:
    def test_at_zero(self):
        # phi(0) = 1 / sqrt(2*pi)
        expected = 1.0 / math.sqrt(2.0 * math.pi)
        assert abs(float(pdf(0.0)) - expected) < TOL

    def test_symmetry(self):
        assert abs(float(pdf(1.5)) - float(pdf(-1.5))) < TOL

    def test_positive_everywhere(self):
        xs = np.linspace(-5, 5, 100)
        assert np.all(pdf(xs) > 0)

    def test_integrates_to_one(self):
        # Numerical integration over [-10, 10]
        xs = np.linspace(-10, 10, 100_000)
        integral = np.trapezoid(pdf(xs), xs)
        assert abs(integral - 1.0) < 1e-5

    def test_array_input(self):
        result = pdf(np.array([0.0, 1.0, -1.0]))
        assert result.shape == (3,)


class TestCDF:
    def test_at_zero(self):
        assert abs(float(cdf(0.0)) - 0.5) < TOL

    def test_symmetry(self):
        # Phi(x) + Phi(-x) = 1
        assert abs(float(cdf(1.96)) + float(cdf(-1.96)) - 1.0) < TOL

    def test_known_values(self):
        # Standard table values
        assert abs(float(cdf(1.645)) - 0.95) < 1e-3   # ~95th percentile
        assert abs(float(cdf(1.96))  - 0.975) < 1e-3  # ~97.5th percentile
        assert abs(float(cdf(2.576)) - 0.995) < 1e-3  # ~99.5th percentile

    def test_limits(self):
        assert float(cdf(-10.0)) < 1e-20
        assert float(cdf(10.0)) > 1.0 - 1e-10

    def test_monotone(self):
        xs = np.linspace(-3, 3, 50)
        vals = cdf(xs)
        assert np.all(np.diff(vals) > 0)


class TestPPF:
    def test_inverse_of_cdf(self):
        for x in [-2.0, -1.0, 0.0, 1.0, 2.0]:
            assert abs(float(ppf(cdf(x))) - x) < LOOSE

    def test_median(self):
        assert abs(float(ppf(0.5)) - 0.0) < TOL

    def test_known_quantiles(self):
        assert abs(float(ppf(0.975)) - 1.96) < 1e-2
        assert abs(float(ppf(0.995)) - 2.576) < 1e-2

    def test_boundary_raises(self):
        with pytest.raises(ValueError):
            ppf(0.0)
        with pytest.raises(ValueError):
            ppf(1.0)
        with pytest.raises(ValueError):
            ppf(-0.1)

    def test_array_input(self):
        result = ppf(np.array([0.25, 0.5, 0.75]))
        assert result.shape == (3,)
        assert result[1] == pytest.approx(0.0, abs=TOL)


class TestGeneralNormal:
    def test_pdf_mean_shift(self):
        # N(2, 1): peak at x=2
        assert float(general_pdf(2.0, mu=2.0, sigma=1.0)) == pytest.approx(
            float(pdf(0.0)), rel=TOL
        )

    def test_pdf_scale(self):
        # Wider sigma -> lower peak
        assert float(general_pdf(0.0, mu=0.0, sigma=2.0)) < float(pdf(0.0))

    def test_cdf_at_mean(self):
        assert float(general_cdf(5.0, mu=5.0, sigma=1.0)) == pytest.approx(0.5, abs=TOL)

    def test_ppf_round_trip(self):
        mu, sigma = 3.0, 0.5
        for p in [0.1, 0.5, 0.9]:
            x = float(general_ppf(p, mu, sigma))
            assert float(general_cdf(x, mu, sigma)) == pytest.approx(p, abs=LOOSE)

    def test_invalid_sigma(self):
        with pytest.raises(ValueError):
            general_pdf(0.0, mu=0.0, sigma=0.0)
        with pytest.raises(ValueError):
            general_cdf(0.0, mu=0.0, sigma=-1.0)
