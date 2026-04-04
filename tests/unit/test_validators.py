"""Unit tests for quantliblab.data.validators.rules."""
from datetime import date

import pytest

from quantliblab.data.validators import (
    ValidationError,
    validate_rate,
    validate_fx_spot,
    validate_price,
    validate_date_series,
)


class TestValidateRate:
    def test_valid_sofr(self):
        validate_rate(0.0530, label="SOFR ON")  # should not raise

    def test_negative_estr(self):
        validate_rate(-0.003, label="ESTR")  # was negative in 2022

    def test_below_floor_raises(self):
        with pytest.raises(ValidationError, match="below the minimum"):
            validate_rate(-0.05, label="SOFR")

    def test_above_ceiling_raises(self):
        with pytest.raises(ValidationError, match="exceeds the maximum"):
            validate_rate(0.30, label="SOFR")

    def test_spike_raises(self):
        with pytest.raises(ValidationError, match="spike threshold"):
            validate_rate(0.053, prev_value=0.03, spike_threshold=0.01)

    def test_normal_daily_move_ok(self):
        validate_rate(0.0533, prev_value=0.0530, spike_threshold=0.02)

    def test_non_numeric_raises(self):
        with pytest.raises(ValidationError):
            validate_rate("n/a", label="SOFR")  # type: ignore


class TestValidateFXSpot:
    def test_valid_eurusd(self):
        validate_fx_spot(1.0850, pair="EURUSD")

    def test_zero_raises(self):
        with pytest.raises(ValidationError, match="must be positive"):
            validate_fx_spot(0.0, pair="EURUSD")

    def test_negative_raises(self):
        with pytest.raises(ValidationError, match="must be positive"):
            validate_fx_spot(-1.08, pair="EURUSD")

    def test_spike_raises(self):
        with pytest.raises(ValidationError, match="spike threshold"):
            validate_fx_spot(1.20, pair="EURUSD", prev_value=1.08,
                             spike_threshold_pct=0.05)

    def test_normal_move_ok(self):
        validate_fx_spot(1.087, pair="EURUSD", prev_value=1.085,
                         spike_threshold_pct=0.05)


class TestValidatePrice:
    def test_valid_crude(self):
        validate_price(82.50, label="WTI crude")

    def test_valid_btc(self):
        validate_price(67_000.0, label="BTC/USD")

    def test_zero_raises(self):
        with pytest.raises(ValidationError, match="must be positive"):
            validate_price(0.0, label="WTI")

    def test_spike_raises(self):
        with pytest.raises(ValidationError, match="spike threshold"):
            validate_price(100.0, label="WTI", prev_value=80.0,
                           spike_threshold_pct=0.15)

    def test_large_crypto_move_ok_with_wider_threshold(self):
        # BTC can move 20% — use wider threshold
        validate_price(80_000.0, label="BTC", prev_value=67_000.0,
                       spike_threshold_pct=0.25)


class TestValidateDateSeries:
    def test_valid_series(self):
        dates = [date(2024, 1, d) for d in [2, 3, 4, 5, 8]]
        validate_date_series(dates)  # weekends skipped, gap <= 7

    def test_not_increasing_raises(self):
        dates = [date(2024, 1, 3), date(2024, 1, 2)]
        with pytest.raises(ValidationError, match="strictly increasing"):
            validate_date_series(dates)

    def test_duplicate_date_raises(self):
        dates = [date(2024, 1, 2), date(2024, 1, 2)]
        with pytest.raises(ValidationError, match="strictly increasing"):
            validate_date_series(dates)

    def test_gap_exceeds_max_raises(self):
        dates = [date(2024, 1, 2), date(2024, 1, 15)]
        with pytest.raises(ValidationError, match="exceeds maximum"):
            validate_date_series(dates, max_gap_days=7)

    def test_no_gaps_allowed_raises(self):
        dates = [date(2024, 1, 2), date(2024, 1, 4)]
        with pytest.raises(ValidationError, match="Consecutive"):
            validate_date_series(dates, allow_gaps=False)

    def test_single_date_ok(self):
        validate_date_series([date(2024, 1, 2)])

    def test_empty_ok(self):
        validate_date_series([])
