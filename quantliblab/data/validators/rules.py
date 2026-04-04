"""
Market data validation rules.

Run before writing to the store to catch bad ticks early.
Each validator raises ValidationError on failure.

Validators provided:
  - validate_rate       : interest rates (bounds, sign, spike)
  - validate_fx_spot    : FX spot prices (positive, spike)
  - validate_price      : generic positive price (commodities, crypto)
  - validate_date_series: date ordering and gap checks
"""
from __future__ import annotations

from datetime import date


class ValidationError(Exception):
    """Raised when a market data point fails a validation rule."""


# ---------------------------------------------------------------------------
# Rate validators
# ---------------------------------------------------------------------------

def validate_rate(
    value: float,
    label: str = "rate",
    min_rate: float = -0.02,   # -2% floor (covers negative ESTR)
    max_rate: float = 0.25,    # 25% ceiling
    prev_value: float | None = None,
    spike_threshold: float = 0.02,  # 200bps one-day move
) -> None:
    """
    Validate an interest rate (e.g. SOFR, SONIA, ESTR).

    Parameters
    ----------
    value           : the rate to validate (e.g. 0.0530 for 5.30%)
    label           : human-readable name for error messages
    min_rate        : minimum allowable rate
    max_rate        : maximum allowable rate
    prev_value      : previous day's rate for spike detection (optional)
    spike_threshold : maximum allowable one-day absolute move
    """
    if not isinstance(value, (int, float)):
        raise ValidationError(f"{label}: expected a number, got {type(value).__name__}")

    if value < min_rate:
        raise ValidationError(
            f"{label}={value:.4f} is below the minimum allowed rate {min_rate:.4f}"
        )

    if value > max_rate:
        raise ValidationError(
            f"{label}={value:.4f} exceeds the maximum allowed rate {max_rate:.4f}"
        )

    if prev_value is not None:
        move = abs(value - prev_value)
        if move > spike_threshold:
            raise ValidationError(
                f"{label}: one-day move of {move:.4f} ({move*100:.0f}bps) "
                f"exceeds spike threshold of {spike_threshold:.4f} "
                f"({spike_threshold*100:.0f}bps). "
                f"prev={prev_value:.4f}, new={value:.4f}"
            )


# ---------------------------------------------------------------------------
# FX validators
# ---------------------------------------------------------------------------

def validate_fx_spot(
    value: float,
    pair: str = "FX",
    prev_value: float | None = None,
    spike_threshold_pct: float = 0.05,  # 5% one-day move
) -> None:
    """
    Validate an FX spot rate.

    Parameters
    ----------
    value                : spot rate (must be strictly positive)
    pair                 : e.g. "EURUSD" — used in error messages
    prev_value           : previous day's rate for spike detection
    spike_threshold_pct  : maximum allowable one-day percentage move
    """
    if value <= 0:
        raise ValidationError(
            f"{pair} spot rate must be positive, got {value}"
        )

    if prev_value is not None:
        if prev_value <= 0:
            raise ValidationError(
                f"{pair} previous spot rate must be positive, got {prev_value}"
            )
        pct_move = abs(value - prev_value) / prev_value
        if pct_move > spike_threshold_pct:
            raise ValidationError(
                f"{pair}: one-day move of {pct_move:.1%} exceeds "
                f"spike threshold of {spike_threshold_pct:.1%}. "
                f"prev={prev_value:.4f}, new={value:.4f}"
            )


# ---------------------------------------------------------------------------
# Generic price validator
# ---------------------------------------------------------------------------

def validate_price(
    value: float,
    label: str = "price",
    prev_value: float | None = None,
    spike_threshold_pct: float = 0.15,  # 15% — commodities/crypto move more
) -> None:
    """
    Validate any positive price (commodity futures, crypto spot).

    Parameters
    ----------
    value               : the price to validate
    label               : e.g. "WTI crude", "BTC/USD"
    prev_value          : previous day's price for spike detection
    spike_threshold_pct : maximum allowable one-day percentage move
    """
    if value <= 0:
        raise ValidationError(f"{label}: price must be positive, got {value}")

    if prev_value is not None:
        if prev_value <= 0:
            raise ValidationError(
                f"{label}: previous price must be positive, got {prev_value}"
            )
        pct_move = abs(value - prev_value) / prev_value
        if pct_move > spike_threshold_pct:
            raise ValidationError(
                f"{label}: one-day move of {pct_move:.1%} exceeds "
                f"spike threshold of {spike_threshold_pct:.1%}. "
                f"prev={prev_value:.4f}, new={value:.4f}"
            )


# ---------------------------------------------------------------------------
# Date series validator
# ---------------------------------------------------------------------------

def validate_date_series(
    dates: list[date],
    allow_gaps: bool = True,
    max_gap_days: int = 7,
) -> None:
    """
    Validate a list of dates for ordering and excessive gaps.

    Parameters
    ----------
    dates        : list of dates (need not be business days)
    allow_gaps   : if False, any gap > 1 day raises an error
    max_gap_days : maximum allowable gap between consecutive dates
    """
    if len(dates) < 2:
        return

    for i in range(1, len(dates)):
        if dates[i] <= dates[i - 1]:
            raise ValidationError(
                f"Dates must be strictly increasing. "
                f"Found {dates[i]} after {dates[i-1]} at position {i}."
            )
        gap = (dates[i] - dates[i - 1]).days
        if not allow_gaps and gap > 1:
            raise ValidationError(
                f"Gap of {gap} days between {dates[i-1]} and {dates[i]}. "
                f"Consecutive dates required."
            )
        if gap > max_gap_days:
            raise ValidationError(
                f"Gap of {gap} days between {dates[i-1]} and {dates[i]} "
                f"exceeds maximum allowed gap of {max_gap_days} days."
            )
