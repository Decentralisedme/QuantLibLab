"""
Tenor parsing and date arithmetic.

Examples:
    Tenor.from_string("3M").add_to(date(2024, 1, 31)) -> date(2024, 4, 30)
    Tenor.from_string("1Y").in_years()                 -> 1.0
"""
from __future__ import annotations

import re
from datetime import date, timedelta
from enum import Enum

from dateutil.relativedelta import relativedelta

_TENOR_RE = re.compile(r"^(\d+)(D|W|M|Y)$", re.IGNORECASE)

# Overnight / Tom-Next shorthands
_SHORTHANDS: dict[str, tuple[int, str]] = {
    "ON": (1, "D"),
    "TN": (2, "D"),
    "SN": (3, "D"),
}


class TenorUnit(str, Enum):
    DAY = "D"
    WEEK = "W"
    MONTH = "M"
    YEAR = "Y"


class Tenor:
    """Relative time period such as 1D, 2W, 3M, 1Y."""

    __slots__ = ("quantity", "unit")

    def __init__(self, quantity: int, unit: TenorUnit) -> None:
        if quantity < 0:
            raise ValueError("Tenor quantity must be non-negative")
        self.quantity = quantity
        self.unit = unit

    @classmethod
    def from_string(cls, s: str) -> "Tenor":
        s = s.strip().upper()
        if s in _SHORTHANDS:
            qty, u = _SHORTHANDS[s]
            return cls(qty, TenorUnit(u))
        m = _TENOR_RE.match(s)
        if not m:
            raise ValueError(f"Cannot parse tenor '{s}'. Expected: 1D, 2W, 3M, 1Y, ON, TN")
        return cls(int(m.group(1)), TenorUnit(m.group(2)))

    def add_to(self, d: date) -> date:
        """Return d + this tenor.  Month-end is handled correctly by dateutil."""
        match self.unit:
            case TenorUnit.DAY:
                return d + timedelta(days=self.quantity)
            case TenorUnit.WEEK:
                return d + timedelta(weeks=self.quantity)
            case TenorUnit.MONTH:
                return d + relativedelta(months=self.quantity)
            case TenorUnit.YEAR:
                return d + relativedelta(years=self.quantity)

    def in_years(self) -> float:
        """Approximate year fraction — useful for rough ordering of tenors."""
        match self.unit:
            case TenorUnit.DAY:   return self.quantity / 365.0
            case TenorUnit.WEEK:  return self.quantity / 52.0
            case TenorUnit.MONTH: return self.quantity / 12.0
            case TenorUnit.YEAR:  return float(self.quantity)

    def __repr__(self) -> str:
        return f"Tenor({self.quantity}{self.unit.value})"

    def __str__(self) -> str:
        return f"{self.quantity}{self.unit.value}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Tenor):
            return NotImplemented
        return self.quantity == other.quantity and self.unit == other.unit

    def __lt__(self, other: "Tenor") -> bool:
        return self.in_years() < other.in_years()

    def __hash__(self) -> int:
        return hash((self.quantity, self.unit))
