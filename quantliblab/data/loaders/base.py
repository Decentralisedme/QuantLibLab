"""
Base loader interface.

All loaders return a list of (date, dict) records ready to pass
directly to store.write_many().
"""
from __future__ import annotations

from datetime import date
from typing import Protocol


class RateLoader(Protocol):
    def fetch(
        self,
        start: date,
        end: date,
    ) -> list[tuple[date, dict[str, float]]]:
        """Fetch rate data for the given date range."""
        ...
