"""
Crypto perpetual swap instrument definition.

A perpetual swap is like a futures contract with no expiry.  Instead of
converging to spot at expiry, the price is anchored to the underlying index
via a periodic funding rate mechanism:

  - If mark > index  → longs pay shorts (positive funding rate)
  - If mark < index  → shorts pay longs (negative funding rate)

Deribit settles funding every hour.  The annualised funding rate is:
  annual_rate = 8h_rate × 3 × 365   (approx)

Instrument names on Deribit:
  BTC-PERPETUAL
  ETH-PERPETUAL
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class Perpetual:
    """
    Snapshot of a perpetual swap contract.

    Parameters
    ----------
    instrument_name : "BTC-PERPETUAL" or "ETH-PERPETUAL"
    mark_price : Deribit mark price in USD (fair value; used for P&L)
    index_price : Composite spot index in USD (multi-exchange weighted average)
    last_price : Last traded price in USD
    bid : Best bid in USD
    ask : Best ask in USD
    volume_usd : 24h notional trading volume in USD
    open_interest : Open interest in contracts (1 contract = 10 USD on Deribit BTC perp)
    current_funding : Current 8h funding rate (decimal, e.g. 0.0001 = 0.01%)
    underlying : "BTC" or "ETH" — set automatically by BTCPerpetual/ETHPerpetual
    funding_8h : Next scheduled 8h funding rate
    interest_rate : Interest component of funding (usually 0 on Deribit)
    timestamp : When this snapshot was taken (UTC)
    """

    instrument_name: str               # "BTC-PERPETUAL" or "ETH-PERPETUAL"
    mark_price: float
    index_price: float
    last_price: Optional[float]
    bid: Optional[float]
    ask: Optional[float]
    volume_usd: float
    open_interest: float
    current_funding: float             # 8h rate, decimal
    underlying: str = ""               # overridden by BTCPerpetual / ETHPerpetual
    funding_8h: Optional[float] = None
    interest_rate: float = 0.0
    timestamp: Optional[datetime] = None

    @property
    def basis(self) -> float:
        """mark_price - index_price in USD."""
        return self.mark_price - self.index_price

    @property
    def basis_bps(self) -> float:
        """Basis as basis points relative to index price."""
        if self.index_price > 0:
            return (self.basis / self.index_price) * 10_000
        return 0.0

    @property
    def annualised_funding(self) -> float:
        """
        Approximate annualised funding rate.
        8h rate × 3 payments/day × 365 days.
        """
        return self.current_funding * 3 * 365

    @property
    def mid(self) -> Optional[float]:
        """Mid-market price in USD."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2.0
        return None


@dataclass
class BTCPerpetual(Perpetual):
    """BTC perpetual swap. Underlying fixed to 'BTC'."""

    def __post_init__(self) -> None:
        self.underlying = "BTC"


@dataclass
class ETHPerpetual(Perpetual):
    """ETH perpetual swap. Underlying fixed to 'ETH'."""

    def __post_init__(self) -> None:
        self.underlying = "ETH"
