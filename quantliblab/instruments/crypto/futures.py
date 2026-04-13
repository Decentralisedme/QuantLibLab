"""
Crypto futures instrument definitions.

A Future is a standardised contract to buy/sell an asset at a fixed price
on a specified expiry date.  On Deribit, BTC and ETH futures are
cash-settled in the base currency (BTC or ETH).

Instrument name format on Deribit:
  BTC-28MAR25   — quarterly expiry
  ETH-27JUN25
  BTC-PERPETUAL — perpetual (see perpetuals.py)

All prices are in USD.  OI is in number of contracts (1 BTC or 1 ETH each).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional


@dataclass
class CryptoFuture:
    """
    A single crypto futures contract snapshot (one row from Deribit book summary).

    Parameters
    ----------
    instrument_name : Deribit instrument name, e.g. "BTC-28MAR25"
    expiry_date : Calendar date of contract expiry
    mark_price : Deribit mark price in USD (used for margin / P&L)
    last_price : Last traded price in USD (may be None if no trade yet today)
    bid : Best bid in USD
    ask : Best ask in USD
    volume_usd : 24h notional trading volume in USD
    open_interest : Open interest in number of contracts
    underlying : "BTC" or "ETH" — set automatically by BTCFuture/ETHFuture
    basis : mark_price - index_price (convenience field; can be None)
    timestamp : When this snapshot was taken (UTC)
    """

    instrument_name: str               # e.g. "BTC-28MAR25"
    expiry_date: date
    mark_price: float
    last_price: Optional[float]
    bid: Optional[float]
    ask: Optional[float]
    volume_usd: float
    open_interest: float               # contracts
    underlying: str = ""               # overridden by BTCFuture / ETHFuture
    basis: Optional[float] = None      # mark - index
    timestamp: Optional[datetime] = None

    @property
    def mid(self) -> Optional[float]:
        """Mid-market price from bid/ask. None if either is missing."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2.0
        return None

    @property
    def spread(self) -> Optional[float]:
        """Bid-ask spread in USD."""
        if self.bid is not None and self.ask is not None:
            return self.ask - self.bid
        return None


@dataclass
class BTCFuture(CryptoFuture):
    """BTC-denominated futures contract. Underlying fixed to 'BTC'."""

    def __post_init__(self) -> None:
        self.underlying = "BTC"


@dataclass
class ETHFuture(CryptoFuture):
    """ETH-denominated futures contract. Underlying fixed to 'ETH'."""

    def __post_init__(self) -> None:
        self.underlying = "ETH"
