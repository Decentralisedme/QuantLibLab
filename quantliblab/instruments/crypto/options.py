"""
Crypto options instrument definitions.

Deribit lists European-style, cash-settled options on BTC and ETH.
Each contract gives the right to buy (call) or sell (put) 1 BTC (or 1 ETH)
at the strike price, settled in the base currency at expiry.

Instrument name format:
  BTC-28MAR25-80000-C   — BTC call, expiry 28 Mar 2025, strike 80,000 USD
  ETH-27JUN25-2500-P    — ETH put,  expiry 27 Jun 2025, strike 2,500 USD

Greeks are provided directly by Deribit (model: Black-Scholes on log-normal IV).
All prices and strikes in USD.  IV in decimal (0.60 = 60%).
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Optional


class OptionKind(str, Enum):
    CALL = "call"
    PUT = "put"


@dataclass
class CryptoOption:
    """
    A single crypto options contract snapshot.

    Parameters
    ----------
    instrument_name : Deribit instrument name, e.g. "BTC-28MAR25-80000-C"
    expiry_date : Calendar date of expiry
    strike : Strike price in USD
    kind : OptionKind.CALL or OptionKind.PUT
    mark_price : Deribit model price in USD
    mark_iv : Implied volatility used for the mark price (decimal, e.g. 0.60)
    underlying : "BTC" or "ETH" — set automatically by BTCOption/ETHOption
    bid_iv : Implied vol for the best bid
    ask_iv : Implied vol for the best ask
    bid : Best bid in USD
    ask : Best ask in USD
    volume_usd : 24h notional volume in USD
    open_interest : Open interest in contracts
    delta : Option delta (−1 to +1 for puts; 0 to +1 for calls)
    gamma : Option gamma
    vega : Option vega (per 1% move in IV)
    theta : Option theta (per calendar day)
    rho : Option rho
    timestamp : When this snapshot was taken (UTC)
    """

    instrument_name: str
    expiry_date: date
    strike: float
    kind: OptionKind
    mark_price: float
    mark_iv: float                      # decimal
    underlying: str = ""               # overridden by BTCOption / ETHOption
    bid_iv: Optional[float] = None
    ask_iv: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    volume_usd: float = 0.0
    open_interest: float = 0.0
    delta: Optional[float] = None
    gamma: Optional[float] = None
    vega: Optional[float] = None
    theta: Optional[float] = None
    rho: Optional[float] = None
    timestamp: Optional[datetime] = None

    @property
    def mid_iv(self) -> Optional[float]:
        """Mid implied volatility. None if bid/ask IV missing."""
        if self.bid_iv is not None and self.ask_iv is not None:
            return (self.bid_iv + self.ask_iv) / 2.0
        return None

    @property
    def mid(self) -> Optional[float]:
        """Mid price in USD. None if bid/ask missing."""
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2.0
        return None

    @property
    def moneyness(self) -> Optional[float]:
        """
        Simple moneyness: strike / mark_price.
        ATM = 1.0, OTM call > 1.0, OTM put < 1.0.
        """
        if self.mark_price and self.mark_price > 0:
            return self.strike / self.mark_price
        return None


@dataclass
class BTCOption(CryptoOption):
    """BTC option contract. Underlying fixed to 'BTC'."""

    def __post_init__(self) -> None:
        self.underlying = "BTC"


@dataclass
class ETHOption(CryptoOption):
    """ETH option contract. Underlying fixed to 'ETH'."""

    def __post_init__(self) -> None:
        self.underlying = "ETH"
