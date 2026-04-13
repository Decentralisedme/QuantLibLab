"""
Deribit data → CSV store bridge.

Fetches live snapshots from the Deribit loader and writes them to the
raw CSV store.  Keeps the same pattern as fred_loader → csv_store.

Datasets written:
  crypto/btc_index.csv   — BTC composite spot index (daily)
  crypto/eth_index.csv   — ETH composite spot index (daily)
  crypto/btc_perp.csv    — BTC perpetual snapshot (daily)
  crypto/eth_perp.csv    — ETH perpetual snapshot (daily)

Futures and options are not stored here — their schema changes as
contracts expire.  Those will be handled in pricing/ when the exact
format needed is clear.
"""
from __future__ import annotations

from datetime import date

from quantliblab.data.loaders.deribit_loader import fetch_index_price, fetch_perpetual
from quantliblab.data.store import write


def save_index(underlying: str, as_of: date) -> None:
    """
    Fetch and save the BTC or ETH composite index price for a given date.

    Parameters
    ----------
    underlying : "BTC" or "ETH"
    as_of      : the date to label this snapshot (typically today or yesterday)
    """
    price = fetch_index_price(underlying)
    write(
        asset_class="crypto",
        dataset=f"{underlying.lower()}_index",
        date_=as_of,
        data={"index_price": round(price, 2)},
    )


def save_perp(underlying: str, as_of: date) -> None:
    """
    Fetch and save the BTC or ETH perpetual swap snapshot for a given date.

    Columns saved:
      mark_price     — Deribit mark price in USD
      index_price    — Composite spot index in USD
      basis_usd      — mark_price - index_price
      current_funding — Current 8h funding rate (decimal)
      funding_8h     — Next 8h funding rate (decimal); None stored as ""
      open_interest  — Open interest in contracts

    Parameters
    ----------
    underlying : "BTC" or "ETH"
    as_of      : the date to label this snapshot
    """
    perp = fetch_perpetual(underlying)
    write(
        asset_class="crypto",
        dataset=f"{underlying.lower()}_perp",
        date_=as_of,
        data={
            "mark_price":      round(perp.mark_price, 2),
            "index_price":     round(perp.index_price, 2),
            "basis_usd":       round(perp.basis, 4),
            "current_funding": round(perp.current_funding, 8),
            "funding_8h":      round(perp.funding_8h, 8) if perp.funding_8h is not None else "",
            "open_interest":   round(perp.open_interest, 0),
        },
    )
