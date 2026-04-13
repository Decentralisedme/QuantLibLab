"""
Unit tests for instruments/crypto — futures, options, perpetuals.

Tests cover:
- Field assignment and __post_init__ underlying override
- Derived properties: mid, spread, basis, basis_bps, annualised_funding
- OptionKind enum values
- moneyness calculation
- None-safe properties when bid/ask absent
"""
from datetime import date, datetime

import pytest

from quantliblab.instruments.crypto import (
    BTCFuture,
    ETHFuture,
    BTCOption,
    ETHOption,
    BTCPerpetual,
    ETHPerpetual,
    OptionKind,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _btc_future(**kwargs) -> BTCFuture:
    defaults = dict(
        instrument_name="BTC-28MAR25",
        expiry_date=date(2025, 3, 28),
        mark_price=82_000.0,
        last_price=81_950.0,
        bid=81_980.0,
        ask=82_020.0,
        volume_usd=500_000_000.0,
        open_interest=12_000.0,
    )
    defaults.update(kwargs)
    return BTCFuture(**defaults)


def _btc_option(**kwargs) -> BTCOption:
    defaults = dict(
        instrument_name="BTC-28MAR25-80000-C",
        expiry_date=date(2025, 3, 28),
        strike=80_000.0,
        kind=OptionKind.CALL,
        mark_price=82_000.0,
        mark_iv=0.60,
        bid_iv=0.58,
        ask_iv=0.62,
        bid=3_100.0,
        ask=3_300.0,
        volume_usd=10_000_000.0,
        open_interest=500.0,
        delta=0.55,
        gamma=0.00002,
        vega=150.0,
        theta=-45.0,
    )
    defaults.update(kwargs)
    return BTCOption(**defaults)


def _btc_perp(**kwargs) -> BTCPerpetual:
    defaults = dict(
        instrument_name="BTC-PERPETUAL",
        mark_price=82_050.0,
        index_price=82_000.0,
        last_price=82_040.0,
        bid=82_030.0,
        ask=82_070.0,
        volume_usd=2_000_000_000.0,
        open_interest=50_000.0,
        current_funding=0.0001,   # 0.01% per 8h
        funding_8h=0.00012,
    )
    defaults.update(kwargs)
    return BTCPerpetual(**defaults)


# ---------------------------------------------------------------------------
# CryptoFuture / BTCFuture / ETHFuture
# ---------------------------------------------------------------------------

class TestCryptoFuture:
    def test_btc_underlying_set(self):
        f = _btc_future()
        assert f.underlying == "BTC"

    def test_eth_underlying_set(self):
        f = ETHFuture(
            instrument_name="ETH-28MAR25",
            expiry_date=date(2025, 3, 28),
            mark_price=2_100.0,
            last_price=2_095.0,
            bid=2_098.0,
            ask=2_102.0,
            volume_usd=50_000_000.0,
            open_interest=8_000.0,
        )
        assert f.underlying == "ETH"

    def test_mid_price(self):
        f = _btc_future(bid=81_980.0, ask=82_020.0)
        assert f.mid == pytest.approx(82_000.0)

    def test_spread(self):
        f = _btc_future(bid=81_980.0, ask=82_020.0)
        assert f.spread == pytest.approx(40.0)

    def test_mid_none_when_bid_missing(self):
        f = _btc_future(bid=None, ask=82_020.0)
        assert f.mid is None

    def test_mid_none_when_ask_missing(self):
        f = _btc_future(bid=81_980.0, ask=None)
        assert f.mid is None

    def test_basis_stored(self):
        f = _btc_future(basis=50.0)
        assert f.basis == 50.0

    def test_basis_defaults_none(self):
        f = _btc_future()
        assert f.basis is None

    def test_timestamp_optional(self):
        ts = datetime(2025, 3, 24, 12, 0, 0)
        f = _btc_future(timestamp=ts)
        assert f.timestamp == ts


# ---------------------------------------------------------------------------
# CryptoOption / BTCOption / ETHOption
# ---------------------------------------------------------------------------

class TestCryptoOption:
    def test_btc_underlying(self):
        o = _btc_option()
        assert o.underlying == "BTC"

    def test_eth_underlying(self):
        o = ETHOption(
            instrument_name="ETH-28MAR25-2500-P",
            expiry_date=date(2025, 3, 28),
            strike=2_500.0,
            kind=OptionKind.PUT,
            mark_price=2_100.0,
            mark_iv=0.70,
        )
        assert o.underlying == "ETH"

    def test_option_kind_call(self):
        o = _btc_option(kind=OptionKind.CALL)
        assert o.kind == OptionKind.CALL
        assert o.kind.value == "call"

    def test_option_kind_put(self):
        o = _btc_option(kind=OptionKind.PUT)
        assert o.kind == OptionKind.PUT
        assert o.kind.value == "put"

    def test_mid_iv(self):
        o = _btc_option(bid_iv=0.58, ask_iv=0.62)
        assert o.mid_iv == pytest.approx(0.60)

    def test_mid_iv_none_when_missing(self):
        o = _btc_option(bid_iv=None, ask_iv=0.62)
        assert o.mid_iv is None

    def test_mid_price(self):
        o = _btc_option(bid=3_100.0, ask=3_300.0)
        assert o.mid == pytest.approx(3_200.0)

    def test_moneyness_otm_call(self):
        # Strike 80k, mark 82k → moneyness < 1 (OTM call has strike < spot, not > spot)
        # moneyness = strike / mark_price = 80000 / 82000 ≈ 0.976
        o = _btc_option(strike=80_000.0, mark_price=82_000.0)
        assert o.moneyness == pytest.approx(80_000.0 / 82_000.0)

    def test_moneyness_atm(self):
        o = _btc_option(strike=82_000.0, mark_price=82_000.0)
        assert o.moneyness == pytest.approx(1.0)

    def test_greeks_stored(self):
        o = _btc_option(delta=0.55, gamma=0.00002, vega=150.0, theta=-45.0)
        assert o.delta == pytest.approx(0.55)
        assert o.gamma == pytest.approx(0.00002)
        assert o.vega == pytest.approx(150.0)
        assert o.theta == pytest.approx(-45.0)

    def test_greeks_optional(self):
        o = BTCOption(
            instrument_name="BTC-28MAR25-80000-C",
            expiry_date=date(2025, 3, 28),
            strike=80_000.0,
            kind=OptionKind.CALL,
            mark_price=82_000.0,
            mark_iv=0.60,
        )
        assert o.delta is None
        assert o.gamma is None


# ---------------------------------------------------------------------------
# Perpetual / BTCPerpetual / ETHPerpetual
# ---------------------------------------------------------------------------

class TestPerpetual:
    def test_btc_underlying(self):
        p = _btc_perp()
        assert p.underlying == "BTC"

    def test_eth_underlying(self):
        p = ETHPerpetual(
            instrument_name="ETH-PERPETUAL",
            mark_price=2_105.0,
            index_price=2_100.0,
            last_price=2_103.0,
            bid=2_102.0,
            ask=2_108.0,
            volume_usd=500_000_000.0,
            open_interest=20_000.0,
            current_funding=0.00005,
        )
        assert p.underlying == "ETH"

    def test_basis(self):
        p = _btc_perp(mark_price=82_050.0, index_price=82_000.0)
        assert p.basis == pytest.approx(50.0)

    def test_basis_bps(self):
        # 50 / 82000 * 10000 ≈ 6.098 bps
        p = _btc_perp(mark_price=82_050.0, index_price=82_000.0)
        assert p.basis_bps == pytest.approx(50.0 / 82_000.0 * 10_000, rel=1e-4)

    def test_annualised_funding(self):
        # 0.0001 * 3 * 365 = 0.1095 = 10.95% annualised
        p = _btc_perp(current_funding=0.0001)
        assert p.annualised_funding == pytest.approx(0.0001 * 3 * 365)

    def test_mid_price(self):
        p = _btc_perp(bid=82_030.0, ask=82_070.0)
        assert p.mid == pytest.approx(82_050.0)

    def test_mid_none_when_missing(self):
        p = _btc_perp(bid=None, ask=82_070.0)
        assert p.mid is None

    def test_negative_funding(self):
        """Shorts pay longs when mark < index."""
        p = _btc_perp(current_funding=-0.00005)
        assert p.annualised_funding < 0

    def test_negative_basis(self):
        """Mark below index — backwardation."""
        p = _btc_perp(mark_price=81_950.0, index_price=82_000.0)
        assert p.basis == pytest.approx(-50.0)
        assert p.basis_bps < 0
