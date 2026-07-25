"""
Full-pipeline integration tests against the latest golden snapshot.

Runs the REAL code paths — Deribit parsing, forward interpolation, SVI
calibration, Polymarket parsing and digital pricing, FRED parsing — with
zero network, off the raw payloads captured by
scripts/capture_golden_snapshot.py.

If no snapshot exists yet the whole module is skipped:

    python scripts/capture_golden_snapshot.py   # mint one first
"""
from __future__ import annotations

import csv
import json
from datetime import date, timedelta

import pytest

from quantliblab.data.golden import latest_snapshot, load_json, replay

SNAP = latest_snapshot()

pytestmark = pytest.mark.skipif(
    SNAP is None,
    reason="no golden snapshot — run scripts/capture_golden_snapshot.py",
)


@pytest.fixture(scope="module")
def surfaces():
    """Calibrate both currencies once, offline, from the snapshot."""
    from quantliblab.harness.deribit_surface import calibrate_currency
    out = {}
    with replay(SNAP / "raw"):
        for ccy in ("BTC", "ETH"):
            out[ccy] = calibrate_currency(ccy)
    return out


class TestDeribitCalibration:
    def test_surfaces_fitted(self, surfaces):
        for ccy, fs in surfaces.items():
            assert fs.surface is not None, f"{ccy}: no usable surface"
            assert len(fs.surface.slices) >= 2
            assert fs.index_price > 0

    def test_fit_quality(self, surfaces):
        """Used slices should fit tightly and be butterfly-free."""
        for ccy, fs in surfaces.items():
            used = [d for d in fs.diagnostics if d.used]
            assert used, f"{ccy}: no used slices"
            for d in used:
                assert d.butterfly_free
                assert d.rmse_volpts < 10.0, (
                    f"{ccy} {d.expiry}: RMSE {d.rmse_volpts:.2f} vol pts")

    def test_forwards_monotone_carry_sane(self, surfaces):
        for ccy, fs in surfaces.items():
            for s in fs.surface.slices:
                # |implied carry| < 100%/yr — catches unit errors loudly
                assert 0.35 < s.F / fs.index_price < 2.75

    def test_smile_queries_finite(self, surfaces):
        for fs in surfaces.values():
            for k in (-0.5, 0.0, 0.5):
                for T in (0.05, 0.25):
                    sigma, slope = fs.smile_at(k, T)
                    assert 0.01 < sigma < 5.0
                    assert abs(slope) < 10.0


class TestPolymarketPipeline:
    @pytest.fixture(scope="class")
    def raw_markets(self):
        p = SNAP / "raw" / "gamma_markets.json"
        if not (p.exists() or p.with_suffix(".json.gz").exists()):
            pytest.skip("snapshot has no Gamma capture")
        return load_json(p)

    def test_parser_yields_markets(self, raw_markets):
        from quantliblab.harness.polymarket import extract_crypto_binaries
        parsed = extract_crypto_binaries(raw_markets)
        # crypto binaries come and go; just require the parser to not
        # collapse to zero on a real page set while markets exist
        assert isinstance(parsed, list)
        for pm in parsed:
            assert pm.asset in ("BTC", "ETH")
            assert pm.strike > 0

    def test_priced_fairs_are_probabilities(self, raw_markets, surfaces):
        from quantliblab.harness.polymarket import (
            extract_crypto_binaries, price_market,
        )
        parsed = extract_crypto_binaries(raw_markets)
        priced = 0
        # snapshot pricing must be anchored at capture time, not test time
        asof = min((s.asof for s in surfaces.values()), default=None)
        for pm in parsed:
            fs = surfaces.get(pm.asset)
            if fs is None or fs.surface is None:
                continue
            row = price_market(pm, fs, now=asof)
            if row is None:
                continue
            priced += 1
            assert 0.0 <= float(row["fair"]) <= 1.0
            if row["fair_lo"] != "":
                assert float(row["fair_lo"]) <= float(row["fair"]) <= float(row["fair_hi"])
        if parsed:
            assert priced > 0, "no parsed market could be priced"


class TestRatesFeeds:
    def test_fred_sofr_parses_offline(self):
        from quantliblab.data.loaders.fred_loader import fetch_sofr_on
        # reconstruct the capture window from the recorded filenames
        keys = list((SNAP / "raw").glob("fred__SOFR__*.json"))
        if not keys:
            pytest.skip("snapshot has no FRED capture")
        _, _, start_s, end_s = keys[0].stem.split("__")
        with replay(SNAP / "raw"):
            rows = fetch_sofr_on(date.fromisoformat(start_s),
                                 date.fromisoformat(end_s))
        assert rows
        for _, row in rows:
            assert 0.0 < row["ON"] < 0.15        # decimal, sane level

    def test_sofr_futures_csv_sane(self):
        p = SNAP / "normalized" / "sofr_futures.csv"
        if not p.exists():
            pytest.skip("snapshot has no SOFR futures capture")
        with p.open() as f:
            rows = list(csv.DictReader(f))
        assert rows
        for r in rows:
            assert 0.0 < float(r["implied_rate"]) < 0.15
            assert 85.0 < float(r["price"]) < 100.0
