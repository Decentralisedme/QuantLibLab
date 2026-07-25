"""Unit tests for golden snapshot record/replay — no network involved:
the live seams are replaced with fakes, then the recorder is layered on
top and the replay is checked against what was recorded."""
from __future__ import annotations

from datetime import date

import pytest

from quantliblab.data import golden
from quantliblab.data.loaders import deribit_loader, fred_loader, nyfed_loader


@pytest.fixture()
def fake_seams(monkeypatch):
    """Replace both live HTTP seams with counting fakes."""
    calls = {"deribit": 0, "fred": 0}

    def fake_get(endpoint, params):
        calls["deribit"] += 1
        return {"endpoint": endpoint, "params": params, "n": calls["deribit"]}

    def fake_fred(series_id, start, end):
        calls["fred"] += 1
        return [{"date": start.isoformat(), "value": "5.30"}]

    monkeypatch.setattr(deribit_loader, "_get", fake_get)
    monkeypatch.setattr(fred_loader, "_fetch_series", fake_fred)
    monkeypatch.setattr(nyfed_loader, "_fetch_series", fake_fred)
    return calls


class TestRecord:
    def test_records_and_caches_within_block(self, tmp_path, fake_seams):
        with golden.record(tmp_path):
            r1 = deribit_loader._get("get_index_price", {"index_name": "btc_usd"})
            r2 = deribit_loader._get("get_index_price", {"index_name": "btc_usd"})
        assert r1 == r2                         # 2nd call served from cache
        assert fake_seams["deribit"] == 1       # upstream hit exactly once
        key = golden.deribit_key("get_index_price", {"index_name": "btc_usd"})
        assert (tmp_path / (key + ".gz")).exists()   # payloads stored gzipped

    def test_seams_restored_after_block(self, tmp_path, fake_seams):
        inner = None
        with golden.record(tmp_path):
            inner = deribit_loader._get
        assert deribit_loader._get is not inner   # recorder removed
        deribit_loader._get("e", {"a": 1})
        assert fake_seams["deribit"] == 1         # back on the (fake) live seam

    def test_fred_recorded_for_both_loaders(self, tmp_path, fake_seams):
        d0, d1 = date(2026, 7, 1), date(2026, 7, 20)
        with golden.record(tmp_path):
            fred_loader._fetch_series("SOFR", d0, d1)
            nyfed_loader._fetch_series("SOFR", d0, d1)   # same cache entry
        assert fake_seams["fred"] == 1
        assert (tmp_path / (golden.fred_key("SOFR", d0, d1) + ".gz")).exists()


class TestReplay:
    def test_replay_serves_recorded_payloads_offline(self, tmp_path, fake_seams):
        with golden.record(tmp_path):
            live = deribit_loader._get("ticker", {"instrument_name": "BTC-PERPETUAL"})
        # kill the upstream entirely: replay must not need it
        deribit_loader._get = None                       # type: ignore[assignment]
        with golden.replay(tmp_path):
            replayed = deribit_loader._get("ticker",
                                           {"instrument_name": "BTC-PERPETUAL"})
        assert replayed == live

    def test_replay_raises_on_missing_payload(self, tmp_path, fake_seams):
        tmp_path.mkdir(exist_ok=True)
        with golden.replay(tmp_path):
            with pytest.raises(golden.GoldenMiss):
                deribit_loader._get("get_index_price", {"index_name": "eth_usd"})

    def test_replay_requires_existing_dir(self):
        with pytest.raises(golden.GoldenMiss):
            with golden.replay("/nonexistent/golden/raw"):
                pass


class TestCompression:
    def test_plain_json_payloads_still_replayable(self, tmp_path, fake_seams):
        """Snapshots captured before gzip support keep working."""
        import json
        key = golden.deribit_key("get_index_price", {"index_name": "btc_usd"})
        (tmp_path / key).write_text(json.dumps({"index_price": 1.0}))
        with golden.replay(tmp_path):
            r = deribit_loader._get("get_index_price", {"index_name": "btc_usd"})
        assert r == {"index_price": 1.0}

    def test_gzip_roundtrip_and_shrinkage(self, tmp_path):
        payload = [{"instrument_name": f"BTC-X-{i}", "mark_iv": 55.5,
                    "bid_price": 0.019, "ask_price": 0.021} for i in range(500)]
        import json
        raw_len = len(json.dumps(payload))
        written = golden.save_json(tmp_path / "book.json", payload)
        assert written.name.endswith(".json.gz")
        assert written.stat().st_size < raw_len / 5     # >5x smaller
        assert golden.load_json(tmp_path / "book.json") == payload


class TestDiscovery:
    def test_list_and_latest(self, tmp_path):
        assert golden.list_snapshots(tmp_path) == []
        assert golden.latest_snapshot(tmp_path) is None
        for label in ("2026-07-20", "2026-07-25"):
            (tmp_path / label / "raw").mkdir(parents=True)
        snaps = golden.list_snapshots(tmp_path)
        assert [p.name for p in snaps] == ["2026-07-20", "2026-07-25"]
        assert golden.latest_snapshot(tmp_path).name == "2026-07-25"
