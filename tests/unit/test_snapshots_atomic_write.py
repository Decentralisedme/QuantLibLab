"""
Unit tests for run_calibration_harness.py's snapshots.csv write path.

snapshots.csv is load-bearing state, not just output — step_price rebuilds
known_pit_events from it on every run. These tests cover the two
robustness properties that matters for: a duplicate PIT observation must
be caught loudly, and a crash mid-write must never corrupt the file or
silently produce that duplicate.

scripts/ isn't a package, so the script is loaded by file path.
"""
from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_calibration_harness.py"


def _load_harness_module():
    spec = importlib.util.spec_from_file_location("run_calibration_harness", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_calibration_harness"] = mod
    spec.loader.exec_module(mod)
    return mod


harness = _load_harness_module()


def _row(event_ticker, asof, is_pit, venue="kalshi", market_id=None):
    row = {f: "" for f in harness.SNAP_FIELDS}
    row.update({
        "asof": asof, "venue": venue, "event_ticker": event_ticker,
        "market_id": market_id or f"{event_ticker}-T1000",
        "is_pit_observation": is_pit,
    })
    return row


@pytest.fixture
def snap_path(tmp_path, monkeypatch):
    p = tmp_path / "snapshots.csv"
    monkeypatch.setattr(harness, "SNAPSHOTS", p)
    return p


# ---------------------------------------------------------------------------
# Uniqueness guard
# ---------------------------------------------------------------------------

class TestPitUniquenessGuard:
    def test_passes_on_clean_history(self):
        rows = [
            _row("EVT-A", "2026-09-19T00:00:00+00:00", "True"),
            _row("EVT-A", "2026-09-20T00:00:00+00:00", "False"),
            _row("EVT-B", "2026-09-19T00:00:00+00:00", "True"),
        ]
        harness._assert_pit_uniqueness(rows)   # must not raise

    def test_ignores_polymarket_rows(self):
        rows = [
            _row("EVT-A", "2026-09-19T00:00:00+00:00", "True", venue="polymarket"),
            _row("EVT-A", "2026-09-20T00:00:00+00:00", "True", venue="polymarket"),
        ]
        harness._assert_pit_uniqueness(rows)   # not kalshi — never checked

    def test_raises_on_duplicate_pit_observation(self):
        rows = [
            _row("EVT-A", "2026-09-19T00:00:00+00:00", "True"),
            _row("EVT-A", "2026-09-26T00:00:00+00:00", "True"),  # same expiry, later poll
        ]
        with pytest.raises(RuntimeError, match="duplicate PIT observation"):
            harness._assert_pit_uniqueness(rows)

    def test_many_strikes_same_poll_is_not_a_duplicate(self):
        """One ladder = many rows, all True, all same asof — that's one
        PIT observation, not many."""
        asof = "2026-09-19T00:00:00+00:00"
        rows = [_row("EVT-A", asof, "True", market_id=f"EVT-A-T{k}") for k in range(50)]
        harness._assert_pit_uniqueness(rows)   # must not raise


# ---------------------------------------------------------------------------
# Atomic write
# ---------------------------------------------------------------------------

class TestAtomicWrite:
    def test_writes_header_and_rows(self, snap_path):
        rows = [_row("EVT-A", "2026-09-19T00:00:00+00:00", "True")]
        harness._write_snapshots_atomic(rows)
        with snap_path.open() as f:
            got = list(csv.DictReader(f))
        assert len(got) == 1
        assert got[0]["event_ticker"] == "EVT-A"

    def test_appends_to_existing_file(self, snap_path):
        harness._write_snapshots_atomic(
            [_row("EVT-A", "2026-09-19T00:00:00+00:00", "True")])
        harness._write_snapshots_atomic(
            [_row("EVT-A", "2026-09-20T00:00:00+00:00", "False")])
        with snap_path.open() as f:
            got = list(csv.DictReader(f))
        assert len(got) == 2

    def test_no_temp_file_left_behind_on_success(self, snap_path):
        harness._write_snapshots_atomic(
            [_row("EVT-A", "2026-09-19T00:00:00+00:00", "True")])
        leftovers = list(snap_path.parent.glob(".snapshots.*.tmp"))
        assert leftovers == []

    def test_duplicate_pit_observation_raises_after_write(self, snap_path, monkeypatch):
        """If the caller mis-tracks known_pit_events and stamps True twice
        for the same expiry, the write still lands (atomicity isn't a
        substitute for the logic check) but must fail loudly right after."""
        harness._write_snapshots_atomic(
            [_row("EVT-A", "2026-09-19T00:00:00+00:00", "True")])
        with pytest.raises(RuntimeError, match="duplicate PIT observation"):
            harness._write_snapshots_atomic(
                [_row("EVT-A", "2026-09-26T00:00:00+00:00", "True")])

    def test_crash_mid_write_leaves_original_file_untouched(self, snap_path, monkeypatch):
        """A crash while building the new file (before os.replace) must
        never corrupt or partially overwrite the existing snapshots.csv —
        and, since nothing new lands, the crashed expiry's is_pit_observation
        is simply never written, so the NEXT run correctly (and safely)
        treats it as still-unseen rather than producing a duplicate True."""
        harness._write_snapshots_atomic(
            [_row("EVT-A", "2026-09-19T00:00:00+00:00", "True")])
        before = snap_path.read_bytes()

        def boom(*a, **kw):
            raise OSError("simulated crash mid-write")

        monkeypatch.setattr(harness.csv.DictWriter, "writerows", boom)
        with pytest.raises(OSError, match="simulated crash"):
            harness._write_snapshots_atomic(
                [_row("EVT-B", "2026-09-20T00:00:00+00:00", "True")])

        assert snap_path.read_bytes() == before, "original file must be untouched by a failed write"
        assert list(snap_path.parent.glob(".snapshots.*.tmp")) == [], \
            "temp file must be cleaned up, not left half-written"

    def test_crash_mid_write_then_retry_does_not_duplicate(self, snap_path):
        """End-to-end version of the scenario in the crash-safety request:
        pricing computes a new expiry's rows, the write dies, the harness
        restarts and reprices the same (still-unqualified-as-seen) expiry —
        the eventual successful write must carry exactly one True for it.

        Patched and restored manually (not via the monkeypatch fixture)
        because monkeypatch.undo() would also revert this test's own
        snap_path -> tmp_path redirection, not just the crash simulation."""
        orig_writerows = harness.csv.DictWriter.writerows
        harness.csv.DictWriter.writerows = lambda *a, **kw: (_ for _ in ()).throw(
            OSError("simulated crash mid-write"))
        try:
            with pytest.raises(OSError):
                harness._write_snapshots_atomic(
                    [_row("EVT-A", "2026-09-19T00:00:00+00:00", "True")])
        finally:
            harness.csv.DictWriter.writerows = orig_writerows
        assert not snap_path.exists()

        known = {r["event_ticker"] for r in harness._read_csv(snap_path)
                 if r.get("venue") == "kalshi"}
        assert known == set(), "nothing was persisted, so nothing is known yet"
        harness._write_snapshots_atomic(
            [_row("EVT-A", "2026-09-19T00:05:00+00:00", "True")])

        rows = harness._read_csv(snap_path)
        trues = [r for r in rows if r["event_ticker"] == "EVT-A"
                 and r["is_pit_observation"] == "True"]
        assert len(trues) == 1
