"""Unit tests for quantliblab.data.store.csv_store."""
import shutil
from datetime import date
from pathlib import Path

import pytest

from quantliblab.data.store import (
    write, write_many, read, read_latest, list_datasets, delete_row,
)

# ---------------------------------------------------------------------------
# Fixtures — use a temp raw directory so tests don't touch real data files
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_raw_root(tmp_path, monkeypatch):
    """Redirect all store I/O to a temporary directory."""
    import quantliblab.data.store.csv_store as mod
    fake_root = tmp_path / "raw"
    for ac in ["fx", "rates", "commodities", "crypto"]:
        (fake_root / ac).mkdir(parents=True)
    monkeypatch.setattr(mod, "_RAW_ROOT", fake_root)


# ---------------------------------------------------------------------------
# Write / read round-trip
# ---------------------------------------------------------------------------

class TestWriteRead:
    def test_single_row(self):
        write("rates", "sofr", date(2024, 1, 2), {"ON": "0.0530", "3M": "0.0545"})
        rows = read("rates", "sofr")
        assert len(rows) == 1
        assert rows[0]["date"] == "2024-01-02"
        assert rows[0]["ON"] == "0.0530"

    def test_multiple_rows_sorted(self):
        write("rates", "sofr", date(2024, 1, 3), {"ON": "0.0531"})
        write("rates", "sofr", date(2024, 1, 2), {"ON": "0.0530"})
        rows = read("rates", "sofr")
        assert rows[0]["date"] == "2024-01-02"
        assert rows[1]["date"] == "2024-01-03"

    def test_update_existing_date(self):
        write("rates", "sofr", date(2024, 1, 2), {"ON": "0.0530"})
        write("rates", "sofr", date(2024, 1, 2), {"ON": "0.0532"})  # correction
        rows = read("rates", "sofr")
        assert len(rows) == 1
        assert rows[0]["ON"] == "0.0532"

    def test_write_many(self):
        records = [
            (date(2024, 1, 2), {"spot": "1.0850"}),
            (date(2024, 1, 3), {"spot": "1.0860"}),
            (date(2024, 1, 4), {"spot": "1.0870"}),
        ]
        write_many("fx", "eurusd", records)
        rows = read("fx", "eurusd")
        assert len(rows) == 3


class TestReadFilters:
    def setup_method(self):
        records = [
            (date(2024, 1, d), {"rate": str(0.05 + d * 0.001)})
            for d in range(2, 8)
        ]
        write_many("rates", "sonia", records)

    def test_no_filter(self):
        assert len(read("rates", "sonia")) == 6

    def test_start_filter(self):
        rows = read("rates", "sonia", start=date(2024, 1, 5))
        assert all(r["date"] >= "2024-01-05" for r in rows)

    def test_end_filter(self):
        rows = read("rates", "sonia", end=date(2024, 1, 4))
        assert all(r["date"] <= "2024-01-04" for r in rows)

    def test_range_filter(self):
        rows = read("rates", "sonia", start=date(2024, 1, 3), end=date(2024, 1, 5))
        assert len(rows) == 3

    def test_missing_file_returns_empty(self):
        assert read("rates", "nonexistent") == []


class TestReadLatest:
    def test_returns_last_row(self):
        write("fx", "usdjpy", date(2024, 1, 2), {"spot": "145.0"})
        write("fx", "usdjpy", date(2024, 1, 3), {"spot": "146.0"})
        latest = read_latest("fx", "usdjpy")
        assert latest is not None
        assert latest["date"] == "2024-01-03"

    def test_missing_file_returns_none(self):
        assert read_latest("fx", "nonexistent") is None


class TestListDatasets:
    def test_lists_created_files(self):
        write("rates", "sofr", date(2024, 1, 2), {"ON": "0.053"})
        write("rates", "sonia", date(2024, 1, 2), {"ON": "0.052"})
        datasets = list_datasets("rates")
        assert "sofr" in datasets
        assert "sonia" in datasets

    def test_invalid_asset_class(self):
        with pytest.raises(ValueError):
            list_datasets("equities")


class TestDeleteRow:
    def test_deletes_existing(self):
        write("rates", "sofr", date(2024, 1, 2), {"ON": "0.053"})
        write("rates", "sofr", date(2024, 1, 3), {"ON": "0.054"})
        deleted = delete_row("rates", "sofr", date(2024, 1, 2))
        assert deleted is True
        rows = read("rates", "sofr")
        assert len(rows) == 1
        assert rows[0]["date"] == "2024-01-03"

    def test_returns_false_if_not_found(self):
        write("rates", "sofr", date(2024, 1, 2), {"ON": "0.053"})
        assert delete_row("rates", "sofr", date(2024, 1, 9)) is False


class TestValidation:
    def test_schema_mismatch_raises(self):
        write("rates", "sofr", date(2024, 1, 2), {"ON": "0.053"})
        with pytest.raises(ValueError, match="Schema mismatch"):
            write("rates", "sofr", date(2024, 1, 3), {"ON": "0.054", "3M": "0.056"})

    def test_invalid_asset_class_raises(self):
        with pytest.raises(ValueError, match="Unknown asset class"):
            write("equities", "spx", date(2024, 1, 2), {"price": "4700"})
