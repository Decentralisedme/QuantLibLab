"""
Golden snapshot record / replay.

Captures the RAW payloads of every external request made through the
library's HTTP seams, so that:

  * the exact bytes the parsers consumed are preserved (parser
    regression tests stay valid even after dataclasses change), and
  * the full pipeline — loaders, SVI calibration, digital pricing —
    can be re-run offline, deterministically, in CI.

Seams patched
-------------
  deribit_loader._get(endpoint, params)      -> Deribit /public payloads
  fred_loader._fetch_series(id, start, end)  -> FRED observation lists
  nyfed_loader._fetch_series                 -> (same function, but bound
                                                by `from ... import` at
                                                module load, so patched
                                                separately)

yfinance (FX spots, SOFR futures) has no clean request seam; the capture
script stores its outputs as "raw-ish" JSON instead — see
scripts/capture_golden_snapshot.py.

Usage
-----
    from quantliblab.data.golden import record, replay

    with record(golden_dir / "raw"):
        calibrate_currency("BTC")     # hits the network once, records all

    with replay(golden_dir / "raw"):
        calibrate_currency("BTC")     # NO network — served from disk;
                                      # unknown requests raise GoldenMiss

Within a single `record()` block, repeated identical requests are served
from the cache, so e.g. calibrate_currency() and a later fetch_futures()
see byte-identical data.
"""
from __future__ import annotations

import gzip
import hashlib
import json
from contextlib import contextmanager
from datetime import date
from pathlib import Path

from quantliblab.data.loaders import deribit_loader, fred_loader, nyfed_loader


class GoldenMiss(RuntimeError):
    """A replayed run requested data that is not in the golden snapshot."""


# ---------------------------------------------------------------------------
# Cache keys — filename-safe, deterministic
# ---------------------------------------------------------------------------

def _safe(text: str) -> str:
    out = "".join(c if (c.isalnum() or c in "=-_.") else "_" for c in text)
    if len(out) > 140:  # keep filenames portable
        out = out[:120] + hashlib.md5(text.encode()).hexdigest()[:16]
    return out


def deribit_key(endpoint: str, params: dict) -> str:
    flat = "__".join(f"{k}={params[k]}" for k in sorted(params))
    return _safe(f"deribit__{endpoint}__{flat}") + ".json"


def fred_key(series_id: str, start: date, end: date) -> str:
    return _safe(f"fred__{series_id}__{start.isoformat()}__{end.isoformat()}") + ".json"


# ---------------------------------------------------------------------------
# Record / replay context managers
# ---------------------------------------------------------------------------

def _payload_path(raw_dir: Path, key: str) -> Path | None:
    """Existing payload file for a key — gzipped preferred, plain accepted."""
    for cand in (raw_dir / (key + ".gz"), raw_dir / key):
        if cand.exists():
            return cand
    return None


def _load(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as f:
            return json.load(f)
    return json.loads(path.read_text())


def save_json(path: Path, payload, compress: bool = True) -> Path:
    """Write JSON, gzipped by default (~8-10x smaller — keeps golden
    snapshots trivially committable). Returns the path actually written."""
    if compress:
        path = path.with_suffix(path.suffix + ".gz") \
            if path.suffix != ".gz" else path
        with gzip.open(path, "wt") as f:
            json.dump(payload, f, sort_keys=True)
        return path
    path.write_text(json.dumps(payload, indent=1, sort_keys=True))
    return path


def load_json(path: Path):
    """Read JSON written by save_json; accepts <name> or <name>.gz."""
    if not path.exists() and path.with_suffix(path.suffix + ".gz").exists():
        path = path.with_suffix(path.suffix + ".gz")
    return _load(path)


@contextmanager
def _patched(deribit_get, fred_fetch):
    """Swap the two seams in, restore on exit (also inside nyfed_loader,
    which binds _fetch_series by name at import time)."""
    saved = (deribit_loader._get, fred_loader._fetch_series,
             nyfed_loader._fetch_series)
    deribit_loader._get = deribit_get
    fred_loader._fetch_series = fred_fetch
    nyfed_loader._fetch_series = fred_fetch
    try:
        yield
    finally:
        (deribit_loader._get, fred_loader._fetch_series,
         nyfed_loader._fetch_series) = saved


@contextmanager
def record(raw_dir: Path | str):
    """Fetch live, saving every payload; identical repeat requests within
    the block (or from a previous partial capture) are served from disk."""
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    real_get, real_fred = deribit_loader._get, fred_loader._fetch_series

    def rec_get(endpoint: str, params: dict):
        hit = _payload_path(raw_dir, deribit_key(endpoint, params))
        if hit is not None:
            return _load(hit)
        result = real_get(endpoint, params)
        save_json(raw_dir / deribit_key(endpoint, params), result)
        return result

    def rec_fred(series_id: str, start: date, end: date):
        hit = _payload_path(raw_dir, fred_key(series_id, start, end))
        if hit is not None:
            return _load(hit)
        result = real_fred(series_id, start, end)
        save_json(raw_dir / fred_key(series_id, start, end), result)
        return result

    with _patched(rec_get, rec_fred):
        yield raw_dir


@contextmanager
def replay(raw_dir: Path | str):
    """Serve every request from the snapshot; NEVER touches the network.
    Raises GoldenMiss for anything not captured."""
    raw_dir = Path(raw_dir)
    if not raw_dir.is_dir():
        raise GoldenMiss(f"golden raw dir not found: {raw_dir}")

    def rep_get(endpoint: str, params: dict):
        hit = _payload_path(raw_dir, deribit_key(endpoint, params))
        if hit is None:
            raise GoldenMiss(f"not in snapshot: deribit {endpoint} {params}")
        return _load(hit)

    def rep_fred(series_id: str, start: date, end: date):
        hit = _payload_path(raw_dir, fred_key(series_id, start, end))
        if hit is None:
            raise GoldenMiss(f"not in snapshot: FRED {series_id} "
                             f"{start}..{end}")
        return _load(hit)

    with _patched(rep_get, rep_fred):
        yield raw_dir


# ---------------------------------------------------------------------------
# Snapshot discovery
# ---------------------------------------------------------------------------

GOLDEN_ROOT = Path(__file__).parents[2] / "tests" / "fixtures" / "golden"


def list_snapshots(root: Path | str = GOLDEN_ROOT) -> list[Path]:
    """Golden snapshot directories, oldest first (dir name = ISO date)."""
    root = Path(root)
    if not root.is_dir():
        return []
    return sorted(p for p in root.iterdir()
                  if p.is_dir() and (p / "raw").is_dir())


def latest_snapshot(root: Path | str = GOLDEN_ROOT) -> Path | None:
    snaps = list_snapshots(root)
    return snaps[-1] if snaps else None
