#!/usr/bin/env python3
"""
capture_golden_snapshot.py — freeze one full day of market data.

    python scripts/capture_golden_snapshot.py            # capture today
    python scripts/capture_golden_snapshot.py 2026-07-25 # label explicitly

Writes tests/fixtures/golden/<YYYY-MM-DD>/
    raw/          exact API payloads (Deribit, FRED, Polymarket Gamma;
                  yfinance outputs stored as "raw-ish" JSON — no clean
                  request seam exists for it)
    normalized/   CSVs derived from the raw payloads: fitted SVI slices,
                  futures strips, O/N rates, FX spots, parsed + priced
                  Polymarket markets
    manifest.json capture metadata + per-feed success/failure

Each feed is captured independently — one dead API does not abort the
snapshot; failures are recorded in the manifest.

The snapshot then powers tests/integration/test_golden_pipeline.py
(full offline pipeline run) and dashboard/curve_viewer.py.
"""
from __future__ import annotations

import csv
import json
import sys
import traceback
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT))

from quantliblab.data.golden import GOLDEN_ROOT, record, save_json   # noqa: E402

FRED_LOOKBACK_DAYS = 30      # window of O/N history worth freezing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def _step(manifest: dict, name: str, fn) -> None:
    """Run one capture step; record outcome instead of crashing."""
    print(f"── {name} ...")
    try:
        detail = fn()
        manifest["feeds"][name] = {"ok": True, "detail": detail}
        print(f"   ok: {detail}")
    except Exception as e:                       # noqa: BLE001 — capture all
        manifest["feeds"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        print(f"   FAILED: {e}")
        traceback.print_exc(limit=2)


# ---------------------------------------------------------------------------
# Capture steps
# ---------------------------------------------------------------------------

def capture_deribit(raw: Path, norm: Path, surfaces: dict) -> str:
    """Calibrate BTC + ETH under the recorder; persist SVI slices,
    diagnostics, and the futures strip per currency."""
    from quantliblab.data.loaders.deribit_loader import fetch_futures
    from quantliblab.harness.deribit_surface import calibrate_currency

    counts = []
    with record(raw):
        for ccy in ("BTC", "ETH"):
            fs = calibrate_currency(ccy)
            surfaces[ccy] = fs

            _write_csv(
                norm / f"svi_{ccy.lower()}.csv",
                ["expiry", "T", "F", "a", "b", "rho", "m", "s",
                 "rmse_volpts", "n_quotes", "used", "reason"],
                [{
                    "expiry": d.expiry.isoformat(), "T": f"{d.T:.6f}",
                    "F": f"{d.F:.2f}",
                    **({"a": f"{s.params.a:.8f}", "b": f"{s.params.b:.8f}",
                        "rho": f"{s.params.rho:.6f}", "m": f"{s.params.m:.6f}",
                        "s": f"{s.params.s:.6f}"}
                       if s is not None else
                       {"a": "", "b": "", "rho": "", "m": "", "s": ""}),
                    "rmse_volpts": (f"{d.rmse_volpts:.3f}"
                                    if d.rmse_volpts == d.rmse_volpts else ""),
                    "n_quotes": d.n_quotes, "used": d.used, "reason": d.reason,
                } for d, s in _pair_diags_slices(fs)],
            )

            futs = fetch_futures(ccy)           # served from recorder cache
            _write_csv(
                norm / f"deribit_{ccy.lower()}_futures.csv",
                ["expiry", "mark_price"],
                [{"expiry": f.expiry_date.isoformat(),
                  "mark_price": f.mark_price} for f in futs],
            )
            used = sum(1 for d in fs.diagnostics if d.used)
            counts.append(f"{ccy}: {used}/{len(fs.diagnostics)} slices, "
                          f"index={fs.index_price:.0f}")
    return "; ".join(counts)


def _pair_diags_slices(fs):
    """Yield (diagnostic, fitted_slice_or_None) matched by expiry T."""
    by_T = ({round(s.T, 9): s for s in fs.surface.slices}
            if fs.surface is not None else {})
    for d in fs.diagnostics:
        yield d, by_T.get(round(d.T, 9))


def capture_fred(raw: Path, norm: Path) -> str:
    from quantliblab.data.loaders.fred_loader import (
        fetch_estr_on, fetch_sofr_on, fetch_sonia_on,
    )
    from quantliblab.data.loaders.nyfed_loader import fetch_sofr_averages

    end = date.today()
    start = end - timedelta(days=FRED_LOOKBACK_DAYS)
    with record(raw):
        on_rows: dict[str, dict] = {}
        for name, fn in (("sofr", fetch_sofr_on), ("sonia", fetch_sonia_on),
                         ("estr", fetch_estr_on)):
            for d, row in fn(start, end):
                on_rows.setdefault(d.isoformat(), {"date": d.isoformat()})[name] = row["ON"]
        avg = fetch_sofr_averages(start, end)

    _write_csv(norm / "rates_on.csv", ["date", "sofr", "sonia", "estr"],
               [on_rows[k] for k in sorted(on_rows)])
    _write_csv(norm / "sofr_averages.csv", ["date", "30D", "90D", "180D"],
               [{"date": d.isoformat(), **row} for d, row in avg])
    return f"{len(on_rows)} O/N rows, {len(avg)} average rows"


def capture_gamma(raw: Path, norm: Path, surfaces: dict) -> str:
    """Raw Gamma pages + parsed markets + fair values vs today's surfaces."""
    from quantliblab.harness.polymarket import (
        extract_crypto_binaries, fetch_gamma_markets, price_market,
    )
    markets = fetch_gamma_markets()
    save_json(raw / "gamma_markets.json", markets)

    parsed = extract_crypto_binaries(markets)
    _write_csv(norm / "polymarket_markets.csv",
               ["market_id", "asset", "type", "strike", "resolution",
                "yes_price", "liquidity", "question"],
               [{"market_id": p.market_id, "asset": p.asset,
                 "type": p.mtype.value, "strike": p.strike,
                 "resolution": p.resolution.isoformat(),
                 "yes_price": p.yes_price if p.yes_price is not None else "",
                 "liquidity": p.liquidity, "question": p.question[:120]}
                for p in parsed])

    priced = []
    for p in parsed:
        fs = surfaces.get(p.asset)
        if fs is None or fs.surface is None:
            continue
        try:
            row = price_market(p, fs)
        except Exception:                        # noqa: BLE001
            continue
        if row:
            priced.append(row)
    if priced:
        _write_csv(norm / "polymarket_priced.csv", list(priced[0].keys()), priced)
    return f"{len(markets)} raw, {len(parsed)} parsed, {len(priced)} priced"


def capture_sofr_futures(raw: Path, norm: Path) -> str:
    """yfinance strips — no request seam, so store outputs as raw-ish JSON."""
    from dataclasses import asdict
    from quantliblab.data.loaders.sofr_futures_loader import (
        fetch_sr1_strip, fetch_sr3_strip,
    )
    sr3 = fetch_sr3_strip()
    sr1 = fetch_sr1_strip()
    dump = {"source": "yfinance last_price (rawish — no request seam)",
            "sr3": [asdict(c) for c in sr3], "sr1": [asdict(c) for c in sr1]}
    (raw / "sofr_futures_yf.json").write_text(
        json.dumps(dump, default=str, indent=1))   # small — keep readable
    _write_csv(norm / "sofr_futures.csv",
               ["contract_type", "ticker", "ref_start", "ref_end",
                "price", "implied_rate"],
               [{"contract_type": c.contract_type, "ticker": c.ticker,
                 "ref_start": c.ref_start.isoformat(),
                 "ref_end": c.ref_end.isoformat(),
                 "price": c.price, "implied_rate": c.implied_rate}
                for c in sr3 + sr1])
    return f"{len(sr3)} SR3 + {len(sr1)} SR1 contracts"


def capture_fx(raw: Path, norm: Path) -> str:
    from quantliblab.data.loaders.fx_loader import fetch_all_fx
    end = date.today()
    data = fetch_all_fx(end - timedelta(days=7), end)
    (raw / "fx_spots_yf.json").write_text(
        json.dumps({p: [(d.isoformat(), row) for d, row in rows]
                    for p, rows in data.items()},
                   indent=1))
    latest = []
    for pair, rows in data.items():
        if rows:
            d, row = rows[-1]
            latest.append({"pair": pair, "date": d.isoformat(),
                           "spot": list(row.values())[0]})
    _write_csv(norm / "fx_spots.csv", ["pair", "date", "spot"], latest)
    return f"{len(latest)} pairs"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    label = sys.argv[1] if len(sys.argv) > 1 else date.today().isoformat()
    golden = GOLDEN_ROOT / label
    raw, norm = golden / "raw", golden / "normalized"
    raw.mkdir(parents=True, exist_ok=True)
    norm.mkdir(parents=True, exist_ok=True)
    print(f"Capturing golden snapshot -> {golden}")

    manifest: dict = {
        "label": label,
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "feeds": {},
    }
    surfaces: dict = {}

    _step(manifest, "deribit_surfaces",
          lambda: capture_deribit(raw, norm, surfaces))
    _step(manifest, "fred_rates", lambda: capture_fred(raw, norm))
    _step(manifest, "polymarket_gamma",
          lambda: capture_gamma(raw, norm, surfaces))
    _step(manifest, "sofr_futures", lambda: capture_sofr_futures(raw, norm))
    _step(manifest, "fx_spots", lambda: capture_fx(raw, norm))

    (golden / "manifest.json").write_text(json.dumps(manifest, indent=2))
    ok = sum(1 for f in manifest["feeds"].values() if f["ok"])
    print(f"Done: {ok}/{len(manifest['feeds'])} feeds captured. "
          f"Manifest: {golden / 'manifest.json'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
