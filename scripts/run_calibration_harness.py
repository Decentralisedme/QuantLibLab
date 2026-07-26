#!/usr/bin/env python3
"""
run_calibration_harness.py — nightly paper-trading loop.

    python scripts/run_calibration_harness.py            # live run
    python scripts/run_calibration_harness.py --selftest # offline logic test

Each run:
  1. Calibrate BTC + ETH surfaces off Deribit; log per-slice diagnostics.
  2. Fetch active Polymarket crypto binaries; parse; price; append every
     (fair, market) pair to data/harness/snapshots.csv.
  3. Check previously-snapshotted markets whose resolution time has
     passed; fetch outcomes; append to data/harness/resolutions.csv.
  4. Score: Brier of OUR fair vs Brier of the MARKET price on the same
     resolved set. THE decision number is the difference:

         skill = Brier(market) - Brier(model)

     skill > 0 sustained over >= ~100 resolutions -> real edge; let
     Mandalorian trade small. skill <= 0 -> the market knows more than
     the model; fix inputs, don't trade. Also prints a calibration
     table (predicted-probability buckets vs realized frequency).

Notes:
  * uses only the LAST snapshot per market before resolution for scoring
    (earlier snapshots are for studying edge decay, not skill).
  * cron example (daily 07:00 UTC, after Deribit's 08:00 expiry is >2d
    away for dailies):  0 7 * * * cd ~/QuantLibLab && .venv/bin/python
    scripts/run_calibration_harness.py >> data/harness/harness.log 2>&1
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s %(name)s %(message)s")
log = logging.getLogger("harness")

DATA_DIR = Path("data/harness")
SNAPSHOTS = DATA_DIR / "snapshots.csv"
RESOLUTIONS = DATA_DIR / "resolutions.csv"

SNAP_FIELDS = ["asof", "market_id", "question", "asset", "type", "strike",
               "resolution", "T_years", "forward", "sigma_at_k", "smile_slope",
               "fair", "fair_lo", "fair_hi", "market_yes", "liquidity",
               "edge_edge_yes", "edge_edge_no", "edge_side"]
RES_FIELDS = ["market_id", "resolved_at", "outcome"]


def _append_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_calibrate() -> dict:
    from quantliblab.harness.deribit_surface import calibrate_currency
    surfaces = {}
    for ccy in ("BTC", "ETH"):
        fs = calibrate_currency(ccy)
        for d in fs.diagnostics:
            log.info("  %s %s T=%.4f F=%.0f quotes=%d rmse=%.2fvp used=%s %s",
                     ccy, d.expiry, d.T, d.F, d.n_quotes,
                     d.rmse_volpts if d.rmse_volpts == d.rmse_volpts else -1,
                     d.used, d.reason)
        if fs.surface is None:
            log.error("%s: no usable surface — skipping pricing", ccy)
        surfaces[ccy] = fs
    return surfaces


def step_price(surfaces: dict) -> list[dict]:
    from quantliblab.harness.polymarket import (
        extract_crypto_binaries, fetch_gamma_markets, price_market,
    )
    raw = fetch_gamma_markets()
    log.info("Gamma returned %d active markets", len(raw))
    markets = extract_crypto_binaries(raw)
    log.info("parsed %d crypto binaries", len(markets))
    rows = []
    for pm in markets:
        fs = surfaces.get(pm.asset)
        if fs is None or fs.surface is None:
            continue
        try:
            row = price_market(pm, fs)
        except Exception as e:
            log.warning("pricing failed for %s: %s", pm.market_id, e)
            continue
        if row:
            rows.append(row)
    _append_csv(SNAPSHOTS, SNAP_FIELDS, rows)
    # print the opportunity board, largest |edge| first
    priced = [r for r in rows if r.get("market_yes") not in ("", None)]
    priced.sort(key=lambda r: abs(float(r["fair"]) - float(r["market_yes"])),
                reverse=True)
    for r in priced[:10]:
        log.info("EDGE %+0.3f  fair=%.3f mkt=%.3f  %s",
                 float(r["fair"]) - float(r["market_yes"]),
                 float(r["fair"]), float(r["market_yes"]), r["question"][:80])
    return rows


def step_resolve() -> None:
    from quantliblab.harness.polymarket import fetch_resolution
    now = datetime.now(timezone.utc)
    snaps = _read_csv(SNAPSHOTS)
    done = {r["market_id"] for r in _read_csv(RESOLUTIONS)}
    pending = sorted({
        r["market_id"] for r in snaps
        if r["market_id"] not in done
        and datetime.fromisoformat(r["resolution"]) < now
    })
    new = []
    for mid in pending:
        try:
            outcome = fetch_resolution(mid)
        except Exception as e:
            log.warning("resolution fetch failed for %s: %s", mid, e)
            continue
        if outcome is not None:
            new.append({"market_id": mid, "resolved_at": now.isoformat(),
                        "outcome": outcome})
    if new:
        _append_csv(RESOLUTIONS, RES_FIELDS, new)
    log.info("resolved %d newly matured markets (%d still pending)",
             len(new), len(pending) - len(new))


def step_score() -> None:
    snaps = _read_csv(SNAPSHOTS)
    res = {r["market_id"]: int(r["outcome"]) for r in _read_csv(RESOLUTIONS)}
    if not res:
        log.info("no resolved markets yet — Brier scoring skipped")
        return
    # last snapshot per resolved market
    last: dict[str, dict] = {}
    for r in snaps:
        if r["market_id"] in res:
            prev = last.get(r["market_id"])
            if prev is None or r["asof"] > prev["asof"]:
                last[r["market_id"]] = r

    model_se, market_se, n = 0.0, 0.0, 0
    buckets = defaultdict(lambda: [0, 0])          # bucket -> [count, hits]
    for mid, r in last.items():
        y = res[mid]
        try:
            fair = float(r["fair"])
        except (ValueError, KeyError):
            continue
        model_se += (fair - y) ** 2
        b = min(int(fair * 10), 9)
        buckets[b][0] += 1
        buckets[b][1] += y
        if r.get("market_yes") not in ("", None):
            market_se += (float(r["market_yes"]) - y) ** 2
        n += 1
    if n == 0:
        return
    brier_model = model_se / n
    brier_market = market_se / n
    log.info("── SCOREBOARD (n=%d resolved) ─────────────────────", n)
    log.info("Brier(model)=%.4f  Brier(market)=%.4f  skill=%+.4f  %s",
             brier_model, brier_market, brier_market - brier_model,
             "MODEL BEATS MARKET" if brier_market > brier_model
             else "market beats model — do not trade yet")
    log.info("calibration: bucket -> predicted mid vs realized freq")
    for b in sorted(buckets):
        cnt, hits = buckets[b]
        log.info("  [%.1f–%.1f)  n=%3d  realized=%.3f",
                 b / 10, (b + 1) / 10, cnt, hits / cnt)
    if n < 100:
        log.info("(< 100 resolutions — treat skill as noise, keep papering)")


# ---------------------------------------------------------------------------
# Offline self-test — no network required
# ---------------------------------------------------------------------------

def selftest() -> int:
    import numpy as np
    from quantliblab.harness.polymarket import MarketType, parse_question
    from quantliblab.volatility.smile.svi import SVIParams
    from quantliblab.volatility.surface.local_vol import LocalVolSurface, SVISlice
    from quantliblab.harness.deribit_surface import FittedSurface
    from quantliblab.harness.polymarket import ParsedMarket, price_market
    from datetime import timedelta

    cases = [
        ("Will Bitcoin be above $105,000 on July 8?", ("BTC", MarketType.EUROPEAN_ABOVE, 105000.0)),
        ("Bitcoin below $95k on August 1?",           ("BTC", MarketType.EUROPEAN_BELOW, 95000.0)),
        ("Will Ethereum reach $4,000 by December 31?",("ETH", MarketType.TOUCH, 4000.0)),
        ("Will BTC dip to $80k by September?",        ("BTC", MarketType.TOUCH, 80000.0)),
        ("Bitcoin Up or Down on July 5?",             (None,)),
        ("What price will Ethereum hit in July?",     (None,)),
        ("Will Bitcoin hit an all-time high in 2026?",(None,)),
        ("Will bitcoin hit $1m before GTA VI?",        (None,)),
    ]
    ok = True
    for q, exp in cases:
        a, t, s = parse_question(q)
        got = (a,) if a is None else (a, t, s)
        status = "PASS" if got == exp else "FAIL"
        ok &= status == "PASS"
        print(f"  [{status}] {q!r} -> {got}")

    # synthetic surface + one market of each type
    F0 = 110_000.0
    s1 = SVISlice(0.10, F0 * 1.005, SVIParams(0.002, 0.30, -0.3, 0.0, 0.15))
    s2 = SVISlice(0.30, F0 * 1.015, SVIParams(0.008, 0.40, -0.25, 0.02, 0.25))
    fs = FittedSurface("BTC", datetime.now(timezone.utc), F0,
                       LocalVolSurface([s1, s2]), [])
    now = datetime.now(timezone.utc)
    for q, mt, K, ymid in [
        ("Will Bitcoin be above $120,000 on <date>?", MarketType.EUROPEAN_ABOVE, 120_000.0, 0.30),
        ("Will Bitcoin reach $125,000 by <date>?",    MarketType.TOUCH,          125_000.0, 0.40),
    ]:
        pm = ParsedMarket("test", q, "BTC", mt, K, now + timedelta(days=60),
                          ymid, 10_000.0, {})
        row = price_market(pm, fs, now)
        sane = row and 0.0 <= float(row["fair"]) <= 1.0
        ok &= bool(sane)
        print(f"  [{'PASS' if sane else 'FAIL'}] {mt.value}: fair={row['fair']}"
              f" band=[{row['fair_lo']},{row['fair_hi']}] edge_side={row.get('edge_side')}")
    print("SELFTEST", "PASSED" if ok else "FAILED")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--skip-resolve", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest()
    surfaces = step_calibrate()
    step_price(surfaces)
    if not args.skip_resolve:
        step_resolve()
    step_score()
    return 0


if __name__ == "__main__":
    sys.exit(main())
