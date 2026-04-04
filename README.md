# QuantLibLab

A multi-asset quantitative pricing library in Python, covering interest rates, FX, commodities, and crypto.

## Asset classes

| Asset class | Instruments | Benchmark rates |
|---|---|---|
| Interest rates | OIS swaps, FRAs | SOFR, SONIA, ESTR |
| FX | Spot, forwards, vanilla options | — |
| Commodities | Futures, options | WTI crude, base metals |
| Crypto | Perps, options | BTC, ETH, Uniswap yield curves |

## Project layout

```
quantliblab/
├── conventions/     # Day counts, business day rules, calendars, tenors
├── math/            # Interpolation, solvers, distributions, statistics
├── data/            # Market data loaders, validators, CSV store
├── curves/          # OIS curve construction and bootstrapping
├── volatility/      # Vol surface parametrizations (SABR, SVI)
├── instruments/     # Product definitions (rates, FX, commodities, crypto)
├── pricing/         # Analytical and Monte Carlo engines, Greeks, P&L
└── utils/           # Dates, logging, config
```

## Status

| Module | Status |
|---|---|
| `conventions` | Done |
| `math` | Done |
| `data` | Done |
| `curves` | Planned |
| `volatility` | Planned |
| `instruments` | Planned |
| `pricing` | Planned |

## Market data sources

### Interest rates

| Rate | Tenor | Source | Series ID | Auth | Lag |
|---|---|---|---|---|---|
| SOFR | Overnight | FRED (NY Fed) | `SOFR` | Free API key | Same day ~8am ET |
| SOFR | 30/90/180-day avg | FRED (NY Fed) | `SOFR30DAYAVG`, `SOFR90DAYAVG`, `SOFR180DAYAVG` | Free API key | Same day |
| SONIA | Overnight | FRED (Bank of England) | `IUDSOIA` | Free API key | T+1 |
| ESTR | Overnight | FRED (ECB) | `ECBESTRVOLWGTTRMDMNRT` | Free API key | T+1 |

**Note on 3-month rates:** Forward-looking term rates (CME Term SOFR 3M, FTSE Term SONIA 3M)
require paid or licensed data feeds. As a free alternative, `SOFR90DAYAVG` is the
backward-looking 90-day compounded average — the correct rate for OIS discounting and
most SOFR-linked instruments.

### FX spot

| Pair | Source | Ticker | Auth | Notes |
|---|---|---|---|---|
| EURUSD | yfinance (Yahoo Finance) | `EURUSD=X` | None | Mid-market closing price |
| GBPUSD | yfinance | `GBPUSD=X` | None | Mid-market closing price |
| USDJPY | yfinance | `USDJPY=X` | None | Mid-market closing price |
| USDCHF | yfinance | `USDCHF=X` | None | Mid-market closing price |
| AUDUSD | yfinance | `AUDUSD=X` | None | Mid-market closing price |
| USDCAD | yfinance | `USDCAD=X` | None | Mid-market closing price |

### Fetching data

```bash
# Fetch yesterday's data (run daily as cron job)
python scripts/fetch_daily_data.py

# Fetch a specific date
python scripts/fetch_daily_data.py 2025-01-15

# Backfill a date range
python scripts/fetch_daily_data.py 2025-01-01 2025-03-31
```

Data is saved to `quantliblab/data/raw/` as CSV files (excluded from git).

## OIS rate curve construction

### Overview

OIS (Overnight Index Swap) curves are **zero coupon curves** bootstrapped
from market instruments. Each pillar is calibrated so that the instrument's
NPV is exactly zero given the discount factors implied by the curve.

### Currencies and benchmarks

| Currency | OIS benchmark | Day count | Calendar | Settlement |
|---|---|---|---|---|
| USD | SOFR | ACT/360 | New York | T+2 |
| GBP | SONIA | ACT/365 | London | T+0 |
| EUR | ESTR | ACT/360 | TARGET2 | T+2 |

### Instrument set (per currency)

| Tenor | Instrument | Calibration method |
|---|---|---|
| ON | Deposit (the O/N fixing itself) | Direct — rate is the zero rate, no solver |
| TN | Deposit | Direct |
| 1W | OIS swap | Newton-Raphson (Brent fallback) |
| 1M | OIS swap | Newton-Raphson (Brent fallback) |
| 3M | OIS swap | Newton-Raphson (Brent fallback) |
| 6M | OIS swap | Newton-Raphson (Brent fallback) |
| 9M | OIS swap (optional — quoted less frequently) | Newton-Raphson (Brent fallback) |
| 12M | OIS swap | Newton-Raphson (Brent fallback) |

FRAs and futures are excluded at this stage (LIBOR-era instruments,
less relevant for OIS curves up to 1Y).

### Bootstrap algorithm

1. Sort pillars by maturity
2. For each pillar, solve for the zero rate `r(t)` such that the instrument NPV = 0:
   - **Deposits:** `r = quoted_rate` (direct, no solver)
   - **OIS swaps:** find `r` so that fixed leg PV = floating leg PV, using Newton-Raphson
3. Store the resulting discount factor `P(t) = exp(-r × year_fraction)`
4. Proceed to the next pillar using all previously solved discount factors

### Interpolation

Method: **flat-forward** (log-linear on discount factors)

Between two calibrated pillars t₁ and t₂, the instantaneous forward rate
is held constant:

```
f(t₁, t₂) = -[log P(t₂) - log P(t₁)] / (t₂ - t₁)
```

This is the market standard for OIS curves. It ensures:
- Forward rates are continuous and non-negative between pillars
- Smooth Greeks (DV01, delta) with no jumps at pillar boundaries
- Correct OIS swap pricing (no phantom forward rate spikes)

Extrapolation beyond the last pillar: flat-forward (hold the last
implied forward rate constant).

### Output columns

| Column | Example | Description |
|---|---|---|
| `instrument` | `OISSwap` | Instrument type used to calibrate this pillar |
| `tenor` | `3M` | Human-readable tenor |
| `maturity_date` | `2025-06-24` | Actual calendar date (business day adjusted) |
| `year_fraction` | `0.2493` | Time in years (per currency day count convention) |
| `zero_rate` | `0.0431` | Calibrated continuously-compounded zero rate |
| `discount_factor` | `0.9893` | `exp(-zero_rate × year_fraction)` |

### Curve API

```python
curve.discount_factor(date)        # P(t) = exp(-r × t)
curve.zero_rate(date)              # r(t) = -log(P(t)) / t
curve.forward_rate(date1, date2)   # f(t1,t2) = -log(P(t2)/P(t1)) / (t2-t1)
```

### Compounding convention

All rates stored and computed in **continuous compounding** (market standard
for OIS discount curves). Conversion to other conventions (annual, semi-annual)
is available via the `conventions` module.

---

## Getting started

**Requirements:** Python 3.11+

```bash
# Clone the repo
git clone https://github.com/Decentralisedme/QuantLibLab.git
cd QuantLibLab

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows

# Install with dev dependencies
pip install -e ".[dev]"
```

## Environment variables

Copy `.env.example` to `.env` and fill in your API keys:

```bash
cp .env.example .env
```

| Variable | Required | Where to get it |
|---|---|---|
| `FRED_API_KEY` | Yes (for rates) | Free at https://fred.stlouisfed.org/docs/api/api_key.html |

## Running tests

```bash
pytest tests/ -v
```

## Related projects

- **OilIntel** — physical oil intelligence app (Raspberry Pi, OpenClaw)
- **Eliza** — crypto agent for on-chain analytics, Deribit vol surfaces, DeFi yield curves
