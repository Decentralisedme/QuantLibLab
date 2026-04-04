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
