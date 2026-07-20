# QuantLibLab

A multi-asset quantitative pricing library in Python, covering interest rates, FX, commodities, and crypto.

## Asset classes

| Asset class | Instruments | Benchmark rates |
|---|---|---|
| Interest rates | OIS swaps, SOFR futures | SOFR, SONIA, ESTR |
| FX | Spot, forwards, vanilla options | — |
| Commodities | Futures, options | WTI crude, base metals |
| Crypto | Perps, futures, options | BTC, ETH |

## Project layout

```
quantliblab/
├── conventions/     # Day counts, business day rules, calendars, tenors
├── math/            # Interpolation, solvers, distributions, statistics
├── data/            # Market data loaders, validators, CSV store
├── curves/          # OIS curve construction and bootstrapping
├── volatility/      # SVI smile fits (smile/svi.py), Dupire local vol (surface/local_vol.py)
├── instruments/     # Product definitions (rates, FX, commodities, crypto)
├── pricing/         # Analytical engines (analytical/digital.py), Monte Carlo, Greeks
├── harness/         # Deribit surface calibration + Polymarket fair-value harness
└── utils/           # Dates, logging, config
dashboard/           # Streamlit live dashboard (OIS curves)
notebooks/           # Jupyter explorers (curves_explorer.ipynb)
scripts/             # fetch_daily_data.py, run_calibration_harness.py
```

## Status

| Module | Status |
|---|---|
| `conventions` | Done |
| `math` | Done |
| `data` | Done |
| `curves` | Done |
| `volatility` (SVI smile, Dupire local vol) | Done |
| `instruments/crypto` | Done |
| `instruments/rates, fx, commodities` | Planned |
| `pricing/analytical` (smile-adjusted digitals, one-touch) | Done |
| `pricing/montecarlo, risk` | Planned |
| `harness` (Deribit ↔ Polymarket paper-trading loop) | Live-testing |

## Market data sources

### Interest rates

| Rate | Tenor | Source | Series ID / Ticker | Auth | Lag |
|---|---|---|---|---|---|
| SOFR | Overnight | FRED (NY Fed) | `SOFR` | Free API key | Same day ~8am ET |
| SOFR | 30/90/180-day avg | FRED (NY Fed) | `SOFR30DAYAVG`, `SOFR90DAYAVG`, `SOFR180DAYAVG` | Free API key | Same day |
| SONIA | Overnight | FRED (Bank of England) | `IUDSOIA` | Free API key | T+1 |
| ESTR | Overnight | FRED (ECB) | `ECBESTRVOLWGTTRMDMNRT` | Free API key | T+1 |
| SOFR 1M futures | Monthly strip | Yahoo Finance | `SR1{M}{YY}.CME` | None | Real-time |
| SOFR 3M futures | Quarterly strip | Yahoo Finance | `SR3{M}{YY}.CME` | None | Real-time |

**SOFR futures tickers** use standard futures month codes:
`F=Jan G=Feb H=Mar J=Apr K=May M=Jun N=Jul Q=Aug U=Sep V=Oct X=Nov Z=Dec`

Example: `SR3M26.CME` = 3-month SOFR futures, June 2026 delivery.

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

---

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

### Instrument set

| Tenor | Instrument | Calibration |
|---|---|---|
| ON | Deposit | Direct — `P = 1/(1+r·τ)`, start = T+0, end = T+1 |
| TN | Deposit | Direct, start = T+1, end = T+2 |
| 1W–12M | OIS swap | Newton-Raphson (Brent fallback) |
| SR1 1M strip | SOFR 1-month futures | Pricing identity (see below) |
| SR3 3M strip | SOFR 3-month futures | Pricing identity (see below) |

### SOFR futures pillars

SR1 and SR3 futures are bootstrapped on top of the deposit/OIS pillars via
`bootstrap_sofr_futures()`. The pricing identity is:

```
P(ref_start) / P(ref_end) = 1 + K · τ_ACT360
```

where K is the futures-implied rate (`(100 − price) / 100`) and τ is the
ACT/360 year fraction over the reference period.

**Reference periods:**

| Type | Reference period | Start | End |
|---|---|---|---|
| SR3 | 3-month quarter | 3rd Wednesday of delivery month − 3 | 3rd Wednesday of delivery month |
| SR1 | Calendar month | 1st of delivery month | Last day of delivery month |

SR3 quarters are contiguous (end of one = start of next).

**Stub handling:** contracts whose reference period has already started
(`ref_start < valuation_date`) are skipped — realised stub compounding
is not in scope at this stage.

**Convexity adjustment** (optional): daily margining creates a small
bias between futures rates and equivalent OIS forwards:
```
CA ≈ σ² · τ₁ · τ₂ / 2     (< 1bp for maturities ≤ 1Y at σ ≈ 1%)
```
Pass `convexity_vol` to `bootstrap_sofr_futures()` to enable.

### Calendars

| Calendar | Used for | Difference |
|---|---|---|
| `NewYorkCalendar` | SOFR OIS swaps, NY Fed fixing | Federal Reserve holidays |
| `CMECalendar` | SOFR futures (SR3 reference quarters) | Same as NY Fed, **except Good Friday** (CME closed, NY Fed open) |
| `LondonCalendar` | SONIA | UK bank holidays |
| `TARGETCalendar` | ESTR | TARGET2 holidays |

### Bootstrap algorithm

1. Sort pillars by maturity
2. For deposits: `P(T) = 1/(1 + r·τ)` — direct formula
3. For OIS swaps: Newton-Raphson on `NPV(r) = (1−P(T)) − K·τ·P(T) = 0`
4. For SOFR futures: `P(ref_end) = P(ref_start) / (1 + K·τ)`, where `P(ref_start)` is interpolated from the running curve
5. All previously solved pillars are held fixed at each step

### Interpolation

Method: **flat-forward** (log-linear on discount factors)

Between two calibrated pillars t₁ and t₂, the instantaneous forward rate
is held constant:

```
f(t₁, t₂) = -[log P(t₂) - log P(t₁)] / (t₂ - t₁)
```

Extrapolation beyond the last pillar: flat-forward (hold the last
implied forward rate constant).

### Output columns

| Column | Example | Description |
|---|---|---|
| `instrument` | `OISSwap`, `SR3Future` | Instrument type used to calibrate this pillar |
| `tenor` | `3M`, `SR3U26` | Human-readable tenor or futures ticker |
| `start_date` | `2026-04-15` | Settlement / effective date |
| `maturity_date` | `2026-07-15` | End date (business day adjusted) |
| `year_fraction` | `0.2493` | Time in years from valuation date |
| `zero_rate` | `0.0431` | Calibrated continuously-compounded zero rate |
| `discount_factor` | `0.9893` | `exp(-zero_rate × year_fraction)` |

### Curve API

```python
from quantliblab.curves import OISCurve, CurveInstrument, InstrumentType, bootstrap_sofr_futures
from quantliblab.data.loaders.sofr_futures_loader import fetch_sr1_strip, fetch_sr3_strip
from datetime import date

val = date(2026, 4, 13)

# --- Step 1: short-end deposits + OIS swaps ---
instruments = [
    CurveInstrument("ON",  InstrumentType.DEPOSIT,  0.0431),
    CurveInstrument("1W",  InstrumentType.OIS_SWAP,  0.0430),
    CurveInstrument("1M",  InstrumentType.OIS_SWAP,  0.0428),
]
sofr = OISCurve.sofr(val, instruments)

# --- Step 2: extend with live SOFR futures ---
sr1 = fetch_sr1_strip(val, n_contracts=9)
sr3 = fetch_sr3_strip(val, n_contracts=5)

from quantliblab.curves.bootstrapper import bootstrap as bs
from quantliblab.conventions.day_count import DayCountBasis
from quantliblab.conventions.calendars import NewYorkCalendar
dep_pillars = bs(val, instruments, DayCountBasis.ACT_360, NewYorkCalendar(), settlement_days=2)

futures_pillars = bootstrap_sofr_futures(val, sr1 + sr3, dep_pillars)
all_pillars = dep_pillars + futures_pillars

# --- Core curve methods ---
sofr.discount_factor(date(2026, 10, 13))   # P(t)
sofr.zero_rate(date(2026, 10, 13))         # r(t), continuously compounded
sofr.forward_rate(date(2026, 7, 13), date(2026, 10, 13))  # f(t1, t2)
sofr.to_dataframe()                         # pillar table as pandas DataFrame
```

---

## FX forward curves

### Overview

FX forward rates are derived from two OIS discount curves and the spot rate
via **Covered Interest Rate Parity (CIP)** — no separate forward market data required.

```
F(T) = S × P_for(T) / P_dom(T)
```

**Swap points** (forward points):
```
Swap points = (F(T) - S) × 10,000   [in pips]
```

### Supported pairs

| Pair | Domestic curve | Foreign curve |
|---|---|---|
| GBPUSD | SOFR (USD) | SONIA (GBP) |
| EURUSD | SOFR (USD) | ESTR (EUR) |
| EURGBP | SONIA (GBP) | ESTR (EUR) |

### FX forward API

```python
from quantliblab.curves import gbpusd_forward

fx = gbpusd_forward(spot=1.2924, sofr_curve=sofr, sonia_curve=sonia)
fx.forward(date)           # F(T)
fx.swap_points(date)       # (F(T) − S) × 10,000 pips
fx.forward_curve()         # DataFrame at ON/1W/1M/3M/6M/9M/12M
```

---

## Crypto instruments (off-chain)

### Instrument types

| Class | File | Represents |
|---|---|---|
| `BTCFuture` / `ETHFuture` | `instruments/crypto/futures.py` | Cash-settled quarterly futures on Deribit |
| `BTCOption` / `ETHOption` | `instruments/crypto/options.py` | European cash-settled options (calls and puts) |
| `BTCPerpetual` / `ETHPerpetual` | `instruments/crypto/perpetuals.py` | Perpetual swaps with funding rate mechanism |

### Data source: Deribit

All crypto market data is sourced from the **Deribit REST API v2** (free, no authentication required).

| Data | Endpoint | Notes |
|---|---|---|
| BTC/ETH spot index | `get_index_price` | Composite multi-exchange weighted average |
| Futures snapshot | `get_book_summary_by_currency?kind=future` | Mark price, bid/ask, volume, OI |
| Options snapshot | `get_book_summary_by_currency?kind=option` | Mark price, IV (converted to decimal), bid/ask, OI |
| Option Greeks | `ticker` (per instrument) | Delta, gamma, vega, theta, rho |
| Perpetual snapshot | `ticker` (BTC-PERPETUAL / ETH-PERPETUAL) | Mark price, index, current and 8h funding rates |
| Funding rate history | `get_funding_rate_history` | Hourly 8h funding rates |

**Options prices** are quoted in BTC/ETH and converted to USD via the index price.
**Implied volatility** is returned as a percentage (e.g. 60.0) and converted to decimal (0.60).

### Crypto loader API

```python
from quantliblab.data.loaders.deribit_loader import (
    fetch_index_price, fetch_futures, fetch_perpetual,
    fetch_options, fetch_option_ticker, fetch_funding_history,
)

btc_spot = fetch_index_price("BTC")
futures  = fetch_futures("BTC")        # list[BTCFuture], sorted by expiry
perp     = fetch_perpetual("BTC")
options  = fetch_options("BTC")        # list[BTCOption], Greeks=None
opt      = fetch_option_ticker("BTC-28MAR25-80000-C")  # with full Greeks
```

---

## Volatility surfaces & the Polymarket harness

The Derman–Kani local-volatility program implemented the modern way: fit an
arbitrage-free **SVI** smile per expiry (Gatheral 2004), then extract local vol
analytically via **Dupire/Gatheral** on the total-variance surface — not
implied trees, and no finite-differencing of noisy quotes.

| Module | Role |
|---|---|
| `volatility/smile/svi.py` | Raw SVI fit per expiry (weighted LSQ on total variance, multi-start, butterfly soft-penalty). Analytic `w`, `w'`, `w''` and the Gatheral–Jacquier `g(k)` no-arbitrage check — `g(k)` is also the Dupire denominator. |
| `volatility/surface/local_vol.py` | `LocalVolSurface`: linear-in-total-variance time interpolation (calendar-arb-preserving), Dupire local variance `= (∂w/∂T) / g(k)`, log-Euler path MC, one-touch MC estimator. |
| `pricing/analytical/digital.py` | Polymarket fair-value engine. European digitals use the **smile-slope correction** — `P(F_T > K) = N(d2) − φ(d2)·√T·(dσ/dk)` — not plain `N(d2)`; with crypto skew the correction is routinely 2–6c on a $1 contract, i.e. the entire edge. One-touch closed form (reflection principle, driftless GBM) for the fast bound. |
| `harness/deribit_surface.py` | Nightly BTC/ETH calibration: forwards from the futures strip (carry-interpolated), OTM two-sided quote filtering, per-expiry SVI fits, RMSE/butterfly/calendar diagnostics. |
| `harness/polymarket.py` | Gamma-API loader; parses questions into European above/below vs one-touch (buckets/dailies skipped — mis-parse costs money, a skip costs nothing); prices each vs the fitted surface. |
| `scripts/run_calibration_harness.py` | The nightly loop: calibrate → price → snapshot → resolve → score. |

### Market taxonomy

- **European** ("BTC above $105k on July 8") → terminal distribution only:
  smile-slope digital, **no local vol needed**.
- **One-touch** ("BTC reaches $150k by Dec 31") → path-dependent: the flat
  barrier-IV closed form and the local-vol MC **bracket** the truth (LV is
  known to under-price one-touch under steep smiles vs stochastic vol).
  Quote inside the band, size to the band width — a band, not a point.

### The go-live gate

Every run appends `(fair, market)` snapshots; matured markets are resolved and
scored. The single decision number is

```
skill = Brier(market) − Brier(model)
```

on the same resolved set (last snapshot per market). `skill > 0` sustained over
~100+ resolutions ⇒ real edge, trade small. `skill ≤ 0` ⇒ the market knows
more than the model — fix inputs, don't trade. A predicted-vs-realized
calibration table is printed alongside.

```bash
python scripts/run_calibration_harness.py             # live nightly run
python scripts/run_calibration_harness.py --selftest  # offline logic test
```

Honest caveats live in the module docstrings: Gamma API schema is unversioned
(parse defensively), Polymarket resolves on Binance/Chainlink prints vs
Deribit's composite index (small basis ⇒ require an edge threshold), fair
values are undiscounted to match $1-payout quoting.


## Dashboard and notebooks

### Streamlit dashboard (`dashboard/ois_curves.py`)

Live OIS curves panel with three side-by-side views (SOFR / SONIA / ESTR):
- Zero rate curve with pillar markers
- Forward rate step function
- Pillar table: Instrument, Start Date, End Date, Maturity, Zero Rate, Ds Factor

```bash
streamlit run dashboard/ois_curves.py
```

### Jupyter notebook (`notebooks/curves_explorer.ipynb`)

Interactive curves explorer with `ipywidgets` sliders (±300bps per tenor):
- Move sliders to shock individual tenors
- Live-updating zero rate and discount factor plots (Plotly)
- Pillar table with formatted rates

```bash
jupyter lab notebooks/curves_explorer.ipynb
```

---

## Getting started

**Requirements:** Python 3.11+

```bash
git clone https://github.com/Decentralisedme/QuantLibLab.git
cd QuantLibLab

python -m venv .venv
source .venv/bin/activate      # macOS/Linux
.venv\Scripts\activate         # Windows

pip install -e ".[dev]"
```

## Environment variables

Copy `.env.example` to `.env`:

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
