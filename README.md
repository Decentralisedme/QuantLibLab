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
| `math` | In progress |
| `data` | Planned |
| `curves` | Planned |
| `volatility` | Planned |
| `instruments` | Planned |
| `pricing` | Planned |

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

## Running tests

```bash
pytest tests/ -v
```

## Environment variables

Copy `.env.example` to `.env` and fill in any API keys for market data sources:

```bash
cp .env.example .env
```

## Related projects

- **OilIntel** — physical oil intelligence app (Raspberry Pi, OpenClaw)
- **Eliza** — crypto agent for on-chain analytics, Deribit vol surfaces, DeFi yield curves
