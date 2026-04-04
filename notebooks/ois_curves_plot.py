"""
Plot SOFR, SONIA and ESTR OIS zero coupon curves.
Run from the project root:
    python notebooks/ois_curves_plot.py
"""
from datetime import date, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from quantliblab.curves import OISCurve, CurveInstrument, InstrumentType

VALUATION = date(2025, 3, 24)

# ── Market quotes (illustrative, close to real levels on that date) ──────────

SOFR_INSTRUMENTS = [
    CurveInstrument("ON",  InstrumentType.DEPOSIT,  0.0431),
    CurveInstrument("1W",  InstrumentType.OIS_SWAP, 0.0430),
    CurveInstrument("1M",  InstrumentType.OIS_SWAP, 0.0428),
    CurveInstrument("3M",  InstrumentType.OIS_SWAP, 0.0435),
    CurveInstrument("6M",  InstrumentType.OIS_SWAP, 0.0445),
    CurveInstrument("12M", InstrumentType.OIS_SWAP, 0.0465),
]

SONIA_INSTRUMENTS = [
    CurveInstrument("ON",  InstrumentType.DEPOSIT,  0.04457),
    CurveInstrument("1W",  InstrumentType.OIS_SWAP, 0.04450),
    CurveInstrument("1M",  InstrumentType.OIS_SWAP, 0.04430),
    CurveInstrument("3M",  InstrumentType.OIS_SWAP, 0.04380),
    CurveInstrument("6M",  InstrumentType.OIS_SWAP, 0.04250),
    CurveInstrument("12M", InstrumentType.OIS_SWAP, 0.04050),
]

ESTR_INSTRUMENTS = [
    CurveInstrument("ON",  InstrumentType.DEPOSIT,  0.02418),
    CurveInstrument("1W",  InstrumentType.OIS_SWAP, 0.02410),
    CurveInstrument("1M",  InstrumentType.OIS_SWAP, 0.02390),
    CurveInstrument("3M",  InstrumentType.OIS_SWAP, 0.02300),
    CurveInstrument("6M",  InstrumentType.OIS_SWAP, 0.02150),
    CurveInstrument("12M", InstrumentType.OIS_SWAP, 0.01950),
]

# ── Build curves ─────────────────────────────────────────────────────────────

sofr  = OISCurve.sofr(VALUATION, SOFR_INSTRUMENTS)
sonia = OISCurve.sonia(VALUATION, SONIA_INSTRUMENTS)
estr  = OISCurve.estr(VALUATION, ESTR_INSTRUMENTS)

# ── Interpolation grid: daily from valuation to 12M ─────────────────────────

dates = [VALUATION + timedelta(days=d) for d in range(1, 366)]

def zero_rates(curve):
    return [curve.zero_rate(d) * 100 for d in dates]   # in %

# ── Plot ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(10, 5))
fig.suptitle(
    f"OIS Zero Coupon Curves — Valuation {VALUATION.isoformat()}",
    fontsize=13, fontweight="bold",
)

colors = {"SOFR": "#1f77b4", "SONIA": "#d62728", "ESTR": "#2ca02c"}

for curve, label in [(sofr, "SOFR (USD)"), (sonia, "SONIA (GBP)"), (estr, "ESTR (EUR)")]:
    key = label.split()[0]
    ax.plot(dates, zero_rates(curve), label=label, color=colors[key], linewidth=2)
    for p in curve.pillars:
        ax.scatter(p.maturity_date, p.zero_rate * 100,
                   color=colors[key], s=50, zorder=5)

ax.set_title("Zero Rates (%)", fontsize=11)
ax.set_ylabel("Rate (%)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")
ax.legend()
ax.grid(True, alpha=0.3)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.2f}%"))

plt.tight_layout()
plt.savefig("notebooks/ois_curves.png", dpi=150, bbox_inches="tight")
print("Saved: notebooks/ois_curves.png")
plt.show()
