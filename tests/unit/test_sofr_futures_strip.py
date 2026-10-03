"""
SOFR futures strip construction tests (defect D-002).

Mixing SR1 and SR3 contracts with overlapping reference periods made the
sequential bootstrap sawtooth (2.47%–5.88% forwards on the 2026-07-25
golden snapshot). The fix is D-002 option 2: SR1 for the front, SR3
beyond, handing over at the first live SR3 so nothing overlaps.

Required by D-002:
  1. Monotone, smooth forwards across the handover point
  2. No pillar forward more than ~50bp from its neighbour unless a known
     policy (FOMC) date lies between them
  3. Reprice the input contracts from the bootstrapped curve; recover quotes

Each runs on real golden-snapshot quotes and on a synthetic strip priced
off a known smooth curve.
"""
from __future__ import annotations

import calendar as _cal
import csv
import math
from datetime import date, timedelta
from itertools import pairwise
from pathlib import Path

import pytest

from quantliblab.conventions.calendars import NewYorkCalendar
from quantliblab.conventions.day_count import DayCountBasis, year_fraction
from quantliblab.curves import CurveInstrument, InstrumentType, bootstrap
from quantliblab.curves.base.rate_curve import CurvePillar, RateCurve
from quantliblab.curves.sofr_futures_helper import bootstrap_sofr_futures, select_strip
from quantliblab.data.loaders.sofr_futures_loader import SOFRFuturesContract, _imm_date

BP = 1e-4
ACT360 = DayCountBasis.ACT_360
GOLDEN = Path(__file__).resolve().parents[1] / "fixtures" / "golden"

# FOMC decision days (second day of each meeting), Fed's published 2026
# schedule. Extend when snapshots reach further — a >50bp step with no
# listed date between is meant to fail loudly.
FOMC_DECISIONS = [
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29), date(2026, 6, 17),
    date(2026, 7, 29), date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 9),
]


# ---------------------------------------------------------------------------
# Curve building
# ---------------------------------------------------------------------------

def _contract(ticker: str, kind: str, start: date, end: date, rate: float) -> SOFRFuturesContract:
    return SOFRFuturesContract(
        ticker, kind, end.year, end.month, start, end, 100.0 - rate * 100.0, rate,
    )


def _build(val: date, on_rate: float, contracts: list[SOFRFuturesContract]):
    """O/N deposit anchor + futures strip -> (curve, futures pillars)."""
    on = bootstrap(
        val, [CurveInstrument("ON", InstrumentType.DEPOSIT, on_rate)],
        ACT360, NewYorkCalendar(), settlement_days=2,
    )
    fut = bootstrap_sofr_futures(val, contracts, on)
    return RateCurve(val, on + fut, ACT360), fut


def _segment_forwards(curve: RateCurve) -> list[tuple[CurvePillar, CurvePillar, float]]:
    """Forward implied by each pillar over the segment since the previous one."""
    return [
        (a, b, curve.forward_rate(a.maturity_date, b.maturity_date))
        for a, b in pairwise(curve.pillars)
    ]


def _implied_rate(curve: RateCurve, c: SOFRFuturesContract) -> float:
    tau = year_fraction(c.ref_start, c.ref_end, ACT360)
    return (curve.discount_factor(c.ref_start) / curve.discount_factor(c.ref_end) - 1.0) / tau


# ---------------------------------------------------------------------------
# Golden snapshots
# ---------------------------------------------------------------------------

def _load_snapshot(label: str):
    """(valuation date, O/N anchor, contracts) from a golden snapshot."""
    norm = GOLDEN / label / "normalized"
    if not (norm / "sofr_futures.csv").exists():
        pytest.skip(f"golden snapshot {label} not on disk")
    with (norm / "sofr_futures.csv").open() as f:
        contracts = [
            _contract(r["ticker"].split(".")[0], r["contract_type"],
                      date.fromisoformat(r["ref_start"]), date.fromisoformat(r["ref_end"]),
                      float(r["implied_rate"]))
            for r in csv.DictReader(f)
        ]
    val = date.fromisoformat(label)
    on_rate = None
    if (norm / "rates_on.csv").exists():
        with (norm / "rates_on.csv").open() as f:
            on_rate = float([r for r in csv.DictReader(f) if r["sofr"]][-1]["sofr"])
    if on_rate is None:
        # No FRED feed in this snapshot: proxy O/N by the SR1 month in progress
        on_rate = next(c.implied_rate for c in contracts
                       if c.contract_type == "SR1" and c.ref_start <= val <= c.ref_end)
    return val, on_rate, contracts


SNAPSHOTS = ["2026-07-25", "2026-10-03"]


# ---------------------------------------------------------------------------
# Synthetic strip off a known smooth curve
# ---------------------------------------------------------------------------

SYN_VAL = date(2026, 7, 27)


def _true_df(d: date) -> float:
    """Instantaneous forward rising linearly 4.00% -> 4.60% over two years."""
    t = 0.0 if d == SYN_VAL else year_fraction(SYN_VAL, d, ACT360)
    return math.exp(-(0.040 * t + 0.003 * t * t / 2))


def _true_forward(d1: date, d2: date) -> float:
    return -math.log(_true_df(d2) / _true_df(d1)) / year_fraction(d1, d2, ACT360)


def _synthetic_strip() -> list[SOFRFuturesContract]:
    """12 SR1 months + 8 SR3 quarters, all priced consistently off _true_df."""
    out = []
    y, m = 2026, 8
    for _ in range(12):
        s, e = date(y, m, 1), date(y, m, _cal.monthrange(y, m)[1])
        tau = year_fraction(s, e, ACT360)
        out.append(_contract(f"SR1-{y}-{m:02d}", "SR1", s, e,
                             (_true_df(s) / _true_df(e) - 1) / tau))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    y, m = 2026, 9
    for _ in range(8):
        ny, nm = (y + 1, m - 9) if m > 9 else (y, m + 3)
        s, e = _imm_date(y, m), _imm_date(ny, nm)
        tau = year_fraction(s, e, ACT360)
        out.append(_contract(f"SR3-{ny}-{nm:02d}", "SR3", s, e,
                             (_true_df(s) / _true_df(e) - 1) / tau))
        y, m = ny, nm
    return out


def _synthetic_curve():
    on_rate = (1 / _true_df(SYN_VAL + timedelta(days=1)) - 1) * 360
    return _build(SYN_VAL, on_rate, _synthetic_strip())


# ---------------------------------------------------------------------------
# Strip selection
# ---------------------------------------------------------------------------

class TestSelectStrip:
    def test_golden_july_handover(self):
        """The D-002 input: N26 and U26 have started; SR1 Sep-onward overlaps SR3."""
        val, _, contracts = _load_snapshot("2026-07-25")
        strip = [c.ticker for c in select_strip(val, contracts)]
        assert strip == ["SR1Q26", "SR3Z26", "SR3H27", "SR3M27",
                         "SR3U27", "SR3Z27", "SR3H28", "SR3M28"]

    def test_no_overlap_and_sr1_only_before_handover(self):
        strip = select_strip(SYN_VAL, _synthetic_strip())
        for a, b in pairwise(strip):
            assert b.ref_start >= a.ref_end
        handover = min(c.ref_start for c in strip if c.contract_type == "SR3")
        assert all(c.ref_end <= handover for c in strip if c.contract_type == "SR1")
        assert all(c.ref_start >= handover for c in strip if c.contract_type == "SR3")

    def test_sr1_only_uses_all_live_sr1(self):
        sr1 = [c for c in _synthetic_strip() if c.contract_type == "SR1"]
        assert select_strip(SYN_VAL, sr1) == sr1

    def test_sr3_only_unchanged(self):
        sr3 = [c for c in _synthetic_strip() if c.contract_type == "SR3"]
        assert select_strip(SYN_VAL, sr3) == sr3

    def test_overlapping_sr3_rejected(self):
        a = _contract("A", "SR3", date(2026, 9, 16), date(2026, 12, 16), 0.04)
        b = _contract("B", "SR3", date(2026, 10, 1), date(2027, 1, 4), 0.04)
        with pytest.raises(ValueError, match="Overlapping"):
            select_strip(SYN_VAL, [a, b])

    def test_unknown_contract_type_rejected(self):
        c = _contract("X", "SR6", date(2026, 9, 16), date(2027, 3, 17), 0.04)
        with pytest.raises(ValueError, match="Unknown contract type"):
            select_strip(SYN_VAL, [c])


# ---------------------------------------------------------------------------
# 1. Monotone, smooth forwards across the handover
# ---------------------------------------------------------------------------

def _handover_window(curve: RateCurve) -> list[float]:
    """Forwards of the last SR1 segment and the two segments after it."""
    segs = _segment_forwards(curve)
    k = next(i for i, (_, b, _) in enumerate(segs) if b.instrument == "SR3Future")
    return [f for _, _, f in segs[max(k - 1, 0):k + 2]]


class TestHandoverSmoothness:
    def test_synthetic_recovers_true_forwards(self):
        """Consistent quotes in, the true forward curve out. Segments spanning a
        gap (O/N -> first SR1, last SR1 -> first SR3 end) carry a flat forward
        across the gap, so sit within ~1bp of a linearly rising truth; pure
        SR3 segments are exact."""
        curve, _ = _synthetic_curve()
        for a, b, f in _segment_forwards(curve):
            tol = 1e-6 * BP if a.instrument == b.instrument == "SR3Future" else 1 * BP
            assert abs(f - _true_forward(a.maturity_date, b.maturity_date)) < tol, b.tenor

    def test_synthetic_monotone_everywhere(self):
        curve, _ = _synthetic_curve()
        fwds = [f for _, _, f in _segment_forwards(curve)]
        assert all(x < y for x, y in pairwise(fwds))

    @pytest.mark.parametrize("label", SNAPSHOTS)
    def test_golden_monotone_across_handover(self, label):
        val, on_rate, contracts = _load_snapshot(label)
        curve, _ = _build(val, on_rate, contracts)
        w = _handover_window(curve)
        assert all(x <= y for x, y in pairwise(w)) or all(x >= y for x, y in pairwise(w)), (
            f"{label}: forwards around handover {[round(x * 100, 3) for x in w]}"
        )


# ---------------------------------------------------------------------------
# 2. No unexplained jumps between neighbouring pillar forwards
# ---------------------------------------------------------------------------

def _unexplained_jumps(curve: RateCurve, limit: float = 50 * BP) -> list[str]:
    segs = _segment_forwards(curve)
    bad = []
    for (a0, _, f0), (_, b1, f1) in pairwise(segs):
        if abs(f1 - f0) > limit and not any(
            a0.maturity_date < d < b1.maturity_date for d in FOMC_DECISIONS
        ):
            bad.append(f"{b1.tenor}: {f0 * 100:.3f}% -> {f1 * 100:.3f}%")
    return bad


class TestNoUnexplainedJumps:
    @pytest.mark.parametrize("label", SNAPSHOTS)
    def test_golden_full_strip(self, label):
        """All captured SR1 + SR3 contracts in — the exact D-002 input."""
        val, on_rate, contracts = _load_snapshot(label)
        curve, _ = _build(val, on_rate, contracts)
        assert _unexplained_jumps(curve) == []

    def test_golden_july_range_matches_sr3_only(self):
        """D-002: SR3 alone gave 3.94%–4.23%. The mixed strip must stay near
        that, not 2.47%–5.88%. The front SR1 adds Aug at ~3.72%."""
        val, on_rate, contracts = _load_snapshot("2026-07-25")
        curve, fut = _build(val, on_rate, contracts)
        fwds = [f for _, b, f in _segment_forwards(curve) if b in fut]
        assert 0.0365 < min(fwds) and max(fwds) < 0.0430

    def test_synthetic(self):
        """Truth rises ~7.6bp per quarter; no step should exceed 10bp."""
        curve, _ = _synthetic_curve()
        assert _unexplained_jumps(curve, limit=10 * BP) == []


# ---------------------------------------------------------------------------
# 3. Reprice the input contracts
# ---------------------------------------------------------------------------

class TestReprice:
    @pytest.mark.parametrize("label", SNAPSHOTS)
    def test_golden_strip_reprices(self, label):
        val, on_rate, contracts = _load_snapshot(label)
        curve, _ = _build(val, on_rate, contracts)
        for c in select_strip(val, contracts):
            assert abs(_implied_rate(curve, c) - c.implied_rate) < 1e-6 * BP, c.ticker

    def test_synthetic_reprices_every_contract(self):
        """Strip contracts reprice exactly. Dropped SR1s reprice within 5bp:
        the one straddling the handover gap (Sep) is worst, at ~4bp, because
        option 2 holds the first SR3's forward flat back across the gap —
        the granularity D-002 accepts losing."""
        curve, _ = _synthetic_curve()
        strip = select_strip(SYN_VAL, _synthetic_strip())
        for c in _synthetic_strip():
            tol = 1e-6 * BP if c in strip else 5 * BP
            assert abs(_implied_rate(curve, c) - c.implied_rate) < tol, c.ticker
