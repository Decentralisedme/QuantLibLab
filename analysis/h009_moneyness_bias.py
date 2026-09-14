"""H-009 decision rule: is Polymarket systematically rich to model at high
log-moneyness k = ln(strike/forward)?

Pre-registered 2026-09-05 (see vault ARRAKIS/40-research/hypotheses/H-009.md).
One attempt: thresholds below are fixed by that document and must not be
tuned to the data.

    1. Sign stability   — per-day slope of edge on k is negative on >=20/28 days.
    2. Pooled significance — pooled OLS slope on k is negative, p<0.05,
       with standard errors clustered by day (28 daily snapshots of the same
       ~30 overlapping contracts are not independent draws).
    3. Economic size    — mean |edge| in the top k-tercile exceeds 4 cents.

Fail any one -> closed.

Usage:
    python analysis/h009_moneyness_bias.py [path/to/snapshots.csv]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# Fixed by the decision rule -- do not tune.
MIN_SIGN_STABLE_DAYS = 20
TOTAL_DAYS_DENOM = 28  # threshold denominator as written, even if fewer days exist
SIG_ALPHA = 0.05
ECON_SIZE_THRESHOLD = 0.04  # 4 cents


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["day"] = df["asof"].str[:10]
    df["k"] = np.log(df["strike"] / df["forward"])
    df["edge"] = df["fair"] - df["market_yes"]
    return df


def per_day_slope(day_df: pd.DataFrame) -> float | None:
    """OLS slope of edge on k for one day. None if k has no variance."""
    k = day_df["k"].to_numpy()
    y = day_df["edge"].to_numpy()
    if np.ptp(k) == 0 or len(k) < 2:
        return None
    coeffs = np.polyfit(k, y, 1)
    return float(coeffs[0])


def condition_sign_stability(df: pd.DataFrame) -> dict:
    slopes = {}
    for day, day_df in df.groupby("day"):
        s = per_day_slope(day_df)
        if s is not None:
            slopes[day] = s
    n_negative = sum(1 for s in slopes.values() if s < 0)
    n_days_scored = len(slopes)
    passed = n_negative >= MIN_SIGN_STABLE_DAYS
    return {
        "slopes_by_day": slopes,
        "n_negative": n_negative,
        "n_days_scored": n_days_scored,
        "threshold": f"{MIN_SIGN_STABLE_DAYS}/{TOTAL_DAYS_DENOM}",
        "passed": passed,
    }


def cluster_robust_ols(y: np.ndarray, x: np.ndarray, clusters: np.ndarray) -> dict:
    """OLS of y on [1, x] with CR1 cluster-robust standard errors, clustered
    on `clusters`. No statsmodels dependency -- plain linear algebra.
    """
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta

    groups = pd.unique(clusters)
    n_groups = len(groups)
    meat = np.zeros((2, 2))
    for g in groups:
        mask = clusters == g
        Xg = X[mask]
        ug = resid[mask]
        score_g = Xg.T @ ug
        meat += np.outer(score_g, score_g)

    k_params = X.shape[1]
    dof_correction = (n_groups / (n_groups - 1)) * ((n - 1) / (n - k_params))
    vcov = dof_correction * (XtX_inv @ meat @ XtX_inv)
    se = np.sqrt(np.diag(vcov))

    slope, slope_se = beta[1], se[1]
    df_resid = n_groups - 1  # cluster-robust t-test uses G-1 dof
    t_stat = slope / slope_se
    p_two_sided = 2 * stats.t.sf(abs(t_stat), df=df_resid)

    return {
        "intercept": float(beta[0]),
        "slope": float(slope),
        "slope_se": float(slope_se),
        "t_stat": float(t_stat),
        "df_resid": int(df_resid),
        "p_two_sided": float(p_two_sided),
        "n_obs": int(n),
        "n_clusters": int(n_groups),
    }


def condition_pooled_significance(df: pd.DataFrame) -> dict:
    fit = cluster_robust_ols(
        df["edge"].to_numpy(), df["k"].to_numpy(), df["day"].to_numpy()
    )
    passed = fit["slope"] < 0 and fit["p_two_sided"] < SIG_ALPHA
    return {**fit, "alpha": SIG_ALPHA, "passed": passed}


def condition_economic_size(df: pd.DataFrame) -> dict:
    # Terciles of k computed on the pooled sample, per the decision rule.
    tercile_edges = df["k"].quantile([1 / 3, 2 / 3]).to_numpy()
    top_tercile = df[df["k"] >= tercile_edges[1]]
    mean_abs_edge = float(np.abs(top_tercile["edge"].to_numpy()).mean())
    passed = mean_abs_edge > ECON_SIZE_THRESHOLD
    return {
        "k_tercile_cutoffs": tercile_edges.tolist(),
        "n_top_tercile": int(len(top_tercile)),
        "mean_abs_edge_top_tercile": mean_abs_edge,
        "threshold": ECON_SIZE_THRESHOLD,
        "passed": passed,
    }


def run(path: Path) -> None:
    df = load(path)
    n_days = df["day"].nunique()

    print(f"Data: {path}")
    print(f"Rows: {len(df)}, distinct days: {n_days}")
    print()

    c1 = condition_sign_stability(df)
    print("Condition 1 -- sign stability")
    print(f"  negative-slope days: {c1['n_negative']}/{c1['n_days_scored']} scored"
          f" (need >= {c1['threshold']})")
    print(f"  PASS" if c1["passed"] else "  FAIL")
    print()

    c2 = condition_pooled_significance(df)
    print("Condition 2 -- pooled significance (clustered by day)")
    print(f"  slope = {c2['slope']:.5f}, cluster-robust SE = {c2['slope_se']:.5f}")
    print(f"  t({c2['df_resid']}) = {c2['t_stat']:.3f}, two-sided p = {c2['p_two_sided']:.4f}")
    print(f"  n_obs = {c2['n_obs']}, n_clusters = {c2['n_clusters']}")
    print(f"  PASS" if c2["passed"] else "  FAIL")
    print()

    c3 = condition_economic_size(df)
    print("Condition 3 -- economic size")
    print(f"  top-tercile k cutoff: {c3['k_tercile_cutoffs'][1]:.4f}"
          f" (n={c3['n_top_tercile']})")
    print(f"  mean |edge| in top k-tercile = {c3['mean_abs_edge_top_tercile']:.4f}"
          f" (need > {c3['threshold']:.2f})")
    print(f"  PASS" if c3["passed"] else "  FAIL")
    print()

    verdict = c1["passed"] and c2["passed"] and c3["passed"]
    print(f"VERDICT: {'PASS -- rejects null, richness persists' if verdict else 'FAIL -- closed per pre-registration'}")


if __name__ == "__main__":
    default_path = Path(__file__).resolve().parents[1] / "data" / "harness" / "snapshots.csv"
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else default_path
    run(path)
