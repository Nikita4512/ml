"""Temporal gate, trend detection, forecasting and the Skill Market Risk Score.

HONESTY RULES built into this module
  * Forecasts/trends are produced ONLY if the data has enough time periods
    (assess_temporal decides: Mode A / A_short / B).
  * In Mode B (narrow date window) no time trend and no forecast is produced.
    Only a cross-sectional proxy (junior- vs senior-leaning) is available,
    always labelled with weak evidence.
  * The legacy watchlist is knowledge-based and clearly labelled as such.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix

from .config import (MIN_MONTHLY_PERIODS, MIN_POSTS_PER_PERIOD, MIN_WEEKLY_PERIODS,
                     OBSOLESCENCE_WEIGHTS, TREND_ALPHA, TREND_MIN_AVG_SHARE, TREND_MIN_REL_SLOPE)
from .skills_taxonomy import LEGACY_WATCHLIST

EVIDENCE_ORDER = {"Weak": 0, "Moderate": 1, "Strong": 2}

# ---------------------------------------------------------------------------
# Module-level methodology constant for the Skill Market Risk Score
RISK_SCORE_METHODOLOGY = (
    "Skill Market Risk Score methodology (heuristic - NOT a validated obsolescence probability):\n"
    "  * demand component (25%): lower demand share -> higher risk\n"
    "  * trend component (50%): declining share over time -> higher risk (requires temporal data)\n"
    "  * seniority component (25%): senior-leaning skill distribution -> higher risk\n"
    "Missing components are dropped and remaining weights renormalised.\n"
    "Score range: 0 (low risk) to 100 (high risk).\n"
    "Confidence levels:\n"
    "  High confidence: all 3 components available, strong/moderate evidence\n"
    "  Medium confidence: 2 components, or moderate evidence only\n"
    "  Low confidence: 1 component, or weak evidence\n"
    "  Insufficient data: score could not be computed"
)


# ---------------------------------------------------------------------------
def assess_temporal(dates: pd.Series, min_monthly: int = MIN_MONTHLY_PERIODS,
                    min_weekly: int = MIN_WEEKLY_PERIODS,
                    min_posts: int = MIN_POSTS_PER_PERIOD) -> dict:
    """Decide whether the dataset supports time-based analysis."""
    d = dates.dropna()
    out = {"mode": "B", "freq": None, "evidence_cap": "Weak", "n_valid_dates": int(len(d)),
           "reasons": [], "period_counts": pd.DataFrame(columns=["period", "postings"])}
    if len(d) == 0:
        out["reasons"].append("No valid listing dates were found.")
        return out
    out["date_min"], out["date_max"] = str(d.min().date()), str(d.max().date())
    out["span_days"] = int((d.max() - d.min()).days)
    monthly = d.dt.to_period("M").value_counts().sort_index()
    weekly = d.dt.to_period("W").value_counts().sort_index()
    q_m, q_w = int((monthly >= min_posts).sum()), int((weekly >= min_posts).sum())
    out["qualifying_months"], out["qualifying_weeks"] = q_m, q_w
    out["period_counts"] = monthly.rename("postings").rename_axis("period").reset_index() \
        .assign(period=lambda x: x["period"].astype(str))
    if q_m >= min_monthly:
        out.update(mode="A", freq="M", evidence_cap="Strong" if q_m >= 12 else "Moderate")
        out["reasons"].append(f"{q_m} months have >= {min_posts} postings: monthly trends are possible.")
    elif q_w >= min_weekly:
        out.update(mode="A_short", freq="W", evidence_cap="Weak")
        out["reasons"].append(
            f"Only {q_m} qualifying month(s) but {q_w} qualifying weeks: short-horizon weekly trends "
            "are shown with WEAK evidence (a few weeks can reflect scraping artefacts, not real change).")
    else:
        out["reasons"].append(
            f"Only {q_m} month(s) / {q_w} week(s) have >= {min_posts} postings (need {min_monthly} months "
            f"or {min_weekly} weeks). Time trends and forecasts are DISABLED to avoid fabricated results.")
    return out


# ---------------------------------------------------------------------------
def skill_share_series(X: csr_matrix, skill_names: List[str], dates: pd.Series, freq: str,
                       min_posts: int = MIN_POSTS_PER_PERIOD, top_n: int = 80,
                       min_postings: int = 30) -> Tuple[pd.DataFrame, pd.Series]:
    """Share of postings per period mentioning each skill (top_n skills only)."""
    valid = dates.notna().to_numpy()
    period = dates.dt.to_period(freq)
    codes, uniques = pd.factorize(period[valid], sort=True)
    n_periods = len(uniques)
    Xv = X[valid]
    rows = np.arange(len(codes))
    from scipy.sparse import coo_matrix
    P = coo_matrix((np.ones(len(codes), dtype=np.float32), (codes, rows)),
                   shape=(n_periods, len(codes))).tocsr()
    counts = np.asarray((P @ Xv.astype(np.float32)).todense())
    n_post = np.asarray(P.sum(axis=1)).ravel()
    keep_p = n_post >= min_posts
    overall = np.asarray(X.sum(axis=0)).ravel()
    order = [j for j in np.argsort(-overall) if overall[j] >= min_postings][:top_n]
    share = counts[keep_p][:, order] / n_post[keep_p][:, None]
    index = pd.PeriodIndex(uniques[keep_p]).to_timestamp()
    series = pd.DataFrame(share, index=index, columns=[skill_names[j] for j in order])
    return series, pd.Series(n_post[keep_p].astype(int), index=index, name="postings")


def _bh_adjust(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction (many skills are tested at once)."""
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.minimum(ranked, 1.0)
    return out


def fit_trends(series: pd.DataFrame, evidence_cap: str = "Weak") -> pd.DataFrame:
    """Linear trend of each skill's share over time + FDR-corrected significance."""
    n = len(series)
    if n < 4:
        return pd.DataFrame()
    x = np.arange(n)
    rows = []
    for skill in series.columns:
        y = series[skill].to_numpy() * 100            # percentage points
        res = stats.linregress(x, y)
        mean = float(y.mean())
        rel = res.slope / mean if mean > 0 else 0.0
        rows.append((skill, mean, res.slope, rel, res.rvalue ** 2, res.pvalue))
    df = pd.DataFrame(rows, columns=["skill", "avg_share_pct", "slope_pp_per_period",
                                     "rel_change_per_period", "r2", "p_value"])
    df["q_value"] = _bh_adjust(df["p_value"].fillna(1.0).to_numpy())
    cap = EVIDENCE_ORDER.get(evidence_cap, 0)

    def label(r):
        if r.avg_share_pct / 100 < TREND_MIN_AVG_SHARE:
            return "Insufficient volume"
        if r.q_value < TREND_ALPHA and r.rel_change_per_period >= TREND_MIN_REL_SLOPE:
            return "Emerging"
        if r.q_value < TREND_ALPHA and r.rel_change_per_period <= -TREND_MIN_REL_SLOPE:
            return "Declining"
        return "Stable"

    def evidence(r):
        level = 2 if (r.q_value < 0.01 and n >= 12) else 1 if r.q_value < TREND_ALPHA else 0
        return [k for k, v in EVIDENCE_ORDER.items() if v == min(level, cap)][0]

    df["status"] = df.apply(label, axis=1)
    df["evidence"] = df.apply(evidence, axis=1)
    df["n_periods"] = n
    return df.sort_values("rel_change_per_period", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
def _linear_forecast(y: np.ndarray, horizon: int, level: float = 0.90):
    n = len(y)
    x = np.arange(n)
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    s = np.sqrt(((y - fitted) ** 2).sum() / max(n - 2, 1))
    xf = np.arange(n, n + horizon)
    mean = intercept + slope * xf
    sxx = ((x - x.mean()) ** 2).sum()
    se = s * np.sqrt(1 + 1 / n + (xf - x.mean()) ** 2 / max(sxx, 1e-9))
    t = stats.t.ppf(0.5 + level / 2, max(n - 2, 1))
    return mean, mean - t * se, mean + t * se


def forecast_skill(series: pd.Series, freq: str, horizon: int = 3, level: float = 0.90,
                   backtest_result: dict | None = None) -> pd.DataFrame:
    """Forecast skill demand share (percent of postings) over `horizon` periods.

    Model selection:
      * If a backtest result with best_model is available, use that model.
      * Otherwise, default to linear_trend.
    Exponential smoothing is only used as best if data is long enough (n >= holdout+10).
    All output values are clipped to [0, 100] (valid demand share range).
    Returns an empty DataFrame with a 'reason' column when history is insufficient (n < 6).
    """
    y = series.to_numpy() * 100
    n = len(y)

    if n < 6:
        return pd.DataFrame({"reason": ["Insufficient history"]})

    # --- determine which model to use ---
    best_model = None
    model_selected_by = "default (insufficient backtest data)"
    if backtest_result and backtest_result.get("status") == "ok":
        best_model = backtest_result.get("best_model")
        model_selected_by = "backtest MAE comparison"

    use_ets = best_model == "exp_smoothing"
    forecast_method = "exp_smoothing" if use_ets else "linear_trend"

    # Convert DatetimeIndex freq aliases to Period-compatible aliases
    # (e.g. 'MS' -> 'M', 'QS' -> 'Q')
    period_freq = freq
    _PERIOD_ALIAS = {"MS": "M", "QS": "Q", "YS": "Y", "AS": "Y"}
    if isinstance(freq, str) and freq.upper() in _PERIOD_ALIAS:
        period_freq = _PERIOD_ALIAS[freq.upper()]

    last = series.index[-1].to_period(period_freq)
    future = pd.period_range(last + 1, periods=horizon, freq=period_freq).to_timestamp()

    if use_ets:
        mean_vals = _ets_forecast(y, horizon)
        # For ETS we don't have a closed-form PI; use linear PI as a heuristic
        _, lo, hi = _linear_forecast(y, horizon, level)
        # Shift interval to be centred on ETS mean (preserve width)
        lin_mean, _, _ = _linear_forecast(y, horizon, level)
        offset = mean_vals - lin_mean
        lo = lo + offset
        hi = hi + offset
    else:
        mean_vals, lo, hi = _linear_forecast(y, horizon, level)

    forecast_pct = np.clip(mean_vals, 0.0, 100.0)
    lower_pct    = np.clip(lo,         0.0, 100.0)
    upper_pct    = np.clip(hi,         0.0, 100.0)

    return pd.DataFrame({
        "period":           future,
        "forecast_pct":     forecast_pct,
        "lower_pct":        lower_pct,
        "upper_pct":        upper_pct,
        "forecast_method":  forecast_method,
        "model_selected_by": model_selected_by,
    })


def _ets_forecast(train: np.ndarray, horizon: int, alpha: float = 0.3) -> np.ndarray:
    """Simple exponential smoothing (level-only ETS) forecast.

    s_t = alpha * y_t + (1 - alpha) * s_{t-1}
    Forecast for all future periods = last smoothed value (flat ETS forecast).
    """
    s = train[0]
    for yt in train[1:]:
        s = alpha * yt + (1.0 - alpha) * s
    return np.full(horizon, s)


def backtest_forecast(series: pd.DataFrame, holdout: int = 2) -> dict:
    """Honest holdout check: train on all but the last `holdout` periods, compare four models.

    Models: linear_trend, naive_last_value, mean_last_3, exp_smoothing (ETS).
    Metrics: MAE, RMSE, sMAPE per model.
    Validation is chronological: the most recent `holdout` periods are withheld as a test set.
    """
    n = len(series)
    if n < holdout + 5:
        return {"status": "skipped", "reason": f"Need >= {holdout + 5} periods, have {n}."}

    model_keys = ["linear_trend", "naive_last_value", "mean_last_3"]
    use_ets = n >= holdout + 10
    if use_ets:
        model_keys.append("exp_smoothing")

    abs_errs: dict = {k: [] for k in model_keys}
    sq_errs: dict  = {k: [] for k in model_keys}
    smape_vals: dict = {k: [] for k in model_keys}

    for skill in series.columns:
        y = series[skill].to_numpy() * 100
        train, test = y[:-holdout], y[-holdout:]

        preds = {
            "linear_trend":    _linear_forecast(train, holdout)[0],
            "naive_last_value": np.full(holdout, train[-1]),
            "mean_last_3":      np.full(holdout, train[-3:].mean()),
        }
        if use_ets:
            preds["exp_smoothing"] = _ets_forecast(train, holdout)

        for k, pred in preds.items():
            ae = np.abs(pred - test)
            abs_errs[k].append(ae.mean())
            sq_errs[k].append((ae ** 2).mean())
            # sMAPE: 200 * |actual - forecast| / (|actual| + |forecast| + eps)
            smape = (200.0 * ae / (np.abs(test) + np.abs(pred) + 1e-8)).mean()
            smape_vals[k].append(smape)

    mae  = {k: float(np.mean(v)) for k, v in abs_errs.items()}
    rmse = {k: float(np.sqrt(np.mean(v))) for k, v in sq_errs.items()}
    smape_mean = {k: float(np.mean(v)) for k, v in smape_vals.items()}

    best_model = min(mae, key=mae.__getitem__)
    n_training = n - holdout

    return {
        "status": "ok",
        "holdout_periods": holdout,
        "n_skills": len(series.columns),
        "validation_method": f"walk-forward holdout (last {holdout} periods withheld)",
        "n_training_periods": n_training,
        "mae_pct_points": mae,
        "rmse_pct_points": rmse,
        "smape_pct": smape_mean,
        "best_model": best_model,
        "linear_beats_naive": bool(mae["linear_trend"] < mae["naive_last_value"]),
        "predictions_clipped": True,
    }


# ---------------------------------------------------------------------------
def obsolescence_table(demand: pd.DataFrame, trend: Optional[pd.DataFrame] = None,
                       skew: Optional[pd.DataFrame] = None, min_postings: int = 30,
                       weights: Dict[str, float] = OBSOLESCENCE_WEIGHTS,
                       evidence_cap: str = "Weak") -> pd.DataFrame:
    """Skill Market Risk Score (0-100 heuristic) from up to three components. NOT a validated obsolescence probability.

    trend  : decline in share over time (only if a temporal mode is available)
    demand : low overall share of postings -> higher risk
    skew   : senior-leaning skills (vs junior) -> higher risk
    Missing components are dropped and the remaining weights renormalised.
    The Legacy-watchlist flag is shown but NOT part of the Skill Market Risk Score.
    """
    base = demand[demand["postings"] >= min_postings][["skill", "category", "postings", "share"]].copy()
    if base.empty:
        return base
    # demand component: lower share => higher risk
    base["demand_c"] = 1 - base["share"].rank(pct=True)
    base["trend_c"] = np.nan
    base["skew_c"] = np.nan
    base["trend_evidence"] = None
    if trend is not None and len(trend):
        # trend component: declining share over time => higher risk
        t = trend.set_index("skill")
        rel = t["rel_change_per_period"].clip(-0.1, 0.1)
        base["trend_c"] = base["skill"].map((0.1 - rel) / 0.2)
        base["trend_evidence"] = base["skill"].map(t["evidence"])
    if skew is not None and len(skew):
        # seniority component: senior-leaning => higher risk
        s = skew.set_index("skill")["log_ratio"].clip(-2, 2)
        base["skew_c"] = base["skill"].map((2 - s) / 4)

    comps = {"trend": "trend_c", "demand": "demand_c", "skew": "skew_c"}
    num = np.zeros(len(base))
    den = np.zeros(len(base))
    used = [[] for _ in range(len(base))]
    for name, col in comps.items():
        v = base[col].to_numpy(dtype=float)
        ok = ~np.isnan(v)
        num[ok] += weights[name] * v[ok]
        den[ok] += weights[name]
        for i in np.where(ok)[0]:
            used[i].append(name)
    base["risk_score"] = np.where(den > 0, num / np.where(den > 0, den, 1) * 100, np.nan)
    base["risk_level"] = pd.cut(base["risk_score"], [-1, 45, 65, 101], labels=["Low", "Moderate", "High"])
    base["components_used"] = [", ".join(u) for u in used]
    base["evidence"] = np.where(base["trend_evidence"].notna(), base["trend_evidence"], "Weak")
    base["legacy_watchlist_flag"] = base["skill"].isin(LEGACY_WATCHLIST)

    # --- sample_size: alias for postings, for display clarity
    base["sample_size"] = base["postings"]

    # --- methodology_note: human-readable explanation attached to every row
    base["methodology_note"] = (
        "Skill Market Risk Score (heuristic): weighted combination of demand rank, "
        "demand trend decline, and seniority distribution. "
        "NOT a statistically validated obsolescence probability."
    )

    # --- confidence_level based on components_used and evidence quality
    def _confidence(row):
        if pd.isna(row["risk_score"]):
            return "Insufficient data"
        n_comp = len([c for c in row["components_used"].split(", ") if c]) if row["components_used"] else 0
        ev = str(row["evidence"])
        if n_comp == 3 and ev in ("Strong", "Moderate"):
            return "High confidence"
        if n_comp == 2 or ev == "Moderate":
            return "Medium confidence"
        if n_comp == 1 or ev == "Weak":
            return "Low confidence"
        return "Insufficient data"

    base["confidence_level"] = base.apply(_confidence, axis=1)

    # --- For rows where ALL components yield NaN (risk_score is NaN),
    #     mark evidence as INSUFFICIENT TEMPORAL EVIDENCE if no trend evidence exists
    nan_mask = base["risk_score"].isna()
    no_trend_evidence_mask = base["trend_evidence"].isna()
    base.loc[nan_mask & no_trend_evidence_mask, "evidence"] = "INSUFFICIENT TEMPORAL EVIDENCE"
    base.loc[nan_mask, "confidence_level"] = "Insufficient data"

    return base.sort_values("risk_score", ascending=False).reset_index(drop=True)


def legacy_watchlist_table(demand: pd.DataFrame) -> pd.DataFrame:
    """Knowledge-based legacy list joined with observed demand from the dataset."""
    d = demand.set_index("skill") if len(demand) else pd.DataFrame()
    rows = []
    for skill, reason in LEGACY_WATCHLIST.items():
        postings = int(d.loc[skill, "postings"]) if skill in d.index else 0
        share = float(d.loc[skill, "share"]) if skill in d.index else 0.0
        rows.append((skill, reason, postings, share, "Knowledge-based (not derived from dataset)"))
    return pd.DataFrame(rows, columns=["skill", "reason", "observed_postings", "observed_share", "basis"]) \
        .sort_values("observed_postings", ascending=False).reset_index(drop=True)


def cross_sectional_signals(skew: Optional[pd.DataFrame], min_total: int = 30) -> pd.DataFrame:
    """Mode B proxy: junior-leaning ~ 'newer demand', senior-leaning ~ 'established/legacy'.

    This is NOT a time trend. Evidence is always Weak.
    """
    if skew is None or skew.empty:
        return pd.DataFrame()
    df = skew[(skew["junior_n"] + skew["senior_n"]) >= min_total].copy()
    df["signal"] = np.where(df["lean"] == "junior-leaning", "Emerging proxy (junior-leaning)",
                            np.where(df["lean"] == "senior-leaning", "Established/legacy-risk proxy (senior-leaning)",
                                     "Balanced"))
    df["evidence"] = "Weak"
    return df[["skill", "junior_n", "senior_n", "log_ratio", "signal", "evidence"]].reset_index(drop=True)
