"""Pure-pandas helpers used by the Streamlit pages (no Streamlit imports => easy to test)."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import trends as trends_mod
from .skills_taxonomy import SKILL_CATEGORY

MODE_LABEL = {
    "A": "Mode A - Temporal (monthly trends)",
    "A_short": "Mode A-short - Weekly trends (weak evidence)",
    "B": "Mode B - Cross-sectional (no time trends)",
}
STATUS_COLOR = {"Emerging": "#34D399", "Declining": "#F87171", "Stable": "#9CA3C7",
                "Insufficient volume": "#4B5280"}
RISK_COLOR = {"High": "#F87171", "Moderate": "#FBBF24", "Low": "#34D399"}


def fmt_int(n) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return "-"


def role_options(art: dict) -> List[str]:
    rp = art.get("role_profile")
    return list(rp.index) if rp is not None and len(rp) else []


def skill_options(art: dict) -> List[str]:
    d = art.get("demand")
    if d is None or d.empty:
        return []
    return d.loc[d["postings"] > 0, "skill"].tolist()


def filter_mask(clean: pd.DataFrame, roles: List[str], levels: List[str]) -> np.ndarray:
    """Boolean row mask aligned with the skill matrix."""
    mask = np.ones(len(clean), dtype=bool)
    if roles:
        mask &= clean["role"].isin(roles).to_numpy()
    if levels:
        mask &= clean["level_group"].isin(levels).to_numpy()
    return mask


def jaccard_from_counts(cooc: pd.DataFrame) -> pd.DataFrame:
    """Co-occurrence counts -> Jaccard similarity (0-1); diagonal set to NaN."""
    c = cooc.to_numpy(dtype=float)
    diag = np.diag(c)
    union = diag[:, None] + diag[None, :] - c
    with np.errstate(divide="ignore", invalid="ignore"):
        j = np.where(union > 0, c / union, 0.0)
    np.fill_diagonal(j, np.nan)
    return pd.DataFrame(j, index=cooc.index, columns=cooc.columns)


def confusion_frame(result: dict, normalise: bool = False) -> pd.DataFrame:
    cm = np.array(result["confusion_matrix"], dtype=float)
    if normalise:
        cm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    classes = result["classes"]
    return pd.DataFrame(cm, index=classes, columns=classes)


def per_class_frame(result: dict) -> pd.DataFrame:
    rows = [{"role": k, "precision": v["precision"], "recall": v["recall"],
             "f1": v["f1-score"], "support": int(v["support"])}
            for k, v in result.get("per_class_report", {}).items()]
    return pd.DataFrame(rows).sort_values("f1", ascending=False).reset_index(drop=True) if rows else pd.DataFrame()


def cv_frame(result: dict) -> pd.DataFrame:
    return pd.DataFrame(result.get("cv_results", []))


def history_forecast(art: dict, skill: str, horizon: int = 3) -> Tuple[Optional[pd.Series], pd.DataFrame]:
    """History (share in %) and a live linear forecast for one skill (Mode A only)."""
    series = art.get("trend_series")
    freq = art["meta"].get("temporal", {}).get("freq")
    if series is None or skill not in series.columns or not freq:
        return None, pd.DataFrame()
    hist = series[skill] * 100
    fc = trends_mod.forecast_skill(series[skill], freq, horizon=horizon)
    return hist, fc


def recommend_future_skills(art: dict, user_skills: List[str], n: int = 10) -> Tuple[pd.DataFrame, str]:
    """Skills worth learning next. Basis depends on the analysis mode (always stated)."""
    user = set(user_skills)
    demand = art["demand"].set_index("skill") if art.get("demand") is not None else pd.DataFrame()
    mode = art["meta"].get("analysis_mode", "B")
    tt = art.get("trend_table")
    if mode in ("A", "A_short") and tt is not None and len(tt):
        sub = tt[(tt["status"] == "Emerging") & (~tt["skill"].isin(user))].head(n).copy()
        sub["category"] = sub["skill"].map(SKILL_CATEGORY)
        sub["demand_pct"] = sub["skill"].map(demand["share"] * 100) if len(demand) else np.nan
        sub["basis"] = "Statistically significant rise over time"
        return sub[["skill", "category", "demand_pct", "rel_change_per_period", "evidence", "basis"]], \
            "Emerging skills from time trends (evidence level shown)"
    cs = art.get("cross_sectional")
    if cs is not None and len(cs):
        sub = cs[(cs["signal"].str.startswith("Emerging")) & (~cs["skill"].isin(user))].head(n).copy()
        sub["category"] = sub["skill"].map(SKILL_CATEGORY)
        sub["demand_pct"] = sub["skill"].map(demand["share"] * 100) if len(demand) else np.nan
        sub["basis"] = "Junior-leaning proxy (NOT a time trend)"
        return sub[["skill", "category", "demand_pct", "log_ratio", "evidence", "basis"]], \
            "Cross-sectional proxy only: skills more common in junior than senior postings (weak evidence)"
    return pd.DataFrame(), "No trend or proxy information is available for this dataset."
