"""Personalised skill-gap analysis.

Inputs come from real dataset artefacts (role profile, demand, trends, obsolescence).
The user only supplies their own skills and a target role.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .skill_extractor import SkillExtractor

TREND_MULTIPLIER = {"Emerging": 1.25, "Declining": 0.75}

PROFICIENCY_LEVELS = {"Beginner": 25, "Intermediate": 50, "Advanced": 75, "Expert": 100}
PROFICIENCY_NOTE = (
    "Binary Coverage Score: % of role-required skills the user possesses (regardless of proficiency). "
    "Proficiency-Weighted Coverage: weighted by role demand and self-reported proficiency level (0-100). "
    "Neither metric is a probability of employment or a validated job-readiness score."
)


def parse_user_skills(text: str, extractor: SkillExtractor) -> Dict[str, List[str]]:
    """Turn free text ("python, pandas; sql\\nexcel") into canonical skills.

    Returns {"recognised": [...], "unrecognised": [...]}.
    """
    recognised = extractor.extract_text(text or "")
    unknown = []
    for item in re.split(r"[,;\n/|]+", text or ""):
        item = item.strip()
        if item and not extractor.extract_text(item):
            unknown.append(item)
    return {"recognised": sorted(recognised), "unrecognised": unknown}


def best_fit_roles(user_skills: List[str], role_profile: pd.DataFrame) -> pd.DataFrame:
    """Cosine similarity between the user's skill vector and each role's demand profile."""
    if role_profile.empty:
        return pd.DataFrame(columns=["role", "similarity"])
    u = np.array([1.0 if s in set(user_skills) else 0.0 for s in role_profile.columns])
    if u.sum() == 0:
        return pd.DataFrame({"role": role_profile.index, "similarity": 0.0})
    P = role_profile.to_numpy()
    sim = (P @ u) / (np.linalg.norm(P, axis=1) * np.linalg.norm(u) + 1e-12)
    return pd.DataFrame({"role": role_profile.index, "similarity": sim}) \
        .sort_values("similarity", ascending=False).reset_index(drop=True)


def compute_gap(user_skills: List[str], target_role: str, role_profile: pd.DataFrame,
                demand: pd.DataFrame, categories: Dict[str, str],
                trend: Optional[pd.DataFrame] = None, min_required_share: float = 0.05,
                proficiency: Optional[Dict[str, int]] = None) -> dict:
    """Readiness score, strengths and a prioritised list of missing skills.

    Parameters
    ----------
    proficiency : dict, optional
        Mapping of skill name (lowercase) -> self-reported proficiency score 0-100.
        Proficiency levels:
            Beginner     -> 25
            Intermediate -> 50
            Advanced     -> 75
            Expert       -> 100
        When provided, ``proficiency_coverage_pct`` weights each skill by both role
        demand and the user's proficiency fraction. When None, proficiency_coverage_pct
        equals binary_coverage_pct (same as the legacy readiness_pct).
    """
    if role_profile is None or target_role not in role_profile.index:
        raise ValueError(f"Role '{target_role}' has no profile (not enough postings in the dataset).")
    user = set(user_skills)
    prof = role_profile.loc[target_role]
    required = prof[prof >= min_required_share].sort_values(ascending=False)
    if required.empty:                                   # tiny datasets: fall back to top 10 skills
        required = prof[prof > 0].sort_values(ascending=False).head(10)
    total = float(required.sum()) or 1.0

    have = [s for s in required.index if s in user]
    miss = [s for s in required.index if s not in user]
    # Binary readiness: demand-weighted fraction of required skills the user possesses
    binary_coverage_pct = float(required[have].sum() / total * 100)

    # ------------------------------------------------------------------
    # Proficiency-weighted coverage
    # ------------------------------------------------------------------
    proficiency_mode: str
    if proficiency is None:
        proficiency_coverage_pct = binary_coverage_pct
        proficiency_mode = "binary"
    else:
        weighted_sum = 0.0
        for s, w in required.items():
            if s in user:
                fraction = proficiency.get(s.lower(), 100) / 100.0
            else:
                fraction = 0.0
            weighted_sum += float(w) * fraction
        proficiency_coverage_pct = weighted_sum / total * 100
        proficiency_mode = "proficiency_weighted"

    # Skills the user has but reported proficiency < 50 (needs improvement)
    skills_with_low_proficiency: List[str] = []
    if proficiency is not None:
        skills_with_low_proficiency = [
            s for s in have
            if proficiency.get(s.lower(), 100) < 50
        ]

    overall = demand.set_index("skill")["share"] if len(demand) else pd.Series(dtype=float)
    tmap = trend.set_index("skill")["status"] if trend is not None and len(trend) else pd.Series(dtype=object)
    evmap = trend.set_index("skill")["evidence"] if trend is not None and len(trend) else pd.Series(dtype=object)

    rows = []
    for s in miss:
        status = tmap.get(s, "n/a")
        mult = TREND_MULTIPLIER.get(status, 1.0)
        rows.append({"skill": s, "category": categories.get(s, "Other"),
                     "role_demand_pct": float(required[s] * 100),
                     "overall_demand_pct": float(overall.get(s, 0) * 100),
                     "trend_status": status, "trend_evidence": evmap.get(s, "n/a"),
                     "priority_score": float(required[s] * 100 * mult)})
    missing_df = pd.DataFrame(rows).sort_values("priority_score", ascending=False).reset_index(drop=True) \
        if rows else pd.DataFrame(columns=["skill", "category", "role_demand_pct", "overall_demand_pct",
                                           "trend_status", "trend_evidence", "priority_score"])
    strengths_df = pd.DataFrame({"skill": have, "category": [categories.get(s, "Other") for s in have],
                                 "role_demand_pct": [float(required[s] * 100) for s in have]})

    # coverage by category (weighted by role demand)
    cat_rows = {}
    for s, w in required.items():
        c = categories.get(s, "Other")
        d = cat_rows.setdefault(c, [0.0, 0.0])
        d[1] += float(w)
        if s in user:
            d[0] += float(w)
    cat_cov = pd.DataFrame([(c, v[0] / v[1] * 100 if v[1] else 0.0, v[1]) for c, v in cat_rows.items()],
                           columns=["category", "coverage_pct", "weight"]).sort_values("weight", ascending=False)

    return {
        "target_role": target_role,
        # Legacy alias — preserved for full backward compatibility
        "readiness_pct": binary_coverage_pct,
        # Named fields
        "binary_coverage_pct": binary_coverage_pct,
        "proficiency_coverage_pct": proficiency_coverage_pct,
        "proficiency_mode": proficiency_mode,
        "skills_with_low_proficiency": skills_with_low_proficiency,
        "n_required": len(required),
        "n_have": len(have),
        "n_missing": len(miss),
        "strengths": strengths_df,
        "missing": missing_df,
        "category_coverage": cat_cov.reset_index(drop=True),
        "extra_skills": sorted(user - set(required.index)),
        "coverage_note": PROFICIENCY_NOTE,
    }


def at_risk_user_skills(user_skills: List[str], obsolescence: Optional[pd.DataFrame]) -> pd.DataFrame:
    """The user's own skills that score Moderate/High Skill Market Risk or are on the legacy watchlist."""
    if obsolescence is None or obsolescence.empty:
        return pd.DataFrame()
    sub = obsolescence[obsolescence["skill"].isin(user_skills)]
    sub = sub[(sub["risk_level"].astype(str).isin(["Moderate", "High"])) | sub["legacy_watchlist_flag"]]
    cols = ["skill", "risk_score", "risk_level", "confidence_level", "evidence", "components_used",
            "legacy_watchlist_flag"]
    return sub[[c for c in cols if c in sub.columns]].reset_index(drop=True)
