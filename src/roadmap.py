"""Learning roadmap built from the skill gap.

What is DATA-DRIVEN: which skills are missing, how much the target role demands them,
trend status, and which skills co-occur in postings (used for project ideas).
What is a HEURISTIC (and labelled so): phase ordering by tier/prerequisites and the
effort estimate in weeks. No course links are invented; only search topics are given.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from .skills_taxonomy import PREREQS, SKILL_CATEGORY, tier_of

PHASE_NAMES = {1: "Phase 1 - Foundation", 2: "Phase 2 - Core skills", 3: "Phase 3 - Advanced / specialisation"}
BASE_WEEKS = {1: 2.0, 2: 3.0, 3: 4.0}          # planning estimate at 10 h/week


def _expand_prereqs(targets: List[str], user: set) -> Dict[str, List[str]]:
    """Add unmet prerequisites. Returns {skill: [skills it is a prerequisite for]}."""
    reason: Dict[str, List[str]] = {}
    stack = list(targets)
    seen = set(targets)
    while stack:
        s = stack.pop()
        for p in PREREQS.get(s, []):
            if p in user:
                continue
            reason.setdefault(p, []).append(s)
            if p not in seen:
                seen.add(p)
                stack.append(p)
    return reason


def build_roadmap(missing_df: pd.DataFrame, user_skills: List[str], role: str,
                  cooc: Optional[pd.DataFrame] = None, max_skills: int = 12,
                  hours_per_week: int = 10) -> pd.DataFrame:
    """Return one row per skill, ordered by phase then dependency then priority."""
    if missing_df is None or missing_df.empty:
        return pd.DataFrame()
    user = set(user_skills)
    top = missing_df.head(max_skills)
    prio = dict(zip(top["skill"], top["priority_score"]))
    info = top.set_index("skill")
    prereq_for = _expand_prereqs(list(top["skill"]), user)
    plan = list(top["skill"]) + [p for p in prereq_for if p not in prio]

    depth: Dict[str, int] = {}

    def get_depth(s: str, guard=()) -> int:
        if s in depth:
            return depth[s]
        if s in guard:
            return 0
        deps = [p for p in PREREQS.get(s, []) if p in plan and p not in user]
        depth[s] = 0 if not deps else 1 + max(get_depth(p, guard + (s,)) for p in deps)
        return depth[s]

    scale = 10.0 / max(hours_per_week, 1)
    rows = []
    for s in plan:
        d = get_depth(s)
        phase = min(3, max(tier_of(s), d + 1))
        if s in prio:
            why = f"{info.loc[s, 'role_demand_pct']:.0f}% of {role} postings mention it"
            status = info.loc[s, "trend_status"]
            if status in ("Emerging", "Declining"):
                why += f"; trend: {status.lower()} ({info.loc[s, 'trend_evidence']} evidence)"
            if s in prereq_for:
                why += f"; also a prerequisite for {', '.join(prereq_for[s][:3])}"
            role_pct = float(info.loc[s, "role_demand_pct"])
            trend_status = status
        else:
            why = f"Prerequisite for {', '.join(prereq_for[s][:3])}"
            role_pct, trend_status = None, "n/a"
        partner = None
        if cooc is not None and s in cooc.index:
            pool = [c for c in cooc.columns if c != s and (c in user or c in plan)]
            if pool:
                partner = cooc.loc[s, pool].sort_values(ascending=False).index[0]
        practice = (f"Build a small project that combines {s} with {partner} "
                    f"(they co-occur in many postings)" if partner else f"Build a small hands-on project using {s}")
        rows.append({"phase": phase, "phase_name": PHASE_NAMES[phase], "skill": s,
                     "category": SKILL_CATEGORY.get(s, "Other"), "why": why,
                     "role_demand_pct": role_pct, "trend_status": trend_status,
                     "depth": d, "priority": prio.get(s, 0.0),
                     "effort_weeks_estimate": round(BASE_WEEKS[tier_of(s)] * scale * 2) / 2,
                     "practice_idea": practice,
                     "search_topic": f"{s} tutorial for beginners"})
    df = pd.DataFrame(rows).sort_values(["phase", "depth", "priority"], ascending=[True, True, False]) \
        .reset_index(drop=True)
    df["step"] = df.index + 1
    df["cumulative_weeks"] = df["effort_weeks_estimate"].cumsum()
    return df.drop(columns=["depth", "priority"])


def roadmap_to_markdown(roadmap: pd.DataFrame, role: str, hours_per_week: int = 10) -> str:
    if roadmap is None or roadmap.empty:
        return f"# Roadmap for {role}\n\nNo missing skills found - you already cover the in-demand skills in the data."
    lines = [f"# Learning roadmap: {role}", "",
             f"_Effort figures are planning estimates at {hours_per_week} h/week, not derived from data._", ""]
    for phase, grp in roadmap.groupby("phase"):
        lines.append(f"## {PHASE_NAMES[phase]}")
        for _, r in grp.iterrows():
            lines.append(f"- **{r['skill']}** ({r['category']}) - {r['why']}. "
                         f"~{r['effort_weeks_estimate']} weeks. Practice: {r['practice_idea']}.")
        lines.append("")
    lines.append(f"**Total estimated time:** ~{roadmap['effort_weeks_estimate'].sum():.1f} weeks.")
    return "\n".join(lines)
