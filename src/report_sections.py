"""Assemble report content ONCE; reused by the HTML, PDF and Excel exports.

Everything comes from the loaded artefacts (and the user's gap result if present).
No numbers are written by hand.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import pandas as pd

from .frontend_logic import MODE_LABEL, fmt_int
from .pdf_report import BarSpec

LIMITATIONS = (
    "1. Skills are found with a curated dictionary; skills outside the dictionary are not counted "
    "(see 'candidate terms' for suggestions).\n"
    "2. Role families come from job-title keyword rules; ambiguous titles become 'Other'.\n"
    "3. Time trends need enough time periods; with a narrow date window only a weak cross-sectional "
    "proxy is shown and NO forecast is produced.\n"
    "4. Skill-salary relationships are associations, not causal effects.\n"
    "5. Job postings measure employer demand, not the whole labour market.\n6. The Skill Market Risk Score is a heuristic composite indicator, NOT a statistically validated probability of skill obsolescence.\n7. Classification accuracy is measured against heuristic (rule-based) labels, not human-validated ground truth."
)


def _top(df: Optional[pd.DataFrame], n: int, cols: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df[[c for c in cols if c in df.columns]].head(n)


def _model_block(res: dict, kind: str):
    if not res or res.get("status") != "ok":
        reason = (res or {}).get("reason") or (res or {}).get("error") or "not available"
        return f"Not trained: {reason}"
    if kind == "role":
        return {"Best model": res["best_model"], "Test accuracy": f"{res['test_accuracy']:.3f}",
                "Test macro-F1": f"{res['test_macro_f1']:.3f}",
                "Baseline macro-F1": f"{res['baseline_macro_f1']:.3f}",
                "Beats baseline": "yes" if res.get("beats_baseline") else "NO",
                "Train / test rows": f"{res['n_train']:,} / {res['n_test']:,}"}
    tm, bm = res["test_metrics"], res["baseline_metrics"]
    return {"Best model": res["best_model"], "Test MAE": f"{tm['mae_usd']:,.0f}",
            "Baseline MAE": f"{bm['mae_usd']:,.0f}", "R2 (log salary)": f"{tm['r2_log']:.3f}",
            "Beats baseline": "yes" if res.get("beats_baseline") else "NO",
            "Rows with salary": f"{res['n_train'] + res['n_test']:,}"}


def build_sections(art: dict, gap: Optional[dict] = None,
                   roadmap: Optional[pd.DataFrame] = None) -> List[Tuple[str, object]]:
    meta = art["meta"]
    cr = meta.get("clean_report", {})
    s: List[Tuple[str, object]] = []
    s.append(("Dataset overview", {
        "Data source": meta.get("data_source"), "Postings analysed": fmt_int(meta.get("rows")),
        "Distinct skills detected": meta.get("unique_skills_found"),
        "Postings with >= 1 skill": f"{meta.get('postings_with_any_skill_pct')}%",
        "Date range": f"{meta.get('date_min')} to {meta.get('date_max')}",
        "Analysis mode": MODE_LABEL.get(meta.get("analysis_mode"), meta.get("analysis_mode")),
        "Processed at": meta.get("processed_at")}))
    s.append(("Data cleaning summary", {
        "Rows read": fmt_int(cr.get("rows_in")), "Dropped (no text)": fmt_int(cr.get("dropped_no_text")),
        "Dropped (duplicate job id)": fmt_int(cr.get("dropped_duplicate_job_id")),
        "Dropped (duplicate content)": fmt_int(cr.get("dropped_duplicate_content")),
        "Rows kept": fmt_int(cr.get("rows_out")), "With valid date": f"{cr.get('pct_with_valid_date')}%",
        "With usable salary": f"{cr.get('pct_with_salary')}%"}))
    d = art["demand"]
    top = d.head(20)
    s.append(("Top 20 in-demand skills (% of postings)",
              BarSpec(top["skill"].tolist(), (top["share"] * 100).round(2).tolist(), "Share of postings mentioning the skill (%)")))
    s.append(("Skill categories", art["category_demand"].assign(share=lambda x: (x["share"] * 100).round(2))
              .rename(columns={"share": "share_pct"}) if art.get("category_demand") is not None else pd.DataFrame()))
    s.append(("Temporal coverage", "\n".join(meta.get("temporal", {}).get("reasons", [])) or "n/a"))

    tt = art.get("trend_table")
    if meta.get("analysis_mode") in ("A", "A_short") and tt is not None and len(tt):
        cols = ["skill", "avg_share_pct", "rel_change_per_period", "q_value", "evidence"]
        s.append(("Emerging skills (FDR-corrected significance)", _top(tt[tt["status"] == "Emerging"], 12, cols)))
        s.append(("Declining skills (FDR-corrected significance)", _top(tt[tt["status"] == "Declining"].iloc[::-1], 12, cols)))
        bt = meta.get("backtest", {})
        if bt.get("status") == "ok":
            m = bt["mae_pct_points"]
            s.append(("Forecast back-test", {
                "Hold-out periods": bt["holdout_periods"], "Skills tested": bt["n_skills"],
                "MAE linear trend (pp)": f"{m['linear_trend']:.3f}", "MAE naive last value (pp)": f"{m['naive_last_value']:.3f}",
                "Linear beats naive": "yes" if bt["linear_beats_naive"] else "NO - treat forecasts cautiously"}))
    else:
        cs = art.get("cross_sectional")
        s.append(("Cross-sectional signals (WEAK evidence, not time trends)",
                  _top(cs, 15, ["skill", "junior_n", "senior_n", "log_ratio", "signal"]) if cs is not None else "Not available."))
    ob = art.get("obsolescence")
    s.append(("Skill Market Risk - highest scores (heuristic indicator, not obsolescence probability)", _top(ob, 15, ["skill", "category", "postings", "risk_score", "risk_level", "confidence_level", "evidence", "components_used"])))
    s.append(("Legacy watchlist (knowledge-based, NOT derived from the dataset)",
              _top(art.get("legacy"), 16, ["skill", "observed_postings", "reason"])))
    metrics = art.get("metrics", {})
    s.append(("ML model 1: role-family classifier", _model_block(metrics.get("role_classifier"), "role")))
    s.append(("ML model 2: salary regressor", _model_block(metrics.get("salary_model"), "salary")))

    if gap:
        r = gap["result"]
        s.append((f"Personal skill gap - target role: {r['target_role']}", {
            "Binary Coverage Score": f"{r['readiness_pct']:.1f}%",
            "Proficiency-Weighted Coverage": f"{r.get('proficiency_coverage_pct', r['readiness_pct']):.1f}%",
            "Coverage method": r.get("proficiency_mode", "binary"),
            "Your skills": ", ".join(gap["user_skills"]) or "-",
            "Required skills for role": r["n_required"], "Covered": r["n_have"], "Missing": r["n_missing"],
            "Coverage note": "Coverage measures skill presence/proficiency, not probability of employment."}))
        s.append(("Priority missing skills", _top(r["missing"], 15, ["skill", "category", "role_demand_pct", "trend_status", "priority_score"])))
    if roadmap is not None and len(roadmap):
        s.append(("Learning roadmap (effort = planning estimate)",
                  _top(roadmap, 20, ["step", "phase_name", "skill", "effort_weeks_estimate", "why"])))
    s.append(("Limitations", LIMITATIONS))
    return s


def sections_for_html(sections):
    """HTML builder does not know BarSpec -> convert to a table with text bars."""
    out = []
    for h, c in sections:
        if isinstance(c, BarSpec):
            vals = list(c.values)
            m = max(vals) if vals else 1
            c = pd.DataFrame({"skill": list(c.labels), "value": vals,
                              "bar": ["#" * max(1, int(round(v / m * 30))) for v in vals]})
        out.append((h, c))
    return out


def build_excel_sheets(art: dict, gap: Optional[dict] = None, roadmap: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
    sheets = {"Skill demand": art.get("demand"), "Category demand": art.get("category_demand"),
              "Trend table": art.get("trend_table"), "Cross-sectional": art.get("cross_sectional"),
              "Obsolescence risk": art.get("obsolescence"), "Legacy watchlist": art.get("legacy"),
              "Salary premium": art.get("salary_premium"), "Seniority skew": art.get("skew"),
              "Candidate skills": art.get("candidate_skills"), "Role profile": art.get("role_profile"),
              "Temporal counts": art.get("temporal_counts")}
    rp = art.get("role_profile")
    if rp is not None:
        sheets["Role profile"] = pd.concat([pd.Series(rp.index, index=rp.index, name="role"), rp], axis=1)
    if gap:
        sheets["Gap - missing"] = gap["result"]["missing"]
        sheets["Gap - strengths"] = gap["result"]["strengths"]
    if roadmap is not None:
        sheets["Roadmap"] = roadmap
    return {k: v for k, v in sheets.items() if v is not None}
