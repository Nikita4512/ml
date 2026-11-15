"""Interactive personalised skill-gap analyzer."""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src import ui_components as ui
from src.frontend_logic import role_options, skill_options
from src.gap_analysis import at_risk_user_skills, best_fit_roles, compute_gap, parse_user_skills
from src.reports import df_to_csv_bytes
from src.skill_extractor import SkillExtractor
from src.skills_taxonomy import SKILL_CATEGORY


def _analyse(art, kind, picked, text, role, min_share, proficiency=None):
    parsed = parse_user_skills(text, SkillExtractor())
    user_skills = sorted(set(picked) | set(parsed["recognised"]))
    if not user_skills:
        st.warning("Please select or type at least one skill you already have.")
        return
    try:
        res = compute_gap(user_skills, role, art["role_profile"], art["demand"], SKILL_CATEGORY,
                          art.get("trend_table"), min_required_share=min_share,
                          proficiency=proficiency if proficiency else None)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.session_state["gap_result"] = {
        "kind": kind, "role": role, "user_skills": user_skills, "unrecognised": parsed["unrecognised"],
        "result": res, "fit": best_fit_roles(user_skills, art["role_profile"]),
        "risk": at_risk_user_skills(user_skills, art.get("obsolescence")), "min_share": min_share,
        "proficiency": proficiency if proficiency else None}


def render(art, kind):
    ui.page_header("Personalised Skill Gap Analyzer", "Compare your skills with what employers ask for in your target role")
    if art is None:
        ui.no_data(kind)
        return
    ui.source_banner(art["meta"])
    roles, options = role_options(art), skill_options(art)
    if not roles:
        st.warning("No role family had enough postings for a profile, so a gap analysis is not possible with this dataset. "
                   "Try processing more rows.")
        return

    c1, c2 = st.columns([3, 2])
    picked = c1.multiselect("Skills you already have", options, key=f"gap_ms_{kind}")
    text = c1.text_area("...or type them (comma separated, e.g. python, pandas, power bi)", key=f"gap_txt_{kind}", height=80)
    role = c2.selectbox("Target role family", roles, key=f"gap_role_{kind}")
    min_share = c2.slider("A skill counts as 'required' if at least this % of the role's postings mention it",
                          1, 30, 5, key=f"gap_min_{kind}") / 100

    # ------------------------------------------------------------------
    # Optional proficiency expander (shown only when skills are selected)
    # ------------------------------------------------------------------
    _PROF_OPTIONS = ["Beginner (25)", "Intermediate (50)", "Advanced (75)", "Expert (100)"]
    _PROF_SCORES = {"Beginner (25)": 25, "Intermediate (50)": 50, "Advanced (75)": 75, "Expert (100)": 100}

    proficiency_dict: dict = {}
    # Collect recognised skills from free text as well for the expander
    from src.skill_extractor import SkillExtractor as _SE
    _all_typed = sorted(set(_SE().extract_text(text or "")))
    _all_selected = sorted(set(picked) | set(_all_typed))

    if _all_selected:
        with st.expander("🎯 Set your proficiency level for each skill (optional)", expanded=False):
            st.caption("Leave as **Expert** if you are unsure — this gives the same result as the binary score.")
            _cols = st.columns(min(3, len(_all_selected)))
            for _i, _skill in enumerate(_all_selected):
                _col = _cols[_i % len(_cols)]
                _sel = _col.selectbox(
                    f"{_skill}",
                    _PROF_OPTIONS,
                    index=3,  # default: Expert (100)
                    key=f"gap_prof_{kind}_{_skill}"
                )
                proficiency_dict[_skill.lower()] = _PROF_SCORES[_sel]

    if st.button("Analyse my skill gap", key=f"gap_go_{kind}", type="primary"):
        _prof_arg = proficiency_dict if any(v < 100 for v in proficiency_dict.values()) else None
        _analyse(art, kind, picked, text, role, min_share, proficiency=_prof_arg)

    g = st.session_state.get("gap_result")
    if not g or g["kind"] != kind:
        st.info("Choose your skills and a target role, then press **Analyse my skill gap**.")
        return
    res = g["result"]
    if g["unrecognised"]:
        st.caption("Not recognised as known skills (ignored): " + ", ".join(g["unrecognised"]))

    st.subheader(f"Result for: {res['target_role']}")
    left, right = st.columns([1, 2])
    with left:
        gauge = go.Figure(go.Indicator(mode="gauge+number", value=round(res["readiness_pct"], 1),
                                       number={"suffix": "%"}, title={"text": "Skill Coverage"},
                                       gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#7C3AED"},
                                              "steps": [{"range": [0, 40], "color": "rgba(248,113,113,0.25)"},
                                                        {"range": [40, 70], "color": "rgba(251,191,36,0.25)"},
                                                        {"range": [70, 100], "color": "rgba(52,211,153,0.25)"}]}))
        ui.show_fig(ui.style_fig(gauge, 280), key=f"gap_gauge_{kind}")
    with right:
        ui.kpi_row([("Required skills", str(res["n_required"]), f">= {g['min_share']:.0%} of role postings"),
                    ("You have", str(res["n_have"]), "of the required skills"),
                    ("Missing", str(res["n_missing"]), "to close the gap")])
        st.caption("Binary Skill Coverage = % of role-required skills you possess (demand-weighted). "
                      "NOT a probability of employment or validated job-readiness score.")
        if res.get("proficiency_mode") == "proficiency_weighted":
            st.metric("Proficiency-Weighted Coverage",
                      f"{res['proficiency_coverage_pct']:.1f}%",
                      help="Weighted by role demand and self-reported proficiency level (0-100).")
        if res.get("skills_with_low_proficiency"):
            st.warning("Skills needing proficiency improvement (below 50): " +
                       ", ".join(res["skills_with_low_proficiency"]))

    tabs = st.tabs(["Missing skills", "Strengths", "Coverage by category", "Best-fit roles", "At-risk skills you hold"])
    with tabs[0]:
        m = res["missing"]
        if m.empty:
            st.success("You already cover every in-demand skill for this role in the data.")
        else:
            top = m.head(15).iloc[::-1]
            ui.show_fig(ui.style_fig(px.bar(top, x="priority_score", y="skill", color="category", orientation="h",
                                            labels={"priority_score": "Priority (role demand x trend factor)", "skill": ""}),
                                     max(360, 24 * len(top) + 120), "Skills to learn first"), key=f"gap_miss_{kind}")
            ui.show_df(m, key=f"gap_miss_tbl_{kind}", height=300)
            st.download_button("Download missing skills (CSV)", df_to_csv_bytes(m), "missing_skills.csv", "text/csv",
                               key=f"gap_dl_{kind}")
    with tabs[1]:
        s = res["strengths"]
        ui.show_df(s, key=f"gap_str_{kind}") if len(s) else st.info("None of your skills are in this role's required set.")
        if res["extra_skills"]:
            st.caption("Other skills you listed (not in this role's required set): " + ", ".join(res["extra_skills"]))
    with tabs[2]:
        cc = res["category_coverage"]
        if len(cc) >= 3:
            fig = go.Figure(go.Scatterpolar(r=cc["coverage_pct"], theta=cc["category"], fill="toself", name="Your coverage"))
            fig.update_layout(polar=dict(radialaxis=dict(range=[0, 100])))
            ui.show_fig(ui.style_fig(fig, 460, "Coverage of required skills by category (%)"), key=f"gap_radar_{kind}")
        elif len(cc):
            ui.show_fig(ui.style_fig(px.bar(cc, x="category", y="coverage_pct"), 360), key=f"gap_cat_{kind}")
        else:
            st.info("No category data.")
    with tabs[3]:
        fit = g["fit"]
        ui.show_fig(ui.style_fig(px.bar(fit.iloc[::-1], x="similarity", y="role", orientation="h"), 400,
                                 "Which role family do your skills resemble most?"), key=f"gap_fit_{kind}")
        st.caption("Cosine similarity between your skill set and each role's demand profile.")
    with tabs[4]:
        risk = g["risk"]
        if risk is None or risk.empty:
            st.success("None of your listed skills are flagged as moderate/high obsolescence risk or on the legacy watchlist.")
        else:
            st.warning("These skills you hold show obsolescence risk signals. Check the evidence column before worrying.")
            ui.show_df(risk, key=f"gap_risk_{kind}")
    ui.goto_button("Next: build my learning roadmap", "Recommendations & Roadmap", key="gap_next")
