"""Future-skill recommendations and a phased learning roadmap."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import ui_components as ui
from src.frontend_logic import recommend_future_skills
from src.reports import df_to_csv_bytes
from src.roadmap import PHASE_NAMES, build_roadmap, roadmap_to_markdown


def render(art, kind):
    ui.page_header("Recommendations & Learning Roadmap", "A phased plan built from your skill gap")
    if art is None:
        ui.no_data(kind)
        return
    ui.source_banner(art["meta"])
    g = st.session_state.get("gap_result")
    if not g or g["kind"] != kind:
        st.info("Run the **Skill Gap Analyzer** first - the roadmap is built from your missing skills.")
        ui.goto_button("Open Skill Gap Analyzer", "Skill Gap Analyzer", key="rm_goto_gap")
        return
    res = g["result"]

    c1, c2 = st.columns(2)
    hours = c1.slider("Study hours per week", 3, 40, 10, key=f"rm_hours_{kind}")
    max_sk = c2.slider("Maximum skills in the plan", 3, 20, 10, key=f"rm_max_{kind}")
    rm = build_roadmap(res["missing"], g["user_skills"], g["role"], art.get("cooccurrence"), max_sk, hours)
    st.session_state["roadmap_df"] = rm

    if rm.empty:
        st.success("No missing skills - you already cover the in-demand skills for this role in the data.")
    else:
        total = float(rm["effort_weeks_estimate"].sum())
        ui.kpi_row([("Target role", g["role"], f"readiness now {res['readiness_pct']:.0f}%"),
                    ("Skills in plan", str(len(rm)), "incl. prerequisites"),
                    ("Estimated duration", f"~{total:.0f} weeks", f"at {hours} h/week (planning estimate)"),
                    ("Phases", str(rm["phase"].nunique()), "foundation to advanced")])
        st.caption("Phase order and weeks are heuristics for planning. Which skills are missing, their demand and trend "
                   "status ARE computed from the dataset.")
        fig = go.Figure()
        for phase, grp in rm.groupby("phase"):
            fig.add_trace(go.Bar(name=PHASE_NAMES[phase], y=grp["skill"], x=grp["effort_weeks_estimate"], orientation="h",
                                 base=grp["cumulative_weeks"] - grp["effort_weeks_estimate"]))
        fig.update_xaxes(title="Weeks from start")
        fig.update_yaxes(autorange="reversed")
        ui.show_fig(ui.style_fig(fig, max(360, 28 * len(rm) + 120), "Learning timeline"), key=f"rm_gantt_{kind}")

        for phase, grp in rm.groupby("phase"):
            with st.expander(f"{PHASE_NAMES[phase]}  ({len(grp)} skills)", expanded=(phase == rm['phase'].min())):
                for _, r in grp.iterrows():
                    st.markdown(f"**{r['step']}. {r['skill']}** - {r['category']}  \n"
                                f"Why: {r['why']}  \nPractice: {r['practice_idea']}  \n"
                                f"Estimated effort: ~{r['effort_weeks_estimate']} weeks  \n"
                                f"Search topic: _{r['search_topic']}_")
        d1, d2 = st.columns(2)
        d1.download_button("Download roadmap (CSV)", df_to_csv_bytes(rm), "learning_roadmap.csv", "text/csv", key=f"rm_csv_{kind}")
        d2.download_button("Download roadmap (Markdown)", roadmap_to_markdown(rm, g["role"], hours), "learning_roadmap.md",
                           "text/markdown", key=f"rm_md_{kind}")

    st.subheader("Future skills worth watching")
    rec, basis = recommend_future_skills(art, g["user_skills"], n=10)
    st.caption(basis)
    if rec.empty:
        st.info("No recommendations available for this dataset.")
    else:
        ui.show_df(rec, key=f"rm_rec_{kind}")
    sp = art.get("salary_premium")
    if sp is not None and len(sp):
        st.subheader("Skills associated with higher pay (not yet in your skill set)")
        cand = sp[~sp["skill"].isin(g["user_skills"]) & (sp["p_value"] < 0.05)].head(8)
        if cand.empty:
            st.caption("No statistically significant salary associations among skills you do not already have.")
        else:
            ui.show_df(cand[["skill", "postings_with_salary", "median_with", "median_without", "premium_pct"]],
                       key=f"rm_sal_{kind}")
            st.caption("Association only - higher pay may reflect seniority or industry, not the skill itself.")
