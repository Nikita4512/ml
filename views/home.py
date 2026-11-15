"""Home page: project introduction, KPI cards and computed headline findings."""
import pandas as pd
import plotly.express as px
import streamlit as st

from src import app_state as S
from src import ui_components as ui
from src.frontend_logic import MODE_LABEL, fmt_int

MODE_SHORT = {"A": "A - Monthly trends", "A_short": "A-short - Weekly", "B": "B - Cross-sectional"}


def _hero() -> None:
    st.markdown(
        "<div class='hero'><h2>Which digital skills are fading, which are rising, and what should you learn next?</h2>"
        "<p>This system reads thousands of real job postings, extracts technical skills with NLP, measures demand, "
        "detects emerging and declining skills, trains machine-learning models, and builds a personalised skill-gap "
        "analysis with a learning roadmap.</p></div>", unsafe_allow_html=True)


def _pipeline_steps() -> None:
    steps = ["CSV upload", "Cleaning", "NLP skill extraction", "Demand analysis", "Trend / risk scoring",
             "ML models", "Skill gap", "Roadmap & reports"]
    st.markdown("<span class='arrow'>&rarr;</span>".join(f"<span class='step'>{s}</span>" for s in steps),
                unsafe_allow_html=True)


def _onboarding() -> None:
    st.info("No processed dataset yet. Choose one of the two options below.")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown(ui.card("Option 1 - Use your real Kaggle CSV",
                            "<p>Open <b>Dataset &amp; Upload</b>, point to <code>postings.csv</code> "
                            "(or run <code>python pipeline.py</code> once), then explore every page.</p>"),
                    unsafe_allow_html=True)
        ui.goto_button("Go to Dataset & Upload", "Dataset & Upload", key="home_goto_upload")
    with c2:
        st.markdown(ui.card("Option 2 - Try the DEMO",
                            "<p>Builds a small <b>synthetic</b> dataset (clearly labelled DEMO) so you can "
                            "see every feature working. Demo numbers are not real findings.</p>"),
                    unsafe_allow_html=True)
        if st.button("Build demo analysis (takes ~20 s)", key="home_build_demo"):
            bar = st.progress(0.0, text="Starting demo build...")
            try:
                S.build_demo(progress=lambda m, f: bar.progress(f, text=m))
            except Exception as exc:
                bar.empty()
                ui.error_box("building the demo", exc)
            else:
                S.flash("success", "Demo analysis is ready. All pages now show DEMO data.")
                st.session_state["_pending_source"] = S.DEMO
                st.rerun()


def render(art, kind):
    ui.page_header("AI-Based Digital Skill Obsolescence & Future Skill Gap Prediction System",
                   "Data-driven skill intelligence from real job postings")
    _hero()
    if art is None:
        _onboarding()
        _pipeline_steps()
        return

    meta, metrics = art["meta"], art.get("metrics", {})
    ui.source_banner(meta)
    n_models = sum(1 for m in metrics.values() if m.get("status") == "ok")
    ui.kpi_row([
        ("Postings analysed", fmt_int(meta["rows"]), f"{meta['postings_with_any_skill_pct']}% mention a skill"),
        ("Skills detected", fmt_int(meta["unique_skills_found"]), "from the curated taxonomy"),
        ("Role families profiled", fmt_int(meta["n_roles_profiled"]), "from job-title rules"),
        ("Date coverage", f"{(meta.get('date_min') or '-')[:7]} to {(meta.get('date_max') or '-')[:7]}",
         f"{meta.get('temporal', {}).get('span_days', 0)} days"),
        ("Analysis mode", MODE_SHORT.get(meta["analysis_mode"], meta["analysis_mode"]), "decided from the data"),
        ("ML models trained", f"{n_models} / 2", "role classifier, salary model"),
    ])
    st.write("")
    _pipeline_steps()
    st.write("")

    c1, c2, c3 = st.columns(3)
    c1.markdown(ui.card("What it does", "<ul><li>Extracts skills from job text</li><li>Ranks demand by role and level</li>"
                        "<li>Flags emerging vs fading skills</li><li>Predicts role family and salary</li></ul>"), unsafe_allow_html=True)
    c2.markdown(ui.card("Personalised for you", "<ul><li>Enter your skills + target role</li><li>See readiness score</li>"
                        "<li>Get prioritised missing skills</li><li>Follow a phased roadmap</li></ul>"), unsafe_allow_html=True)
    c3.markdown(ui.card("Built to be honest", "<ul><li>Nothing hard-coded</li><li>Baselines beside every metric</li>"
                        "<li>No forecast without enough dates</li><li>Demo data always labelled</li></ul>"), unsafe_allow_html=True)
    st.write("")

    left, right = st.columns(2)
    top = art["demand"].head(15).iloc[::-1].assign(share_pct=lambda d: d["share"] * 100)
    fig = px.bar(top, x="share_pct", y="skill", color="category", orientation="h",
                 labels={"share_pct": "% of postings", "skill": ""})
    with left:
        ui.show_fig(ui.style_fig(fig, 460, "Top 15 skills by share of postings"), key="home_top_skills")
    tc = art.get("temporal_counts")
    with right:
        if tc is not None and len(tc):
            fig2 = px.bar(tc, x="period", y="postings", labels={"period": "Month", "postings": "Postings"})
            ui.show_fig(ui.style_fig(fig2, 460, "Postings per month (data coverage)"), key="home_monthly")
        else:
            st.info("No valid posting dates were found, so no coverage chart is available.")

    st.subheader("Headline findings (computed from the loaded data)")
    d = art["demand"]
    lines = [f"Most requested skill: **{d.iloc[0]['skill']}** ({d.iloc[0]['share'] * 100:.1f}% of postings)."]
    cd = art.get("category_demand")
    if cd is not None and len(cd):
        lines.append(f"Most common skill category: **{cd.iloc[0]['category']}** ({cd.iloc[0]['share'] * 100:.1f}% of postings).")
    tt = art.get("trend_table")
    if meta["analysis_mode"] in ("A", "A_short") and tt is not None and len(tt):
        lines.append(f"Time trends: **{int((tt['status'] == 'Emerging').sum())} emerging** and "
                     f"**{int((tt['status'] == 'Declining').sum())} declining** skills (FDR-corrected, "
                     f"evidence capped at {meta['temporal']['evidence_cap'].lower()}).")
    else:
        lines.append("Time trends are **not available** for this dataset (not enough dated postings). "
                     "Only a weak junior-vs-senior proxy is shown.")
    rc = metrics.get("role_classifier", {})
    if rc.get("status") == "ok":
        lines.append(f"Role classifier ({rc['best_model']}): macro-F1 **{rc['test_macro_f1']:.3f}** on held-out data "
                     f"vs baseline {rc['baseline_macro_f1']:.3f}.")
    st.markdown("\n".join(f"- {l}" for l in lines))
    with st.expander("Why this analysis mode?"):
        st.write(MODE_LABEL.get(meta["analysis_mode"], ""))
        for r in meta.get("temporal", {}).get("reasons", []):
            st.write("- " + r)
