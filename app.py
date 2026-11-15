"""Streamlit entry point.   Run with:   streamlit run app.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))      # make `src` and `views` importable

import streamlit as st

st.set_page_config(page_title="Skill Obsolescence & Skill Gap System", page_icon=":brain:",
                   layout="wide", initial_sidebar_state="expanded")

import src.app_state as S                                 # noqa: E402
from src import ui_components as ui                             # noqa: E402
from views import about, demand, gap, home, models, reports, roadmap, upload   # noqa: E402
from views import trends as trends_view                         # noqa: E402

PAGES = {
    "Home": home,
    "Dataset & Upload": upload,
    "Skill Demand": demand,
    "Emerging & Declining": trends_view,
    "ML Models": models,
    "Skill Gap Analyzer": gap,
    "Recommendations & Roadmap": roadmap,
    "Reports & Downloads": reports,
    "About & Viva Guide": about,
}


def main() -> None:
    ui.inject_css()

    # values queued by pages (widgets cannot be changed after they are drawn, so we apply them first)
    if "_pending_source" in st.session_state:
        st.session_state["source"] = st.session_state.pop("_pending_source")
    if "_pending_page" in st.session_state:
        st.session_state["page"] = st.session_state.pop("_pending_page")
    if "source" not in st.session_state:
        st.session_state["source"] = S.REAL if S.available(S.REAL) else S.DEMO
    st.session_state.setdefault("page", "Home")

    with st.sidebar:
        st.markdown("<div class='side-title'>Skill Obsolescence &<br>Skill Gap System</div>"
                    "<div class='side-sub'>AI-powered job-market skill intelligence</div>", unsafe_allow_html=True)
        st.radio("Navigation", list(PAGES), key="page", label_visibility="collapsed")
        st.divider()
        st.radio("Data source", [S.REAL, S.DEMO], key="source",
                 help="REAL = your Kaggle CSV. DEMO = small synthetic dataset for demonstrations.")
        kind = st.session_state["source"]
        if S.available(kind):
            st.caption("Results loaded.")
        else:
            st.caption("No results yet for this source.")
        st.caption("Demo data is synthetic and always labelled DEMO.")

    # clear personal results when the data source changes
    if st.session_state.get("_last_source") != kind:
        st.session_state["_last_source"] = kind
        for k in ("gap_result", "roadmap_df", "report_pdf", "report_excel", "report_html"):
            st.session_state.pop(k, None)

    flash = st.session_state.pop("flash", None)
    if flash:
        getattr(st, flash[0], st.info)(flash[1])

    try:
        art = S.get_artifacts(kind)
        if art is None and kind == S.DEMO:
            bar = st.progress(0.0, text="Building the demo analysis (first time only)...")
            art = S.build_demo(progress=lambda m, f: bar.progress(f, text=m))
            bar.empty()
            art = S.get_artifacts(kind)
    except Exception as exc:
        ui.error_box("loading results", exc)
        art = None

    page = st.session_state["page"]
    try:
        PAGES[page].render(art, kind)
    except Exception as exc:
        ui.error_box(f"rendering the '{page}' page", exc)


main()
