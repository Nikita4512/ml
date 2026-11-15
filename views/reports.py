"""Downloadable reports: CSV tables, Excel workbook, HTML and PDF."""
import streamlit as st

from src import ui_components as ui
from src.pdf_report import build_pdf_bytes
from src.report_sections import build_excel_sheets, build_sections, sections_for_html
from src.reports import build_excel_bytes, build_html_report, df_to_csv_bytes


def _gap_for(kind):
    g = st.session_state.get("gap_result")
    return g if g and g["kind"] == kind else None


def render(art, kind):
    ui.page_header("Reports & Downloads", "Export tables and full analytical reports")
    if art is None:
        ui.no_data(kind)
        return
    meta = art["meta"]
    ui.source_banner(meta)
    gap = _gap_for(kind)
    roadmap = st.session_state.get("roadmap_df") if gap else None
    tag = "demo" if meta.get("is_demo") else "real"

    if gap:
        st.success(f"Your personal skill-gap result for **{gap['role']}** will be included in the reports.")
    else:
        st.info("Tip: run the Skill Gap Analyzer first to add a personalised section to the reports.")

    st.subheader("Full reports")
    c1, c2, c3 = st.columns(3)
    title = "Skill Obsolescence & Future Skill Gap Report" + (" (DEMO DATA)" if meta.get("is_demo") else "")
    try:
        sections = build_sections(art, gap, roadmap)
    except Exception as exc:
        ui.error_box("preparing the report", exc)
        return

    with c1:
        st.markdown("**PDF report**")
        if st.button("Generate PDF", key="rep_pdf_btn"):
            try:
                st.session_state["report_pdf"] = build_pdf_bytes(title, meta, sections, bool(meta.get("is_demo")))
            except Exception as exc:
                ui.error_box("creating the PDF", exc)
        if st.session_state.get("report_pdf"):
            st.download_button("Download PDF", st.session_state["report_pdf"], f"skill_report_{tag}.pdf",
                               "application/pdf", key="rep_pdf_dl")
    with c2:
        st.markdown("**Excel workbook**")
        if st.button("Generate Excel", key="rep_xlsx_btn"):
            try:
                st.session_state["report_excel"] = build_excel_bytes(build_excel_sheets(art, gap, roadmap), meta)
            except Exception as exc:
                ui.error_box("creating the Excel file", exc)
        if st.session_state.get("report_excel"):
            st.download_button("Download Excel", st.session_state["report_excel"], f"skill_tables_{tag}.xlsx",
                               "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key="rep_xlsx_dl")
    with c3:
        st.markdown("**HTML report**")
        if st.button("Generate HTML", key="rep_html_btn"):
            try:
                st.session_state["report_html"] = build_html_report(title, meta, sections_for_html(sections),
                                                                    bool(meta.get("is_demo"))).encode("utf-8")
            except Exception as exc:
                ui.error_box("creating the HTML report", exc)
        if st.session_state.get("report_html"):
            st.download_button("Download HTML", st.session_state["report_html"], f"skill_report_{tag}.html",
                               "text/html", key="rep_html_dl")

    st.subheader("Individual tables (CSV)")
    tables = {"Skill demand": art.get("demand"), "Category demand": art.get("category_demand"),
              "Trend table": art.get("trend_table"), "Obsolescence risk": art.get("obsolescence"),
              "Cross-sectional signals": art.get("cross_sectional"), "Legacy watchlist": art.get("legacy"),
              "Salary premium": art.get("salary_premium"), "Candidate skills (TF-IDF)": art.get("candidate_skills")}
    tables = {k: v for k, v in tables.items() if v is not None and len(v)}
    if not tables:
        st.info("No tables available.")
        return
    name = st.selectbox("Table", list(tables), key="rep_table")
    ui.show_df(tables[name].head(200), key="rep_preview", height=320)
    st.download_button(f"Download '{name}' (CSV)", df_to_csv_bytes(tables[name]),
                       f"{name.lower().replace(' ', '_').replace('(', '').replace(')', '')}_{tag}.csv", "text/csv", key="rep_csv_dl")
