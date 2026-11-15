"""Backend smoke test (needs NO Streamlit/Plotly).   Run:  python tests/test_backend_smoke.py

Runs the full pipeline on the synthetic DEMO csv in a temporary folder and checks gap analysis,
roadmap and all report formats. The numbers printed are DEMO numbers, not real findings.
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src import gap_analysis as g                       # noqa: E402
from src.demo_data import ensure_demo_csv                # noqa: E402
from src.orchestrator import run_pipeline                # noqa: E402
from src.pdf_report import build_pdf_bytes               # noqa: E402
from src.report_sections import build_excel_sheets, build_sections, sections_for_html   # noqa: E402
from src.reports import build_excel_bytes, build_html_report                           # noqa: E402
from src.roadmap import build_roadmap, roadmap_to_markdown                             # noqa: E402
from src.skill_extractor import SkillExtractor           # noqa: E402
from src.skills_taxonomy import SKILL_CATEGORY           # noqa: E402


def main() -> None:
    tmp = Path(tempfile.mkdtemp())
    art = run_pipeline(ensure_demo_csv(), tmp / "out", tmp / "models", source_kind="demo")
    meta = art["meta"]
    print(f"pipeline OK: rows={meta['rows']} mode={meta['analysis_mode']} skills={meta['unique_skills_found']}")
    assert meta["is_demo"] is True
    parsed = g.parse_user_skills("python, sql, excel", SkillExtractor())
    role = list(art["role_profile"].index)[0]
    res = g.compute_gap(parsed["recognised"], role, art["role_profile"], art["demand"], SKILL_CATEGORY, art["trend_table"])
    rm = build_roadmap(res["missing"], parsed["recognised"], role, art["cooccurrence"])
    gap = {"role": role, "user_skills": parsed["recognised"], "result": res}
    sections = build_sections(art, gap, rm)
    pdf = build_pdf_bytes("Test", meta, sections, True)
    xlsx = build_excel_bytes(build_excel_sheets(art, gap, rm), meta)
    html = build_html_report("Test", meta, sections_for_html(sections), True)
    assert pdf[:4] == b"%PDF" and xlsx[:2] == b"PK" and "<html" in html
    assert roadmap_to_markdown(rm, role)
    print(f"gap/roadmap/reports OK (pdf {len(pdf)//1024} KB, xlsx {len(xlsx)//1024} KB, html {len(html)//1024} KB)")
    print("ALL BACKEND CHECKS PASSED")


if __name__ == "__main__":
    main()
