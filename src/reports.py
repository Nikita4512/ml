"""Downloadable reports: Excel workbook, CSV bytes and a self-contained HTML report.

Every report starts with a data-provenance banner so REAL results and DEMO data
can never be confused.
"""
from __future__ import annotations

import html
import io
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def _flatten_meta(meta: dict) -> pd.DataFrame:
    rows = []
    for k, v in meta.items():
        if isinstance(v, (dict, list)):
            v = str(v)[:500]
        rows.append((k, v))
    return pd.DataFrame(rows, columns=["item", "value"])


def build_excel_bytes(sheets: Dict[str, pd.DataFrame], meta: Optional[dict] = None) -> bytes:
    """Write a multi-sheet .xlsx. Sheet names are trimmed to Excel's 31-char limit."""
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        if meta:
            _flatten_meta(meta).to_excel(writer, sheet_name="About", index=False)
        used = set()
        for name, df in sheets.items():
            if df is None or len(df) == 0:
                continue
            safe = "".join(ch for ch in name if ch not in '[]:*?/\\')[:31] or "Sheet"
            base, i = safe, 2
            while safe in used:
                safe = f"{base[:28]}_{i}"
                i += 1
            used.add(safe)
            out = df.copy()
            for col in out.columns:                      # Excel cannot store timezone/period objects
                if str(out[col].dtype).startswith(("period", "datetime64[ns,")):
                    out[col] = out[col].astype(str)
            out.to_excel(writer, sheet_name=safe, index=out.index.name is not None)
            ws = writer.sheets[safe]
            for col_cells in ws.columns:                 # simple auto column width
                width = max(len(str(c.value)) if c.value is not None else 0 for c in col_cells[:200])
                ws.column_dimensions[col_cells[0].column_letter].width = min(max(width + 2, 10), 60)
    return buf.getvalue()


CSS = """
body{background:#0B1026;color:#E5E7EF;font-family:Segoe UI,Arial,sans-serif;margin:0;padding:32px;}
h1{color:#A78BFA;margin-bottom:4px} h2{color:#A78BFA;border-bottom:1px solid #2a3263;padding-bottom:6px;margin-top:36px}
.banner{padding:12px 16px;border-radius:8px;margin:16px 0;font-weight:600}
.real{background:#0f3d2e;border:1px solid #34D399;color:#bff5dc}
.demo{background:#4a3410;border:1px solid #FBBF24;color:#ffe9b0}
table{border-collapse:collapse;width:100%;margin:12px 0;background:#131A3A;font-size:13px}
th{background:#7C3AED;color:#fff;text-align:left;padding:8px} td{padding:6px 8px;border-bottom:1px solid #232b57}
tr:nth-child(even) td{background:#161e44} .muted{color:#9CA3C7;font-size:12px}
.kpi{display:inline-block;background:#131A3A;border:1px solid #2a3263;border-radius:10px;padding:12px 18px;margin:6px 10px 6px 0}
.kpi b{display:block;font-size:22px;color:#A78BFA}
"""

Section = Tuple[str, Union[str, pd.DataFrame, dict, object]]


def build_html_report(title: str, meta: dict, sections: List[Section], is_demo: bool = False) -> str:
    """Self-contained HTML (open in a browser, then 'Print > Save as PDF' for a PDF).

    Each section is (heading, content). Content may be: text, a DataFrame, a dict of
    KPIs (rendered as cards) or a Plotly figure (rendered with the Plotly CDN script).
    """
    label = "DEMO / SYNTHETIC DATA - not real results" if is_demo else "REAL DATASET RESULTS"
    parts = [f"<html><head><meta charset='utf-8'><title>{html.escape(title)}</title><style>{CSS}</style></head><body>",
             f"<h1>{html.escape(title)}</h1><div class='muted'>Generated {datetime.now():%Y-%m-%d %H:%M}</div>",
             f"<div class='banner {'demo' if is_demo else 'real'}'>{label}</div>"]
    kpis = {k: v for k, v in meta.items() if isinstance(v, (int, float, str)) and k in
            ("rows", "unique_skills_found", "date_min", "date_max", "analysis_mode", "data_source")}
    parts.extend(f"<div class='kpi'><b>{html.escape(str(v))}</b>{html.escape(str(k).replace('_', ' '))}</div>"
                 for k, v in kpis.items())
    plotly_loaded = False
    for heading, content in sections:
        parts.append(f"<h2>{html.escape(heading)}</h2>")
        if isinstance(content, pd.DataFrame):
            if content.empty:
                parts.append("<p class='muted'>No data available for this section.</p>")
            else:
                parts.append(content.head(200).to_html(index=False, border=0, float_format=lambda x: f"{x:,.3f}"))
        elif isinstance(content, dict):
            parts.extend(f"<div class='kpi'><b>{html.escape(str(v))}</b>{html.escape(str(k))}</div>"
                         for k, v in content.items())
        elif hasattr(content, "to_html"):                       # Plotly figure
            parts.append(content.to_html(full_html=False, include_plotlyjs=False if plotly_loaded else "cdn"))
            plotly_loaded = True
        else:
            parts.append("<p>" + html.escape(str(content)).replace("\n", "<br>") + "</p>")
    parts.append("<p class='muted'>All numbers are computed from the loaded dataset. "
                 "Skill-salary and trend findings are associations, not causal claims.</p></body></html>")
    return "\n".join(parts)
