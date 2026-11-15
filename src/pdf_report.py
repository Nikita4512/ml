"""PDF report generator (uses reportlab only; no browser/Kaleido needed)."""
from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime
from typing import List, Sequence, Tuple, Union

import pandas as pd
from reportlab.graphics.charts.barcharts import HorizontalBarChart
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

PURPLE = colors.HexColor("#7C3AED")
NAVY = colors.HexColor("#0B1026")
LAVENDER = colors.HexColor("#F3EEFF")


@dataclass
class BarSpec:
    """A horizontal bar chart: labels + values (+ title/unit)."""
    labels: Sequence[str]
    values: Sequence[float]
    title: str = ""
    unit: str = ""


def _safe(text) -> str:
    """Reportlab's default fonts are Latin-1; replace anything else instead of crashing."""
    s = "" if text is None else str(text)
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return s.encode("latin-1", "replace").decode("latin-1")


def _fmt(v) -> str:
    if isinstance(v, float):
        if pd.isna(v):
            return ""
        return f"{v:,.3f}" if abs(v) < 1000 else f"{v:,.0f}"
    return str(v)


def _df_table(df: pd.DataFrame, cell_style: ParagraphStyle, head_style: ParagraphStyle,
              width: float, max_rows: int = 25, max_cols: int = 7) -> Table:
    df = df.iloc[:max_rows, :max_cols]
    data = [[Paragraph(_safe(c), head_style) for c in df.columns]]
    for _, row in df.iterrows():
        data.append([Paragraph(_safe(_fmt(v))[:60], cell_style) for v in row])
    t = Table(data, colWidths=[width / max(len(df.columns), 1)] * len(df.columns), repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), PURPLE),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LAVENDER]),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D4C9F5")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return t


def _bar_drawing(spec: BarSpec, width: float) -> Drawing:
    n = max(len(spec.labels), 1)
    height = 30 + 13 * n
    d = Drawing(width, height)
    chart = HorizontalBarChart()
    chart.x, chart.y, chart.width, chart.height = 120, 15, width - 150, height - 30
    chart.data = [list(spec.values)[::-1]]
    chart.categoryAxis.categoryNames = [_safe(l)[:28] for l in list(spec.labels)[::-1]]
    chart.categoryAxis.labels.fontSize = 7
    chart.valueAxis.labels.fontSize = 7
    chart.valueAxis.valueMin = 0
    chart.bars[0].fillColor = PURPLE
    chart.barSpacing = 2
    d.add(chart)
    return d


Content = Union[str, pd.DataFrame, dict, BarSpec]


def build_pdf_bytes(title: str, meta: dict, sections: List[Tuple[str, Content]], is_demo: bool = False) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=16 * mm, rightMargin=16 * mm,
                            topMargin=16 * mm, bottomMargin=16 * mm, title=_safe(title))
    width = A4[0] - 32 * mm
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=PURPLE, fontSize=18, leading=22)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=PURPLE, fontSize=12, spaceBefore=10)
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=9, leading=12)
    cell = ParagraphStyle("cell", parent=body, fontSize=7, leading=9)
    head = ParagraphStyle("head", parent=cell, textColor=colors.white, fontName="Helvetica-Bold")
    story = [Paragraph(_safe(title), h1),
             Paragraph(f"Generated {datetime.now():%Y-%m-%d %H:%M}", body), Spacer(1, 6)]

    label = "DEMO / SYNTHETIC DATA - these are NOT real results" if is_demo else "REAL DATASET RESULTS"
    banner = Table([[Paragraph(f"<b>{label}</b>", ParagraphStyle("b", parent=body, textColor=colors.white))]],
                   colWidths=[width])
    banner.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, -1),
                                 colors.HexColor("#B45309") if is_demo else colors.HexColor("#047857")),
                                ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
    story += [banner, Spacer(1, 8)]

    for heading, content in sections:
        block = [Paragraph(_safe(heading), h2)]
        if isinstance(content, pd.DataFrame):
            block.append(_df_table(content, cell, head, width) if len(content)
                         else Paragraph("No data available for this section.", body))
        elif isinstance(content, BarSpec):
            if len(content.labels):
                if content.title:
                    block.append(Paragraph(_safe(content.title), body))
                block.append(_bar_drawing(content, width))
            else:
                block.append(Paragraph("No data available for this section.", body))
        elif isinstance(content, dict):
            rows = [[Paragraph(f"<b>{_safe(k)}</b>", cell), Paragraph(_safe(v), cell)] for k, v in content.items()]
            if rows:
                t = Table(rows, colWidths=[width * 0.38, width * 0.62])
                t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#D4C9F5")),
                                       ("BACKGROUND", (0, 0), (0, -1), LAVENDER)]))
                block.append(t)
        else:
            for para in str(content).split("\n"):
                block.append(Paragraph(_safe(para) or "&nbsp;", body))
        story.append(KeepTogether(block[:2]))
        story.extend(block[2:])

    def footer(canvas, _doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.grey)
        canvas.drawString(16 * mm, 9 * mm, "Skill Obsolescence & Skill Gap System - all figures computed from the loaded dataset")
        canvas.drawRightString(A4[0] - 16 * mm, 9 * mm, f"Page {_doc.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return buf.getvalue()
