"""Data cleaning and preprocessing.

Input : raw dataframe with canonical column names (from data_loader).
Output: cleaned dataframe + a report dict describing exactly what was removed.
"""
from __future__ import annotations

import html
import re
from typing import Tuple

import numpy as np
import pandas as pd

from .config import SALARY_MAX, SALARY_MIN
from .data_loader import parse_datetime

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")

# multiply a pay-period salary by this to get an approximate yearly figure
PAY_FACTOR = {"HOURLY": 2080, "WEEKLY": 52, "BIWEEKLY": 26, "MONTHLY": 12,
              "YEARLY": 1, "ANNUAL": 1, "ANNUALLY": 1}

JUNIOR = {"internship", "entry level", "associate"}
SENIOR = {"mid-senior level", "director", "executive", "senior"}
MID = {"mid level", "mid"}


def level_group_of(value) -> str:
    """Map an experience-level text to junior / mid / senior / unknown."""
    if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
        return "unknown"
    v = str(value).strip().lower()
    if v in JUNIOR:
        return "junior"
    if v in SENIOR:
        return "senior"
    if v in MID:
        return "mid"
    return "unknown"


def _clean_text(s: pd.Series) -> pd.Series:
    """Strip HTML tags/entities and collapse whitespace."""
    s = s.fillna("").astype(str)
    s = s.str.replace(TAG_RE, " ", regex=True)
    has_entity = s.str.contains("&", regex=False)
    if has_entity.any():
        s = s.where(~has_entity, s.map(html.unescape))
    return s.str.replace(WS_RE, " ", regex=True).str.strip()


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(np.nan, index=df.index)


def compute_salary(df: pd.DataFrame) -> pd.Series:
    """Best-effort yearly salary: normalized_salary, else midpoint x pay-period factor."""
    sal = _num(df, "normalized_salary")
    low, high, med = _num(df, "min_salary"), _num(df, "max_salary"), _num(df, "med_salary")
    mid = med.fillna((low + high) / 2).fillna(low).fillna(high)
    if "pay_period" in df.columns:
        factor = df["pay_period"].astype(str).str.upper().str.strip().map(PAY_FACTOR).fillna(1)
    else:
        factor = 1
    sal = sal.fillna(mid * factor)
    return sal.where(sal.between(SALARY_MIN, SALARY_MAX))      # outliers -> NaN


def clean_postings(df: pd.DataFrame, drop_content_duplicates: bool = True) -> Tuple[pd.DataFrame, dict]:
    """Clean the canonical dataframe. Nothing is imputed: missing stays missing."""
    report = {"rows_in": int(len(df))}
    df = df.copy()

    for col in ("title", "description", "skills_desc", "company", "location"):
        df[col] = _clean_text(df[col]) if col in df.columns else ""

    # 1) rows with no usable text cannot contain skills
    has_text = (df["title"] != "") | (df["description"] != "") | (df["skills_desc"] != "")
    report["dropped_no_text"] = int((~has_text).sum())
    df = df[has_text].reset_index(drop=True)

    # 2) duplicates
    if "job_id" not in df.columns or df["job_id"].isna().all():
        df["job_id"] = np.arange(len(df)).astype(str)
    before = len(df)
    df = df.drop_duplicates(subset="job_id", keep="first")
    report["dropped_duplicate_job_id"] = int(before - len(df))
    if drop_content_duplicates:
        before = len(df)
        key = pd.util.hash_pandas_object(
            pd.DataFrame({"t": df["title"].str.lower(), "c": df["company"].str.lower(),
                          "d": df["description"].str.lower().str[:300]}), index=False)
        df = df[~key.duplicated(keep="first")]
        report["dropped_duplicate_content"] = int(before - len(df))
    df = df.reset_index(drop=True)

    # 3) dates, levels, salary
    df["listed_date"] = parse_datetime(df["listed_time"]) if "listed_time" in df.columns else pd.NaT
    lvl = df["experience_level"] if "experience_level" in df.columns else pd.Series(pd.NA, index=df.index)
    df["experience_level"] = lvl.fillna("Not specified")
    df["level_group"] = df["experience_level"].map(level_group_of)
    wt = df["work_type"] if "work_type" in df.columns else pd.Series(pd.NA, index=df.index)
    df["work_type"] = wt.fillna("Not specified")
    df["salary_annual"] = compute_salary(df)

    keep = ["job_id", "title", "company", "location", "experience_level", "level_group",
            "work_type", "listed_date", "salary_annual", "description", "skills_desc"]
    df = df[keep]

    report.update({
        "rows_out": int(len(df)),
        "pct_with_description": round(float((df["description"] != "").mean() * 100), 1) if len(df) else 0.0,
        "pct_with_valid_date": round(float(df["listed_date"].notna().mean() * 100), 1) if len(df) else 0.0,
        "pct_with_salary": round(float(df["salary_annual"].notna().mean() * 100), 1) if len(df) else 0.0,
        "level_counts": df["level_group"].value_counts().to_dict(),
    })
    return df, report
