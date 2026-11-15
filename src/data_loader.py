"""Large-CSV loading, automatic schema detection and dataset summary.

Works with a file path OR a file-like object (e.g. a Streamlit upload).
The CSV is read in chunks and only the columns we need are kept, so the
~166 MB LinkedIn file loads comfortably on a normal laptop.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from .config import CHUNK_SIZE


class DatasetError(Exception):
    """Raised with a beginner-friendly message when the dataset cannot be used."""


# canonical name -> possible column names in the file (checked in order, lower-case)
COLUMN_ALIASES: Dict[str, List[str]] = {
    "job_id": ["job_id", "id", "jobid", "posting_id"],
    "title": ["title", "job_title", "jobtitle", "position", "job_position"],
    "description": ["description", "job_description", "jobdescription", "job_summary", "details"],
    "skills_desc": ["skills_desc", "required_skills", "skills", "job_skills", "skill_description"],
    "company": ["company_name", "company", "employer", "company_title"],
    "location": ["location", "job_location", "city"],
    "experience_level": ["formatted_experience_level", "experience_level", "job_level",
                         "seniority_level", "seniority", "experience"],
    "work_type": ["formatted_work_type", "work_type", "job_type", "employment_type"],
    "listed_time": ["listed_time", "original_listed_time", "date_posted", "posted_date",
                    "post_date", "listing_date", "first_seen", "posted_at", "created_at", "date"],
    "normalized_salary": ["normalized_salary"],
    "min_salary": ["min_salary", "salary_min"],
    "max_salary": ["max_salary", "salary_max"],
    "med_salary": ["med_salary", "salary_med", "median_salary"],
    "pay_period": ["pay_period", "salary_period"],
}
TEXT_COLUMNS = ["title", "description", "skills_desc"]
ALWAYS_PRESENT = ["title", "description", "skills_desc", "company", "location",
                  "experience_level", "work_type"]


def detect_schema(columns, manual: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Map canonical names -> real column names found in the file.

    `manual` lets the UI override a guess, e.g. {"description": "job_desc"}.
    """
    lower = {str(c).strip().lower(): c for c in columns}
    mapping: Dict[str, str] = {}
    used = set()
    for canon, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            if alias in lower and lower[alias] not in used:
                mapping[canon] = lower[alias]
                used.add(lower[alias])
                break
    for canon, col in (manual or {}).items():
        if col in list(columns):
            mapping[canon] = col
    return mapping


def find_postings_file(raw_dir: Path) -> Path:
    """Pick the most likely postings CSV inside `raw_dir` (searches sub-folders too)."""
    csvs = list(Path(raw_dir).rglob("*.csv"))
    if not csvs:
        raise DatasetError(f"No CSV files found in {raw_dir}. Put the Kaggle CSV (e.g. postings.csv) there.")
    named = [p for p in csvs if "posting" in p.name.lower()]
    pool = named or csvs
    return max(pool, key=lambda p: p.stat().st_size)


def _rewind(source) -> None:
    if hasattr(source, "seek"):
        source.seek(0)


def read_header(source) -> List[str]:
    """Return the CSV column names without loading the data."""
    try:
        head = pd.read_csv(source, nrows=0, encoding="utf-8", encoding_errors="replace")
    except FileNotFoundError:
        raise DatasetError(f"File not found: {source}")
    except pd.errors.EmptyDataError:
        raise DatasetError("The CSV file is empty.")
    except pd.errors.ParserError as exc:
        raise DatasetError(f"The file does not look like a valid CSV: {exc}")
    finally:
        _rewind(source)
    return list(head.columns)


def load_postings(
    source,
    manual_mapping: Optional[Dict[str, str]] = None,
    chunksize: int = CHUNK_SIZE,
    max_rows: Optional[int] = None,
    progress: Optional[Callable[[int], None]] = None,
) -> Tuple[pd.DataFrame, Dict[str, str], dict]:
    """Read the CSV in chunks, keep only useful columns, rename to canonical names.

    Returns (dataframe, mapping canonical->original, info dict).
    """
    columns = read_header(source)
    mapping = detect_schema(columns, manual_mapping)
    if not any(c in mapping for c in TEXT_COLUMNS):
        raise DatasetError(
            "Could not find any text column (title / description / skills). "
            f"Columns in your file: {columns[:30]}. Use the manual column selector."
        )
    usecols = [c for c in columns if c in set(mapping.values())]
    rename = {orig: canon for canon, orig in mapping.items()}

    parts, total = [], 0
    try:
        reader = pd.read_csv(
            source, usecols=usecols, dtype=str, chunksize=chunksize,
            encoding="utf-8", encoding_errors="replace", on_bad_lines="skip",
        )
        for chunk in reader:
            chunk = chunk.rename(columns=rename)
            if max_rows is not None and total + len(chunk) > max_rows:
                chunk = chunk.iloc[: max_rows - total]
            parts.append(chunk)
            total += len(chunk)
            if progress:
                progress(total)
            if max_rows is not None and total >= max_rows:
                break
    except pd.errors.EmptyDataError:
        raise DatasetError("The CSV file is empty.")
    except MemoryError:
        raise DatasetError("Out of memory. Try the 'max rows' option to load a smaller sample.")
    if not parts:
        raise DatasetError("No rows could be read from the file.")

    df = pd.concat(parts, ignore_index=True)
    for col in ALWAYS_PRESENT:                      # make sure downstream code can rely on them
        if col not in df.columns:
            df[col] = pd.NA
    info = {"rows_read": int(len(df)), "columns_in_file": columns,
            "columns_used": usecols}
    return df, mapping, info


def parse_datetime(series: pd.Series) -> pd.Series:
    """Convert epoch (ms or s) or text dates to datetime; invalid/implausible -> NaT."""
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().mean() > 0.5:
        median = num.dropna().median()
        unit = "ms" if median > 1e11 else "s"
        dt = pd.to_datetime(num, unit=unit, errors="coerce")
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dt = pd.to_datetime(series, errors="coerce", utc=True).dt.tz_convert(None)
    lo = pd.Timestamp("2000-01-01")
    hi = pd.Timestamp.now() + pd.Timedelta(days=366)
    return dt.where((dt >= lo) & (dt <= hi))


def summarize_dataset(df: pd.DataFrame, mapping: Dict[str, str]) -> dict:
    """Quick facts for the inspection page (works on the raw canonical dataframe)."""
    summary = {"rows": int(len(df)), "columns_detected": mapping}
    summary["null_pct"] = {c: round(float(df[c].isna().mean() * 100), 1) for c in df.columns}
    if "listed_time" in df.columns:
        dates = parse_datetime(df["listed_time"]).dropna()
        summary["date_min"] = str(dates.min()) if len(dates) else None
        summary["date_max"] = str(dates.max()) if len(dates) else None
        summary["date_valid_pct"] = round(len(dates) / max(len(df), 1) * 100, 1)
    if "description" in df.columns:
        lengths = df["description"].dropna().str.len()
        summary["avg_description_chars"] = int(lengths.mean()) if len(lengths) else 0
    sal_cols = [c for c in ("normalized_salary", "min_salary", "max_salary", "med_salary") if c in df.columns]
    if sal_cols:
        summary["salary_available_pct"] = round(float(df[sal_cols].notna().any(axis=1).mean() * 100), 1)
    summary["data_mode_note"] = "REAL"
    summary["currency_assumption"] = "Currency not validated - assumed to match dataset origin."
    return summary


def validate_dataset_quality(df: pd.DataFrame, mapping: Dict[str, str]) -> dict:
    """Return a data-quality assessment dict for display in the upload view.

    Data mode is always REAL here; the orchestrator sets demo mode separately.
    Numbers come from df - nothing is fabricated.
    """
    issues = []
    # Critical columns check
    missing_critical = [c for c in ["title", "description"] if c not in mapping]
    if missing_critical:
        issues.append(f"CRITICAL: missing column(s): {missing_critical}. Skill extraction will be impaired.")
    if "listed_time" not in mapping:
        issues.append("No date column detected. Time trends and forecasting will be DISABLED (Mode B).")
    sal_cols_present = any(c in mapping for c in ["normalized_salary", "min_salary", "max_salary", "med_salary"])
    if not sal_cols_present:
        issues.append("No salary columns detected. Salary analysis will be skipped.")
    # Temporal suitability
    if "listed_time" not in mapping:
        temporal_suitability = "NO_DATE_COLUMN"
        n_months = 0
    else:
        dates = parse_datetime(df["listed_time"]).dropna() if "listed_time" in df.columns else pd.Series(dtype="datetime64[ns]")
        n_months = int(dates.dt.to_period("M").nunique()) if len(dates) else 0
        if n_months >= 6:
            temporal_suitability = "SUITABLE"
        elif n_months >= 3:
            temporal_suitability = "MARGINAL"
            issues.append(f"Only {n_months} calendar months of data. Forecasting requires >= 6 months.")
        else:
            temporal_suitability = "INSUFFICIENT"
            if n_months > 0:
                issues.append(f"Only {n_months} calendar month(s) with data. Time trends will be DISABLED.")
    # Data quality score (0-100)
    score = 0
    desc_col = df.get("description") if "description" in df.columns else None
    if "description" in mapping and desc_col is not None and (desc_col != "").mean() > 0.5:
        score += 30
    if "listed_time" in mapping:
        score += 25
        if temporal_suitability == "SUITABLE":
            score += 20
    if "title" in mapping:
        score += 15
    if sal_cols_present:
        score += 10
    grade = "Excellent" if score >= 80 else "Good" if score >= 60 else "Fair" if score >= 40 else "Poor"
    sal_note = ("Currency not validated; assumed to match dataset origin. "
                "Do not assume USD for international datasets.")
    if sal_cols_present:
        issues.append(sal_note)
    return {
        "issues": issues,
        "data_mode": "REAL",
        "column_coverage": dict(mapping),
        "missing_critical_cols": missing_critical,
        "temporal_suitability": temporal_suitability,
        "n_date_months": n_months,
        "salary_note": sal_note,
        "data_quality_score": score,
        "data_quality_grade": grade,
        "duplicate_check_note": "Duplicate detection runs during cleaning (job_id + content hash).",
        "skipped_malformed_rows": "Skipped by pandas on_bad_lines='skip' (exact count unavailable).",
    }
