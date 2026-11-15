"""Session/state helpers for the Streamlit app (no layout code here).

Two completely separate result folders are used so DEMO data can never overwrite
or be confused with REAL results:
    data/processed/        <- real dataset (path or upload)
    data/processed_demo/   <- synthetic demo data
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import streamlit as st

from .config import DATA_DIR, MODELS_DIR, PROCESSED_DIR
from .demo_data import ensure_demo_csv
from .orchestrator import artifacts_exist, load_artifacts, run_pipeline

REAL = "Real dataset"
DEMO = "Demo (synthetic)"
DEMO_PROCESSED_DIR = DATA_DIR / "processed_demo"
DEMO_MODELS_DIR = MODELS_DIR / "demo"


def dirs_for(kind: str):
    """(processed_dir, models_dir) for the chosen data source."""
    return (DEMO_PROCESSED_DIR, DEMO_MODELS_DIR) if kind == DEMO else (PROCESSED_DIR, MODELS_DIR)


def available(kind: str) -> bool:
    return artifacts_exist(dirs_for(kind)[0])


def get_artifacts(kind: str) -> Optional[dict]:
    """Load saved results for `kind` once per session (cached in session_state)."""
    version = st.session_state.get("art_version", 0)
    key = f"art::{kind}::{version}"
    if key in st.session_state:
        return st.session_state[key]
    if not available(kind):
        return None
    art = load_artifacts(dirs_for(kind)[0])
    st.session_state[key] = art
    return art


def invalidate() -> None:
    """Drop cached results (call after a pipeline run) and clear personal results."""
    for k in [k for k in st.session_state.keys() if str(k).startswith("art::")]:
        del st.session_state[k]
    st.session_state["art_version"] = st.session_state.get("art_version", 0) + 1
    for k in ("gap_result", "report_excel", "report_html", "report_pdf"):
        st.session_state.pop(k, None)


def build_demo(progress: Optional[Callable[[str, float], None]] = None) -> dict:
    """Generate the synthetic CSV (if needed) and run the SAME pipeline on it."""
    csv = ensure_demo_csv()
    processed, models = dirs_for(DEMO)
    art = run_pipeline(csv, processed, models, source_kind="demo", progress=progress)
    invalidate()
    return art


def models_dir_for(kind: str) -> Path:
    return dirs_for(kind)[1]


def flash(kind: str, message: str) -> None:
    """Queue a message to show at the top of the next page render."""
    st.session_state["flash"] = (kind, message)
