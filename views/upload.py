"""Dataset page: upload / pick a CSV, validate columns, preview, then run the full pipeline."""
from pathlib import Path

import pandas as pd
import streamlit as st

from src import app_state as S
from src import ui_components as ui
from src.config import CHUNK_SIZE, MODELS_DIR, PROCESSED_DIR, RAW_DIR
from src.data_loader import (COLUMN_ALIASES, DatasetError, detect_schema, find_postings_file,
                             load_postings, read_header, summarize_dataset)
from src.orchestrator import run_pipeline

MAP_FIELDS = [("title", "Job title"), ("description", "Job description"), ("skills_desc", "Skills text (optional)"),
              ("listed_time", "Posting date"), ("experience_level", "Experience level"),
              ("normalized_salary", "Yearly salary (optional)")]


def _rewind(source) -> None:
    if hasattr(source, "seek"):
        source.seek(0)


def _choose_source():
    method = st.radio("How do you want to provide the CSV?",
                      ["Local file path (recommended for the large Kaggle file)", "Browser upload"],
                      key="up_method", horizontal=True)
    if method.startswith("Local"):
        default = ""
        try:
            default = str(find_postings_file(RAW_DIR))
        except DatasetError:
            pass
        raw = st.text_input("Full path to the CSV file", value=default, key="up_path",
                            help="Example: C:\\Users\\you\\skill_gap_project\\data\\raw\\postings.csv")
        raw = raw.strip().strip('"').strip("'")
        if not raw:
            st.caption("Tip: copy the Kaggle CSV into data\\raw\\ and it will be found automatically.")
            return None, ""
        p = Path(raw)
        if not p.exists():
            st.error(f"File not found: {p}")
        elif p.is_dir():
            st.error("That is a folder. Please give the path to the .csv file itself.")
        else:
            if p.suffix.lower() != ".csv":
                st.warning("The file does not end in .csv - trying anyway.")
            st.caption(f"Found file: {p.name} ({p.stat().st_size / 1e6:.1f} MB)")
            return p, p.name
        return None, ""
    up = st.file_uploader("Upload a CSV file (up to ~1 GB)", type=["csv"], key="up_file")
    if up is not None:
        st.caption(f"Uploaded: {up.name} ({up.size / 1e6:.1f} MB)")
        return up, up.name
    return None, ""


def _mapping_panel(header):
    manual = {}
    auto = detect_schema(header)
    with st.expander("Column mapping (auto-detected - override only if a guess is wrong)", expanded=False):
        cols = st.columns(2)
        for i, (key, label) in enumerate(MAP_FIELDS):
            options = ["(auto)"] + list(header)
            choice = cols[i % 2].selectbox(f"{label}  [auto: {auto.get(key, 'not found')}]", options,
                                           key=f"map_{key}")
            if choice != "(auto)":
                manual[key] = choice
    final = detect_schema(header, manual)
    table = pd.DataFrame([{"field": k, "your column": final.get(k, "-- not found --"),
                           "status": "OK" if k in final else "missing"} for k, _ in MAP_FIELDS])
    ui.show_df(table, key="up_mapping_table")
    return manual, final


def render(art, kind):
    ui.page_header("Dataset & Upload", "Validate your CSV, inspect its columns, then run the full analysis")
    st.info("The Kaggle file is ~166 MB. It is read in chunks and only needed columns are kept, so a normal laptop is "
            "enough. Using the **local path** option avoids uploading through the browser.")
    source, label = _choose_source()

    if source is not None:
        try:
            header = read_header(source)
        except DatasetError as exc:
            st.error(f"Dataset problem: {exc}")
            source = None
        except Exception as exc:
            ui.error_box("reading the CSV header", exc)
            source = None

    if source is not None:
        st.success(f"CSV header read successfully: {len(header)} columns.")
        manual, final = _mapping_panel(header)
        has_text = [k for k in ("title", "description", "skills_desc") if k in final]
        if not has_text:
            st.error("No text column (title / description / skills) was found. Use the column mapping above.")
        else:
            if "description" not in final:
                st.warning("No description column found - skills can only be extracted from the other text columns.")
            if "listed_time" not in final:
                st.info("No date column - time trends and forecasts will be disabled (Mode B).")
            if not any(k in final for k in ("normalized_salary", "min_salary", "max_salary", "med_salary")):
                st.info("No salary columns - the salary model will be skipped.")

            if st.button("Preview first 2,000 rows", key="up_preview"):
                try:
                    _rewind(source)
                    df, mapping, _ = load_postings(source, manual or None, max_rows=2000)
                    s = summarize_dataset(df, mapping)
                    ui.kpi_row([("Rows previewed", f"{s['rows']:,}", "sample only"),
                                ("Valid dates", f"{s.get('date_valid_pct', 'n/a')}%", f"{s.get('date_min', '-')}"),
                                ("Has salary", f"{s.get('salary_available_pct', 'n/a')}%", "any salary column"),
                                ("Avg description", f"{s.get('avg_description_chars', 0):,} chars", "length")])
                    from src.data_loader import validate_dataset_quality
                    dq = validate_dataset_quality(df, mapping)
                    score_color = "green" if dq["data_quality_score"] >= 80 else "orange" if dq["data_quality_score"] >= 60 else "red"
                    st.markdown(f"**Data Mode: REAL DATASET** | Quality Score: **{dq['data_quality_score']}/100** ({dq['data_quality_grade']})")
                    for issue in dq["issues"]:
                        st.warning(issue)
                    if dq["temporal_suitability"] == "SUITABLE":
                        st.success(f"Temporal coverage: SUITABLE for forecasting ({dq['n_date_months']} months)")
                    elif dq["temporal_suitability"] == "NO_DATE_COLUMN":
                        st.info("No date column found - time trends will be disabled (Mode B).")
                    st.dataframe(df.head(10).drop(columns=[c for c in ("description",) if c in df.columns]))
                    nulls = pd.Series(s["null_pct"], name="% missing").rename_axis("column").reset_index()
                    ui.show_df(nulls, key="up_nulls")
                except DatasetError as exc:
                    st.error(f"Dataset problem: {exc}")
                except Exception as exc:
                    ui.error_box("previewing the file", exc)
                finally:
                    _rewind(source)

            st.subheader("Run the full analysis")
            c1, c2, c3 = st.columns(3)
            max_rows = c1.number_input("Max rows to read (0 = all)", min_value=0, value=0, step=10000, key="up_maxrows")
            train = c2.checkbox("Train ML models", value=True, key="up_train",
                                help="Untick for a faster run without the two ML models.")
            chunk = c3.selectbox("Chunk size", [25_000, 50_000, 100_000], index=1, key="up_chunk")
            if st.button("Run full analysis", key="up_run", type="primary"):
                bar = st.progress(0.0, text="Starting...")
                try:
                    _rewind(source)
                    run_pipeline(source, PROCESSED_DIR, MODELS_DIR,
                                 source_kind="real" if isinstance(source, Path) else "upload",
                                 source_label=f"file: {label}", max_rows=int(max_rows) or None,
                                 chunksize=int(chunk), train_models=bool(train), manual_mapping=manual or None,
                                 progress=lambda m, f: bar.progress(f, text=m))
                except DatasetError as exc:
                    bar.empty()
                    st.error(f"Dataset problem: {exc}")
                except MemoryError:
                    bar.empty()
                    st.error("Out of memory. Set 'Max rows' to a smaller number (e.g. 50,000) and try again.")
                except Exception as exc:
                    bar.empty()
                    ui.error_box("running the analysis", exc)
                else:
                    S.invalidate()
                    S.flash("success", f"Analysis of {label} finished. All pages now show your REAL dataset.")
                    st.session_state["_pending_source"] = S.REAL
                    st.session_state["_pending_page"] = "Home"
                    st.rerun()

    st.divider()
    st.subheader(f"Currently loaded: {kind}")
    if art is None:
        st.caption("Nothing processed yet for this data source.")
        return
    ui.source_banner(art["meta"])
    cr = art["meta"].get("clean_report", {})
    ui.kpi_row([("Rows read", f"{cr.get('rows_in', 0):,}", "before cleaning"),
                ("Rows kept", f"{cr.get('rows_out', 0):,}", "after cleaning"),
                ("Duplicates removed", f"{cr.get('dropped_duplicate_job_id', 0) + cr.get('dropped_duplicate_content', 0):,}", "job id + content"),
                ("With valid date", f"{cr.get('pct_with_valid_date', 0)}%", "of kept rows"),
                ("With salary", f"{cr.get('pct_with_salary', 0)}%", "usable yearly salary")])
    with st.expander("Detected columns and analysis-mode reasoning"):
        st.json(art["meta"].get("columns_detected", {}))
        for r in art["meta"].get("temporal", {}).get("reasons", []):
            st.write("- " + r)
