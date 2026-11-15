"""End-to-end pipeline: CSV -> clean -> skills -> analysis -> trends -> models -> saved artefacts.

`run_pipeline` is used by BOTH the command line (pipeline.py) and the Streamlit app.
`load_artifacts` reads everything back so the dashboard never re-processes the big CSV.
Every dataset-dependent decision (temporal mode, which models run) is made here at runtime.
"""
from __future__ import annotations

import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, save_npz

from . import analysis, trends
from .cleaning import clean_postings
from .config import (CHUNK_SIZE, MIN_GROUP_FOR_SKEW, MIN_POSTS_PER_PERIOD, MIN_ROLE_POSTINGS,
                     MIN_SKILL_POSTINGS, MODELS_DIR, PROCESSED_DIR)
from .data_loader import load_postings
from .models import train_role_classifier, train_salary_model
from .role_mapper import map_roles
from .skill_extractor import SkillExtractor, discover_candidate_terms

Progress = Optional[Callable[[str, float], None]]


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, (pd.Timestamp,)):
        return str(o)
    return str(o)


def _save_json(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, default=_json_default, indent=2), encoding="utf-8")


def _save_csv(df: Optional[pd.DataFrame], path: Path, index: bool = False) -> None:
    if df is not None and len(df):
        df.to_csv(path, index=index)
    elif path.exists():
        path.unlink()                         # remove stale file from a previous run


def run_pipeline(source, out_dir: Path = PROCESSED_DIR, models_dir: Path = MODELS_DIR,
                 source_kind: str = "real", source_label: Optional[str] = None,
                 max_rows: Optional[int] = None, chunksize: int = CHUNK_SIZE,
                 train_models: bool = True, manual_mapping: Optional[Dict[str, str]] = None,
                 progress: Progress = None) -> dict:
    """Process a CSV (path or file object) and save all artefacts to `out_dir`.

    source_kind: "real" | "upload" | "demo"  (demo = synthetic data, clearly labelled everywhere)
    """
    out_dir, models_dir = Path(out_dir), Path(models_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    def step(msg: str, frac: float) -> None:
        if progress:
            progress(msg, min(max(frac, 0.0), 1.0))

    # 1) load ----------------------------------------------------------------
    step("Reading CSV in chunks...", 0.02)
    raw, mapping, info = load_postings(
        source, manual_mapping, chunksize, max_rows,
        progress=lambda n: step(f"Loaded {n:,} rows...", 0.05 + min(n / 200_000, 1) * 0.10))

    # 2) clean ---------------------------------------------------------------
    step("Cleaning and preprocessing...", 0.17)
    df, clean_report = clean_postings(raw)
    del raw
    if df.empty:
        raise ValueError("No usable rows remained after cleaning.")
    df["role"] = map_roles(df["title"])

    # 3) skill extraction ----------------------------------------------------
    step("Extracting skills (NLP)...", 0.22)
    extractor = SkillExtractor()
    text = (df["skills_desc"] + " . " + df["description"]).tolist()      # title deliberately excluded
    X = extractor.extract_many(text, progress=lambda f: step("Extracting skills (NLP)...", 0.22 + 0.30 * f))
    skill_names = extractor.skill_names
    df["n_skills"] = np.asarray(X.sum(axis=1)).ravel().astype(int)

    # 4) descriptive analysis -------------------------------------------------
    step("Analysing skill demand...", 0.55)
    demand = analysis.skill_demand(X, skill_names, extractor.categories, min_postings=1)
    cat_demand = analysis.category_demand(X, skill_names, extractor.categories)
    cooc = analysis.cooccurrence(X, skill_names, top_n=40)
    role_profile, role_counts = analysis.role_skill_profile(X, df["role"], skill_names, MIN_ROLE_POSTINGS)
    skew, skew_note = analysis.seniority_skew(X, df["level_group"], skill_names, MIN_GROUP_FOR_SKEW)
    premium, premium_note = analysis.salary_premium(X, df["salary_annual"], skill_names)
    candidates = discover_candidate_terms(pd.Series(text), extractor)

    # 5) temporal gate + trends ----------------------------------------------
    step("Checking temporal coverage...", 0.62)
    temporal = trends.assess_temporal(df["listed_date"])
    trend_df = series = n_post = forecast_df = None
    backtest = {"status": "skipped", "reason": "Temporal mode is B (not enough time periods)."}
    if temporal["mode"] in ("A", "A_short"):
        step("Fitting skill trends and forecasts...", 0.66)
        series, n_post = trends.skill_share_series(X, skill_names, df["listed_date"], temporal["freq"],
                                                   MIN_POSTS_PER_PERIOD, top_n=80, min_postings=MIN_SKILL_POSTINGS)
        if len(series) >= 4:
            trend_df = trends.fit_trends(series, temporal["evidence_cap"])
            backtest = trends.backtest_forecast(series, holdout=2)
            if trend_df is not None and len(trend_df) and temporal["mode"] == "A":
                pick = pd.concat([trend_df[trend_df["status"] == "Emerging"].head(5),
                                  trend_df[trend_df["status"] == "Declining"].tail(5)])
                fc = []
                for skill in pick["skill"]:
                    f = trends.forecast_skill(series[skill], temporal["freq"], horizon=3,
                                              backtest_result=backtest)
                    if len(f) and "reason" not in f.columns:
                        fc.append(f.assign(skill=skill))
                forecast_df = pd.concat(fc, ignore_index=True) if fc else None
        else:
            temporal["reasons"].append("Fewer than 4 usable periods after filtering; trends skipped.")
    cross = trends.cross_sectional_signals(skew)
    obsolescence = trends.obsolescence_table(demand, trend_df, skew, MIN_SKILL_POSTINGS,
                                             evidence_cap=temporal["evidence_cap"])
    legacy = trends.legacy_watchlist_table(demand)

    # 6) models ---------------------------------------------------------------
    metrics = {}
    if train_models:
        step("Training role classifier...", 0.72)
        try:
            metrics["role_classifier"] = train_role_classifier(
                df, X, skill_names, models_dir, progress=lambda m: step(m, 0.78))
        except Exception as exc:
            metrics["role_classifier"] = {"status": "error", "error": str(exc), "trace": traceback.format_exc()[-800:]}
        step("Training salary model...", 0.90)
        try:
            metrics["salary_model"] = train_salary_model(df, X, skill_names, models_dir,
                                                         progress=lambda m: step(m, 0.93))
        except Exception as exc:
            metrics["salary_model"] = {"status": "error", "error": str(exc), "trace": traceback.format_exc()[-800:]}
    else:
        metrics = {"role_classifier": {"status": "skipped", "reason": "Model training disabled for this run."},
                   "salary_model": {"status": "skipped", "reason": "Model training disabled for this run."}}

    # 7) save -----------------------------------------------------------------
    step("Saving results...", 0.96)
    label = source_label or ("DEMO (synthetic sample data)" if source_kind == "demo" else "REAL DATASET")
    meta = {
        "data_source": label, "source_kind": source_kind, "is_demo": source_kind == "demo",
        "rows": int(len(df)), "unique_skills_found": int((demand["postings"] > 0).sum()),
        "postings_with_any_skill_pct": round(float((df["n_skills"] > 0).mean() * 100), 1),
        "date_min": temporal.get("date_min"), "date_max": temporal.get("date_max"),
        "analysis_mode": temporal["mode"], "temporal": {k: v for k, v in temporal.items() if k != "period_counts"},
        "columns_detected": mapping, "clean_report": clean_report, "load_info": info,
        "seniority_note": skew_note, "salary_premium_note": premium_note,
        "backtest": {
            **backtest,
            # surface new fields explicitly at the top-level of backtest for easy dashboard access
            "best_model": backtest.get("best_model"),
            "validation_method": backtest.get("validation_method"),
            "n_training_periods": backtest.get("n_training_periods"),
        },
        "n_roles_profiled": int(len(role_profile)),
        "role_counts": role_counts.to_dict(), "processed_at": datetime.now().isoformat(timespec="seconds"),
        "runtime_seconds": round(time.time() - t_start, 1),
    }
    clean_cols = ["job_id", "title", "company", "location", "experience_level", "level_group",
                  "work_type", "listed_date", "salary_annual", "role", "n_skills"]
    _save_csv(df[clean_cols], out_dir / "clean.csv")
    save_npz(out_dir / "skill_matrix.npz", X)
    _save_json(skill_names, out_dir / "skills.json")
    _save_json(meta, out_dir / "meta.json")
    _save_json(metrics, out_dir / "metrics.json")
    _save_csv(demand, out_dir / "demand.csv")
    _save_csv(cat_demand, out_dir / "category_demand.csv")
    _save_csv(cooc, out_dir / "cooccurrence.csv", index=True)
    _save_csv(role_profile, out_dir / "role_profile.csv", index=True)
    _save_csv(skew, out_dir / "skew.csv")
    _save_csv(premium, out_dir / "salary_premium.csv")
    _save_csv(candidates, out_dir / "candidate_skills.csv")
    _save_csv(temporal["period_counts"], out_dir / "temporal_counts.csv")
    _save_csv(trend_df, out_dir / "trend_table.csv")
    _save_csv(None if series is None else series.rename_axis("period"), out_dir / "trend_series.csv", index=True)
    _save_csv(forecast_df, out_dir / "forecast.csv")
    _save_csv(cross, out_dir / "cross_sectional.csv")
    _save_csv(obsolescence, out_dir / "obsolescence.csv")
    _save_csv(legacy, out_dir / "legacy_watchlist.csv")
    step("Done", 1.0)
    return load_artifacts(out_dir)


def _read(path: Path, **kw) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, **kw)
    except pd.errors.EmptyDataError:
        return None


def artifacts_exist(out_dir: Path = PROCESSED_DIR) -> bool:
    return (Path(out_dir) / "meta.json").exists() and (Path(out_dir) / "skill_matrix.npz").exists()


def load_artifacts(out_dir: Path = PROCESSED_DIR) -> dict:
    """Load everything saved by run_pipeline. Missing optional tables come back as None."""
    out_dir = Path(out_dir)
    if not artifacts_exist(out_dir):
        raise FileNotFoundError(f"No processed results in {out_dir}. Run `python pipeline.py` first.")
    art = {
        "meta": json.loads((out_dir / "meta.json").read_text(encoding="utf-8")),
        "metrics": json.loads((out_dir / "metrics.json").read_text(encoding="utf-8")) if (out_dir / "metrics.json").exists() else {},
        "skills": json.loads((out_dir / "skills.json").read_text(encoding="utf-8")),
        "matrix": load_npz(out_dir / "skill_matrix.npz").tocsr(),
        "clean": _read(out_dir / "clean.csv", parse_dates=["listed_date"]),
        "demand": _read(out_dir / "demand.csv"),
        "category_demand": _read(out_dir / "category_demand.csv"),
        "cooccurrence": _read(out_dir / "cooccurrence.csv", index_col=0),
        "role_profile": _read(out_dir / "role_profile.csv", index_col=0),
        "skew": _read(out_dir / "skew.csv"),
        "salary_premium": _read(out_dir / "salary_premium.csv"),
        "candidate_skills": _read(out_dir / "candidate_skills.csv"),
        "temporal_counts": _read(out_dir / "temporal_counts.csv"),
        "trend_table": _read(out_dir / "trend_table.csv"),
        "trend_series": _read(out_dir / "trend_series.csv", index_col=0, parse_dates=True),
        "forecast": _read(out_dir / "forecast.csv", parse_dates=["period"]),
        "cross_sectional": _read(out_dir / "cross_sectional.csv"),
        "obsolescence": _read(out_dir / "obsolescence.csv"),
        "legacy": _read(out_dir / "legacy_watchlist.csv"),
    }
    return art
