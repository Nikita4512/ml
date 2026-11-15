# AI-Based Digital Skill Obsolescence and Future Skill Gap Prediction System

A data-driven analytics platform that processes job posting datasets to extract skills via NLP, measure demand trends, forecast future demand, train ML classifiers, identify personalised skill gaps, and generate actionable learning roadmaps — all surfaced through an interactive Streamlit dashboard.

> **Honest disclosure:** This system provides heuristic-based indicators derived from job posting data. It does **not** produce validated predictions of actual labour market outcomes. All scores, trends, and risk indicators should be interpreted as exploratory analytics, not definitive forecasts.

---

## 1. Problem Statement

The digital skills landscape is evolving at an unprecedented pace. Technologies that dominated hiring requirements five years ago may be declining today, while entirely new competencies emerge with each product cycle. This rapid churn creates a three-sided challenge:

- **HR professionals** struggle to anticipate which skills will remain relevant when designing job descriptions, training budgets, and workforce plans.
- **Job seekers** lack visibility into whether their current skill set aligns with market demand or is trending toward obsolescence.
- **Educational institutions** need evidence-based signals to update curricula before graduates enter a misaligned labour market.

This system addresses the challenge by ingesting large-scale job posting data (e.g., Kaggle's LinkedIn Job Postings dataset) and executing an end-to-end analytical pipeline: skill extraction via curated NLP taxonomy, temporal trend detection with statistical correction, forecasting with backtested model selection, ML-based role classification and salary regression, personalised gap analysis with optional proficiency weighting, and phased learning roadmap generation. Results are delivered as interactive dashboards and exportable reports (PDF, Excel, HTML, CSV).

A synthetic demo dataset of 6,000 fabricated postings with planted trends is included so that every feature can be explored without requiring access to a real dataset.

---

## 2. Architecture Overview

```
CSV File ──> Chunked Loader ──> Cleaner ──> NLP Skill Extractor
                                                    │
                ┌───────────────────────────────────┘
                ▼
        Demand Analysis ──> Temporal Gate ──> Trends / Forecast
                │                                    │
                ▼                                    ▼
        Role Mapper ──> ML Models          Skill Market Risk Score
                │           │
                ▼           ▼
        Gap Analyzer ──> Roadmap ──> Reports (PDF/Excel/HTML)
                                         │
                                    Streamlit Dashboard
```

The pipeline is orchestrated by `src/orchestrator.py`, which manages stage sequencing, intermediate serialisation, and real/demo data-source routing. Each stage is independently testable and produces deterministic outputs for a given input.

---

## 3. Technology Stack

| Component | Technology | Purpose |
|---|---|---|
| Dashboard | Streamlit ≥ 1.40 | Interactive web-based exploration and reporting UI |
| NLP | Regex + curated taxonomy (180 skills) | Dictionary-based skill extraction (not ML NER) |
| ML Classification | scikit-learn (LogisticRegression, LinearSVC, RandomForest) | Job role classification from posting text |
| ML Regression | scikit-learn (Ridge, RandomForest) | Salary prediction from skills and metadata |
| Forecasting | scipy, numpy | Linear trend, exponential smoothing, naïve and moving-average baselines |
| Visualisation | Plotly ≥ 5.18 | Interactive, publication-quality charts |
| Reports | ReportLab, openpyxl, Jinja2-style HTML | PDF, Excel, and HTML report generation |
| Data Processing | pandas ≥ 2.0, scipy.sparse | Efficient chunked loading and sparse-matrix skill representation |

**Runtime:** Python 3.10, 3.11, or 3.12.

---

## 4. Data Flow

The pipeline executes the following stages sequentially:

1. **Chunked CSV Loading** — Reads arbitrarily large files in chunks (handles 100 k+ rows). Auto-detects the dataset schema from 14 known column aliases.
2. **Data Quality Validation** — Computes a quality score (0–100), letter grade, temporal-suitability verdict, and currency warnings before any transformation.
3. **Cleaning** — Strips HTML from descriptions, detects and flags duplicates, parses heterogeneous date formats, and normalises salary fields.
4. **NLP Skill Extraction** — Matches cleaned text against a curated dictionary of 180 skills with aliases using longest-match-first phrase lookup. Output is a sparse binary matrix (postings × skills).
5. **TF-IDF Discovery** — Extracts frequent 1–2 gram terms that are **not** in the taxonomy and surfaces them as candidate new skills (suggestions only; not added automatically).
6. **Role Mapping** — Assigns each posting to a role family via keyword rules on job titles, producing confidence scores and flagging weak-label provenance.
7. **Demand Analysis** — Computes skill counts, demand shares, co-occurrence matrices, role profiles, seniority skew, and salary premiums.
8. **Temporal Gating** — Evaluates date coverage: **Mode A** (≥ 6 months of data → trends and forecasting enabled) vs. **Mode B** (narrow window → cross-sectional analysis only; trends disabled by design).
9. **Trend Detection** — Fits per-skill trend lines with Benjamini–Hochberg correction for multiple testing. Computes the **Skill Market Risk Score**, a weighted heuristic combining demand level, trend direction, and seniority skew.
10. **Forecasting with Backtesting** — Evaluates four candidate models (naïve, moving average, linear trend, exponential smoothing) on a chronological holdout; selects the winner by validation MAE.
11. **ML Model Training** — Trains a role classifier (TF-IDF + skill features, 4 algorithm candidates, stratified 5-fold CV) and a salary regressor (Ridge and Random Forest with mean baseline).
12. **Personalised Skill-Gap Analysis** — Compares the user's declared skills against role demand profiles, with optional proficiency weighting.
13. **Phased Learning Roadmap** — Generates a prioritised, multi-phase study plan based on gap severity and skill prerequisites.
14. **Report Generation** — Exports results as PDF, Excel, HTML, or CSV with full data-source attribution.

---

## 5. NLP Methodology

Skill extraction uses a **transparent, deterministic dictionary-matching** approach rather than ML-based Named Entity Recognition (NER).

**Taxonomy.** A curated dictionary of 180 skills, each with canonical name, category, and alias list. Aliases handle common variations such as `C++`, `C#`, `.NET`, `Node.js`, `CI/CD`, and abbreviated forms.

**Extraction Pipeline:**

1. **Text normalisation** — lowercase, punctuation handling, whitespace collapsing.
2. **Tokenisation** — sentence and word-level tokenisation.
3. **Longest-match-first phrase lookup** — scans 1- to 4-word windows, preferring longer matches to avoid partial hits (e.g., "machine learning" is matched as one skill, not two unrelated tokens).
4. **Sparse binary matrix output** — each posting is represented as a binary vector over the 180-skill vocabulary; stored as a `scipy.sparse` matrix for memory efficiency.

**TF-IDF Discovery.** After dictionary extraction, a TF-IDF pass (1–2 grams) over the residual text identifies high-frequency terms absent from the taxonomy. These are surfaced in the dashboard as candidate additions — they are **not** automatically incorporated.

**Limitations.** Skills not present in the 180-entry taxonomy are invisible to the extractor. TF-IDF discovery partially mitigates this but requires manual review. The approach is intentionally transparent and reproducible at the cost of recall.

---

## 6. ML Methodology

### 6.1 Role Classifier

| Aspect | Detail |
|---|---|
| **Features** | Sparse skill matrix concatenated with TF-IDF of the job description. The title column is **excluded** from the TF-IDF vocabulary to reduce label leakage. |
| **Candidates** | Logistic Regression, Linear SVC, Random Forest |
| **Selection** | Stratified 5-fold cross-validation on the training split only |
| **Evaluation** | Held-out 20 % test set; majority-class baseline comparison |
| **Metrics** | Accuracy, macro-F1, per-class precision/recall, confusion matrix |

> [!CAUTION]
> Role labels are **weak labels** generated by keyword rules applied to job titles — they are **not** human-validated ground truth. Reported classification metrics measure the model's agreement with these heuristic labels. Because job descriptions frequently echo the title, accuracy figures can be **optimistically inflated**. This limitation is disclosed explicitly in the UI.

### 6.2 Salary Regressor

| Aspect | Detail |
|---|---|
| **Features** | Skill vector + one-hot encoded role + experience level |
| **Candidates** | Ridge Regression, Random Forest, mean-salary baseline |
| **Guard** | Only trained when ≥ 500 postings contain salary data |
| **Metrics** | MAE, RMSE, R² |

---

## 7. Evaluation Methodology

Rigorous evaluation is central to the pipeline's credibility. The following practices are enforced:

- **Baselines.** Every model is compared against a naïve baseline: majority class (classification), global mean (regression), naïve persistence and moving average (forecasting).
- **Metrics.** Classification: accuracy, macro-F1, per-class precision/recall. Regression: MAE, RMSE, R². Forecasting: MAE, RMSE, sMAPE.
- **Chronological holdout.** Forecasting evaluation uses a last-N-period holdout; walk-forward expansion when data length permits.
- **Model selection.** Performed exclusively on the training split via cross-validation. The test set is never used for fitting or model selection.
- **Forecast clipping.** All demand-share predictions are clipped to the valid [0 %, 100 %] range.

> [!IMPORTANT]
> Classification evaluation is conducted against heuristic labels derived from job-title keyword rules. Strong test-set performance does **not** establish genuine human-validated role classification accuracy. This caveat is surfaced in every relevant dashboard page.

---

## 8. Limitations

The following limitations should be considered when interpreting results:

1. **Dictionary-limited NLP.** Skills absent from the 180-entry taxonomy are undetectable. TF-IDF discovery partially mitigates this by surfacing frequent uncaptured terms as candidates.
2. **Weak labels.** Role labels are produced by keyword rules on job titles, not by human annotators. Classifier metrics reflect agreement with these rules, not ground-truth role assignment.
3. **Heuristic risk score.** The Skill Market Risk Score is a weighted composite of demand level, trend direction, and seniority skew. It is **not** a statistically validated obsolescence probability.
4. **Trend extrapolation ≠ prediction.** Linear and exponential-smoothing trend extrapolation assumes that historical patterns continue. These are not causal models and carry no guarantee of future accuracy.
5. **Coverage ≠ employability.** Skill-coverage scores measure overlap between a user's skills and a role's demand profile. They do **not** represent a probability of employment.
6. **Synthetic demo data.** The bundled demo dataset is entirely synthetic with artificially planted trends. It does **not** represent actual market conditions and exists solely to demonstrate system features.
7. **Currency not validated.** Salary analysis assumes a single, homogeneous currency across all postings. No automatic currency detection or conversion is performed.
8. **Temporal data requirement.** At least 6 months of dated postings (≥ 50 per month) are needed for meaningful trend and forecast analysis. The system automatically disables these features when data is insufficient (Mode B).
9. **Correlational, not causal.** Salary–skill associations are observational correlations. The system cannot determine whether possessing a skill **causes** higher compensation.
10. **No sklearn Pipeline for CV.** TF-IDF vectorisation is fitted on the training split prior to cross-validation folds. For fully unbiased fold-level evaluation, an integrated `sklearn.Pipeline` would be required.
11. **Job postings ≠ labour market.** Postings represent employer demand as expressed on a single platform, not the complete state of the labour market.
12. **Single-platform bias.** Results reflect the source platform's demographic, geographic, and industry composition and may not generalise to other platforms or offline hiring.

---

## 9. Installation

### Prerequisites

- Python **3.10**, **3.11**, or **3.12** (ensure "Add to PATH" is ticked during installation).
- A terminal with access to `pip` and `venv`.

### Setup

```powershell
# 1. Unzip the project and open a terminal in the project root

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4. Verify the installation by running the test suite
python -m pytest tests/ -v
```

> [!NOTE]
> **PowerShell execution policy.** If `venv\Scripts\activate` fails with a script-execution error, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` and try again.

> [!TIP]
> **VS Code users.** Open the project folder in VS Code, select the `venv` interpreter (`Ctrl+Shift+P` → *Python: Select Interpreter*), and use the integrated terminal for all subsequent commands.

---

## 10. Execution

### Demo Mode (no dataset required)

```powershell
streamlit run app.py
```

Select **"Demo (synthetic)"** in the sidebar. The system generates 6,000 synthetic postings with planted trends and runs the full pipeline in-memory.

### Real Dataset Mode

```powershell
# 1. Place your CSV in the raw data folder
#    Copy your file to:  data\raw\

# 2. Inspect the dataset before processing
python inspect_dataset.py

# 3. (Optional) Trial run with a row limit
python pipeline.py --max-rows 20000

# 4. Full pipeline run
python pipeline.py

# 5. Launch the dashboard
streamlit run app.py
#    Select "Real dataset" in the sidebar
```

**Pipeline options:**

| Flag | Effect |
|---|---|
| `--max-rows N` | Process only the first *N* rows (useful for trial runs) |
| `--no-models` | Skip ML model training (faster iteration) |
| `--postings "path"` | Specify a custom CSV path instead of scanning `data\raw\` |

### Analysis Modes (automatic)

The system selects the analysis mode automatically based on temporal coverage:

| Mode | Condition | Behaviour |
|---|---|---|
| **Mode A** | ≥ 6 months of dated postings with sufficient density | Full analysis: demand, trends, forecasting, backtesting |
| **Mode B** | Narrow date window or undated postings | Cross-sectional analysis only; trend and forecast pages are disabled |

Mode B is a **design decision**, not a bug — extrapolating trends from insufficient data would produce misleading results.

---

## 11. Dataset Requirements

### Required and Optional Columns

The loader auto-detects columns from 14 known aliases. The table below lists the canonical names:

| Column | Required? | Purpose |
|---|---|---|
| `title` | **Yes** | Role mapping, display |
| `description` | **Yes** | Skill extraction, TF-IDF features |
| `listed_time` / `date_posted` | For trends | Temporal analysis (determines Mode A vs. Mode B) |
| `normalized_salary` / `min_salary` / `max_salary` | For salary model | Salary regression training |
| `experience_level` | Optional | Seniority analysis and skew metrics |
| `company` | Optional | Display and filtering |
| `location` | Optional | Display and filtering |

### Size Guidelines

| Requirement | Minimum | Recommended |
|---|---|---|
| General analysis | 1,000 postings | 5,000+ postings |
| Temporal forecasting (Mode A) | 6 months of dated postings, ≥ 50 per month | 12+ months |
| Salary regression | 500 postings with salary data | 1,000+ |

---

## Folder Structure

```
skill_gap_project/
├── app.py                          # Streamlit application entry point
├── pipeline.py                     # CLI pipeline runner
├── inspect_dataset.py              # Pre-processing dataset inspector
├── requirements.txt                # Python dependencies
├── README.md
├── setup_windows.bat               # One-click environment setup (Windows)
├── run_app.bat                     # One-click app launcher (Windows)
├── .streamlit/config.toml          # Streamlit theme and server config
│
├── data/
│   ├── raw/                        # ← Place your CSV here
│   ├── processed/                  # Pipeline output (real data)
│   ├── processed_demo/             # Pipeline output (demo data)
│   └── sample/demo_postings.csv    # Bundled synthetic demo dataset
│
├── models/                         # Saved ML model artefacts
│
├── src/                            # Backend modules
│   ├── config.py                   # Thresholds, paths, theme constants
│   ├── data_loader.py              # Chunked CSV loading, schema detection, quality validation
│   ├── cleaning.py                 # HTML removal, duplicates, date parsing, salary normalisation
│   ├── skills_taxonomy.py          # 180 skills + aliases + legacy watchlist + prerequisites
│   ├── skill_extractor.py          # NLP skill extraction + TF-IDF discovery
│   ├── role_mapper.py              # Job title → role family (rule-based, with confidence)
│   ├── analysis.py                 # Demand, co-occurrence, role profiles, seniority skew, salary premium
│   ├── trends.py                   # Temporal gate, trends, forecast, backtest, Skill Market Risk Score
│   ├── models.py                   # Role classifier + salary regressor (with baselines)
│   ├── gap_analysis.py             # Personalised gap with optional proficiency weighting
│   ├── roadmap.py                  # Phased learning roadmap generator
│   ├── reports.py                  # Excel / CSV / HTML report builders
│   ├── orchestrator.py             # Full pipeline orchestration; save/load results
│   ├── demo_data.py                # Synthetic demo-data generator
│   ├── frontend_logic.py           # Helper functions for dashboard pages
│   ├── report_sections.py          # Report content assembly (HTML/PDF/Excel)
│   ├── pdf_report.py               # PDF generation via ReportLab
│   ├── app_state.py                # Session state and real/demo data-source handling
│   └── ui_components.py            # CSS theme, KPI cards, Plotly styling helpers
│
├── views/                          # Frontend pages
│   ├── home.py                     # Landing / overview page
│   ├── upload.py                   # Dataset upload and quality report
│   ├── demand.py                   # Skill demand analysis
│   ├── trends.py                   # Trends and forecasting
│   ├── models.py                   # ML model performance
│   ├── gap.py                      # Personalised skill-gap analysis
│   ├── roadmap.py                  # Learning roadmap
│   ├── reports.py                  # Report download page
│   └── about.py                    # Methodology and limitations
│
└── tests/
    ├── test_audit_corrections.py   # 41 pytest cases covering all 5 audit corrections
    ├── test_backend_smoke.py       # End-to-end backend smoke test
    └── ui_smoke_mock.py            # Page rendering against mocked Streamlit/Plotly
```

---

*This project was developed as part of academic research into AI-driven labour market analytics. For questions, issues, or contributions, please open an issue in the project repository.*
