"""About page: architecture, methodology, limitations and viva Q&A (static explanations, no fake numbers)."""
import streamlit as st

from src import ui_components as ui

QA = [
    ("What problem does this project solve?",
     "Skills change quickly. The system measures from real job postings which skills are in demand, which look like they are "
     "fading or emerging, and shows a learner which skills they are missing for a target role."),
    ("Where does the data come from?",
     "The Kaggle LinkedIn Job Postings dataset. A small SYNTHETIC demo dataset is also included; it is always labelled DEMO."),
    ("How are skills extracted (NLP)?",
     "Text is lower-cased and tokenised, then 1-4 word phrases are matched against a curated skill dictionary (longest match "
     "first). A TF-IDF step suggests frequent terms that are not in the dictionary."),
    ("Why a dictionary and not a neural NER model?",
     "It is transparent, fast on 100k+ postings and needs no labelled training data. The limitation is that unknown skills "
     "are missed, which is why the TF-IDF discovery table exists."),
    ("Which ML models are used and why?",
     "Role-family classification (Logistic Regression, Linear SVM, Random Forest) on skills + TF-IDF features; salary regression "
     "(Ridge, Random Forest). Models are chosen by cross-validation on training data and judged on a held-out test set next to "
     "a baseline."),
    ("How do you know the model is genuine and not memorising?",
     "Stratified train/test split, TF-IDF fitted on training data only, model selection never touches the test set, and a "
     "majority-class baseline is shown. The job title is not a feature."),
    ("Why might accuracy still look high?",
     "Role labels come from title keyword rules and descriptions often repeat the title. This limitation is displayed in the app."),
    ("How are emerging and declining skills found?",
     "For each skill the share of postings per time period is regressed on time. p-values are corrected for multiple testing "
     "(Benjamini-Hochberg) and a minimum effect size is required."),
    ("What if the dataset covers only a few weeks?",
     "The temporal gate switches trends and forecasts OFF. Only a weak junior-vs-senior proxy is shown. This prevents "
     "fabricated trend claims."),
    ("How is the forecast validated?",
     "A back-test holds out the last periods, forecasts them, and compares the linear trend with naive baselines."),
    ("How is the obsolescence score defined?",
     "A transparent 0-100 mix of decline in demand over time, low overall demand and senior-leaning usage. Unavailable "
     "components are dropped and weights renormalised. The legacy watchlist is knowledge-based and shown separately."),
    ("How is the skill gap computed?",
     "For the target role, each skill's demand share is its weight; readiness is the weighted share covered by your skills. "
     "Missing skills are ranked by role demand x a trend factor."),
    ("What are the main limitations?",
     "Dictionary coverage, rule-based roles, postings are not the whole labour market, salary is only available for a subset, "
     "and associations are not causation."),
]


def render(art, kind):
    ui.page_header("About, Architecture & Viva Guide", "How the system works and how to explain it")
    st.markdown("### Architecture")
    st.code(
        "Kaggle CSV ──> data_loader (chunked, schema detection) ──> cleaning\n"
        "          ──> skill_extractor (NLP) + role_mapper ──> skill matrix (sparse)\n"
        "          ──> analysis (demand, co-occurrence, role profiles, seniority skew, salary premium)\n"
        "          ──> trends (temporal gate -> trends / forecast / obsolescence)\n"
        "          ──> models (role classifier, salary regressor, evaluation)\n"
        "          ──> saved artefacts (data/processed) ──> Streamlit pages\n"
        "Streamlit pages ──> gap_analysis ──> roadmap ──> reports (CSV / Excel / HTML / PDF)", language="text")
    st.markdown("### Honesty rules built into the system")
    st.markdown("- Nothing is hard-coded: every metric, count and trend is computed from the loaded file.\n"
                "- Demo data is synthetic, has planted trends, and is labelled everywhere.\n"
                "- Forecasts are disabled when the dates cannot support them.\n"
                "- Every model metric is shown next to a baseline.")
    st.markdown("### Viva questions and short answers")
    for i, (q, a) in enumerate(QA, 1):
        with st.expander(f"{i}. {q}"):
            st.write(a)
