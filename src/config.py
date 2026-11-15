"""Central configuration: paths, thresholds and theme colours.

Change a number here and every module picks it up.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLE_DIR = DATA_DIR / "sample"
MODELS_DIR = ROOT / "models"
for _d in (RAW_DIR, PROCESSED_DIR, SAMPLE_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---- Loading -------------------------------------------------------------
CHUNK_SIZE = 50_000          # rows per chunk when reading the big CSV

# ---- Analysis thresholds -------------------------------------------------
MIN_SKILL_POSTINGS = 30      # ignore skills seen in fewer postings than this
MIN_ROLE_POSTINGS = 50       # a role needs this many postings to get a profile
MIN_GROUP_FOR_SKEW = 200     # min junior AND senior postings for seniority skew

# ---- Temporal (trend) gate ----------------------------------------------
MIN_MONTHLY_PERIODS = 6      # need >= this many qualifying months for Mode A
MIN_WEEKLY_PERIODS = 8       # need >= this many qualifying weeks for Mode A-short
MIN_POSTS_PER_PERIOD = 200   # a period only counts if it has this many postings
TREND_ALPHA = 0.05           # significance level (after FDR correction)
TREND_MIN_AVG_SHARE = 0.005  # skills averaging < 0.5% of postings are too noisy
TREND_MIN_REL_SLOPE = 0.02   # >= 2% relative change per period to call a trend

# ---- Machine learning ----------------------------------------------------
RANDOM_STATE = 42
MIN_CLASS_N = 100            # role classes with fewer postings are dropped
MAX_TRAIN_ROWS = 60_000      # cap rows used for the role classifier
CV_MAX_ROWS = 20_000         # cap rows used for model comparison (cross-validation)
MIN_SALARY_ROWS = 500        # need this many salary rows to train a salary model
SALARY_MIN, SALARY_MAX = 15_000, 1_000_000   # annual salary sanity bounds

# ---- Obsolescence score weights (shown in the UI, renormalised if a part is missing)
OBSOLESCENCE_WEIGHTS = {"trend": 0.50, "demand": 0.25, "skew": 0.25}

# ---- Theme ---------------------------------------------------------------
THEME = {
    "bg": "#0B1026", "panel": "#131A3A", "primary": "#7C3AED",
    "accent": "#A78BFA", "text": "#E5E7EF", "muted": "#9CA3C7",
    "good": "#34D399", "warn": "#FBBF24", "bad": "#F87171",
}
