"""Central configuration: every tunable constant lives here so runs are reproducible."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

SEED = 42

# ---- synthetic data ---------------------------------------------------------
N_PRODUCTS = 60
START_DATE = "2021-01-04"      # a Monday; weeks are labelled by their Monday
N_WEEKS = 208                  # 4 calendar years (first 53 weeks are lag warm-up)
RAW_FILE = DATA_DIR / "raw_sales.csv"
GROUND_TRUTH_FILE = DATA_DIR / "synthetic_ground_truth.csv"
CLEAN_FILE = DATA_DIR / "clean_weekly_sales.csv"

# ---- forecasting setup ------------------------------------------------------
HORIZONS = (1, 2, 3)           # direct multi-horizon: weeks ahead of the forecast origin
WARMUP = 53                    # first target week index with a full 52-week lag history
VAL_WEEKS = 26                 # validation block (strictly before the holdout)
TEST_WEEKS = 26                # final holdout (touched once)
CV_SPLITS = 5                  # walk-forward folds (inside the training block only)
CV_TEST_WEEKS = 10

# ---- reorder policy assumptions (all stated in the README) -------------------
LEAD_TIME = 2                  # weeks between placing and receiving an order
REVIEW_PERIOD = 1              # weeks between orders (review every week)
SERVICE_LEVEL = 0.95           # target cycle service level -> z = 1.645
HOLDING_RATE_ANNUAL = 0.20     # holding cost as a share of unit price per year
LOST_MARGIN = 0.30             # margin lost per unit of unmet demand (share of price)
BOOTSTRAP_B = 2000
