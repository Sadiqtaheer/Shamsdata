from pathlib import Path

# Put this shams_ml folder INSIDE the same folder that contains your CSV files,
# or change DATA_DIR below.
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR if (PROJECT_DIR / "transactions.csv").exists() else PROJECT_DIR.parent

# If you instead keep CSVs in shams_ml/data/, uncomment:
# DATA_DIR = PROJECT_DIR / "data"

MODEL_DIR = PROJECT_DIR / "saved_models"
REPORT_DIR = PROJECT_DIR / "reports"

RANDOM_STATE = 42
CHURN_INACTIVITY_DAYS = 30
FEATURE_WINDOW_DAYS = 90
FORECAST_HORIZONS = (7, 30, 90)

# For Data plans only, estimated gross margin is calculated as:
# successful transaction amount - data_plans.api_amount
# Set False if api_amount is NOT your actual supplier/API cost.
USE_API_AMOUNT_AS_DATA_COST = True
