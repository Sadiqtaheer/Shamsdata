from pathlib import Path
import pandas as pd

DATA_FILES = {
    "transactions": "transactions.csv",
    "users": "users.csv",
    "wallets": "wallets.csv",
    "wallet_transactions": "wallet_transactions.csv",
    "funding_accounts": "funding_accounts.csv",
    "data_transactions": "data_transactions.csv",
    "data_plans": "data_plans.csv",
    "airtime_transactions": "airtime_transactions.csv",
    "cable_transactions": "cable_subscription_transactions.csv",
    "electricity_transactions": "electricity_bill_transactions.csv",
}

DATE_HINTS = ("created_at", "updated_at", "last_login", "deleted_at",
              "kyc_verified_at", "email_verified_at", "expire_date")

def load_all(data_dir: Path):
    data = {}
    missing = []
    for name, filename in DATA_FILES.items():
        path = Path(data_dir) / filename
        if not path.exists():
            missing.append(str(path))
            continue
        df = pd.read_csv(path, low_memory=False)
        for col in df.columns:
            if col in DATE_HINTS or col.endswith("_at"):
                df[col] = pd.to_datetime(df[col], errors="coerce")
        data[name] = df
    if missing:
        raise FileNotFoundError(
            "Missing required CSV files:\n" + "\n".join(missing) +
            "\n\nPlace shams_ml beside the CSVs, or edit DATA_DIR in config.py."
        )
    return data
