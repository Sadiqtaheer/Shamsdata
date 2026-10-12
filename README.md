# Shams ML Business Intelligence Project

## Dashboard

This working folder contains the CSV data, training scripts, trained models, and a Streamlit dashboard.
From PowerShell in this folder, run:

    .\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501

Then open http://localhost:8501. Alternatively run `start_dashboard.ps1`.
The dashboard runs on this computer with Operations, Churn risk, and Business
projection tabs. More analytics contains Forecasts, Data plan performance,
Customers, Product trends, and Anomalies. Historical filters apply
to Operations; headline metrics, forecasts and customer scores use the complete dataset.
Churn tables support customer name/ID search and pagination. Only customer IDs
and names are loaded from the user table for display.
Business projection uses simple trends from completed calendar months, excluding
the final recorded month; trained daily forecasts remain in More analytics.
Amounts are displayed in Nigerian naira. Successful transaction value includes
wallet funding and must not be interpreted as company earnings.

The dashboard monitors the ten source CSVs every 5 seconds. When their contents
change, it waits until the files have been unchanged for 15 seconds, retrains the
models, and refreshes the dashboard automatically. If the dashboard is closed
while data changes, retraining starts the next time it opens. Add rows to the
existing tables without changing their columns; no code edits are needed. The
dashboard shows retraining status, and failures are logged to
`reports/auto_training.log`.

To start training manually:

    .\.venv\Scripts\python.exe train.py

Check dashboard behavior:

    .\.venv\Scripts\python.exe verify_dashboard.py

`reports/run_manifest.json` records training time, data coverage, package versions,
and SHA-256 hashes of the ten source tables. `forecast_backtest_metrics.json`
compares six forecasts with a repeating-week baseline on the last 30 days.
Classifiers and the value regressor are evaluated chronologically and then
refitted on all available labelled rows. Dynamic current account settings are
excluded from transaction prediction; current plan metadata is still a limitation.

The Python environment here uses the bundled Codex Python runtime. On another
computer, create a fresh environment and install `requirements.txt`.

This folder is designed to be dropped directly into your dataset folder.

## Recommended folder layout

    Shamsdataset/
    ├── airtime_transactions.csv
    ├── cable_subscription_transactions.csv
    ├── data_plans.csv
    ├── data_transactions.csv
    ├── electricity_bill_transactions.csv
    ├── funding_accounts.csv
    ├── transactions.csv
    ├── users.csv
    ├── wallets.csv
    ├── wallet_transactions.csv
    └── shams_ml/
        ├── train.py
        ├── predict.py
        ├── config.py
        ├── data_loader.py
        ├── feature_engineering.py
        ├── requirements.txt
        ├── saved_models/
        └── reports/

The code uses all 10 business data tables. PHONE_FIELD_MANIFEST.csv and README.txt
are documentation/metadata, so they are not ML training tables.

## Setup

Open a terminal inside `shams_ml`:

    python -m venv .venv

Windows:

    .venv\Scripts\activate

macOS/Linux:

    source .venv/bin/activate

Then:

    pip install -r requirements.txt
    python train.py

## Models/analytics produced

1. Transaction success/failure classifier
2. Customer churn classifier
3. Next-30-day customer value model
4. Company transaction-volume forecast
5. Successful transaction-value/revenue forecast
6. Success/failure-rate forecast
7. Product/service rising/falling trend analysis
8. Customer segmentation
9. Transaction anomaly detection
10. Data-product gross-margin history where cost information exists
11. Data-plan historical outcome ranking and daily data-transaction forecast

Data-plan comparisons rank plans by observed success rate after a configurable
minimum sample size. Margin is estimated only when current catalog API cost is
available; current catalog plan details do not represent historical pricing.

## Reports

After training, `reports/` contains:

- enriched_transactions.csv
- transaction_success_predictions.csv
- transaction_success_metrics.json
- customer_churn_and_value.csv
- customer_models_metrics.json
- company_daily_forecast.csv
- company_forecast_summary.json
- data_plan_performance.csv
- product_trends.csv
- customer_segments.csv
- transaction_anomalies.csv
- estimated_data_profit_history.csv
- training_summary.json

## Important profit note

A transaction amount is NOT automatically profit.

The project only estimates gross margin for data-plan transactions where a matching
`data_plans.api_amount` exists. It assumes:

    estimated gross margin = successful transaction amount - api_amount

This assumption is controlled by `USE_API_AMOUNT_AS_DATA_COST` in config.py.
If `api_amount` does not represent the company's actual supplier/API cost, turn
that setting off and replace it with your true cost data.

For airtime, cable, electricity, wallet funding, etc., add actual supplier cost,
commission, gateway fee and operating-cost data before calling the result profit.

## Security / leakage protection

Customer CSVs, generated reports, trained models, and local secrets are excluded
from version control. Supply the CSV files locally before training; this repository
contains the application and ML source code, not the customer dataset.

The models intentionally do NOT train on raw passwords, API keys/tokens,
transaction PINs, BVN/NIN, bank account numbers, phone numbers, email addresses,
IP addresses, API responses, or similar secrets/identifiers.

The success/failure model also excludes post-outcome information such as
`balance_after` and `api_response`, because those can leak the answer.

## Forecast interpretation

Forecasts are estimates, not guarantees. The project uses chronological data so
future observations are not randomly mixed into past training data. Retrain
regularly as new transactions arrive.

Forecasts exclude the final recorded date because it may be a partial export day.
For each measure, gradient boosting and a repeating-week baseline are compared
on up to three earlier rolling windows. Separate final holdouts report errors
for 7, 30, and 90 days. The delivered daily trajectory consistently uses the
30-day selection; the other horizons provide additional validation. Forecast
errors are measured before totals are reconciled. Forecasts begin after the
last complete recorded day, rather than today's date.

Saved `forecast_*.joblib` files now contain a bundle with `selected_model`,
`model` (only for gradient boosting), `weekly_history`, and training metadata.
The dashboard reads the generated forecast CSV rather than these bundles.

Classifier reports include Brier score, log loss, and calibration curves.
These measure probability reliability; probabilities have not been recalibrated.
Temporal validation keeps equal timestamps together and customer label windows
must end before the next partition begins. Insufficient dates produce an explicit
training error instead of a misleading split within the same date.

Run `python verify_ml.py` for ML boundary checks and
`python verify_dashboard.py` for dashboard integration checks after training.

## Churn definition

By default, churn means a previously active customer makes no transaction during
the next 30 days. Change `CHURN_INACTIVITY_DAYS` in `config.py` if the business
uses a different definition.
