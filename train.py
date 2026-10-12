from pathlib import Path
import json
import warnings
from datetime import datetime, timezone
import hashlib
import platform
import sklearn
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier, ExtraTreesRegressor, RandomForestClassifier,
    RandomForestRegressor, IsolationForest, HistGradientBoostingRegressor
)
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, roc_auc_score, accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score, brier_score_loss, log_loss
)
from sklearn.calibration import calibration_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans

from config import (
    DATA_DIR, MODEL_DIR, REPORT_DIR, RANDOM_STATE,
    CHURN_INACTIVITY_DAYS, FEATURE_WINDOW_DAYS,
    FORECAST_HORIZONS, USE_API_AMOUNT_AS_DATA_COST
)
from data_loader import load_all
from feature_engineering import (
    build_enriched_transactions, daily_company_metrics,
    product_daily_metrics, user_features_asof
)

MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR.mkdir(exist_ok=True)

def save_json(obj, name):
    with open(REPORT_DIR / name, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)

def make_pipeline(X, estimator):
    cat = [c for c in X.columns if pd.api.types.is_string_dtype(X[c]) or str(X[c].dtype).startswith("category")]
    num = [c for c in X.columns if c not in cat]
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median"))
        ]), num),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", min_frequency=5))
        ]), cat)
    ])
    return Pipeline([("preprocess", pre), ("model", estimator)])

def chronological_split(df, frac=.80):
    return chronological_validation_split(df, "created_at", frac)

def chronological_validation_split(df, time_col, frac=.85):
    ordered = df.sort_values(time_col).reset_index(drop=True)
    times = ordered[time_col].drop_duplicates().sort_values().reset_index(drop=True)
    if len(times) > 1:
        i = min(len(times) - 1, max(1, int(len(times) * frac)))
        boundary = times.iloc[i]
        fit = ordered[ordered[time_col] < boundary]
        validation = ordered[ordered[time_col] >= boundary]
        if not fit.empty and not validation.empty:
            return fit, validation
    raise ValueError(f"Need at least two distinct {time_col} values for temporal validation.")


def positive_probability(model, X):
    """Resolve class 1 explicitly, including models trained on one class."""
    classes = list(model.classes_)
    if 1 not in classes:
        return np.zeros(len(X))
    return model.predict_proba(X)[:, classes.index(1)]


def probability_metrics(y, probability):
    observed, predicted = calibration_curve(y, probability, n_bins=10)
    return {
        "brier_score": float(brier_score_loss(y, probability)),
        "log_loss": float(log_loss(y, np.column_stack([1 - probability, probability]), labels=[0, 1])),
        "calibration_curve": [
            {"predicted_probability": float(p), "observed_frequency": float(o)}
            for p, o in zip(predicted, observed)
        ],
    }

def classifier_candidates():
    return {
        "random_forest": RandomForestClassifier(
            n_estimators=250, min_samples_leaf=3, class_weight="balanced",
            max_samples=.85, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=250, min_samples_leaf=3, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

def select_classifier_candidate(candidates, X_fit, y_fit, X_validation, y_validation):
    metric = "roc_auc" if y_validation.nunique() > 1 and y_fit.nunique() > 1 else "accuracy"
    scores = {}
    for name, candidate in candidates.items():
        candidate.fit(X_fit, y_fit)
        prediction = candidate.predict(X_validation)
        if metric == "roc_auc" and len(candidate.classes_) == 2:
            score = roc_auc_score(y_validation, positive_probability(candidate, X_validation))
        else:
            score = accuracy_score(y_validation, prediction)
        scores[name] = float(score)
    selected = max(scores, key=scores.get)
    return selected, scores, metric

def select_regressor_candidate(candidates, X_fit, y_fit, X_validation, y_validation):
    scores = {}
    for name, candidate in candidates.items():
        candidate.fit(X_fit, y_fit)
        scores[name] = float(mean_absolute_error(
            y_validation, candidate.predict(X_validation)
        ))
    return min(scores, key=scores.get), scores

def train_transaction_success(tx):
    # Only information plausibly available at/before transaction initiation.
    # Explicitly excludes status, response, balance_after, current wallet state,
    # identifiers, phone/IP/reference values and post-outcome fields.
    candidates = [
        "amount", "vending_medium", "transaction_channel", "service",
        "balance_before", "hour", "dayofweek", "day", "month", "is_weekend",
        "detail_network_id", "detail_data_plan_id", "data_plan_type_id",
        "size", "volume", "validity", "numeric_value"
    ]
    df = tx[tx["status"].astype(str).str.lower().isin(["success", "failed"])].dropna(subset=["created_at"]).copy()
    if df.created_at.nunique() < 3:
        raise ValueError("Transaction training needs at least three distinct timestamps.")
    cols = [c for c in candidates if c in df.columns]
    train, test = chronological_split(df)
    Xtr, ytr = train[cols], train["is_success"]
    Xte, yte = test[cols], test["is_success"]

    fit, validation = chronological_validation_split(train, "created_at")
    Xfit, yfit = fit[cols], fit["is_success"]
    Xvalidation, yvalidation = validation[cols], validation["is_success"]
    candidates = {
        name: make_pipeline(Xfit, estimator)
        for name, estimator in classifier_candidates().items()
    }
    selected, validation_scores, selection_metric = select_classifier_candidate(
        candidates, Xfit, yfit, Xvalidation, yvalidation
    )
    model = clone(candidates[selected])
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    prob = positive_probability(model, Xte)
    metrics = {
        "test_rows": len(test),
        "selected_model": selected,
        "validation_selection_metric": selection_metric,
        "validation_candidate_scores": validation_scores,
        "accuracy": float(accuracy_score(yte, pred)),
        "majority_baseline_accuracy": float((yte == ytr.mode().iloc[0]).mean()),
        "roc_auc": float(roc_auc_score(yte, prob)) if yte.nunique() > 1 else None,
        "classification_report": classification_report(yte, pred, labels=[0, 1], output_dict=True, zero_division=0),
        **probability_metrics(yte, prob)
    }
    model.fit(df[cols], df["is_success"])
    joblib.dump({"model": model, "features": cols},
                MODEL_DIR / "transaction_success.joblib")
    save_json(metrics, "transaction_success_metrics.json")

    out = test[["id", "created_at", "service", "amount", "status"]].copy()
    out["predicted_success_probability"] = prob
    out.to_csv(REPORT_DIR / "transaction_success_predictions.csv", index=False)
    return metrics

def build_user_snapshots(tx):
    first = tx["created_at"].min().floor("D")
    last = tx["created_at"].max().floor("D")
    # Monthly cutoffs with enough past and future.
    cutoffs = pd.date_range(
        first + pd.Timedelta(days=FEATURE_WINDOW_DAYS),
        last - pd.Timedelta(days=CHURN_INACTIVITY_DAYS),
        freq="30D"
    )
    frames = []
    for cutoff in cutoffs:
        feat = user_features_asof(tx, cutoff, FEATURE_WINDOW_DAYS)
        if feat.empty:
            continue
        future_end = cutoff + pd.Timedelta(days=CHURN_INACTIVITY_DAYS)
        future = tx[(tx["created_at"] >= cutoff) & (tx["created_at"] < future_end)]
        future_stats = future.groupby("user_id").agg(
            future_tx=("id", "count"),
            future_revenue=("revenue", "sum")
        ).reset_index()
        feat = feat.merge(future_stats, on="user_id", how="left").fillna(
            {"future_tx": 0, "future_revenue": 0}
        )
        feat["churn"] = (feat["future_tx"] == 0).astype(int)
        feat["cutoff"] = cutoff
        frames.append(feat)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def train_churn_and_clv(tx):
    snap = build_user_snapshots(tx)
    if len(snap) < 50 or snap["cutoff"].nunique() < 3:
        raise ValueError("Churn/value training needs 50 snapshot rows and three distinct cutoffs.")
    snap = snap.sort_values("cutoff")
    split_date = snap["cutoff"].drop_duplicates().sort_values().iloc[
        max(0, int(snap["cutoff"].nunique() * .8) - 1)
    ]
    train = snap[snap["cutoff"] <= split_date]
    test = snap[snap["cutoff"] > split_date]
    # Labels must be completely observed before the next partition begins.
    train = train[train.cutoff + pd.Timedelta(days=CHURN_INACTIVITY_DAYS) <= test.cutoff.min()]

    features = [c for c in [
        "tx_90d", "success_90d", "spend_90d", "avg_amount_90d",
        "std_amount_90d", "services_90d", "recency_days", "window_tenure_days"
    ] if c in snap]
    fit, validation = chronological_validation_split(train, "cutoff")
    fit = fit[fit.cutoff + pd.Timedelta(days=CHURN_INACTIVITY_DAYS) <= validation.cutoff.min()]
    if fit.empty:
        raise ValueError("Insufficient non-overlapping customer label windows.")
    candidates = classifier_candidates()
    selected, validation_scores, selection_metric = select_classifier_candidate(
        candidates, fit[features], fit["churn"],
        validation[features], validation["churn"]
    )
    churn = clone(candidates[selected])
    churn.fit(train[features], train["churn"])
    cp = churn.predict(test[features])
    cprob = positive_probability(churn, test[features])
    cm = {
        "accuracy": float(accuracy_score(test["churn"], cp)),
        "roc_auc": float(roc_auc_score(test["churn"], cprob))
            if test["churn"].nunique() > 1 else None,
        "churn_definition": f"No transaction in the next {CHURN_INACTIVITY_DAYS} days",
        "selected_model": selected,
        "validation_selection_metric": selection_metric,
        "validation_candidate_scores": validation_scores,
        **probability_metrics(test["churn"], cprob)
    }

    regression_fit, regression_validation = fit, validation
    regression_candidates = {
        "random_forest": RandomForestRegressor(
            n_estimators=250, min_samples_leaf=3, max_samples=.85,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "extra_trees": ExtraTreesRegressor(
            n_estimators=250, min_samples_leaf=3,
            random_state=RANDOM_STATE, n_jobs=-1
        ),
        "hist_gradient_boosting": HistGradientBoostingRegressor(
            max_iter=200, learning_rate=.05, l2_regularization=1.0,
            random_state=RANDOM_STATE
        ),
    }
    selected_value_model, value_validation_scores = select_regressor_candidate(
        regression_candidates,
        regression_fit[features], regression_fit["future_revenue"],
        regression_validation[features], regression_validation["future_revenue"]
    )
    clv = clone(regression_candidates[selected_value_model])
    clv.fit(train[features], train["future_revenue"])
    vp = np.maximum(0, clv.predict(test[features]))
    vm = {
        "target": f"next_{CHURN_INACTIVITY_DAYS}_day_successful_transaction_value",
        "mae": float(mean_absolute_error(test["future_revenue"], vp)),
        "rmse": float(mean_squared_error(test["future_revenue"], vp) ** .5),
        "r2": float(r2_score(test["future_revenue"], vp)),
        "selected_model": selected_value_model,
        "validation_selection_metric": "mae",
        "validation_candidate_scores": value_validation_scores
    }

    cm["test_rows"] = len(test)
    cm["majority_baseline_accuracy"] = float((test["churn"] == train["churn"].mode().iloc[0]).mean())
    # Evaluate on future snapshots first, then fit delivery models on all labels.
    churn.fit(snap[features], snap["churn"])
    clv.fit(snap[features], snap["future_revenue"])
    joblib.dump({"model": churn, "features": features}, MODEL_DIR / "churn_model.joblib")
    joblib.dump({"model": clv, "features": features}, MODEL_DIR / "customer_value_model.joblib")

    latest_cutoff = tx["created_at"].max() + pd.Timedelta(seconds=1)
    latest = user_features_asof(tx, latest_cutoff, FEATURE_WINDOW_DAYS)
    if not latest.empty:
        latest["churn_probability"] = positive_probability(churn, latest[features])
        latest["predicted_next_30d_value"] = np.maximum(0, clv.predict(latest[features]))
        latest.sort_values("churn_probability", ascending=False).to_csv(
            REPORT_DIR / "customer_churn_and_value.csv", index=False
        )
    save_json({"churn": cm, "customer_value": vm}, "customer_models_metrics.json")
    return {"churn": cm, "customer_value": vm}

def lagged_supervised(series, max_lag=28):
    s = series.astype(float).reset_index(drop=True)
    frame = pd.DataFrame({"y": s})
    for lag in [1, 2, 3, 7, 14, 21, 28]:
        frame[f"lag_{lag}"] = s.shift(lag)
    frame["roll7"] = s.shift(1).rolling(7).mean()
    frame["roll14"] = s.shift(1).rolling(14).mean()
    frame["roll28"] = s.shift(1).rolling(28).mean()
    return frame.dropna()

def recursive_forecast(series, horizon):
    hist = list(series.astype(float))
    sup = lagged_supervised(pd.Series(hist))
    X, y = sup.drop(columns="y"), sup["y"]
    model = HistGradientBoostingRegressor(
        max_iter=250, learning_rate=.05, random_state=RANDOM_STATE
    )
    model.fit(X, y)
    preds = []
    for _ in range(horizon):
        s = pd.Series(hist)
        row = {
            "lag_1": s.iloc[-1], "lag_2": s.iloc[-2], "lag_3": s.iloc[-3],
            "lag_7": s.iloc[-7], "lag_14": s.iloc[-14],
            "lag_21": s.iloc[-21], "lag_28": s.iloc[-28],
            "roll7": s.iloc[-7:].mean(),
            "roll14": s.iloc[-14:].mean(),
            "roll28": s.iloc[-28:].mean(),
        }
        p = max(0.0, float(model.predict(pd.DataFrame([row]))[0]))
        preds.append(p)
        hist.append(p)
    return model, preds

def weekly_forecast(series, horizon):
    return np.resize(series.iloc[-7:].to_numpy(dtype=float), horizon)


def evaluate_forecast(series, horizon):
    """Select on earlier origins; evaluate once on an untouched final horizon."""
    holdout_start = len(series) - horizon
    origins = [holdout_start - horizon * offset for offset in (3, 2, 1)]
    origins = [origin for origin in origins if origin >= 60]
    if holdout_start < 60 or not origins:
        return "weekly_baseline", {
            "selected_model": "weekly_baseline", "holdout_days": horizon,
            "mae": None, "weekly_baseline_mae": None,
            "warning": "Insufficient history for separate selection and evaluation.",
        }
    candidate_errors = {"gradient_boosting": [], "weekly_baseline": []}
    for origin in origins:
        history = series.iloc[:origin]
        actual = series.iloc[origin:origin + horizon]
        _, predictions = recursive_forecast(history, horizon)
        candidate_errors["gradient_boosting"].append(mean_absolute_error(actual, predictions))
        candidate_errors["weekly_baseline"].append(mean_absolute_error(actual, weekly_forecast(history, horizon)))
    scores = {name: float(np.mean(errors)) for name, errors in candidate_errors.items()}
    selected = min(scores, key=scores.get)
    history, actual = series.iloc[:holdout_start], series.iloc[holdout_start:]
    _, ml = recursive_forecast(history, horizon)
    baseline = weekly_forecast(history, horizon)
    prediction = ml if selected == "gradient_boosting" else baseline
    return selected, {
        "holdout_days": horizon,
        "selected_model": selected,
        "selection_folds": len(origins),
        "selection_candidate_mae": scores,
        "mae": float(mean_absolute_error(actual, prediction)),
        "gradient_boosting_mae": float(mean_absolute_error(actual, ml)),
        "weekly_baseline_mae": float(mean_absolute_error(actual, baseline)),
    }


def train_company_forecasts(tx):
    daily = daily_company_metrics(tx).sort_values("date")
    # The final recorded date may be an incomplete export day.
    # Exclude it from fitting and evaluation; forecasts start on that date.
    excluded_date = daily["date"].max()
    daily = daily[daily["date"] < excluded_date]
    if len(daily) < 7:
        raise ValueError("Forecasting needs at least seven observed complete days.")
    full_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(full_dates).fillna(0)
    daily.index.name = "date"
    metrics = ["transactions", "data_transactions", "successful_transactions", "failed_transactions",
               "transaction_value", "revenue", "active_users"]
    future = pd.DataFrame(index=pd.date_range(
        daily.index.max() + pd.Timedelta(days=1),
        periods=max(FORECAST_HORIZONS), freq="D"
    ))
    future.index.name = "date"

    backtest = {}
    for col in metrics:
        horizons = {}
        for horizon in FORECAST_HORIZONS:
            _, evaluation = evaluate_forecast(daily[col], horizon)
            horizons[str(horizon)] = evaluation
        # Use the 30-day selection consistently for the full daily trajectory.
        selection = horizons["30"]
        selected = selection.get("selected_model", "weekly_baseline")
        backtest[col] = {
            **selection, "horizons": horizons,
            "excluded_final_date": str(excluded_date.date()),
            "evaluation_scope": "Before component reconciliation",
        }
        model = None
        if selected == "gradient_boosting":
            model, preds = recursive_forecast(daily[col], max(FORECAST_HORIZONS))
        else:
            preds = weekly_forecast(daily[col], max(FORECAST_HORIZONS))
        future[col] = preds
        joblib.dump({"model": model, "selected_model": selected,
                     "weekly_history": daily[col].iloc[-7:].tolist(),
                     "training_end": str(daily.index.max()),
                     "selection_horizon": 30}, MODEL_DIR / f"forecast_{col}.joblib")

    if "successful_transactions" in future and "transactions" in future:
        # Reconcile separately forecast counts to their total.
        subtotal = future["successful_transactions"] + future["failed_transactions"]
        scale = np.minimum(1.0, future["transactions"] / subtotal.clip(lower=1))
        future["successful_transactions"] *= scale
        future["failed_transactions"] *= scale
        if "data_transactions" in future:
            future["data_transactions"] = np.minimum(
                future["data_transactions"], future["transactions"]
            )
        future["revenue"] = np.minimum(future["revenue"], future["transaction_value"])
        future["success_rate"] = (
            future["successful_transactions"] / future["transactions"].clip(lower=1)
        ).clip(0, 1)
        future["failure_rate"] = (
            future["failed_transactions"] / future["transactions"].clip(lower=1)
        ).clip(0, 1)

    future.to_csv(REPORT_DIR / "company_daily_forecast.csv")
    summary = {}
    for h in FORECAST_HORIZONS:
        part = future.iloc[:h]
        summary[f"next_{h}_days"] = {
            c: float(part[c].sum()) for c in
            ["transactions", "data_transactions", "transaction_value", "revenue"]
            if c in part
        }
        if "success_rate" in part:
            summary[f"next_{h}_days"]["avg_success_rate"] = float(part["success_rate"].mean())
            summary[f"next_{h}_days"]["avg_failure_rate"] = float(part["failure_rate"].mean())
    save_json(summary, "company_forecast_summary.json")
    save_json(backtest, "forecast_backtest_metrics.json")
    return summary

def data_plan_performance(tx, minimum_transactions=30):
    """Summarize observed plan outcomes without treating current catalog prices as history."""
    data = tx[
        tx["service"].eq("data")
        & tx["detail_data_plan_id"].notna()
        & tx["status"].astype(str).str.lower().isin(["success", "failed"])
    ].copy()
    output_path = REPORT_DIR / "data_plan_performance.csv"
    if data.empty:
        pd.DataFrame().to_csv(output_path, index=False)
        return {"warning": "No data transactions matched a data plan."}

    metadata = [
        column for column in [
            "size", "volume", "validity", "numeric_value", "data_plan_type_id",
            "plan_sell_amount", "plan_api_cost"
        ] if column in data.columns
    ]
    grouped = {
        "transactions": ("id", "count"),
        "successful_transactions": ("is_success", "sum"),
        "failed_transactions": ("is_failed", "sum"),
        "unique_customers": ("user_id", "nunique"),
        "sales_value": ("revenue", "sum"),
        "average_transaction_value": ("amount", "mean"),
        "estimated_gross_margin": (
            "estimated_gross_margin", lambda values: values.sum(min_count=1)
        ),
        "transactions_with_cost_data": (
            "estimated_gross_margin", lambda values: values.notna().sum()
        ),
    }
    for column in metadata:
        grouped[column] = (column, "first")
    plans = data.groupby("detail_data_plan_id").agg(**grouped).reset_index()
    plans["success_rate"] = (
        plans["successful_transactions"] / plans["transactions"].clip(lower=1)
    )
    plans["failure_rate"] = (
        plans["failed_transactions"] / plans["transactions"].clip(lower=1)
    )
    plans["estimated_margin_per_cost_matched_transaction"] = (
        plans["estimated_gross_margin"] /
        plans["transactions_with_cost_data"].replace(0, np.nan)
    )
    plans["reliable_sample"] = plans["transactions"] >= minimum_transactions
    plans["sample_note"] = np.where(
        plans["reliable_sample"],
        "At least 30 transactions",
        "Small sample; compare cautiously",
    )
    plans = plans.sort_values(
        ["reliable_sample", "success_rate", "estimated_margin_per_cost_matched_transaction", "transactions"],
        ascending=[False, False, False, False],
        na_position="last",
    ).reset_index(drop=True)
    plans["performance_rank"] = pd.NA
    reliable = plans["reliable_sample"]
    plans.loc[reliable, "performance_rank"] = np.arange(1, int(reliable.sum()) + 1)
    plans.to_csv(output_path, index=False)
    return {
        "plans_scored": int(len(plans)),
        "plans_with_reliable_sample": int(reliable.sum()),
        "minimum_transactions_for_ranking": minimum_transactions,
        "ranking_basis": "Observed success rate, then estimated margin per cost-matched transaction, then transaction count.",
    }

def product_trends(tx):
    p = product_daily_metrics(tx)
    maxdate = p["date"].max()
    rows = []
    for service, g in p.groupby("service"):
        g = g.set_index("date").sort_index()
        idx = pd.date_range(maxdate - pd.Timedelta(days=89), maxdate, freq="D")
        g = g.reindex(idx).fillna(0)
        recent = g.iloc[-30:]
        previous = g.iloc[-60:-30]
        for metric in ["transactions", "revenue", "users"]:
            r = recent[metric].sum()
            q = previous[metric].sum()
            growth = ((r - q) / q) if q > 0 else (1.0 if r > 0 else 0.0)
            rows.append({
                "service": service, "metric": metric,
                "previous_30d": float(q), "recent_30d": float(r),
                "growth_rate": float(growth),
                "trend": "rising" if growth > .10 else
                         "falling" if growth < -.10 else "stable"
            })
    out = pd.DataFrame(rows)
    out.to_csv(REPORT_DIR / "product_trends.csv", index=False)
    return out

def train_segmentation_and_anomalies(tx):
    cutoff = tx["created_at"].max() + pd.Timedelta(seconds=1)
    uf = user_features_asof(tx, cutoff, FEATURE_WINDOW_DAYS)
    features = ["tx_90d", "success_90d", "spend_90d", "avg_amount_90d",
                "services_90d", "recency_days"]
    features = [c for c in features if c in uf]
    if len(uf) >= 10:
        scaler = StandardScaler()
        X = scaler.fit_transform(uf[features])
        k = min(5, max(2, len(uf)//20))
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        uf["customer_segment"] = km.fit_predict(X)
        uf.to_csv(REPORT_DIR / "customer_segments.csv", index=False)
        joblib.dump({"scaler": scaler, "model": km, "features": features},
                    MODEL_DIR / "customer_segmentation.joblib")

    # Transaction anomaly model: amount + time + basic service encoding.
    a = tx[["id", "created_at", "user_id", "service", "amount", "hour",
            "dayofweek", "balance_before"]].copy()
    a["service_code"], service_categories = pd.factorize(a["service"])
    af = ["amount", "hour", "dayofweek", "balance_before", "service_code"]
    for c in af:
        a[c] = pd.to_numeric(a[c], errors="coerce")
        a[c] = a[c].fillna(a[c].median())
    iso = IsolationForest(contamination=.01, random_state=RANDOM_STATE, n_jobs=-1)
    a["anomaly_flag"] = iso.fit_predict(a[af])
    a["anomaly_score"] = -iso.score_samples(a[af])
    a.sort_values("anomaly_score", ascending=False).to_csv(
        REPORT_DIR / "transaction_anomalies.csv", index=False
    )
    joblib.dump({"model": iso, "features": af, "service_categories": list(service_categories)},
                MODEL_DIR / "anomaly_model.joblib")

def profit_report(tx):
    # Profit is ONLY estimated for successful data transactions with known API cost.
    x = tx[tx["estimated_gross_margin"].notna()].copy()
    if x.empty:
        return {"warning": "No defensible cost data available for profit estimation."}
    x["date"] = x["created_at"].dt.floor("D")
    d = x.groupby("date").agg(
        data_revenue=("revenue", "sum"),
        estimated_data_gross_margin=("estimated_gross_margin", "sum"),
        transactions=("id", "count")
    ).reset_index()
    d["estimated_margin_rate"] = (
        d["estimated_data_gross_margin"] / d["data_revenue"].replace(0, np.nan)
    )
    d.to_csv(REPORT_DIR / "estimated_data_profit_history.csv", index=False)
    return {
        "scope": "Data transactions only",
        "assumption": "data_plans.api_amount is treated as supplier/API cost",
        "estimated_total_gross_margin": float(x["estimated_gross_margin"].sum()),
        "successful_data_revenue_with_cost_match": float(x["revenue"].sum())
    }

def main():
    print("Loading all tables from:", DATA_DIR)
    data = load_all(DATA_DIR)
    tx = build_enriched_transactions(data, USE_API_AMOUNT_AS_DATA_COST)
    tx.to_csv(REPORT_DIR / "enriched_transactions.csv", index=False)

    results = {}
    print("1/7 Training transaction success/failure model...")
    results["transaction_success"] = train_transaction_success(tx)

    print("2/7 Training churn + customer-value models...")
    results["customers"] = train_churn_and_clv(tx)

    print("3/7 Training company forecasts...")
    results["forecast"] = train_company_forecasts(tx)

    print("4/7 Calculating product rise/fall trends...")
    trends = product_trends(tx)
    results["product_trends_rows"] = len(trends)

    results["data_plan_performance"] = data_plan_performance(tx)

    print("5/7 Training segmentation + anomaly detection...")
    train_segmentation_and_anomalies(tx)

    print("6/7 Building profit/margin report...")
    results["profit"] = profit_report(tx)

    print("7/7 Saving run summary...")
    save_json(results, "training_summary.json")
    from data_loader import DATA_FILES
    save_json({
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "transaction_rows": len(tx),
        "data_start": tx["created_at"].min(),
        "data_end": tx["created_at"].max(),
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "source_sha256": {f: hashlib.sha256((DATA_DIR / f).read_bytes()).hexdigest() for f in DATA_FILES.values()},
        "evaluation": "Chronological classifier/customer holdouts; customer label windows end before the next partition. Forecast candidates selected on earlier rolling origins and evaluated on separate final 7/30/90-day holdouts. Delivery forecasts use the 30-day selection. Classifiers/regressor refit on all available labelled rows.",
        "limitations": ["Historical plan attributes come from the current catalog.",
                         "Forecasts begin after the last complete recorded day; the final export day is excluded.",
                         "Revenue field means successful transaction value, not company earnings."]
    }, "run_manifest.json")
    print("\nDONE.")
    print("Models:", MODEL_DIR)
    print("Reports:", REPORT_DIR)

if __name__ == "__main__":
    main()
