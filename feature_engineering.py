import numpy as np
import pandas as pd

SENSITIVE_USER_COLUMNS = [
    "name", "username", "api_key", "phone", "photo", "address", "email",
    "bvn", "nin", "password", "api_token", "original_token",
    "transaction_pin", "remember_token", "account_reference", "last_login_ip"
]

def _service_name(type_value):
    s = str(type_value)
    if "DataTransaction" in s: return "data"
    if "AirtimeTransaction" in s: return "airtime"
    if "WalletTransaction" in s: return "wallet"
    if "CableSubscriptionTransaction" in s: return "cable"
    if "ElectricityBillTransaction" in s: return "electricity"
    return "other"

def build_enriched_transactions(data, use_api_cost=True):
    """
    Creates ONE safe analytical transaction table while using the service tables.
    It deliberately excludes raw secrets, phone numbers, account numbers, IPs,
    API responses and other high-risk/leaky fields from ML features.
    """
    tx = data["transactions"].copy()
    tx["created_at"] = pd.to_datetime(tx["created_at"], errors="coerce")
    tx["service"] = tx["transactionable_type"].map(_service_name)
    tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce").fillna(0)
    tx["is_success"] = (tx["status"].astype(str).str.lower() == "success").astype(int)
    tx["is_failed"] = (tx["status"].astype(str).str.lower() == "failed").astype(int)
    tx["revenue"] = np.where(tx["is_success"].eq(1), tx["amount"], 0.0)

    # Data transaction -> plan enrichment.
    dt = data["data_transactions"][["id", "data_plan_id", "mobile_network_id"]].copy()
    dt = dt.rename(columns={
        "id": "transactionable_id",
        "data_plan_id": "detail_data_plan_id",
        "mobile_network_id": "detail_network_id"
    })
    dp_cols = ["id", "size", "volume", "validity", "numeric_value", "amount",
               "api_amount", "active", "data_plan_type_id"]
    dp = data["data_plans"][[c for c in dp_cols if c in data["data_plans"].columns]].copy()
    dp = dp.rename(columns={
        "id": "detail_data_plan_id",
        "amount": "plan_sell_amount",
        "api_amount": "plan_api_cost",
        "active": "plan_active"
    })
    data_detail = dt.merge(dp, on="detail_data_plan_id", how="left")
    data_detail["service"] = "data"

    # Airtime detail.
    at = data["airtime_transactions"][["id", "mobile_network_id"]].copy()
    at = at.rename(columns={"id": "transactionable_id",
                            "mobile_network_id": "detail_network_id"})
    at["service"] = "airtime"

    # Cable detail.
    cb = data["cable_transactions"][["id", "cable_network_id",
                                     "cable_subscription_plan_id"]].copy()
    cb = cb.rename(columns={"id": "transactionable_id",
                            "cable_network_id": "detail_network_id",
                            "cable_subscription_plan_id": "detail_plan_id"})
    cb["service"] = "cable"

    # Electricity detail.
    el = data["electricity_transactions"][["id", "electricity_distributor_id",
                                           "meter_type"]].copy()
    el = el.rename(columns={"id": "transactionable_id",
                            "electricity_distributor_id": "detail_network_id"})
    el["service"] = "electricity"

    # Wallet detail.
    wt = data["wallet_transactions"][["id", "type", "method",
                                      "payment_gateway"]].copy()
    wt = wt.rename(columns={"id": "transactionable_id",
                            "type": "wallet_tx_type",
                            "method": "wallet_method",
                            "payment_gateway": "wallet_gateway"})
    wt["service"] = "wallet"

    details = pd.concat([data_detail, at, cb, el, wt],
                        ignore_index=True, sort=False)
    tx = tx.merge(details, on=["transactionable_id", "service"], how="left")

    # User features (safe, non-secret).
    u = data["users"].copy()
    safe_user_cols = [c for c in [
        "id", "active", "created_at", "last_login", "kyc_verified_at",
        "email_verified_at", "user_package_id", "recovered",
        "enable_api", "account_status"
    ] if c in u.columns]
    u = u[safe_user_cols].rename(columns={
        "id": "user_id",
        "active": "user_active",
        "created_at": "user_created_at"
    })
    tx = tx.merge(u, on="user_id", how="left")

    # Current wallet state. Useful for company/customer analytics, but excluded
    # from historical transaction-success model to avoid temporal leakage.
    w = data["wallets"].copy()
    for c in ["balance", "bonus_balance"]:
        if c in w: w[c] = pd.to_numeric(w[c], errors="coerce")
    wag = w.groupby("user_id").agg(
        wallet_count=("id", "count"),
        current_wallet_balance=("balance", "sum"),
        current_bonus_balance=("bonus_balance", "sum"),
    ).reset_index()
    tx = tx.merge(wag, on="user_id", how="left")

    # Funding-account presence/count (not raw bank/account details).
    fa = data["funding_accounts"].copy()
    fag = fa.groupby("user_id").agg(
        funding_account_count=("id", "count"),
        active_funding_accounts=("active", "sum")
    ).reset_index()
    tx = tx.merge(fag, on="user_id", how="left")

    tx["estimated_gross_margin"] = np.nan
    if use_api_cost and "plan_api_cost" in tx:
        cost = pd.to_numeric(tx["plan_api_cost"], errors="coerce")
        mask = tx["service"].eq("data") & tx["is_success"].eq(1) & cost.notna()
        tx.loc[mask, "estimated_gross_margin"] = tx.loc[mask, "amount"] - cost[mask]

    tx["hour"] = tx["created_at"].dt.hour
    tx["dayofweek"] = tx["created_at"].dt.dayofweek
    tx["day"] = tx["created_at"].dt.day
    tx["month"] = tx["created_at"].dt.month
    tx["is_weekend"] = tx["dayofweek"].isin([5, 6]).astype(int)

    return tx.sort_values("created_at").reset_index(drop=True)

def daily_company_metrics(tx):
    x = tx.dropna(subset=["created_at"]).copy()
    x["date"] = x["created_at"].dt.floor("D")
    out = x.groupby("date").agg(
        transactions=("id", "count"),
        data_transactions=("service", lambda values: values.eq("data").sum()),
        successful_transactions=("is_success", "sum"),
        failed_transactions=("is_failed", "sum"),
        transaction_value=("amount", "sum"),
        revenue=("revenue", "sum"),
        active_users=("user_id", "nunique"),
    ).reset_index()
    out["success_rate"] = out["successful_transactions"] / out["transactions"].clip(lower=1)
    out["failure_rate"] = out["failed_transactions"] / out["transactions"].clip(lower=1)
    return out

def product_daily_metrics(tx):
    x = tx.dropna(subset=["created_at"]).copy()
    x["date"] = x["created_at"].dt.floor("D")
    return x.groupby(["date", "service"]).agg(
        transactions=("id", "count"),
        successes=("is_success", "sum"),
        revenue=("revenue", "sum"),
        users=("user_id", "nunique")
    ).reset_index()

def user_features_asof(tx, cutoff, window_days=90):
    start = cutoff - pd.Timedelta(days=window_days)
    h = tx[(tx["created_at"] < cutoff) & (tx["created_at"] >= start)].copy()
    if h.empty:
        return pd.DataFrame()
    g = h.groupby("user_id").agg(
        tx_90d=("id", "count"),
        success_90d=("is_success", "mean"),
        spend_90d=("revenue", "sum"),
        avg_amount_90d=("amount", "mean"),
        std_amount_90d=("amount", "std"),
        services_90d=("service", "nunique"),
        last_tx=("created_at", "max"),
        first_tx_window=("created_at", "min"),
    ).reset_index()
    g["recency_days"] = (cutoff - g["last_tx"]).dt.total_seconds() / 86400
    g["window_tenure_days"] = (g["last_tx"] - g["first_tx_window"]).dt.total_seconds() / 86400
    g = g.drop(columns=["last_tx", "first_tx_window"])
    return g.fillna(0)
