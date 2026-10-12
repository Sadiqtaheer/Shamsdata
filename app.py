"""Local Shams business dashboard. Run with: streamlit run app.py."""
import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from auto_training import AutoTrainer
from config import DATA_DIR
from data_loader import DATA_FILES

ROOT = Path(__file__).resolve().parent
REPORTS = ROOT / "reports"
st.set_page_config(page_title="Shamsdata insights", page_icon=":material/monitoring:", layout="wide")


@st.cache_data(ttl=300, max_entries=24, show_spinner=False)
def read_csv(name, modified):
    return pd.read_csv(REPORTS / name)


def table(name):
    path = REPORTS / name
    if not path.exists():
        st.info(f"This report is not available yet: {name}. Complete training first.")
        st.stop()
    return read_csv(name, path.stat().st_mtime_ns)


def report(name):
    path = REPORTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def money(value):
    return f"₦{value:,.0f}"


def download(df, filename):
    st.download_button("Download this table", df.to_csv(index=False).encode("utf-8"), filename,
                       mime="text/csv", icon=":material/download:")




def display(df):
    st.dataframe(df, hide_index=True, width="stretch")


@st.cache_data(ttl=300, max_entries=2, show_spinner=False)
def load_transactions(modified):
    # Only these analytics fields leave the server; raw account details are excluded.
    cols = ["id", "created_at", "user_id", "service", "amount", "status", "revenue", "is_success", "is_failed", "vending_medium"]
    tx = pd.read_csv(REPORTS / "enriched_transactions.csv", usecols=cols, parse_dates=["created_at"])
    return tx.dropna(subset=["created_at"])



from insights_views import operations, churn, projection


@st.cache_data(ttl=300, show_spinner=False)
def customer_names(modified):
    return pd.read_csv(ROOT / "users.csv", usecols=["id", "name"]).rename(
        columns={"id": "user_id", "name": "customer_name"})


@st.cache_resource(show_spinner=False)
def get_auto_trainer():
    return AutoTrainer(ROOT, DATA_DIR, REPORTS, DATA_FILES).start()


@st.fragment(run_every="5s")
def automatic_training_notice(trainer):
    state = trainer.snapshot()
    if state["state"] == "running":
        st.info("New source data detected. Models are retraining; the dashboard will refresh automatically.")
    elif state["state"] == "error":
        st.error("Automatic retraining failed. See reports/auto_training.log; the system will retry.")
    elif state["generation"] > st.session_state.get("auto_training_generation", 0):
        st.session_state["auto_training_generation"] = state["generation"]
        st.rerun()


trainer = get_auto_trainer()
st.title("Shamsdata insights", icon=":material/monitoring:")
st.caption("Operational outcomes and business insights from your transaction history")
automatic_training_notice(trainer)
path = REPORTS / "enriched_transactions.csv"
if not path.exists():
    st.info("Train the models first to generate the dashboard reports.")
    st.stop()
tx = load_transactions(path.stat().st_mtime_ns)
manifest = report("run_manifest.json")
summary = report("training_summary.json")
if tx.empty:
    st.info("No dated transactions are available.")
    st.stop()
names = customer_names((ROOT / "users.csv").stat().st_mtime_ns)
tx = tx.merge(names, on="user_id", how="left", validate="many_to_one")
tx["customer_name"] = tx.customer_name.fillna("Unknown")
last_date = tx.created_at.max()
with st.container(horizontal=True):
    st.metric("Transactions", f"{len(tx):,}", border=True)
    st.metric("Successful transactions", f"{tx.is_success.sum():,}", border=True)
    st.metric("Success rate", f"{tx.is_success.mean():.1%}", border=True)
    st.metric("Failed transactions", f"{tx.is_failed.sum():,}", border=True)
    st.metric("Successful transaction value", money(tx.revenue.sum()), border=True)
st.caption(f"Data through {last_date:%d %b %Y} · NGN · Successful value includes wallet funding and is not company earnings.")
tabs = st.tabs(["Operations", "Churn risk", "Business projection", "More analytics"])
with tabs[0]:
    operations(tx)
with tabs[1]:
    churn(table("customer_churn_and_value.csv"), names, summary.get("customers", {}).get("churn", {}))
with tabs[2]:
    projection(tx)
with tabs[3]:
    section = st.selectbox("Workspace", ["Forecasts", "Data plan performance", "Customers", "Product trends", "Anomalies", "Model performance"], key="analytics_section")
    if section == "Forecasts":
        st.caption("Company-wide predictions using all services. Forecasts start after the last recorded transaction.")
        horizon = st.number_input("Forecast horizon (days)", min_value=7, max_value=90, value=30, step=1, key="forecast_horizon_count")
        future = table("company_daily_forecast.csv").iloc[:horizon].copy()
        future["date"] = pd.to_datetime(future["date"])
        with st.container(horizontal=True):
            st.metric("Expected transactions", f"{future.transactions.sum():,.0f}", border=True)
            st.metric("Expected successful value", money(future.revenue.sum()), border=True)
            rate = future.successful_transactions.sum() / max(future.transactions.sum(), 1)
            st.metric("Expected success rate", f"{rate:.1%}", border=True)
        with st.container(horizontal=True):
            choice = st.selectbox("Forecast measure", ["Transactions", "Data transactions", "Successful value", "Active users"], key="forecast_measure")
            trend_view = st.segmented_control("Time view", ["Daily", "7-day average"], default="Daily", key="forecast_trend_view")
        field = {"Transactions":"transactions", "Data transactions":"data_transactions", "Successful value":"revenue", "Active users":"active_users"}[choice]
        history = tx.set_index("created_at").resample("D").agg(
            transactions=("id", "count"),
            data_transactions=("service", lambda values: values.eq("data").sum()),
            revenue=("revenue", "sum"),
            active_users=("user_id", "nunique"),
        ).tail(96).reset_index().rename(columns={"created_at": "date"})
        history = history[history["date"] < future.date.min()]
        chart_data = pd.concat([
            history[["date", field]].assign(Series="Actual"),
            future[["date", field]].assign(Series="Forecast"),
        ], ignore_index=True).rename(columns={field: "Value"})
        if trend_view == "7-day average":
            chart_data["Value"] = chart_data["Value"].rolling(7, min_periods=7).mean()
        chart_data = chart_data[
            chart_data["date"] >= history["date"].max() - pd.Timedelta(days=89)
        ].dropna(subset=["Value"])
        hist = chart_data[chart_data["Series"] == "Actual"]
        pred = chart_data[chart_data["Series"] == "Forecast"]
        axis_title = {
            "transactions": "Transactions per day",
            "data_transactions": "Data transactions per day",
            "revenue": "Successful transaction value per day (NGN)",
            "active_users": "Unique active users per day",
        }[field]
        if trend_view == "7-day average":
            axis_title = f"7-day average of {axis_title.lower()}"
        chart = alt.Chart(chart_data).mark_line(strokeWidth=2.5).encode(
            x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %y", tickCount=7, labelAngle=0)),
            y=alt.Y("Value:Q", title=axis_title, axis=alt.Axis(format=",.0f"), scale=alt.Scale(zero=True)),
            color=alt.Color("Series:N", scale=alt.Scale(domain=["Actual", "Forecast"], range=["#087F8C", "#D58B28"]), legend=alt.Legend(orient="top", title=None)),
            strokeDash=alt.StrokeDash("Series:N", scale=alt.Scale(domain=["Actual", "Forecast"], range=[[1, 0], [6, 4]]), legend=None),
            tooltip=[alt.Tooltip("date:T", title="Date", format="%d %b %Y"), alt.Tooltip("Series:N", title="Period"), alt.Tooltip("Value:Q", title=axis_title, format=",.1f")],
        ).properties(height=350, title=f"{choice}: actuals and forecast ({trend_view.lower()})")
        actual_points = alt.Chart(hist).mark_point(
            filled=True, size=55, color="#087F8C", stroke="white", strokeWidth=1
        ).encode(
            x=alt.X("date:T"),
            y=alt.Y("Value:Q"),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
                alt.Tooltip("Series:N", title="Period"),
                alt.Tooltip("Value:Q", title=axis_title, format=",.1f"),
            ],
        )
        forecast_points = alt.Chart(pred).mark_point(
            filled=True, size=55, color="#D58B28", stroke="white", strokeWidth=1
        ).encode(
            x=alt.X("date:T"),
            y=alt.Y("Value:Q"),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%d %b %Y"),
                alt.Tooltip("Series:N", title="Period"),
                alt.Tooltip("Value:Q", title=axis_title, format=",.1f"),
            ],
        )
        forecast_window = alt.Chart(pd.DataFrame({
            "start": [future.date.min()], "end": [future.date.max()]
        })).mark_rect(color="#D58B28", opacity=.08).encode(
            x=alt.X("start:T"), x2=alt.X2("end:T")
        )
        boundary = alt.Chart(pd.DataFrame({"date":[future.date.min()]})).mark_rule(color="#D58B28", strokeDash=[3,3]).encode(x="date:T")
        label = boundary.mark_text(text="Forecast begins", align="left", dx=6, dy=12, color="#98611C").encode(y=alt.value(14))
        st.altair_chart(forecast_window + chart + actual_points + forecast_points + boundary + label, width="stretch")
        caption = (
            f"Actuals through {hist.date.max():%d %b %Y}; forecast from "
            f"{future.date.min():%d %b %Y} to {future.date.max():%d %b %Y}. "
            "Forecasts are point estimates; the potentially incomplete final export day is excluded from actuals and fitting."
        )
        if field == "revenue":
            caption += " Successful value is transaction value, not company earnings."
        st.caption(caption)
        backtest = report("forecast_backtest_metrics.json")
        if field in backtest:
            b = backtest[field]
            st.caption(f"Forecast method: {b.get('selected_model', 'gradient_boosting').replace('_', ' ')}. Selected using earlier validation windows.")
            if b.get("mae") is not None and b["mae"] > b["weekly_baseline_mae"]:
                st.warning(f"On the last 30 recorded days, this model was less accurate than repeating the previous week. Model average error: {b['mae']:,.1f}; weekly baseline: {b['weekly_baseline_mae']:,.1f}.")
        st.subheader("Daily forecast breakdown", icon=":material/show_chart:")
        graph_groups = [
            ("Transaction counts", {"transactions": "All transactions", "successful_transactions": "Successful", "failed_transactions": "Failed"}, "Transactions", False),
            ("Transaction values", {"transaction_value": "Total value", "revenue": "Successful value"}, "Value (NGN)", False),
            ("Active users", {"active_users": "Active users"}, "Users", False),
            ("Success and failure rates", {"success_rate": "Success rate", "failure_rate": "Failure rate"}, "Rate", True),
        ]
        for offset in (0, 2):
            columns = st.columns(2)
            for container, (title, fields, axis_title, percentage) in zip(columns, graph_groups[offset:offset + 2]):
                with container, st.container(border=True):
                    st.subheader(title)
                    graph_data = future[["date", *fields]].rename(columns=fields).melt(id_vars="date", var_name="Measure", value_name="Forecast")
                    value_format = ".1%" if percentage else ",.1f"
                    st.altair_chart(alt.Chart(graph_data).mark_line(
                        point=alt.OverlayMarkDef(size=55, filled=True, stroke="white", strokeWidth=1)
                    ).encode(
                        x=alt.X("date:T", title="Date", axis=alt.Axis(format="%b %y", tickCount=6, labelAngle=0)),
                        y=alt.Y("Forecast:Q", title=axis_title, axis=alt.Axis(format=".0%" if percentage else ",.0f"), scale=alt.Scale(domain=[0, 1]) if percentage else alt.Scale(zero=True)),
                        color=alt.Color("Measure:N", legend=alt.Legend(orient="bottom", title=None)),
                        tooltip=[alt.Tooltip("date:T", title="Date", format="%d %b %Y"), "Measure:N", alt.Tooltip("Forecast:Q", format=value_format)],
                    ).properties(height=280), width="stretch")
        with st.expander("View forecast data table"):
            display(future.rename(columns={"revenue":"successful_value_ngn", "transaction_value":"total_value_ngn"}))
        download(future.rename(columns={"revenue":"successful_value_ngn"}), "company_forecast.csv")

    elif section == "Customers":
        customers = table("customer_churn_and_value.csv")
        segments = table("customer_segments.csv")
        customers = customers.merge(segments[["user_id", "customer_segment"]], on="user_id", how="left")
        customers["Segment"] = customers.customer_segment.map(lambda x: f"Segment {int(x) + 1}")
        st.caption("Customers active in the last 90 days. Churn means no transaction during the next 30 days; values are predictions as of the dataset end.")
        threshold = st.number_input("High churn risk threshold (%)", min_value=10, max_value=90, value=60, step=5, key="risk_threshold_percent") / 100
        high = customers[customers.churn_probability >= threshold]
        with st.container(horizontal=True):
            st.metric("Scored customers", f"{len(customers):,}", border=True)
            st.metric("High churn risk", f"{len(high):,}", border=True)
            st.metric("Predicted 30-day value", money(customers.predicted_next_30d_value.sum()), border=True)
        left, right = st.columns(2)
        with left, st.container(border=True):
            st.subheader("Risk and customer value")
            chart = alt.Chart(customers).mark_circle(size=85, opacity=.7).encode(
                x=alt.X("churn_probability:Q", title="Chance of no transaction in next 30 days", scale=alt.Scale(domain=[0,1]), axis=alt.Axis(format="%")),
                y=alt.Y("predicted_next_30d_value:Q", title="Predicted 30-day value (NGN)", axis=alt.Axis(format=",.0f")),
                color=alt.Color("Segment:N", title="Segment", legend=alt.Legend(orient="top")),
                tooltip=[alt.Tooltip("user_id:N", title="Customer ID"), "Segment:N", alt.Tooltip("churn_probability:Q", title="Inactivity risk", format=".1%"), alt.Tooltip("predicted_next_30d_value:Q", title="Predicted value (NGN)", format=",.0f")]).properties(height=300)
            cutoff = alt.Chart(pd.DataFrame({"risk":[threshold]})).mark_rule(color="#C74747", strokeDash=[5,4]).encode(x="risk:Q", tooltip=[alt.Tooltip("risk:Q", title="High-risk cutoff", format=".0%")])
            st.altair_chart(chart + cutoff)
        with right, st.container(border=True):
            st.subheader("Customer segments")
            counts = customers.groupby("customer_segment").size().rename("Customers").reset_index()
            counts["Segment"] = counts.customer_segment.map(lambda x: f"Segment {int(x) + 1}")
            st.bar_chart(counts, x="Segment", y="Customers", color="#087F8C", height=300)
            st.caption("Segments group similar activity patterns. Numbers are cluster labels, not quality rankings.")
        only_high = st.checkbox("Show only high-risk customers", value=True, key="only_high")
        visible = (high if only_high else customers).sort_values("churn_probability", ascending=False)
        visible = visible[["user_id", "churn_probability", "predicted_next_30d_value", "tx_90d", "recency_days", "customer_segment"]].copy()
        visible["churn_probability"] = (visible.churn_probability * 100).round(1)
        visible["customer_segment"] += 1
        visible = visible.rename(columns={"user_id":"Customer ID", "churn_probability":"Churn risk (%)", "predicted_next_30d_value":"Predicted 30-day value (NGN)", "tx_90d":"90-day transactions", "recency_days":"Days since last transaction", "customer_segment":"Segment"})
        display(visible)
        download(visible, "customer_priority_list.csv")

    elif section == "Product trends":
        st.caption("Most recent 30 days compared with the preceding 30 days, ending at the dataset's last recorded date.")
        trends = table("product_trends.csv")
        metric = st.selectbox("Compare", ["Transactions", "Successful value", "Transacting users"], key="product_metric")
        field = {"Transactions":"transactions", "Successful value":"revenue", "Transacting users":"users"}[metric]
        part = trends[trends.metric == field].copy()
        if field == "users":
            st.caption("User counts are summed daily active users; a customer can appear on multiple days.")
        bars = part.melt(id_vars="service", value_vars=["previous_30d", "recent_30d"], var_name="Period", value_name="Value")
        bars["Period"] = bars.Period.map({"previous_30d":"Previous 30 days", "recent_30d":"Recent 30 days"})
        st.altair_chart(alt.Chart(bars).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("service:N", title="Service", axis=alt.Axis(labelAngle=0)), xOffset="Period:N", y=alt.Y("Value:Q", title="Summed daily active users" if field == "users" else metric + (" (NGN)" if field == "revenue" else ""), axis=alt.Axis(format=",.0f")),
            color=alt.Color("Period:N", scale=alt.Scale(domain=["Previous 30 days", "Recent 30 days"], range=["#AABCC7", "#087F8C"]), legend=alt.Legend(orient="top", title=None)), tooltip=[alt.Tooltip("service:N", title="Service"), "Period", alt.Tooltip("Value:Q", title=metric, format=",.0f")]).properties(height=320))
        part["Growth (%)"] = (part.growth_rate * 100).round(1)
        part.loc[part.previous_30d == 0, "Growth (%)"] = float("nan")
        display(part[["service", "previous_30d", "recent_30d", "Growth (%)", "trend"]])
        st.caption("Rising/falling uses a ±10% threshold. Growth is undefined when the previous period is zero.")
        download(part, "product_trends.csv")
        with st.expander("Data-plan margin estimate"):
            st.warning("This estimate assumes the current data_plans.api_amount is supplier cost. It excludes operating costs and other services, and is not verified profit.")
            margin = table("estimated_data_profit_history.csv")
            st.metric("Estimated data gross margin", money(margin.estimated_data_gross_margin.sum()))
            margin["date"] = pd.to_datetime(margin.date)
            st.line_chart(margin.set_index("date"), y="estimated_data_gross_margin", y_label="Estimated margin (NGN)")

    elif section == "Data plan performance":
        plans = table("data_plan_performance.csv")
        if plans.empty:
            st.info("No data-plan transactions with final outcomes were available in the latest training run.")
            st.stop()
        st.caption("Historical data-plan outcomes across the available transaction history. The comparison describes observed reliability, not guaranteed future quality.")
        minimum_volume = st.number_input(
            "Minimum transactions to rank a plan", min_value=1, max_value=500,
            value=30, step=5, key="data_plan_minimum_volume"
        )
        comparable = plans[plans.transactions >= minimum_volume].copy()
        comparable = comparable.sort_values(
            ["success_rate", "estimated_margin_per_cost_matched_transaction", "transactions"],
            ascending=[False, False, False],
            na_position="last",
        )
        with st.container(horizontal=True):
            st.metric("Plans meeting sample threshold", f"{len(comparable):,}", border=True)
            st.metric("Matched data transactions", f"{plans.transactions.sum():,.0f}", border=True)
            st.metric("Successful data sales", money(plans.sales_value.sum()), border=True)
        st.caption("Success rate uses final success/failed outcomes. Current catalog cost is used only for estimated margin; historical catalog prices are unavailable.")
        if comparable.empty:
            st.info("No plans meet this transaction-count threshold yet. Lower the minimum to inspect plans with smaller samples.")
        else:
            top = comparable.head(15).copy()
            top["Plan"] = top.detail_data_plan_id.map(lambda plan_id: f"Plan {int(plan_id)}")
            st.subheader("Plans with the highest observed success rate")
            chart = alt.Chart(top).mark_bar().encode(
                x=alt.X("success_rate:Q", title="Observed success rate", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
                y=alt.Y("Plan:N", sort=alt.SortField(field="success_rate", order="descending"), title=None),
                color=alt.value("#087F8C"),
                tooltip=[
                    alt.Tooltip("Plan:N"),
                    alt.Tooltip("transactions:Q", title="Transactions", format=",.0f"),
                    alt.Tooltip("success_rate:Q", title="Success rate", format=".1%"),
                    alt.Tooltip("estimated_margin_per_cost_matched_transaction:Q", title="Estimated margin per cost-matched sale (NGN)", format=",.0f"),
                ],
            ).properties(height=max(240, len(top) * 28))
            st.altair_chart(chart, width="stretch")
        columns = [
            "performance_rank", "detail_data_plan_id", "size", "volume", "validity",
            "plan_sell_amount", "plan_api_cost", "transactions", "successful_transactions",
            "failed_transactions", "unique_customers", "success_rate", "failure_rate",
            "sales_value", "estimated_gross_margin", "transactions_with_cost_data",
            "estimated_margin_per_cost_matched_transaction", "sample_note",
        ]
        visible = plans[[column for column in columns if column in plans]].rename(columns={
            "performance_rank": "Rank",
            "detail_data_plan_id": "Plan ID",
            "size": "Plan size",
            "volume": "Unit",
            "validity": "Current catalog validity",
            "plan_sell_amount": "Current catalog sell price (NGN)",
            "plan_api_cost": "Current catalog API cost (NGN)",
            "transactions": "Transactions",
            "successful_transactions": "Successful",
            "failed_transactions": "Failed",
            "unique_customers": "Unique customers",
            "success_rate": "Success rate",
            "failure_rate": "Failure rate",
            "sales_value": "Successful sales (NGN)",
            "estimated_gross_margin": "Estimated gross margin (NGN)",
            "transactions_with_cost_data": "Cost-matched transactions",
            "estimated_margin_per_cost_matched_transaction": "Estimated margin per cost-matched sale (NGN)",
            "sample_note": "Sample size",
        }).copy()
        for column in ["Success rate", "Failure rate"]:
            visible[column] = visible[column].map(lambda value: f"{value:.1%}")
        display(visible)
        download(plans, "data_plan_performance.csv")

    elif section == "Anomalies":
        anomalies = table("transaction_anomalies.csv")
        anomalies["created_at"] = pd.to_datetime(anomalies.created_at)
        st.caption("Unusual transactions detected from amount, time, balance, and service. A flag is a review signal, not evidence of fraud.")
        services = st.multiselect("Services", sorted(anomalies.service.unique()), default=sorted(anomalies.service.unique()), key="anomaly_services")
        selected = anomalies[anomalies.service.isin(services)]
        flagged = selected[selected.anomaly_flag == -1]
        with st.container(horizontal=True):
            st.metric("Flagged transactions", f"{len(flagged):,}", border=True)
            st.metric("Flagged amount", money(flagged.amount.sum()), border=True)
            st.metric("Flag rate", f"{len(flagged) / max(len(selected), 1):.1%}", border=True)
        if flagged.empty:
            st.info("No flagged transactions for the selected services.")
        else:
            daily_flags = flagged.set_index("created_at").resample("D").size().rename("Flagged transactions")
            st.altair_chart(alt.Chart(daily_flags.reset_index()).mark_bar(color="#D58B28").encode(
                x=alt.X("created_at:T", title="Transaction date", axis=alt.Axis(format="%d %b")),
                y=alt.Y("Flagged transactions:Q", title="Transactions flagged per day", axis=alt.Axis(tickMinStep=1)),
                tooltip=[alt.Tooltip("created_at:T", title="Date", format="%d %b %Y"), "Flagged transactions:Q"]).properties(height=280))
        visible = flagged[["id", "created_at", "user_id", "service", "amount", "anomaly_score"]].sort_values("anomaly_score", ascending=False)
        download(visible, "flagged_transactions.csv")

    elif section == "Model performance":
        st.caption("Held-out evaluation uses later dates than training. Saved delivery models are then fitted on all available labelled data.")
        success = summary.get("transaction_success", {})
        churn = summary.get("customers", {}).get("churn", {})
        value = summary.get("customers", {}).get("customer_value", {})
        if not success or not churn:
            st.info("Complete training to see evaluation results.")
            st.stop()
        with st.container(horizontal=True):
            st.metric("Transaction accuracy", f"{success['accuracy']:.1%}", border=True)
            st.metric("Transaction ROC AUC", f"{success['roc_auc']:.3f}" if success.get('roc_auc') is not None else "Unavailable", border=True)
            st.metric("Churn accuracy", f"{churn['accuracy']:.1%}", border=True)
            st.metric("Customer value R²", f"{value['r2']:.3f}", border=True)
        left, right = st.columns(2)
        with left, st.container(border=True):
            st.subheader("Transaction success / failure")
            st.caption(f"Evaluated on {success['test_rows']:,} later transactions.")
            cr = pd.DataFrame({"Failed":success["classification_report"]["0"], "Successful":success["classification_report"]["1"]}).T
            display(cr.reset_index(names="Outcome").round(3))
            if "majority_baseline_accuracy" in success:
                st.caption(f"Always predicting the most common training class: {success['majority_baseline_accuracy']:.1%} accuracy.")
        with right, st.container(border=True):
            st.subheader("Customer models")
            auc = churn.get("roc_auc")
            st.markdown(f"**Churn ROC AUC:** {auc:.3f}" if auc is not None else "**Churn ROC AUC:** Unavailable (one test class)")
            if "brier_score" in churn:
                st.caption(f"Churn Brier score: {churn['brier_score']:.3f}; transaction Brier score: {success['brier_score']:.3f}. Lower is better.")
                st.caption(f"Churn log loss: {churn['log_loss']:.3f}; transaction log loss: {success['log_loss']:.3f}. Lower is better. Probabilities have not been recalibrated.")
            st.markdown(f"**Value average absolute error:** {money(value['mae'])}")
            st.markdown(f"**Value root mean squared error:** {money(value['rmse'])}")
            st.caption("Value is next-30-day successful transaction value, not lifetime profit. AUC measures class ranking; it does not guarantee calibrated probabilities.")
        st.subheader("Forecast validation")
        backtest = report("forecast_backtest_metrics.json")
        if backtest:
            validation = pd.DataFrame({name: {key: value for key, value in result.items() if key in ["mae", "weekly_baseline_mae", "holdout_days", "selected_model"]} for name, result in backtest.items()}).T.reset_index(names="Measure")
            validation["Beats weekly baseline"] = validation.mae < validation.weekly_baseline_mae
            display(validation.rename(columns={"mae":"Model MAE", "weekly_baseline_mae":"Weekly baseline MAE", "holdout_days":"Holdout days"}))
            st.caption("Models are selected on earlier rolling windows, then evaluated on a separate final holdout. Lower MAE is better. Errors are measured before component reconciliation; the final export day is excluded.")
            horizon_rows = [{"Measure": name, "Horizon days": int(horizon), **{key: result.get(key) for key in ["selected_model", "mae", "weekly_baseline_mae", "selection_folds"]}} for name, item in backtest.items() for horizon, result in item.get("horizons", {}).items()]
            if horizon_rows:
                display(pd.DataFrame(horizon_rows))
        st.subheader("Training record")
        if manifest:
            st.json({k:v for k,v in manifest.items() if k != "source_sha256"}, expanded=False)
            st.download_button("Download training record", json.dumps(manifest, indent=2), "run_manifest.json", mime="application/json")
        st.caption("Segmentation and anomaly detection are unsupervised; no labelled accuracy claim is made. Historical plan metadata comes from the current catalog. Forecasts exclude the final export day.")
