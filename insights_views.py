"""Reference-style transaction insight views backed by the local reports."""
import altair as alt
import pandas as pd
import streamlit as st


def paginate(data, key, size=50):
    pages = max(1, (len(data) + size - 1) // size)
    page = min(st.session_state.get(key, 1), pages)
    rows = data.iloc[(page - 1) * size:page * size]
    st.dataframe(rows, hide_index=True, width="stretch", column_config={
        "Churn risk (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
        "Highest failure risk (%)": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%"),
    })
    left, middle, right = st.columns([1, 2, 1])
    if left.button("Previous", key=key + "_previous", disabled=page <= 1):
        st.session_state[key] = page - 1
        st.rerun()
    middle.caption(f"Page {page} of {pages} | Showing {len(rows):,} of {len(data):,} customers")
    if right.button("Next", key=key + "_next", disabled=page >= pages):
        st.session_state[key] = page + 1
        st.rerun()


def period_summary(data, granularity):
    units = {"Weeks": "W-SUN", "Months": "M", "Years": "Y"}
    source = data.copy()
    if granularity == "Weeks":
        source = source[source.created_at >= source.created_at.max() - pd.DateOffset(months=6)]
    source["period"] = source.created_at.dt.to_period(units[granularity]).dt.start_time
    return source.groupby("period", as_index=False).agg(
        transactions=("id", "size"), failure_rate=("is_failed", "mean"), value=("revenue", "sum"))


def operations(data):
    with st.expander("Filter transaction history"):
        dates = st.date_input("Date range", (data.created_at.min().date(), data.created_at.max().date()), key="dates")
        services = st.multiselect("Services", sorted(data.service.unique()), default=sorted(data.service.unique()), key="services")
    if len(dates) != 2:
        st.info("Choose a start and end date.")
        return
    data = data[data.created_at.between(pd.Timestamp(dates[0]), pd.Timestamp(dates[1]) + pd.Timedelta(days=1), inclusive="left") & data.service.isin(services)]
    if data.empty:
        st.info("No transactions match these filters.")
        return
    granularity = st.segmented_control("Time granularity", ["Weeks", "Months", "Years"], default="Months", key="operations_granularity") or "Months"
    rate_view = st.segmented_control("Failure-rate view", ["Weekly (last 6 months)", "Monthly (full history)", "Yearly (full history)"], default="Weekly (last 6 months)", key="failure_rate_view")
    rate_granularity = {"Monthly (full history)": "Months", "Yearly (full history)": "Years"}.get(rate_view, "Weeks")
    volume = period_summary(data, granularity)
    volume["Period"] = volume.period.dt.strftime("%d %b %Y" if granularity == "Weeks" else "%b %Y" if granularity == "Months" else "%Y")
    rates = period_summary(data, rate_granularity)
    rates["Period"] = rates.period.dt.strftime("%d %b %Y" if rate_granularity == "Weeks" else "%b %Y" if rate_granularity == "Months" else "%Y")
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader({"Weeks": "Weekly", "Months": "Monthly", "Years": "Yearly"}[granularity] + " transaction volume", icon=":material/bar_chart:")
        st.altair_chart(alt.Chart(volume).mark_bar().encode(
            x=alt.X("Period:N", title="Period (highest to lowest volume)", sort="-y", axis=alt.Axis(labelAngle=-45)), y=alt.Y("transactions:Q", title="Transactions"),
            tooltip=[alt.Tooltip("period:T", title="Period"), alt.Tooltip("transactions:Q", format=",.0f")]).properties(height=310), width="stretch")
    with right, st.container(border=True):
        st.subheader(rate_view or "Weekly failure rate", icon=":material/error:")
        st.altair_chart(alt.Chart(rates).mark_bar().encode(
            x=alt.X("failure_rate:Q", title="Failure rate", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%")),
            y=alt.Y("Period:N", sort="-x", title=None), tooltip=["Period", alt.Tooltip("failure_rate:Q", format=".2%")]).properties(height=310), width="stretch")
    left, right = st.columns(2)
    for container, field, title, icon in [(left, "service", "Transactions by service", "category"), (right, "vending_medium", "Transactions by provider", "storefront")]:
        counts = data.fillna({field: "Unknown"}).groupby(field, as_index=False).agg(transactions=("id", "size"), failed=("is_failed", "sum"), failure_rate=("is_failed", "mean")).nlargest(10, "transactions")
        with container, st.container(border=True):
            st.subheader(title, icon=f":material/{icon}:")
            st.altair_chart(alt.Chart(counts).mark_bar().encode(
                x=alt.X("transactions:Q", title="Transactions"), y=alt.Y(field + ":N", sort="-x", title=None),
                tooltip=[field, alt.Tooltip("transactions:Q", format=",.0f"), "failed", alt.Tooltip("failure_rate:Q", format=".2%")]).properties(height=280), width="stretch")
    st.caption("Charts use the selected history. Weekly views show its latest six months; headline metrics use all history.")


def risk_explorer(data, predictions):
    st.subheader("Risk explorer", icon=":material/manage_search:")
    scored = data.merge(predictions[["id", "predicted_success_probability"]], on="id", how="inner", validate="one_to_one")
    scored["failure_risk"] = 1 - scored.predicted_success_probability
    with st.container(border=True):
        first, second, third, fourth = st.columns(4)
        with first:
            dates = st.date_input("Transaction date range", (data.created_at.min().date(), data.created_at.max().date()), key="risk_dates")
        with second:
            floor = st.number_input("Minimum failure risk (%)", min_value=0, max_value=100, value=50, step=5, key="failure_floor_percent") / 100
        with third:
            statuses = st.multiselect("Actual status", ["success", "failed"], default=["success", "failed"], key="risk_status")
        with fourth:
            query = st.text_input("Search customer name or ID", key="risk_search").strip()
    if len(dates) != 2:
        st.info("Choose a start and end date.")
        return
    selected = scored[scored.created_at.between(pd.Timestamp(dates[0]), pd.Timestamp(dates[1]) + pd.Timedelta(days=1), inclusive="left") & scored.failure_risk.ge(floor) & scored.status.isin(statuses)]
    if query:
        selected = selected[selected.customer_name.str.contains(query, case=False, regex=False) | selected.user_id.astype(str).str.contains(query, regex=False)]
    grouped = selected.groupby(["user_id", "customer_name"], as_index=False).agg(
        Transactions=("id", "size"), Successful=("is_success", "sum"), Failed=("is_failed", "sum"),
        Value=("revenue", "sum"), Latest=("created_at", "max"), Risk=("failure_risk", "max"))
    grouped = grouped.sort_values(["Successful", "Risk"], ascending=False)
    grouped["Risk"] = (grouped.Risk * 100).round(1)
    grouped = grouped.rename(columns={"user_id": "Customer ID", "customer_name": "Customer name", "Value": "Successful value (NGN)", "Latest": "Last transaction", "Risk": "Highest failure risk (%)"})
    if grouped.empty:
        st.info("No customers match these risk filters.")
    paginate(grouped, "risk_page")
    st.caption("Counts include only transactions matching the filters. Scores come from the saved model fitted on available history; they are review signals, not held-out performance.")


def churn(customers, names, metrics):
    data = customers.merge(names, on="user_id", how="left", validate="one_to_one")
    data["customer_name"] = data.customer_name.fillna("Unknown")
    with st.container(horizontal=True):
        st.metric("Users scored", f"{len(data):,}", border=True)
        st.metric("High churn risk", f"{data.churn_probability.ge(.5).sum():,}", border=True)
        st.metric("Average churn risk", f"{data.churn_probability.mean():.1%}", border=True)
        auc = metrics.get("roc_auc")
        st.metric("Model ROC-AUC", f"{auc:.4f}" if auc is not None else "Unavailable", border=True)
    with st.container(border=True):
        st.subheader("Users most likely to become inactive", icon=":material/person_search:")
        floor_percent = st.number_input(
            "Minimum churn risk (%)", min_value=0, max_value=100, value=50, step=5,
            key="churn_floor_percent",
            help="Use minus to lower the threshold and plus to raise it, or type a percentage.",
        )
        floor = floor_percent / 100
        st.caption(f"Showing customers with churn risk of {floor_percent}% or higher. Lower thresholds include more customers.")
        size = st.number_input(
            "Users to display", min_value=10, max_value=200, value=50, step=10,
            key="churn_display_count",
            help="Number of users per page. Use minus to show fewer or plus to show more.",
        )
        query = st.text_input("Search customer name or ID", key="churn_search").strip()
        data = data[data.churn_probability.ge(floor)]
        if query:
            data = data[data.customer_name.str.contains(query, case=False, regex=False) | data.user_id.astype(str).str.contains(query, regex=False)]
        data = data.sort_values("churn_probability", ascending=False)
        data = data[["customer_name", "user_id", "tx_90d", "spend_90d", "recency_days", "churn_probability", "predicted_next_30d_value"]].copy()
        data.churn_probability = (data.churn_probability * 100).round(1)
        data = data.rename(columns={"customer_name": "Customer name", "user_id": "Customer ID", "tx_90d": "Transactions in last 90 days", "spend_90d": "Successful value in last 90 days (NGN)", "recency_days": "Days since last transaction", "churn_probability": "Churn risk (%)", "predicted_next_30d_value": "Predicted 30-day value (NGN)"})
        if data.empty:
            st.info("No customers match these churn filters.")
        paginate(data, "churn_page", size)
        st.caption("Latest available snapshot: risk of no transaction in the next 30 days. High risk uses a 50% cutoff.")


def projection(data):
    st.subheader("Business projection", icon=":material/insights:")
    source = data.assign(month=data.created_at.dt.to_period("M").dt.to_timestamp())
    monthly = source.groupby(["service", "month"]).agg(transactions=("id", "size"), value=("revenue", "sum"), failure_rate=("is_failed", "mean"))
    # Exclude the final month because the final recorded day may be incomplete.
    months = pd.date_range(source.month.min(), source.month.max(), freq="MS")[:-1]
    if len(months) < 2:
        st.info("At least two completed months are needed for a trend projection.")
        return
    def extrapolate(values):
        recent = pd.Series(values[-6:], dtype=float).dropna()
        if recent.empty:
            return 0.
        if len(recent) == 1:
            return max(0., recent.iloc[0])
        x = pd.Series(range(len(recent)), dtype=float)
        y = pd.Series(recent.to_numpy())
        slope = ((x - x.mean()) * (y - y.mean())).sum() / ((x - x.mean()) ** 2).sum()
        return max(0., y.iloc[-1] + slope)
    rows, chart_rows = [], []
    for service in sorted(data.service.unique()):
        history = monthly.loc[service].reindex(months)
        history[["transactions", "value"]] = history[["transactions", "value"]].fillna(0)
        recent = history.transactions.tail(3).sum()
        previous = history.transactions.iloc[-6:-3].sum()
        growth = recent / previous - 1 if len(months) >= 6 and previous else float("nan")
        predicted = {field: extrapolate(history[field].tolist()) for field in ["transactions", "value", "failure_rate"]}
        predicted["failure_rate"] = min(1., predicted["failure_rate"])
        direction = "Insufficient history" if pd.isna(growth) else "Growing" if growth > .05 else "Declining" if growth < -.05 else "Stable"
        rows.append({"Service": service, "Growth": growth, "Direction": direction, "Projected transactions": predicted["transactions"], "Projected value (NGN)": predicted["value"]})
        for month, record in history.tail(6).iterrows():
            chart_rows.append({"Service": service, "Month": month, "Period": "Actual", **record.to_dict()})
        # Repeat the final actual point to connect each dashed projection segment.
        chart_rows.append({"Service": service, "Month": months[-1], "Period": "Projected", **history.iloc[-1].to_dict()})
        chart_rows.append({"Service": service, "Month": months[-1] + pd.offsets.MonthBegin(1), "Period": "Projected", **predicted})
    result = pd.DataFrame(rows)
    totals = monthly.groupby("month").transactions.sum().reindex(months, fill_value=0)
    previous = totals.iloc[-6:-3].sum()
    growth = totals.tail(3).sum() / previous - 1 if len(months) >= 6 and previous else None
    with st.container(horizontal=True):
        st.metric("3-month transaction trend", f"{growth:.1%}" if growth is not None else "N/A")
        st.metric("Projected month transactions", f"{result['Projected transactions'].sum():,.0f}")
        st.metric("Projected month value", f"₦{result['Projected value (NGN)'].sum():,.0f}")
    st.caption(f"Projection for {source.month.max():%B %Y}, using completed months through {months[-1]:%B %Y}. Simple six-month linear trends; partial final month excluded. These differ from the trained daily forecasts in More analytics.")
    service = st.selectbox("Service projection", ["All services", *sorted(data.service.unique())], key="projection_service")
    chart_data = pd.DataFrame(chart_rows)
    if service != "All services":
        chart_data = chart_data[chart_data.Service.eq(service)]
    def chart(field, title):
        y = alt.Y(field + ":Q", title=title, axis=alt.Axis(format=".0%" if field == "failure_rate" else ",.0f"))
        if field == "failure_rate":
            y = y.scale(domain=[0, 1])
        return alt.Chart(chart_data).mark_line(point=True).encode(
            x=alt.X("Month:T", axis=alt.Axis(format="%Y %b")), y=y, color="Service:N",
            strokeDash=alt.StrokeDash("Period:N", scale=alt.Scale(domain=["Actual", "Projected"], range=[[1, 0], [6, 4]])),
            tooltip=["Service", alt.Tooltip("Month:T", format="%Y %b"), "Period", alt.Tooltip(field + ":Q", format=".1%" if field == "failure_rate" else ",.0f")]).properties(height=300)
    st.subheader("Transaction volume projection", icon=":material/show_chart:")
    st.altair_chart(chart("transactions", "Transactions"), width="stretch")
    left, right = st.columns(2)
    with left, st.container(border=True):
        st.subheader("Successful value projection", icon=":material/payments:")
        st.altair_chart(chart("value", "Successful value (NGN)"), width="stretch")
    with right, st.container(border=True):
        st.subheader("Failure-rate projection", icon=":material/error:")
        st.altair_chart(chart("failure_rate", "Failure rate"), width="stretch")
    with st.container(border=True):
        st.subheader("Service growth direction", icon=":material/trending_up:")
        growth_rows = result.dropna(subset=["Growth"])
        if growth_rows.empty:
            st.info("Six completed months and a nonzero previous period are needed to compare growth.")
        else:
            st.altair_chart(alt.Chart(growth_rows).mark_bar().encode(x=alt.X("Growth:Q", axis=alt.Axis(format="%")), y=alt.Y("Service:N", sort="-x"), color="Direction:N", tooltip=["Service", "Direction", alt.Tooltip("Growth:Q", format=".1%")]), width="stretch")
