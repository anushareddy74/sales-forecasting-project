"""
Day 6: Interactive Streamlit dashboard for sales forecasting.
Run with: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet

st.set_page_config(page_title="Sales Forecasting Dashboard", layout="wide")


@st.cache_data
def load_data(path="data/train.csv"):
    df = pd.read_csv(path)
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
    return df


@st.cache_data
def get_weekly_series(df, category=None, region=None):
    filtered = df.copy()
    if category and category != "All":
        filtered = filtered[filtered["Category"] == category]
    if region and region != "All":
        filtered = filtered[filtered["Region"] == region]
    weekly = filtered.set_index("Order Date")["Sales"].resample("W").sum()
    return weekly


@st.cache_resource
def train_model(_prophet_df):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        seasonality_mode="additive",
        changepoint_prior_scale=0.1
    )
    model.fit(_prophet_df)
    return model


def main():
    st.title("📈 Sales Forecasting Dashboard")
    st.caption("Final-year Data Science project — Superstore sales analysis & forecasting")

    df = load_data()

    # Sidebar filters
    st.sidebar.header("Filters")
    category = st.sidebar.selectbox("Category", ["All"] + sorted(df["Category"].unique().tolist()))
    region = st.sidebar.selectbox("Region", ["All"] + sorted(df["Region"].unique().tolist()))
    forecast_weeks = st.sidebar.slider("Weeks to forecast", min_value=4, max_value=26, value=12)

    weekly = get_weekly_series(df, category, region)

    if len(weekly) < 10:
        st.warning("Not enough data for this filter combination to build a reliable forecast.")
        st.line_chart(weekly)
        return

    # Key metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sales", f"${weekly.sum():,.0f}")
    col2.metric("Avg Weekly Sales", f"${weekly.mean():,.0f}")
    col3.metric("Peak Week Sales", f"${weekly.max():,.0f}")

    # Historical trend
    st.subheader("Historical Weekly Sales")
    st.line_chart(weekly)

    # Forecast
    st.subheader(f"Forecast: Next {forecast_weeks} Weeks")
    with st.spinner("Training model and generating forecast..."):
        prophet_df = weekly.reset_index()
        prophet_df.columns = ["ds", "y"]
        model = train_model(prophet_df)
        future = model.make_future_dataframe(periods=forecast_weeks, freq="W")
        forecast = model.predict(future)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(prophet_df["ds"], prophet_df["y"], label="Historical", color="#2b6cb0")
    forecast_future = forecast[forecast["ds"] > prophet_df["ds"].max()]
    ax.plot(forecast_future["ds"], forecast_future["yhat"], label="Forecast", color="#e53e3e", linestyle="--")
    ax.fill_between(forecast_future["ds"], forecast_future["yhat_lower"], forecast_future["yhat_upper"],
                     color="#e53e3e", alpha=0.15, label="Confidence interval")
    ax.legend()
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    plt.tight_layout()
    st.pyplot(fig)

    # Forecast table
    with st.expander("View forecast data table"):
        display_df = forecast_future[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
        display_df.columns = ["Date", "Predicted Sales", "Lower Bound", "Upper Bound"]
        display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
        st.dataframe(display_df.round(2), use_container_width=True)

    st.markdown("---")
    st.caption("Built with Python, Pandas, Prophet, and Streamlit | [GitHub Repo](https://github.com/anushareddy74/sales-forecasting-project)")


if __name__ == "__main__":
    main()
