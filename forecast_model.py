"""
Day 4-5: Build and evaluate a sales forecasting model using Prophet.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet


def load_weekly_sales(path="data/train.csv"):
    df = pd.read_csv(path)
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
    weekly = df.set_index("Order Date")["Sales"].resample("W").sum()
    return weekly


def prepare_prophet_df(weekly_series):
    """Prophet requires columns named 'ds' (date) and 'y' (value)."""
    prophet_df = weekly_series.reset_index()
    prophet_df.columns = ["ds", "y"]
    return prophet_df


def train_test_split(prophet_df, test_weeks=12):
    """Hold out the last N weeks for testing."""
    train = prophet_df.iloc[:-test_weeks]
    test = prophet_df.iloc[-test_weeks:]
    return train, test


def train_model(train_df):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        seasonality_mode="additive",  # more stable on shorter/noisier series
        changepoint_prior_scale=0.1   # allows a bit more trend flexibility
    )
    model.fit(train_df)
    return model


def evaluate(model, test_df):
    future = model.make_future_dataframe(periods=len(test_df), freq="W")
    forecast = model.predict(future)

    # Get predictions for the test period only
    predicted = forecast.set_index("ds").loc[test_df["ds"], "yhat"]
    actual = test_df.set_index("ds")["y"]

    mae = np.mean(np.abs(actual.values - predicted.values))
    rmse = np.sqrt(np.mean((actual.values - predicted.values) ** 2))

    # Naive baseline: predict last known value repeated
    naive_pred = np.full(len(test_df), train_test_split_last_value)
    naive_mae = np.mean(np.abs(actual.values - naive_pred))

    print(f"Model MAE:  {mae:.2f}")
    print(f"Model RMSE: {rmse:.2f}")
    print(f"Naive baseline MAE: {naive_mae:.2f}")
    print(f"Improvement over baseline: {(1 - mae/naive_mae) * 100:.1f}%")

    return forecast


def plot_forecast(model, forecast, out="notebooks/forecast.png"):
    fig = model.plot(forecast)
    plt.title("Sales Forecast")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_components(model, forecast, out="notebooks/forecast_components.png"):
    fig = model.plot_components(forecast)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def load_monthly_sales(path="data/train.csv"):
    df = pd.read_csv(path)
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
    monthly = df.set_index("Order Date")["Sales"].resample("M").sum()
    return monthly


def run_cross_validation(model, initial="730 days", period="90 days", horizon="90 days"):
    """
    Prophet's built-in rolling-origin cross-validation: much more robust than a
    single train/test split since it tests the model across multiple time windows.
    """
    from prophet.diagnostics import cross_validation, performance_metrics
    df_cv = cross_validation(model, initial=initial, period=period, horizon=horizon)
    df_metrics = performance_metrics(df_cv)
    return df_cv, df_metrics


def compare_weekly_vs_monthly():
    print("=" * 50)
    print("WEEKLY MODEL")
    print("=" * 50)
    weekly = load_weekly_sales()
    weekly_df = prepare_prophet_df(weekly)
    w_train, w_test = train_test_split(weekly_df, test_weeks=12)
    w_model = train_model(w_train)
    w_forecast = evaluate_with_baseline(w_model, w_test, w_train["y"].iloc[-1])

    print("\n" + "=" * 50)
    print("MONTHLY MODEL")
    print("=" * 50)
    monthly = load_monthly_sales()
    monthly_df = prepare_prophet_df(monthly)
    m_train, m_test = train_test_split(monthly_df, test_weeks=3)  # last 3 months
    m_model = train_model(m_train)
    m_forecast = evaluate_with_baseline(m_model, m_test, m_train["y"].iloc[-1])

    return weekly_df, weekly, monthly_df


def evaluate_with_baseline(model, test_df, last_train_value):
    future = model.make_future_dataframe(periods=len(test_df), freq="D")
    # Match frequency to test_df's actual spacing
    future = model.make_future_dataframe(periods=len(test_df), freq=pd.infer_freq(test_df["ds"]) or "W")
    forecast = model.predict(future)
    predicted = forecast.set_index("ds").reindex(test_df["ds"])["yhat"]
    actual = test_df.set_index("ds")["y"]

    mae = np.mean(np.abs(actual.values - predicted.values))
    rmse = np.sqrt(np.mean((actual.values - predicted.values) ** 2))
    naive_pred = np.full(len(test_df), last_train_value)
    naive_mae = np.mean(np.abs(actual.values - naive_pred))

    print(f"Model MAE:  {mae:.2f}")
    print(f"Model RMSE: {rmse:.2f}")
    print(f"Naive baseline MAE: {naive_mae:.2f}")
    print(f"Improvement over baseline: {(1 - mae/naive_mae) * 100:.1f}%")
    return forecast


if __name__ == "__main__":
    weekly = load_weekly_sales()
    prophet_df = prepare_prophet_df(weekly)

    train_df, test_df = train_test_split(prophet_df, test_weeks=12)
    train_test_split_last_value = train_df["y"].iloc[-1]  # for naive baseline

    print(f"Training on {len(train_df)} weeks, testing on {len(test_df)} weeks")

    model = train_model(train_df)
    forecast = evaluate(model, test_df)

    plot_forecast(model, forecast)
    plot_components(model, forecast)
    print("Saved forecast.png and forecast_components.png to notebooks/")

    # Day 5: Cross-validation for a more robust accuracy estimate
    print("\n" + "=" * 50)
    print("CROSS-VALIDATION (rolling window)")
    print("=" * 50)
    full_model = train_model(prophet_df)
    try:
        df_cv, df_metrics = run_cross_validation(full_model)
        print(df_metrics[["horizon", "mae", "rmse"]].head(10))
        df_metrics.to_csv("data/cv_metrics.csv", index=False)
        print("Saved cross-validation metrics to data/cv_metrics.csv")
    except Exception as e:
        print(f"Cross-validation skipped (needs more data or took too long): {e}")

    # Day 5: Compare weekly vs monthly aggregation
    print("\n" + "=" * 50)
    print("WEEKLY vs MONTHLY COMPARISON")
    print("=" * 50)
    compare_weekly_vs_monthly()

    # Train final model on ALL data and forecast 12 weeks into the future
    final_model = train_model(prophet_df)
    future = final_model.make_future_dataframe(periods=12, freq="W")
    future_forecast = final_model.predict(future)
    future_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(12).to_csv(
        "data/future_forecast.csv", index=False
    )
    print("Saved 12-week future forecast to data/future_forecast.csv")
