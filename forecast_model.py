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
        seasonality_mode="multiplicative"  # good fit since seasonality grows with the trend
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

    # Train final model on ALL data and forecast 12 weeks into the future
    final_model = train_model(prophet_df)
    future = final_model.make_future_dataframe(periods=12, freq="W")
    future_forecast = final_model.predict(future)
    future_forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(12).to_csv(
        "data/future_forecast.csv", index=False
    )
    print("Saved 12-week future forecast to data/future_forecast.csv")
