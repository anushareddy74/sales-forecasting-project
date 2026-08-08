"""
Day 1-2: Load, clean, and aggregate the Superstore sales dataset.
"""
import pandas as pd

def load_and_clean(path="data/train.csv"):
    df = pd.read_csv(path)

    # Convert date column
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")

    # Postal Code has a few missing values but isn't needed for forecasting
    df = df.drop(columns=["Postal Code"], errors="ignore")

    return df


def aggregate_weekly(df, category=None, region=None):
    """Aggregate Sales to weekly totals, optionally filtered by category/region."""
    filtered = df.copy()
    if category:
        filtered = filtered[filtered["Category"] == category]
    if region:
        filtered = filtered[filtered["Region"] == region]

    weekly = filtered.set_index("Order Date")["Sales"].resample("W").sum()
    return weekly


if __name__ == "__main__":
    df = load_and_clean()
    print("Loaded:", df.shape)

    weekly = aggregate_weekly(df)
    print("Weekly series:", weekly.shape)
    weekly.to_csv("data/weekly_sales.csv")
    print("Saved data/weekly_sales.csv")
