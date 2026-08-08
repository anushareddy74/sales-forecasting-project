"""
Day 3: Exploratory analysis & visualization.
Generates trend, seasonality, category, and outlier plots into notebooks/.
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


def load_data(path="data/train.csv"):
    df = pd.read_csv(path)
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
    return df


def plot_weekly_trend(df, out="notebooks/weekly_trend.png"):
    weekly = df.set_index("Order Date")["Sales"].resample("W").sum()
    fig, ax = plt.subplots(figsize=(12, 5))
    weekly.plot(ax=ax, color="#2b6cb0")
    ax.set_title("Weekly Sales Trend (2015-2018)")
    ax.set_ylabel("Sales")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    return weekly


def plot_monthly_seasonality(df, out="notebooks/monthly_seasonality.png"):
    monthly = df.groupby([df["Order Date"].dt.year, df["Order Date"].dt.month])["Sales"].sum().reset_index()
    monthly.columns = ["Year", "Month", "Sales"]
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(data=monthly, x="Month", y="Sales", hue="Year", marker="o", ax=ax, palette="viridis")
    ax.set_title("Monthly Sales by Year (Seasonality Check)")
    ax.set_xticks(range(1, 13))
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_category_sales(df, out="notebooks/sales_by_category.png"):
    fig, ax = plt.subplots(figsize=(8, 5))
    df.groupby("Category")["Sales"].sum().sort_values().plot(kind="barh", ax=ax, color="#38a169")
    ax.set_title("Total Sales by Category")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


def plot_outlier_check(df, out="notebooks/outlier_check.png"):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(y=df["Sales"], ax=ax, color="#e53e3e")
    ax.set_title("Sales Distribution (Outlier Check)")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()


if __name__ == "__main__":
    df = load_data()
    weekly = plot_weekly_trend(df)
    plot_monthly_seasonality(df)
    plot_category_sales(df)
    plot_outlier_check(df)

    print("Peak week:", weekly.idxmax().date(), "-> Sales:", round(weekly.max(), 2))
    print("Lowest week:", weekly.idxmin().date(), "-> Sales:", round(weekly.min(), 2))
    print("Saved 4 plots to notebooks/")
