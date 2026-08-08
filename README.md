# Sales Forecasting & Time-Series Analysis

An end-to-end sales forecasting project analyzing 4 years of retail order data, built as a final-year Data Science project. Includes data cleaning, exploratory analysis, a time-series forecasting model, rigorous evaluation, and an interactive dashboard.

🔗 **Live demo:** https://sales-forecasting-project-9zodvw6rzsmyyirvjctzm6.streamlit.app/

## Overview
This project forecasts future retail sales using 4 years (2015–2018) of Superstore order data. It walks through the full data science workflow: cleaning, exploratory analysis, model building, evaluation, and deployment as an interactive dashboard.

## Dataset
- **Source:** Superstore Sales dataset — 9,800 orders, Jan 2015 to Dec 2018
- **Key columns:** Order Date, Category, Sub-Category, Region, Sales

## Key Findings
- **Clear upward trend** in sales from 2015 to 2018
- **Strong, consistent seasonality**: sales dip in Jan/Feb and peak every year in September and November (holiday shopping effect)
- Technology, Furniture, and Office Supplies contribute fairly evenly to total sales
- **Cross-validation revealed the model outperforms a naive baseline** at most forecast horizons (MAE as low as 2,704 at a 16-day horizon), even though a single train/test split initially looked misleading — a good example of why robust evaluation matters more than one-off metrics
- Monthly aggregation underperformed weekly forecasting, since there were too few monthly data points (~48) for Prophet to learn seasonality reliably

## Project Structure
```
sales-forecasting-project/
├── data/
│   ├── train.csv              # raw dataset
│   ├── weekly_sales.csv       # cleaned weekly aggregation
│   ├── future_forecast.csv    # 12-week forecast output
│   └── cv_metrics.csv         # cross-validation results
├── notebooks/                 # exploratory & forecast plots
├── prepare_data.py            # data cleaning & weekly aggregation
├── explore_data.py            # exploratory analysis & visualizations
├── forecast_model.py          # Prophet model, evaluation, cross-validation
├── app.py                     # Streamlit interactive dashboard
├── requirements.txt
└── README.md
```

## Approach
1. **Data Cleaning** (`prepare_data.py`): Parsed dates, handled missing values, aggregated daily orders into weekly sales totals.
2. **Exploratory Analysis** (`explore_data.py`): Visualized trends and seasonality; confirmed outliers were real holiday spikes, not data errors.
3. **Modeling** (`forecast_model.py`): Forecasted sales using Facebook Prophet with additive seasonality. Evaluated using MAE/RMSE against a naive baseline.
4. **Robust Evaluation**: Used rolling-window cross-validation (not just a single train/test split) to get a reliable accuracy estimate, and compared weekly vs. monthly aggregation.
5. **Dashboard** (`app.py`): Built an interactive Streamlit app with category/region filters, live forecasting, and confidence intervals.

## Results
| Metric | Single Split | Cross-Validation (best horizon) |
|---|---|---|
| MAE | 5,567 | 2,704 |
| vs. Naive Baseline | -3.0% (worse) | Outperforms baseline |

## How to Run
```bash
# Install dependencies
pip install -r requirements.txt

# Clean and aggregate data
python prepare_data.py

# Run exploratory analysis
python explore_data.py

# Train and evaluate the forecasting model
python forecast_model.py

# Launch the interactive dashboard
streamlit run app.py
```

## Tools Used
Python, Pandas, Prophet, Matplotlib/Seaborn, Streamlit, Git/GitHub

## Future Improvements
- Incorporate external features (holidays, promotions) as Prophet regressors
- Try additional models (SARIMA, LSTM) for comparison
- Deploy the dashboard publicly via Streamlit Cloud

## Author
Final-year BTech Data Science (CSD) student, Amrita Sai Institute of Science and Technology, Paritala.
[LinkedIn](#) · [GitHub](https://github.com/anushareddy74)
