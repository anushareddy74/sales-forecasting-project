# Sales Forecasting & Time-Series Analysis

Forecasting future retail sales using historical order data, built as a final-year Data Science project.

## Overview
This project analyzes 4 years (2015–2018) of Superstore order data to identify sales trends and seasonality, then builds a forecasting model to predict future sales. A Streamlit dashboard lets users explore historical sales and view forecasts by category/region.

## Dataset
- **Source:** Superstore Sales dataset (9,800 orders, 2015–2018)
- **Key columns:** Order Date, Category, Sub-Category, Region, Sales

## Project Structure
```
sales-forecasting-project/
├── data/
│   └── train.csv           # raw dataset
├── notebooks/               # exploratory analysis
├── prepare_data.py          # data cleaning & weekly aggregation
├── requirements.txt
└── README.md
```

## Approach
1. **Data Cleaning:** Parsed dates, handled missing values, aggregated daily orders into weekly sales totals.
2. **Exploratory Analysis:** Visualized trends and seasonality (strong end-of-year spikes).
3. **Modeling:** Forecasted future sales using Facebook Prophet, evaluated with MAE/RMSE against a naive baseline.
4. **Dashboard:** Built an interactive Streamlit app to visualize historical sales and forecasts by category and region.

## How to Run
```bash
pip install -r requirements.txt
python prepare_data.py
streamlit run app.py
```

## Tools Used
Python, Pandas, Prophet, Matplotlib/Seaborn, Streamlit

## Author
Final-year BTech Data Science (CSD) student, Amrita Sai Institute of Science and Technology, Paritala.

