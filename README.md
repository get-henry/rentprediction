# Rent Prediction Using Zillow Data

This repository contains Python scripts for predicting rental prices using Zillow's **Observed Rent Index (ZORI)** dataset. The project uses various machine learning models to analyze and forecast rent trends, including:

1. **Linear Regression**: A straightforward model for short-term predictions.
2. **TensorFlow LSTM**: A memory-based deep learning model for capturing sequential trends.
3. **Prophet**: A time-series forecasting model for identifying long-term trends and seasonality.

---

## Features
- **Data Processing**: The script processes Zillow's **ZORI dataset** for either a single region (e.g., "United States") or multiple regions.
- **Model Evaluation**: Metrics like MAE, RMSE, and R² are calculated to assess model accuracy.
- **Forecasting**:
  - Linear Regression predicts rent based on the last 3 months.
  - LSTM uses deep learning for sequential forecasting.
  - Prophet identifies long-term trends and forecasts future rent prices.
- **Visualization**:
  - Predicted vs. Actual Prices.
  - Residual Plots for Linear Regression.
  - Prophet’s forecast plots with confidence intervals.

---

## Getting Started

### Prerequisites
1. Python 3.10 or later.
2. Required libraries:
   ```bash
   pip install tensorflow keras scikit-learn matplotlib pandas numpy prophet scikeras
   ```

### Data Source

This project uses Zillow’s ZORI (Smoothed, Seasonally Adjusted) dataset. You can download the dataset from Zillow Research Data and place it in the data/ directory.

### Recommended dataset:	
- Metro Level ZORI (Smoothed, Seasonally Adjusted): [Download here](https://www.zillow.com/research/data/)

---
## How to Use
1.	Download the Dataset:
  -	Ensure the dataset is saved in the data/ folder with the name Metro_zori_uc_sfrcondomfr_sm_sa_month.csv.
2.	Run the Script:
 ```bash
  python rentprediction.py
  ```
3.	Choose Analysis Mode:
	-	To process data for the entire USA: The script processes only the row corresponding to "United States".
	-	To process data for all regions: Modify the script to iterate through all rows (see the for region_name, region_data section in the code).
4.	Outputs:
	-	Predicted rent prices for the next month.
	-	Visualizations for predictions and residuals.

---

## Script Overview

### Main Functions
1.	exec_linear_regression(lrdata)
	-	Predicts rent using a linear model based on the last 3 months of data.
	-	Outputs predicted vs. actual plots and residual plots.
2.	exec_tensorflow(tdata)
	-	Uses LSTM to predict rent trends based on sequential data.
	-	Implements k-fold cross-validation and evaluates model performance.
3.	exec_prophet(pdata)
	-	Forecasts rent using Prophet, highlighting trends and seasonality.
	-	Produces future rent predictions with confidence intervals.
4.	evaluate_model(model, X, y, X_test, y_test, predictions_rescaled)
	-	Evaluates the performance of models using metrics like MAE, RMSE, and R².

---
## Future Improvements
-	Add support for visualizing trends for all regions.
-	Optimize LSTM for better prediction accuracy.
-	Incorporate additional datasets like ZORDI for analyzing rental demand.
