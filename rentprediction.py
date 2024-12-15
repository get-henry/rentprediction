# Import necessary libraries
import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
#import argparse
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import matplotlib.pyplot as plt

# Suppress FutureWarnings from Prophet and Pandas
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses INFO and WARNING messages
tf.get_logger().setLevel('ERROR')  # Log only errors

def import_csv_from_relative_path(relative_path):
    try:
        # Get the absolute path to the current script's directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Construct the full path to the CSV file
        csv_path = os.path.join(base_dir, relative_path)
        
        # Read the CSV file into a DataFrame
        data = pd.read_csv(csv_path)
        print(f"CSV file imported successfully from: {csv_path}")
        return data
    except FileNotFoundError:
        print(f"File not found at the specified path: {relative_path}")
    except Exception as e:
        print(f"An error occurred while importing the CSV: {e}")
    
def exec_linear_regression(lrdata):
    # Work on a copy to prevent modifying the global DataFrame
    lrdata = lrdata.copy()
    
    # Create lag features (e.g., predict using last 3 months)
    lrdata["Lag1"] = lrdata["RentPrice"].shift(1)
    lrdata["Lag2"] = lrdata["RentPrice"].shift(2)
    lrdata["Lag3"] = lrdata["RentPrice"].shift(3)
    lrdata = lrdata.dropna()
    
    #Allow the function to dynamically accept the number of lag features:
    #def exec_linear_regression(lrdata, lags=3):
    #for i in range(1, lags + 1):
    #    lrdata[f"Lag{i}"] = lrdata["RentPrice"].shift(i)
    #lrdata = lrdata.dropna()
    
    # Features and target
    X = lrdata[["Lag1", "Lag2", "Lag3"]]
    y = lrdata["RentPrice"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    rmse_scores = (-scores) ** 0.5
    
    print("Mean Absolute Error:", mae)
    print("Mean Squared Error:", mse)
    print("Root Mean Squared Error:", rmse)
    print("R-squared:", r2)
    print("Cross-validated RMSE:", rmse_scores.mean())
    
    # Predict next month's rent
    latest_data = lrdata[["Lag1", "Lag2", "Lag3"]].iloc[-1].values.reshape(1, -1)
    next_month_rent = model.predict(latest_data)
    print("Predicted Rent for Next Month:", next_month_rent[0])
    
    # Plot Predicted vs Actual
    plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Ideal Fit Line')
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.legend()
    plt.title("Predicted vs Actual Prices")
    plt.show()
    
    # Plot Residuals
    residuals = y_test - y_pred
    plt.scatter(y_test, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Actual Prices")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()
    
    return(lrdata)

def build_lstm_model():
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(3, 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def exec_tensorflow(tdata):
    
    # Work on a copy to prevent modifying the global DataFrame
    tdata = tdata.copy()

    # Normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(tdata.values)
    
    # Create sequences for LSTM (e.g., last 3 months to predict next month)
    X, y = [], []
    for i in range(3, len(scaled_data)):
        X.append(scaled_data[i-3:i, 0])  # Last 3 months
        y.append(scaled_data[i, 0])      # Next month's rent
    X, y = np.array(X), np.array(y)
    
    # Split into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Reshape for LSTM input
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # Reshape to (n_samples, time_steps, n_features)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Build the LSTM model
    # Wrap LSTM model for scikit-learn compatibility
    model = KerasRegressor(build_fn=build_lstm_model, epochs=20, batch_size=16, verbose=0)
    
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict and rescale
    #predictions = model.predict(X_test)
    #predictions = scaler.inverse_transform(predictions)
    
    predictions = model.predict(X_test).reshape(-1, 1)  # Reshape predictions to 2D
    predictions_rescaled = scaler.inverse_transform(predictions)
    
    # Evaluate the model
    mae = mean_absolute_error(y_test, predictions_rescaled)
    mse = mean_squared_error(y_test, predictions_rescaled)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions_rescaled)
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    rmse_scores = (-scores) ** 0.5

    print("LSTM Mean Absolute Error:", mae)
    print("LSTM Mean Squared Error:", mse)
    print("LSTM Root Mean Squared Error:", rmse)
    print("LSTM R-squared:", r2)
    print("LSTM Cross-validated RMSE:", rmse_scores.mean())
    
    
    # Predict next month's rent
    latest_sequence = scaled_data[-3:].reshape(1, 3, 1)  # Last 3 months
    next_month_rent = scaler.inverse_transform(model.predict(latest_sequence).reshape(-1, 1))
    print("Predicted Rent for Next Month:", next_month_rent[0][0])
    
    return(tdata)


    
def exec_prophet(pdata):
    # Work on a copy to prevent modifying the global DataFrame
    pdata = pdata.copy()
    
    # Step 3: Prepare data for Prophet
    pdata = pdata.reset_index()
    pdata.columns = ["ds", "y"]  # Prophet requires columns named 'ds' (date) and 'y' (value)
    
    # Step 4: Initialize and train the Prophet model
    model = Prophet()
    model.fit(pdata)
    
    # Step 5: Create future dates for prediction (next 12 months)
    future_dates = model.make_future_dataframe(periods=12, freq="M")
    
    # Step 6: Predict future rental prices
    forecast = model.predict(future_dates)
    
    # Step 7: Display predictions for the next 12 months
    future_predictions = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(12)
    print(future_predictions)
    
    # Optional: Plot the forecast
    model.plot(forecast)
    
    return(pdata)
    
# Example usage
if __name__ == "__main__":
    
    print(tf.__version__)
        
    # Specify the relative path to your CSV file
    relative_csv_path = "data/Metro_zori_uc_sfrcondomfr_sm_month.csv"  # Adjust this path as needed
    
    # Import the CSV
    data = import_csv_from_relative_path(relative_csv_path)
    
    # Step 1: Focus on a specific region (e.g., "United States")
    region_data = data[data["RegionName"] == "United States"]
    
    # Step 2: Prepare the time-series data
    time_series_data = region_data.drop(
        columns=["RegionID", "SizeRank", "RegionName", "RegionType", "StateName"]
    ).T
    
    # Rename columns for clarity
    time_series_data.columns = ["RentPrice"]
    
    # Convert the index to a datetime object
    time_series_data.index = pd.to_datetime(time_series_data.index)
    
    #parser = argparse.ArgumentParser(description="Run forecasting models.")
    #parser.add_argument("--method", type=str, choices=["prophet", "linear", "tensorflow"], required=True)
    #args = parser.parse_args()

    #if args.method == "prophet":
    #    exec_prophet(time_series_data)
    #elif args.method == "linear":
    #    exec_linear_regression(time_series_data)
    #elif args.method == "tensorflow":
    #    exec_tensorflow(time_series_data)
    
    
    tdata_p = exec_tensorflow(time_series_data)
    lgdata_p = exec_linear_regression(time_series_data)
    pdata_p = exec_prophet(time_series_data)