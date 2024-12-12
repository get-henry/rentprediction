# Import necessary libraries
import os
import numpy as np
import pandas as pd
#from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt

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

def exec_prophet(data):
    # Step 4: Initialize and train the Prophet model
    model = Prophet()
    model.fit(prophet_data)
    
    # Step 5: Create future dates for prediction (next 12 months)
    future_dates = model.make_future_dataframe(periods=12, freq="M")
    
    # Step 6: Predict future rental prices
    forecast = model.predict(future_dates)
    
    # Step 7: Display predictions for the next 12 months
    future_predictions = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(12)
    print(future_predictions)
    
    # Optional: Plot the forecast
    model.plot(forecast)
    
def exec_linear_regression(region_data):
    # Create lag features (e.g., predict using last 3 months)
    region_data["Lag1"] = region_data["RentPrice"].shift(1)
    region_data["Lag2"] = region_data["RentPrice"].shift(2)
    region_data["Lag3"] = region_data["RentPrice"].shift(3)
    region_data = region_data.dropna()
    
    # Features and target
    X = region_data[["Lag1", "Lag2", "Lag3"]]
    y = region_data["RentPrice"]
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print("Mean Squared Error:", mse)
    print("R-squared:", r2)
    
    # Step 7: Visualize predictions vs actual values
    plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', label='Ideal Fit Line')
    plt.xlabel("Actual Prices")
    plt.ylabel("Predicted Prices")
    plt.legend()
    plt.title("Predicted vs Actual Prices")
    plt.show()
    
    # Predict next month's rent
    latest_data = region_data[["Lag1", "Lag2", "Lag3"]].iloc[-1].values.reshape(1, -1)
    next_month_rent = model.predict(latest_data)
    print("Predicted Rent for Next Month:", next_month_rent[0])

def exec_tensorflow(region_data):
    # Normalize data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(region_data.values)
    
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
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Build the LSTM model
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    
    # Train the model
    model.fit(X_train, y_train, epochs=20, batch_size=16, validation_data=(X_test, y_test))
    
    # Predict and rescale
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    
    # Predict next month's rent
    latest_sequence = scaled_data[-3:].reshape(1, 3, 1)  # Last 3 months
    next_month_rent = scaler.inverse_transform(model.predict(latest_sequence))
    print("Predicted Rent for Next Month:", next_month_rent[0][0])

# Example usage
if __name__ == "__main__":
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
    
    # Step 3: Prepare data for Prophet
    prophet_data = time_series_data.reset_index()
    prophet_data.columns = ["ds", "y"]  # Prophet requires columns named 'ds' (date) and 'y' (value)
    
    #exec_prophet(prophet_data) #ModuleNotFoundError: No module named 'prophet'
    exec_linear_regression(time_series_data)
    exec_tensorflow(time_series_data)