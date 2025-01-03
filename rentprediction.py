# Import necessary libraries
import os
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
from prophet import Prophet
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from keras import Input
from keras.models import Sequential
from keras.layers import Dense, LSTM
from scikeras.wrappers import KerasRegressor
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
        

def evaluate_model(model, X, y, X_test, y_test, predictions_rescaled):
    """
    Evaluates the performance of a machine learning model.

    Parameters:
    - model: Trained model to evaluate.
    - X: Input features (used for cross-validation).
    - y: Target values (used for cross-validation).
    - X_test: Test features used for evaluation.
    - y_test: True target values for the test set.
    - predictions_rescaled: Predictions made by the model, rescaled to the original range.

    Returns:
    - A dictionary of evaluation metrics.
    """
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_test, predictions_rescaled)
    mse = mean_squared_error(y_test, predictions_rescaled)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions_rescaled)
    
    # Perform cross-validation
    scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=5)
    rmse_scores = (-scores) ** 0.5

    # Print evaluation metrics
    print("Evaluation Metrics:")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    print(f"Cross-validated RMSE: {rmse_scores.mean():.4f}")
    
    # Return metrics as a dictionary
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "CV_RMSE": rmse_scores.mean()
    }
    
def exec_linear_regression(lrdata):
    
    print("Linear Regression")
    
    # Work on a copy to prevent modifying the global DataFrame
    lrdata = lrdata.copy()
    
    # Create lag features (e.g., predict using last 3 months)
    lrdata["Lag1"] = lrdata["RentPrice"].shift(1)
    lrdata["Lag2"] = lrdata["RentPrice"].shift(2)
    lrdata["Lag3"] = lrdata["RentPrice"].shift(3)
    lrdata = lrdata.dropna()
    
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
    
    # Evaluate the model using the evaluation function
    evaluation = evaluate_model(
        model=model,
        X=X,
        y=y,
        X_test=X_test,
        y_test=y_test,
        predictions_rescaled=y_pred  # Rescaled predictions are just `y_pred` for Linear Regression
    )
    
    
    # Predict next month's rent
    latest_data = lrdata[["Lag1", "Lag2", "Lag3"]].iloc[-1].values.reshape(1, -1)
    next_month_rent = model.predict(latest_data)
    print("Predicted Rent for Next Month:", next_month_rent[0])
    
    # Plot 1: Predicted vs Actual
    plt.figure(figsize=(8, 6))  # Create a new figure
    plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Fit Line')
    plt.xlabel("Actual Prices", fontsize=12)
    plt.ylabel("Predicted Prices", fontsize=12)
    plt.legend()
    plt.title("Linear Regression: Predicted vs Actual Prices", fontsize=14)
    plt.grid(True)  # Add grid for better visualization
    plt.show()  # Display the plot
    
    # Plot 2: Residuals
    plt.figure(figsize=(8, 6))  # Create a new figure
    residuals = y_test - y_pred
    plt.scatter(y_test, residuals, color='purple', alpha=0.6)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Actual Prices", fontsize=12)
    plt.ylabel("Residuals", fontsize=12)
    plt.title("Linear Regression: Residual Plot", fontsize=14)
    plt.grid(True)  # Add grid for better visualization
    plt.show()  # Display the plot
    
    return(lrdata)

def build_lstm_model():
    model = Sequential([
        Input(shape=(3, 1)),  # Define the input shape explicitly
        LSTM(50, return_sequences=True),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def k_fold_cross_validation(X, y, k=5):
    fold_size = len(X) // k
    maes, rmses, r2s = [], [], []

    for i in range(k):
        # Split into train and test sets
        start = i * fold_size
        end = start + fold_size

        X_test = X[start:end]
        y_test = y[start:end]

        X_train = np.concatenate((X[:start], X[end:]), axis=0)
        y_train = np.concatenate((y[:start], y[end:]), axis=0)

        # Reshape for LSTM input
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

        # Build and train the model
        model = build_lstm_model()
        model.fit(X_train, y_train, epochs=10, batch_size=16, verbose=0)

        # Predict and evaluate
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, predictions)

        maes.append(mae)
        rmses.append(rmse)
        r2s.append(r2)

    # Return average metrics
    return {
        "MAE": np.mean(maes),
        "RMSE": np.mean(rmses),
        "R²": np.mean(r2s)
    }

def exec_tensorflow(tdata):
    
    print("TensorFlow LSTM")
    
    # Work on a copy to prevent modifying the global DataFrame
    tdata = tdata.copy()

    # Normalize data
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaled_X = scaler_X.fit_transform(tdata.values)
    scaled_y = scaler_y.fit_transform(tdata["RentPrice"].values.reshape(-1, 1))
    
    # Create sequences for LSTM (e.g., last 3 months to predict next month)
    X, y = [], []
    for i in range(3, len(scaled_X)):
        X.append(scaled_X[i-3:i])  # Last 3 months
        y.append(scaled_y[i])      # Next month's rent
    X, y = np.array(X), np.array(y)
    
    # Perform manual cross-validation
    metrics = k_fold_cross_validation(X, y, k=5)
    
    print("Cross-Validated Metrics:")
    print("Average MAE:", metrics["MAE"])
    print("Average RMSE:", metrics["RMSE"])
    print("Average R²:", metrics["R²"])
    
    # Split into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    # Reshape for LSTM input
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))  # Reshape to (n_samples, time_steps, n_features)
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
    
    # Build the LSTM model
    model = KerasRegressor(model=build_lstm_model, epochs=20, batch_size=16, verbose=0)
    # Train the model
    model.fit(X_train, y_train)
    
    # Predict and rescale
    predictions = model.predict(X_test).reshape(-1, 1)  # Ensure 2D shape for scaler
    predictions_rescaled = scaler_y.inverse_transform(predictions)
    
    # Evaluate the model
    evaluation = evaluate_model(
        model=model,
        X=X,
        y=y,
        X_test=X_test,
        y_test=y_test,
        predictions_rescaled=predictions_rescaled
    )
    
    # Predict next month's rent
    # Extract the last 3 months for prediction
    latest_sequence = scaled_X[-3:].reshape(1, 3, 1)  # Shape: (1, time_steps, n_features)
    # Predict next month's scaled rent
    next_month_prediction_scaled = model.predict(latest_sequence).reshape(-1, 1)
    # Rescale the prediction back to the original range
    next_month_rent = scaler_y.inverse_transform(next_month_prediction_scaled)
    print("Predicted Rent for Next Month:", next_month_rent[0][0])
    
    return(tdata)


    
def exec_prophet(pdata):
    
    print("Prophet")
    
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
    
    # Predict next month's rent
    next_month_prediction = forecast.iloc[-1]["yhat"]
    print(f"\nPredicted Rent for Next Month: {next_month_prediction:.2f}")
    
    # Optional: Plot the forecast
    fig = model.plot(forecast)
    
    # Add title and labels
    fig.suptitle("Prophet Forecast: Rent Prediction", fontsize=16)
    plt.xlabel("Date")
    plt.ylabel("Rent Price")
    plt.show()
    
    return(pdata)
    
if __name__ == "__main__":
    
    # Specify the relative path to your CSV file
    relative_csv_path = "data/Metro_zori_uc_sfrcondomfr_sm_sa_month.csv"  # Adjust this path as needed
    
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
        
    tdata_p = exec_tensorflow(time_series_data)
    lgdata_p = exec_linear_regression(time_series_data)
    pdata_p = exec_prophet(time_series_data)