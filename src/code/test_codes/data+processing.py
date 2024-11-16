"""
This application implements a predictive time series forecasting model using ARIMA (AutoRegressive Integrated Moving Average) 
to predict the future trend of stock prices based on historical data. The project aims to predict stock price trends for 
a given ticker using the ARIMA model, leveraging financial data obtained via the Yahoo Finance API.

The following steps are performed:
1. Load and preprocess historical stock price data.
2. Fit an ARIMA model to the historical data to capture the underlying patterns.
3. Forecast future stock prices for a specified number of days ahead.
4. Visualize the historical data alongside the forecasted values to assess the accuracy and trends.

Key Features:
- Stock price trend prediction using ARIMA
- Data processing and preparation for model training
- Visualization of original data and forecasted predictions

Dependencies:
- pandas
- numpy
- matplotlib
- statsmodels
- yfinance

The model leverages ARIMA to model the time series data, considering autoregressive, differencing, and moving average components 
to make future predictions based on past data trends. The forecast results are visualized alongside the actual historical prices.

By adjusting the ARIMA model parameters (p, d, q), the application allows for fine-tuning to better match the data's underlying patterns.
"""

import pandas as pd

from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
# Load the market data with proper handling of unnamed columns and date parsing
def preprocess_data(file_path):
    # Read the CSV, skipping the first 2 rows of metadata (the 'Ticker' row and the blank row)
    data = pd.read_csv(file_path, skiprows=2, parse_dates=['Date'], index_col='Date')
    
    # Display the column names to check if everything is parsed correctly
    # print("Columns:", data.columns)
    
    # Manually rename columns, excluding unnamed columns
    data.columns = ['Price', 'Adj Close', 'Close', 'Low', 'High', 'Volume']
    
    # Drop the unnamed columns
    data.drop(columns=[col for col in data.columns if 'Unnamed' in col], inplace=True)
    
    # Display the first few rows to confirm correct loading
    print(data.head())

    # Dropping any rows with missing values
    data.dropna(inplace=True)

    # Feature Engineering:
    # Adding moving averages
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()
    
    # Adding daily returns
    data['Daily_Return'] = data['Close'].pct_change()
    
    # Adding volatility (rolling window of 50 days)
    data['Volatility'] = data['Daily_Return'].rolling(window=50).std()

    # Normalize the 'Close' prices using MinMaxScaler
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    data['Normalized_Close'] = scaler.fit_transform(data[['Close']])

    return data

# Example usage
file_path = 'AAPL_market_data.csv'
processed_data = preprocess_data(file_path)

# Print the first few rows to verify
print(processed_data.head())


# Function to fit an ARIMA model
def fit_arima_model(data, order=(10, 1, 0)):
    # Use the 'Close' column for prediction
    model = ARIMA(data['Close'], order=order)
    model_fit = model.fit()
    
    # Make predictions
    forecast = model_fit.forecast(steps=30)  # Predict next 30 days
    return forecast, model_fit

# Fit the ARIMA model and predict trends
forecast, model_fit = fit_arima_model(processed_data)

# Convert the forecast into a pandas series with proper date index if applicable
forecast_index = pd.date_range(start=processed_data.index[-2], periods=31, freq='B')[1:]
forecast_series = pd.Series(forecast, index=forecast_index)
# Plot the predictions
plt.figure(figsize=(10, 6))
plt.plot(processed_data['Close'], label='Actual Prices')
plt.plot(forecast_series, label='Forecast', color='red')
plt.plot(range(len(processed_data), len(processed_data) + 30), forecast, label='Forecast', color='red')
plt.title('Stock Price Trend Prediction using ARIMA')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

