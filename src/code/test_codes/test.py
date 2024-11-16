import yfinance as yf
import pandas as pd

# Fetch historical data for a specific stock (e.g., Apple)
def fetch_market_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    return stock_data

# Example usage
ticker = 'MSFT'
start_date = '2020-01-01'
end_date = '2024-01-01'
market_data = fetch_market_data(ticker, start_date, end_date)

# Save to CSV
market_data.to_csv(f'{ticker}_market_data.csv')
print(f"Market data for {ticker} saved.")