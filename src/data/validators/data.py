import yfinance as yf
import pandas as pd
import numpy as np
from typing import Optional, Tuple
import logging
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataAcquisition:
    """
    Class to handle market data fetching and preprocessing
    """
    def __init__(self):
        self.required_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'
        ]

    def fetch_and_clean_data(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        handle_missing: bool = True
    ) -> Tuple[pd.DataFrame, dict]:
        """
        Fetch and clean market data for a given ticker
        
        Parameters:
        -----------
        ticker : str
            Stock ticker symbol
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        handle_missing : bool
            Whether to handle missing values
            
        Returns:
        --------
        Tuple[pd.DataFrame, dict]
            Clean dataframe and data quality metrics
        """
        try:
            logger.info(f"Fetching data for {ticker} from {start_date} to {end_date}")
            
            # Add extra days to handle potential missing data
            start_dt = datetime.strptime(start_date, '%Y-%m-%d') - timedelta(days=5)
            end_dt = datetime.strptime(end_date, '%Y-%m-%d') + timedelta(days=5)
            
            # Fetch data
            stock_data = yf.download(
                ticker,
                start=start_dt.strftime('%Y-%m-%d'),
                end=end_dt.strftime('%Y-%m-%d'),
                progress=False
            )
            
            # Initial data quality check
            if stock_data.empty:
                raise ValueError(f"No data retrieved for {ticker}")
            
            # Clean and process the data
            clean_data = self._clean_market_data(stock_data, handle_missing)
            
            # Trim to exact date range
            clean_data = clean_data[start_date:end_date]
            
            # Generate data quality metrics
            quality_metrics = self._calculate_quality_metrics(clean_data)
            
            logger.info(f"Successfully processed data for {ticker}")
            
            return clean_data, quality_metrics
            
        except Exception as e:
            logger.error(f"Error processing {ticker}: {str(e)}")
            raise

    def _clean_market_data(self, df: pd.DataFrame, handle_missing: bool) -> pd.DataFrame:
        """Clean and preprocess market data"""
        # Make copy to avoid modifying original data
        df = df.copy()
        
        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        
        # Standardize column names
        df.columns = [col.replace(' ', '_') for col in df.columns]
        
        # Check for required columns
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        if handle_missing:
            # Handle missing values
            df = self._handle_missing_values(df)
        
        # Remove duplicates
        df = df[~df.index.duplicated(keep='first')]
        
        # Sort by date
        df = df.sort_index()
        
        # Remove rows with invalid values
        df = df[df['Close'] > 0]
        df = df[df['Volume'] >= 0]
        
        # Add date-related features
        df['Year'] = df.index.year
        df['Month'] = df.index.month
        df['Day'] = df.index.day
        df['DayOfWeek'] = df.index.dayofweek
        
        return df

    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Forward fill prices (carry forward last known price)
        price_columns = ['Open', 'High', 'Low', 'Close', 'Adj_Close']
        df[price_columns] = df[price_columns].fillna(method='ffill')
        
        # Backward fill if there are still missing values
        df[price_columns] = df[price_columns].fillna(method='bfill')
        
        # Fill missing volume with 0
        df['Volume'] = df['Volume'].fillna(0)
        
        return df

    def _calculate_quality_metrics(self, df: pd.DataFrame) -> dict:
        """Calculate data quality metrics"""
        metrics = {
            'total_rows': len(df),
            'date_range': f"{df.index.min()} to {df.index.max()}",
            'missing_values': df.isnull().sum().to_dict(),
            'price_range': {
                'min': df['Close'].min(),
                'max': df['Close'].max(),
                'mean': df['Close'].mean()
            },
            'volume_stats': {
                'min': df['Volume'].min(),
                'max': df['Volume'].max(),
                'mean': df['Volume'].mean()
            }
        }
        return metrics

def main():
    """Example usage of the MarketDataAcquisition class"""
    # Initialize data acquirer
    data_acquirer = MarketDataAcquisition()
    
    # Example parameters
    ticker = 'MSFT'
    start_date = '2020-01-01'
    end_date = '2024-01-01'
    
    try:
        # Fetch and clean data
        clean_data, quality_metrics = data_acquirer.fetch_and_clean_data(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date
        )
        
        # Save to CSV
        output_file = f'{ticker}_market_data.csv'
        clean_data.to_csv(output_file)
        
        # Log results
        logger.info(f"Market data for {ticker} saved to {output_file}")
        logger.info("Data quality metrics:")
        for metric, value in quality_metrics.items():
            logger.info(f"{metric}: {value}")
            
    except Exception as e:
        logger.error(f"Failed to process {ticker}: {str(e)}")
        raise

if __name__ == "__main__":
    main()