import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class PortfolioAnalyzer:
    """
    A comprehensive tool for financial analysis and portfolio management.
    Suitable for wealth management applications with focus on risk metrics
    and investment analysis.
    """
    
    def __init__(self):
        self.scaler = MinMaxScaler()
        
    def preprocess_data(self, file_path: str) -> pd.DataFrame:
        """
        Preprocesses financial data with advanced feature engineering
        and risk metrics calculation.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file containing market data
            
        Returns:
        --------
        pd.DataFrame
            Processed dataframe with additional features and risk metrics
        """
        # Load and clean data
        data = pd.read_csv(file_path, skiprows=2, parse_dates=['Date'], index_col='Date')
        data.columns = ['Price', 'Adj Close', 'Close', 'Low', 'High', 'Volume']
        data.drop(columns=[col for col in data.columns if 'Unnamed' in col], inplace=True)
        data.dropna(inplace=True)
        
        # Technical Indicators
        self._add_technical_indicators(data)
        
        # Risk Metrics
        self._add_risk_metrics(data)
        
        # Value at Risk (VaR) calculation
        data['VaR_95'] = self._calculate_var(data['Daily_Return'], confidence_level=0.95)
        data['VaR_99'] = self._calculate_var(data['Daily_Return'], confidence_level=0.99)
        
        return data
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> None:
        """
        Adds technical analysis indicators to the dataset.
        """
        # Moving Averages
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        
        # Bollinger Bands
        data['BB_middle'] = data['Close'].rolling(window=20).mean()
        bb_std = data['Close'].rolling(window=20).std()
        data['BB_upper'] = data['BB_middle'] + (bb_std * 2)
        data['BB_lower'] = data['BB_middle'] - (bb_std * 2)
        
        # Relative Strength Index (RSI)
        delta = data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
    
    def _add_risk_metrics(self, data: pd.DataFrame) -> None:
        """
        Calculates and adds various risk metrics to the dataset.
        """
        # Returns
        data['Daily_Return'] = data['Close'].pct_change()
        data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Volatility metrics
        data['Volatility_30d'] = data['Daily_Return'].rolling(window=30).std() * np.sqrt(252)
        data['Volatility_60d'] = data['Daily_Return'].rolling(window=60).std() * np.sqrt(252)
        
        # Maximum Drawdown
        rolling_max = data['Close'].rolling(window=252, min_periods=1).max()
        drawdown = data['Close'] / rolling_max - 1.0
        data['Max_Drawdown'] = drawdown.rolling(window=252, min_periods=1).min()
        
        # Sharpe Ratio (assuming risk-free rate of 2%)
        rf_rate = 0.02
        excess_returns = data['Daily_Return'] - rf_rate/252
        data['Sharpe_Ratio'] = (excess_returns.rolling(window=252).mean() / 
                               data['Daily_Return'].rolling(window=252).std()) * np.sqrt(252)
    
    def _calculate_var(self, returns: pd.Series, confidence_level: float) -> float:
        """
        Calculates Value at Risk using historical simulation method.
        """
        return returns.quantile(1 - confidence_level)
    
    def fit_arima_model(self, data: pd.DataFrame, 
                       order: Tuple[int, int, int]=(5,1,2),
                       forecast_days: int=30) -> Tuple[pd.Series, float]:
        """
        Fits an ARIMA model with optimal parameters and provides forecasts
        with confidence intervals."""
        model = ARIMA(data['Close'], order=order)
        model_fit = model.fit()
        
        # Generate forecast with confidence intervals
        forecast = model_fit.forecast(steps=forecast_days)
        forecast_ci = model_fit.get_forecast(steps=forecast_days).conf_int()
        
        # Create forecast index
        forecast_index = pd.date_range(
            start=data.index[-1] + pd.Timedelta(days=1),
            periods=forecast_days,
            freq='B'
        )
        
        return pd.Series(forecast, index=forecast_index), model_fit.aic
    
    def generate_analysis_report(self, data: pd.DataFrame) -> Dict:
    
        latest_price = data['Close'].iloc[-1]
        annual_return = (data['Close'].iloc[-1] / data['Close'].iloc[0]) ** (252/len(data)) - 1
        
        report = {
            'latest_price': latest_price,
            'annual_return': annual_return,
            'annual_volatility': data['Volatility_30d'].iloc[-1],
            'sharpe_ratio': data['Sharpe_Ratio'].iloc[-1],
            'max_drawdown': data['Max_Drawdown'].iloc[-1],
            'var_95': data['VaR_95'].iloc[-1],
            'var_99': data['VaR_99'].iloc[-1],
            'rsi': data['RSI'].iloc[-1],
            'is_oversold': data['RSI'].iloc[-1] < 30,
            'is_overbought': data['RSI'].iloc[-1] > 70,
        }
        
        return report
    
    def plot_analysis(self, data: pd.DataFrame, forecast: pd.Series) -> None:
    
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 15))
        
        # Price and Forecasts Plot
        ax1.plot(data.index, data['Close'], label='Actual Price')
        ax1.plot(data['SMA_50'], label='50-day SMA', alpha=0.7)
        ax1.plot(data['SMA_200'], label='200-day SMA', alpha=0.7)
        ax1.plot(forecast.index, forecast, 'r--', label='Forecast')
        ax1.fill_between(data.index, data['BB_upper'], data['BB_lower'], alpha=0.1)
        ax1.set_title('Price Trends and Technical Indicators')
        ax1.legend()
        
        # Volatility and Returns Plot
        ax2.plot(data.index, data['Volatility_30d'], label='30-day Volatility')
        ax2.plot(data.index, data['Daily_Return'].rolling(window=30).mean(), 
                label='30-day Avg Return')
        ax2.set_title('Volatility and Returns Analysis')
        ax2.legend()
        
        # Risk Metrics Plot
        ax3.plot(data.index, data['Max_Drawdown'], label='Maximum Drawdown')
        ax3.plot(data.index, data['Sharpe_Ratio'], label='Sharpe Ratio')
        ax3.set_title('Risk Metrics')
        ax3.legend()
        
        plt.tight_layout()
        plt.show()

# Example usage
def main():
    analyzer = PortfolioAnalyzer()
    
    # Process data
    data = analyzer.preprocess_data('../data/AAPL_market_data.csv')
    
    # Generate forecast
    forecast, aic = analyzer.fit_arima_model(data)
    
    # Generate analysis report
    report = analyzer.generate_analysis_report(data)
    print("\nAnalysis Report:")
    for key, value in report.items():
        print(f"{key}: {value:.4f}")
    
    # Create visualizations
    analyzer.plot_analysis(data, forecast)

if __name__ == "__main__":
    main()