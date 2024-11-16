import yfinance as yf
import polars as pl

class MarketDataCollector:
    def __init__(self, symbols: list[str]):
        self.symbols = symbols
        
    def fetch_historical_data(self, period: str = "1y") -> pl.DataFrame:
        """Fetch historical market data using yfinance"""
        data = []
        for symbol in self.symbols:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            hist['symbol'] = symbol
            data.append(hist)
        return pl.from_pandas(pd.concat(data))

    def process_data(self, df: pl.DataFrame) -> pl.DataFrame:
        """Process and clean market data"""
        return df.with_columns([
            pl.col('Close').pct_change().alias('returns'),
            pl.col('Volume').log().alias('log_volume')
        ]).drop_nulls()