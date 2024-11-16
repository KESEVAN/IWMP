import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import logging
import yaml
import joblib
from pathlib import Path
import warnings

# ML imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import lightgbm as lgb
from prophet import Prophet

# Azure OpenAI for market sentiment analysis
from azure.ai.textanalytics import TextAnalyticsClient
from azure.core.credentials import AzureKeyCredential
import langchain
from langchain.llms import AzureOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn

# Data validation
from pydantic import BaseModel, Field
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketDataSchema(BaseModel):
    """Pydantic model for data validation"""
    date: datetime
    # open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: int = Field(gt=0)
    adj_close: float = Field(gt=0)

@dataclass
class ModelConfig:
    """Configuration for model training and prediction"""
    feature_cols: List[str]
    target_col: str
    test_size: float
    cv_folds: int
    prophet_params: Dict
    lgbm_params: Dict

class DataPipeline:
    """
    Data ingestion and preprocessing pipeline with validation
    """
    def __init__(self, config_path: str):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.scaler = MinMaxScaler()
        
    def validate_data(self, df: pd.DataFrame) -> bool:
        """Validate data using Pydantic schema"""
        try:
            print(df.head())
            for _, row in df.iterrows():
                MarketDataSchema(
                    date=row.name,  # Assuming the index is the Date
                    price=row['Price'],
                    adj_close=row['Adj Close'],
                    close=row['Close'],
                    high=row['High'],
                    low=row['Low'],
                    # open=row['Open'],
                    volume=row['Volume']
                )
            return True
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            return False

    def process_market_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process market data with feature engineering"""
        df = self._add_technical_indicators(df)
        df = self._add_risk_metrics(df)
        df = self._add_market_sentiment(df)
        return df

    def _add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators"""
        # Traditional indicators
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        # df['RSI'] = self._calculate_rsi(df['Close'])
        
        # Volatility indicators
        # df['ATR'] = self._calculate_atr(df)
        # df['BB_upper'], df['BB_lower'] = self._calculate_bollinger_bands(df['Close'])
        
        return df

    def _add_risk_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate risk metrics"""
        df['Daily_Returns'] = df['Close'].pct_change()
        df['Volatility'] = df['Daily_Returns'].rolling(window=20).std() * np.sqrt(252)
        df['VaR_95'] = df['Daily_Returns'].rolling(window=252).quantile(0.05)
        # df['Sharpe_Ratio'] = self._calculate_sharpe_ratio(df)
        return df

    def _add_market_sentiment(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add market sentiment using Azure OpenAI"""
        # Initialize Azure OpenAI client
        llm = AzureOpenAI(
            deployment_name="text-davinci-003",
            model_name="gpt-3.5-turbo"
        )
        
        # Create sentiment analysis chain
        template = """
        Analyze the market sentiment for {symbol} based on the following metrics:
        Price: {price}
        Volume: {volume}
        RSI: {rsi}
        
        Provide a sentiment score between -1 and 1.
        """
        
        prompt = PromptTemplate(
            input_variables=["symbol", "price", "volume", "rsi"],
            template=template
        )
        
        chain = LLMChain(llm=llm, prompt=prompt)
        
        # Calculate sentiment scores (simplified for demonstration)
        df['Sentiment_Score'] = df.apply(
            lambda x: float(chain.run(
                symbol="STOCK",
                price=x['Close'],
                volume=x['Volume'],
                rsi=x['RSI']
            )) if not pd.isna(x['RSI']) else np.nan,
            axis=1
        )
        
        return df

class ModelPipeline:
    """
    ML model training and prediction pipeline
    """
    def __init__(self, config: ModelConfig):
        self.config = config
        self.models = {
            'prophet': None,
            'lgbm': None
        }
        
    def train_models(self, df: pd.DataFrame) -> Dict:
        """Train multiple models and track experiments with MLflow"""
        with mlflow.start_run():
            # Train Prophet model
            prophet_model = self._train_prophet(df)
            self.models['prophet'] = prophet_model
            
            # Train LightGBM model
            lgbm_model = self._train_lgbm(df)
            self.models['lgbm'] = lgbm_model
            
            # Log models and parameters
            mlflow.log_params(self.config.prophet_params)
            mlflow.log_params(self.config.lgbm_params)
            
            # Log models
            mlflow.sklearn.log_model(lgbm_model, "lgbm_model")
            
            return self.models

    def _train_prophet(self, df: pd.DataFrame) -> Prophet:
        """Train Facebook Prophet model"""
        prophet_df = df.reset_index()[['Date', 'Close']].rename(
            columns={'Date': 'ds', 'Close': 'y'}
        )
        
        model = Prophet(**self.config.prophet_params)
        model.fit(prophet_df)
        
        return model

    def _train_lgbm(self, df: pd.DataFrame) -> lgb.LGBMRegressor:
        """Train LightGBM model"""
        X = df[self.config.feature_cols]
        y = df[self.config.target_col]
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        
        model = lgb.LGBMRegressor(**self.config.lgbm_params)
        
        # Train with cross-validation
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            
        return model

    def make_predictions(self, df: pd.DataFrame, horizon: int = 30) -> Dict[str, pd.DataFrame]:
        """Generate predictions from all models"""
        predictions = {}
        
        # Prophet predictions
        future = self.models['prophet'].make_future_dataframe(periods=horizon)
        prophet_pred = self.models['prophet'].predict(future)
        predictions['prophet'] = prophet_pred[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        
        # LightGBM predictions
        X = df[self.config.feature_cols]
        lgbm_pred = pd.DataFrame(
            self.models['lgbm'].predict(X),
            index=df.index,
            columns=['prediction']
        )
        predictions['lgbm'] = lgbm_pred
        
        return predictions

class WealthManagementPipeline:
    """
    Main pipeline orchestrator
    """
    def __init__(self, config_path: str):
        self.data_pipeline = DataPipeline(config_path)
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            self.model_config = ModelConfig(**config['model_config'])
        self.model_pipeline = ModelPipeline(self.model_config)

    def run(self, data_path: str) -> Dict:
        """Run the complete pipeline"""
        try:
            # Load and validate data
            # df = pd.read_csv(data_path, parse_dates=['Date'], index_col='Date')
            df = pd.read_csv(data_path, skiprows=2, parse_dates=['Date'], index_col='Date')
             # Manually rename columns, excluding unnamed columns
            df.columns = ['Price', 'Adj Close', 'Close', 'Low', 'High', 'Volume']
            
            # Drop the unnamed columns
            df.drop(columns=[col for col in df.columns if 'Unnamed' in col], inplace=True)
            
            if not self.data_pipeline.validate_data(df):
                raise ValueError("Data validation failed")

            # Process data
            processed_df = self.data_pipeline.process_market_data(df)

            # Train models
            models = self.model_pipeline.train_models(processed_df)

            # Generate predictions
            predictions = self.model_pipeline.make_predictions(processed_df)

            return {
                'processed_data': processed_df,
                'predictions': predictions,
                'models': models
            }

        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    config_path = "config.yaml"
    pipeline = WealthManagementPipeline(config_path)
    results = pipeline.run("MSFT_market_data.csv")