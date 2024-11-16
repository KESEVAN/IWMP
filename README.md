# Intelligent Wealth Management Pipeline (iWMP)
A machine learning pipeline for financial market analysis and prediction, combining traditional statistical methods with modern ML approaches.

## Overview
This project implements a comprehensive financial analysis system that processes market data through multiple modeling techniques (ARIMA, Prophet, LightGBM) while incorporating sentiment analysis. The goal is to provide accurate market predictions and risk assessments for wealth management applications.

## Tech Stack
- **Core:** Python, pandas, numpy
- **ML/Statistics:** LightGBM, Prophet, ARIMA (statsmodels)
- **AI/NLP:** Azure OpenAI, LangChain
- **MLOps:** MLflow, YAML config
- **Validation:** pydantic

## Current Features
- Technical indicator calculation (SMA)
- Risk metrics (VaR)
- Market sentiment analysis
- Experiment tracking with MLflow

## Project Structure
```
├── wealth.py           # Main pipeline implementation
├── ARIMA.py           # ARIMA model and portfolio analysis
├── config.yaml        # Configuration parameters
└── README.md         # Documentation
```

## Setup and Installation
```bash
git clone https://github.com/KESEVAN/iWMP.git
pip install -r requirements.txt
```

## Work in Progress
- [ ] Real-time data integration
- [ ] More indicators Sharp Ratio, Sortino Ratio, etc.
- [ ] Need to add Boulinger Bands, MACD, RSI, etc.
- [ ] API development
- [ ] Documentation improvements
- [ ] Unit tests

## Contributing
Feel free to open issues or submit pull requests. This project is under active development.

## License
MIT License