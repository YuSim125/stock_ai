# Stock AI — ML-Powered Stock Recommendation System

An end-to-end machine learning system that ingests market and
macroeconomic data, identifies patterns using ML models, and
recommends stock bundles ranked by predicted performance.

## Tech Stack
- **Data:** yFinance, FRED API
- **ML:** XGBoost, LSTM (PyTorch)
- **Ensemble:** Combined model scoring with confidence weighting
- **Optimization:** KMeans clustering for diversified bundles
- **Backtesting:** Walk-forward simulation with Sharpe, drawdown metrics
- **Dashboard:** Streamlit (coming soon)

## Project Structure

```
stock-ai/
├── pipeline/
│   ├── fetch_data.py               # Phase 1: S&P 500 data ingestion (502 stocks)
│   └── feature_engineering.py     # Phase 2: 30+ technical + macro signals
├── models/
│   ├── train_model.py              # Phase 3a: XGBoost baseline model
│   ├── lstm_model.py               # Phase 3b: LSTM deep learning model
│   ├── ensemble_bundle.py          # Phase 4: Ensemble scoring + bundle engine
│   └── results/                    # Model outputs, charts, scores
├── backtest/
│   └── backtest.py                 # Phase 5: Walk-forward backtesting
├── data/                           # gitignored - generated locally
├── sp500_tickers.txt               # Full S&P 500 ticker list
├── .env.example                    # API key template
└── requirements.txt
```

## Phases

- [x] Phase 1 - Data Pipeline (502 S&P 500 stocks, 1.2M rows, yFinance + FRED macro)
- [x] Phase 2 - Feature Engineering (30+ technical + macro signals, 1.08M rows)
- [x] Phase 3a - XGBoost Baseline Model (50% accuracy, 62.9% win rate on high-confidence picks)
- [x] Phase 3b - LSTM Deep Learning Model (57.2% accuracy, 58.3% win rate, RTX 2070 GPU trained)
- [x] Phase 4 - Ensemble Model + Portfolio Bundle Engine (3 diversified bundles, avg score 0.69)
- [x] Phase 5 - Backtesting & Validation (18.5% return, 0.886 Sharpe, -15% max drawdown)
- [ ] Phase 6 - Streamlit Dashboard (live recommendations, charts, portfolio tracker)

## Key Results

| Metric | Value |
|---|---|
| Stocks tracked | 502 (full S&P 500) |
| Training data | 1.08M rows (2015–2022) |
| High-confidence win rate | 58.3–62.9% |
| Backtest return (2023–2024) | 18.5% |
| Backtest Sharpe ratio | 0.886 |
| Max drawdown | -15.0% |
| Bundle avg ensemble score | 0.69 |

## How It Works

1. **Data Pipeline** — pulls daily OHLCV prices for all S&P 500 stocks via yFinance plus 7 macroeconomic indicators from the FRED API
2. **Feature Engineering** — computes 30+ signals per stock including RSI, MACD, Bollinger Bands, moving average crossovers, volume ratios, and macro regime features
3. **ML Models** — XGBoost learns tabular patterns; LSTM learns 60-day sequential patterns using PyTorch on GPU
4. **Ensemble Scoring** — combines both model probabilities with a confidence bonus when both agree, scoring all 500 stocks
5. **Bundle Engine** — clusters top stocks by return correlation using KMeans, picks diversified holdings across sectors
6. **Backtesting** — walk-forward simulation with transaction costs, compared against SPY and random selection

## Setup

1. Clone the repo
2. `pip install -r requirements.txt`
3. Add your FRED API key to `.env` (see `.env.example`)
4. Download S&P 500 tickers: `python -c "import pandas as pd; df=pd.read_csv('https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv'); df['Symbol'].to_csv('sp500_tickers.txt', index=False, header=False)"`
5. Run `python pipeline/fetch_data.py`
6. Run `python pipeline/feature_engineering.py`
7. Run `python models/train_model.py`
8. Run `python models/lstm_model.py`
9. Run `python models/ensemble_bundle.py`
10. Run `python backtest/backtest.py`

## Disclaimer

This project is for educational purposes only. Model outputs are not financial advice. Always do your own research before making investment decisions.