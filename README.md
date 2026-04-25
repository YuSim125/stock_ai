# Stock AI — ML-Powered Stock Recommendation System

An end-to-end machine learning system that ingests market and 
macroeconomic data, identifies patterns using ML models, and 
recommends stock bundles ranked by predicted performance.

## Tech Stack
- **Data:** yFinance, FRED API
- **ML:** XGBoost, LSTM (PyTorch), FinBERT
- **Optimization:** PyPortfolioOpt
- **Dashboard:** Streamlit

## Project Structure
```
stock-ai/
├── pipeline/
│   ├── fetch_data.py           # Phase 1: data ingestion
│   └── feature_engineering.py # Phase 2: feature engineering
├── models/
│   └── train_model.py          # Phase 3: ML model training
├── data/                       # gitignored - generated locally
└── models/results/             # gitignored - generated locally
```

## Phases
- [x] Phase 1 - Data Pipeline (yFinance + FRED macro data)
- [x] Phase 2 - Feature Engineering (30+ technical + macro signals)
- [x] Phase 3 - XGBoost Baseline Model
- [ ] Phase 4 - LSTM Deep Learning Model
- [ ] Phase 5 - Portfolio Bundle Engine
- [ ] Phase 6 - Backtesting & Validation
- [ ] Phase 7 - Streamlit Dashboard

## Setup
1. Clone the repo
2. `pip install -r requirements.txt`
3. Add your FRED API key to `.env`
4. Run `python pipeline/fetch_data.py`