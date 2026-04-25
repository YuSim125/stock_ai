# ============================================================
#  feature_engineering.py  —  Stock AI Project: Phase 2
#  Takes raw master dataset and builds meaningful signals
#  that the ML model will actually learn patterns from
# ============================================================
#
#  HOW TO RUN:
#  python pipeline/feature_engineering.py
#
#  INPUT:  data/processed/master_dataset.csv  (from Phase 1)
#  OUTPUT: data/processed/features_dataset.csv (ready for ML)
# ============================================================

import os
import pandas as pd
import numpy as np

# ─────────────────────────────────────────────────────────
#  LOAD THE MASTER DATASET FROM PHASE 1
# ─────────────────────────────────────────────────────────

INPUT_PATH  = "data/processed/master_dataset.csv"
OUTPUT_PATH = "data/processed/features_dataset.csv"

print("=" * 55)
print("  Stock AI — Feature Engineering Starting")
print("=" * 55)

print(f"\n📂 Loading master dataset from {INPUT_PATH}...")
df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
print(f"   Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
print(f"   Stocks: {df['Ticker'].unique().tolist()}\n")


# ─────────────────────────────────────────────────────────
#  HELPER: process one stock at a time
#  We do this because indicators like moving averages
#  should NEVER bleed across different stocks
# ─────────────────────────────────────────────────────────

def engineer_features(stock_df):
    """
    Takes one stock's data and adds all technical +
    macro + target features. Returns enriched DataFrame.
    """
    # Work on a copy so we don't modify the original
    d = stock_df.copy().sort_index()

    # ── SECTION 1: PRICE MOMENTUM FEATURES ──────────────
    # These capture whether a stock is trending up or down
    # and how strongly

    # Returns over different time windows
    # "How much did the stock gain/lose over the last N days?"
    d["Return_5d"]  = d["Close"].pct_change(5)   # 1 week
    d["Return_10d"] = d["Close"].pct_change(10)  # 2 weeks
    d["Return_20d"] = d["Close"].pct_change(20)  # 1 month
    d["Return_60d"] = d["Close"].pct_change(60)  # 3 months

    # Moving average crossovers
    # When the short MA crosses above the long MA = bullish signal
    # When it crosses below = bearish signal
    d["MA_10"]  = d["Close"].rolling(10).mean()
    d["MA_20"]  = d["Close"].rolling(20).mean()
    d["MA_50"]  = d["Close"].rolling(50).mean()
    d["MA_200"] = d["Close"].rolling(200).mean()

    # Distance from moving averages (as % above/below)
    # e.g. +0.05 means price is 5% above the 50-day average
    d["Price_vs_MA20"]  = (d["Close"] - d["MA_20"])  / d["MA_20"]
    d["Price_vs_MA50"]  = (d["Close"] - d["MA_50"])  / d["MA_50"]
    d["Price_vs_MA200"] = (d["Close"] - d["MA_200"]) / d["MA_200"]

    # MA crossover signal: positive = short MA above long MA (bullish)
    d["MA_Cross_20_50"]  = d["MA_20"]  - d["MA_50"]
    d["MA_Cross_50_200"] = d["MA_50"]  - d["MA_200"]

    # ── SECTION 2: RSI (Relative Strength Index) ─────────
    # RSI measures if a stock is "overbought" or "oversold"
    # RSI > 70 = overbought (might drop soon)
    # RSI < 30 = oversold (might bounce back up)
    # RSI ranges from 0 to 100

    def compute_rsi(series, period=14):
        delta = series.diff()                          # daily price change
        gain  = delta.clip(lower=0)                    # only positive days
        loss  = -delta.clip(upper=0)                   # only negative days
        avg_gain = gain.rolling(period).mean()
        avg_loss = loss.rolling(period).mean()
        rs  = avg_gain / (avg_loss + 1e-10)            # ratio of gains to losses
        rsi = 100 - (100 / (1 + rs))
        return rsi

    d["RSI_14"] = compute_rsi(d["Close"], 14)
    d["RSI_7"]  = compute_rsi(d["Close"], 7)   # shorter = more sensitive

    # ── SECTION 3: MACD ──────────────────────────────────
    # MACD = Moving Average Convergence Divergence
    # One of the most widely used momentum indicators
    # Positive MACD = upward momentum, Negative = downward

    ema_12 = d["Close"].ewm(span=12, adjust=False).mean()  # fast EMA
    ema_26 = d["Close"].ewm(span=26, adjust=False).mean()  # slow EMA
    d["MACD"]        = ema_12 - ema_26                     # MACD line
    d["MACD_Signal"] = d["MACD"].ewm(span=9, adjust=False).mean()  # signal line
    d["MACD_Hist"]   = d["MACD"] - d["MACD_Signal"]       # histogram (momentum shift)

    # ── SECTION 4: BOLLINGER BANDS ────────────────────────
    # Bollinger Bands show if a stock is unusually high or low
    # relative to its recent trading range
    # Price touching upper band = potentially overbought
    # Price touching lower band = potentially oversold

    bb_window = 20
    bb_std    = 2
    rolling_mean = d["Close"].rolling(bb_window).mean()
    rolling_std  = d["Close"].rolling(bb_window).std()
    d["BB_Upper"] = rolling_mean + (bb_std * rolling_std)
    d["BB_Lower"] = rolling_mean - (bb_std * rolling_std)
    # %B: where is the price within the bands? 1 = upper band, 0 = lower band
    d["BB_Pct"] = (d["Close"] - d["BB_Lower"]) / (d["BB_Upper"] - d["BB_Lower"] + 1e-10)
    # Band width: how wide are the bands? Wider = more volatile market
    d["BB_Width"] = (d["BB_Upper"] - d["BB_Lower"]) / (rolling_mean + 1e-10)

    # ── SECTION 5: VOLUME FEATURES ───────────────────────
    # Volume = how many shares were traded that day
    # High volume on an up day = strong buying conviction
    # High volume on a down day = strong selling pressure

    d["Volume_MA_20"]  = d["Volume"].rolling(20).mean()
    # Volume ratio: is today's volume higher or lower than average?
    d["Volume_Ratio"]  = d["Volume"] / (d["Volume_MA_20"] + 1e-10)
    # On-Balance Volume: running total — rises on up days, falls on down days
    d["OBV"] = (np.sign(d["Close"].diff()) * d["Volume"]).fillna(0).cumsum()

    # ── SECTION 6: VOLATILITY FEATURES ───────────────────
    # Volatility tells us how risky/unpredictable a stock is
    # Models use this to adjust confidence in predictions

    d["Volatility_5d"]  = d["Daily_Return"].rolling(5).std()
    d["Volatility_20d"] = d["Daily_Return"].rolling(20).std()
    d["Volatility_60d"] = d["Daily_Return"].rolling(60).std()

    # Volatility regime: is the stock more or less volatile than usual?
    d["Vol_Regime"] = d["Volatility_20d"] / (d["Volatility_60d"] + 1e-10)

    # Average True Range (ATR): measures daily price range volatility
    high_low   = d["High"] - d["Low"]
    high_close = (d["High"] - d["Close"].shift()).abs()
    low_close  = (d["Low"]  - d["Close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    d["ATR_14"] = true_range.rolling(14).mean()

    # ── SECTION 7: MACRO FEATURE ENGINEERING ─────────────
    # Transform raw macro data into signals the model can use

    if "Fed_Funds_Rate" in d.columns:
        # Rate of change in interest rates (is the Fed raising or cutting?)
        d["Rate_Change_20d"] = d["Fed_Funds_Rate"].diff(20)
        # Rate regime: is the rate environment high or low historically?
        d["Rate_Level"] = d["Fed_Funds_Rate"].rolling(252).rank(pct=True)

    if "Inflation_CPI" in d.columns:
        # Month-over-month inflation change
        d["Inflation_Change"] = d["Inflation_CPI"].pct_change(20)

    if "VIX_Fear_Index" in d.columns:
        # Fear regime: is market fear elevated?
        d["VIX_MA_20"]   = d["VIX_Fear_Index"].rolling(20).mean()
        d["VIX_Spike"]   = d["VIX_Fear_Index"] / (d["VIX_MA_20"] + 1e-10)

    if "Yield_Curve" in d.columns:
        # Inverted yield curve = recession warning (very powerful signal)
        d["Yield_Inverted"] = (d["Yield_Curve"] < 0).astype(int)

    # ── SECTION 8: THE TARGET VARIABLE ───────────────────
    # This is what the model is trying to PREDICT
    # "Given everything above, will this stock be higher in N days?"
    #
    # We use FUTURE returns as the label:
    # shift(-20) looks 20 days into the future
    # The model never sees this during training on past data —
    # it only uses it to learn "what happened after these conditions"

    # Forward return: how much will price change over next 20 trading days?
    d["Target_Return_20d"] = d["Close"].pct_change(20).shift(-20)

    # Binary label: did it go UP or DOWN?
    # 1 = stock went up (buy signal), 0 = stock went down (avoid)
    d["Target_Up"] = (d["Target_Return_20d"] > 0).astype(int)

    return d


# ─────────────────────────────────────────────────────────
#  RUN FEATURE ENGINEERING ON EACH STOCK SEPARATELY
# ─────────────────────────────────────────────────────────

print("⚙️  Engineering features for each stock...\n")

all_featured = []

for ticker in df["Ticker"].unique():
    print(f"  Processing {ticker}...")

    # Filter to just this stock's rows
    stock_df = df[df["Ticker"] == ticker].copy()

    # Apply all features
    featured = engineer_features(stock_df)
    all_featured.append(featured)

    print(f"  ✅ {ticker}: {featured.shape[1]} features, {len(featured)} rows")

# Stack all stocks back together
features_df = pd.concat(all_featured).sort_index()

# ─────────────────────────────────────────────────────────
#  CLEAN UP
#  Drop rows with NaN values caused by rolling windows
#  (first ~200 rows per stock won't have MA_200 yet etc.)
# ─────────────────────────────────────────────────────────

rows_before = len(features_df)
features_df = features_df.dropna()
rows_after  = len(features_df)
dropped     = rows_before - rows_after

print(f"\n🧹 Dropped {dropped} rows with NaN (from rolling window warmup)")
print(f"   Final dataset: {rows_after} rows × {features_df.shape[1]} columns")

# ─────────────────────────────────────────────────────────
#  SAVE THE FEATURES DATASET
# ─────────────────────────────────────────────────────────

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
features_df.to_csv(OUTPUT_PATH)

print(f"\n✅ Features dataset saved → {OUTPUT_PATH}")

# ─────────────────────────────────────────────────────────
#  QUICK SANITY CHECK
#  Print a summary so you can verify everything looks right
# ─────────────────────────────────────────────────────────

print("\n" + "─" * 55)
print("  FEATURE SUMMARY")
print("─" * 55)

feature_groups = {
    "Price & Returns":    ["Return_5d", "Return_20d", "Return_60d"],
    "Moving Averages":    ["MA_20", "MA_50", "Price_vs_MA50"],
    "Momentum (RSI)":     ["RSI_14", "RSI_7"],
    "Momentum (MACD)":    ["MACD", "MACD_Signal", "MACD_Hist"],
    "Bollinger Bands":    ["BB_Pct", "BB_Width"],
    "Volume":             ["Volume_Ratio", "OBV"],
    "Volatility":         ["Volatility_20d", "ATR_14", "Vol_Regime"],
    "Macro":              ["Rate_Change_20d", "VIX_Spike", "Yield_Inverted"],
    "Target (label)":     ["Target_Return_20d", "Target_Up"],
}

for group, cols in feature_groups.items():
    existing = [c for c in cols if c in features_df.columns]
    if existing:
        print(f"\n  {group}:")
        print(features_df[existing].describe().round(4).to_string())

print("\n" + "=" * 55)
print("  Phase 2 complete! Ready for ML model building.")
print("=" * 55)