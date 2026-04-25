# ============================================================
#  fetch_data.py  —  Stock AI Project: Data Pipeline
#  Pulls stock prices + macro economic data and saves to CSV
# ============================================================

import os
import pandas as pd
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv

# Load your secret API key from the .env file
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

# ─────────────────────────────────────────────────────────
#  SETTINGS  —  change these any time
# ─────────────────────────────────────────────────────────

# The stocks you want to track
# Add or remove any ticker symbols you want here
STOCKS = [
    "AAPL",  # Apple
    "MSFT",  # Microsoft
    "GOOGL", # Google
    "AMZN",  # Amazon
    "NVDA",  # Nvidia
    "JPM",   # JPMorgan (bank)
    "XOM",   # ExxonMobil (energy)
    "JNJ",   # Johnson & Johnson (healthcare)
    "SPY",   # S&P 500 ETF (market benchmark)
]

# How far back to pull data
START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"

# Where to save the data
OUTPUT_DIR = "data/raw"


# ─────────────────────────────────────────────────────────
#  PART 1: PULL STOCK PRICE DATA
#  For each stock we grab: Open, High, Low, Close, Volume
#  "Close" is the price at end of each trading day
# ─────────────────────────────────────────────────────────

def fetch_stock_data(tickers, start, end, output_dir):
    """
    Downloads daily price history for a list of stock tickers
    and saves each one as its own CSV file.
    """
    print("\n📈 Fetching stock price data...")

    # Make sure the output folder exists
    os.makedirs(output_dir, exist_ok=True)

    for ticker in tickers:
        print(f"  Downloading {ticker}...")

        # yfinance does all the heavy lifting here
        # It pulls data straight from Yahoo Finance for free
        stock = yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,   # adjusts prices for splits/dividends automatically
            progress=False      # silences the download progress bar
        )

        # yfinance sometimes returns multi-level columns — flatten them
        if isinstance(stock.columns, pd.MultiIndex):
            stock.columns = stock.columns.get_level_values(0)

        if stock.empty:
            print(f"  ⚠️  No data found for {ticker}, skipping.")
            continue

        # Add the ticker name as a column so we know which stock it is later
        stock["Ticker"] = ticker

        # Also calculate a few simple features while we're here:

        # Daily return: how much % did the stock move each day?
        # e.g. +0.02 means it went up 2% that day
        stock["Daily_Return"] = stock["Close"].pct_change()

        # 20-day moving average: smooths out noise, shows the trend
        stock["MA_20"] = stock["Close"].rolling(window=20).mean()

        # 50-day moving average: longer-term trend
        stock["MA_50"] = stock["Close"].rolling(window=50).mean()

        # Volatility: how wildly does the stock swing?
        # Measured as the standard deviation of daily returns over 20 days
        stock["Volatility_20"] = stock["Daily_Return"].rolling(window=20).std()

        # Save to CSV
        filepath = os.path.join(output_dir, f"{ticker}.csv")
        stock.to_csv(filepath)
        print(f"  ✅ Saved {len(stock)} rows → {filepath}")

    print("Stock data download complete!\n")


# ─────────────────────────────────────────────────────────
#  PART 2: PULL MACRO ECONOMIC DATA FROM FRED
#  These are the "big picture" economic signals
#  that affect the entire market
# ─────────────────────────────────────────────────────────

def fetch_macro_data(api_key, start, end, output_dir):
    """
    Downloads key macroeconomic indicators from the
    Federal Reserve's free FRED database.
    """
    print("🌍 Fetching macroeconomic data from FRED...")

    if not api_key:
        print("  ⚠️  No FRED API key found. Skipping macro data.")
        print("  Get a free key at: https://fred.stlouisfed.org/docs/api/api_key.html")
        return

    fred = Fred(api_key=api_key)

    # Each entry is: "series_id": "friendly name"
    # FRED series IDs are their internal codes — these are the most useful ones
    macro_series = {
        "DFF":     "Fed_Funds_Rate",       # Interest rate the Fed charges banks (huge market driver)
        "T10Y2Y":  "Yield_Curve",          # 10yr minus 2yr treasury — negative = recession warning
        "CPIAUCSL":"Inflation_CPI",        # Consumer Price Index — measures inflation
        "UNRATE":  "Unemployment_Rate",    # % of people without jobs
        "GDP":     "GDP_Growth",           # Total economic output (quarterly)
        "VIXCLS":  "VIX_Fear_Index",       # Market fear/volatility index — spikes during crashes
        "DTWEXBGS":"Dollar_Strength",      # US dollar strength vs other currencies
    }

    all_macro = pd.DataFrame()

    for series_id, friendly_name in macro_series.items():
        print(f"  Downloading {friendly_name} ({series_id})...")

        try:
            series = fred.get_series(
                series_id,
                observation_start=start,
                observation_end=end
            )

            # Convert to a nice DataFrame column
            df = series.to_frame(name=friendly_name)
            all_macro = pd.concat([all_macro, df], axis=1)
            print(f"  ✅ Got {len(df)} data points")

        except Exception as e:
            print(f"  ⚠️  Could not fetch {series_id}: {e}")

    if all_macro.empty:
        print("  No macro data was downloaded.")
        return

    # FRED data comes in different frequencies (daily, monthly, quarterly)
    # We forward-fill so every calendar day has a value
    # e.g. GDP is only reported quarterly, so we repeat that value for each day
    # until the next quarterly update comes in
    all_macro = all_macro.resample("D").interpolate(method="time")

    # Trim to our date range
    all_macro = all_macro.loc[start:end]

    # Save all macro data in one file
    filepath = os.path.join(output_dir, "macro_data.csv")
    all_macro.to_csv(filepath)
    print(f"\n✅ Macro data saved → {filepath}")
    print(f"   Shape: {all_macro.shape[0]} rows × {all_macro.shape[1]} columns\n")


# ─────────────────────────────────────────────────────────
#  PART 3: COMBINE EVERYTHING INTO ONE MASTER DATASET
#  Merge each stock's data with the macro data
#  so every row has both stock info + economic context
# ─────────────────────────────────────────────────────────

def combine_data(tickers, output_dir):
    """
    Merges each stock's price data with the macro data
    and saves a single combined master CSV.
    """
    print("🔗 Combining stock + macro data...")

    # Load macro data
    macro_path = os.path.join(output_dir, "macro_data.csv")
    if not os.path.exists(macro_path):
        print("  ⚠️  No macro data found. Skipping combination step.")
        return

    macro = pd.read_csv(macro_path, index_col=0, parse_dates=True)

    all_stocks = []

    for ticker in tickers:
        stock_path = os.path.join(output_dir, f"{ticker}.csv")
        if not os.path.exists(stock_path):
            continue

        stock = pd.read_csv(stock_path, index_col=0, parse_dates=True)

        # Merge: match each stock's date with the macro data on that same date
        # "left" join means we keep all stock dates even if macro data is missing
        combined = stock.merge(macro, left_index=True, right_index=True, how="left")

        all_stocks.append(combined)

    if not all_stocks:
        print("  No stock data found to combine.")
        return

    # Stack all stocks into one big dataset
    master = pd.concat(all_stocks)

    # Save the master file
    processed_dir = output_dir.replace("raw", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    filepath = os.path.join(processed_dir, "master_dataset.csv")
    master.to_csv(filepath)

    print(f"✅ Master dataset saved → {filepath}")
    print(f"   {len(tickers)} stocks × {len(master)} total rows")
    print(f"   Columns: {list(master.columns)}\n")


# ─────────────────────────────────────────────────────────
#  RUN EVERYTHING
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Stock AI — Data Pipeline Starting")
    print("=" * 55)

    # Step 1: Download all stock price data
    fetch_stock_data(STOCKS, START_DATE, END_DATE, OUTPUT_DIR)

    # Step 2: Download macro economic data
    fetch_macro_data(FRED_API_KEY, START_DATE, END_DATE, OUTPUT_DIR)

    # Step 3: Merge everything into one master dataset
    combine_data(STOCKS, OUTPUT_DIR)

    print("=" * 55)
    print("  Pipeline complete! Check your data/ folder.")
    print("=" * 55)