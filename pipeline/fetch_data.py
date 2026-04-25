# ============================================================
#  fetch_data.py  —  Stock AI Project: Data Pipeline (v2)
#  EXPANDED TO S&P 500
#  Pulls all 500 S&P stocks + macro data and saves to CSV
# ============================================================
#
#  HOW TO RUN:
#  python pipeline/fetch_data.py
#
#  ⚠️  This will take 15-25 minutes to download all 500 stocks
#  Leave it running — it saves progress as it goes so if it
#  stops you can resume without restarting from scratch
# ============================================================

import os
import time
import pandas as pd
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv

# Load your secret API key from the .env file
load_dotenv()
FRED_API_KEY = os.getenv("FRED_API_KEY")

# ─────────────────────────────────────────────────────────
#  SETTINGS
# ─────────────────────────────────────────────────────────

START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"
OUTPUT_DIR = "data/raw"

# How many seconds to wait between downloads
# This prevents Yahoo Finance from blocking you for too many requests
DELAY_BETWEEN_DOWNLOADS = 0.5


# ─────────────────────────────────────────────────────────
#  STEP 0: GET ALL S&P 500 TICKERS AUTOMATICALLY
#  Wikipedia maintains an up-to-date list of S&P 500 companies
#  We scrape it so we never have to manually update the list
# ─────────────────────────────────────────────────────────

def get_sp500_tickers():
    """
    Reads S&P 500 tickers from the local sp500_tickers.txt file.
    """
    print("📋 Reading S&P 500 tickers from sp500_tickers.txt...")

    with open("sp500_tickers.txt", "r") as f:
        tickers = [line.strip() for line in f.readlines() if line.strip()]

    print(f"   ✅ Loaded {len(tickers)} tickers\n")
    return tickers


# ─────────────────────────────────────────────────────────
#  PART 1: PULL STOCK PRICE DATA
#  Downloads each stock one at a time with smart resuming
#  so you don't lose progress if something interrupts
# ─────────────────────────────────────────────────────────

def fetch_stock_data(tickers, start, end, output_dir):
    """
    Downloads daily price history for all S&P 500 tickers.
    Skips tickers that already have a saved CSV file so
    you can resume if the download gets interrupted.
    """
    print("📈 Fetching stock price data for all S&P 500 stocks...")
    print(f"   This will take approximately 15-25 minutes\n")

    os.makedirs(output_dir, exist_ok=True)

    success_count  = 0
    skip_count     = 0
    fail_count     = 0
    failed_tickers = []

    for i, ticker in enumerate(tickers):
        filepath = os.path.join(output_dir, f"{ticker}.csv")

        # RESUME FEATURE: skip if we already downloaded this one
        if os.path.exists(filepath):
            skip_count += 1
            continue

        # Progress indicator every 10 stocks
        if (i + 1) % 10 == 0 or i == 0:
            pct = ((i + 1) / len(tickers)) * 100
            print(f"   Progress: {i+1}/{len(tickers)} ({pct:.0f}%) | "
                  f"✅ {success_count} saved | ❌ {fail_count} failed")

        try:
            stock = yf.download(
                ticker,
                start=start,
                end=end,
                auto_adjust=True,
                progress=False
            )

            # Flatten multi-level columns if present
            if isinstance(stock.columns, pd.MultiIndex):
                stock.columns = stock.columns.get_level_values(0)

            # Skip if no data came back
            if stock.empty or len(stock) < 100:
                failed_tickers.append(ticker)
                fail_count += 1
                continue

            # Add ticker column and basic features
            stock["Ticker"]        = ticker
            stock["Daily_Return"]  = stock["Close"].pct_change()
            stock["MA_20"]         = stock["Close"].rolling(window=20).mean()
            stock["MA_50"]         = stock["Close"].rolling(window=50).mean()
            stock["Volatility_20"] = stock["Daily_Return"].rolling(window=20).std()

            stock.to_csv(filepath)
            success_count += 1

            # Small delay to avoid getting rate-limited by Yahoo Finance
            time.sleep(DELAY_BETWEEN_DOWNLOADS)

        except Exception as e:
            failed_tickers.append(ticker)
            fail_count += 1
            time.sleep(1)

    print(f"\n   ✅ Downloaded:  {success_count} stocks")
    print(f"   ⏭️  Skipped (already existed): {skip_count} stocks")
    print(f"   ❌ Failed:     {fail_count} stocks")

    if failed_tickers:
        print(f"   Failed tickers: {failed_tickers}")

    # Save list of successfully downloaded tickers for reference
    all_csvs = [f.replace(".csv", "") for f in os.listdir(output_dir)
                if f.endswith(".csv") and f != "macro_data.csv"]
    tickers_path = os.path.join(output_dir, "_downloaded_tickers.txt")
    with open(tickers_path, "w") as f:
        f.write("\n".join(sorted(all_csvs)))
    print(f"   📋 Ticker list saved → {tickers_path}\n")

    return all_csvs


# ─────────────────────────────────────────────────────────
#  PART 2: PULL MACRO ECONOMIC DATA FROM FRED
#  Same as before — this part is fast (under 30 seconds)
# ─────────────────────────────────────────────────────────

def fetch_macro_data(api_key, start, end, output_dir):
    """
    Downloads key macroeconomic indicators from FRED.
    Skips if already downloaded.
    """
    print("🌍 Fetching macroeconomic data from FRED...")

    macro_path = os.path.join(output_dir, "macro_data.csv")
    if os.path.exists(macro_path):
        print("   ⏭️  Macro data already exists, skipping download\n")
        return

    if not api_key:
        print("   ⚠️  No FRED API key found. Skipping macro data.")
        return

    fred = Fred(api_key=api_key)

    macro_series = {
        "DFF":     "Fed_Funds_Rate",
        "T10Y2Y":  "Yield_Curve",
        "CPIAUCSL":"Inflation_CPI",
        "UNRATE":  "Unemployment_Rate",
        "GDP":     "GDP_Growth",
        "VIXCLS":  "VIX_Fear_Index",
        "DTWEXBGS":"Dollar_Strength",
    }

    all_macro = pd.DataFrame()

    for series_id, friendly_name in macro_series.items():
        print(f"   Downloading {friendly_name}...")
        try:
            series = fred.get_series(
                series_id,
                observation_start=start,
                observation_end=end
            )
            df = series.to_frame(name=friendly_name)
            all_macro = pd.concat([all_macro, df], axis=1)
            print(f"   ✅ {friendly_name}: {len(df)} data points")
        except Exception as e:
            print(f"   ⚠️  Could not fetch {series_id}: {e}")

    if all_macro.empty:
        return

    all_macro = all_macro.resample("D").interpolate(method="time")
    all_macro = all_macro.loc[start:end]
    all_macro.to_csv(macro_path)

    print(f"\n   ✅ Macro data saved → {macro_path}")
    print(f"      Shape: {all_macro.shape[0]} rows × {all_macro.shape[1]} columns\n")


# ─────────────────────────────────────────────────────────
#  PART 3: COMBINE EVERYTHING INTO ONE MASTER DATASET
#  Processes in batches of 50 to keep memory usage low
# ─────────────────────────────────────────────────────────

def combine_data(downloaded_tickers, output_dir):
    """
    Merges all stock CSVs with macro data into one master file.
    Processes in batches of 50 stocks to keep memory usage low.
    """
    print("🔗 Combining stock + macro data into master dataset...")
    print("   Processing in batches of 50 stocks to manage memory...\n")

    macro_path = os.path.join(output_dir, "macro_data.csv")
    if not os.path.exists(macro_path):
        print("   ⚠️  No macro data found. Skipping.")
        return

    macro = pd.read_csv(macro_path, index_col=0, parse_dates=True)

    processed_dir = output_dir.replace("raw", "processed")
    os.makedirs(processed_dir, exist_ok=True)
    master_path = os.path.join(processed_dir, "master_dataset.csv")

    BATCH_SIZE = 50
    batches    = [downloaded_tickers[i:i+BATCH_SIZE]
                  for i in range(0, len(downloaded_tickers), BATCH_SIZE)]

    first_batch = True
    total_rows  = 0

    for batch_num, batch in enumerate(batches):
        print(f"   Batch {batch_num+1}/{len(batches)}: merging {len(batch)} stocks...")
        batch_frames = []

        for ticker in batch:
            stock_path = os.path.join(output_dir, f"{ticker}.csv")
            if not os.path.exists(stock_path):
                continue
            try:
                stock    = pd.read_csv(stock_path, index_col=0, parse_dates=True)
                combined = stock.merge(
                    macro, left_index=True, right_index=True, how="left"
                )
                batch_frames.append(combined)
            except Exception as e:
                print(f"     ⚠️  Error processing {ticker}: {e}")

        if not batch_frames:
            continue

        batch_df    = pd.concat(batch_frames)
        total_rows += len(batch_df)

        # Write header only on first batch, append the rest
        batch_df.to_csv(
            master_path,
            mode="w" if first_batch else "a",
            header=first_batch
        )
        first_batch = False
        print(f"   ✅ Batch {batch_num+1} done ({len(batch_df):,} rows)")

    print(f"\n✅ Master dataset complete → {master_path}")
    print(f"   Total: {len(downloaded_tickers)} stocks | {total_rows:,} rows\n")


# ─────────────────────────────────────────────────────────
#  RUN EVERYTHING
# ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Stock AI — S&P 500 Data Pipeline")
    print("=" * 55)

    # Get full S&P 500 ticker list from Wikipedia
    tickers = get_sp500_tickers()

    # Download all stock data (resumes automatically if interrupted)
    downloaded = fetch_stock_data(tickers, START_DATE, END_DATE, OUTPUT_DIR)

    # Download macro data (skips if already done)
    fetch_macro_data(FRED_API_KEY, START_DATE, END_DATE, OUTPUT_DIR)

    # Combine everything into one master dataset
    combine_data(downloaded, OUTPUT_DIR)

    print("=" * 55)
    print("  Pipeline complete! Check your data/ folder.")
    print(f"  Stocks downloaded: {len(downloaded)}")
    print("=" * 55)