# ============================================================
#  ensemble_bundle.py  —  Stock AI Project: Phase 4
#  Ensemble Model + Portfolio Bundle Engine
#
#  This script:
#  1. Loads both trained models (XGBoost + LSTM)
#  2. Combines their predictions into one ensemble score
#  3. Ranks all 500 stocks by that score
#  4. Groups top stocks into diversified bundles
#  5. Outputs your final stock recommendations
# ============================================================
#
#  HOW TO RUN:
#  pip install pyportfolioopt
#  python models/ensemble_bundle.py
#
#  INPUT:  models/xgboost_model.json
#          models/lstm_model.pth
#          data/processed/features_dataset.csv
#  OUTPUT: models/results/bundle_recommendations.csv
#          models/results/bundle_report.txt
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ─────────────────────────────────────────────────────────
#  SETTINGS
# ─────────────────────────────────────────────────────────

INPUT_PATH   = "data/processed/features_dataset.csv"
MODEL_DIR    = "models"
RESULTS_DIR  = "models/results"

# How many stocks per bundle
BUNDLE_SIZE = 10

# How many bundles to generate
NUM_BUNDLES = 3

# Only consider stocks where BOTH models agree with this confidence
CONFIDENCE_THRESHOLD = 0.55

# Sequence length must match what LSTM was trained on
SEQUENCE_LENGTH = 60

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLS = [
    "Return_5d", "Return_10d", "Return_20d", "Return_60d",
    "Price_vs_MA20", "Price_vs_MA50", "Price_vs_MA200",
    "MA_Cross_20_50", "MA_Cross_50_200",
    "RSI_14", "RSI_7",
    "MACD", "MACD_Signal", "MACD_Hist",
    "BB_Pct", "BB_Width",
    "Volume_Ratio",
    "Volatility_5d", "Volatility_20d", "Volatility_60d",
    "Vol_Regime", "ATR_14",
    "Fed_Funds_Rate", "Yield_Curve", "Inflation_CPI",
    "Unemployment_Rate", "VIX_Fear_Index",
    "Rate_Change_20d", "VIX_Spike", "Yield_Inverted",
]

TARGET_COL = "Target_Up"


# ─────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────

print("=" * 55)
print("  Stock AI — Ensemble + Bundle Engine")
print("=" * 55)

print(f"\n📂 Loading features dataset...")
df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
available_features = [c for c in FEATURE_COLS if c in df.columns]
print(f"   Loaded: {df.shape[0]:,} rows | {len(available_features)} features")

# Use the most recent data available for scoring
# We score stocks on their LATEST data to get current recommendations
latest_date = df.index.max()
print(f"   Latest date in dataset: {latest_date.date()}")

# Get the last 200 rows per stock (enough for sequences + context)
print("\n📊 Extracting latest data per stock...")
recent_df = df.groupby("Ticker").tail(200).copy()
print(f"   Stocks available: {recent_df['Ticker'].nunique()}")


# ─────────────────────────────────────────────────────────
#  NORMALIZE FEATURES
#  Use the same scaling approach as training
# ─────────────────────────────────────────────────────────

scaler = StandardScaler()

# Fit on all data (we're scoring, not training)
recent_df[available_features] = scaler.fit_transform(
    recent_df[available_features].fillna(0)
)


# ─────────────────────────────────────────────────────────
#  LOAD XGBOOST MODEL
# ─────────────────────────────────────────────────────────

print("\n🤖 Loading XGBoost model...")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model(os.path.join(MODEL_DIR, "xgboost_model.json"))
print("   ✅ XGBoost loaded")


# ─────────────────────────────────────────────────────────
#  LOAD LSTM MODEL
# ─────────────────────────────────────────────────────────

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.5):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0,
            batch_first = True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(DEVICE)
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])
        return self.fc(out).squeeze()


print("🧠 Loading LSTM model...")
lstm_model = StockLSTM(input_size=len(available_features), hidden_size=64).to(DEVICE)
lstm_model.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR, "lstm_model.pth"),
        map_location=DEVICE
    )
)
lstm_model.eval()
print("   ✅ LSTM loaded")


# ─────────────────────────────────────────────────────────
#  SCORE EVERY STOCK WITH BOTH MODELS
#
#  For each stock we:
#  1. Get XGBoost probability (using latest row of features)
#  2. Get LSTM probability (using last 60 days as sequence)
#  3. Combine into ensemble score
# ─────────────────────────────────────────────────────────

print(f"\n🔮 Scoring all stocks with ensemble model...")
print(f"   Device: {DEVICE}\n")

results = []
tickers = recent_df["Ticker"].unique()

for i, ticker in enumerate(tickers):
    if (i + 1) % 50 == 0:
        print(f"   Scoring ticker {i+1}/{len(tickers)}...")

    stock_data = recent_df[recent_df["Ticker"] == ticker].copy()

    if len(stock_data) < SEQUENCE_LENGTH + 5:
        continue

    try:
        # ── XGBOOST SCORE ───────────────────────────────
        # Use the most recent single row of features
        latest_row = stock_data[available_features].iloc[[-1]]
        xgb_prob = xgb_model.predict_proba(latest_row)[0][1]

        # ── LSTM SCORE ───────────────────────────────────
        # Use the last 60 days as a sequence
        sequence = stock_data[available_features].values[-SEQUENCE_LENGTH:]
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            lstm_prob = lstm_model(sequence_tensor).item()

        # ── ENSEMBLE SCORE ────────────────────────────────
        # Weighted average: slightly favor XGBoost since it
        # showed better calibration on this dataset
        # Adjust weights as you see fit
        ensemble_score = (0.55 * xgb_prob) + (0.45 * lstm_prob)

        # ── AGREEMENT BONUS ───────────────────────────────
        # When both models agree (both > 0.55), boost the score
        # This is the most reliable signal
        both_bullish = xgb_prob > CONFIDENCE_THRESHOLD and \
                       lstm_prob > CONFIDENCE_THRESHOLD
        agreement_bonus = 0.05 if both_bullish else 0.0
        final_score = min(ensemble_score + agreement_bonus, 1.0)

        # Get additional context for the report
        latest = stock_data.iloc[-1]

        results.append({
            "Ticker":         ticker,
            "Ensemble_Score": round(final_score, 4),
            "XGBoost_Prob":   round(xgb_prob, 4),
            "LSTM_Prob":      round(lstm_prob, 4),
            "Both_Bullish":   both_bullish,
            "RSI_14":         round(latest.get("RSI_14", 50), 1),
            "Return_20d":     round(latest.get("Return_20d", 0) * 100, 2),
            "Volatility_20d": round(latest.get("Volatility_20d", 0) * 100, 4),
        })

    except Exception as e:
        continue

# Convert to DataFrame and sort by ensemble score
scores_df = pd.DataFrame(results)
scores_df = scores_df.sort_values("Ensemble_Score", ascending=False)

print(f"\n   ✅ Scored {len(scores_df)} stocks")
print(f"   Both models bullish on: {scores_df['Both_Bullish'].sum()} stocks")

print(f"\n🏆 Top 20 stocks by ensemble score:")
print(scores_df[["Ticker", "Ensemble_Score", "XGBoost_Prob",
                  "LSTM_Prob", "Both_Bullish", "RSI_14",
                  "Return_20d"]].head(20).to_string(index=False))


# ─────────────────────────────────────────────────────────
#  BUILD PORTFOLIO BUNDLES
#
#  We don't just want the top 10 stocks — we want
#  DIVERSIFIED bundles that spread risk across sectors.
#
#  Strategy:
#  1. Take top 60 stocks by ensemble score
#  2. Use KMeans clustering to group them by behavior
#  3. Pick the best stock from each cluster
#  4. This ensures we don't end up with 10 tech stocks
# ─────────────────────────────────────────────────────────

print(f"\n📦 Building {NUM_BUNDLES} diversified bundles...")

# Work with top candidates only
top_candidates = scores_df[
    scores_df["Ensemble_Score"] > CONFIDENCE_THRESHOLD
].head(60).copy()

print(f"   High-confidence candidates: {len(top_candidates)}")

if len(top_candidates) < NUM_BUNDLES * BUNDLE_SIZE:
    print("   ⚠️  Not enough high-confidence stocks, lowering threshold...")
    top_candidates = scores_df.head(NUM_BUNDLES * BUNDLE_SIZE * 2).copy()

# Get return history for clustering
# Cluster stocks by their correlation pattern
print("   Building correlation matrix for diversification...")

cluster_data = []
cluster_tickers = []

for ticker in top_candidates["Ticker"]:
    stock_hist = df[df["Ticker"] == ticker]["Return_5d"].tail(120)
    if len(stock_hist) >= 100:
        cluster_data.append(stock_hist.values[-100:])
        cluster_tickers.append(ticker)

if len(cluster_data) >= NUM_BUNDLES * BUNDLE_SIZE:
    cluster_matrix = np.array(cluster_data)

    # KMeans clustering: group stocks with similar return patterns
    # Each cluster = stocks that tend to move together
    # We pick from different clusters to ensure diversification
    n_clusters = NUM_BUNDLES * BUNDLE_SIZE
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(cluster_matrix)

    top_candidates_clustered = top_candidates[
        top_candidates["Ticker"].isin(cluster_tickers)
    ].copy()
    top_candidates_clustered["Cluster"] = cluster_labels[
        :len(top_candidates_clustered)
    ]

    # From each cluster, pick the highest scored stock
    best_per_cluster = top_candidates_clustered.sort_values(
        "Ensemble_Score", ascending=False
    ).groupby("Cluster").first().reset_index()

    best_per_cluster = best_per_cluster.sort_values(
        "Ensemble_Score", ascending=False
    )

else:
    best_per_cluster = top_candidates.copy()

# Split into bundles
bundles = []
for b in range(NUM_BUNDLES):
    start = b * BUNDLE_SIZE
    end   = start + BUNDLE_SIZE
    bundle_stocks = best_per_cluster.iloc[start:end]

    if len(bundle_stocks) == 0:
        continue

    # Equal weight within each bundle for simplicity
    # (Phase 5 will add mean-variance optimization)
    weight = round(1.0 / len(bundle_stocks), 4)
    bundle_stocks = bundle_stocks.copy()
    bundle_stocks["Weight"]      = weight
    bundle_stocks["Bundle"]      = f"Bundle_{b+1}"
    bundle_stocks["Bundle_Rank"] = b + 1
    bundles.append(bundle_stocks)

if not bundles:
    print("   ⚠️  Could not build bundles. Check model outputs.")
else:
    all_bundles = pd.concat(bundles)

    # ─────────────────────────────────────────────────────
    #  PRINT THE FINAL RECOMMENDATIONS
    # ─────────────────────────────────────────────────────

    print("\n" + "=" * 55)
    print("  📊 STOCK BUNDLE RECOMMENDATIONS")
    print("=" * 55)

    report_lines = []
    report_lines.append("STOCK AI — BUNDLE RECOMMENDATIONS")
    report_lines.append(f"Generated from data through: {latest_date.date()}")
    report_lines.append("=" * 55)

    for b in range(1, NUM_BUNDLES + 1):
        bundle = all_bundles[all_bundles["Bundle_Rank"] == b]
        if bundle.empty:
            continue

        avg_score = bundle["Ensemble_Score"].mean()
        both_bullish_count = bundle["Both_Bullish"].sum()

        header = f"\n🏦 BUNDLE {b}  |  Avg Score: {avg_score:.3f}  |  Both models agree: {both_bullish_count}/{len(bundle)}"
        print(header)
        print(f"   {'Ticker':<8} {'Score':>7} {'XGB':>7} {'LSTM':>7} {'RSI':>6} {'20d Ret':>8} {'Weight':>8}")
        print(f"   {'─'*55}")

        report_lines.append(header)

        for _, row in bundle.iterrows():
            line = (f"   {row['Ticker']:<8} "
                    f"{row['Ensemble_Score']:>7.4f} "
                    f"{row['XGBoost_Prob']:>7.4f} "
                    f"{row['LSTM_Prob']:>7.4f} "
                    f"{row['RSI_14']:>6.1f} "
                    f"{row['Return_20d']:>7.1f}% "
                    f"{row['Weight']:>7.1%}")
            print(line)
            report_lines.append(line)

        print(f"\n   💡 Interpretation:")
        if avg_score > 0.65:
            interp = "Strong buy signal — both models highly confident"
        elif avg_score > 0.58:
            interp = "Moderate buy signal — good ensemble agreement"
        else:
            interp = "Speculative — monitor closely before acting"
        print(f"   {interp}")

    print("\n" + "=" * 55)
    print("  ⚠️  DISCLAIMER")
    print("  These are model predictions, not financial advice.")
    print("  Always do your own research before investing.")
    print("=" * 55)

    # ─────────────────────────────────────────────────────
    #  SAVE OUTPUTS
    # ─────────────────────────────────────────────────────

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Save full scores for all stocks
    scores_path = os.path.join(RESULTS_DIR, "all_stock_scores.csv")
    scores_df.to_csv(scores_path, index=False)
    print(f"\n💾 All stock scores saved → {scores_path}")

    # Save bundle recommendations
    bundles_path = os.path.join(RESULTS_DIR, "bundle_recommendations.csv")
    all_bundles.to_csv(bundles_path, index=False)
    print(f"💾 Bundle recommendations saved → {bundles_path}")

    # Save text report
    report_path = os.path.join(RESULTS_DIR, "bundle_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"💾 Bundle report saved → {report_path}")

print("\n" + "=" * 55)
print("  Phase 4 complete!")
print("  Next: Backtesting to validate performance")
print("=" * 55)