# ============================================================
#  train_model.py  —  Stock AI Project: Phase 3
#  XGBoost Baseline Model
#  Trains on engineered features to predict if a stock
#  will go UP or DOWN over the next 20 trading days
# ============================================================
#
#  HOW TO RUN:
#  pip install xgboost scikit-learn matplotlib
#  python models/train_model.py
#
#  INPUT:  data/processed/features_dataset.csv  (from Phase 2)
#  OUTPUT: models/xgboost_model.json            (saved model)
#          models/results/                       (charts + report)
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, confusion_matrix
)
from sklearn.preprocessing import StandardScaler

# ─────────────────────────────────────────────────────────
#  SETTINGS
# ─────────────────────────────────────────────────────────

INPUT_PATH   = "data/processed/features_dataset.csv"
MODEL_DIR    = "models"
RESULTS_DIR  = "models/results"

# These are the features we feed into the model
# Every column we engineered in Phase 2 becomes an input signal
FEATURE_COLS = [
    # Price momentum
    "Return_5d", "Return_10d", "Return_20d", "Return_60d",
    # Moving average signals
    "Price_vs_MA20", "Price_vs_MA50", "Price_vs_MA200",
    "MA_Cross_20_50", "MA_Cross_50_200",
    # RSI
    "RSI_14", "RSI_7",
    # MACD
    "MACD", "MACD_Signal", "MACD_Hist",
    # Bollinger Bands
    "BB_Pct", "BB_Width",
    # Volume
    "Volume_Ratio",
    # Volatility
    "Volatility_5d", "Volatility_20d", "Volatility_60d",
    "Vol_Regime", "ATR_14",
    # Macro
    "Fed_Funds_Rate", "Yield_Curve", "Inflation_CPI",
    "Unemployment_Rate", "VIX_Fear_Index",
    "Rate_Change_20d", "VIX_Spike", "Yield_Inverted",
]

# What the model is trying to predict
# 1 = stock went up after 20 days, 0 = stock went down
TARGET_COL = "Target_Up"


# ─────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────

print("=" * 55)
print("  Stock AI — XGBoost Model Training")
print("=" * 55)

print(f"\n📂 Loading features from {INPUT_PATH}...")
df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
print(f"   Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# Only keep feature columns that actually exist in our dataset
# (some macro columns may be missing depending on FRED availability)
available_features = [c for c in FEATURE_COLS if c in df.columns]
missing_features   = [c for c in FEATURE_COLS if c not in df.columns]

if missing_features:
    print(f"   ⚠️  Missing features (will skip): {missing_features}")

print(f"   Using {len(available_features)} features for training\n")

# ─────────────────────────────────────────────────────────
#  WALK-FORWARD TRAIN / TEST SPLIT
#
#  This is the most important concept in financial ML.
#  We CANNOT randomly shuffle the data like in normal ML.
#  Why? Because that would let the model "cheat" by seeing
#  future data during training (called "look-ahead bias").
#
#  Instead we split by TIME:
#  Train → 2015 to 2022  (the past the model learns from)
#  Test  → 2023 to 2024  (the future it has never seen)
#
#  This mimics how it works in real life — you train on
#  history and predict the future.
# ─────────────────────────────────────────────────────────

TRAIN_END = "2022-12-31"
TEST_START = "2023-01-01"

print("✂️  Splitting data by time (no look-ahead bias)...")

train_df = df[df.index <= TRAIN_END]
test_df  = df[df.index >= TEST_START]

X_train = train_df[available_features]
y_train = train_df[TARGET_COL]

X_test  = test_df[available_features]
y_test  = test_df[TARGET_COL]

print(f"   Train: {len(X_train):,} rows  ({train_df.index.min().date()} → {train_df.index.max().date()})")
print(f"   Test:  {len(X_test):,} rows  ({test_df.index.min().date()} → {test_df.index.max().date()})")
print(f"   Train positive rate: {y_train.mean():.1%} (stocks that went up)")
print(f"   Test positive rate:  {y_test.mean():.1%} (stocks that went up)\n")


# ─────────────────────────────────────────────────────────
#  TRAIN THE XGBOOST MODEL
#
#  XGBoost = Extreme Gradient Boosting
#  It builds hundreds of small decision trees, each one
#  learning from the mistakes of the previous one.
#  Together they form a very powerful prediction engine.
#
#  Think of it like 300 analysts each looking at the data
#  and voting — the majority vote wins.
# ─────────────────────────────────────────────────────────

print("🤖 Training XGBoost model...")

model = xgb.XGBClassifier(
    n_estimators      = 300,    # number of trees to build
    max_depth         = 5,      # how deep each tree can go (deeper = more complex)
    learning_rate     = 0.05,   # how much each tree corrects the previous one
    subsample         = 0.8,    # use 80% of rows per tree (prevents overfitting)
    colsample_bytree  = 0.8,    # use 80% of features per tree (prevents overfitting)
    min_child_weight  = 5,      # minimum samples needed to split a node
    gamma             = 0.1,    # minimum gain needed to make a split
    use_label_encoder = False,
    eval_metric       = "logloss",
    random_state      = 42,
    verbosity         = 0,
)

# Train — this is where the magic happens
# The model reads every row of training data and adjusts
# its 300 trees to minimize prediction errors
model.fit(
    X_train, y_train,
    eval_set              = [(X_test, y_test)],
    verbose               = False,
)

print("   ✅ Training complete!\n")


# ─────────────────────────────────────────────────────────
#  EVALUATE THE MODEL
#  How well does it predict on data it has NEVER seen?
# ─────────────────────────────────────────────────────────

print("📊 Evaluating model performance on test data...")

# Get predictions
# predict()      → hard label: 1 (up) or 0 (down)
# predict_proba()→ confidence score: 0.0 to 1.0
y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # probability of going UP

# Core metrics
accuracy = accuracy_score(y_test, y_pred)
auc      = roc_auc_score(y_test, y_pred_proba)

print(f"\n   Accuracy : {accuracy:.1%}  (baseline to beat: {y_test.mean():.1%})")
print(f"   AUC Score: {auc:.3f}      (0.5 = random, 1.0 = perfect)")
print(f"\n   Detailed breakdown:")
print(classification_report(y_test, y_pred, target_names=["DOWN", "UP"]))


# ─────────────────────────────────────────────────────────
#  FEATURE IMPORTANCE
#  Which signals did the model find most useful?
#  This is one of the most valuable outputs — it tells you
#  what actually drives stock performance in your data
# ─────────────────────────────────────────────────────────

importance_df = pd.DataFrame({
    "Feature":    available_features,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n🔍 Top 15 most important features:")
print(importance_df.head(15).to_string(index=False))


# ─────────────────────────────────────────────────────────
#  SIMULATE TRADING PERFORMANCE
#  What if we only bought stocks when the model was
#  confident (probability > 60%)? How would that do?
# ─────────────────────────────────────────────────────────

print("\n💰 Simulating model-guided trading strategy...")

test_results = test_df[["Ticker", "Target_Return_20d"]].copy()
test_results["Predicted_Prob"] = y_pred_proba
test_results["Predicted_Up"]   = y_pred

# Strategy: only trade when model confidence > 60%
CONFIDENCE_THRESHOLD = 0.60
high_conf = test_results[test_results["Predicted_Prob"] > CONFIDENCE_THRESHOLD]

if len(high_conf) > 0:
    strategy_return = high_conf["Target_Return_20d"].mean()
    buyhold_return  = test_results["Target_Return_20d"].mean()
    win_rate        = (high_conf["Target_Return_20d"] > 0).mean()

    print(f"   High-confidence trades: {len(high_conf):,} ({len(high_conf)/len(test_results):.1%} of all signals)")
    print(f"   Win rate on those trades: {win_rate:.1%}")
    print(f"   Avg return per trade (model): {strategy_return:.2%}")
    print(f"   Avg return per trade (buy & hold): {buyhold_return:.2%}")
    lift = strategy_return - buyhold_return
    print(f"   Model lift over buy & hold: {lift:+.2%}")


# ─────────────────────────────────────────────────────────
#  SAVE RESULTS AND CHARTS
# ─────────────────────────────────────────────────────────

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Save the trained model
model_path = os.path.join(MODEL_DIR, "xgboost_model.json")
model.save_model(model_path)
print(f"\n💾 Model saved → {model_path}")

# Chart 1: Feature Importance
plt.figure(figsize=(10, 7))
top15 = importance_df.head(15)
plt.barh(top15["Feature"][::-1], top15["Importance"][::-1], color="#4F86C6")
plt.xlabel("Importance Score")
plt.title("XGBoost — Top 15 Most Predictive Features")
plt.tight_layout()
chart1_path = os.path.join(RESULTS_DIR, "feature_importance.png")
plt.savefig(chart1_path, dpi=120)
plt.close()
print(f"📈 Feature importance chart saved → {chart1_path}")

# Chart 2: Prediction confidence distribution
plt.figure(figsize=(10, 5))
plt.hist(y_pred_proba[y_test == 1], bins=40, alpha=0.6,
         color="green", label="Actually went UP")
plt.hist(y_pred_proba[y_test == 0], bins=40, alpha=0.6,
         color="red",   label="Actually went DOWN")
plt.axvline(x=0.5, color="black", linestyle="--", label="Decision boundary (0.5)")
plt.axvline(x=CONFIDENCE_THRESHOLD, color="orange",
            linestyle="--", label=f"High confidence ({CONFIDENCE_THRESHOLD})")
plt.xlabel("Model Confidence (Probability of Going UP)")
plt.ylabel("Number of Predictions")
plt.title("XGBoost — Prediction Confidence Distribution")
plt.legend()
plt.tight_layout()
chart2_path = os.path.join(RESULTS_DIR, "confidence_distribution.png")
plt.savefig(chart2_path, dpi=120)
plt.close()
print(f"📈 Confidence distribution chart saved → {chart2_path}")

# Save summary report as JSON
summary = {
    "accuracy":           round(accuracy, 4),
    "auc_score":          round(auc, 4),
    "baseline":           round(float(y_test.mean()), 4),
    "train_rows":         len(X_train),
    "test_rows":          len(X_test),
    "features_used":      len(available_features),
    "top_5_features":     importance_df["Feature"].head(5).tolist(),
    "high_conf_trades":   len(high_conf) if len(high_conf) > 0 else 0,
    "high_conf_win_rate": round(float(win_rate), 4) if len(high_conf) > 0 else 0,
}
report_path = os.path.join(RESULTS_DIR, "model_summary.json")
with open(report_path, "w") as f:
    json.dump(summary, f, indent=2)
print(f"📋 Model summary saved → {report_path}")

print("\n" + "=" * 55)
print("  Phase 3 (Baseline) complete!")
print(f"  Accuracy: {accuracy:.1%}  |  AUC: {auc:.3f}")
print("  Next: LSTM model for time-series patterns")
print("=" * 55)