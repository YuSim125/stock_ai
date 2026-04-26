# ============================================================
#  lstm_model.py  —  Stock AI Project: Phase 3B
#  LSTM Deep Learning Model
#
#  XGBoost treated every day as independent.
#  LSTM understands that prices move in SEQUENCES over time.
#  It looks at the last 60 days together as a pattern,
#  just like a human analyst would review a chart.
# ============================================================
#
#  HOW TO RUN:
#  pip install torch scikit-learn pandas numpy matplotlib
#  python models/lstm_model.py
#
#  INPUT:  data/processed/features_dataset.csv
#  OUTPUT: models/lstm_model.pth
#          models/results/lstm_results.png
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score

# ─────────────────────────────────────────────────────────
#  SETTINGS
# ─────────────────────────────────────────────────────────

INPUT_PATH  = "data/processed/features_dataset.csv"
MODEL_DIR   = "models"
RESULTS_DIR = "models/results"

# Sequence length: how many past days the LSTM looks at
# 60 days = ~3 months of trading history per prediction
SEQUENCE_LENGTH = 30

# Training settings
BATCH_SIZE  = 2048    # how many sequences to process at once
EPOCHS      = 50   # how many times to train through the data
LEARNING_RATE = 0.0003

# LSTM architecture
HIDDEN_SIZE  = 64  # number of memory cells in the LSTM
NUM_LAYERS   = 2    # stack 2 LSTM layers for more depth
DROPOUT      = 0.5   # randomly disable 30% of neurons to prevent overfitting

# Time split (same as XGBoost for fair comparison)
TRAIN_END  = "2022-12-31"
TEST_START = "2023-01-01"

# Features to feed into the LSTM
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

# Use GPU if available, otherwise CPU
# GPU makes training much faster but CPU works fine too
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🖥️  Using device: {DEVICE}")
if DEVICE.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")


# ─────────────────────────────────────────────────────────
#  LOAD AND PREPARE DATA
# ─────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("  Stock AI — LSTM Model Training")
print("=" * 55)

print(f"\n📂 Loading features dataset...")
df = pd.read_csv(INPUT_PATH, index_col=0, parse_dates=True)
print(f"   Loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")

# Keep only columns that exist
available_features = [c for c in FEATURE_COLS if c in df.columns]
print(f"   Using {len(available_features)} features\n")


# ─────────────────────────────────────────────────────────
#  NORMALIZE THE FEATURES
#
#  Neural networks learn much better when all input values
#  are on a similar scale (roughly -3 to +3).
#  Without this, a feature like "OBV" (millions) would
#  completely overpower "RSI_14" (0 to 100).
#
#  StandardScaler converts each feature to:
#  mean = 0, standard deviation = 1
# ─────────────────────────────────────────────────────────

print("📐 Normalizing features...")

train_df = df[df.index <= TRAIN_END].copy()
test_df  = df[df.index >= TEST_START].copy()

scaler = StandardScaler()

# IMPORTANT: fit the scaler ONLY on training data
# Never let test data influence the scaling — that's cheating
train_df[available_features] = scaler.fit_transform(
    train_df[available_features]
)
test_df[available_features] = scaler.transform(
    test_df[available_features]
)

print(f"   Train: {len(train_df):,} rows")
print(f"   Test:  {len(test_df):,} rows\n")


# ─────────────────────────────────────────────────────────
#  BUILD SEQUENCES
#
#  This is the key difference from XGBoost.
#  Instead of feeding the model one day at a time,
#  we feed it a WINDOW of 60 consecutive days.
#
#  Each training sample looks like:
#  Input:  [day1_features, day2_features, ... day60_features]
#  Output: did the stock go UP on day 80? (1 or 0)
#
#  The LSTM reads through all 60 days in order and builds
#  up a memory of the pattern before making its prediction.
# ─────────────────────────────────────────────────────────

def build_sequences(stock_df, feature_cols, target_col, seq_len):
    """
    Converts a stock's daily data into overlapping sequences.
    Each sequence is 60 days of features + 1 label.
    """
    X, y = [], []

    features = stock_df[feature_cols].values
    targets  = stock_df[target_col].values

    # Slide a window of size seq_len across the data
    for i in range(seq_len, len(stock_df)):
        # Input: the 60 days BEFORE day i
        X.append(features[i - seq_len : i])
        # Label: what happened on day i
        y.append(targets[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


print("🔄 Building sequences (this takes a few minutes for 500 stocks)...")

# Build sequences separately for each stock
# This prevents sequences from bleeding across different companies
train_X_list, train_y_list = [], []
test_X_list,  test_y_list  = [], []

tickers = df["Ticker"].unique()
for i, ticker in enumerate(tickers):
    if (i + 1) % 50 == 0:
        print(f"   Processing ticker {i+1}/{len(tickers)}...")

    # Training sequences
    t_train = train_df[train_df["Ticker"] == ticker]
    if len(t_train) > SEQUENCE_LENGTH + 10:
        X_t, y_t = build_sequences(t_train, available_features,
                                   TARGET_COL, SEQUENCE_LENGTH)
        train_X_list.append(X_t)
        train_y_list.append(y_t)

    # Test sequences
    t_test = test_df[test_df["Ticker"] == ticker]
    if len(t_test) > SEQUENCE_LENGTH + 10:
        X_e, y_e = build_sequences(t_test, available_features,
                                   TARGET_COL, SEQUENCE_LENGTH)
        test_X_list.append(X_e)
        test_y_list.append(y_e)

# Stack all stocks together
X_train = np.concatenate(train_X_list, axis=0)
y_train = np.concatenate(train_y_list, axis=0)
X_test  = np.concatenate(test_X_list,  axis=0)
y_test  = np.concatenate(test_y_list,  axis=0)

print(f"\n   Train sequences: {X_train.shape[0]:,}")
print(f"   Test sequences:  {X_test.shape[0]:,}")
print(f"   Sequence shape:  {X_train.shape[1]} days × {X_train.shape[2]} features\n")


# ─────────────────────────────────────────────────────────
#  PYTORCH DATASET AND DATALOADER
#
#  PyTorch needs data wrapped in a Dataset class.
#  The DataLoader handles batching and shuffling automatically.
# ─────────────────────────────────────────────────────────

class StockSequenceDataset(Dataset):
    """Wraps numpy arrays into a PyTorch Dataset."""
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


train_dataset = StockSequenceDataset(X_train, y_train)
test_dataset  = StockSequenceDataset(X_test,  y_test)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,       # shuffle training data each epoch
    num_workers=0
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"   Batches per epoch: {len(train_loader):,}\n")


# ─────────────────────────────────────────────────────────
#  DEFINE THE LSTM MODEL
#
#  Architecture:
#  Input (60 days × 30 features)
#    → LSTM Layer 1 (128 hidden units) — learns short patterns
#    → LSTM Layer 2 (128 hidden units) — learns longer patterns
#    → Dropout (prevents overfitting)
#    → Fully Connected Layer (128 → 1)
#    → Sigmoid (converts to 0-1 probability)
# ─────────────────────────────────────────────────────────

class StockLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(StockLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # The core LSTM — reads sequences and builds memory
        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            dropout     = dropout if num_layers > 1 else 0,
            batch_first = True      # input shape: (batch, sequence, features)
        )

        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(dropout)

        # Final classifier: converts LSTM output to a single probability
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()            # output between 0 and 1
        )

    def forward(self, x):
        # Initialize hidden state and cell state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_size).to(DEVICE)

        # Pass sequence through LSTM
        # out shape: (batch, sequence_length, hidden_size)
        out, _ = self.lstm(x, (h0, c0))

        # Take only the LAST timestep's output
        # This is the LSTM's final "summary" of the 60-day sequence
        out = self.dropout(out[:, -1, :])

        # Pass through classifier
        out = self.fc(out)
        return out.squeeze()


# Initialize the model
model = StockLSTM(
    input_size  = len(available_features),
    hidden_size = HIDDEN_SIZE,
    num_layers  = NUM_LAYERS,
    dropout     = DROPOUT
).to(DEVICE)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"🧠 LSTM Model Architecture:")
print(f"   Input:       {len(available_features)} features × {SEQUENCE_LENGTH} days")
print(f"   LSTM layers: {NUM_LAYERS} × {HIDDEN_SIZE} hidden units")
print(f"   Parameters:  {total_params:,}")
print(f"   Device:      {DEVICE}\n")


# ─────────────────────────────────────────────────────────
#  TRAINING SETUP
#
#  Loss function: BCELoss (Binary Cross Entropy)
#  Perfect for yes/no prediction problems
#
#  Optimizer: Adam — adapts the learning rate automatically
#  much better than basic gradient descent
# ─────────────────────────────────────────────────────────

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Learning rate scheduler: reduce LR if training plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=3
)


# ─────────────────────────────────────────────────────────
#  TRAINING LOOP
# ─────────────────────────────────────────────────────────

print(f"🚀 Training LSTM for {EPOCHS} epochs...")
print(f"   Batch size: {BATCH_SIZE} | Learning rate: {LEARNING_RATE}\n")

train_losses = []
test_losses  = []
test_aucs    = []
best_auc     = 0
best_epoch   = 0

for epoch in range(EPOCHS):

    # ── TRAINING PHASE ──────────────────────────────────
    model.train()
    epoch_loss = 0

    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(DEVICE)
        batch_y = batch_y.to(DEVICE)

        optimizer.zero_grad()           # clear previous gradients
        predictions = model(batch_X)    # forward pass
        loss = criterion(predictions, batch_y)  # calculate error
        loss.backward()                 # backpropagate
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # prevent exploding gradients
        optimizer.step()                # update weights

        epoch_loss += loss.item()

    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # ── EVALUATION PHASE ────────────────────────────────
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():   # don't calculate gradients during evaluation
        eval_loss = 0
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            preds   = model(batch_X)
            loss    = criterion(preds, batch_y)
            eval_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())

    avg_test_loss = eval_loss / len(test_loader)
    test_losses.append(avg_test_loss)

    auc = roc_auc_score(all_labels, all_preds)
    test_aucs.append(auc)

    scheduler.step(avg_test_loss)

    # Save best model
    if auc > best_auc:
        best_auc   = auc
        best_epoch = epoch + 1
        torch.save(model.state_dict(),
                   os.path.join(MODEL_DIR, "lstm_model.pth"))

    # Print progress every epochs
    if True:
        preds_binary = [1 if p > 0.5 else 0 for p in all_preds]
        acc = accuracy_score(all_labels, preds_binary)
        print(f"   Epoch {epoch+1:02d}/{EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Test Loss: {avg_test_loss:.4f} | "
              f"AUC: {auc:.4f} | "
              f"Acc: {acc:.1%}")


# ─────────────────────────────────────────────────────────
#  FINAL EVALUATION
# ─────────────────────────────────────────────────────────

print(f"\n✅ Best model: Epoch {best_epoch} with AUC {best_auc:.4f}")
print(f"\n📊 Final Evaluation on Test Set:")

# Load best saved model
model.load_state_dict(
    torch.load(os.path.join(MODEL_DIR, "lstm_model.pth"),
               map_location=DEVICE)
)
model.eval()

all_preds, all_labels = [], []
with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(DEVICE)
        preds   = model(batch_X)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.numpy())

all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
preds_bin  = (all_preds > 0.5).astype(int)

accuracy = accuracy_score(all_labels, preds_bin)
auc      = roc_auc_score(all_labels, all_preds)

print(f"   Accuracy:  {accuracy:.1%}")
print(f"   AUC Score: {auc:.4f}")

# High confidence strategy
high_conf_mask   = all_preds > 0.60
high_conf_labels = all_labels[high_conf_mask]
win_rate = high_conf_labels.mean() if high_conf_mask.sum() > 0 else 0

print(f"\n   High-confidence predictions (>60%): {high_conf_mask.sum():,}")
print(f"   Win rate on those:                  {win_rate:.1%}")

print(f"\n📈 LSTM vs XGBoost comparison:")
print(f"   {'Metric':<20} {'XGBoost':>10} {'LSTM':>10}")
print(f"   {'─'*40}")
print(f"   {'Accuracy':<20} {'50.0%':>10} {str(round(accuracy*100, 1))+'%':>10}")
print(f"   {'AUC Score':<20} {'0.529':>10} {round(auc, 3):>10}")


# ─────────────────────────────────────────────────────────
#  SAVE TRAINING CURVE CHART
# ─────────────────────────────────────────────────────────

os.makedirs(RESULTS_DIR, exist_ok=True)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss curve
ax1.plot(train_losses, label="Train Loss", color="#4F86C6")
ax1.plot(test_losses,  label="Test Loss",  color="#E07B54")
ax1.axvline(x=best_epoch-1, color="green", linestyle="--",
            label=f"Best epoch ({best_epoch})")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.set_title("LSTM Training & Test Loss")
ax1.legend()

# AUC curve
ax2.plot(test_aucs, color="#5BA85C", label="Test AUC")
ax2.axhline(y=0.529, color="orange", linestyle="--",
            label="XGBoost AUC (0.529)")
ax2.axhline(y=0.5, color="red", linestyle=":",
            label="Random baseline (0.5)")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("AUC Score")
ax2.set_title("LSTM AUC Over Training")
ax2.legend()

plt.tight_layout()
chart_path = os.path.join(RESULTS_DIR, "lstm_training_curve.png")
plt.savefig(chart_path, dpi=120)
plt.close()

print(f"\n📈 Training curve saved → {chart_path}")
print(f"\n💾 Best model saved → models/lstm_model.pth")

print("\n" + "=" * 55)
print(f"  Phase 3B complete!")
print(f"  Accuracy: {accuracy:.1%}  |  AUC: {auc:.4f}")
print(f"  Best epoch: {best_epoch}/{EPOCHS}")
print("  Next: Build the Portfolio Bundle Engine")
print("=" * 55)