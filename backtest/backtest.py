# ============================================================
#  backtest.py  —  Stock AI Project: Phase 5
#  Backtesting Engine
#
#  Simulates trading with the ensemble model's signals
#  on historical data to measure real performance.
#
#  KEY PRINCIPLE: Walk-forward testing only.
#  We only trade on signals the model could have known
#  at that point in time — no cheating with future data.
# ============================================================
#
#  HOW TO RUN:
#  pip install pyportfolioopt matplotlib
#  python backtest/backtest.py
#
#  INPUT:  models/results/all_stock_scores.csv
#          data/processed/features_dataset.csv
#  OUTPUT: backtest/results/backtest_report.txt
#          backtest/results/performance_chart.png
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from datetime import datetime

# ─────────────────────────────────────────────────────────
#  SETTINGS
# ─────────────────────────────────────────────────────────

FEATURES_PATH = "data/processed/features_dataset.csv"
SCORES_PATH   = "models/results/all_stock_scores.csv"
OUTPUT_DIR    = "backtest/results"

# Simulation settings
INITIAL_CAPITAL    = 10000   # starting portfolio value ($)
STOCKS_PER_BUNDLE  = 10      # how many stocks to hold at once
REBALANCE_DAYS     = 20      # rebalance every 20 trading days (~1 month)
CONFIDENCE_CUTOFF  = 0.55    # minimum ensemble score to consider buying
TRANSACTION_COST   = 0.001   # 0.1% per trade (realistic for retail)

# Backtest period — using 2023-2024 as out-of-sample test
BACKTEST_START = "2023-01-01"
BACKTEST_END   = "2024-12-31"

print("=" * 55)
print("  Stock AI — Backtesting Engine")
print("=" * 55)


# ─────────────────────────────────────────────────────────
#  LOAD DATA
# ─────────────────────────────────────────────────────────

print(f"\n📂 Loading data...")
df = pd.read_csv(FEATURES_PATH, index_col=0, parse_dates=True)
scores_df = pd.read_csv(SCORES_PATH)

# Filter to backtest period
backtest_df = df[
    (df.index >= BACKTEST_START) &
    (df.index <= BACKTEST_END)
].copy()

print(f"   Features dataset: {df.shape[0]:,} rows")
print(f"   Backtest period:  {BACKTEST_START} → {BACKTEST_END}")
print(f"   Rows in period:   {len(backtest_df):,}")
print(f"   Stocks scored:    {len(scores_df)}")


# ─────────────────────────────────────────────────────────
#  BUILD PRICE PIVOT TABLE
#  Rows = dates, Columns = tickers, Values = close price
#  This makes it easy to calculate portfolio returns
# ─────────────────────────────────────────────────────────

print("\n📊 Building price matrix...")
price_pivot = backtest_df.pivot_table(
    index=backtest_df.index,
    columns="Ticker",
    values="Close"
)
price_pivot = price_pivot.fillna(method="ffill")  # forward fill missing prices

print(f"   Price matrix: {price_pivot.shape[0]} days × {price_pivot.shape[1]} stocks")

# Get trading days in backtest period
trading_days = price_pivot.index.tolist()
print(f"   Trading days: {len(trading_days)}")


# ─────────────────────────────────────────────────────────
#  BACKTEST SIMULATION
#
#  Every 20 trading days we:
#  1. Re-score all stocks using model signals
#  2. Pick the top STOCKS_PER_BUNDLE stocks
#  3. Rebalance portfolio to equal weight
#  4. Track daily returns until next rebalance
#
#  We compare against:
#  - SPY (S&P 500 buy & hold) as benchmark
#  - Random stock selection as sanity check
# ─────────────────────────────────────────────────────────

print(f"\n🔄 Running backtest simulation...")
print(f"   Capital: ${INITIAL_CAPITAL:,} | Rebalance: every {REBALANCE_DAYS} days")
print(f"   Stocks per bundle: {STOCKS_PER_BUNDLE} | Min confidence: {CONFIDENCE_CUTOFF}\n")

# Get top stocks from model scores
top_stocks = scores_df[
    scores_df["Ensemble_Score"] >= CONFIDENCE_CUTOFF
].sort_values("Ensemble_Score", ascending=False)

# Filter to stocks we have price data for
available_tickers = set(price_pivot.columns)
top_stocks = top_stocks[top_stocks["Ticker"].isin(available_tickers)]

print(f"   High-confidence stocks available: {len(top_stocks)}")

# ── STRATEGY 1: MODEL PORTFOLIO ──────────────────────────
model_portfolio_value  = [INITIAL_CAPITAL]
model_daily_returns    = []
current_holdings       = []
rebalance_dates        = []

# ── STRATEGY 2: SPY BUY & HOLD ───────────────────────────
spy_portfolio_value = [INITIAL_CAPITAL]
spy_daily_returns   = []

# ── STRATEGY 3: RANDOM SELECTION (sanity check) ──────────
np.random.seed(42)
random_tickers = list(available_tickers)[:STOCKS_PER_BUNDLE * 3]
random_picks   = np.random.choice(random_tickers, STOCKS_PER_BUNDLE, replace=False)
random_portfolio_value = [INITIAL_CAPITAL]
random_daily_returns   = []

# Run day by day
for day_idx, date in enumerate(trading_days[1:], 1):
    prev_date = trading_days[day_idx - 1]

    # ── REBALANCE CHECK ───────────────────────────────────
    # Every REBALANCE_DAYS we pick new stocks
    if day_idx % REBALANCE_DAYS == 1 or len(current_holdings) == 0:
        # Pick top N stocks by ensemble score that we have prices for
        new_holdings = []
        for _, row in top_stocks.iterrows():
            ticker = row["Ticker"]
            if (ticker in price_pivot.columns and
                not pd.isna(price_pivot.loc[date, ticker]) and
                not pd.isna(price_pivot.loc[prev_date, ticker])):
                new_holdings.append(ticker)
            if len(new_holdings) >= STOCKS_PER_BUNDLE:
                break

        if new_holdings:
            # Apply transaction cost on rebalance
            transaction_fee = model_portfolio_value[-1] * TRANSACTION_COST
            model_portfolio_value[-1] -= transaction_fee
            current_holdings = new_holdings
            rebalance_dates.append(date)

    # ── DAILY RETURN CALCULATION ──────────────────────────

    # Model portfolio daily return
    if current_holdings:
        daily_rets = []
        for ticker in current_holdings:
            if (ticker in price_pivot.columns and
                not pd.isna(price_pivot.loc[date, ticker]) and
                not pd.isna(price_pivot.loc[prev_date, ticker]) and
                price_pivot.loc[prev_date, ticker] > 0):

                ret = (price_pivot.loc[date, ticker] -
                       price_pivot.loc[prev_date, ticker]) / \
                       price_pivot.loc[prev_date, ticker]
                daily_rets.append(ret)

        if daily_rets:
            portfolio_ret = np.mean(daily_rets)  # equal weight
            new_value = model_portfolio_value[-1] * (1 + portfolio_ret)
            model_portfolio_value.append(new_value)
            model_daily_returns.append(portfolio_ret)
        else:
            model_portfolio_value.append(model_portfolio_value[-1])
            model_daily_returns.append(0)
    else:
        model_portfolio_value.append(model_portfolio_value[-1])
        model_daily_returns.append(0)

    # SPY buy & hold daily return
    if ("SPY" in price_pivot.columns and
        not pd.isna(price_pivot.loc[date, "SPY"]) and
        not pd.isna(price_pivot.loc[prev_date, "SPY"]) and
        price_pivot.loc[prev_date, "SPY"] > 0):

        spy_ret = (price_pivot.loc[date, "SPY"] -
                   price_pivot.loc[prev_date, "SPY"]) / \
                   price_pivot.loc[prev_date, "SPY"]
        spy_portfolio_value.append(spy_portfolio_value[-1] * (1 + spy_ret))
        spy_daily_returns.append(spy_ret)
    else:
        spy_portfolio_value.append(spy_portfolio_value[-1])
        spy_daily_returns.append(0)

    # Random portfolio daily return
    rand_rets = []
    for ticker in random_picks:
        if (ticker in price_pivot.columns and
            not pd.isna(price_pivot.loc[date, ticker]) and
            not pd.isna(price_pivot.loc[prev_date, ticker]) and
            price_pivot.loc[prev_date, ticker] > 0):

            ret = (price_pivot.loc[date, ticker] -
                   price_pivot.loc[prev_date, ticker]) / \
                   price_pivot.loc[prev_date, ticker]
            rand_rets.append(ret)

    if rand_rets:
        random_portfolio_value.append(
            random_portfolio_value[-1] * (1 + np.mean(rand_rets))
        )
        random_daily_returns.append(np.mean(rand_rets))
    else:
        random_portfolio_value.append(random_portfolio_value[-1])
        random_daily_returns.append(0)


# ─────────────────────────────────────────────────────────
#  CALCULATE PERFORMANCE METRICS
# ─────────────────────────────────────────────────────────

def calc_metrics(daily_returns, portfolio_values, name):
    """Calculate key performance metrics for a strategy."""
    rets = np.array(daily_returns)

    # Total return
    total_return = (portfolio_values[-1] - portfolio_values[0]) / \
                    portfolio_values[0]

    # Annualized return (252 trading days per year)
    n_days = len(rets)
    annual_return = (1 + total_return) ** (252 / n_days) - 1

    # Sharpe Ratio: return per unit of risk
    # Higher = better. Above 1.0 is good, above 2.0 is excellent
    if rets.std() > 0:
        sharpe = (rets.mean() * 252) / (rets.std() * np.sqrt(252))
    else:
        sharpe = 0

    # Maximum Drawdown: worst peak-to-trough drop
    # e.g. -0.15 means the portfolio fell 15% from its peak at worst
    peak = portfolio_values[0]
    max_dd = 0
    for val in portfolio_values:
        if val > peak:
            peak = val
        dd = (val - peak) / peak
        if dd < max_dd:
            max_dd = dd

    # Win rate: % of days with positive return
    win_rate = (rets > 0).mean()

    return {
        "Strategy":       name,
        "Total Return":   f"{total_return:.1%}",
        "Annual Return":  f"{annual_return:.1%}",
        "Sharpe Ratio":   f"{sharpe:.3f}",
        "Max Drawdown":   f"{max_dd:.1%}",
        "Win Rate":       f"{win_rate:.1%}",
        "Final Value":    f"${portfolio_values[-1]:,.0f}",
        # Raw values for comparison
        "_total_return":  total_return,
        "_sharpe":        sharpe,
        "_max_dd":        max_dd,
    }


model_metrics  = calc_metrics(model_daily_returns,  model_portfolio_value,  "AI Model")
spy_metrics    = calc_metrics(spy_daily_returns,    spy_portfolio_value,    "SPY Buy & Hold")
random_metrics = calc_metrics(random_daily_returns, random_portfolio_value, "Random Selection")

# Alpha = model return above SPY
alpha = model_metrics["_total_return"] - spy_metrics["_total_return"]


# ─────────────────────────────────────────────────────────
#  PRINT RESULTS
# ─────────────────────────────────────────────────────────

print("\n" + "=" * 55)
print("  📊 BACKTEST RESULTS")
print(f"  Period: {BACKTEST_START} → {BACKTEST_END}")
print(f"  Starting capital: ${INITIAL_CAPITAL:,}")
print("=" * 55)

metrics_display = [model_metrics, spy_metrics, random_metrics]
headers = ["Metric", "AI Model", "SPY B&H", "Random"]
rows = [
    ["Total Return",  model_metrics["Total Return"],
                      spy_metrics["Total Return"],
                      random_metrics["Total Return"]],
    ["Annual Return", model_metrics["Annual Return"],
                      spy_metrics["Annual Return"],
                      random_metrics["Annual Return"]],
    ["Sharpe Ratio",  model_metrics["Sharpe Ratio"],
                      spy_metrics["Sharpe Ratio"],
                      random_metrics["Sharpe Ratio"]],
    ["Max Drawdown",  model_metrics["Max Drawdown"],
                      spy_metrics["Max Drawdown"],
                      random_metrics["Max Drawdown"]],
    ["Win Rate",      model_metrics["Win Rate"],
                      spy_metrics["Win Rate"],
                      random_metrics["Win Rate"]],
    ["Final Value",   model_metrics["Final Value"],
                      spy_metrics["Final Value"],
                      random_metrics["Final Value"]],
]

print(f"\n  {'Metric':<18} {'AI Model':>12} {'SPY B&H':>12} {'Random':>12}")
print(f"  {'─'*54}")
for row in rows:
    print(f"  {row[0]:<18} {row[1]:>12} {row[2]:>12} {row[3]:>12}")

print(f"\n  Alpha vs SPY: {alpha:+.1%}")
if alpha > 0:
    print(f"  The AI model OUTPERFORMED the S&P 500 by {alpha:.1%}")
else:
    print(f"  The AI model underperformed the S&P 500 by {abs(alpha):.1%}")

print(f"\n  Rebalances executed: {len(rebalance_dates)}")
print(f"  Current holdings:    {current_holdings}")


# ─────────────────────────────────────────────────────────
#  GENERATE PERFORMANCE CHART
# ─────────────────────────────────────────────────────────

print(f"\n📈 Generating performance chart...")

dates_plot = trading_days[:len(model_portfolio_value)]

fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig)

# Chart 1: Portfolio value over time
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(dates_plot, model_portfolio_value,
         color="#4F86C6", linewidth=2, label="AI Model")
ax1.plot(dates_plot[:len(spy_portfolio_value)], spy_portfolio_value,
         color="#E07B54", linewidth=2, label="SPY Buy & Hold")
ax1.plot(dates_plot[:len(random_portfolio_value)], random_portfolio_value,
         color="#999999", linewidth=1.5, linestyle="--", label="Random")

# Mark rebalance points
for rd in rebalance_dates:
    if rd in dates_plot:
        idx = dates_plot.index(rd)
        if idx < len(model_portfolio_value):
            ax1.axvline(x=rd, color="#4F86C6", alpha=0.2, linewidth=0.8)

ax1.set_title(f"Portfolio Performance ({BACKTEST_START} → {BACKTEST_END})",
              fontsize=13, fontweight="bold")
ax1.set_ylabel("Portfolio Value ($)")
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(
    plt.FuncFormatter(lambda x, p: f"${x:,.0f}")
)

# Chart 2: Monthly returns heatmap-style bar chart
ax2 = fig.add_subplot(gs[1, 0])
monthly_model = pd.Series(
    model_daily_returns,
    index=trading_days[1:len(model_daily_returns)+1]
).resample("ME").sum()

colors = ["#5BA85C" if r > 0 else "#E07B54" for r in monthly_model]
ax2.bar(range(len(monthly_model)), monthly_model * 100, color=colors)
ax2.axhline(y=0, color="black", linewidth=0.8)
ax2.set_title("AI Model Monthly Returns", fontsize=11, fontweight="bold")
ax2.set_ylabel("Return (%)")
ax2.set_xlabel("Month")
ax2.grid(True, alpha=0.3, axis="y")

# Chart 3: Drawdown chart
ax3 = fig.add_subplot(gs[1, 1])
model_vals = np.array(model_portfolio_value)
running_max = np.maximum.accumulate(model_vals)
drawdown = (model_vals - running_max) / running_max * 100

spy_vals = np.array(spy_portfolio_value)
spy_running_max = np.maximum.accumulate(spy_vals)
spy_drawdown = (spy_vals - spy_running_max) / spy_running_max * 100

ax3.fill_between(dates_plot[:len(drawdown)], drawdown,
                 0, color="#E07B54", alpha=0.5, label="AI Model")
ax3.fill_between(dates_plot[:len(spy_drawdown)], spy_drawdown,
                 0, color="#999999", alpha=0.3, label="SPY")
ax3.set_title("Drawdown Chart", fontsize=11, fontweight="bold")
ax3.set_ylabel("Drawdown (%)")
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
os.makedirs(OUTPUT_DIR, exist_ok=True)
chart_path = os.path.join(OUTPUT_DIR, "performance_chart.png")
plt.savefig(chart_path, dpi=120, bbox_inches="tight")
plt.close()
print(f"   Chart saved → {chart_path}")


# ─────────────────────────────────────────────────────────
#  SAVE TEXT REPORT
# ─────────────────────────────────────────────────────────

report = f"""
STOCK AI — BACKTEST REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
Period: {BACKTEST_START} to {BACKTEST_END}
Starting Capital: ${INITIAL_CAPITAL:,}
{'='*55}

PERFORMANCE SUMMARY
{'─'*55}
{'Metric':<20} {'AI Model':>12} {'SPY B&H':>12} {'Random':>12}
{'─'*55}
"""
for row in rows:
    report += f"{row[0]:<20} {row[1]:>12} {row[2]:>12} {row[3]:>12}\n"

report += f"""
Alpha vs SPY: {alpha:+.1%}
Rebalances:   {len(rebalance_dates)}
{'='*55}

SETTINGS USED
{'─'*55}
Stocks per bundle:   {STOCKS_PER_BUNDLE}
Rebalance frequency: every {REBALANCE_DAYS} trading days
Min confidence:      {CONFIDENCE_CUTOFF}
Transaction cost:    {TRANSACTION_COST:.1%} per rebalance
"""

report_path = os.path.join(OUTPUT_DIR, "backtest_report.txt")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"   Report saved → {report_path}")

print("\n" + "=" * 55)
print("  Phase 5 complete!")
print(f"  AI Model: {model_metrics['Total Return']} total return")
print(f"  SPY:      {spy_metrics['Total Return']} total return")
print(f"  Alpha:    {alpha:+.1%}")
print("  Next: Streamlit Dashboard")
print("=" * 55)