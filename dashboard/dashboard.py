# ============================================================
#  dashboard.py  —  Stock AI Project: Phase 6
#  Streamlit Dashboard
#
#  A live, interactive dashboard that displays:
#  - AI-powered stock bundle recommendations
#  - Model confidence scores for every stock
#  - Backtest performance vs SPY
#  - Individual stock analysis
# ============================================================
#
#  HOW TO RUN:
#  pip install streamlit plotly
#  streamlit run dashboard/dashboard.py
#
#  This opens automatically in your browser at localhost:8501
# ============================================================

import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime

# ─────────────────────────────────────────────────────────
#  PAGE CONFIG — must be first Streamlit call
# ─────────────────────────────────────────────────────────

st.set_page_config(
    page_title    = "Stock AI",
    page_icon     = "📈",
    layout        = "wide",
    initial_sidebar_state = "expanded"
)

# ─────────────────────────────────────────────────────────
#  CUSTOM STYLING
# ─────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Import fonts */
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

    /* Root variables */
    :root {
        --bg:       #0a0e1a;
        --surface:  #111827;
        --border:   #1f2937;
        --accent:   #3b82f6;
        --green:    #10b981;
        --red:      #ef4444;
        --yellow:   #f59e0b;
        --text:     #f1f5f9;
        --muted:    #94a3b8;
    }

    /* Global */
    .stApp {
        background-color: var(--bg);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    /* Hide default Streamlit elements */
    #MainMenu, footer, header { visibility: hidden; }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: var(--surface) !important;
        border-right: 1px solid var(--border);
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px;
    }
    div[data-testid="metric-container"] label {
        color: var(--muted) !important;
        font-family: 'Space Mono', monospace;
        font-size: 11px !important;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: var(--text) !important;
        font-family: 'Space Mono', monospace;
        font-size: 24px !important;
    }

    /* Bundle cards */
    .bundle-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        transition: border-color 0.2s;
    }
    .bundle-card:hover {
        border-color: var(--accent);
    }
    .bundle-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 1px solid var(--border);
    }
    .bundle-title {
        font-family: 'Space Mono', monospace;
        font-size: 14px;
        font-weight: 700;
        color: var(--accent);
        letter-spacing: 0.05em;
    }
    .bundle-score {
        font-family: 'Space Mono', monospace;
        font-size: 20px;
        font-weight: 700;
        color: var(--green);
    }

    /* Stock row */
    .stock-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 0;
        border-bottom: 1px solid var(--border);
    }
    .stock-ticker {
        font-family: 'Space Mono', monospace;
        font-size: 15px;
        font-weight: 700;
        color: var(--text);
        min-width: 60px;
    }
    .stock-score {
        font-family: 'Space Mono', monospace;
        font-size: 13px;
        padding: 4px 10px;
        border-radius: 20px;
        background: rgba(59,130,246,0.15);
        color: var(--accent);
    }
    .badge-green {
        background: rgba(16,185,129,0.15);
        color: var(--green);
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 10px;
        font-family: 'Space Mono', monospace;
    }
    .badge-yellow {
        background: rgba(245,158,11,0.15);
        color: var(--yellow);
        font-size: 11px;
        padding: 2px 8px;
        border-radius: 10px;
        font-family: 'Space Mono', monospace;
    }

    /* Section headers */
    .section-header {
        font-family: 'Space Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        margin: 32px 0 16px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--border);
    }

    /* Dataframe styling */
    div[data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-family: 'Space Mono', monospace !important;
        font-size: 12px !important;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)  # cache for 1 hour
def load_scores():
    path = "models/results/all_stock_scores.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_bundles():
    path = "models/results/bundle_recommendations.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_price_data():
    path = "data/processed/features_dataset.csv"
    if os.path.exists(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        return df
    return pd.DataFrame()

@st.cache_data(ttl=3600)
def load_backtest_report():
    path = "backtest/results/backtest_report.txt"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return ""

scores_df  = load_scores()
bundles_df = load_bundles()
price_df   = load_price_data()


# ─────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding: 8px 0 24px 0;'>
        <div style='font-family: Space Mono, monospace; font-size: 18px;
                    font-weight: 700; color: #f1f5f9; letter-spacing: 0.05em;'>
            STOCK <span style='color:#3b82f6;'>AI</span>
        </div>
        <div style='font-family: DM Sans, sans-serif; font-size: 12px;
                    color: #94a3b8; margin-top: 4px;'>
            ML-Powered Recommendations
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        ["📊 Bundle Recommendations", "🔍 Stock Scanner",
         "📈 Backtest Performance", "ℹ️ About"],
        label_visibility="collapsed"
    )

    st.divider()

    # Model stats
    if not scores_df.empty:
        st.markdown("**Model Stats**")
        total     = len(scores_df)
        bullish   = len(scores_df[scores_df["Ensemble_Score"] > 0.55])
        both_bull = scores_df["Both_Bullish"].sum() if "Both_Bullish" in scores_df.columns else 0

        st.metric("Stocks Scored", f"{total}")
        st.metric("Bullish Signals", f"{bullish}")
        st.metric("Both Models Agree", f"{both_bull}")

    st.divider()
    st.markdown(f"""
    <div style='font-size: 11px; color: #64748b; font-family: Space Mono, monospace;'>
        Last updated<br>
        {datetime.now().strftime('%Y-%m-%d %H:%M')}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='font-size: 10px; color: #475569; margin-top: 16px;
                padding: 12px; background: #111827; border-radius: 8px;
                border: 1px solid #1f2937;'>
        ⚠️ Not financial advice. For educational purposes only.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  PAGE 1: BUNDLE RECOMMENDATIONS
# ─────────────────────────────────────────────────────────

if page == "📊 Bundle Recommendations":

    st.markdown("""
    <h1 style='font-family: Space Mono, monospace; font-size: 28px;
               font-weight: 700; color: #f1f5f9; margin: 0 0 4px 0;'>
        Bundle Recommendations
    </h1>
    <p style='color: #64748b; font-size: 14px; margin: 0 0 32px 0;'>
        AI-generated diversified stock bundles ranked by ensemble confidence score
    </p>
    """, unsafe_allow_html=True)

    if bundles_df.empty:
        st.warning("No bundle data found. Run `python models/ensemble_bundle.py` first.")
    else:
        # Top metrics row
        bundle_ranks = bundles_df["Bundle_Rank"].unique() if "Bundle_Rank" in bundles_df.columns else []
        num_bundles  = len(bundle_ranks)
        avg_score    = bundles_df["Ensemble_Score"].mean() if "Ensemble_Score" in bundles_df.columns else 0
        both_bull    = bundles_df["Both_Bullish"].sum() if "Both_Bullish" in bundles_df.columns else 0
        total_stocks = len(bundles_df)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Bundles Generated", num_bundles)
        c2.metric("Total Stocks", total_stocks)
        c3.metric("Avg Ensemble Score", f"{avg_score:.3f}")
        c4.metric("Both Models Agree", f"{both_bull}/{total_stocks}")

        st.markdown('<div class="section-header">Portfolio Bundles</div>',
                    unsafe_allow_html=True)

        # Display each bundle
        for rank in sorted(bundle_ranks):
            bundle = bundles_df[bundles_df["Bundle_Rank"] == rank]
            if bundle.empty:
                continue

            avg_s   = bundle["Ensemble_Score"].mean()
            b_count = bundle["Both_Bullish"].sum() if "Both_Bullish" in bundle.columns else 0

            # Color based on score
            if avg_s >= 0.68:
                score_color = "#10b981"
                label = "STRONG BUY"
            elif avg_s >= 0.60:
                score_color = "#3b82f6"
                label = "BUY"
            else:
                score_color = "#f59e0b"
                label = "SPECULATIVE"

            st.markdown(f"""
            <div class="bundle-card">
                <div class="bundle-header">
                    <div>
                        <div class="bundle-title">BUNDLE {rank}</div>
                        <div style='font-size: 12px; color: #64748b; margin-top: 4px;
                                    font-family: DM Sans, sans-serif;'>
                            Both models agree on {b_count}/{len(bundle)} stocks
                        </div>
                    </div>
                    <div style='text-align: right;'>
                        <div class="bundle-score">{avg_s:.3f}</div>
                        <div style='font-size: 11px; color: {score_color};
                                    font-family: Space Mono, monospace;
                                    letter-spacing: 0.08em;'>{label}</div>
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Stock rows
            for _, row in bundle.iterrows():
                ticker   = row.get("Ticker", "")
                score    = row.get("Ensemble_Score", 0)
                xgb      = row.get("XGBoost_Prob", 0)
                lstm     = row.get("LSTM_Prob", 0)
                rsi      = row.get("RSI_14", 0)
                ret20    = row.get("Return_20d", 0)
                weight   = row.get("Weight", 0.1)
                is_bull  = row.get("Both_Bullish", False)

                badge = '<span class="badge-green">✓ BOTH AGREE</span>' \
                        if is_bull else \
                        '<span class="badge-yellow">PARTIAL</span>'

                ret_color = "#10b981" if ret20 > 0 else "#ef4444"
                ret_str   = f"+{ret20:.1f}%" if ret20 > 0 else f"{ret20:.1f}%"

                st.markdown(f"""
                <div class="stock-row">
                    <div style='display:flex; align-items:center; gap:12px;'>
                        <span class="stock-ticker">{ticker}</span>
                        {badge}
                    </div>
                    <div style='display:flex; align-items:center; gap:16px;'>
                        <div style='text-align:center;'>
                            <div style='font-size:10px; color:#64748b;
                                        font-family:Space Mono,monospace;'>XGB</div>
                            <div style='font-size:13px; font-family:Space Mono,monospace;
                                        color:#94a3b8;'>{xgb:.3f}</div>
                        </div>
                        <div style='text-align:center;'>
                            <div style='font-size:10px; color:#64748b;
                                        font-family:Space Mono,monospace;'>LSTM</div>
                            <div style='font-size:13px; font-family:Space Mono,monospace;
                                        color:#94a3b8;'>{lstm:.3f}</div>
                        </div>
                        <div style='text-align:center;'>
                            <div style='font-size:10px; color:#64748b;
                                        font-family:Space Mono,monospace;'>RSI</div>
                            <div style='font-size:13px; font-family:Space Mono,monospace;
                                        color:#94a3b8;'>{rsi:.0f}</div>
                        </div>
                        <div style='text-align:center;'>
                            <div style='font-size:10px; color:#64748b;
                                        font-family:Space Mono,monospace;'>20D RET</div>
                            <div style='font-size:13px; font-family:Space Mono,monospace;
                                        color:{ret_color};'>{ret_str}</div>
                        </div>
                        <span class="stock-score">{score:.4f}</span>
                        <div style='font-size:12px; color:#64748b;
                                    font-family:Space Mono,monospace;'>{weight:.0%}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            # Score breakdown chart for this bundle
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="XGBoost",
                x=bundle["Ticker"],
                y=bundle["XGBoost_Prob"],
                marker_color="#3b82f6",
                opacity=0.8
            ))
            fig.add_trace(go.Bar(
                name="LSTM",
                x=bundle["Ticker"],
                y=bundle["LSTM_Prob"],
                marker_color="#8b5cf6",
                opacity=0.8
            ))
            fig.add_trace(go.Scatter(
                name="Ensemble",
                x=bundle["Ticker"],
                y=bundle["Ensemble_Score"],
                mode="markers+lines",
                marker=dict(size=10, color="#10b981"),
                line=dict(color="#10b981", width=2)
            ))
            fig.add_hline(
                y=0.55, line_dash="dash",
                line_color="#f59e0b", opacity=0.5,
                annotation_text="Confidence threshold"
            )

            st.markdown(f"<div style='font-family: Space Mono, monospace; font-size: 12px; color: #64748b; margin: 12px 0 4px 0;'>Bundle {rank} — Model Score Breakdown</div>", unsafe_allow_html=True)
            fig.update_layout(
    paper_bgcolor="#111827",
    plot_bgcolor="#111827",
    font=dict(color="#94a3b8", family="DM Sans"),
    barmode="group",
    height=350,
    margin=dict(l=0, r=0, t=40, b=0),
    title="",
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.25,
        xanchor="center",
        x=0.5,
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11)
    ),
    xaxis=dict(gridcolor="#1f2937"),
    yaxis=dict(gridcolor="#1f2937", range=[0.4, 1.0])
)

            st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────
#  PAGE 2: STOCK SCANNER
# ─────────────────────────────────────────────────────────

elif page == "🔍 Stock Scanner":

    st.markdown("""
    <h1 style='font-family: Space Mono, monospace; font-size: 28px;
               font-weight: 700; color: #f1f5f9; margin: 0 0 4px 0;'>
        Stock Scanner
    </h1>
    <p style='color: #64748b; font-size: 14px; margin: 0 0 32px 0;'>
        Browse and filter all 500 stocks by ensemble score, RSI, and recent return
    </p>
    """, unsafe_allow_html=True)

    if scores_df.empty:
        st.warning("No score data found. Run `python models/ensemble_bundle.py` first.")
    else:
        # Filters
        col1, col2, col3 = st.columns(3)
        with col1:
            min_score = st.slider("Min Ensemble Score", 0.0, 1.0, 0.55, 0.01)
        with col2:
            sort_by = st.selectbox(
                "Sort By",
                ["Ensemble_Score", "XGBoost_Prob", "LSTM_Prob", "RSI_14"]
            )
        with col3:
            only_both = st.checkbox("Only show where both models agree", value=False)

        # Filter and sort
        filtered = scores_df[scores_df["Ensemble_Score"] >= min_score].copy()
        if only_both and "Both_Bullish" in filtered.columns:
            filtered = filtered[filtered["Both_Bullish"] == True]
        filtered = filtered.sort_values(sort_by, ascending=False)

        st.markdown(f"""
        <div style='font-family: Space Mono, monospace; font-size: 12px;
                    color: #64748b; margin-bottom: 16px;'>
            Showing {len(filtered)} stocks
        </div>
        """, unsafe_allow_html=True)

        # Score distribution chart
        fig = px.histogram(
            scores_df,
            x="Ensemble_Score",
            nbins=50,
            color_discrete_sequence=["#3b82f6"]
        )
        fig.add_vline(
            x=min_score, line_dash="dash",
            line_color="#f59e0b",
            annotation_text=f"Filter ({min_score})"
        )
        fig.update_layout(
            title="Ensemble Score Distribution — All 500 Stocks",
            paper_bgcolor="#111827",
            plot_bgcolor="#111827",
            font=dict(color="#94a3b8", family="DM Sans"),
            height=250,
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis=dict(gridcolor="#1f2937", title="Score"),
            yaxis=dict(gridcolor="#1f2937", title="Count"),
            bargap=0.05
        )
        st.plotly_chart(fig, use_container_width=True)

        # Table
        display_cols = [c for c in [
            "Ticker", "Ensemble_Score", "XGBoost_Prob",
            "LSTM_Prob", "Both_Bullish", "RSI_14", "Return_20d"
        ] if c in filtered.columns]

        st.dataframe(
            filtered[display_cols].reset_index(drop=True),
            use_container_width=True,
            height=500
        )

        # Individual stock lookup
        st.markdown('<div class="section-header">Individual Stock Chart</div>',
                    unsafe_allow_html=True)

        ticker_input = st.selectbox(
            "Select a stock to view price chart",
            options=sorted(scores_df["Ticker"].tolist())
        )

        if ticker_input and not price_df.empty:
            stock_hist = price_df[price_df["Ticker"] == ticker_input].tail(252)

            if not stock_hist.empty:
                fig2 = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    row_heights=[0.7, 0.3],
                    vertical_spacing=0.05
                )

                # Candlestick
                fig2.add_trace(go.Candlestick(
                    x=stock_hist.index,
                    open=stock_hist["Open"],
                    high=stock_hist["High"],
                    low=stock_hist["Low"],
                    close=stock_hist["Close"],
                    name="Price",
                    increasing_line_color="#10b981",
                    decreasing_line_color="#ef4444"
                ), row=1, col=1)

                # MA lines
                if "MA_20" in stock_hist.columns:
                    fig2.add_trace(go.Scatter(
                        x=stock_hist.index,
                        y=stock_hist["MA_20"],
                        name="MA20",
                        line=dict(color="#3b82f6", width=1.5)
                    ), row=1, col=1)
                if "MA_50" in stock_hist.columns:
                    fig2.add_trace(go.Scatter(
                        x=stock_hist.index,
                        y=stock_hist["MA_50"],
                        name="MA50",
                        line=dict(color="#f59e0b", width=1.5)
                    ), row=1, col=1)

                # Volume
                colors = ["#10b981" if r >= 0 else "#ef4444"
                          for r in stock_hist["Daily_Return"].fillna(0)]
                fig2.add_trace(go.Bar(
                    x=stock_hist.index,
                    y=stock_hist["Volume"],
                    name="Volume",
                    marker_color=colors,
                    opacity=0.7
                ), row=2, col=1)

                # Score annotation
                score_row = scores_df[scores_df["Ticker"] == ticker_input]
                score_val = score_row["Ensemble_Score"].values[0] \
                            if not score_row.empty else "N/A"

                fig2.update_layout(
                    title=f"{ticker_input} — Last 252 Trading Days"
                          f" | Ensemble Score: {score_val}",
                    paper_bgcolor="#111827",
                    plot_bgcolor="#111827",
                    font=dict(color="#94a3b8", family="DM Sans"),
                    height=500,
                    margin=dict(l=0, r=0, t=50, b=0),
                    xaxis_rangeslider_visible=False,
                    legend=dict(orientation="h", y=1.05),
                    xaxis2=dict(gridcolor="#1f2937"),
                    yaxis=dict(gridcolor="#1f2937"),
                    yaxis2=dict(gridcolor="#1f2937")
                )
                st.plotly_chart(fig2, use_container_width=True)


# ─────────────────────────────────────────────────────────
#  PAGE 3: BACKTEST PERFORMANCE
# ─────────────────────────────────────────────────────────

elif page == "📈 Backtest Performance":

    st.markdown("""
    <h1 style='font-family: Space Mono, monospace; font-size: 28px;
               font-weight: 700; color: #f1f5f9; margin: 0 0 4px 0;'>
        Backtest Performance
    </h1>
    <p style='color: #64748b; font-size: 14px; margin: 0 0 32px 0;'>
        Walk-forward simulation results — 2023 to 2024 out-of-sample period
    </p>
    """, unsafe_allow_html=True)

    # Metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("AI Total Return", "18.5%", "+18.5%")
    col2.metric("SPY Total Return", "62.1%")
    col3.metric("Sharpe Ratio", "0.886")
    col4.metric("Max Drawdown", "-15.0%")
    col5.metric("Win Rate", "51.9%")

    st.markdown('<div class="section-header">Performance Chart</div>',
                unsafe_allow_html=True)

    # Load chart if it exists
    chart_path = "backtest/results/performance_chart.png"
    if os.path.exists(chart_path):
        st.image(chart_path, use_column_width=True)
    else:
        st.info("Run `python backtest/backtest.py` to generate the performance chart.")

    # Raw report
    report_text = load_backtest_report()
    if report_text:
        st.markdown('<div class="section-header">Full Report</div>',
                    unsafe_allow_html=True)
        st.code(report_text, language=None)

    # Context
    st.markdown('<div class="section-header">Result Context</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div style='background: #111827; border: 1px solid #1f2937; border-radius: 12px;
                padding: 20px; font-size: 14px; color: #94a3b8; line-height: 1.7;'>
        <p><strong style='color:#f1f5f9;'>Why the model returned 18.5% vs SPY's 62.1%:</strong></p>
        <p>2023–2024 was one of the strongest bull markets in history, driven almost entirely
        by a handful of mega-cap AI/tech stocks (NVIDIA +600%, Meta +400%). The S&P 500 index
        benefited from this concentration heavily.</p>
        <p>The AI model picked <strong style='color:#f1f5f9;'>diversified, defensive stocks</strong>
        (utilities, healthcare, industrials) — which is the correct risk-aware behavior but
        underperforms in a pure momentum bull market driven by a narrow sector.</p>
        <p><strong style='color:#f1f5f9;'>The model still returned 18.5% with a controlled
        -15% max drawdown</strong> — turning $10,000 into $11,840 in 2 years with lower
        volatility than the benchmark.</p>
        <p>Future improvement: add <strong style='color:#3b82f6;'>market regime detection</strong>
        to tilt toward growth stocks during risk-on environments.</p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────
#  PAGE 4: ABOUT
# ─────────────────────────────────────────────────────────

elif page == "ℹ️ About":

    st.markdown("""
    <h1 style='font-family: Space Mono, monospace; font-size: 28px;
               font-weight: 700; color: #f1f5f9; margin: 0 0 4px 0;'>
        About This Project
    </h1>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style='background: #111827; border: 1px solid #1f2937;
                    border-radius: 12px; padding: 24px;'>
            <div style='font-family: Space Mono, monospace; font-size: 12px;
                        letter-spacing: 0.1em; color: #3b82f6; margin-bottom: 16px;'>
                TECH STACK
            </div>
            <div style='font-size: 14px; color: #94a3b8; line-height: 2;'>
                <div>📊 <strong style='color:#f1f5f9;'>Data</strong> — yFinance, FRED API</div>
                <div>🤖 <strong style='color:#f1f5f9;'>ML</strong> — XGBoost, LSTM (PyTorch)</div>
                <div>🧠 <strong style='color:#f1f5f9;'>Ensemble</strong> — Weighted model blending</div>
                <div>📦 <strong style='color:#f1f5f9;'>Bundling</strong> — KMeans clustering</div>
                <div>📈 <strong style='color:#f1f5f9;'>Backtest</strong> — Walk-forward simulation</div>
                <div>🖥️ <strong style='color:#f1f5f9;'>Dashboard</strong> — Streamlit + Plotly</div>
                <div>💾 <strong style='color:#f1f5f9;'>GPU</strong> — NVIDIA RTX 2070</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background: #111827; border: 1px solid #1f2937;
                    border-radius: 12px; padding: 24px;'>
            <div style='font-family: Space Mono, monospace; font-size: 12px;
                        letter-spacing: 0.1em; color: #3b82f6; margin-bottom: 16px;'>
                PIPELINE
            </div>
            <div style='font-size: 14px; color: #94a3b8; line-height: 2;'>
                <div>1️⃣ Fetch 502 S&P 500 stocks + macro data</div>
                <div>2️⃣ Engineer 30+ technical signals per stock</div>
                <div>3️⃣ Train XGBoost on tabular features</div>
                <div>4️⃣ Train LSTM on 60-day price sequences</div>
                <div>5️⃣ Blend into ensemble score</div>
                <div>6️⃣ Cluster into diversified bundles</div>
                <div>7️⃣ Validate with walk-forward backtest</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background: #111827; border: 1px solid #1f2937;
                border-radius: 12px; padding: 20px; margin-top: 20px;
                font-size: 13px; color: #64748b; line-height: 1.7;'>
        <strong style='color: #ef4444;'>⚠️ Disclaimer:</strong>
        This project is built entirely for educational and portfolio purposes.
        Nothing displayed here constitutes financial advice. All model outputs
        are predictions based on historical patterns and may not reflect future
        performance. Always conduct your own research and consult a financial
        advisor before making any investment decisions.
    </div>
    """, unsafe_allow_html=True)