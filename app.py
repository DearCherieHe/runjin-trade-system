from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_sources.loaders import (
    load_crypto_prices,
    load_financials,
    load_kronos_forecast,
    load_paper_trades,
    load_prices,
    load_risk_rules,
    load_watchlist_notes,
)
from src.kline.indicators import add_indicators, classify_regime, relative_strength
from src.long_term.scoring import SCORE_COLUMNS, build_score_table
from src.long_term.thesis import get_ticker_profile, latest_financial_snapshot
from src.quant_bot.paper_trader import run_crypto_paper, run_us_stock_paper
from src.reports.weekly_report import build_weekly_report


ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="RunJin Trade System", layout="wide")


RUNJIN_CSS = """
<style>
:root {
  --rj-bg: #030504;
  --rj-bg-2: #070908;
  --rj-panel: #151716;
  --rj-panel-2: #202221;
  --rj-line: #303631;
  --rj-line-soft: rgba(255,255,255,0.08);
  --rj-text: #e8ebe8;
  --rj-muted: #9ca39d;
  --rj-dim: #6f776f;
  --rj-green: #2ed17c;
  --rj-green-soft: rgba(46,209,124,0.16);
  --rj-gold: #d8cb5f;
  --rj-pink: #ff78b7;
  --rj-purple: #a891ff;
  --rj-red: #ff5f64;
  --rj-mono: "SF Mono", "JetBrains Mono", "Menlo", monospace;
  --rj-sans: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

.stApp {
  background:
    linear-gradient(90deg, rgba(46,209,124,0.18) 0 2px, transparent 2px 100%),
    radial-gradient(circle at 16% 0%, rgba(46,209,124,0.10), transparent 28%),
    radial-gradient(circle at 88% 8%, rgba(216,203,95,0.06), transparent 24%),
    var(--rj-bg);
  color: var(--rj-text);
}

.block-container {
  max-width: 1480px;
  padding: 2.1rem 2.4rem 3.5rem;
}

[data-testid="stSidebar"] {
  background: #060706;
  border-right: 1px solid var(--rj-line);
}

[data-testid="stSidebar"] * {
  color: var(--rj-muted);
}

[data-testid="stSidebar"] [role="radiogroup"] label {
  border-radius: 6px;
  padding: 6px 8px;
  margin: 2px 0;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
  background: rgba(255,255,255,0.05);
}

h1, h2, h3 {
  color: var(--rj-text);
  letter-spacing: 0;
}

h1 {
  font-size: clamp(2rem, 4vw, 4.8rem);
  line-height: 0.96;
  font-weight: 850;
  margin-bottom: 0.35rem;
}

h2, h3 {
  font-weight: 760;
}

p, li, label, .stMarkdown {
  color: var(--rj-muted);
}

code {
  color: var(--rj-gold);
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 4px;
}

.runjin-hero {
  border: 1px solid var(--rj-line);
  background:
    linear-gradient(135deg, rgba(46,209,124,0.13), transparent 36%),
    linear-gradient(180deg, #101211 0%, #080a09 100%);
  padding: 28px 30px 24px;
  margin: 0 0 22px;
  box-shadow: 0 24px 80px rgba(0,0,0,0.32);
}

.runjin-kicker {
  display: flex;
  align-items: center;
  gap: 12px;
  color: var(--rj-green);
  font-family: var(--rj-mono);
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin-bottom: 12px;
}

.runjin-kicker::before {
  content: "";
  display: inline-block;
  width: 32px;
  height: 2px;
  background: var(--rj-green);
}

.runjin-title {
  font-size: clamp(2.5rem, 5.8vw, 6.5rem);
  line-height: 0.88;
  font-weight: 900;
  color: var(--rj-text);
  letter-spacing: -0.02em;
  text-wrap: pretty;
}

.runjin-subtitle {
  max-width: 920px;
  margin-top: 16px;
  color: #b8beb8;
  font-size: 15px;
  line-height: 1.65;
}

.runjin-ribbon {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 18px;
}

.runjin-ribbon span {
  border: 1px solid var(--rj-line);
  background: rgba(255,255,255,0.035);
  color: var(--rj-muted);
  font-family: var(--rj-mono);
  font-size: 11px;
  padding: 5px 8px;
  border-radius: 5px;
}

.runjin-section {
  color: var(--rj-green);
  font-family: var(--rj-mono);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  font-size: 12px;
  margin: 22px 0 8px;
}

.runjin-note {
  border-left: 2px solid var(--rj-green);
  background: rgba(46,209,124,0.07);
  padding: 12px 14px;
  color: #cbd1cb;
  font-family: var(--rj-mono);
  font-size: 12px;
  line-height: 1.6;
  margin: 12px 0 18px;
}

.runjin-thesis {
  border: 1px solid var(--rj-line);
  background: var(--rj-panel);
  padding: 18px;
  min-height: 170px;
}

.runjin-thesis h4 {
  color: var(--rj-green);
  font-family: var(--rj-mono);
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  margin: 0 0 9px;
}

.runjin-thesis p {
  margin: 0;
  color: #d5d9d4;
  line-height: 1.55;
}

.runjin-signal {
  border: 1px solid var(--rj-line);
  background: #101211;
  padding: 12px 14px;
  font-family: var(--rj-mono);
  color: var(--rj-text);
  margin: 12px 0 16px;
}

.runjin-signal b {
  color: var(--rj-green);
}

[data-testid="stMetric"] {
  background: linear-gradient(180deg, #171918, #101211);
  border: 1px solid var(--rj-line);
  border-radius: 0;
  padding: 15px 15px 14px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
}

[data-testid="stMetricLabel"] {
  color: var(--rj-muted);
  font-family: var(--rj-mono);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.07em;
}

[data-testid="stMetricValue"] {
  color: var(--rj-text);
  font-family: var(--rj-mono);
  font-size: 24px;
}

[data-testid="stMetricDelta"] {
  color: var(--rj-green);
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--rj-line);
  background: var(--rj-panel);
}

[data-testid="stCaptionContainer"] {
  color: var(--rj-dim);
  font-family: var(--rj-mono);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}

.stTabs [data-baseweb="tab-list"] {
  gap: 4px;
  border-bottom: 1px solid var(--rj-line);
}

.stTabs [data-baseweb="tab"] {
  background: transparent;
  color: var(--rj-muted);
  border-radius: 0;
  font-family: var(--rj-mono);
}

.stTabs [aria-selected="true"] {
  color: var(--rj-green) !important;
  border-bottom: 2px solid var(--rj-green);
}

.stSelectbox, .stMultiSelect, .stRadio {
  color: var(--rj-text);
}

[data-baseweb="select"] > div, [data-baseweb="input"] > div {
  background: #0e100f;
  border-color: var(--rj-line);
  border-radius: 5px;
}

[data-testid="stAlert"] {
  background: rgba(46,209,124,0.08);
  border: 1px solid rgba(46,209,124,0.30);
  color: var(--rj-text);
}

hr {
  border-color: var(--rj-line);
}

.js-plotly-plot {
  border: 1px solid var(--rj-line);
  background: var(--rj-panel);
}

@media (max-width: 800px) {
  .block-container { padding: 1.4rem 1rem 2.4rem; }
  .runjin-hero { padding: 20px; }
  .runjin-title { font-size: 3rem; }
}
</style>
"""


def inject_design_system():
    st.markdown(RUNJIN_CSS, unsafe_allow_html=True)


def page_header(kicker: str, title: str, subtitle: str, ribbons=None):
    ribbon_html = ""
    if ribbons:
        ribbon_html = '<div class="runjin-ribbon">' + "".join(f"<span>{item}</span>" for item in ribbons) + "</div>"
    st.markdown(
        f"""
        <div class="runjin-hero">
          <div class="runjin-kicker">{kicker}</div>
          <div class="runjin-title">{title}</div>
          <div class="runjin-subtitle">{subtitle}</div>
          {ribbon_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_label(text: str):
    st.markdown(f'<div class="runjin-section">{text}</div>', unsafe_allow_html=True)


@st.cache_data
def load_all_data():
    prices = load_prices()
    crypto = load_crypto_prices()
    financials = load_financials()
    notes = load_watchlist_notes()
    forecasts = load_kronos_forecast()
    paper_trades = load_paper_trades()
    risk_rules = load_risk_rules()
    scored = build_score_table(notes)
    return prices, crypto, financials, scored, forecasts, paper_trades, risk_rules


def fmt_int(value):
    return f"{value:,.0f}"


def fmt_pct(value):
    return f"{value * 100:.0f}%"


def named_table(name, df):
    st.caption(f"TABLE / {name}")
    st.dataframe(df, use_container_width=True, hide_index=True)


def metric_row(label, value, help_text=None):
    st.metric(label, value, help=help_text)


def style_figure(fig, height=None):
    fig.update_layout(
        paper_bgcolor="#151716",
        plot_bgcolor="#151716",
        font=dict(color="#d7dbd5", family="SF Mono, Menlo, monospace", size=12),
        title=dict(font=dict(color="#e8ebe8", size=15), x=0.02),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(255,255,255,0)",
            font=dict(color="#9ca39d"),
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=48, r=28, t=52, b=44),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", linecolor="#303631", zerolinecolor="#303631"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", linecolor="#303631", zerolinecolor="#303631"),
    )
    if height:
        fig.update_layout(height=height)
    return fig


def plot_kline(df, title, forecast=None):
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["date"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="OHLC",
            increasing_line_color="#2ed17c",
            increasing_fillcolor="rgba(46,209,124,0.50)",
            decreasing_line_color="#ff5f64",
            decreasing_fillcolor="rgba(255,95,100,0.50)",
        )
    )
    if "ma20" in df:
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma20"], name="MA20", line=dict(width=1.5, color="#d8cb5f")))
    if "ma60" in df:
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma60"], name="MA60", line=dict(width=1.5, color="#a891ff")))
    if forecast is not None and not forecast.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast["date"],
                y=forecast["predicted_close"],
                name="Kronos-style forecast (research only)",
                line=dict(dash="dash", color="#ff78b7"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast["date"],
                y=forecast["upper_band"],
                name="Forecast upper",
                line=dict(width=0),
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast["date"],
                y=forecast["lower_band"],
                name="Forecast band",
                fill="tonexty",
                line=dict(width=0),
                opacity=0.18,
            )
        )
    fig.update_layout(title=title, xaxis_rangeslider_visible=False, height=470, legend_title="")
    style_figure(fig, 470)
    st.plotly_chart(fig, use_container_width=True)


def page_dashboard(prices, crypto, scored, risk_rules):
    page_header(
        "RunJin / Manifested Discipline",
        "润金交易系统",
        "A dark research cockpit for long-term compounding and short-term paper-trading discipline. The interface is designed as a daily operating room: narrative, risk, signal, and review stay visible without turning into noise.",
        ["金水相生", "Offline sample mode", "No leverage", "No real orders"],
    )

    nvda = prices.loc[prices["ticker"] == "NVDA"]
    btc = crypto.loc[crypto["symbol"] == "BTC-USD"]
    stock_bt, stock_metrics, _, stock_status, stock_reason = run_us_stock_paper(nvda, risk_rules)
    crypto_bt, crypto_metrics, _, crypto_status, crypto_reason = run_crypto_paper(btc, risk_rules)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Watchlist", fmt_int(len(scored)))
    col2.metric("Deep research", fmt_int((scored["bucket"] == "Deep research candidate").sum()))
    col3.metric("US paper bot", stock_status, stock_reason)
    col4.metric("Crypto paper bot", crypto_status, crypto_reason)

    section_label("Portfolio Simulation")
    equity = pd.DataFrame(
        {
            "date": stock_bt["date"],
            "US stock bot": stock_bt["equity"],
        }
    )
    crypto_equity = crypto_bt[["datetime", "equity"]].rename(columns={"datetime": "date", "equity": "Crypto bot"})
    merged = pd.merge_asof(
        equity.sort_values("date"),
        crypto_equity.sort_values("date"),
        on="date",
        direction="nearest",
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["US stock bot"], name="US stock bot", line=dict(color="#2ed17c")))
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["Crypto bot"], name="Crypto bot", line=dict(color="#d8cb5f")))
    fig.update_layout(title="Paper Equity Curves", yaxis_title="USD")
    style_figure(fig, 360)
    st.plotly_chart(fig, use_container_width=True)

    alerts = []
    for name, status, reason in [("US stock bot", stock_status, stock_reason), ("Crypto bot", crypto_status, crypto_reason)]:
        if status != "CONTINUE":
            alerts.append({"system": name, "status": status, "reason": reason})
    if alerts:
        named_table("Risk alerts", pd.DataFrame(alerts))
    else:
        st.markdown('<div class="runjin-note">Risk desk clear: no stop condition is triggered in the current sample simulation.</div>', unsafe_allow_html=True)

    named_table("Top long-term candidates", scored[["ticker", "company", "tags", "score_label", "bucket", "growth_evidence"]].head(8))


def page_watchlist(scored):
    page_header(
        "Long Book / Equity Myth to Proof",
        "长期观察池",
        "This desk ranks companies by narrative space, acceleration, margin quality, cash flow, value-chain position, disagreement, and valuation tolerance. The goal is to find stories already converting into financial proof.",
        ["Narrative", "Growth proof", "Invalidation"],
    )
    bucket_filter = st.multiselect("Bucket", sorted(scored["bucket"].unique()), default=list(sorted(scored["bucket"].unique())))
    view = scored.loc[scored["bucket"].isin(bucket_filter)].copy()
    named_table(
        "Ranked long-term observation pool",
        view[
            [
                "ticker",
                "company",
                "tags",
                "score_label",
                "bucket",
                "narrative",
                "growth_evidence",
                "invalidation",
            ]
        ],
    )

    section_label("Score Breakdown")
    selected = st.selectbox("Ticker", view["ticker"].tolist())
    row = view.loc[view["ticker"] == selected].iloc[0]
    score_df = pd.DataFrame({"component": SCORE_COLUMNS, "score": [row[col] for col in SCORE_COLUMNS]})
    fig = go.Figure(go.Bar(x=score_df["component"], y=score_df["score"], marker_color="#2ed17c"))
    fig.update_layout(title=f"{selected} component score", yaxis=dict(range=[0, 5]))
    style_figure(fig, 340)
    st.plotly_chart(fig, use_container_width=True)


def page_stock_detail(prices, financials, scored, forecasts):
    page_header(
        "Single Name / Thesis Ledger",
        "个股研究台",
        "One ticker at a time: price structure, financial trend, thesis, catalysts, invalidation, and buy plan sit in the same frame so the long-term decision is auditable.",
        ["K-line", "Financial trend", "Decision log"],
    )
    ticker = st.selectbox("Select ticker", scored["ticker"].tolist())
    profile, ticker_financials = get_ticker_profile(scored, financials, ticker)
    ticker_prices = add_indicators(prices.loc[prices["ticker"] == ticker].copy())
    forecast = forecasts.loc[forecasts["ticker"] == ticker]

    col1, col2, col3, col4 = st.columns(4)
    snapshot = latest_financial_snapshot(ticker_financials)
    col1.metric("Score", profile["score_label"])
    col2.metric("Revenue growth", fmt_pct(snapshot["revenue_growth_yoy"]))
    col3.metric("Gross margin", fmt_pct(snapshot["gross_margin"]))
    col4.metric("OCF margin", fmt_pct(snapshot["operating_cash_flow_margin"]))

    plot_kline(ticker_prices, f"{ticker} K-line", forecast)

    section_label("Thesis Ledger")
    thesis_cols = st.columns(2)
    with thesis_cols[0]:
        st.markdown(f'<div class="runjin-thesis"><h4>Narrative</h4><p>{profile["narrative"]}</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="runjin-thesis"><h4>Growth Evidence</h4><p>{profile["growth_evidence"]}</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="runjin-thesis"><h4>Catalysts</h4><p>{profile["catalysts"]}</p></div>', unsafe_allow_html=True)
    with thesis_cols[1]:
        st.markdown(f'<div class="runjin-thesis"><h4>Invalidation</h4><p>{profile["invalidation"]}</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="runjin-thesis"><h4>Buy Plan</h4><p>{profile["buy_plan"]}</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="runjin-thesis"><h4>Bucket</h4><p>{profile["bucket"]}</p></div>', unsafe_allow_html=True)

    fin_view = ticker_financials.copy()
    for col in ["revenue_growth_yoy", "gross_margin", "operating_cash_flow_margin", "net_income_margin"]:
        fin_view[col] = fin_view[col].map(fmt_pct)
    named_table("Financial trend", fin_view)


def page_kline_lab(prices, forecasts):
    page_header(
        "K-line Lab / Pattern Is Context",
        "K线解读台",
        "Indicators are treated as context, not prophecy. The Kronos-style line is a local research stub and remains visually separated from executable trading signals.",
        ["MA20 / MA60", "RSI / MACD", "Research-only forecast"],
    )
    tickers = sorted(prices["ticker"].unique())
    ticker = st.selectbox("Ticker", tickers, index=tickers.index("NVDA") if "NVDA" in tickers else 0)
    ticker_prices = add_indicators(prices.loc[prices["ticker"] == ticker].copy())
    forecast = forecasts.loc[forecasts["ticker"] == ticker]
    benchmark = prices.loc[prices["ticker"] == "TSM"].copy()
    rs = relative_strength(ticker_prices, benchmark) if ticker != "TSM" else pd.DataFrame()

    col1, col2, col3 = st.columns(3)
    col1.metric("Regime", classify_regime(ticker_prices))
    col2.metric("Latest RSI", f"{ticker_prices['rsi14'].dropna().iloc[-1]:.0f}")
    col3.metric("20D volatility", fmt_pct(ticker_prices["volatility_20"].dropna().iloc[-1]))

    st.markdown('<div class="runjin-note">Kronos-style forecast is a local research-only stub. It does not generate trade instructions.</div>', unsafe_allow_html=True)
    plot_kline(ticker_prices, f"{ticker} Indicators + Research Forecast", forecast)

    indicator_view = ticker_prices.tail(30)[["date", "close", "ma20", "ma60", "rsi14", "macd", "macd_signal", "bb_lower", "bb_upper"]].copy()
    named_table("Latest indicator values", indicator_view.round(2))

    if not rs.empty:
        fig = go.Figure(go.Scatter(x=rs["date"], y=rs["relative_strength"], name=f"{ticker} vs TSM"))
        fig.update_layout(title=f"Relative Strength: {ticker} vs TSM")
        style_figure(fig, 320)
        st.plotly_chart(fig, use_container_width=True)


def page_short_bot(prices, crypto, risk_rules, paper_trades):
    page_header(
        "Short Book / Small Cashflow Engine",
        "短线模拟机器人",
        "The short book is deliberately small: paper trading only, no leverage, no broker API, no exchange keys. The design makes stop status more important than return.",
        ["Dry-run", "Risk first", "Stop conditions visible"],
    )

    asset_class = st.radio("Asset class", ["US stock daily", "Crypto hourly"], horizontal=True)
    if asset_class == "US stock daily":
        ticker = st.selectbox("Ticker", sorted(prices["ticker"].unique()), index=0)
        data = prices.loc[prices["ticker"] == ticker].copy()
        bt, metrics, trades, status, reason = run_us_stock_paper(data, risk_rules)
        time_col = "date"
    else:
        ticker = st.selectbox("Symbol", sorted(crypto["symbol"].unique()), index=0)
        data = crypto.loc[crypto["symbol"] == ticker].copy()
        bt, metrics, trades, status, reason = run_crypto_paper(data, risk_rules)
        time_col = "datetime"

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Status", status, reason)
    col2.metric("Return", fmt_pct(metrics["total_return"]))
    col3.metric("Max drawdown", fmt_pct(metrics["max_drawdown"]))
    col4.metric("Win rate", fmt_pct(metrics["win_rate"]))
    col5.metric("Trades", fmt_int(metrics["trade_count"]))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt[time_col], y=bt["equity"], name="Equity", line=dict(color="#2ed17c")))
    fig.update_layout(title=f"{ticker} Paper Equity", yaxis_title="USD")
    style_figure(fig, 360)
    st.plotly_chart(fig, use_container_width=True)

    latest = bt.tail(1).iloc[0]
    st.markdown(
        f'<div class="runjin-signal"><b>Current signal</b> {int(latest["signal"])} &nbsp; / &nbsp; '
        f'<b>Reason</b> {latest["signal_reason"]} &nbsp; / &nbsp; '
        f'<b>Position</b> {latest["position"]:.2%}</div>',
        unsafe_allow_html=True,
    )

    named_table("Recent strategy trades", trades)
    named_table("Sample paper trade ledger", paper_trades)


def page_weekly_report(prices, crypto, scored, risk_rules):
    page_header(
        "Weekly Review / Keep the System Honest",
        "周度复盘",
        "A compact operating memo: long-term candidates, bot state, drawdown, and whether to continue, pause, or review. The ritual matters as much as the signal.",
        ["Continue", "Pause", "Review"],
    )
    stock_bt, stock_metrics, _, stock_status, stock_reason = run_us_stock_paper(prices.loc[prices["ticker"] == "NVDA"].copy(), risk_rules)
    crypto_bt, crypto_metrics, _, crypto_status, crypto_reason = run_crypto_paper(crypto.loc[crypto["symbol"] == "BTC-USD"].copy(), risk_rules)
    report = build_weekly_report(
        scored,
        {
            "US stock daily trend": {"metrics": stock_metrics, "status": stock_status, "reason": stock_reason},
            "Crypto hourly mean reversion": {"metrics": crypto_metrics, "status": crypto_status, "reason": crypto_reason},
        },
    )
    st.markdown(report)


def main():
    inject_design_system()
    prices, crypto, financials, scored, forecasts, paper_trades, risk_rules = load_all_data()
    st.sidebar.markdown(
        """
        <div style="padding: 10px 4px 18px;">
          <div style="font-size: 24px; font-weight: 850; color: #e8ebe8; line-height: 1;">润金</div>
          <div style="font-family: SF Mono, Menlo, monospace; color: #2ed17c; font-size: 11px; letter-spacing: .08em; margin-top: 8px;">RUNJIN TRADE SYSTEM</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    page = st.sidebar.radio(
        "Workspace",
        ["Dashboard", "Long Watchlist", "Stock Detail", "K-line Lab", "Short Bot", "Weekly Report"],
    )
    st.sidebar.caption("MODE / offline sample data")
    st.sidebar.caption("BOUNDARY / V0.1 never places real orders.")

    if page == "Dashboard":
        page_dashboard(prices, crypto, scored, risk_rules)
    elif page == "Long Watchlist":
        page_watchlist(scored)
    elif page == "Stock Detail":
        page_stock_detail(prices, financials, scored, forecasts)
    elif page == "K-line Lab":
        page_kline_lab(prices, forecasts)
    elif page == "Short Bot":
        page_short_bot(prices, crypto, risk_rules, paper_trades)
    elif page == "Weekly Report":
        page_weekly_report(prices, crypto, scored, risk_rules)


if __name__ == "__main__":
    main()
