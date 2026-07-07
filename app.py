from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_sources.loaders import (
    get_data_source_status,
    load_crypto_prices,
    load_financials,
    load_kronos_forecast,
    load_market_universe,
    load_prices,
    load_risk_rules,
    load_watchlist_notes,
)
from src.data_sources.finance_mcp import load_finance_mcp_capabilities, load_finance_mcp_research
from src.data_sources.market_universe import ensure_market_universe_columns
from src.backtest_lab.engine import (
    BACKTEST_SYSTEM_MAP,
    BacktestEngineUnavailable,
    DEFAULT_STRATEGY_SPEC,
    DEFAULT_PORTFOLIO_SPEC,
    EXAMPLE_STRATEGY_SPECS,
    EXAMPLE_PORTFOLIO_SPECS,
    run_portfolio_backtest,
    run_strategy_backtest,
)
from src.backtest_lab.batch import BATCH_STRATEGY_GRIDS, run_batch_backtest
from src.kline.indicators import add_indicators, classify_regime, relative_strength
from src.kline.abu_research import atr_research, gap_analysis, rolling_correlation_matrix, similar_paths
from src.kline.timeframes import apply_asof, coverage_summary, resample_ohlcv
from src.data_sources.live_sources import fetch_yfinance_prices
from src.long_term.scoring import SCORE_COLUMNS, build_score_table
from src.long_term.thesis import get_ticker_profile, latest_financial_snapshot
from src.quant_bot.paper_trader import run_crypto_paper, run_us_stock_paper
from src.reports.weekly_report import build_weekly_report
from src.trade_skills.capital_rotation import cohort_rotation
from src.trade_skills.intraday import intraday_signal_pack
from src.trade_skills.journal import build_markdown_note, default_deep_dive_sections, list_journal_entries
from src.trade_skills.sepa import sepa_dashboard, sepa_entry_plan


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
    linear-gradient(90deg, rgba(46,209,124,0.11), transparent 34%),
    linear-gradient(180deg, #101211 0%, #090b0a 100%);
  padding: 24px 28px 22px;
  margin: 0 0 20px;
  box-shadow: 0 18px 60px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.035);
  position: relative;
  overflow: hidden;
}

.runjin-hero::after {
  content: "";
  position: absolute;
  right: 26px;
  top: 24px;
  width: 140px;
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(46,209,124,0.55), transparent);
}

.runjin-kicker {
  display: flex;
  align-items: center;
  gap: 12px;
  color: var(--rj-green);
  font-family: var(--rj-mono);
  font-size: 11px;
  letter-spacing: 0.11em;
  text-transform: uppercase;
  margin-bottom: 10px;
  opacity: 0.92;
}

.runjin-kicker::before {
  content: "";
  display: inline-block;
  width: 32px;
  height: 2px;
  background: var(--rj-green);
}

.runjin-title {
  font-size: clamp(2.2rem, 4.4vw, 4.9rem);
  line-height: 0.98;
  font-weight: 820;
  color: var(--rj-text);
  letter-spacing: -0.012em;
  text-wrap: pretty;
  max-width: 980px;
}

.runjin-subtitle {
  max-width: 1080px;
  margin-top: 14px;
  color: #aeb6ae;
  font-size: 14px;
  line-height: 1.7;
  text-wrap: pretty;
}

.runjin-ribbon {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 16px;
}

.runjin-ribbon span {
  border: 1px solid var(--rj-line);
  background: rgba(255,255,255,0.025);
  color: #a5ada5;
  font-family: var(--rj-mono);
  font-size: 10.5px;
  padding: 5px 9px;
  border-radius: 6px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.025);
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
  .runjin-hero { padding: 18px; }
  .runjin-title { font-size: 2.55rem; }
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
def load_all_data(data_mode):
    prices = load_prices(data_mode=data_mode)
    crypto = load_crypto_prices(data_mode=data_mode)
    financials = load_financials(data_mode=data_mode)
    market_universe = load_market_universe(data_mode=data_mode)
    notes = load_watchlist_notes()
    forecasts = load_kronos_forecast(data_mode=data_mode)
    finance_research, finance_mcp_status = load_finance_mcp_research(data_mode=data_mode)
    risk_rules = load_risk_rules()
    scored = build_score_table(notes)
    source_status = get_data_source_status()
    source_status["finance_mcp"] = finance_mcp_status
    return prices, crypto, financials, market_universe, scored, forecasts, finance_research, risk_rules, source_status


def fmt_int(value):
    return f"{value:,.0f}"


def fmt_pct(value):
    return f"{value * 100:.0f}%"


def fmt_number(value, digits=1, suffix=""):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):,.{digits}f}{suffix}"
    except (TypeError, ValueError):
        return "N/A"


def coerce_datetime_key(df, column="date"):
    normalized = df.copy()
    normalized[column] = pd.to_datetime(normalized[column], errors="coerce")
    if getattr(normalized[column].dt, "tz", None) is not None:
        normalized[column] = normalized[column].dt.tz_convert(None)
    normalized[column] = normalized[column].astype("datetime64[ns]")
    return normalized.dropna(subset=[column]).sort_values(column).reset_index(drop=True)


def named_table(name, df):
    st.caption(f"TABLE / {name}")
    st.dataframe(df, use_container_width=True, hide_index=True)


def metric_row(label, value, help_text=None):
    st.metric(label, value, help=help_text)


def has_crypto_symbol(crypto, symbol="BTC-USD"):
    return not crypto.empty and "symbol" in crypto.columns and not crypto.loc[crypto["symbol"] == symbol].empty


def disabled_bot_metrics():
    return {
        "total_return": 0,
        "max_drawdown": 0,
        "win_rate": 0,
        "profit_factor": 0,
        "trade_count": 0,
    }


def style_figure(fig, height=None, time_axis=False):
    range_selector = None
    range_slider = None
    bottom_margin = 44
    if time_axis:
        range_selector = dict(
            buttons=[
                dict(count=7, label="1W", step="day", stepmode="backward"),
                dict(count=1, label="1M", step="month", stepmode="backward"),
                dict(count=3, label="1Q", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1Y", step="year", stepmode="backward"),
                dict(label="ALL", step="all"),
            ],
            bgcolor="#101211",
            activecolor="#2ed17c",
            bordercolor="#303631",
            borderwidth=1,
            font=dict(color="#d7dbd5", size=11),
            x=0,
            y=1.12,
            xanchor="left",
            yanchor="top",
        )
        range_slider = dict(
            visible=True,
            bgcolor="#101211",
            bordercolor="#303631",
            borderwidth=1,
            thickness=0.08,
        )
        bottom_margin = 72
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
        margin=dict(l=48, r=28, t=64 if time_axis else 52, b=bottom_margin),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            linecolor="#303631",
            zerolinecolor="#303631",
            rangeselector=range_selector,
            rangeslider=range_slider,
        ),
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
    fig.update_layout(title=title, height=470, legend_title="")
    style_figure(fig, 470, time_axis=True)
    st.plotly_chart(fig, use_container_width=True)


def forecast_controls(key_prefix):
    return st.checkbox(
        "Show research forecast",
        value=False,
        key=f"{key_prefix}_show_forecast",
        help="Off by default. Local sample forecasts are disabled in live-data mode.",
    )


def filter_forecast_for_chart(forecast, kline_df, max_price_ratio=2.5):
    if forecast is None or forecast.empty or kline_df.empty:
        return pd.DataFrame()
    latest_close = float(kline_df["close"].dropna().iloc[-1])
    if latest_close <= 0:
        return pd.DataFrame()
    filtered = forecast.copy()
    for col in ["predicted_close", "lower_band", "upper_band"]:
        filtered = filtered.loc[filtered[col].between(latest_close / max_price_ratio, latest_close * max_price_ratio)]
    return filtered.reset_index(drop=True)


def build_ticker_kline(prices, ticker, timeframe, as_of=None, data_mode="sample"):
    base = prices.loc[prices["ticker"] == ticker].copy()
    source_note = "watchlist daily source"
    if data_mode in {"live_auto", "live"} and (timeframe == "1H" or base.empty):
        try:
            interval = "1h" if timeframe == "1H" else "1d"
            period = "730d" if timeframe == "1H" else "max"
            base = fetch_yfinance_prices([ticker], period=period, interval=interval)
            source_note = f"yfinance {interval} on-demand; free-source coverage can vary by market"
        except Exception as exc:
            if data_mode == "live":
                raise
            source_note = f"hourly live unavailable, using daily source: {exc}"
    if base.empty:
        return base, f"no OHLCV rows for {ticker}; configure a live source or choose a seeded watchlist ticker"
    kline = resample_ohlcv(base, timeframe, "date")
    kline = apply_asof(kline, as_of)
    return kline, source_note


def kline_controls(prices, ticker, key_prefix, data_mode):
    timeframe = st.selectbox(
        "K-line timeframe",
        ["1H", "1D", "1W", "1M", "1Q", "1Y"],
        index=1,
        key=f"{key_prefix}_timeframe",
        help="Hourly uses live on-demand data when available. Weekly/monthly/quarterly/yearly are OHLCV resamples from historical daily bars.",
    )
    ticker_prices = prices.loc[prices["ticker"] == ticker].copy()
    ticker_prices["date"] = pd.to_datetime(ticker_prices["date"], errors="coerce")
    min_date = ticker_prices["date"].min().date()
    max_date = ticker_prices["date"].max().date()
    as_of = st.date_input(
        "Replay as of",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        key=f"{key_prefix}_asof",
        help="Use this date as the current moment. The chart only shows bars available up to that date.",
    )
    st.caption(f"Replay range: {min_date} -> {max_date} / data mode: {data_mode}")
    show_forecast = forecast_controls(key_prefix)
    return timeframe, as_of, show_forecast


def page_dashboard(prices, crypto, market_universe, scored, finance_research, risk_rules, source_status):
    page_header(
        "RunJin / Manifested Discipline",
        "润金交易系统",
        "A dark research cockpit for long-term compounding and short-term paper-trading discipline. The interface is designed as a daily operating room: narrative, risk, signal, and review stay visible without turning into noise.",
        ["金水相生", "Live data mode", "No leverage", "No real orders"],
    )
    status_df = pd.DataFrame(
        [
            {"layer": key, **value}
            for key, value in source_status.items()
        ]
    )
    named_table("Data source status", status_df)

    nvda = prices.loc[prices["ticker"] == "NVDA"]
    stock_bt, stock_metrics, _, stock_status, stock_reason = run_us_stock_paper(nvda, risk_rules)
    if has_crypto_symbol(crypto, "BTC-USD"):
        btc = crypto.loc[crypto["symbol"] == "BTC-USD"]
        crypto_bt, crypto_metrics, _, crypto_status, crypto_reason = run_crypto_paper(btc, risk_rules)
    else:
        crypto_bt = pd.DataFrame()
        crypto_metrics = disabled_bot_metrics()
        crypto_status = "PAUSE"
        crypto_reason = "Crypto live source unavailable"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Market universe", fmt_int(len(market_universe)))
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
    equity = coerce_datetime_key(equity, "date")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity["date"], y=equity["US stock bot"], name="US stock bot", line=dict(color="#2ed17c")))
    if not crypto_bt.empty:
        crypto_equity = crypto_bt[["datetime", "equity"]].rename(columns={"datetime": "date", "equity": "Crypto bot"})
        crypto_equity = coerce_datetime_key(crypto_equity, "date")
        fig.add_trace(go.Scatter(x=crypto_equity["date"], y=crypto_equity["Crypto bot"], name="Crypto bot", line=dict(color="#d8cb5f")))
    fig.update_layout(title="Paper Equity Curves", yaxis_title="USD")
    style_figure(fig, 360, time_axis=True)
    st.plotly_chart(fig, use_container_width=True)

    alerts = []
    for name, status, reason in [("US stock bot", stock_status, stock_reason), ("Crypto bot", crypto_status, crypto_reason)]:
        if status != "CONTINUE":
            alerts.append({"system": name, "status": status, "reason": reason})
    if alerts:
        named_table("Risk alerts", pd.DataFrame(alerts))
    else:
        st.markdown('<div class="runjin-note">Risk desk clear: no stop condition is triggered in the current live-data simulation.</div>', unsafe_allow_html=True)

    high_priority = finance_research.loc[finance_research["importance"] >= 4].head(6)
    if not high_priority.empty:
        named_table("FinanceMCP high-priority research inputs", high_priority)

    named_table("Top long-term candidates", scored[["ticker", "company", "tags", "score_label", "bucket", "growth_evidence"]].head(8))


def page_market_universe(market_universe):
    page_header(
        "Market Universe / Broad Coverage",
        "全市场股票池",
        "The tradable stock pool covers US stocks, A-shares, Hong Kong stocks, and Singapore stocks, then excludes micro caps below USD 300M equivalent before research or chart selection.",
        ["US", "A-share", "HK", "SG", "USD 300M floor"],
    )
    if market_universe.empty:
        st.warning("No market universe rows are available. Configure full listing CSV files in configs/live_sources.yaml.")
        return
    market_universe = ensure_market_universe_columns(market_universe)

    markets = sorted(market_universe["market"].dropna().unique())
    exchanges = sorted(market_universe["exchange"].dropna().unique())
    currencies = sorted(market_universe["currency"].dropna().unique())
    col1, col2, col3, col4 = st.columns([1, 1.2, 1, 1.2])
    selected_markets = col1.multiselect("Market", markets, default=markets)
    selected_exchanges = col2.multiselect("Exchange", exchanges, default=exchanges)
    selected_currencies = col3.multiselect("Currency", currencies, default=currencies)
    min_cap = col4.number_input("Min market cap USD", min_value=0, value=300_000_000, step=50_000_000)

    view = market_universe.loc[
        market_universe["market"].isin(selected_markets)
        & market_universe["exchange"].isin(selected_exchanges)
        & market_universe["currency"].isin(selected_currencies)
        & (market_universe["market_cap_usd"] >= min_cap)
    ].copy()
    market_counts = view.groupby("market_group", as_index=False)["ticker"].count().rename(columns={"ticker": "count"})
    cap_by_market = view.groupby("market_group", as_index=False)["market_cap_usd"].sum()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Symbols", fmt_int(len(view)))
    c2.metric("Markets", fmt_int(view["market_group"].nunique()))
    c3.metric("Exchanges", fmt_int(view["exchange"].nunique()))
    c4.metric("Median cap", f"${fmt_int(view['market_cap_usd'].median())}" if not view.empty else "$0")

    if not market_counts.empty:
        fig = go.Figure(go.Bar(x=market_counts["market_group"], y=market_counts["count"], marker_color="#2ed17c"))
        fig.update_layout(title="Universe count by market", yaxis_title="Symbols")
        style_figure(fig, 320)
        st.plotly_chart(fig, use_container_width=True)

    if not cap_by_market.empty:
        named_table("Market cap by market", cap_by_market.assign(market_cap_usd=cap_by_market["market_cap_usd"].map(lambda value: f"${fmt_int(value)}")))

    table_columns = [
        "ticker",
        "yahoo_ticker",
        "company",
        "market_group",
        "market_rank",
        "market",
        "exchange",
        "currency",
        "market_cap_usd",
        "source",
    ]
    table = view[[col for col in table_columns if col in view.columns]].copy()
    table["market_cap_usd"] = table["market_cap_usd"].map(lambda value: f"${fmt_int(value)}")
    named_table("Filtered multi-market stock universe", table)


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


def page_finance_mcp_radar(finance_research, scored, source_status):
    page_header(
        "Finance MCP / Data Edge Radar",
        "全市场情报雷达",
        "A research-only layer inspired by FinanceMCP: market news, macro calendar, money flow, index membership, fundamentals, China-market candidates, crypto context, and technical signals are collected before they become trade decisions.",
        ["Research only", "Multi-source", "No auto orders"],
    )

    status = source_status.get("finance_mcp", {})
    st.markdown(
        f'<div class="runjin-note">FinanceMCP adapter: {status.get("mode", "unknown")} / {status.get("source", "unknown")} / {status.get("message", "")}</div>',
        unsafe_allow_html=True,
    )

    capabilities = load_finance_mcp_capabilities()
    named_table("FinanceMCP capability map", capabilities)

    domains = sorted(finance_research["domain"].dropna().unique())
    tickers = ["ALL"] + sorted(finance_research["symbol"].dropna().unique())
    col1, col2, col3 = st.columns([1.2, 1.1, 1])
    selected_domains = col1.multiselect("Research domains", domains, default=domains)
    selected_symbol = col2.selectbox("Symbol", tickers)
    min_importance = col3.slider("Min importance", 1, 5, 3)

    view = finance_research.loc[
        finance_research["domain"].isin(selected_domains)
        & (finance_research["importance"] >= min_importance)
    ].copy()
    if selected_symbol != "ALL":
        view = view.loc[view["symbol"] == selected_symbol]

    watch_symbols = set(scored["ticker"].tolist()) | {"BTC-USD", "ETH-USD", "MARKET"}
    watch_hits = view.loc[view["symbol"].isin(watch_symbols)].copy()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Signals", fmt_int(len(view)))
    col2.metric("Watchlist hits", fmt_int(len(watch_hits)))
    col3.metric("High importance", fmt_int((view["importance"] >= 4).sum()))
    col4.metric("Domains", fmt_int(view["domain"].nunique()))

    if not view.empty:
        domain_counts = view.groupby("domain", as_index=False)["importance"].sum().sort_values("importance", ascending=False)
        fig = go.Figure(go.Bar(x=domain_counts["domain"], y=domain_counts["importance"], marker_color="#2ed17c"))
        fig.update_layout(title="Research pressure by domain", yaxis_title="Importance score")
        style_figure(fig, 340)
        st.plotly_chart(fig, use_container_width=True)

    named_table("Filtered research radar", view)
    if not watch_hits.empty:
        named_table("Watchlist-linked signals", watch_hits)


def page_stock_detail(prices, financials, scored, forecasts, data_mode):
    page_header(
        "Single Name / Thesis Ledger",
        "个股研究台",
        "One ticker at a time: price structure, financial trend, thesis, catalysts, invalidation, and buy plan sit in the same frame so the long-term decision is auditable.",
        ["K-line", "Financial trend", "Decision log"],
    )
    ticker = st.selectbox("Select ticker", scored["ticker"].tolist())
    profile, ticker_financials = get_ticker_profile(scored, financials, ticker)
    timeframe, as_of, show_forecast = kline_controls(prices, ticker, "detail", data_mode)
    raw_kline, source_note = build_ticker_kline(prices, ticker, timeframe, as_of, data_mode)
    ticker_prices = add_indicators(raw_kline)
    use_forecast = show_forecast and timeframe == "1D" and as_of >= prices.loc[prices["ticker"] == ticker, "date"].max().date()
    forecast = filter_forecast_for_chart(forecasts.loc[forecasts["ticker"] == ticker], ticker_prices) if use_forecast else pd.DataFrame()

    col1, col2, col3, col4 = st.columns(4)
    snapshot = latest_financial_snapshot(ticker_financials)
    col1.metric("Score", profile["score_label"])
    col2.metric("Revenue growth", fmt_pct(snapshot["revenue_growth_yoy"]))
    col3.metric("Gross margin", fmt_pct(snapshot["gross_margin"]))
    col4.metric("OCF margin", fmt_pct(snapshot["operating_cash_flow_margin"]))

    st.markdown(
        f'<div class="runjin-note">K-line coverage: {coverage_summary(ticker_prices)} / {source_note}. Forecast overlay is off by default and filtered for price-scale sanity.</div>',
        unsafe_allow_html=True,
    )
    plot_kline(ticker_prices, f"{ticker} {timeframe} K-line / Replay as of {as_of}", forecast)

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


def page_kline_lab(prices, forecasts, market_universe, data_mode):
    page_header(
        "K-line Lab / Pattern Is Context",
        "K线解读台",
        "Indicators are treated as context, not prophecy. The Kronos-style line is a local research stub and remains visually separated from executable trading signals.",
        ["MA20 / MA60", "RSI / MACD", "ABU-style research"],
    )
    price_tickers = sorted(prices["ticker"].unique())
    universe_tickers = sorted(market_universe["yahoo_ticker"].dropna().unique()) if not market_universe.empty else []
    tickers = sorted(set(price_tickers) | set(universe_tickers))
    ticker = st.selectbox("Ticker", tickers, index=tickers.index("NVDA") if "NVDA" in tickers else 0)
    ticker_has_loaded_prices = ticker in set(price_tickers)
    control_prices = prices
    if not ticker_has_loaded_prices and data_mode in {"live_auto", "live"}:
        try:
            control_prices = fetch_yfinance_prices([ticker], period="1y", interval="1d")
        except Exception as exc:
            if data_mode == "live":
                raise
            st.warning(f"On-demand daily data unavailable for {ticker}: {exc}")
            return
    if control_prices.loc[control_prices["ticker"] == ticker].empty:
        st.warning(f"No daily rows are available for {ticker}.")
        return
    timeframe, as_of, show_forecast = kline_controls(control_prices, ticker, "lab", data_mode)
    raw_kline, source_note = build_ticker_kline(prices, ticker, timeframe, as_of, data_mode)
    if raw_kline.empty:
        st.warning(source_note)
        return
    ticker_prices = add_indicators(raw_kline)
    latest_source_date = control_prices.loc[control_prices["ticker"] == ticker, "date"].max()
    use_forecast = (
        show_forecast
        and timeframe == "1D"
        and pd.notna(latest_source_date)
        and as_of >= pd.to_datetime(latest_source_date).date()
        and ticker in set(forecasts["ticker"])
    )
    forecast = filter_forecast_for_chart(forecasts.loc[forecasts["ticker"] == ticker], ticker_prices) if use_forecast else pd.DataFrame()
    benchmark_raw = prices.loc[prices["ticker"] == "TSM"].copy()
    benchmark = apply_asof(resample_ohlcv(benchmark_raw, timeframe, "date"), as_of) if not benchmark_raw.empty else pd.DataFrame()
    rs = relative_strength(ticker_prices, benchmark) if ticker != "TSM" and not benchmark.empty else pd.DataFrame()

    latest_indicator = ticker_prices.dropna().tail(1)
    kdj_state = "Insufficient data"
    if not latest_indicator.empty:
        latest_row = latest_indicator.iloc[0]
        if latest_row["kdj_j"] > 100:
            kdj_state = "KDJ extended"
        elif latest_row["kdj_j"] < 0:
            kdj_state = "KDJ washed out"
        elif latest_row["kdj_k"] > latest_row["kdj_d"]:
            kdj_state = "KDJ improving"
        else:
            kdj_state = "KDJ cooling"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Regime", classify_regime(ticker_prices))
    latest_rsi = ticker_prices["rsi14"].dropna()
    latest_volatility = ticker_prices["volatility_20"].dropna()
    col2.metric("Latest RSI", f"{latest_rsi.iloc[-1]:.0f}" if not latest_rsi.empty else "N/A")
    col3.metric("20D volatility", fmt_pct(latest_volatility.iloc[-1]) if not latest_volatility.empty else "N/A")
    col4.metric("KDJ state", kdj_state)

    st.markdown(
        f'<div class="runjin-note">K-line coverage: {coverage_summary(ticker_prices)} / {source_note}. Forecast overlay is off by default, research-only, hidden during replay, and filtered for price-scale sanity.</div>',
        unsafe_allow_html=True,
    )
    plot_kline(ticker_prices, f"{ticker} {timeframe} Indicators / Replay as of {as_of}", forecast)

    indicator_view = ticker_prices.tail(30)[
        ["date", "close", "ma20", "ma60", "rsi14", "macd", "macd_signal", "bb_lower", "bb_upper", "kdj_k", "kdj_d", "kdj_j"]
    ].copy()
    named_table("Latest indicator values", indicator_view.round(2))

    if not rs.empty:
        fig = go.Figure(go.Scatter(x=rs["date"], y=rs["relative_strength"], name=f"{ticker} vs TSM"))
        fig.update_layout(title=f"Relative Strength: {ticker} vs TSM")
        style_figure(fig, 320, time_axis=True)
        st.plotly_chart(fig, use_container_width=True)

    section_label("ABU-style Research Context")
    st.markdown(
        '<div class="runjin-note">Gap, ATR, correlation, and similar-path diagnostics are research-only context rebuilt for RunJin. They do not create executable trading instructions.</div>',
        unsafe_allow_html=True,
    )
    gap_view = gap_analysis(ticker_prices).tail(12)
    atr_view = atr_research(ticker_prices).tail(30)
    corr_candidates = [item for item in ["NVDA", "TSLA", "AMD", "AVGO", "TSM", "PLTR"] if item in set(prices["ticker"])]
    corr_view = rolling_correlation_matrix(prices, corr_candidates, window=60)
    similar_view = similar_paths(prices, ticker if ticker in set(prices["ticker"]) else "NVDA", corr_candidates, window=60)
    col_gap, col_atr = st.columns(2)
    with col_gap:
        named_table("Gap signals", gap_view.round(4) if not gap_view.empty else pd.DataFrame([{"note": "No significant gaps in current window"}]))
    with col_atr:
        named_table("ATR research", atr_view[["date", "close", "atr21", "atr_pct"]].round(4) if not atr_view.empty else pd.DataFrame())
    col_corr, col_similar = st.columns(2)
    with col_corr:
        named_table("Rolling correlation matrix", corr_view.round(3) if not corr_view.empty else pd.DataFrame([{"note": "Not enough overlapping symbols"}]))
    with col_similar:
        named_table("Similar recent paths", similar_view.round(3) if not similar_view.empty else pd.DataFrame([{"note": "Not enough comparable history"}]))


def page_symbol_cockpit(prices, crypto, financials, scored, forecasts, data_mode):
    page_header(
        "Trade Skills / Symbol Cockpit",
        "个股驾驶舱",
        "A one-name operating page inspired by trade-skills: thesis, SEPA trend template, K-line replay, intraday scenario, and journal prompts stay in one place so each decision has context.",
        ["Prediction", "Environment", "Review", "Note", "Research only"],
    )
    ticker = st.selectbox("Symbol", scored["ticker"].tolist(), key="cockpit_ticker")
    profile, ticker_financials = get_ticker_profile(scored, financials, ticker)
    timeframe, as_of, show_forecast = kline_controls(prices, ticker, "cockpit", data_mode)
    raw_kline, source_note = build_ticker_kline(prices, ticker, timeframe, as_of, data_mode)
    if raw_kline.empty:
        st.warning(source_note)
        return
    ticker_prices = add_indicators(raw_kline)
    use_forecast = show_forecast and timeframe == "1D" and ticker in set(forecasts["ticker"])
    forecast = filter_forecast_for_chart(forecasts.loc[forecasts["ticker"] == ticker], ticker_prices) if use_forecast else pd.DataFrame()

    tabs = st.tabs(["Prediction", "Environment", "Review", "Note"])
    with tabs[0]:
        plot_kline(ticker_prices, f"{ticker} cockpit K-line / {timeframe} / Replay as of {as_of}", forecast)
        benchmark = prices.loc[prices["ticker"] == "NVDA"].copy()
        sepa = sepa_dashboard(ticker_prices, benchmark if ticker != "NVDA" else pd.DataFrame())
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("SEPA verdict", sepa["verdict"])
        latest_rsi = ticker_prices["rsi14"].dropna()
        c2.metric("RSI", f"{latest_rsi.iloc[-1]:.0f}" if not latest_rsi.empty else "N/A")
        c3.metric("Regime", classify_regime(ticker_prices))
        c4.metric("Bars", fmt_int(len(ticker_prices)))
        named_table("SEPA trend checks", sepa["checks"] if not sepa["checks"].empty else sepa["summary"])
    with tabs[1]:
        snapshot = latest_financial_snapshot(ticker_financials)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Score", profile["score_label"])
        col2.metric("Revenue growth", fmt_pct(snapshot["revenue_growth_yoy"]))
        col3.metric("Gross margin", fmt_pct(snapshot["gross_margin"]))
        col4.metric("OCF margin", fmt_pct(snapshot["operating_cash_flow_margin"]))
        named_table("Financial proof", ticker_financials)
        st.markdown(f'<div class="runjin-note">Data coverage: {coverage_summary(ticker_prices)} / {source_note}</div>', unsafe_allow_html=True)
    with tabs[2]:
        st.markdown(f'<div class="runjin-thesis"><h4>Narrative</h4><p>{profile["narrative"]}</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="runjin-thesis"><h4>Invalidation</h4><p>{profile["invalidation"]}</p></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="runjin-thesis"><h4>Buy Plan</h4><p>{profile["buy_plan"]}</p></div>', unsafe_allow_html=True)
    with tabs[3]:
        sections = pd.DataFrame(default_deep_dive_sections(ticker, profile))
        named_table("Deep-dive journal prompts", sections)
        st.text_area("Markdown note draft", value=build_markdown_note(ticker, sections), height=360, key="cockpit_note_draft")


def page_sepa_lab(prices):
    page_header(
        "Trade Skills / SEPA Lab",
        "SEPA 趋势模板",
        "A Minervini-style trend template rebuilt locally for RunJin. It checks 50/150/200MA structure, 52-week location, volume context, and relative strength before a stock becomes a serious breakout candidate.",
        ["50MA", "150MA", "200MA", "52W", "Relative strength"],
    )
    ticker = st.selectbox("Ticker", sorted(prices["ticker"].unique()), index=0, key="sepa_ticker")
    benchmark_ticker = st.selectbox("Benchmark", sorted(prices["ticker"].unique()), index=0, key="sepa_benchmark")
    data = prices.loc[prices["ticker"] == ticker].copy()
    benchmark = prices.loc[prices["ticker"] == benchmark_ticker].copy() if benchmark_ticker != ticker else pd.DataFrame()
    result = sepa_dashboard(data, benchmark)
    levels = result["levels"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Verdict", result["verdict"])
    col2.metric("Passes", fmt_int((result["checks"]["status"] == "pass").sum()) if not result["checks"].empty else "N/A")
    col3.metric("Fails", fmt_int((result["checks"]["status"] == "fail").sum()) if not result["checks"].empty else "N/A")
    col4.metric("Bars", fmt_int(len(levels)))
    if not levels.empty:
        fig = go.Figure()
        fig.add_trace(go.Candlestick(x=levels["date"], open=levels["open"], high=levels["high"], low=levels["low"], close=levels["close"], name="OHLC"))
        for col, color in [("ma50", "#2ed17c"), ("ma150", "#d8cb5f"), ("ma200", "#a891ff")]:
            if col in levels:
                fig.add_trace(go.Scatter(x=levels["date"], y=levels[col], name=col.upper(), line=dict(color=color, width=1.4)))
        fig.update_layout(title=f"{ticker} SEPA structure")
        style_figure(fig, 440, time_axis=True)
        st.plotly_chart(fig, use_container_width=True)
    named_table("SEPA checks", result["checks"] if not result["checks"].empty else result["summary"])
    named_table("Hypothetical entry plan", sepa_entry_plan(levels))


def page_intraday_signal(prices, crypto, data_mode):
    page_header(
        "Trade Skills / Intraday Signal",
        "盘中多周期信号",
        "A 5m/15m/1h-style scenario desk. V0.1 uses live hourly data when available and falls back to bundled hourly crypto data; lower timeframes are clearly labeled as proxies when source granularity is not fine enough.",
        ["5m proxy", "15m proxy", "1h", "Risk gate"],
    )
    asset_class = st.radio("Asset class", ["Crypto hourly", "US stock hourly/live"], horizontal=True, key="intraday_asset")
    if asset_class == "Crypto hourly":
        if crypto.empty:
            st.warning("Crypto hourly data is unavailable.")
            return
        symbol = st.selectbox("Symbol", sorted(crypto["symbol"].unique()), key="intraday_crypto")
        data = crypto.loc[crypto["symbol"] == symbol].copy()
        title = symbol
        time_col = "datetime"
    else:
        symbol = st.selectbox("Ticker", sorted(prices["ticker"].unique()), key="intraday_stock")
        try:
            data = fetch_yfinance_prices([symbol], period="730d", interval="1h") if data_mode in {"live", "live_auto"} else prices.loc[prices["ticker"] == symbol].copy()
        except Exception as exc:
            if data_mode == "live":
                raise
            st.warning(f"Hourly source unavailable for {symbol}; using daily bars as a coarse proxy. {exc}")
            data = prices.loc[prices["ticker"] == symbol].copy()
        title = symbol
        time_col = "date"
    pack = intraday_signal_pack(data)
    if pack["signals"].empty:
        named_table("Risk gate", pack["risk_gate"])
        return
    col1, col2, col3 = st.columns(3)
    primary = pack["scenarios"].iloc[0]
    col1.metric("Primary direction", primary["direction"])
    col2.metric("Probability", f"{primary['probability']:.0f}%")
    col3.metric("Bars", fmt_int(len(data)))
    chart_data = data.copy()
    chart_data[time_col] = pd.to_datetime(chart_data[time_col], errors="coerce")
    fig = go.Figure(go.Candlestick(x=chart_data[time_col], open=chart_data["open"], high=chart_data["high"], low=chart_data["low"], close=chart_data["close"], name="OHLC"))
    fig.update_layout(title=f"{title} intraday source chart")
    style_figure(fig, 390, time_axis=True)
    st.plotly_chart(fig, use_container_width=True)
    named_table("Multi-timeframe signals", pack["signals"])
    named_table("Scenario plan", pack["scenarios"])
    named_table("Risk gate", pack["risk_gate"])


def page_capital_rotation(prices):
    page_header(
        "Trade Skills / Capital Rotation",
        "资金轮动雷达",
        "Cohort-level flow comparison for retail-friendly narrative tracking: semiconductors, AI software, EV/autonomy, AI cloud, and storage are compared by normalized price flow and breadth.",
        ["Cohort flow", "Breadth", "Narrative label"],
    )
    lookback = st.slider("Lookback bars", 21, 126, 63, step=21)
    result = cohort_rotation(prices, lookback=lookback)
    st.markdown(f'<div class="runjin-note">Rotation label: <strong>{result["label"]}</strong>. Price flow is a research proxy, not fund-flow truth.</div>', unsafe_allow_html=True)
    curves = result["curves"]
    if not curves.empty:
        fig = go.Figure()
        for col in [c for c in curves.columns if c != "date"]:
            fig.add_trace(go.Scatter(x=curves["date"], y=curves[col], name=col))
        fig.update_layout(title="Normalized cohort flow", yaxis_title="Indexed performance")
        style_figure(fig, 430, time_axis=True)
        st.plotly_chart(fig, use_container_width=True)
    scores = result["scores"].copy()
    if not scores.empty:
        scores["lookback_return"] = scores["lookback_return"].map(lambda value: f"{value:.1%}")
        scores["positive_breadth"] = scores["positive_breadth"].map(lambda value: f"{value:.0%}")
    named_table("Cohort rotation scores", scores)


def page_research_journal(scored):
    page_header(
        "Trade Skills / Research Journal",
        "研究日志",
        "A durable note layer for stock deep dives. The current V0.1 generates structured prompts and markdown drafts; writing to disk can be added once you approve the exact note workflow.",
        ["Business", "Fundamentals", "Technicals", "Catalysts", "Invalidation"],
    )
    ticker = st.selectbox("Ticker", scored["ticker"].tolist(), key="journal_ticker")
    profile = scored.loc[scored["ticker"] == ticker].iloc[0].to_dict()
    sections = pd.DataFrame(default_deep_dive_sections(ticker, profile))
    named_table("Six-lens deep dive", sections)
    st.text_area("Markdown draft", value=build_markdown_note(ticker, sections), height=420, key="journal_draft")
    entries = list_journal_entries()
    named_table("Existing journal entries", entries if not entries.empty else pd.DataFrame([{"note": "No local journal markdown files yet"}]))


def page_backtest_lab(prices, crypto):
    page_header(
        "Backtest Lab / Strategy Proof",
        "策略回测平台",
        "A safer backtesting desk built on backtesting.py: you define a constrained YAML strategy spec, the engine runs on OHLCV bars, and the output shows equity, drawdown, trades, and robustness warnings. V0.1 does not execute arbitrary Python.",
        ["backtesting.py", "No eval", "No leverage", "Research only"],
    )
    st.markdown(
        '<div class="runjin-note">Input is a controlled YAML strategy spec, not raw Python code. This keeps the platform reproducible and avoids executing unsafe user code inside Streamlit.</div>',
        unsafe_allow_html=True,
    )

    asset_class = st.radio("Asset class", ["US stock", "Crypto"], horizontal=True, key="bt_asset_class")
    if asset_class == "US stock":
        ticker = st.selectbox("Ticker", sorted(prices["ticker"].unique()), index=0, key="bt_ticker")
        raw = prices.loc[prices["ticker"] == ticker].copy()
        time_col = "date"
        timeframe = st.selectbox("Backtest timeframe", ["1D", "1W", "1M"], index=0, key="bt_stock_timeframe")
        raw = resample_ohlcv(raw, timeframe, "date")
    else:
        if crypto.empty:
            st.warning("Crypto live data is unavailable. Switch to US stock or Live auto/Sample only for crypto backtests.")
            return
        ticker = st.selectbox("Symbol", sorted(crypto["symbol"].unique()), index=0, key="bt_symbol")
        raw = crypto.loc[crypto["symbol"] == ticker].copy()
        time_col = "datetime"
        timeframe = st.selectbox("Backtest timeframe", ["1H", "1D"], index=0, key="bt_crypto_timeframe")
        if timeframe == "1D":
            raw = resample_ohlcv(raw.rename(columns={"datetime": "date"}), "1D", "date").rename(columns={"date": "datetime"})

    raw[time_col] = pd.to_datetime(raw[time_col], errors="coerce")
    min_date = raw[time_col].min().date()
    max_date = raw[time_col].max().date()
    col1, col2, col3 = st.columns([1, 1, 1.2])
    start_date = col1.date_input("Start", value=min_date, min_value=min_date, max_value=max_date, key="bt_start")
    end_date = col2.date_input("End", value=max_date, min_value=min_date, max_value=max_date, key="bt_end")
    example_name = col3.selectbox("Template", list(EXAMPLE_STRATEGY_SPECS.keys()), key="bt_template")

    if st.button("Load template", key="bt_load_template"):
        st.session_state["bt_strategy_spec"] = EXAMPLE_STRATEGY_SPECS[example_name]

    default_spec = st.session_state.get("bt_strategy_spec", EXAMPLE_STRATEGY_SPECS.get(example_name, DEFAULT_STRATEGY_SPEC))
    strategy_spec = st.text_area(
        "Strategy YAML",
        value=default_spec,
        height=260,
        key="bt_strategy_spec",
        help="Edit parameters safely. Supported templates: sma_crossover, rsi_mean_reversion, bollinger_reversion, macd_trend.",
    )

    mask = (raw[time_col].dt.date >= start_date) & (raw[time_col].dt.date <= end_date)
    backtest_data = raw.loc[mask].copy()
    st.caption(f"Backtest dataset: {ticker} / {timeframe} / {len(backtest_data):,} bars / {start_date} -> {end_date}")

    if st.button("Run backtest", type="primary", key="bt_run"):
        with st.spinner("Running backtest with backtesting.py..."):
            try:
                result = run_strategy_backtest(backtest_data, time_col, strategy_spec)
            except BacktestEngineUnavailable as exc:
                st.error(str(exc))
                st.code("pip install -r requirements.txt", language="bash")
                return
            except Exception as exc:
                st.error(f"Backtest failed: {exc}")
                return
        st.session_state["bt_result"] = result

    result = st.session_state.get("bt_result")
    if result:
        for warning in result.warnings:
            st.warning(warning)

        stats = result.stats
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Return", fmt_number(stats.get("Return [%]"), 1, "%"))
        col2.metric("Max drawdown", fmt_number(stats.get("Max. Drawdown [%]"), 1, "%"))
        col3.metric("Sharpe", fmt_number(stats.get("Sharpe Ratio"), 2))
        col4.metric("Win rate", fmt_number(stats.get("Win Rate [%]"), 1, "%"))
        col5.metric("Trades", fmt_int(stats.get("# Trades", 0) or 0))

        if result.ump_verdict is not None and not result.ump_verdict.empty:
            verdict = result.ump_verdict["verdict"].iloc[0]
            st.markdown(
                f'<div class="runjin-note">UMP-lite verdict: <strong>{verdict.upper()}</strong>. Rule-based裁判只做研究拦截提示，不自动下单。</div>',
                unsafe_allow_html=True,
            )
            named_table("UMP-lite verdict", result.ump_verdict)
        if result.assumptions is not None and not result.assumptions.empty:
            named_table("Execution assumptions", result.assumptions)

        equity = result.equity_curve.copy()
        date_col = "Date" if "Date" in equity.columns else equity.columns[0]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity[date_col], y=equity["Equity"], name="Equity", line=dict(color="#2ed17c")))
        if "DrawdownPct" in equity.columns:
            fig.add_trace(
                go.Scatter(
                    x=equity[date_col],
                    y=equity["DrawdownPct"] * 100,
                    name="Drawdown %",
                    yaxis="y2",
                    line=dict(color="#ff5f64", width=1),
                )
            )
            fig.update_layout(yaxis2=dict(title="Drawdown %", overlaying="y", side="right", gridcolor="rgba(255,255,255,0)"))
        fig.update_layout(title=f"{result.name} equity curve", yaxis_title="Equity")
        style_figure(fig, 410, time_axis=True)
        st.plotly_chart(fig, use_container_width=True)

        summary = pd.DataFrame([{"metric": key, "value": value} for key, value in stats.items() if key not in {"Start", "End"}])
        named_table("Backtest statistics", summary)
        if result.metrics_detail is not None and not result.metrics_detail.empty:
            named_table("ABU-style metrics detail", result.metrics_detail)
        if result.slippage_detail is not None and not result.slippage_detail.empty:
            slippage_cols = [col for col in ["date", "close", "execution_reference", "gap_pct", "gap_guard_flag", "slippage_bps"] if col in result.slippage_detail.columns]
            named_table("Slippage reference sample", result.slippage_detail[slippage_cols].round(4))

        trades = result.trades.copy()
        if not trades.empty:
            display_cols = [col for col in ["EntryTime", "ExitTime", "Size", "EntryPrice", "ExitPrice", "PnL", "ReturnPct", "Duration"] if col in trades.columns]
            named_table("Backtest trades", trades[display_cols] if display_cols else trades)
        else:
            st.markdown('<div class="runjin-note">No trades were generated. Adjust the strategy parameters or date range.</div>', unsafe_allow_html=True)

    section_label("Batch Backtest Leaderboard")
    st.markdown(
        '<div class="runjin-note">Vectorbt-style batch lab: scan many symbols and parameter variants at once, then rank strategy candidates before deeper single-asset review.</div>',
        unsafe_allow_html=True,
    )
    available_tickers = sorted(prices["ticker"].dropna().astype(str).unique())
    default_batch_tickers = available_tickers[: min(20, len(available_tickers))]
    col1, col2, col3 = st.columns([1.4, 0.8, 0.8])
    batch_tickers = col1.multiselect("Batch universe", available_tickers, default=default_batch_tickers, key="batch_tickers")
    max_batch_tickers = col2.slider("Max symbols", 5, min(200, max(5, len(available_tickers))), min(30, max(5, len(available_tickers))), 5, key="batch_max_tickers")
    max_variants = col3.slider("Variants / strategy", 1, 12, 6, 1, key="batch_max_variants")
    batch_strategies = st.multiselect(
        "Strategy templates",
        list(BATCH_STRATEGY_GRIDS.keys()),
        default=["sma_crossover", "rsi_mean_reversion", "macd_trend"],
        key="batch_strategies",
    )
    if st.button("Run batch leaderboard", type="primary", key="batch_run"):
        with st.spinner("Running vectorbt-style batch scan..."):
            try:
                batch_result = run_batch_backtest(
                    prices,
                    tickers=batch_tickers,
                    strategies=batch_strategies,
                    max_tickers=max_batch_tickers,
                    max_variants_per_strategy=max_variants,
                )
            except Exception as exc:
                st.error(f"Batch backtest failed: {exc}")
                return
        st.session_state["batch_result"] = batch_result

    batch_result = st.session_state.get("batch_result")
    if batch_result:
        for warning in batch_result.warnings[:8]:
            st.warning(warning)
        leaderboard = batch_result.leaderboard.copy()
        if leaderboard.empty:
            st.markdown('<div class="runjin-note">No batch results were generated. Add more price history or select different templates.</div>', unsafe_allow_html=True)
        else:
            top = leaderboard.iloc[0]
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Best ticker", top["ticker"])
            col2.metric("Best return", fmt_number(top["return_pct"], 1, "%"))
            col3.metric("Best drawdown", fmt_number(top["max_drawdown_pct"], 1, "%"))
            col4.metric("Best Sharpe", fmt_number(top["sharpe"], 2))
            col5.metric("Runs", fmt_int(len(leaderboard)))
            display = leaderboard.head(80).copy()
            for col in ["return_pct", "max_drawdown_pct", "sharpe", "win_rate_pct", "score"]:
                display[col] = pd.to_numeric(display[col], errors="coerce").round(2)
            named_table("Strategy leaderboard", display)

            curves = batch_result.equity_curves
            if not curves.empty:
                top_keys = leaderboard.head(5)[["ticker", "strategy", "params"]]
                fig = go.Figure()
                for _, row in top_keys.iterrows():
                    mask = (
                        (curves["ticker"] == row["ticker"])
                        & (curves["strategy"] == row["strategy"])
                        & (curves["params"] == row["params"])
                    )
                    curve = curves.loc[mask].copy()
                    if curve.empty:
                        continue
                    normalized = curve["equity"] / curve["equity"].iloc[0] * 100
                    fig.add_trace(go.Scatter(x=curve["date"], y=normalized, name=f"{row['ticker']} {row['strategy']}"))
                fig.update_layout(title="Top batch equity curves", yaxis_title="Indexed equity")
                style_figure(fig, 410, time_axis=True)
                st.plotly_chart(fig, use_container_width=True)

    section_label("Portfolio Backtest")
    st.markdown(
        '<div class="runjin-note">Portfolio Lab absorbs the best bt/qstrader idea: test allocation rules, rebalancing, turnover, and cost before thinking about execution.</div>',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns([1.2, 1])
    portfolio_template = col1.selectbox("Portfolio template", list(EXAMPLE_PORTFOLIO_SPECS.keys()), key="portfolio_template")
    if col2.button("Load portfolio template", key="portfolio_load"):
        st.session_state["portfolio_spec"] = EXAMPLE_PORTFOLIO_SPECS[portfolio_template]

    portfolio_spec = st.text_area(
        "Portfolio YAML",
        value=st.session_state.get("portfolio_spec", EXAMPLE_PORTFOLIO_SPECS.get(portfolio_template, DEFAULT_PORTFOLIO_SPEC)),
        height=245,
        key="portfolio_spec",
        help="Templates: equal_weight_rebalance, momentum_top_n, inverse_volatility. No leverage; max_position_pct caps concentration.",
    )
    if st.button("Run portfolio backtest", type="primary", key="portfolio_run"):
        with st.spinner("Running portfolio rebalance simulation..."):
            try:
                portfolio_result = run_portfolio_backtest(prices, portfolio_spec)
            except Exception as exc:
                st.error(f"Portfolio backtest failed: {exc}")
                return
        st.session_state["portfolio_result"] = portfolio_result

    portfolio_result = st.session_state.get("portfolio_result")
    if portfolio_result:
        for warning in portfolio_result.warnings:
            st.warning(warning)
        stats = portfolio_result.stats
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Return", fmt_number(stats.get("Return [%]"), 1, "%"))
        col2.metric("Max drawdown", fmt_number(stats.get("Max. Drawdown [%]"), 1, "%"))
        col3.metric("Sharpe", fmt_number(stats.get("Sharpe Ratio"), 2))
        col4.metric("Rebalances", fmt_int(stats.get("Rebalances", 0) or 0))
        col5.metric("Total cost", f"${fmt_int(stats.get('Total Cost [$]', 0) or 0)}")

        equity = portfolio_result.equity_curve.copy()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=equity["date"], y=equity["Equity"], name="Equity", line=dict(color="#2ed17c")))
        fig.add_trace(go.Scatter(x=equity["date"], y=equity["DrawdownPct"] * 100, name="Drawdown %", yaxis="y2", line=dict(color="#ff5f64", width=1)))
        fig.update_layout(
            title=f"{portfolio_result.name} portfolio equity",
            yaxis_title="Equity",
            yaxis2=dict(title="Drawdown %", overlaying="y", side="right", gridcolor="rgba(255,255,255,0)"),
        )
        style_figure(fig, 410, time_axis=True)
        st.plotly_chart(fig, use_container_width=True)
        named_table("Portfolio statistics", pd.DataFrame([{"metric": key, "value": value} for key, value in stats.items()]))
        if not portfolio_result.rebalance_log.empty:
            named_table("Rebalance log", portfolio_result.rebalance_log.tail(30).round(4))
        if not portfolio_result.weights.empty:
            latest_weights = portfolio_result.weights.tail(1).drop(columns=["date"], errors="ignore").T.reset_index()
            latest_weights.columns = ["symbol", "weight"]
            latest_weights = latest_weights.sort_values("weight", ascending=False)
            named_table("Latest weights", latest_weights)

    section_label("Backtest System Map")
    named_table("Borrowed strengths from mature systems", pd.DataFrame(BACKTEST_SYSTEM_MAP))


def page_short_bot(prices, crypto, risk_rules):
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
        if crypto.empty:
            st.warning("Crypto live data is unavailable. The crypto paper bot is paused until BTC/ETH hourly data is available.")
            return
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
    style_figure(fig, 360, time_axis=True)
    st.plotly_chart(fig, use_container_width=True)

    latest = bt.tail(1).iloc[0]
    st.markdown(
        f'<div class="runjin-signal"><b>Current signal</b> {int(latest["signal"])} &nbsp; / &nbsp; '
        f'<b>Reason</b> {latest["signal_reason"]} &nbsp; / &nbsp; '
        f'<b>Position</b> {latest["position"]:.2%}</div>',
        unsafe_allow_html=True,
    )

    named_table("Recent strategy trades", trades)


def page_weekly_report(prices, crypto, scored, risk_rules):
    page_header(
        "Weekly Review / Keep the System Honest",
        "周度复盘",
        "A compact operating memo: long-term candidates, bot state, drawdown, and whether to continue, pause, or review. The ritual matters as much as the signal.",
        ["Continue", "Pause", "Review"],
    )
    stock_bt, stock_metrics, _, stock_status, stock_reason = run_us_stock_paper(prices.loc[prices["ticker"] == "NVDA"].copy(), risk_rules)
    if has_crypto_symbol(crypto, "BTC-USD"):
        _, crypto_metrics, _, crypto_status, crypto_reason = run_crypto_paper(crypto.loc[crypto["symbol"] == "BTC-USD"].copy(), risk_rules)
    else:
        crypto_metrics = disabled_bot_metrics()
        crypto_status = "PAUSE"
        crypto_reason = "Crypto live source unavailable"
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
    st.sidebar.markdown(
        """
        <div style="padding: 10px 4px 18px;">
          <div style="font-size: 24px; font-weight: 850; color: #e8ebe8; line-height: 1;">润金</div>
          <div style="font-family: SF Mono, Menlo, monospace; color: #2ed17c; font-size: 11px; letter-spacing: .08em; margin-top: 8px;">RUNJIN TRADE SYSTEM</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    data_mode_label = st.sidebar.radio(
        "Data mode",
        ["Live auto", "Sample only", "Live strict"],
        index=0,
        help="Live auto tries current market sources first and falls back to bundled samples when a public source is blocked or empty.",
    )
    data_mode = {
        "Live auto": "live_auto",
        "Sample only": "sample",
        "Live strict": "live",
    }[data_mode_label]
    try:
        prices, crypto, financials, market_universe, scored, forecasts, finance_research, risk_rules, source_status = load_all_data(data_mode)
    except Exception as exc:
        st.sidebar.error(f"Live strict failed: {exc}")
        st.sidebar.caption("Switch to Live auto or Sample only to keep the workspace running when public data sources are unavailable.")
        st.stop()
    page = st.sidebar.radio(
        "Workspace",
        [
            "Dashboard",
            "Market Universe",
            "Long Watchlist",
            "Finance MCP Radar",
            "Symbol Cockpit",
            "Stock Detail",
            "K-line Lab",
            "SEPA Lab",
            "Intraday Signal",
            "Capital Rotation",
            "Research Journal",
            "Backtest Lab",
            "Short Bot",
            "Weekly Report",
        ],
    )
    st.sidebar.caption(f"MODE / {data_mode}")
    st.sidebar.caption("BOUNDARY / V0.1 never places real orders.")

    if page == "Dashboard":
        page_dashboard(prices, crypto, market_universe, scored, finance_research, risk_rules, source_status)
    elif page == "Market Universe":
        page_market_universe(market_universe)
    elif page == "Long Watchlist":
        page_watchlist(scored)
    elif page == "Finance MCP Radar":
        page_finance_mcp_radar(finance_research, scored, source_status)
    elif page == "Symbol Cockpit":
        page_symbol_cockpit(prices, crypto, financials, scored, forecasts, data_mode)
    elif page == "Stock Detail":
        page_stock_detail(prices, financials, scored, forecasts, data_mode)
    elif page == "K-line Lab":
        page_kline_lab(prices, forecasts, market_universe, data_mode)
    elif page == "SEPA Lab":
        page_sepa_lab(prices)
    elif page == "Intraday Signal":
        page_intraday_signal(prices, crypto, data_mode)
    elif page == "Capital Rotation":
        page_capital_rotation(prices)
    elif page == "Research Journal":
        page_research_journal(scored)
    elif page == "Backtest Lab":
        page_backtest_lab(prices, crypto)
    elif page == "Short Bot":
        page_short_bot(prices, crypto, risk_rules)
    elif page == "Weekly Report":
        page_weekly_report(prices, crypto, scored, risk_rules)


if __name__ == "__main__":
    main()
