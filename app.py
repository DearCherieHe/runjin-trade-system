from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_sources.loaders import (
    get_data_source_status,
    load_crypto_prices,
    load_financials,
    load_kronos_forecast,
    load_prices,
    load_risk_rules,
    load_watchlist_notes,
)
from src.kline.indicators import add_indicators, classify_regime, relative_strength
from src.kline.timeframes import apply_asof, coverage_summary, resample_ohlcv
from src.data_sources.live_sources import fetch_yfinance_prices
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
    notes = load_watchlist_notes()
    forecasts = load_kronos_forecast(data_mode=data_mode)
    risk_rules = load_risk_rules()
    scored = build_score_table(notes)
    source_status = get_data_source_status()
    return prices, crypto, financials, scored, forecasts, risk_rules, source_status


def fmt_int(value):
    return f"{value:,.0f}"


def fmt_pct(value):
    return f"{value * 100:.0f}%"


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
    if timeframe == "1H" and data_mode in {"live_auto", "live"}:
        try:
            base = fetch_yfinance_prices([ticker], period="730d", interval="1h")
            source_note = "yfinance 1h on-demand; free intraday history is limited"
        except Exception as exc:
            if data_mode == "live":
                raise
            source_note = f"hourly live unavailable, using daily source: {exc}"
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


def page_dashboard(prices, crypto, scored, risk_rules, source_status):
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
    equity = coerce_datetime_key(equity, "date")
    crypto_equity = coerce_datetime_key(crypto_equity, "date")
    merged = pd.merge_asof(
        equity,
        crypto_equity,
        on="date",
        direction="nearest",
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["US stock bot"], name="US stock bot", line=dict(color="#2ed17c")))
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["Crypto bot"], name="Crypto bot", line=dict(color="#d8cb5f")))
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


def page_kline_lab(prices, forecasts, data_mode):
    page_header(
        "K-line Lab / Pattern Is Context",
        "K线解读台",
        "Indicators are treated as context, not prophecy. The Kronos-style line is a local research stub and remains visually separated from executable trading signals.",
        ["MA20 / MA60", "RSI / MACD", "Research-only forecast"],
    )
    tickers = sorted(prices["ticker"].unique())
    ticker = st.selectbox("Ticker", tickers, index=tickers.index("NVDA") if "NVDA" in tickers else 0)
    timeframe, as_of, show_forecast = kline_controls(prices, ticker, "lab", data_mode)
    raw_kline, source_note = build_ticker_kline(prices, ticker, timeframe, as_of, data_mode)
    ticker_prices = add_indicators(raw_kline)
    use_forecast = show_forecast and timeframe == "1D" and as_of >= prices.loc[prices["ticker"] == ticker, "date"].max().date()
    forecast = filter_forecast_for_chart(forecasts.loc[forecasts["ticker"] == ticker], ticker_prices) if use_forecast else pd.DataFrame()
    benchmark_raw = prices.loc[prices["ticker"] == "TSM"].copy()
    benchmark = apply_asof(resample_ohlcv(benchmark_raw, timeframe, "date"), as_of)
    rs = relative_strength(ticker_prices, benchmark) if ticker != "TSM" else pd.DataFrame()

    col1, col2, col3 = st.columns(3)
    col1.metric("Regime", classify_regime(ticker_prices))
    col2.metric("Latest RSI", f"{ticker_prices['rsi14'].dropna().iloc[-1]:.0f}")
    col3.metric("20D volatility", fmt_pct(ticker_prices["volatility_20"].dropna().iloc[-1]))

    st.markdown(
        f'<div class="runjin-note">K-line coverage: {coverage_summary(ticker_prices)} / {source_note}. Forecast overlay is off by default, research-only, hidden during replay, and filtered for price-scale sanity.</div>',
        unsafe_allow_html=True,
    )
    plot_kline(ticker_prices, f"{ticker} {timeframe} Indicators / Replay as of {as_of}", forecast)

    indicator_view = ticker_prices.tail(30)[["date", "close", "ma20", "ma60", "rsi14", "macd", "macd_signal", "bb_lower", "bb_upper"]].copy()
    named_table("Latest indicator values", indicator_view.round(2))

    if not rs.empty:
        fig = go.Figure(go.Scatter(x=rs["date"], y=rs["relative_strength"], name=f"{ticker} vs TSM"))
        fig.update_layout(title=f"Relative Strength: {ticker} vs TSM")
        style_figure(fig, 320, time_axis=True)
        st.plotly_chart(fig, use_container_width=True)


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
        prices, crypto, financials, scored, forecasts, risk_rules, source_status = load_all_data(data_mode)
    except Exception as exc:
        st.sidebar.error(f"Live strict failed: {exc}")
        st.sidebar.caption("Switch to Live auto or Sample only to keep the workspace running when public data sources are unavailable.")
        st.stop()
    page = st.sidebar.radio(
        "Workspace",
        ["Dashboard", "Long Watchlist", "Stock Detail", "K-line Lab", "Short Bot", "Weekly Report"],
    )
    st.sidebar.caption(f"MODE / {data_mode}")
    st.sidebar.caption("BOUNDARY / V0.1 never places real orders.")

    if page == "Dashboard":
        page_dashboard(prices, crypto, scored, risk_rules, source_status)
    elif page == "Long Watchlist":
        page_watchlist(scored)
    elif page == "Stock Detail":
        page_stock_detail(prices, financials, scored, forecasts, data_mode)
    elif page == "K-line Lab":
        page_kline_lab(prices, forecasts, data_mode)
    elif page == "Short Bot":
        page_short_bot(prices, crypto, risk_rules)
    elif page == "Weekly Report":
        page_weekly_report(prices, crypto, scored, risk_rules)


if __name__ == "__main__":
    main()
