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

st.set_page_config(page_title="Trade Lab", layout="wide")


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
    st.caption(f"Table: {name}")
    st.dataframe(df, use_container_width=True, hide_index=True)


def metric_row(label, value, help_text=None):
    st.metric(label, value, help=help_text)


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
        )
    )
    if "ma20" in df:
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma20"], name="MA20", line=dict(width=1.5)))
    if "ma60" in df:
        fig.add_trace(go.Scatter(x=df["date"], y=df["ma60"], name="MA60", line=dict(width=1.5)))
    if forecast is not None and not forecast.empty:
        fig.add_trace(
            go.Scatter(
                x=forecast["date"],
                y=forecast["predicted_close"],
                name="Kronos-style forecast (research only)",
                line=dict(dash="dash"),
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
    st.plotly_chart(fig, use_container_width=True)


def page_dashboard(prices, crypto, scored, risk_rules):
    st.title("Trade Lab")
    st.caption("Long-term stock observation desk + short-term paper-trading lab. Offline demo, no real orders.")

    nvda = prices.loc[prices["ticker"] == "NVDA"]
    btc = crypto.loc[crypto["symbol"] == "BTC-USD"]
    stock_bt, stock_metrics, _, stock_status, stock_reason = run_us_stock_paper(nvda, risk_rules)
    crypto_bt, crypto_metrics, _, crypto_status, crypto_reason = run_crypto_paper(btc, risk_rules)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Watchlist", fmt_int(len(scored)))
    col2.metric("Deep research", fmt_int((scored["bucket"] == "Deep research candidate").sum()))
    col3.metric("US paper bot", stock_status, stock_reason)
    col4.metric("Crypto paper bot", crypto_status, crypto_reason)

    st.subheader("Portfolio Simulation")
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
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["US stock bot"], name="US stock bot"))
    fig.add_trace(go.Scatter(x=merged["date"], y=merged["Crypto bot"], name="Crypto bot"))
    fig.update_layout(title="Paper Equity Curves", yaxis_title="USD", height=360)
    st.plotly_chart(fig, use_container_width=True)

    alerts = []
    for name, status, reason in [("US stock bot", stock_status, stock_reason), ("Crypto bot", crypto_status, crypto_reason)]:
        if status != "CONTINUE":
            alerts.append({"system": name, "status": status, "reason": reason})
    if alerts:
        named_table("Risk alerts", pd.DataFrame(alerts))
    else:
        st.success("No risk stop triggered in current sample simulation.")

    named_table("Top long-term candidates", scored[["ticker", "company", "tags", "score_label", "bucket", "growth_evidence"]].head(8))


def page_watchlist(scored):
    st.title("Long Watchlist")
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

    st.subheader("Score Breakdown")
    selected = st.selectbox("Ticker", view["ticker"].tolist())
    row = view.loc[view["ticker"] == selected].iloc[0]
    score_df = pd.DataFrame({"component": SCORE_COLUMNS, "score": [row[col] for col in SCORE_COLUMNS]})
    fig = go.Figure(go.Bar(x=score_df["component"], y=score_df["score"]))
    fig.update_layout(yaxis=dict(range=[0, 5]), height=340)
    st.plotly_chart(fig, use_container_width=True)


def page_stock_detail(prices, financials, scored, forecasts):
    st.title("Stock Detail")
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

    st.subheader("Thesis")
    thesis_cols = st.columns(2)
    with thesis_cols[0]:
        st.markdown(f"**Narrative**\n\n{profile['narrative']}")
        st.markdown(f"**Growth evidence**\n\n{profile['growth_evidence']}")
        st.markdown(f"**Catalysts**\n\n{profile['catalysts']}")
    with thesis_cols[1]:
        st.markdown(f"**Invalidation**\n\n{profile['invalidation']}")
        st.markdown(f"**Buy plan**\n\n{profile['buy_plan']}")
        st.markdown(f"**Bucket**\n\n{profile['bucket']}")

    fin_view = ticker_financials.copy()
    for col in ["revenue_growth_yoy", "gross_margin", "operating_cash_flow_margin", "net_income_margin"]:
        fin_view[col] = fin_view[col].map(fmt_pct)
    named_table("Financial trend", fin_view)


def page_kline_lab(prices, forecasts):
    st.title("K-line Lab")
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

    st.info("Kronos-style forecast is a local research-only stub. It does not generate trade instructions.")
    plot_kline(ticker_prices, f"{ticker} Indicators + Research Forecast", forecast)

    indicator_view = ticker_prices.tail(30)[["date", "close", "ma20", "ma60", "rsi14", "macd", "macd_signal", "bb_lower", "bb_upper"]].copy()
    named_table("Latest indicator values", indicator_view.round(2))

    if not rs.empty:
        fig = go.Figure(go.Scatter(x=rs["date"], y=rs["relative_strength"], name=f"{ticker} vs TSM"))
        fig.update_layout(title=f"Relative Strength: {ticker} vs TSM", height=320)
        st.plotly_chart(fig, use_container_width=True)


def page_short_bot(prices, crypto, risk_rules, paper_trades):
    st.title("Short Bot")
    st.caption("Paper trading only. No leverage, no broker API, no exchange keys.")

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
    fig.add_trace(go.Scatter(x=bt[time_col], y=bt["equity"], name="Equity"))
    fig.update_layout(title=f"{ticker} Paper Equity", yaxis_title="USD", height=360)
    st.plotly_chart(fig, use_container_width=True)

    latest = bt.tail(1).iloc[0]
    st.markdown(
        f"**Current signal:** `{int(latest['signal'])}` | **Reason:** {latest['signal_reason']} | "
        f"**Position:** {latest['position']:.2%}"
    )

    named_table("Recent strategy trades", trades)
    named_table("Sample paper trade ledger", paper_trades)


def page_weekly_report(prices, crypto, scored, risk_rules):
    st.title("Weekly Report")
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
    prices, crypto, financials, scored, forecasts, paper_trades, risk_rules = load_all_data()
    page = st.sidebar.radio(
        "Trade Lab",
        ["Dashboard", "Long Watchlist", "Stock Detail", "K-line Lab", "Short Bot", "Weekly Report"],
    )
    st.sidebar.caption("Mode: offline sample data")
    st.sidebar.caption("V0.1 never places real orders.")

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
