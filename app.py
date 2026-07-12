from pathlib import Path
from typing import Optional
from html import escape

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

APP_ROOT = Path(__file__).resolve().parent
SAMPLE_DIR = APP_ROOT / "data" / "sample"

try:
    from src.data_sources.loaders import (
        get_data_source_status,
        load_crypto_prices,
        load_financials,
        load_future_industry_map,
        load_kronos_forecast,
        load_market_universe,
        load_prices,
        load_risk_rules,
        load_watchlist_notes,
    )
except ImportError as loaders_import_error:
    LOADERS_IMPORT_ERROR = loaders_import_error

    def get_data_source_status() -> dict:
        return {
            "startup": {
                "mode": "sample_fallback",
                "source": "local_sample",
                "message": f"Data loader import failed; using bundled samples: {LOADERS_IMPORT_ERROR}",
            }
        }

    def load_prices(data_mode=None) -> pd.DataFrame:
        return pd.read_csv(SAMPLE_DIR / "us_stock_ohlcv.csv", parse_dates=["date"])

    def load_crypto_prices(data_mode=None) -> pd.DataFrame:
        return pd.read_csv(SAMPLE_DIR / "crypto_ohlcv_hourly.csv", parse_dates=["datetime"])

    def load_financials(data_mode=None) -> pd.DataFrame:
        return pd.read_csv(SAMPLE_DIR / "financial_metrics.csv", parse_dates=["quarter"])

    def load_kronos_forecast(data_mode=None) -> pd.DataFrame:
        return pd.read_csv(SAMPLE_DIR / "kronos_forecast_sample.csv", parse_dates=["date"])

    def load_market_universe(data_mode=None) -> pd.DataFrame:
        prices = load_prices(data_mode)
        return prices[["ticker"]].drop_duplicates().assign(market="US", source="sample_fallback")

    def load_risk_rules() -> dict:
        return {
            "capital": {"starting_cash": 100000, "max_position_pct": 0.12, "no_leverage": True},
            "risk_limits": {
                "max_daily_loss_pct": 0.03,
                "max_drawdown_pct": 0.12,
                "per_trade_stop_loss_pct": 0.06,
                "min_cash_pct": 0.25,
            },
        }

    def load_watchlist_notes() -> pd.DataFrame:
        return pd.read_csv(SAMPLE_DIR / "watchlist_notes.csv")

    def load_future_industry_map() -> pd.DataFrame:
        return pd.read_csv(SAMPLE_DIR / "future_industry_map.csv")

from src.data_sources.finance_mcp import load_finance_mcp_capabilities, load_finance_mcp_research
from src.data_sources.market_universe import ensure_market_universe_columns
from src.data_sources.sync import result_to_frames, sync_single_quote
from src.core.cache import CacheManager
from src.core.llm_providers import provider_table, task_policy_table
from src.ashare_workbench.agent_framework import astock_agent_tables, astock_decision_prompt
from src.ashare_workbench.levels import key_price_levels
from src.market_workbench.core import (
    STRATEGY_DESCRIPTIONS,
    add_market_indicators,
    filter_screener,
    latest_snapshot as market_latest_snapshot,
    market_rotation,
    monitor_verdict,
    screen_market,
    surge_ladder,
)
from src.market_workbench.data import build_market_workbench_data
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
from src.kline.launch_points import launch_point_analysis
from src.kline.timeframes import apply_asof, coverage_summary, resample_ohlcv
from src.kline.volume_price import analyze_volume_price_state, latest_volume_price_note
try:
    from src.data_sources.live_sources import fetch_yfinance_prices
except ImportError:
    def fetch_yfinance_prices(tickers, period="max", interval="1d") -> pd.DataFrame:
        prices = pd.read_csv(SAMPLE_DIR / "us_stock_ohlcv.csv", parse_dates=["date"])
        return prices[prices["ticker"].isin(tickers)].copy()
from src.long_term.scoring import SCORE_COLUMNS, build_score_table
from src.long_term.industry_map import build_future_profile, future_theme_summary, industry_layer_scores, relationship_comparison
from src.long_term.batch_analysis import analysis_progress_rows, batch_long_analysis
from src.long_term.thesis import (
    EXIT_OBSERVATION_CONDITIONS,
    HOLD_OBSERVATION_CONDITIONS,
    TENBAGGER_DISCOVERY_FRAMEWORK,
    get_ticker_profile,
    latest_financial_snapshot,
)
from src.long_term.ollama_assistant import (
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    ask_ollama,
    load_ollama_config,
    ollama_status,
    save_ollama_config,
)
from src.quant_bot.paper_trader import run_crypto_paper, run_us_stock_paper
from src.reports.exporter import export_markdown_report, report_export_capabilities
from src.reports.weekly_report import build_weekly_report
from src.trade_skills.capital_rotation import cohort_rotation
from src.trade_skills.intraday import intraday_signal_pack
from src.trade_skills.journal import build_markdown_note, default_deep_dive_sections, list_journal_entries
from src.trade_skills.sepa import sepa_dashboard, sepa_entry_plan


ROOT = APP_ROOT

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
  padding: 2rem 2rem 3.2rem;
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
  padding: 9px 10px;
  margin: 4px 0;
  border: 1px solid transparent;
  transition: background .15s ease, border-color .15s ease;
}

[data-testid="stSidebar"] [role="radiogroup"] label:hover {
  background: rgba(255,255,255,0.05);
}

[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
  background: rgba(46,209,124,0.13);
  border-color: rgba(46,209,124,0.35);
}

[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) * {
  color: #55e49b !important;
}

[data-testid="stSidebar"] [role="radiogroup"] [data-testid="stWidgetLabel"],
[data-testid="stSidebar"] [role="radiogroup"] input {
  display: none;
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
  border: 0;
  background: transparent;
  padding: 8px 0 12px;
  margin: 0 0 16px;
  box-shadow: none;
  position: relative;
  overflow: visible;
}

.runjin-hero::after {
  content: none;
}

.runjin-kicker {
  display: none;
}

.runjin-title-row {
  display: flex;
  align-items: center;
  gap: 12px;
  min-height: 40px;
}

.runjin-title-row::before {
  content: "";
  width: 5px;
  height: 28px;
  border-radius: 99px;
  background: linear-gradient(180deg, #54f3df, var(--rj-green));
  box-shadow: 0 0 16px rgba(46,209,124,0.35);
  flex: 0 0 auto;
}

.runjin-title {
  font-size: 30px;
  line-height: 1.25;
  font-weight: 820;
  color: var(--rj-text);
  letter-spacing: 0;
  max-width: 980px;
}

.runjin-subtitle {
  display: none;
}

.runjin-page-note {
  max-width: 1040px;
  margin: 8px 0 0 17px;
  color: #818981;
  font-family: var(--rj-mono);
  font-size: 14px;
  line-height: 1.45;
  text-wrap: pretty;
}

.runjin-ribbon {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 12px 0 0 17px;
}

.runjin-ribbon span {
  border: 1px solid var(--rj-line);
  background: rgba(255,255,255,0.035);
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
  border-radius: 8px;
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

.runjin-kpi-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 18px;
  margin: 8px 0 20px;
}

.runjin-kpi-card {
  height: 116px;
  border: 1px solid var(--rj-line);
  border-radius: 8px;
  background: linear-gradient(180deg, #171918, #101211);
  padding: 18px 18px 16px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
  overflow: hidden;
}

.runjin-kpi-label {
  color: var(--rj-muted);
  font-family: var(--rj-mono);
  font-size: 12px;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.runjin-kpi-value {
  color: var(--rj-text);
  font-family: var(--rj-mono);
  font-size: 28px;
  font-weight: 780;
  line-height: 1.15;
  margin-top: 12px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.runjin-kpi-note {
  color: #9eb8a4;
  font-family: var(--rj-sans);
  font-size: 13px;
  line-height: 1.2;
  margin-top: 10px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}

.runjin-work-grid {
  display: grid;
  grid-template-columns: minmax(0, 1.45fr) minmax(320px, 0.8fr);
  gap: 20px;
  align-items: start;
}

.runjin-panel {
  border: 1px solid var(--rj-line);
  border-radius: 8px;
  background: var(--rj-panel);
  padding: 16px;
}

.runjin-list {
  display: grid;
  gap: 10px;
  margin-top: 10px;
}

.runjin-list-item {
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 8px;
  background: rgba(255,255,255,0.025);
  padding: 11px 12px;
}

.runjin-list-title {
  color: var(--rj-text);
  font-weight: 740;
  margin-bottom: 4px;
}

.runjin-list-meta {
  color: var(--rj-muted);
  font-size: 12px;
  line-height: 1.35;
}

[data-testid="stMetric"] {
  background: linear-gradient(180deg, #171918, #101211);
  border: 1px solid var(--rj-line);
  border-radius: 8px;
  padding: 15px 15px 14px;
  box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
  min-height: 112px;
  height: 112px;
  display: flex;
  flex-direction: column;
  justify-content: center;
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
  line-height: 1.15;
}

[data-testid="stMetricDelta"] {
  color: var(--rj-green);
  min-height: 24px;
  margin-top: 8px;
  max-width: 100%;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

[data-testid="stMetric"] [data-testid="stMetricDelta"] svg {
  flex: 0 0 auto;
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--rj-line);
  background: var(--rj-panel);
}

[data-testid="stCaptionContainer"] {
  color: var(--rj-dim);
  font-family: var(--rj-sans);
  text-transform: none;
  letter-spacing: 0;
  font-weight: 700;
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
  border-radius: 8px;
}

@media (max-width: 800px) {
  .block-container { padding: 1.4rem 1rem 2.4rem; }
  .runjin-title { font-size: 25px; }
  .runjin-kpi-grid, .runjin-work-grid { grid-template-columns: 1fr; }
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
          <div class="runjin-title-row"><div class="runjin-title">{title}</div></div>
          <div class="runjin-subtitle">{subtitle}</div>
          {ribbon_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_label(text: str):
    st.markdown(f'<div class="runjin-section">{text}</div>', unsafe_allow_html=True)


def render_research_checklist(title: str, items, glue: Optional[str] = None):
    if glue:
        body = f"<p>{glue.join(items)}</p>"
    else:
        body = "<ul>" + "".join(f"<li>{item}</li>" for item in items) + "</ul>"
    st.markdown(f'<div class="runjin-thesis"><h4>{title}</h4>{body}</div>', unsafe_allow_html=True)


def render_kpi_grid(items):
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        note = escape(str(item.get("note", ""))) or "&nbsp;"
        col.markdown(
            '<div class="runjin-kpi-card">'
            f'<div class="runjin-kpi-label">{escape(str(item["label"]))}</div>'
            f'<div class="runjin-kpi-value">{escape(str(item["value"]))}</div>'
            f'<div class="runjin-kpi-note">{note}</div>'
            "</div>",
            unsafe_allow_html=True,
        )


@st.cache_data
def load_all_data(data_mode):
    prices = load_prices(data_mode=data_mode)
    crypto = load_crypto_prices(data_mode=data_mode)
    financials = load_financials(data_mode=data_mode)
    market_universe = load_market_universe(data_mode=data_mode)
    notes = load_watchlist_notes()
    industry_map = load_future_industry_map()
    forecasts = load_kronos_forecast(data_mode=data_mode)
    finance_research, finance_mcp_status = load_finance_mcp_research(data_mode=data_mode)
    risk_rules = load_risk_rules()
    scored = build_score_table(notes)
    source_status = get_data_source_status()
    source_status["finance_mcp"] = finance_mcp_status
    return prices, crypto, financials, market_universe, scored, industry_map, forecasts, finance_research, risk_rules, source_status


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


def status_cn(status):
    return {
        "CONTINUE": "运行中",
        "PAUSE": "暂停",
        "STOP": "停止",
        "REVIEW": "复核",
        "PASS": "通过",
        "FAIL": "失败",
        "success": "成功",
        "failed": "失败",
    }.get(str(status), str(status))


def bucket_cn(bucket):
    return {
        "Deep research candidate": "深度研究",
        "Observation pool": "观察池",
        "Low priority": "低优先级",
    }.get(str(bucket), str(bucket))


def watchlist_display(df):
    view = df.copy()
    if "bucket" in view:
        view["bucket"] = view["bucket"].map(bucket_cn)
    rename_map = {
        "ticker": "代码",
        "company": "公司",
        "tags": "标签",
        "score_label": "评分",
        "bucket": "层级",
        "narrative": "长期逻辑",
        "growth_evidence": "验证证据",
        "invalidation": "失效条件",
        "catalysts": "催化剂",
        "buy_plan": "计划",
    }
    return view.rename(columns={key: value for key, value in rename_map.items() if key in view.columns})


def build_ollama_research_prompt(ticker, profile, ticker_financials, user_question):
    latest = latest_financial_snapshot(ticker_financials)
    if latest:
        financial_context = (
            f"最新财务: 营收同比 {fmt_pct(latest['revenue_growth_yoy'])}, "
            f"毛利率 {fmt_pct(latest['gross_margin'])}, "
            f"经营现金流率 {fmt_pct(latest['operating_cash_flow_margin'])}, "
            f"净利率 {fmt_pct(latest['net_income_margin'])}, "
            f"营收增速变化 {fmt_pct(latest['revenue_growth_delta'])}."
        )
    else:
        financial_context = "暂无财务序列。"

    return f"""
你是润金交易系统里的本地投研小助手。请用中文回答，风格要短、清楚、可执行。
边界：只做投研分析和问题拆解，不给确定性买卖指令，不承诺收益。

当前股票：{ticker} / {profile['company']}
叙事：{profile['narrative']}
成长证据：{profile['growth_evidence']}
催化剂：{profile['catalysts']}
失效条件：{profile['invalidation']}
买入计划：{profile['buy_plan']}
评分：{profile['score_label']} / 分层：{profile['bucket']}
{financial_context}

继续跟踪/持有条件：
{chr(10).join('- ' + item for item in HOLD_OBSERVATION_CONDITIONS)}

放弃条件：
{chr(10).join('- ' + item for item in EXIT_OBSERVATION_CONDITIONS)}

识别牛股框架：
{" + ".join(TENBAGGER_DISCOVERY_FRAMEWORK)}

用户问题：{user_question}

请按这个格式输出：
1. 核心判断：一句话。
2. 证据：3条以内。
3. 风险/失效点：3条以内。
4. 下一步要核实什么：3条以内。
""".strip()


def coerce_datetime_key(df, column="date"):
    normalized = df.copy()
    normalized[column] = pd.to_datetime(normalized[column], errors="coerce")
    if getattr(normalized[column].dt, "tz", None) is not None:
        normalized[column] = normalized[column].dt.tz_convert(None)
    normalized[column] = normalized[column].astype("datetime64[ns]")
    return normalized.dropna(subset=[column]).sort_values(column).reset_index(drop=True)


def named_table(name, df):
    st.caption(name)
    st.dataframe(df, use_container_width=True, hide_index=True)


def render_data_source_debug(source_status):
    status_df = pd.DataFrame(
        [
            {"layer": key, **value}
            for key, value in source_status.items()
        ]
    )
    with st.expander("Debug / data plumbing", expanded=False):
        named_table("Data source status", status_df)


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


def plot_kline(df, title, forecast=None, signal_markers=None):
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
    if signal_markers is not None and not signal_markers.empty:
        for action, markers in signal_markers.groupby("action"):
            fig.add_trace(
                go.Scatter(
                    x=markers["date"],
                    y=markers["close"],
                    mode="markers+text",
                    name=f"Volume-price: {action}",
                    text=markers["marker_text"],
                    textposition="top center",
                    marker=dict(
                        size=11,
                        color=markers["color"],
                        symbol="diamond",
                        line=dict(color="#151716", width=1),
                    ),
                    hovertext=markers["note"],
                    hoverinfo="text+x+y",
                )
            )
    fig.update_layout(title=title, height=470, legend_title="")
    style_figure(fig, 470, time_axis=True)
    st.plotly_chart(fig, use_container_width=True)


def forecast_controls(key_prefix):
    return st.checkbox(
        "显示研究预测",
        value=False,
        key=f"{key_prefix}_show_forecast",
        help="默认关闭；只作为研究参考，不作为交易信号。",
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
    control_cols = st.columns([1, 1, 1])
    timeframe = control_cols[0].selectbox(
        "K线周期",
        ["1H", "1D", "1W", "1M", "1Q", "1Y"],
        index=1,
        key=f"{key_prefix}_timeframe",
        help="小时线优先使用实时数据；周/月/季/年线由日线重采样。",
    )
    ticker_prices = prices.loc[prices["ticker"] == ticker].copy()
    ticker_prices["date"] = pd.to_datetime(ticker_prices["date"], errors="coerce")
    min_date = ticker_prices["date"].min().date()
    max_date = ticker_prices["date"].max().date()
    as_of = control_cols[1].date_input(
        "回放到",
        value=max_date,
        min_value=min_date,
        max_value=max_date,
        key=f"{key_prefix}_asof",
        help="把这一天当作当前时点，只显示当时已经出现的数据。",
    )
    with control_cols[2]:
        show_forecast = forecast_controls(key_prefix)
    return timeframe, as_of, show_forecast


def page_dashboard(prices, crypto, market_universe, scored, finance_research, risk_rules, source_status):
    page_header(
        "RunJin / Manifested Discipline",
        "润金交易系统",
        "A dark research cockpit for long-term compounding and short-term paper-trading discipline. The interface is designed as a daily operating room: narrative, risk, signal, and review stay visible without turning into noise.",
        ["金水相生", "实时数据", "不加杠杆", "不实盘下单"],
    )
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

    alerts = []
    for name, status, reason in [("US stock bot", stock_status, stock_reason), ("Crypto bot", crypto_status, crypto_reason)]:
        if status != "CONTINUE":
            alerts.append({"system": name, "status": status, "reason": reason})

    render_kpi_grid(
        [
            {"label": "股票池", "value": fmt_int(len(market_universe)), "note": "可研究标的"},
            {"label": "深度研究", "value": fmt_int((scored["bucket"] == "Deep research candidate").sum()), "note": "高优先级候选"},
            {"label": "美股模拟", "value": status_cn(stock_status), "note": stock_reason},
            {"label": "加密模拟", "value": status_cn(crypto_status), "note": crypto_reason},
        ]
    )

    equity = pd.DataFrame(
        {
            "date": stock_bt["date"],
            "US stock bot": stock_bt["equity"],
        }
    )
    equity = coerce_datetime_key(equity, "date")
    chart_equity = equity.tail(1260)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_equity["date"], y=chart_equity["US stock bot"], name="US stock bot", line=dict(color="#2ed17c")))
    if not crypto_bt.empty:
        crypto_equity = crypto_bt[["datetime", "equity"]].rename(columns={"datetime": "date", "equity": "Crypto bot"})
        crypto_equity = coerce_datetime_key(crypto_equity, "date")
        fig.add_trace(go.Scatter(x=crypto_equity.tail(1260)["date"], y=crypto_equity.tail(1260)["Crypto bot"], name="Crypto bot", line=dict(color="#d8cb5f")))
    fig.update_layout(title="模拟账户曲线（近5年）", yaxis_title="USD")
    style_figure(fig, 340, time_axis=True)

    left_col, right_col = st.columns([1.45, 0.8])
    with left_col:
        section_label("账户曲线")
        st.plotly_chart(fig, use_container_width=True)
        candidate_view = watchlist_display(scored[["ticker", "company", "score_label", "bucket", "growth_evidence"]].head(6))
        named_table("今日优先观察", candidate_view)
    with right_col:
        section_label("今日处理")
        top_rows = scored.head(3)
        st.markdown('<div class="runjin-list-title">优先股票</div>', unsafe_allow_html=True)
        for row in top_rows.itertuples():
            st.markdown(
                '<div class="runjin-list-item">'
                f'<div class="runjin-list-title">{escape(str(row.ticker))} · {escape(str(row.company))}</div>'
                f'<div class="runjin-list-meta">{escape(str(row.score_label))} / {escape(bucket_cn(row.bucket))}</div>'
                "</div>",
                unsafe_allow_html=True,
            )
        st.markdown('<div style="height:14px"></div><div class="runjin-list-title">风险提醒</div>', unsafe_allow_html=True)
        if alerts:
            for alert in alerts:
                st.markdown(
                '<div class="runjin-list-item">'
                f'<div class="runjin-list-title">{escape(str(alert["system"]))} · {escape(status_cn(alert["status"]))}</div>'
                f'<div class="runjin-list-meta">{escape(str(alert["reason"]))}</div>'
                "</div>",
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                '<div class="runjin-list-item"><div class="runjin-list-title">风险正常</div>'
                '<div class="runjin-list-meta">当前模拟盘没有触发暂停条件</div></div>',
                unsafe_allow_html=True,
            )
    high_priority = finance_research.loc[finance_research["importance"] >= 4].head(4)
    if not high_priority.empty:
        with st.expander("高优先级研究输入", expanded=False):
            named_table("Research inputs", high_priority)
    render_data_source_debug(source_status)


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
    section_label("Long-Term Observation Discipline")
    checklist_cols = st.columns(3)
    with checklist_cols[0]:
        render_research_checklist("Continue / Hold", HOLD_OBSERVATION_CONDITIONS)
    with checklist_cols[1]:
        render_research_checklist("Exit / Stop Tracking", EXIT_OBSERVATION_CONDITIONS)
    with checklist_cols[2]:
        render_research_checklist("Tenbagger Discovery", TENBAGGER_DISCOVERY_FRAMEWORK, " + ")

    bucket_filter = st.multiselect("Bucket", sorted(scored["bucket"].unique()), default=list(sorted(scored["bucket"].unique())))
    view = scored.loc[scored["bucket"].isin(bucket_filter)].copy()
    named_table(
        "Ranked long-term observation pool",
        watchlist_display(view[
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
        ]),
    )

    section_label("Score Breakdown")
    selected = st.selectbox("Ticker", view["ticker"].tolist())
    row = view.loc[view["ticker"] == selected].iloc[0]
    score_df = pd.DataFrame({"component": SCORE_COLUMNS, "score": [row[col] for col in SCORE_COLUMNS]})
    fig = go.Figure(go.Bar(x=score_df["component"], y=score_df["score"], marker_color="#2ed17c"))
    fig.update_layout(title=f"{selected} component score", yaxis=dict(range=[0, 5]))
    style_figure(fig, 340)
    st.plotly_chart(fig, use_container_width=True)


def page_tenbagger_profile(prices, financials, scored, industry_map, forecasts, data_mode):
    page_header(
        "Future 10x Map / One Stock at a Time",
        "未来10倍股 Profile",
        "A long-term research cockpit that starts from future industries, value-chain layers, monopoly potential, gross-margin power, and upstream/downstream relationships before looking at price.",
        ["10-year", "Value chain", "Moat", "Peers", "Proof"],
    )
    if industry_map.empty:
        st.warning("No future industry map is available.")
        return
    theme_filter = st.selectbox("Future industry", ["ALL"] + sorted(industry_map["mega_theme"].dropna().unique().tolist()))
    filtered = industry_map if theme_filter == "ALL" else industry_map.loc[industry_map["mega_theme"] == theme_filter]
    ticker = st.selectbox("Stock", filtered["ticker"].drop_duplicates().tolist(), key="tenbagger_ticker")
    profile = build_future_profile(industry_map, scored, ticker)
    if not profile:
        st.warning(f"No future profile is configured for {ticker}.")
        return

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mega theme", profile.get("mega_theme", "N/A"))
    col2.metric("Chain layer", profile.get("chain_layer", "N/A"))
    col3.metric("Moat score", f"{profile.get('moat_total', 0):.0f} / 20")
    col4.metric("Moat label", profile.get("moat_label", "N/A"))

    st.markdown(
        f"""
        <div class="runjin-thesis">
          <h4>{profile.get("ticker")} / {profile.get("company")}</h4>
          <p><strong>Role:</strong> {profile.get("chain_role", "")}</p>
          <p><strong>Why it can 10x:</strong> {profile.get("why_it_can_10x", "")}</p>
          <p><strong>Key question:</strong> {profile.get("key_questions", "")}</p>
          <p><strong>Risk flags:</strong> {profile.get("risk_flags", "")}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    discovery_cols = st.columns(3)
    with discovery_cols[0]:
        render_research_checklist("Tenbagger Discovery", TENBAGGER_DISCOVERY_FRAMEWORK, " + ")
    with discovery_cols[1]:
        render_research_checklist("Continue / Hold", HOLD_OBSERVATION_CONDITIONS)
    with discovery_cols[2]:
        render_research_checklist("Exit / Stop Tracking", EXIT_OBSERVATION_CONDITIONS)

    score_cols = ["monopoly_score", "gross_margin_power", "irreplaceability", "ten_year_optionality", "capital_intensity"]
    score_df = pd.DataFrame(
        {
            "dimension": ["Monopoly", "Gross margin", "Irreplaceable", "10Y optionality", "Capital intensity"],
            "score": [float(profile.get(col, 0) or 0) for col in score_cols],
        }
    )
    fig = go.Figure(go.Bar(x=score_df["dimension"], y=score_df["score"], marker_color=["#2ed17c", "#d8cb5f", "#a891ff", "#ff78b7", "#ff5f64"]))
    fig.update_layout(title=f"{ticker} value-chain quality", yaxis=dict(range=[0, 5]))
    style_figure(fig, 330)
    st.plotly_chart(fig, use_container_width=True)

    tabs = st.tabs(["Industry Map", "Relationship Compare", "Company Proof", "K-line Context"])
    with tabs[0]:
        theme_summary = future_theme_summary(industry_map)
        named_table("Future industry map", theme_summary.round(2) if not theme_summary.empty else pd.DataFrame())
        layer_scores = industry_layer_scores(industry_map, ticker)
        if not layer_scores.empty:
            fig_layer = go.Figure(
                go.Bar(
                    x=layer_scores["chain_layer"],
                    y=layer_scores["avg_moat"],
                    text=layer_scores["tickers"],
                    marker_color=["#2ed17c" if selected else "#303631" for selected in layer_scores["selected"]],
                )
            )
            fig_layer.update_layout(title=f"{profile.get('mega_theme')} value-chain layer quality", yaxis_title="Average moat score")
            style_figure(fig_layer, 330)
            st.plotly_chart(fig_layer, use_container_width=True)
            named_table("Theme value-chain layers", layer_scores.round(2))
    with tabs[1]:
        compare = relationship_comparison(industry_map, scored, ticker)
        display_cols = [
            "relationship",
            "ticker",
            "company",
            "chain_layer",
            "chain_role",
            "moat_total",
            "score_label",
            "why_it_can_10x",
            "key_questions",
        ]
        named_table("Upstream / peers / downstream comparison", compare[[col for col in display_cols if col in compare.columns]].round(2))
    with tabs[2]:
        scored_row = scored.loc[scored["ticker"] == ticker] if ticker in set(scored["ticker"]) else pd.DataFrame()
        ticker_financials = financials.loc[financials["ticker"] == ticker].sort_values("quarter") if ticker in set(financials["ticker"]) else pd.DataFrame()
        if not scored_row.empty:
            row = scored_row.iloc[0]
            c1, c2, c3 = st.columns(3)
            c1.metric("Long score", row["score_label"])
            c2.metric("Bucket", row["bucket"])
            c3.metric("Value-chain score", f"{row['value_chain_position']} / 5")
            named_table("Long thesis", scored_row[["ticker", "company", "narrative", "growth_evidence", "catalysts", "invalidation", "buy_plan"]])
        else:
            st.info("This ticker is in the future industry map but not yet in the scored long watchlist.")
        if not ticker_financials.empty:
            named_table("Financial proof", ticker_financials)
    with tabs[3]:
        raw_kline, source_note = build_ticker_kline(prices, ticker, "1D", None, data_mode)
        if raw_kline.empty:
            st.info(f"No K-line data loaded for {ticker}. {source_note}")
        else:
            ticker_prices = add_indicators(raw_kline)
            forecast = filter_forecast_for_chart(forecasts.loc[forecasts["ticker"] == ticker], ticker_prices) if ticker in set(forecasts["ticker"]) else pd.DataFrame()
            st.markdown(f'<div class="runjin-note">K-line is context only: {coverage_summary(ticker_prices)} / {source_note}</div>', unsafe_allow_html=True)
            plot_kline(ticker_prices, f"{ticker} long-term K-line context", forecast)


def plot_market_kline(symbol_df, ticker, market):
    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=symbol_df["date"],
            open=symbol_df["open"],
            high=symbol_df["high"],
            low=symbol_df["low"],
            close=symbol_df["close"],
            name="OHLC",
            increasing_line_color="#2ed17c",
            increasing_fillcolor="rgba(46,209,124,0.50)",
            decreasing_line_color="#ff5f64",
            decreasing_fillcolor="rgba(255,95,100,0.50)",
        )
    )
    for ma, color in [("ma20", "#d8cb5f"), ("ma60", "#a891ff"), ("ma120", "#ff78b7")]:
        if ma in symbol_df:
            fig.add_trace(go.Scatter(x=symbol_df["date"], y=symbol_df[ma], name=ma.upper(), line=dict(width=1.4, color=color)))
    fig.update_layout(title=f"{ticker} {market} K-line / research only", height=430)
    style_figure(fig, 430, time_axis=True)
    st.plotly_chart(fig, use_container_width=True)


def page_finance_mcp_radar(finance_research, scored, source_status, prices, crypto):
    page_header(
        "Data Edge Radar / Multi-Market Quant Desk",
        "全市场情报 + 多市场智能量化工作台",
        "FinanceMCP-style intelligence remains the macro data edge; TickFlow-style screening, rotation, surge ladders, key levels, and monitor rules now cover A-share, US stocks, Hong Kong stocks, and crypto in one research-only workspace.",
        ["A / US / HK / Crypto", "Offline ready", "No auto orders"],
    )

    status = source_status.get("finance_mcp", {})
    st.markdown(
        f'<div class="runjin-note">FinanceMCP adapter: {status.get("mode", "unknown")} / {status.get("source", "unknown")} / {status.get("message", "")}. Multi-market workbench uses bundled samples by default and is research-only.</div>',
        unsafe_allow_html=True,
    )

    market_prices, market_meta = build_market_workbench_data(prices, crypto)
    enriched = add_market_indicators(market_prices, market_meta)
    screened = screen_market(enriched)
    rotation = market_rotation(enriched)
    ladder = surge_ladder(enriched)
    markets = ["ALL"] + [item for item in ["A_SHARE", "US", "HK", "CRYPTO"] if item in set(enriched["market"])]

    section = st.radio(
        "Workbench section",
        ["智能选股", "情绪/突破梯队", "板块轮动", "关键价位", "监控规则", "Research Radar"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if section == "智能选股":
        strategy_options = ["ALL"] + list(STRATEGY_DESCRIPTIONS.keys())
        col1, col2, col3, col4 = st.columns([0.9, 1.2, 1, 1])
        market_filter = col1.selectbox("Market", markets)
        strategy = col2.selectbox("Strategy template", strategy_options)
        min_score = col3.slider("Min signal score", 0, 60, 24)
        concept_options = ["ALL"] + sorted(screened.loc[screened["market"].eq(market_filter) if market_filter != "ALL" else screened["market"].notna(), "concept"].dropna().unique().tolist())
        concept_filter = col4.selectbox("Theme", concept_options)
        view = filter_screener(screened, market_filter, strategy, min_score)
        if concept_filter != "ALL":
            view = view.loc[view["concept"] == concept_filter]
        description = STRATEGY_DESCRIPTIONS.get(strategy, "显示所有 TickFlow-style 选股模板命中的多市场候选。")
        st.markdown(f'<div class="runjin-note">{description}</div>', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Candidates", fmt_int(len(view)))
        col2.metric("Avg score", fmt_number(view["score"].mean() if not view.empty else 0, 0))
        col3.metric("Surge hits", fmt_int(view["strong_surge"].sum() if not view.empty else 0))
        col4.metric("Markets", fmt_int(view["market"].nunique() if not view.empty else 0))
        table_cols = ["market", "ticker", "company", "concept", "score", "strategies", "reasons", "close", "return_1d", "momentum_20", "volume_ratio", "rsi14", "volatility_20"]
        named_table("Multi-market TickFlow-style screener", view[table_cols].round(4) if not view.empty else pd.DataFrame([{"note": "No candidates match current filters"}]))
        if not view.empty:
            top = view.head(12)
            fig = go.Figure(go.Bar(x=top["ticker"], y=top["score"], marker_color="#2ed17c", text=top["market"] + " / " + top["concept"]))
            fig.update_layout(title="Top multi-market signal scores", yaxis_title="Score")
            style_figure(fig, 330)
            st.plotly_chart(fig, use_container_width=True)

    elif section == "情绪/突破梯队":
        market_filter = st.selectbox("Market", markets, key="surge_market")
        view = ladder if market_filter == "ALL" else ladder.loc[ladder["market"] == market_filter]
        col1, col2, col3 = st.columns(3)
        col1.metric("Triggered symbols", fmt_int((view["surge_streak"] > 0).sum()))
        col2.metric("Max streak", fmt_int(view["surge_streak"].max() if not view.empty else 0))
        col3.metric("Hot themes", fmt_int(view.loc[view["surge_streak"] > 0, "concept"].nunique() if not view.empty else 0))
        st.markdown(
            '<div class="runjin-note">A股显示涨停/连板语义；美股、港股、加密货币使用强势上涨、放量和60日新高构造突破梯队。这是情绪研究，不生成交易指令。</div>',
            unsafe_allow_html=True,
        )
        named_table("Surge / breakout ladder", view.round(4))

    elif section == "板块轮动":
        market_filter = st.selectbox("Market", markets, key="rotation_market")
        view = rotation if market_filter == "ALL" else rotation.loc[rotation["market"] == market_filter]
        named_table("Theme rotation", view.round(4))
        if not view.empty:
            fig = go.Figure()
            x_labels = view["market"] + " / " + view["concept"]
            fig.add_trace(go.Bar(x=x_labels, y=view["avg_20d"], name="20D return", marker_color="#2ed17c"))
            fig.add_trace(go.Scatter(x=x_labels, y=view["avg_volume_ratio"], name="Volume ratio", yaxis="y2", line=dict(color="#d8cb5f", width=2)))
            fig.update_layout(title="Theme rotation: price momentum + volume heat", yaxis_title="20D return", yaxis2=dict(overlaying="y", side="right", title="Volume ratio"))
            style_figure(fig, 380)
            st.plotly_chart(fig, use_container_width=True)

    elif section == "关键价位":
        market_filter = st.selectbox("Market", [item for item in markets if item != "ALL"], key="levels_market")
        tickers = sorted(enriched.loc[enriched["market"] == market_filter, "ticker"].unique())
        default_symbol = {"A_SHARE": "688981", "US": "NVDA", "HK": "0700.HK", "CRYPTO": "BTC-USD"}.get(market_filter, tickers[0])
        default_idx = tickers.index(default_symbol) if default_symbol in tickers else 0
        ticker = st.selectbox("Symbol", tickers, index=default_idx)
        symbol_df = enriched.loc[enriched["ticker"] == ticker].copy()
        plot_market_kline(symbol_df.tail(180), ticker, market_filter)
        levels = key_price_levels(symbol_df)
        col1, col2 = st.columns([1.2, 1])
        with col1:
            named_table("Key price levels", levels.round(4))
        with col2:
            latest = symbol_df.tail(1).iloc[0]
            verdict = monitor_verdict(latest)
            st.metric("Monitor status", verdict["status"].upper())
            st.markdown(f'<div class="runjin-note">{verdict["alerts"]}</div>', unsafe_allow_html=True)
            snapshot_cols = ["date", "close", "ma20", "ma60", "ma120", "rsi14", "volume_ratio", "atr_pct", "momentum_20", "momentum_60"]
            named_table("Latest signal snapshot", symbol_df[snapshot_cols].tail(1).round(4))

    elif section == "监控规则":
        market_filter = st.selectbox("Market", markets, key="monitor_market")
        latest = market_latest_snapshot(enriched)
        if market_filter != "ALL":
            latest = latest.loc[latest["market"] == market_filter]
        verdicts = latest.apply(lambda row: monitor_verdict(row), axis=1, result_type="expand")
        monitor = pd.concat([latest[["market", "ticker", "company", "concept", "date", "close", "rsi14", "volume_ratio", "volatility_20", "strong_surge"]].reset_index(drop=True), verdicts], axis=1)
        status_filter = st.selectbox("Status", ["ALL"] + sorted(monitor["status"].unique().tolist()))
        if status_filter != "ALL":
            monitor = monitor.loc[monitor["status"] == status_filter]
        col1, col2, col3 = st.columns(3)
        col1.metric("Monitor rows", fmt_int(len(monitor)))
        col2.metric("Review/Risk", fmt_int(monitor["status"].isin(["review", "risk"]).sum()))
        col3.metric("Volume alerts", fmt_int((monitor["volume_ratio"] >= 2).sum()))
        named_table("Multi-market monitor rules", monitor.sort_values(["status", "volume_ratio"], ascending=[True, False]).round(4))

    else:
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

    section_label("Observation Checklist")
    observation_cols = st.columns(3)
    with observation_cols[0]:
        render_research_checklist("Continue / Hold", HOLD_OBSERVATION_CONDITIONS)
    with observation_cols[1]:
        render_research_checklist("Exit / Stop Tracking", EXIT_OBSERVATION_CONDITIONS)
    with observation_cols[2]:
        render_research_checklist("Tenbagger Discovery", TENBAGGER_DISCOVERY_FRAMEWORK, " + ")

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
    ticker = st.selectbox("股票", tickers, index=tickers.index("NVDA") if "NVDA" in tickers else 0)
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
    benchmark_symbol = "SPY" if "SPY" in set(prices["ticker"]) else "TSM"
    benchmark_raw = prices.loc[prices["ticker"] == benchmark_symbol].copy()
    benchmark = apply_asof(resample_ohlcv(benchmark_raw, timeframe, "date"), as_of) if not benchmark_raw.empty else pd.DataFrame()
    rs = relative_strength(ticker_prices, benchmark) if ticker != "TSM" and not benchmark.empty else pd.DataFrame()

    latest_indicator = ticker_prices.dropna().tail(1)
    kdj_state = "数据不足"
    if not latest_indicator.empty:
        latest_row = latest_indicator.iloc[0]
        if latest_row["kdj_j"] > 100:
            kdj_state = "偏高"
        elif latest_row["kdj_j"] < 0:
            kdj_state = "超卖"
        elif latest_row["kdj_k"] > latest_row["kdj_d"]:
            kdj_state = "转强"
        else:
            kdj_state = "降温"

    regime_label = {
        "Range or transition": "震荡/转换",
        "Uptrend": "上升趋势",
        "Downtrend": "下降趋势",
        "High volatility": "高波动",
    }.get(classify_regime(ticker_prices), classify_regime(ticker_prices))
    latest_rsi = ticker_prices["rsi14"].dropna()
    latest_volatility = ticker_prices["volatility_20"].dropna()
    render_kpi_grid(
        [
            {"label": "结构", "value": regime_label, "note": "趋势状态"},
            {"label": "RSI", "value": f"{latest_rsi.iloc[-1]:.0f}" if not latest_rsi.empty else "N/A", "note": "动量温度"},
            {"label": "20日波动", "value": fmt_pct(latest_volatility.iloc[-1]) if not latest_volatility.empty else "N/A", "note": "风险强度"},
            {"label": "KDJ", "value": kdj_state, "note": "短线状态"},
        ]
    )

    with st.expander("数据说明", expanded=False):
        st.caption(f"{coverage_summary(ticker_prices)} / {source_note}")
    volume_price_context = f"{ticker}|{timeframe}|{as_of}"
    if st.button("Show volume-price state", key="vp_show"):
        st.session_state["volume_price_context"] = volume_price_context
    show_volume_price = st.session_state.get("volume_price_context") == volume_price_context
    volume_price = analyze_volume_price_state(ticker_prices, benchmark) if show_volume_price else None
    markers = volume_price["markers"] if volume_price else None
    plot_kline(ticker_prices, f"{ticker} {timeframe} Indicators / Replay as of {as_of}", forecast, markers)

    if volume_price:
        st.markdown(f'<div class="runjin-note">{volume_price["summary"]}</div>', unsafe_allow_html=True)
        signal_view = volume_price["signals"].head(30).copy()
        if not signal_view.empty:
            signal_view["date"] = pd.to_datetime(signal_view["date"]).dt.strftime("%Y-%m-%d")
            named_table(
                "Volume-price rule marks",
                signal_view[["date", "close", "volume_ratio", "price_zone", "volume_state", "price_state", "label", "action", "note"]].round(4),
            )
        else:
            named_table("Volume-price rule marks", pd.DataFrame([{"note": "No volume-price rule was triggered in the recent window"}]))

    indicator_view = ticker_prices.tail(30)[
        ["date", "close", "ma20", "ma60", "rsi14", "macd", "macd_signal", "bb_lower", "bb_upper", "kdj_k", "kdj_d", "kdj_j"]
    ].copy()
    named_table("Latest indicator values", indicator_view.round(2))

    if not rs.empty:
        fig = go.Figure(go.Scatter(x=rs["date"], y=rs["relative_strength"], name=f"{ticker} vs TSM"))
        fig.update_layout(title=f"Relative Strength: {ticker} vs TSM")
        style_figure(fig, 320, time_axis=True)
        st.plotly_chart(fig, use_container_width=True)

    section_label("Launch Point Research")
    launch = launch_point_analysis(ticker_prices)
    st.markdown(
        f'<div class="runjin-note">{launch["summary"]}<br>Charts are records of market participants\' expectations, fear, greed, and forced behavior. Similar patterns are useful research context, not deterministic prophecy.</div>',
        unsafe_allow_html=True,
    )
    launch_signals = launch["signals"]
    if not launch_signals.empty:
        signal_view = launch_signals.head(20).copy()
        signal_view["date"] = pd.to_datetime(signal_view["date"]).dt.strftime("%Y-%m-%d")
        named_table("Launch point candidates", signal_view)
    else:
        named_table("Launch point candidates", pd.DataFrame([{"note": "No major launch-point candidate in the current window"}]))
    col_levels, col_plan = st.columns(2)
    with col_levels:
        levels = launch["levels"]
        named_table("Original-control proxy levels", levels.round(3) if not levels.empty else pd.DataFrame([{"note": "No nearby high-evidence pivot levels"}]))
    with col_plan:
        named_table("Launch risk plan", launch["plan"])

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


def page_ollama_research_assistant(scored, financials):
    page_header(
        "Local AI / Research Assistant",
        "投研小助手",
        "Ask a local Ollama model to pressure-test the current thesis, invalidation points, and next research checks.",
        ["本地模型", "投研问答", "不下单"],
    )
    ollama_config = load_ollama_config()
    base_url = ollama_config["base_url"]
    default_model = ollama_config["default_model"]
    status = ollama_status(base_url=base_url)
    status_label = "已连接" if status["available"] else "未连接"
    col1, col2, col3 = st.columns(3)
    col1.metric("Ollama", status_label)
    col2.metric("默认模型", default_model)
    col3.metric("本地地址", base_url.replace("http://", ""))

    if not status["available"]:
        if "localhost" in base_url or "127.0.0.1" in base_url:
            st.warning("当前是线上页面，不能直接访问你电脑里的本地 Ollama。要使用本地模型，请在本机运行本系统，或把 Ollama 地址改成可被线上访问的远程地址。")
        else:
            st.warning("Ollama 还没有连上。请到 `系统能力 -> 模型能力` 设置地址和模型，然后测试连接。")
        st.markdown(
            f'<div class="runjin-note">当前配置：<strong>{base_url}</strong> / <strong>{default_model}</strong></div>',
            unsafe_allow_html=True,
        )
        with st.expander("错误信息", expanded=False):
            st.code(status["message"])
        return

    models = status["models"] or [default_model]
    default_index = models.index(default_model) if default_model in models else 0
    model = st.selectbox("模型", models, index=default_index)
    ticker = st.selectbox("股票", scored["ticker"].tolist(), key="ollama_ticker")
    profile = scored.loc[scored["ticker"] == ticker].iloc[0]
    ticker_financials = financials.loc[financials["ticker"] == ticker].sort_values("quarter")

    quick_questions = {
        "这只票现在最该验证什么？": "这只股票当前最关键的验证点是什么？请按产业、公司、财务、价格结构拆解。",
        "继续跟踪还是降级？": "根据继续跟踪条件和放弃条件，这只股票应该继续跟踪、降级观察还是暂时剔除？",
        "牛股框架打分": "用识别牛股框架逐项评估这只股票，每项给出通过/待验证/不通过。",
        "找反方证据": "请站在反方角度，列出最可能证伪这个长期 thesis 的证据。",
    }
    selected_question = st.radio("常用问题", list(quick_questions.keys()), horizontal=True)
    question = st.text_area(
        "你的问题",
        value=quick_questions[selected_question],
        height=110,
        key="ollama_question",
    )

    if st.button("问小助手", type="primary", key="ollama_ask"):
        with st.spinner("本地模型思考中..."):
            prompt = build_ollama_research_prompt(ticker, profile, ticker_financials, question)
            try:
                answer = ask_ollama(prompt, model=model, base_url=base_url)
            except Exception as exc:
                st.error(f"Ollama 调用失败：{exc}")
                return
        st.markdown(answer)

    with st.expander("当前投研上下文", expanded=False):
        st.markdown(f"**{ticker} / {profile['company']}**")
        st.markdown(f"- 叙事：{profile['narrative']}")
        st.markdown(f"- 成长证据：{profile['growth_evidence']}")
        st.markdown(f"- 失效条件：{profile['invalidation']}")
        st.markdown(f"- 买入计划：{profile['buy_plan']}")


def page_single_stock_sync():
    page_header(
        "Data Sync / Single Stock",
        "单股同步",
        "Sync one quote with visible main route, fallback route, failure reason, and market_quotes persistence state.",
        ["主链路", "回退链路", "落库状态"],
    )
    col1, col2, col3 = st.columns([1, 1, 1])
    symbol = col1.text_input("代码", value="NVDA", key="sync_symbol")
    market = col2.selectbox("市场", ["US", "A_SHARE", "HK", "CN"], index=0, key="sync_market")
    ttl = col3.number_input("缓存秒数", min_value=0, max_value=3600, value=120, step=30, key="sync_ttl")
    if st.button("同步", type="primary", key="sync_single_quote"):
        with st.spinner("同步行情中..."):
            result = sync_single_quote(symbol.strip(), market=market, cache_ttl_seconds=int(ttl))
        st.session_state["single_sync_result"] = result

    result = st.session_state.get("single_sync_result")
    if result:
        summary, attempts, quotes = result_to_frames(result)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("状态", result["status"])
        c2.metric("主链路", result["main_route"])
        c3.metric("回退链路", result["fallback_route"])
        c4.metric("market_quotes", "已写入" if result["market_quotes_saved"] else "未写入")
        named_table("Sync summary", summary)
        named_table("Route attempts", attempts)
        named_table("market_quotes row", quotes)


def page_batch_analysis(scored, financials):
    page_header(
        "Batch Research / Multi Stock",
        "批量分析",
        "Analyze multiple watchlist names in one pass and rank by long-term score plus latest financial proof.",
        ["多股", "排序", "进度"],
    )
    default_tickers = scored.head(5)["ticker"].tolist()
    tickers = st.multiselect("股票", scored["ticker"].tolist(), default=default_tickers, key="batch_long_tickers")
    progress = analysis_progress_rows(tickers)
    named_table("Analysis progress", progress)
    if st.button("开始批量分析", type="primary", key="batch_long_run"):
        with st.spinner("批量分析中..."):
            result = batch_long_analysis(scored, financials, tickers)
        st.session_state["batch_long_result"] = result
    result = st.session_state.get("batch_long_result")
    if result is not None:
        view = result.copy()
        for col in ["revenue_growth_yoy", "gross_margin", "ocf_margin"]:
            if col in view:
                view[col] = view[col].map(lambda value: fmt_pct(value) if pd.notna(value) else "N/A")
        named_table("Batch long analysis", view)


def page_astock_agent_framework(scored):
    page_header(
        "A-Share Agents / Decision Chain",
        "A股研判",
        "把市场、情绪、新闻、基本面、政策、资金和解禁放进同一条决策链，先判断是否值得继续跟踪，再决定仓位和失效条件。",
        ["A股约束", "多空辩论", "风险分层"],
    )
    tables = astock_agent_tables()

    symbol_options = scored["ticker"].tolist() if "ticker" in scored else []
    default_symbol = symbol_options[0] if symbol_options else "300124.SZ"
    natural_task = st.text_input(
        "自然语言任务",
        value="分析汇川技术中线还能不能继续跟踪",
        key="ashare_natural_task",
    )
    col1, col2, col3 = st.columns([1, 1, 2])
    symbol = col1.selectbox("股票", symbol_options or [default_symbol], key="astock_agent_symbol")
    horizon = col2.selectbox("周期", ["短线", "中线", "长线"], index=1, key="astock_agent_horizon")
    thesis = col3.text_input(
        "核心假设",
        value="产业趋势向上，公司位置靠前，财报逐步验证，估值仍有安全边际，周/月线开始止跌放量。",
        key="astock_agent_thesis",
    )

    prompt = astock_decision_prompt(symbol, f"{natural_task}。{thesis}", horizon)
    render_kpi_grid(
        [
            {"label": "入口", "value": "一句话", "note": "先说问题，再选股票"},
            {"label": "第一步", "value": "7类分析", "note": "先拆证据"},
            {"label": "第二步", "value": "多空辩论", "note": "主动找反证"},
            {"label": "输出", "value": "研报卡", "note": "动作、仓位、失效位"},
        ]
    )

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["研判链路", "分析师分工", "研报卡片", "信号验证", "系统保障", "A股约束", "提示词"])
    with tab1:
        named_table("决策流程", tables["pipeline"])
        named_table("Vibe 研究流", tables["vibe_research_flow"])
        named_table("日常跟踪闭环", tables["operating_loop"])
        with st.expander("数据源分工", expanded=False):
            named_table("A股数据路由", tables["data_routes"])
    with tab2:
        named_table("7类分析师检查表", tables["analysts"])
    with tab3:
        named_table("结构化研报字段", tables["report_card"])
        with st.expander("自然语言任务示例", expanded=False):
            named_table("任务解析示例", tables["intent_examples"])
    with tab4:
        named_table("信号准入闸门", tables["signal_gates"])
        named_table("策略操作系统", tables["strategy_os"])
        named_table("模拟信号生命周期", tables["paper_signal_lifecycle"])
        with st.expander("策略代码契约", expanded=False):
            named_table("策略代码契约", tables["strategy_code_contract"])
        with st.expander("执行安全模型", expanded=False):
            named_table("执行安全模型", tables["safety_model"])
        with st.expander("跟单边界", expanded=False):
            named_table("跟单与排行榜边界", tables["copy_trading_boundaries"])
    with tab5:
        named_table("TradingAgents 原版可吸收能力", tables["upstream_practices"])
        named_table("决策记忆字段", tables["decision_memory"])
        named_table("影子账户复盘", tables["shadow_account"])
    with tab6:
        named_table("A股交易约束", tables["constraints"])
    with tab7:
        st.text_area("可复制给投研小助手的提示词", value=prompt, height=360, key="astock_agent_prompt")
        if st.button("用 Ollama 跑一版研判", type="primary", key="astock_agent_ollama"):
            config = load_ollama_config()
            status = ollama_status(base_url=config["base_url"])
            if not status["available"]:
                st.warning("Ollama 还没有连接。线上 Streamlit 页面不能访问你本机的 localhost，需要本地运行系统或配置远程可访问的 Ollama。")
            else:
                with st.spinner("正在生成 A股研判..."):
                    answer = ask_ollama(prompt, model=config["default_model"], base_url=config["base_url"])
                st.markdown(answer)


def page_system_capabilities(source_status):
    page_header(
        "System / Capabilities",
        "系统能力",
        "Hidden operating layer for model providers, task policies, cache status, and data-source diagnostics.",
        ["模型", "缓存", "数据"],
    )
    tabs = st.tabs(["模型能力", "缓存管理", "数据源状态", "导出能力"])
    with tabs[0]:
        section_label("Ollama Settings")
        ollama_config = load_ollama_config()
        with st.form("ollama_settings_form"):
            base_url = st.text_input("Ollama 地址", value=ollama_config["base_url"], help="通常是 http://localhost:11434")
            default_model = st.text_input("默认模型", value=ollama_config["default_model"], help="例如 qwen2.5:7b")
            save_clicked = st.form_submit_button("保存设置")
        if save_clicked:
            path = save_ollama_config(base_url, default_model)
            st.success(f"已保存：{path}")

        test_config = load_ollama_config()
        if st.button("测试 Ollama 连接", key="test_ollama_connection"):
            status = ollama_status(base_url=test_config["base_url"])
            if status["available"]:
                st.success("Ollama 已连接")
                named_table("Installed Ollama models", pd.DataFrame([{"model": model} for model in status["models"]]))
            else:
                st.error("Ollama 未连接")
                st.code(status["message"])
        st.markdown(
            f'<div class="runjin-note">如果本机没有模型，先在终端执行：<strong>ollama pull {test_config["default_model"]}</strong></div>',
            unsafe_allow_html=True,
        )

        section_label("Provider Catalog")
        named_table("LLM providers", pd.DataFrame(provider_table()))
        named_table("Task model policy", pd.DataFrame(task_policy_table()))
    with tabs[1]:
        cache = CacheManager()
        entries = pd.DataFrame(cache.list_entries())
        named_table("Cache entries", entries if not entries.empty else pd.DataFrame([{"note": "No cache entries yet"}]))
        namespace = st.text_input("清理命名空间", value="quote", key="cache_namespace")
        if st.button("清理缓存", key="cache_clear"):
            deleted = cache.clear_namespace(namespace.strip())
            st.success(f"已清理 {deleted} 个缓存文件")
    with tabs[2]:
        render_data_source_debug(source_status)
    with tabs[3]:
        named_table("Report export capabilities", pd.DataFrame(report_export_capabilities()))


def page_backtest_lab(prices, crypto):
    page_header(
        "Backtest Lab / Strategy Proof",
        "策略回测平台",
        "A safer backtesting desk built on backtesting.py: you define a constrained YAML strategy spec, the engine runs on OHLCV bars, and the output shows equity, drawdown, trades, and robustness warnings. V0.1 does not execute arbitrary Python.",
        ["策略验证", "不加杠杆", "研究用途"],
    )

    asset_class = st.radio("资产类型", ["美股", "加密"], horizontal=True, key="bt_asset_class")
    if asset_class == "美股":
        ticker = st.selectbox("股票", sorted(prices["ticker"].unique()), index=0, key="bt_ticker")
        raw = prices.loc[prices["ticker"] == ticker].copy()
        time_col = "date"
        timeframe = st.selectbox("周期", ["1D", "1W", "1M"], index=0, key="bt_stock_timeframe")
        raw = resample_ohlcv(raw, timeframe, "date")
    else:
        if crypto.empty:
            st.warning("Crypto live data is unavailable. The workspace is fixed to realtime data only, so no sample fallback is shown.")
            return
        ticker = st.selectbox("标的", sorted(crypto["symbol"].unique()), index=0, key="bt_symbol")
        raw = crypto.loc[crypto["symbol"] == ticker].copy()
        time_col = "datetime"
        timeframe = st.selectbox("周期", ["1H", "1D"], index=0, key="bt_crypto_timeframe")
        if timeframe == "1D":
            raw = resample_ohlcv(raw.rename(columns={"datetime": "date"}), "1D", "date").rename(columns={"date": "datetime"})

    raw[time_col] = pd.to_datetime(raw[time_col], errors="coerce")
    min_date = raw[time_col].min().date()
    max_date = raw[time_col].max().date()
    col1, col2, col3 = st.columns([1, 1, 1.2])
    default_start = max(min_date, pd.Timestamp(max_date).to_pydatetime().date().replace(year=max_date.year - 5))
    start_date = col1.date_input("开始", value=default_start, min_value=min_date, max_value=max_date, key="bt_start")
    end_date = col2.date_input("结束", value=max_date, min_value=min_date, max_value=max_date, key="bt_end")
    example_name = col3.selectbox("策略模板", list(EXAMPLE_STRATEGY_SPECS.keys()), key="bt_template")

    if st.button("载入模板", key="bt_load_template"):
        st.session_state["bt_strategy_spec"] = EXAMPLE_STRATEGY_SPECS[example_name]

    default_spec = st.session_state.get("bt_strategy_spec", EXAMPLE_STRATEGY_SPECS.get(example_name, DEFAULT_STRATEGY_SPEC))
    with st.expander("高级参数 YAML", expanded=False):
        strategy_spec = st.text_area(
            "策略 YAML",
            value=default_spec,
            height=260,
            key="bt_strategy_spec",
            help="安全编辑参数。支持 sma_crossover, rsi_mean_reversion, bollinger_reversion, macd_trend。",
        )

    mask = (raw[time_col].dt.date >= start_date) & (raw[time_col].dt.date <= end_date)
    backtest_data = raw.loc[mask].copy()
    st.caption(f"回测数据：{ticker} / {timeframe} / {len(backtest_data):,} 根K线 / {start_date} -> {end_date}")

    if st.button("运行回测", type="primary", key="bt_run"):
        with st.spinner("回测中..."):
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
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        col1.metric("Return", fmt_number(stats.get("Return [%]"), 1, "%"))
        col2.metric("Max drawdown", fmt_number(stats.get("Max. Drawdown [%]"), 1, "%"))
        col3.metric("Sharpe", fmt_number(stats.get("Sharpe Ratio"), 2))
        col4.metric("Info ratio", fmt_number(stats.get("ABU information_ratio"), 2))
        col5.metric("Trades / year", fmt_number(stats.get("ABU trades_per_year"), 1))
        col6.metric("DD duration", f"{fmt_int(stats.get('ABU max_drawdown_duration_days', 0) or 0)}d")
        col7.metric("Trades", fmt_int(stats.get("# Trades", 0) or 0))
        if (stats.get("ABU trades_per_year") is not None and stats.get("ABU trades_per_year") < 4) or (stats.get("ABU max_drawdown_duration_days") is not None and stats.get("ABU max_drawdown_duration_days") > 120):
            st.markdown(
                '<div class="runjin-note">Performance quality warning: low trading frequency or long drawdown duration can make a strategy unsuitable as the main cash-flow engine even when total return looks acceptable.</div>',
                unsafe_allow_html=True,
            )

        if result.ump_verdict is not None and not result.ump_verdict.empty:
            verdict = result.ump_verdict["verdict"].iloc[0]
            st.markdown(
                f'<div class="runjin-note">UMP-lite verdict: <strong>{verdict.upper()}</strong>. Rule-based裁判只做研究拦截提示，不自动下单。</div>',
                unsafe_allow_html=True,
            )
            named_table("UMP-lite verdict", result.ump_verdict)
        if result.assumptions is not None and not result.assumptions.empty:
            named_table("Execution assumptions", result.assumptions)
        if result.transaction_cost_audit is not None and not result.transaction_cost_audit.empty:
            worst_status = "FAIL" if result.transaction_cost_audit["status"].eq("fail").any() else "REVIEW" if result.transaction_cost_audit["status"].eq("review").any() else "PASS"
            st.markdown(
                f'<div class="runjin-note">Transaction cost guard: <strong>{worst_status}</strong>. This estimates commission, half-spread, market impact, and latency slippage in bps, then checks whether costs erase the strategy edge.</div>',
                unsafe_allow_html=True,
            )
            named_table("Transaction cost guard", result.transaction_cost_audit)
        if result.survivorship_audit is not None and not result.survivorship_audit.empty:
            worst_status = "FAIL" if result.survivorship_audit["status"].eq("fail").any() else "REVIEW" if result.survivorship_audit["status"].eq("review").any() else "PASS"
            st.markdown(
                f'<div class="runjin-note">Survivorship bias guard: <strong>{worst_status}</strong>. This checks whether the historical universe includes delisted, bankrupt, acquired, and point-in-time membership data; current-listed-only data can overstate returns.</div>',
                unsafe_allow_html=True,
            )
            named_table("Survivorship bias guard", result.survivorship_audit)
        if result.regime_audit is not None and not result.regime_audit.empty:
            worst_status = "FAIL" if result.regime_audit["status"].eq("fail").any() else "REVIEW" if result.regime_audit["status"].eq("review").any() else "PASS"
            st.markdown(
                f'<div class="runjin-note">Recent regime guard: <strong>{worst_status}</strong>. This compares recent performance with full-period and early-period results, then flags cost-stationarity, survivor-set, and market-structure risks in old history.</div>',
                unsafe_allow_html=True,
            )
            named_table("Recent regime guard", result.regime_audit)
        if result.lookahead_audit is not None and not result.lookahead_audit.empty:
            audit_status = result.lookahead_audit["status"].iloc[0]
            st.markdown(
                f'<div class="runjin-note">Look-ahead audit: <strong>{str(audit_status).upper()}</strong>. The engine reruns the strategy on truncated data and compares the shared position file with the full-data run.</div>',
                unsafe_allow_html=True,
            )
            named_table("Look-ahead bias audit", result.lookahead_audit)
            if result.lookahead_details is not None and not result.lookahead_details.empty:
                named_table("Look-ahead audit details", result.lookahead_details)
        if result.drawdown_tolerance is not None and not result.drawdown_tolerance.empty:
            worst_status = "FAIL" if result.drawdown_tolerance["status"].eq("fail").any() else "REVIEW" if result.drawdown_tolerance["status"].eq("review").any() else "PASS"
            st.markdown(
                f'<div class="runjin-note">Drawdown tolerance guard: <strong>{worst_status}</strong>. This compares maximum drawdown depth and longest time underwater with your stated tolerance.</div>',
                unsafe_allow_html=True,
            )
            named_table("Drawdown tolerance guard", result.drawdown_tolerance)
        if result.snooping_audit is not None and not result.snooping_audit.empty:
            worst_status = "FAIL" if result.snooping_audit["status"].eq("fail").any() else "REVIEW" if result.snooping_audit["status"].eq("review").any() else "PASS"
            st.markdown(
                f'<div class="runjin-note">Data-snooping audit: <strong>{worst_status}</strong>. This checks parameter count, sample-per-parameter, assumed trial count, qualitative choices, a conservative Sharpe haircut proxy, and Bailey-style sample-size thresholds.</div>',
                unsafe_allow_html=True,
            )
            named_table("Data-snooping bias audit", result.snooping_audit)
        if result.out_of_sample_audit is not None and not result.out_of_sample_audit.empty:
            worst_status = "FAIL" if result.out_of_sample_audit["status"].eq("fail").any() else "REVIEW" if result.out_of_sample_audit["status"].eq("review").any() else "PASS"
            st.markdown(
                f'<div class="runjin-note">Out-of-sample audit: <strong>{worst_status}</strong>. The same final parameters are tested on a reserved recent segment to see whether performance survives outside the training period.</div>',
                unsafe_allow_html=True,
            )
            named_table("Out-of-sample audit", result.out_of_sample_audit)

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
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Return", fmt_number(stats.get("Return [%]"), 1, "%"))
        col2.metric("Max drawdown", fmt_number(stats.get("Max. Drawdown [%]"), 1, "%"))
        col3.metric("Sharpe", fmt_number(stats.get("Sharpe Ratio"), 2))
        col4.metric("Info ratio", fmt_number(stats.get("Information Ratio"), 2))
        col5.metric("Rebalances", fmt_int(stats.get("Rebalances", 0) or 0))
        col6.metric("Total cost", f"${fmt_int(stats.get('Total Cost [$]', 0) or 0)}")

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
    nvda_benchmark = prices.loc[prices["ticker"] == "SPY"].copy() if "SPY" in set(prices["ticker"]) else pd.DataFrame()
    volume_price_note = latest_volume_price_note(add_indicators(prices.loc[prices["ticker"] == "NVDA"].copy()), nvda_benchmark)
    report = build_weekly_report(
        scored,
        {
            "US stock daily trend": {"metrics": stock_metrics, "status": stock_status, "reason": stock_reason},
            "Crypto hourly mean reversion": {"metrics": crypto_metrics, "status": crypto_status, "reason": crypto_reason},
        },
        volume_price_note=volume_price_note,
    )
    st.markdown(report)
    if st.button("导出 Markdown", key="weekly_export_md"):
        path = export_markdown_report("weekly_trade_lab_review", report)
        st.success(f"已导出：{path}")


def page_long_research_desk(prices, crypto, financials, market_universe, scored, industry_map, forecasts, finance_research, source_status, data_mode):
    subpage_label = st.radio(
        "研究模块",
        ["投研小助手", "单股同步", "批量分析", "A股研判", "牛股框架", "股票池", "观察清单", "研究雷达", "研究日志"],
        horizontal=True,
        index=6,
        key="long_research_layer",
    )
    subpage = {
        "投研小助手": "Research Assistant",
        "单股同步": "Single Sync",
        "批量分析": "Batch Analysis",
        "A股研判": "A-Share Agent Framework",
        "牛股框架": "10x Stock Profile",
        "股票池": "Market Universe",
        "观察清单": "Long Watchlist",
        "研究雷达": "Finance Radar",
        "研究日志": "Research Journal",
    }[subpage_label]
    if subpage == "Research Assistant":
        page_ollama_research_assistant(scored, financials)
    elif subpage == "Single Sync":
        page_single_stock_sync()
    elif subpage == "Batch Analysis":
        page_batch_analysis(scored, financials)
    elif subpage == "A-Share Agent Framework":
        page_astock_agent_framework(scored)
    elif subpage == "10x Stock Profile":
        page_tenbagger_profile(prices, financials, scored, industry_map, forecasts, data_mode)
    elif subpage == "Market Universe":
        page_market_universe(market_universe)
    elif subpage == "Long Watchlist":
        page_watchlist(scored)
    elif subpage == "Finance Radar":
        page_finance_mcp_radar(finance_research, scored, source_status, prices, crypto)
    elif subpage == "Research Journal":
        page_research_journal(scored)


def page_signal_lab(prices, crypto, forecasts, market_universe, data_mode):
    subpage_label = st.radio(
        "分析模块",
        ["K线启动点", "SEPA趋势", "盘中情景"],
        horizontal=True,
        key="trading_signal_layer",
    )
    subpage = {
        "K线启动点": "K-line & Launch Points",
        "SEPA趋势": "SEPA Trend Template",
        "盘中情景": "Intraday Scenarios",
    }[subpage_label]
    if subpage == "K-line & Launch Points":
        page_kline_lab(prices, forecasts, market_universe, data_mode)
    elif subpage == "SEPA Trend Template":
        page_sepa_lab(prices)
    elif subpage == "Intraday Scenarios":
        page_intraday_signal(prices, crypto, data_mode)


def main():
    inject_design_system()
    st.sidebar.markdown(
        """
        <div style="padding: 10px 4px 18px;">
          <div style="font-size: 22px; font-weight: 850; color: #e8ebe8; line-height: 1;">润金交易系统</div>
          <div style="font-family: SF Mono, Menlo, monospace; color: #2ed17c; font-size: 11px; letter-spacing: .08em; margin-top: 8px;">v0.2</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    data_mode = "live"
    try:
        prices, crypto, financials, market_universe, scored, industry_map, forecasts, finance_research, risk_rules, source_status = load_all_data(data_mode)
    except Exception as exc:
        st.sidebar.error(f"Realtime data failed: {exc}")
        st.sidebar.caption("The app is now configured to use real-time research data only, so it stops instead of falling back to bundled samples.")
        st.stop()
    nav_labels = {
        "Dashboard": "D  仪表盘",
        "Research Desk": "R  长期研究",
        "Signal Lab": "T  技术分析",
        "Capital Rotation": "M  资金轮动",
        "Backtest Lab": "B  回测复盘",
        "Paper Bot": "A  模拟交易",
        "Weekly Review": "LG 周报",
        "System Capabilities": "⚙ 系统能力",
    }
    page_reverse = {label: key for key, label in nav_labels.items()}
    page_label = st.sidebar.radio(
        "导航",
        [
            nav_labels["Dashboard"],
            nav_labels["Research Desk"],
            nav_labels["Signal Lab"],
            nav_labels["Capital Rotation"],
            nav_labels["Backtest Lab"],
            nav_labels["Paper Bot"],
            nav_labels["Weekly Review"],
            nav_labels["System Capabilities"],
        ],
        label_visibility="collapsed",
    )
    page = page_reverse[page_label]
    with st.sidebar.expander("系统状态", expanded=False):
        st.caption(f"MODE / {data_mode}")
        st.caption("BOUNDARY / V0.2 never places real orders.")

    if page == "Dashboard":
        page_dashboard(prices, crypto, market_universe, scored, finance_research, risk_rules, source_status)
    elif page == "Research Desk":
        page_long_research_desk(prices, crypto, financials, market_universe, scored, industry_map, forecasts, finance_research, source_status, data_mode)
    elif page == "Signal Lab":
        page_signal_lab(prices, crypto, forecasts, market_universe, data_mode)
    elif page == "Capital Rotation":
        page_capital_rotation(prices)
    elif page == "Backtest Lab":
        page_backtest_lab(prices, crypto)
    elif page == "Paper Bot":
        page_short_bot(prices, crypto, risk_rules)
    elif page == "Weekly Review":
        page_weekly_report(prices, crypto, scored, risk_rules)
    elif page == "System Capabilities":
        page_system_capabilities(source_status)


if __name__ == "__main__":
    main()
