import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_sources.loaders import (
    load_crypto_prices,
    load_financials,
    load_market_universe,
    load_prices,
    load_risk_rules,
    load_watchlist_config,
    load_watchlist_notes,
)
from src.data_sources.finance_mcp import load_finance_mcp_capabilities, load_finance_mcp_research
from src.backtest_lab.engine import (
    BacktestEngineUnavailable,
    DEFAULT_STRATEGY_SPEC,
    load_strategy_spec,
    prepare_ohlcv,
    run_strategy_backtest,
    validate_strategy_spec,
)
from src.kline.indicators import add_indicators
from src.long_term.scoring import SCORE_COLUMNS, build_score_table
from src.quant_bot.paper_trader import run_crypto_paper, run_us_stock_paper
from src.quant_bot.risk import evaluate_risk
from src.reports.weekly_report import build_weekly_report


def assert_true(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    watchlist = load_watchlist_config()["watchlist"]
    tickers = [item["ticker"] for item in watchlist]
    prices = load_prices(data_mode="sample")
    crypto = load_crypto_prices(data_mode="sample")
    financials = load_financials(data_mode="sample")
    market_universe = load_market_universe(data_mode="sample")
    notes = load_watchlist_notes()
    risk_rules = load_risk_rules()
    scored = build_score_table(notes)

    required_universe_cols = ["ticker", "yahoo_ticker", "market", "exchange", "currency", "market_cap_usd"]
    for col in required_universe_cols:
        assert_true(col in market_universe.columns, f"Market universe missing {col}")
    assert_true({"US", "A_SHARE_SH", "A_SHARE_SZ", "HK", "SG"}.issubset(set(market_universe["market"])), "Market universe missing required markets")
    assert_true((market_universe["market_cap_usd"] >= 300_000_000).all(), "Market universe includes micro caps below USD 300M")

    for ticker in tickers:
        assert_true(not prices.loc[prices["ticker"] == ticker].empty, f"Missing price rows for {ticker}")
        assert_true(not notes.loc[notes["ticker"] == ticker].empty, f"Missing notes for {ticker}")
        assert_true(not financials.loc[financials["ticker"] == ticker].empty, f"Missing financials for {ticker}")

    nvda_indicators = add_indicators(prices.loc[prices["ticker"] == "NVDA"].copy())
    required_indicator_cols = ["ma20", "ma60", "rsi14", "macd", "bb_lower", "bb_upper", "volatility_20", "kdj_k", "kdj_d", "kdj_j"]
    for col in required_indicator_cols:
        assert_true(nvda_indicators[col].notna().sum() > 0, f"Indicator {col} is empty")

    score_check = notes[SCORE_COLUMNS].sum(axis=1).reset_index(drop=True)
    assert_true((scored.sort_values("ticker")[SCORE_COLUMNS].sum(axis=1).reset_index(drop=True) >= 0).all(), "Invalid scores")
    assert_true(score_check.max() <= 35, "Score exceeds 35")
    assert_true(set(scored["bucket"]).issubset({"Deep research candidate", "Observation pool", "Low priority"}), "Invalid bucket")

    stock_bt, stock_metrics, stock_trades, stock_status, _ = run_us_stock_paper(
        prices.loc[prices["ticker"] == "NVDA"].copy(), risk_rules
    )
    crypto_bt, crypto_metrics, crypto_trades, crypto_status, _ = run_crypto_paper(
        crypto.loc[crypto["symbol"] == "BTC-USD"].copy(), risk_rules
    )
    assert_true(stock_metrics["trade_count"] > 0, "US stock bot produced no trades")
    assert_true(crypto_metrics["trade_count"] > 0, "Crypto bot produced no trades")
    assert_true(not stock_trades.empty, "US stock trade log empty")
    assert_true(not crypto_trades.empty, "Crypto trade log empty")
    assert_true(stock_status in {"CONTINUE", "STOP", "PAUSE"}, "Invalid stock bot status")
    assert_true(crypto_status in {"CONTINUE", "STOP", "PAUSE"}, "Invalid crypto bot status")

    finance_research, finance_status = load_finance_mcp_research(data_mode="sample")
    finance_capabilities = load_finance_mcp_capabilities()
    assert_true(not finance_research.empty, "FinanceMCP research radar is empty")
    assert_true(not finance_capabilities.empty, "FinanceMCP capability map is empty")
    assert_true(finance_status["mode"] in {"sample", "external_csv", "mcp_ready_sample_fallback"}, "Invalid FinanceMCP status")

    strategy_spec = load_strategy_spec(DEFAULT_STRATEGY_SPEC)
    prepared = prepare_ohlcv(prices.loc[prices["ticker"] == "NVDA"].copy(), "date")
    normalized_spec, warnings = validate_strategy_spec(strategy_spec, len(prepared))
    assert_true(not prepared.empty, "Backtest OHLCV preparation returned no rows")
    assert_true(normalized_spec["template"] == "sma_crossover", "Backtest strategy spec did not normalize")
    try:
        backtest_result = run_strategy_backtest(prices.loc[prices["ticker"] == "NVDA"].copy(), "date", DEFAULT_STRATEGY_SPEC)
        assert_true("Return [%]" in backtest_result.stats, "Backtest stats missing return")
    except BacktestEngineUnavailable:
        pass

    stressed = stock_bt.copy()
    stressed.loc[stressed.index[-1], "equity"] = stressed["equity"].iloc[-2] * 0.90
    stressed["drawdown"] = stressed["equity"] / stressed["equity"].cummax() - 1
    status, reason = evaluate_risk(stressed, risk_rules)
    assert_true(status == "STOP", f"Risk stress did not stop bot: {reason}")

    report = build_weekly_report(
        scored,
        {
            "US stock daily trend": {"metrics": stock_metrics, "status": stock_status},
            "Crypto hourly mean reversion": {"metrics": crypto_metrics, "status": crypto_status},
        },
    )
    for token in ["Long-term observation desk", "Paper bot review", "continue"]:
        assert_true(token in report, f"Report missing {token}")

    print("All Trade Lab demo checks passed.")


if __name__ == "__main__":
    main()
