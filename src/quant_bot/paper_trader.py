from src.quant_bot.backtest import build_trade_log, run_long_only_backtest, summarize_backtest
from src.quant_bot.risk import evaluate_risk
from src.quant_bot.strategies import crypto_mean_reversion_signal, us_stock_trend_signal


def run_us_stock_paper(price_df, risk_rules):
    config = risk_rules["strategies"]["us_stock_daily"]
    signal_df = us_stock_trend_signal(
        price_df,
        fast_ma=config["fast_ma"],
        slow_ma=config["slow_ma"],
        max_volatility=config["max_annualized_volatility"],
    )
    bt = run_long_only_backtest(
        signal_df,
        time_col="date",
        starting_cash=risk_rules["capital"]["starting_cash"],
        position_pct=risk_rules["capital"]["max_position_pct"],
    )
    status, reason = evaluate_risk(bt, risk_rules)
    return bt, summarize_backtest(bt), build_trade_log(bt, "date"), status, reason


def run_crypto_paper(price_df, risk_rules):
    config = risk_rules["strategies"]["crypto_hourly"]
    signal_df = crypto_mean_reversion_signal(
        price_df,
        lower_rsi=config["lower_rsi"],
        upper_rsi=config["upper_rsi"],
    )
    bt = run_long_only_backtest(
        signal_df,
        time_col="datetime",
        starting_cash=risk_rules["capital"]["starting_cash"],
        position_pct=risk_rules["capital"]["max_position_pct"],
    )
    status, reason = evaluate_risk(bt, risk_rules)
    return bt, summarize_backtest(bt), build_trade_log(bt, "datetime"), status, reason
