import numpy as np
import pandas as pd


def run_long_only_backtest(signal_df, time_col="date", starting_cash=100000, position_pct=0.1):
    data = signal_df.copy().sort_values(time_col)
    data["asset_return"] = data["close"].pct_change().fillna(0)
    data["position"] = data["signal"].shift(1).fillna(0) * position_pct
    data["strategy_return"] = data["position"] * data["asset_return"]
    data["equity"] = starting_cash * (1 + data["strategy_return"]).cumprod()
    data["drawdown"] = data["equity"] / data["equity"].cummax() - 1
    data["trade_flag"] = data["signal"].diff().fillna(data["signal"]).abs()
    return data


def summarize_backtest(backtest_df):
    if backtest_df.empty:
        return {
            "total_return": 0,
            "max_drawdown": 0,
            "win_rate": 0,
            "profit_factor": 0,
            "trade_count": 0,
        }
    returns = backtest_df["strategy_return"]
    positive = returns[returns > 0].sum()
    negative = returns[returns < 0].sum()
    return {
        "total_return": backtest_df["equity"].iloc[-1] / backtest_df["equity"].iloc[0] - 1,
        "max_drawdown": backtest_df["drawdown"].min(),
        "win_rate": float((returns > 0).mean()),
        "profit_factor": float(positive / abs(negative)) if negative != 0 else np.inf,
        "trade_count": int(backtest_df["trade_flag"].sum()),
    }


def build_trade_log(backtest_df, time_col="date"):
    data = backtest_df.loc[backtest_df["trade_flag"] > 0, [time_col, "close", "signal", "signal_reason"]].copy()
    data["action"] = data["signal"].map({1: "Enter long", 0: "Exit / flat"}).fillna("Hold")
    return data.tail(20).reset_index(drop=True)
