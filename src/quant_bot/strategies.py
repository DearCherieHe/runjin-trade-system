from src.kline.indicators import add_indicators


def us_stock_trend_signal(price_df, fast_ma=20, slow_ma=60, max_volatility=0.75):
    data = add_indicators(price_df)
    data["signal"] = 0
    long_mask = (data[f"ma{fast_ma}"] > data[f"ma{slow_ma}"]) & (data["volatility_20"] <= max_volatility)
    data.loc[long_mask, "signal"] = 1
    data["signal_reason"] = "No position"
    data.loc[long_mask, "signal_reason"] = "Fast MA above slow MA with acceptable volatility"
    return data


def crypto_mean_reversion_signal(price_df, lower_rsi=35, upper_rsi=65):
    data = add_indicators(price_df.rename(columns={"datetime": "date"}))
    data["signal"] = 0
    buy_mask = (data["close"] < data["bb_lower"]) | (data["rsi14"] < lower_rsi)
    sell_mask = (data["close"] > data["bb_mid"]) | (data["rsi14"] > upper_rsi)
    data.loc[buy_mask, "signal"] = 1
    data.loc[sell_mask, "signal"] = 0
    data["signal_reason"] = "No position"
    data.loc[buy_mask, "signal_reason"] = "Oversold versus Bollinger/RSI"
    data.loc[sell_mask, "signal_reason"] = "Mean reversion exit zone"
    return data.rename(columns={"date": "datetime"})
