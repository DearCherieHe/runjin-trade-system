from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_STRATEGY_SPEC = """name: RunJin SMA trend test
template: sma_crossover
cash: 100000
commission_pct: 0.10
position_size: 0.95
stop_loss_pct: 6
take_profit_pct: 0
parameters:
  fast_window: 20
  slow_window: 60
"""


EXAMPLE_STRATEGY_SPECS = {
    "SMA crossover trend": DEFAULT_STRATEGY_SPEC,
    "RSI mean reversion": """name: RunJin RSI mean reversion
template: rsi_mean_reversion
cash: 100000
commission_pct: 0.10
position_size: 0.50
stop_loss_pct: 4
take_profit_pct: 8
parameters:
  rsi_window: 14
  lower_rsi: 35
  upper_rsi: 65
""",
    "Bollinger mean reversion": """name: RunJin Bollinger reversion
template: bollinger_reversion
cash: 100000
commission_pct: 0.10
position_size: 0.50
stop_loss_pct: 5
take_profit_pct: 9
parameters:
  window: 20
  std: 2
""",
    "MACD trend": """name: RunJin MACD trend
template: macd_trend
cash: 100000
commission_pct: 0.10
position_size: 0.75
stop_loss_pct: 6
take_profit_pct: 0
parameters:
  fast: 12
  slow: 26
  signal: 9
""",
}


class BacktestEngineUnavailable(RuntimeError):
    pass


@dataclass
class BacktestResult:
    name: str
    template: str
    stats: dict[str, Any]
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    data: pd.DataFrame
    warnings: list[str]


def load_strategy_spec(raw_spec: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        return _parse_simple_strategy_yaml(raw_spec)

    spec = yaml.safe_load(raw_spec) or {}
    if not isinstance(spec, dict):
        raise ValueError("Strategy spec must be a YAML object.")
    return spec


def _parse_simple_strategy_yaml(raw_spec: str) -> dict[str, Any]:
    spec: dict[str, Any] = {}
    current_section = None
    for raw_line in raw_spec.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("  ") and current_section:
            key, value = _split_yaml_pair(line.strip())
            spec.setdefault(current_section, {})[key] = _coerce_scalar(value)
            continue
        key, value = _split_yaml_pair(line.strip())
        if value == "":
            spec[key] = {}
            current_section = key
        else:
            spec[key] = _coerce_scalar(value)
            current_section = None
    return spec


def _split_yaml_pair(line: str) -> tuple[str, str]:
    if ":" not in line:
        raise ValueError(f"Invalid strategy spec line: {line}")
    key, value = line.split(":", 1)
    return key.strip(), value.strip().strip('"').strip("'")


def _coerce_scalar(value: str):
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value


def prepare_ohlcv(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    required = [time_col, "open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing OHLCV columns: {', '.join(missing)}")

    data = df[required].copy()
    data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
    data = data.dropna(subset=[time_col, "open", "high", "low", "close"]).sort_values(time_col)
    data = data.rename(
        columns={
            time_col: "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    data = data.set_index("Date")
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data = data[~data.index.duplicated(keep="last")]
    return data


def validate_strategy_spec(spec: dict[str, Any], bars: int) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    template = str(spec.get("template", "sma_crossover")).strip()
    if template not in {"sma_crossover", "rsi_mean_reversion", "bollinger_reversion", "macd_trend"}:
        raise ValueError(f"Unsupported template: {template}")

    cash = float(spec.get("cash", 100000))
    if cash <= 0:
        raise ValueError("cash must be positive.")

    commission_pct = float(spec.get("commission_pct", 0.1))
    if commission_pct < 0 or commission_pct > 2:
        raise ValueError("commission_pct must be between 0 and 2.")

    position_size = float(spec.get("position_size", 0.95))
    if position_size <= 0 or position_size > 1:
        raise ValueError("position_size must be > 0 and <= 1. V0.1 does not allow leverage.")

    stop_loss_pct = float(spec.get("stop_loss_pct", 0))
    take_profit_pct = float(spec.get("take_profit_pct", 0))
    if stop_loss_pct < 0 or take_profit_pct < 0:
        raise ValueError("stop_loss_pct and take_profit_pct cannot be negative.")
    if stop_loss_pct == 0:
        warnings.append("No per-trade stop loss is configured. Use this only for research, not live trading.")

    params = spec.get("parameters", {}) or {}
    if not isinstance(params, dict):
        raise ValueError("parameters must be a YAML object.")

    max_window = _max_lookback(template, params)
    if bars < max_window + 20:
        warnings.append(f"Only {bars} bars available; this is thin for a {template} strategy with lookback {max_window}.")

    normalized = {
        "name": str(spec.get("name", template)).strip() or template,
        "template": template,
        "cash": cash,
        "commission": commission_pct / 100,
        "position_size": position_size,
        "stop_loss_pct": stop_loss_pct / 100,
        "take_profit_pct": take_profit_pct / 100,
        "parameters": params,
    }
    return normalized, warnings


def run_strategy_backtest(raw_data: pd.DataFrame, time_col: str, raw_spec: str) -> BacktestResult:
    try:
        from backtesting import Backtest, Strategy
        from backtesting.lib import crossover
    except ImportError as exc:
        raise BacktestEngineUnavailable(
            "backtesting.py is not installed. Run `pip install -r requirements.txt` and restart Streamlit."
        ) from exc

    data = prepare_ohlcv(raw_data, time_col)
    if data.empty:
        raise ValueError("No OHLCV rows available for backtest.")

    spec = load_strategy_spec(raw_spec)
    normalized, warnings = validate_strategy_spec(spec, len(data))
    params = normalized["parameters"]

    class RunJinStrategy(Strategy):
        fast_window = int(params.get("fast_window", 20))
        slow_window = int(params.get("slow_window", 60))
        rsi_window = int(params.get("rsi_window", 14))
        lower_rsi = float(params.get("lower_rsi", 35))
        upper_rsi = float(params.get("upper_rsi", 65))
        window = int(params.get("window", 20))
        std = float(params.get("std", 2))
        fast = int(params.get("fast", 12))
        slow = int(params.get("slow", 26))
        signal = int(params.get("signal", 9))
        position_size = normalized["position_size"]
        stop_loss_pct = normalized["stop_loss_pct"]
        take_profit_pct = normalized["take_profit_pct"]

        def init(self):
            close = self.data.Close
            high = self.data.High
            low = self.data.Low
            self.ma_fast = self.I(_sma, close, self.fast_window)
            self.ma_slow = self.I(_sma, close, self.slow_window)
            self.rsi = self.I(_rsi, close, self.rsi_window)
            self.bb_mid, self.bb_upper, self.bb_lower = self.I(_bollinger, close, self.window, self.std)
            self.macd, self.macd_signal = self.I(_macd, close, self.fast, self.slow, self.signal)
            self.kdj_k, self.kdj_d, self.kdj_j = self.I(_kdj, high, low, close)

        def next(self):
            template = normalized["template"]
            if template == "sma_crossover":
                enter = crossover(self.ma_fast, self.ma_slow)
                exit_signal = crossover(self.ma_slow, self.ma_fast)
            elif template == "rsi_mean_reversion":
                enter = self.rsi[-1] < self.lower_rsi
                exit_signal = self.rsi[-1] > self.upper_rsi
            elif template == "bollinger_reversion":
                enter = self.data.Close[-1] < self.bb_lower[-1]
                exit_signal = self.data.Close[-1] > self.bb_mid[-1]
            elif template == "macd_trend":
                enter = crossover(self.macd, self.macd_signal)
                exit_signal = crossover(self.macd_signal, self.macd)
            else:
                enter = False
                exit_signal = False

            if not self.position and enter:
                price = self.data.Close[-1]
                sl = price * (1 - self.stop_loss_pct) if self.stop_loss_pct > 0 else None
                tp = price * (1 + self.take_profit_pct) if self.take_profit_pct > 0 else None
                self.buy(size=self.position_size, sl=sl, tp=tp)
            elif self.position and exit_signal:
                self.position.close()

    bt = Backtest(
        data,
        RunJinStrategy,
        cash=normalized["cash"],
        commission=normalized["commission"],
        trade_on_close=True,
        exclusive_orders=True,
        hedging=False,
    )
    stats = bt.run()
    equity_curve = stats["_equity_curve"].reset_index()
    trades = stats["_trades"].copy()
    summary = _stats_to_dict(stats)
    return BacktestResult(
        name=normalized["name"],
        template=normalized["template"],
        stats=summary,
        equity_curve=equity_curve,
        trades=trades,
        data=data.reset_index(),
        warnings=warnings,
    )


def _stats_to_dict(stats) -> dict[str, Any]:
    keys = [
        "Start",
        "End",
        "Duration",
        "Exposure Time [%]",
        "Equity Final [$]",
        "Equity Peak [$]",
        "Return [%]",
        "Buy & Hold Return [%]",
        "Return (Ann.) [%]",
        "Volatility (Ann.) [%]",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Calmar Ratio",
        "Max. Drawdown [%]",
        "Avg. Drawdown [%]",
        "# Trades",
        "Win Rate [%]",
        "Best Trade [%]",
        "Worst Trade [%]",
        "Avg. Trade [%]",
        "Profit Factor",
        "SQN",
    ]
    result = {}
    for key in keys:
        value = stats.get(key, None)
        if isinstance(value, (np.floating, np.integer)):
            value = float(value)
        result[key] = value
    return result


def _max_lookback(template: str, params: dict[str, Any]) -> int:
    if template == "sma_crossover":
        return max(int(params.get("fast_window", 20)), int(params.get("slow_window", 60)))
    if template == "rsi_mean_reversion":
        return int(params.get("rsi_window", 14))
    if template == "bollinger_reversion":
        return int(params.get("window", 20))
    if template == "macd_trend":
        return max(int(params.get("fast", 12)), int(params.get("slow", 26)), int(params.get("signal", 9)))
    return 1


def _sma(values, window):
    return pd.Series(values).rolling(int(window)).mean().to_numpy()


def _rsi(values, window):
    close = pd.Series(values)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(int(window)).mean()
    loss = -delta.clip(upper=0).rolling(int(window)).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).to_numpy()


def _bollinger(values, window, std_mult):
    close = pd.Series(values)
    mid = close.rolling(int(window)).mean()
    std = close.rolling(int(window)).std()
    upper = mid + float(std_mult) * std
    lower = mid - float(std_mult) * std
    return mid.to_numpy(), upper.to_numpy(), lower.to_numpy()


def _macd(values, fast, slow, signal):
    close = pd.Series(values)
    macd = close.ewm(span=int(fast), adjust=False).mean() - close.ewm(span=int(slow), adjust=False).mean()
    macd_signal = macd.ewm(span=int(signal), adjust=False).mean()
    return macd.to_numpy(), macd_signal.to_numpy()


def _kdj(high, low, close, window=9):
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    lowest_low = low.rolling(int(window)).min()
    highest_high = high.rolling(int(window)).max()
    rsv = (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    j = 3 * k - 2 * d
    return k.to_numpy(), d.to_numpy(), j.to_numpy()
