from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.backtest_lab.costs import describe_execution_assumptions, resolve_commission_rate, slippage_reference
from src.backtest_lab.bias import lookahead_audit_frame, run_lookahead_audit
from src.backtest_lab.metrics import abu_style_metrics, metrics_detail_frame
from src.backtest_lab.position_sizing import resolve_position_size
from src.backtest_lab.snooping import data_snooping_audit_frame
from src.backtest_lab.ump_lite import evaluate_ump_lite


DEFAULT_STRATEGY_SPEC = """name: RunJin SMA trend test
template: sma_crossover
cash: 100000
commission_pct: 0.10
commission_model: us_equity_basic
slippage_model: hl_mean_gap_guard
trade_on_close: false
position_model: fixed_fraction
benchmark: SPY
position_size: 0.95
stop_loss_pct: 6
take_profit_pct: 0
lookahead_check:
  enabled: true
  truncation_bars: 30
snooping_check:
  enabled: true
  max_parameters: 5
  min_bars_per_parameter: 30
  assumed_trials: 1
risk_judge:
  enabled: true
  max_volatility: 0.75
  max_drawdown: 0.12
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

BACKTEST_SYSTEM_MAP = [
    {
        "project": "backtesting.py",
        "best_idea": "Fast single-instrument Strategy/Backtest workflow with detailed stats and trades.",
        "runjin_integration": "Single Asset tab executes vetted YAML templates through backtesting.py.",
        "status": "wired",
    },
    {
        "project": "backtrader",
        "best_idea": "Event-driven mindset, broker simulation, commission/slippage, orders, multiple timeframes.",
        "runjin_integration": "Execution constraints, no leverage, commission fields, and future order/slippage roadmap.",
        "status": "absorbed_design",
    },
    {
        "project": "qstrader",
        "best_idea": "Portfolio/event architecture with clear data, strategy, portfolio, and execution boundaries.",
        "runjin_integration": "Backtest engine adapters are isolated from Streamlit UI and paper bot execution.",
        "status": "absorbed_design",
    },
    {
        "project": "QuantResearch",
        "best_idea": "Research notebooks for portfolio optimization, VaR, factors, mean reversion, pairs, and regimes.",
        "runjin_integration": "Research diagnostics and future experiment logging live under Backtest Lab docs.",
        "status": "research_roadmap",
    },
    {
        "project": "bt",
        "best_idea": "Composable portfolio algos, rebalancing, reusable blocks, and detailed comparison reports.",
        "runjin_integration": "Portfolio tab supports equal weight, momentum top-N, inverse volatility, turnover and cost.",
        "status": "wired_lightweight",
    },
    {
        "project": "Gekko BacktestTool",
        "best_idea": "Crypto-bot style rapid strategy comparison and parameter iteration.",
        "runjin_integration": "Crypto OHLCV can be tested in Single Asset tab; parameter sweeps are next-stage.",
        "status": "absorbed_design",
    },
]

DEFAULT_PORTFOLIO_SPEC = """name: RunJin AI infrastructure basket
template: momentum_top_n
cash: 100000
commission_pct: 0.10
rebalance_days: 20
max_position_pct: 0.25
parameters:
  lookback_days: 60
  top_n: 4
universe:
  - NVDA
  - AVGO
  - AMD
  - TSM
  - PLTR
  - TSLA
"""


EXAMPLE_PORTFOLIO_SPECS = {
    "Momentum top-N": DEFAULT_PORTFOLIO_SPEC,
    "Equal weight basket": """name: RunJin equal-weight watchlist
template: equal_weight_rebalance
cash: 100000
commission_pct: 0.10
rebalance_days: 20
max_position_pct: 0.20
parameters: {}
universe:
  - NVDA
  - AVGO
  - AMD
  - TSM
  - PLTR
""",
    "Inverse volatility": """name: RunJin inverse-volatility basket
template: inverse_volatility
cash: 100000
commission_pct: 0.10
rebalance_days: 20
max_position_pct: 0.25
parameters:
  lookback_days: 40
universe:
  - NVDA
  - AVGO
  - AMD
  - TSM
  - PLTR
  - TSLA
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
    assumptions: pd.DataFrame | None = None
    metrics_detail: pd.DataFrame | None = None
    ump_verdict: pd.DataFrame | None = None
    slippage_detail: pd.DataFrame | None = None
    lookahead_audit: pd.DataFrame | None = None
    lookahead_details: pd.DataFrame | None = None
    snooping_audit: pd.DataFrame | None = None


@dataclass
class PortfolioBacktestResult:
    name: str
    template: str
    stats: dict[str, Any]
    equity_curve: pd.DataFrame
    rebalance_log: pd.DataFrame
    weights: pd.DataFrame
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


def load_portfolio_spec(raw_spec: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError:
        return _parse_simple_portfolio_yaml(raw_spec)

    spec = yaml.safe_load(raw_spec) or {}
    if not isinstance(spec, dict):
        raise ValueError("Portfolio spec must be a YAML object.")
    return spec


def _parse_simple_portfolio_yaml(raw_spec: str) -> dict[str, Any]:
    spec: dict[str, Any] = {}
    current_section = None
    for raw_line in raw_spec.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if current_section == "universe" and stripped.startswith("- "):
            spec.setdefault("universe", []).append(stripped[2:].strip())
            continue
        if line.startswith("  ") and current_section == "parameters":
            key, value = _split_yaml_pair(stripped)
            spec.setdefault("parameters", {})[key] = _coerce_scalar(value)
            continue
        key, value = _split_yaml_pair(stripped)
        if value == "":
            spec[key] = [] if key == "universe" else {}
            current_section = key
        else:
            spec[key] = _coerce_scalar(value)
            current_section = None
    return spec


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
    risk_judge = spec.get("risk_judge", {}) or {}
    if not isinstance(risk_judge, dict):
        raise ValueError("risk_judge must be a YAML object.")

    commission_model = str(spec.get("commission_model", "pct_only")).strip() or "pct_only"
    slippage_model = str(spec.get("slippage_model", "close")).strip() or "close"
    trade_on_close = bool(spec.get("trade_on_close", False))
    if trade_on_close:
        warnings.append("trade_on_close is enabled. This is valid only when the strategy can truly decide at the period close.")
    position_model = str(spec.get("position_model", "fixed_fraction")).strip() or "fixed_fraction"
    benchmark = str(spec.get("benchmark", "")).strip()
    lookahead_check = spec.get("lookahead_check", {"enabled": True, "truncation_bars": 30}) or {}
    if not isinstance(lookahead_check, dict):
        raise ValueError("lookahead_check must be a YAML object.")
    snooping_check = spec.get("snooping_check", {"enabled": True, "max_parameters": 5, "min_bars_per_parameter": 30, "assumed_trials": 1}) or {}
    if not isinstance(snooping_check, dict):
        raise ValueError("snooping_check must be a YAML object.")
    position_parameters = spec.get("position_parameters", {}) or {}
    if not isinstance(position_parameters, dict):
        raise ValueError("position_parameters must be a YAML object.")

    max_window = _max_lookback(template, params)
    if bars < max_window + 20:
        warnings.append(f"Only {bars} bars available; this is thin for a {template} strategy with lookback {max_window}.")

    normalized = {
        "name": str(spec.get("name", template)).strip() or template,
        "template": template,
        "cash": cash,
        "commission": resolve_commission_rate(commission_model, commission_pct / 100),
        "commission_model": commission_model,
        "slippage_model": slippage_model,
        "trade_on_close": trade_on_close,
        "position_model": position_model,
        "position_parameters": position_parameters,
        "benchmark": benchmark,
        "position_size": position_size,
        "stop_loss_pct": stop_loss_pct / 100,
        "take_profit_pct": take_profit_pct / 100,
        "risk_judge": risk_judge,
        "lookahead_check": lookahead_check,
        "snooping_check": snooping_check,
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
    normalized["position_size"] = resolve_position_size(raw_data, normalized, warnings)
    slippage_detail = slippage_reference(raw_data, normalized["slippage_model"]).tail(20)
    lookahead_config = normalized["lookahead_check"]
    lookahead_result = None
    lookahead_audit = None
    lookahead_details = None
    if bool(lookahead_config.get("enabled", True)):
        lookahead_result = run_lookahead_audit(
            data,
            normalized["template"],
            params,
            normalized["trade_on_close"],
            int(lookahead_config.get("truncation_bars", 30)),
        )
        lookahead_audit = lookahead_audit_frame(lookahead_result)
        lookahead_details = lookahead_result.details
        if lookahead_result.status == "fail":
            warnings.append("Look-ahead audit failed: positions changed when future rows were truncated.")

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
        trade_on_close=normalized["trade_on_close"],
        exclusive_orders=True,
        hedging=False,
    )
    stats = bt.run()
    equity_curve = stats["_equity_curve"].reset_index()
    trades = stats["_trades"].copy()
    summary = _stats_to_dict(stats)
    snooping_audit = data_snooping_audit_frame(normalized, summary, len(data)) if bool(normalized["snooping_check"].get("enabled", True)) else pd.DataFrame()
    if not snooping_audit.empty and snooping_audit["status"].isin(["review", "fail"]).any():
        warnings.append("Data-snooping audit found review/fail items. Treat optimized performance with a haircut.")
    metrics_detail = abu_style_metrics(equity_curve, trades, data.reset_index())
    summary.update({f"ABU {key}": value for key, value in metrics_detail.items() if value is not None})
    ump_verdict = evaluate_ump_lite(raw_data, equity_curve, trades, normalized["risk_judge"])
    return BacktestResult(
        name=normalized["name"],
        template=normalized["template"],
        stats=summary,
        equity_curve=equity_curve,
        trades=trades,
        data=data.reset_index(),
        warnings=warnings,
        assumptions=describe_execution_assumptions(
            normalized["commission_model"],
            normalized["slippage_model"],
            normalized["trade_on_close"],
            normalized["position_model"],
            normalized["benchmark"],
        ),
        metrics_detail=metrics_detail_frame(metrics_detail),
        ump_verdict=pd.DataFrame({"verdict": [ump_verdict.verdict] * len(ump_verdict.reasons), "reason": ump_verdict.reasons}),
        slippage_detail=slippage_detail,
        lookahead_audit=lookahead_audit,
        lookahead_details=lookahead_details,
        snooping_audit=snooping_audit,
    )


def run_portfolio_backtest(raw_prices: pd.DataFrame, raw_spec: str) -> PortfolioBacktestResult:
    spec = load_portfolio_spec(raw_spec)
    normalized, warnings = validate_portfolio_spec(spec)
    prices = _prepare_price_matrix(raw_prices, normalized["universe"])
    if prices.empty or prices.shape[1] < 2:
        raise ValueError("Portfolio backtest needs at least two symbols with overlapping price history.")

    returns = prices.pct_change().fillna(0)
    rebalance_days = normalized["rebalance_days"]
    commission = normalized["commission"]
    cash = normalized["cash"]
    max_position = normalized["max_position_pct"]
    params = normalized["parameters"]

    equity = cash
    current_weights = pd.Series(0.0, index=prices.columns)
    rows = []
    weights_rows = []
    rebalance_rows = []

    for i, date in enumerate(prices.index):
        daily_return = float((current_weights * returns.loc[date]).sum())
        equity *= 1 + daily_return
        turnover = 0.0
        cost = 0.0

        should_rebalance = i == 0 or i % rebalance_days == 0
        if should_rebalance:
            target_weights = _portfolio_target_weights(
                normalized["template"],
                prices.iloc[: i + 1],
                params,
                max_position,
            )
            turnover = float((target_weights - current_weights).abs().sum())
            cost = equity * turnover * commission
            equity -= cost
            rebalance_rows.append(
                {
                    "date": date,
                    "turnover": turnover,
                    "cost": cost,
                    "equity_after_cost": equity,
                    "active_positions": int((target_weights > 0).sum()),
                    "top_weight": float(target_weights.max()) if not target_weights.empty else 0.0,
                }
            )
            current_weights = target_weights

        rows.append({"date": date, "Equity": equity, "Return": daily_return, "Turnover": turnover, "Cost": cost})
        weights_row = {"date": date}
        weights_row.update(current_weights.to_dict())
        weights_rows.append(weights_row)

    equity_curve = pd.DataFrame(rows)
    equity_curve["DrawdownPct"] = equity_curve["Equity"] / equity_curve["Equity"].cummax() - 1
    stats = _portfolio_stats(equity_curve, prices, cash)
    return PortfolioBacktestResult(
        name=normalized["name"],
        template=normalized["template"],
        stats=stats,
        equity_curve=equity_curve,
        rebalance_log=pd.DataFrame(rebalance_rows),
        weights=pd.DataFrame(weights_rows),
        warnings=warnings,
    )


def validate_portfolio_spec(spec: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    template = str(spec.get("template", "equal_weight_rebalance")).strip()
    if template not in {"equal_weight_rebalance", "momentum_top_n", "inverse_volatility"}:
        raise ValueError(f"Unsupported portfolio template: {template}")

    universe = spec.get("universe", [])
    if not isinstance(universe, list) or len(universe) < 2:
        raise ValueError("universe must include at least two tickers.")
    universe = [str(item).strip().upper() for item in universe if str(item).strip()]

    cash = float(spec.get("cash", 100000))
    commission_pct = float(spec.get("commission_pct", 0.1))
    rebalance_days = int(spec.get("rebalance_days", 20))
    max_position_pct = float(spec.get("max_position_pct", 0.25))
    params = spec.get("parameters", {}) or {}
    if not isinstance(params, dict):
        raise ValueError("parameters must be a YAML object.")
    if cash <= 0:
        raise ValueError("cash must be positive.")
    if commission_pct < 0 or commission_pct > 2:
        raise ValueError("commission_pct must be between 0 and 2.")
    if rebalance_days < 1:
        raise ValueError("rebalance_days must be at least 1.")
    if max_position_pct <= 0 or max_position_pct > 1:
        raise ValueError("max_position_pct must be > 0 and <= 1.")
    if max_position_pct * len(universe) < 0.99:
        warnings.append("max_position_pct is tight enough that the portfolio may hold residual cash.")

    return {
        "name": str(spec.get("name", template)).strip() or template,
        "template": template,
        "universe": universe,
        "cash": cash,
        "commission": commission_pct / 100,
        "rebalance_days": rebalance_days,
        "max_position_pct": max_position_pct,
        "parameters": params,
    }, warnings


def _prepare_price_matrix(raw_prices: pd.DataFrame, universe: list[str]) -> pd.DataFrame:
    required = {"date", "ticker", "close"}
    missing = required - set(raw_prices.columns)
    if missing:
        raise ValueError(f"Missing price columns: {', '.join(sorted(missing))}")
    data = raw_prices.loc[raw_prices["ticker"].isin(universe), ["date", "ticker", "close"]].copy()
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["close"] = pd.to_numeric(data["close"], errors="coerce")
    matrix = data.dropna().pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()
    return matrix.ffill().dropna(axis=1, how="all").dropna()


def _portfolio_target_weights(template: str, price_history: pd.DataFrame, params: dict[str, Any], max_position: float) -> pd.Series:
    columns = price_history.columns
    weights = pd.Series(0.0, index=columns)
    if template == "equal_weight_rebalance":
        selected = list(columns)
    elif template == "momentum_top_n":
        lookback = int(params.get("lookback_days", 60))
        top_n = int(params.get("top_n", min(5, len(columns))))
        if len(price_history) <= lookback:
            selected = list(columns[:top_n])
        else:
            momentum = price_history.iloc[-1] / price_history.iloc[-lookback] - 1
            selected = momentum.sort_values(ascending=False).head(top_n).index.tolist()
    elif template == "inverse_volatility":
        lookback = int(params.get("lookback_days", 40))
        vol = price_history.pct_change().tail(lookback).std().replace(0, np.nan)
        inv_vol = (1 / vol).replace([np.inf, -np.inf], np.nan).dropna()
        if inv_vol.empty:
            selected = list(columns)
            raw_weights = pd.Series(1 / len(selected), index=selected)
        else:
            raw_weights = inv_vol / inv_vol.sum()
            selected = raw_weights.index.tolist()
        capped = raw_weights.clip(upper=max_position)
        weights.loc[capped.index] = capped
        if weights.sum() > 1:
            weights = weights / weights.sum()
        return weights
    else:
        selected = list(columns)

    if selected:
        equal = min(max_position, 1 / len(selected))
        weights.loc[selected] = equal
    return weights


def _portfolio_stats(equity_curve: pd.DataFrame, prices: pd.DataFrame, starting_cash: float) -> dict[str, Any]:
    daily_returns = equity_curve["Equity"].pct_change().dropna()
    periods = max(len(equity_curve), 1)
    total_return = equity_curve["Equity"].iloc[-1] / starting_cash - 1
    annualized_return = (1 + total_return) ** (252 / periods) - 1 if periods > 1 else 0
    annualized_vol = daily_returns.std() * np.sqrt(252) if not daily_returns.empty else 0
    sharpe = annualized_return / annualized_vol if annualized_vol and annualized_vol > 0 else np.nan
    max_drawdown = equity_curve["DrawdownPct"].min()
    benchmark = prices.mean(axis=1)
    benchmark_return = benchmark.iloc[-1] / benchmark.iloc[0] - 1 if len(benchmark) > 1 else 0
    return {
        "Start": equity_curve["date"].iloc[0],
        "End": equity_curve["date"].iloc[-1],
        "Equity Final [$]": float(equity_curve["Equity"].iloc[-1]),
        "Return [%]": float(total_return * 100),
        "Benchmark Equal Basket Return [%]": float(benchmark_return * 100),
        "Return (Ann.) [%]": float(annualized_return * 100),
        "Volatility (Ann.) [%]": float(annualized_vol * 100),
        "Sharpe Ratio": float(sharpe) if pd.notna(sharpe) else None,
        "Max. Drawdown [%]": float(max_drawdown * 100),
        "Rebalances": int((equity_curve["Turnover"] > 0).sum()),
        "Total Cost [$]": float(equity_curve["Cost"].sum()),
        "Avg Turnover": float(equity_curve.loc[equity_curve["Turnover"] > 0, "Turnover"].mean() or 0),
    }


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
