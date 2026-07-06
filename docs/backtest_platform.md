# Backtest Platform

RunJin's Backtest Lab is designed as a research platform, not an execution engine.

## V0.1 Scope

- Engine: `backtesting.py`
- Input: constrained YAML strategy specs
- Assets: US stock OHLCV and crypto OHLCV already loaded by the app
- Outputs: equity curve, drawdown overlay, statistics, trade log, portfolio rebalance log, and latest weights
- Safety: no leverage, no real orders, no exchange keys, no arbitrary Python execution

## Borrowed Strengths

RunJin does not install every mature backtesting framework into one Streamlit app. It borrows their strongest ideas and keeps a narrow implementation surface:

| System | Strength integrated into RunJin |
|---|---|
| `backtesting.py` | Single-instrument Strategy/Backtest execution, statistics, equity curve, and trade log. |
| `backtrader` | Event-driven discipline, broker-simulation boundaries, commission assumptions, and future slippage/order-model roadmap. |
| `qstrader` | Clean separation between data, strategy, portfolio, execution, and reporting. |
| `QuantResearch` | Research workflow mindset: factors, risk, VaR, regimes, notebooks, and experiment history. |
| `bt` | Portfolio allocation, rebalancing, reusable strategy blocks, comparison-oriented reporting. |
| `Gekko BacktestTool` | Crypto-bot oriented rapid iteration and parameter testing, without connecting real exchange keys. |

## Portfolio Templates

The Portfolio Lab is a lightweight portfolio-rebalance simulator inspired by `bt` and qstrader-style portfolio boundaries:

- `equal_weight_rebalance`
- `momentum_top_n`
- `inverse_volatility`

Example:

```yaml
name: RunJin AI infrastructure basket
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
```

## Why YAML Specs Instead Of Raw Python

Letting users paste arbitrary Python into Streamlit would make the app unsafe on a hosted server. V0.1 supports editable strategy specs that map to vetted templates:

- `sma_crossover`
- `rsi_mean_reversion`
- `bollinger_reversion`
- `macd_trend`

This gives enough freedom to test ideas while keeping the engine reproducible and auditable.

## Strategy Spec Shape

```yaml
name: RunJin SMA trend test
template: sma_crossover
cash: 100000
commission_pct: 0.10
position_size: 0.95
stop_loss_pct: 6
take_profit_pct: 0
parameters:
  fast_window: 20
  slow_window: 60
```

## Upgrade Path

1. Add a job table for long-running parameter sweeps.
2. Add vectorbt for large grid searches and portfolio-level signal matrices.
3. Add walk-forward validation and train/test splits.
4. Add slippage models, liquidity constraints, and benchmark comparison.
5. Add result persistence so every backtest can be reproduced from a saved spec and data snapshot.
