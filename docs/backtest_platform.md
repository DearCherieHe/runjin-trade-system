# Backtest Platform

RunJin's Backtest Lab is designed as a research platform, not an execution engine.

## V0.1 Scope

- Engine: `backtesting.py`
- Input: constrained YAML strategy specs
- Assets: US stock OHLCV and crypto OHLCV already loaded by the app
- Outputs: equity curve, drawdown overlay, statistics, and trade log
- Safety: no leverage, no real orders, no exchange keys, no arbitrary Python execution

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
