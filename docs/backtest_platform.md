# Backtest Platform

RunJin's Backtest Lab is designed as a research platform, not an execution engine.

## V0.1 Scope

- Engine: `backtesting.py`
- Input: constrained YAML strategy specs
- Assets: US stock OHLCV and crypto OHLCV already loaded by the app
- Outputs: equity curve, drawdown overlay, statistics, trade log, portfolio rebalance log, and latest weights
- Safety: no leverage, no real orders, no exchange keys, no arbitrary Python execution
- Bias control: default next-bar execution, truncated-data look-ahead audit, and data-snooping audit

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
| `ABU / abupy` | Factor, position, slippage, commission, metrics, and裁判 boundaries, reimplemented without copying GPL source. |

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
commission_model: us_equity_basic
slippage_model: hl_mean_gap_guard
trade_on_close: false
position_model: atr_risk
benchmark: SPY
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
```

## ABU-style Research Layer

The ABU-inspired layer adds execution assumptions and risk review without turning the app into a live trading engine:

- Cost models: `pct_only`, `us_equity_basic`, `a_share_basic`, `hk_equity_basic`, `crypto_basic`.
- Slippage models: `close`, `hl_mean`, `hl_mean_gap_guard`.
- Execution timing: `trade_on_close: false` is the default, so signals based on the current bar execute on the next bar. Set it to `true` only when the strategy can genuinely decide at the period close.
- Position models: `fixed_fraction`, `atr_risk`, `kelly_lite`.
- UMP-lite verdicts: `allow`, `review`, `block`.

K-line Lab also includes research-only gap, ATR, rolling-correlation, and similar-path diagnostics.

## Look-Ahead Bias Guard

RunJin now runs a truncated-data audit for single-asset strategy backtests when `lookahead_check.enabled` is true:

1. Generate a position file from the full historical data.
2. Remove the latest `truncation_bars` rows from the same history.
3. Generate a second position file from the truncated history.
4. Compare both position files through their shared dates.

If the shared history positions differ, the strategy likely used future rows indirectly and the audit returns `fail`. This check does not prove a strategy is profitable, but it catches a common class of inflated backtests.

## Data-Snooping Bias Guard

RunJin adds a second audit for overfitting risk:

- `adjustable_parameter_count`: counts strategy parameters plus explicit sizing/exit knobs. The default rule of thumb is no more than five.
- `bars_per_parameter`: warns when there is too little history for the number of knobs.
- `assumed_trials`: asks you to count repeated manual tweaks, batch scans, and qualitative design choices as trials.
- `qualitative_choice_count`: flags choices such as template, execution timing, cost model, slippage model, and position model.
- `deflated_sharpe_proxy`: applies a simple Sharpe haircut based on assumed trial count. It is an explainable proxy, not the exact Bailey Deflated Sharpe Ratio.
- `sample_size_true_sharpe_ge_0`: applies practical Bailey-style thresholds: Sharpe >= 1 needs at least 681 data points, while Sharpe >= 2 needs at least 174 data points, to support confidence that true Sharpe is at least 0.
- `sample_size_true_sharpe_ge_1`: requires Sharpe >= 1.5 and at least 2,739 data points to support confidence that true Sharpe is at least 1.

The audit is intentionally conservative. A `review` result does not mean a strategy is invalid; it means the backtest performance should be discounted before you trust it.

## Batch Strategy Leaderboard

The vectorbt-inspired batch layer scans many symbols and parameter variants without executing user Python code or placing orders.

- Strategy templates: `sma_crossover`, `rsi_mean_reversion`, `macd_trend`, `bollinger_reversion`.
- Controls: selected universe, max symbols, selected templates, and max variants per template.
- Outputs: ranked leaderboard, return, max drawdown, Sharpe, win rate, trade count, score, and top indexed equity curves.
- Boundary: this is a research scanner. Large full-market runs should move into a persisted job table before being scheduled.

## Upgrade Path

1. Add a job table for long-running parameter sweeps.
2. Add optional vectorbt acceleration for very large grid searches and portfolio-level signal matrices.
3. Add walk-forward validation and train/test splits.
4. Add liquidity constraints and point-in-time data snapshots.
5. Add result persistence so every backtest can be reproduced from a saved spec and data snapshot.
