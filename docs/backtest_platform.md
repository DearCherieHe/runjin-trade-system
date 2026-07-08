# Backtest Platform

RunJin's Backtest Lab is designed as a research platform, not an execution engine.

## V0.1 Scope

- Engine: `backtesting.py`
- Input: constrained YAML strategy specs
- Assets: US stock OHLCV and crypto OHLCV already loaded by the app
- Outputs: equity curve, drawdown overlay, statistics, trade log, portfolio rebalance log, and latest weights
- Primary performance lens: Information Ratio, Sharpe Ratio, drawdown depth, drawdown duration, and trade frequency; simple return is not enough.
- Safety: no leverage, no real orders, no exchange keys, no arbitrary Python execution
- Bias control: default next-bar execution, truncated-data look-ahead audit, data-snooping audit, out-of-sample audit, survivorship-bias disclosure, and recent-regime review

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
out_of_sample_check:
  enabled: true
  train_pct: 0.70
  min_test_bars: 174
  max_sharpe_decay: 0.75
drawdown_tolerance:
  max_drawdown_pct: 10
  max_drawdown_duration_days: 90
  max_drawdown_duration_bars: 63
transaction_cost_check:
  enabled: true
  half_spread_bps: 5
  market_impact_bps: 0
  latency_slippage_bps: 1
  max_per_side_cost_bps: 20
  max_annual_cost_drag_pct: 6
  min_avg_trade_edge_multiple: 2
survivorship_check:
  enabled: true
  data_universe: current_listed_symbols
  includes_delisted: false
  includes_bankrupt: false
  includes_acquired: false
  point_in_time_membership: false
  low_price_bias: false
  low_valuation_bias: false
  distress_bias: false
  low_price_threshold: 5
regime_check:
  enabled: true
  recent_years: 3
  min_recent_bars: 174
  max_history_years_for_equal_weight: 5
  max_recent_sharpe_decay: 0.75
  min_recent_sharpe: 0
  current_costs_applied_to_full_history: true
  known_regime_breaks: ""
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

- Performance quality: Information Ratio, Sharpe, annualized return/volatility, max drawdown, max drawdown duration, and trades per year.
- Drawdown tolerance: compare peak-to-trough loss and longest underwater period against your stated threshold.
- Cost models: `pct_only`, `us_equity_basic`, `a_share_basic`, `hk_equity_basic`, `crypto_basic`.
- Slippage models: `close`, `hl_mean`, `hl_mean_gap_guard`.
- Transaction cost guard: commission, half-spread, market impact, and latency slippage are estimated in bps, then converted into round-trip and annualized strategy drag.
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

## Out-Of-Sample Guard

RunJin also reserves the most recent part of the history as an out-of-sample segment:

- `out_of_sample_split`: confirms the train/test split is valid. Default is 70% train and 30% test.
- `oos_min_test_bars`: checks that the test set has enough bars. Default minimum is 174 bars.
- `oos_return_reasonable`: flags strategies that collapse in the reserved test period.
- `oos_sharpe_decay`: compares train Sharpe and test Sharpe; large decay is treated as overfit risk.
- `oos_sample_size_true_sharpe_ge_0`: applies the same practical sample-size logic to the test set.

This is not full moving optimization yet. It is the first necessary guard: the final parameter set must remain reasonable on a recent period that was not supposed to guide model design.

## Drawdown Tolerance Guard

RunJin treats drawdown as a survivability constraint, not just a chart statistic:

- `max_drawdown_depth_tolerance`: compares the deepest peak-to-trough equity loss with `drawdown_tolerance.max_drawdown_pct`.
- `max_drawdown_duration_days_tolerance`: compares the longest calendar time spent underwater with `drawdown_tolerance.max_drawdown_duration_days`.
- `max_drawdown_duration_bars_tolerance`: compares the longest bar-based underwater duration with `drawdown_tolerance.max_drawdown_duration_bars`.

Maximum drawdown depth and maximum drawdown duration often occur in different periods. The guard reports both so you can ask the practical question: would you still be able to keep trading this strategy without being forced to stop?

## Transaction Cost Guard

RunJin treats every buy and every sell as a separate transaction. The guard estimates:

- `per_side_transaction_cost`: commission plus half-spread, market impact, and latency slippage.
- `round_trip_transaction_cost`: buy cost plus sell cost.
- `annual_cost_drag`: round-trip cost multiplied by estimated trades per year.
- `average_trade_edge_vs_cost`: whether average trade edge is large enough to survive round-trip cost.
- `cost_adjusted_return_proxy`: conservative return after estimated cost drag.
- `cost_adjusted_sharpe_proxy`: conservative Sharpe after annualized cost drag.

This prevents high-turnover strategies from looking good only because the backtest forgot spreads, market impact, or latency slippage.

## Survivorship Bias Guard

RunJin treats survivorship bias as a data-quality disclosure, not a solved problem. If a historical database only contains symbols that are still listed today, it can overstate strategy performance because bankrupt, delisted, merged, or acquired losers are missing.

The guard checks:

- `data_universe_declared`: whether the strategy spec names the universe source.
- `includes_delisted_bankrupt_acquired`: whether delisted, bankrupt, and acquired symbols are included.
- `point_in_time_membership`: whether universe membership reflects what was actually tradable at each historical date.
- `strategy_survivorship_vulnerability`: whether the strategy is especially exposed because it targets low-price, low-valuation, or distressed names.

The bundled sample and common current-listing data sources should be treated as `review` unless you explicitly provide delisted/bankrupt/acquired coverage and point-in-time membership. A vulnerable low-price or distressed strategy should be treated as `fail` until tested on survivorship-free data.

## Recent Regime Guard

RunJin does not treat a ten-year headline backtest as equally useful across all years. Older periods can look better because fewer funds competed for the same edge, spreads and liquidity were different, missing delisted stocks distort early history, and market rules or macro regimes changed.

The guard checks:

- `backtest_history_span`: whether the test is long enough to require regime context instead of equal weighting.
- `recent_regime_window`: whether the latest period has enough bars to judge current usefulness.
- `recent_vs_full_performance`: whether recent Sharpe and return still support the strategy.
- `early_performance_inflation`: whether early-period performance dominates recent evidence.
- `historical_cost_stationarity`: whether today's cost assumptions were applied across all historical years.
- `regime_change_disclosure`: whether known regulatory, market-structure, or macro breaks were declared.
- `nonstationarity_warning`: whether recent volatility diverges sharply from older history.

This guard does not say old data is useless. It says old data should be discounted when markets are non-stationary. For strategy adoption, the latest few years and out-of-sample behavior should carry more decision weight than a pretty full-period chart.

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
