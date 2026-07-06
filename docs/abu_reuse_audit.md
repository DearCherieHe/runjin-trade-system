# ABU Reuse Audit

RunJin reviewed the local ABU source tree at `/Users/serena/Documents/Notes/Trade/Github/abu-master`.

## Legal Boundary

ABU is licensed under GPL-3.0. RunJin does not copy `abupy` source files into the main package by default. The V0.1 integration is a clean-room style rewrite of selected ideas using RunJin's existing pandas/numpy modules.

Direct code reuse should only happen in a clearly separated GPL-compatible area, such as a future `vendor/abupy_gpl/` adapter, after accepting the license obligations.

## Useful ABU Ideas

| ABU area | Idea absorbed into RunJin | RunJin location |
|---|---|---|
| `TradeBu` | Separate orders, actions, capital, benchmark, and execution assumptions. | `src/backtest_lab/engine.py` |
| `MetricsBu` | Report risk and trade quality beyond simple return. | `src/backtest_lab/metrics.py` |
| `BetaBu` | ATR-based position sizing instead of fixed exposure only. | `src/backtest_lab/position_sizing.py` |
| `SlippageBu` / `TradeBu.ABuCommission` | Explicit commission and slippage assumptions. | `src/backtest_lab/costs.py` |
| `UmpBu` | A裁判 layer that can block or review risky trades. | `src/backtest_lab/ump_lite.py` |
| `TLineBu` / `SimilarBu` | Gap, ATR, correlation, and similar-path research context. | `src/kline/abu_research.py` |

## Not Reused Directly

- ABU's old execution engine is not loaded inside Streamlit.
- ABU's full ML / UMP training stack is not enabled in V0.1.
- ABU's notebook UI widgets are not ported.
- ABU's data download stack is not used as a live data dependency.

## Engineering Notes

ABU 0.4.0 targets older Python/pandas environments and includes APIs such as `DataFrame.append`, which are incompatible with modern pandas. RunJin keeps the dependency surface small and uses current pandas-compatible implementations.

All ABU-style outputs remain research-only and do not place or recommend live orders.
