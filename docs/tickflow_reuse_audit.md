# TickFlow Stock Panel Reuse Audit

RunJin integrates the local `tickflow-stock-panel-main` project as a multi-market research workbench inside the existing `Finance MCP Radar` page. The source repository is MIT licensed, so the V0.1 implementation can reuse strategy ideas and lightweight logic without pulling in GPL constraints.

## Reused Ideas

- Screener workflow: strategy filters, signal reasons, ranking table, and monitor-style alerts across A-share, US stocks, Hong Kong stocks, and crypto.
- Indicator pipeline: moving averages, MACD, RSI, KDJ, Bollinger Bands, ATR, volume ratio, rolling highs/lows, and limit-up/down signals.
- Strategy templates: trend breakout, MA golden cross, MACD golden cross, volume-price surge, low-volatility leader, oversold bounce, Bollinger breakout, bullish alignment, limit-up momentum, support pullback, and low reversal.
- Rotation view: concept/industry/theme grouping with recent performance and leaders.
- Surge ladder: A-share limit-up ladder plus US/HK/crypto strong-breakout ladder using volume, momentum, and rolling-high context.
- Price levels: volume POC approximation, pivots, 60-day high/low, ATR stops, Fibonacci pullback levels, and gap midpoint.

## RunJin Adaptation

- No FastAPI or React service is started.
- No Polars, DuckDB, vectorbt, or TickFlow runtime dependency is required for demo startup.
- Offline sample A-share OHLCV, Hong Kong OHLCV, and multi-market concept mapping live under `data/sample`.
- The workbench remains research-only and does not place orders.
- The user-facing integration replaces the weaker FinanceMCP-only page body with a combined market intelligence and A-share quant workbench.

## Boundary

This module is designed for screening, replay, and research context. It is not a broker connector, exchange connector, or real-time execution system.
