# Trade Lab Demo V0.1

Trade Lab is a Streamlit research desk for two related workflows:

1. A long-term US stock observation desk for narrative, growth, financial proof, catalysts, and invalidation rules.
2. A short-term paper-trading lab for small, no-leverage strategy experiments on US stocks and crypto.

The app is realtime-first in the Streamlit UI. Bundled sample files remain for validation and development, but the primary workspace is configured to stop rather than silently fall back when live research feeds fail.

## Pages

- Dashboard: summary metrics, paper bot state, and risk alerts.
- Long-term Investing workbench: Research Desk and Weekly Review.
- Research Desk: market universe, long watchlist, finance radar, symbol cockpit, and research journal.
- Mid/Short Trading workbench: Signal Lab, Capital Rotation, Backtest Lab, Paper Bot, and Weekly Review.
- Signal Lab: K-line launch-point research, SEPA trend template, and intraday scenarios.
- Paper Bot: US stock trend-following and crypto mean-reversion paper simulations.
- Weekly Review: generated review of watchlist and bot status.

## Deferred

- Live brokerage connections.
- Real exchange API credentials.
- Real Kronos model download, training, or inference.
- Leveraged strategies.

## Launch Point Research

The K-line layer treats charts as behavioral records: price and volume express the crowd's expectations, fear, greed, forced selling, and breakout chasing. Similar historical structures can be useful context for future scenarios, but they are not deterministic predictions.

V0.1 studies four launch-point families:

- Pattern launch: large bullish structures and breakout candles.
- Gap launch: upward gap behavior and whether the gap holds.
- Original-control proxy: high-evidence pivot support/resistance as a practical approximation.
- Bollinger launch: lower-band reclaim, squeeze release, and upper-band expansion.

Every candidate must show an entry zone, invalidation level, trailing-stop reference, confidence score, and note. The panel is research-only and never places orders.
