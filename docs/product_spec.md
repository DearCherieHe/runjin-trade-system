# Trade Lab Demo V0.1

Trade Lab is a Streamlit research desk for two related workflows:

1. A long-term future-industry map for finding potential 10x stocks through value-chain position, monopoly potential, gross-margin power, and financial proof.
2. A short-term paper-trading lab for small, no-leverage strategy experiments on US stocks and crypto.

The app is realtime-first in the Streamlit UI. Bundled sample files remain for validation and development, but the primary workspace is configured to stop rather than silently fall back when live research feeds fail.

## Pages

- Dashboard: summary metrics, paper bot state, and risk alerts.
- Long-term Investing workbench: Research Desk and Weekly Review.
- Research Desk: one-stock 10x profile, future industry map, long watchlist, finance radar, and research journal.
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

## Future 10x Stock Profile

The long-term workbench starts from industries, not tickers. Each candidate is mapped into:

- Mega theme: AI compute, AI Agent, humanoid/autonomy, AI medical, additive manufacturing, brain-computer interface, or other future industry.
- Value-chain layer: equipment, core components, materials, software, data, or platform.
- Relationship map: upstream suppliers, direct peers, and downstream demand owners.
- Quality scores: monopoly potential, gross-margin power, irreplaceability, ten-year optionality, and capital intensity.
- Proof questions: why the company could become a 10x candidate and what evidence would invalidate the thesis.

The page shows one stock at a time through a selector. The goal is to identify which layer of a future industry is most likely to become a high-margin toll road, not merely which terminal product is exciting.
