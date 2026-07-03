# Trade Lab Demo V0.1

Trade Lab is a local Streamlit demo for two related workflows:

1. A long-term US stock observation desk for narrative, growth, financial proof, catalysts, and invalidation rules.
2. A short-term paper-trading lab for small, no-leverage strategy experiments on US stocks and crypto.

The demo is offline-first. The app loads bundled sample data from `data/sample` and does not need API keys or network access.

## Pages

- Dashboard: summary metrics, paper bot state, and risk alerts.
- Long Watchlist: ranked long-term candidate table.
- Stock Detail: K-line, financials, thesis, catalysts, invalidation, and buy plan.
- K-line Lab: deterministic indicators and research-only Kronos placeholder forecast.
- Short Bot: US stock trend-following and crypto mean-reversion paper simulations.
- Weekly Report: generated review of watchlist and bot status.

## Deferred

- Live brokerage connections.
- Real exchange API credentials.
- Real Kronos model download, training, or inference.
- Leveraged strategies.
