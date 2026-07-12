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

The long-term module keeps a shared observation discipline:

- Continue / hold conditions: long-term industry demand is intact, the company remains a core supplier, financial reports are not deteriorating continuously, the technology route has not been replaced, valuation already reflects pessimistic expectations, and the weekly/monthly chart has not broken a long-term platform on heavy volume.
- Exit / stop-tracking conditions: demand logic is invalidated, customers are lost or products fall behind, gross margin collapses for a long period, financial condition worsens, peers become clearly stronger while the name only follows the concept, or price breaks below a long-term platform and fails to rebound with strength.
- Tenbagger discovery framework: industry trend up + company position ahead + financial reports gradually confirm + valuation has a margin of safety + monthly/weekly chart starts to stop falling with volume.

The page shows one stock at a time through a selector. The goal is to identify which layer of a future industry is most likely to become a high-margin toll road, not merely which terminal product is exciting.

## Interface Discipline

The product should feel like an operating cockpit, not an engineering console:

- First screen priority: show status, decision, risk, and the next useful action.
- Hide plumbing: data sources, modes, fallbacks, and adapter details belong in collapsed debug areas unless they block the user.
- Keep navigation flat: sidebar entries should be short Chinese work areas with compact letter cues.
- Avoid repeated structures: each page should have one primary table or chart group, with secondary details behind tabs or expanders.
- Prefer business labels over implementation labels: use `牛股框架`, `观察清单`, `技术分析`, and `模拟交易` instead of internal module names.

## Local Research Assistant

The long-term research desk includes a local Ollama assistant. It defaults to `qwen2.5:7b` through `http://localhost:11434` and is used only after the user asks a question. The assistant receives the selected ticker thesis, financial snapshot, hold/exit checklist, and tenbagger framework, then returns a concise research answer with evidence, invalidation points, and next checks. It must stay framed as research support, not order execution or guaranteed advice.

## TradingAgents-CN Inspired Capabilities

The system borrows product ideas from TradingAgents-CN while keeping the implementation native to RunJin and avoiding proprietary frontend/backend copying:

- Single-stock sync result: show main route, fallback route, failure reason, and `market_quotes` persistence status.
- AKShare quote fallback: `stock_bid_ask_em -> stock_zh_a_spot -> stock_zh_a_spot_em -> stock_zh_a_hist`.
- Cache management: file cache is implemented now; MongoDB/Redis remain future backends behind the same cache interface.
- LLM provider catalog: providers and task-model policies are visible in `系统能力`.
- Batch analysis: multiple watchlist names can be ranked in one pass by score and latest financial proof.
- Report export: Markdown export is available now; Word/PDF are listed as planned export backends.
- Debug visibility: data-source status and low-level plumbing stay under `系统能力`, not on the first screen.
