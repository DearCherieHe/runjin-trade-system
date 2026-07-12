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

## TradingAgents-Astock Inspired Capabilities

The A-share research flow borrows the decision structure from TradingAgents-Astock, but presents it as a compact trader-facing checklist instead of an engineering graph:

- A-share analyst chain: market, sentiment, news, fundamentals, policy, hot-money flow, and unlock/insider-selling checks.
- Debate structure: Bull vs Bear first, then Research Manager synthesis, then aggressive/neutral/conservative risk opinions.
- A-share constraints: T+1, price limits, 100-share lots, ST/delisting risk, and trading-session behavior must affect the final plan.
- Data routing principle: prefer stable free data sources for routine quotes, and reserve Eastmoney/Tonghuashun-style sources for unique A-share data such as hot money, unlocks, and consensus.
- Output discipline: every result should end with one action, position ceiling, invalidation condition, and the next data point to verify.

## TradingAgents-AShare Inspired Capabilities

KylinMountain/TradingAgents-AShare adds a stronger daily-operator lens. RunJin should absorb the workflow, not the UI complexity:

- Natural-language task entry: let the trader type `分析汇川技术中线还能不能拿`, then infer symbol, horizon, and analysis priority.
- Multi-agent collaboration: keep the analyst, bull/bear, trader, risk, and portfolio-manager chain visible as a decision process.
- Scheduled watchlist review: core A-share names should support daily or weekly re-analysis with clear success/failure status.
- Holding tracker: cost, position weight, P/L, latest report date, and active risk should sit beside the research conclusion.
- Structured report cards: every saved report should expose conclusion, confidence, target price, stop/invalidation level, core risks, and next check.
- Model readiness: model warmup and API test details belong in `系统能力`, not in the main trading workflow.

## TauricResearch TradingAgents Inspired Capabilities

The upstream TradingAgents project is most useful as a process architecture reference. RunJin should absorb the durable operating ideas:

- Role chain: analyst evidence, bull/bear debate, trader plan, risk debate, and portfolio-manager decision.
- Fast/deep model split: lightweight extraction can use quick models; final debate and portfolio decisions should use the stronger model.
- Debate budget: let the user control debate rounds, and require each round to add new evidence instead of repeating prior claims.
- Decision memory: store action, confidence, price context, invalidation condition, and later review notes for each report.
- Checkpoint recovery: long analysis should resume from the last completed stage after interruption.
- Multi-market discipline: keep one research workflow across A-shares, HK, US, and crypto, but never erase market-specific trading rules.

## Vibe-Trading Inspired Capabilities

HKUDS/Vibe-Trading is useful for the trader-facing loop from natural language to analysis, validation, report, and memory:

- Natural-language workflow: a plain trading question should become a structured research task without forcing the user through many fields.
- Tool-backed research: the system should gather market data, fundamentals, news, comparable history, and prior notes before drafting a view.
- Backtest before belief: when a thesis implies a repeatable rule, RunJin should run a quick historical check before treating the story as usable.
- Persistent memory: reports, invalidation points, and later outcomes should feed future analysis instead of disappearing after one session.
- Shadow account review: real or simulated trades should be compared with prior research cards to identify impulsive entries, loss-adding, early profit taking, concept chasing, and ignored liquidity warnings.

## AI-Trader Inspired Capabilities

HKUDS/AI-Trader points toward social signals, simulated trading, leaderboards, and agent-published trade ideas. RunJin should absorb the discipline layer without turning into an auto-execution product:

- Signal gatekeeping: every AI-generated idea must pass research-card completeness, risk budget, market constraint, anti-thesis, and duplicate-signal checks.
- Paper execution first: signals should enter simulated execution before any real-money consideration.
- Lifecycle tracking: record signal publication, simulated fill, holding updates, exit reason, and final performance.
- Signal quality scoreboard: rank sources by win rate, payoff ratio, drawdown, and stability, not just recent flashy returns.
- Copy-trading boundary: RunJin should not place real orders or encourage blind following; any real trade remains a user-confirmed external action.

## QuantDinger Inspired Capabilities

QuantDinger is useful as a reference for an AI-native quantitative trading operating system. RunJin should absorb its research-to-simulation discipline and safety model:

- Strategy OS flow: AI research, strategy development, backtest verification, paper execution, and risk monitoring should be connected as one lifecycle.
- Code contract: AI-generated strategies must expose entry, exit, invalidation, sizing, and exception handling before they can be tested.
- Reproducible backtests: every backtest should preserve parameters, data window, trades, curve, drawdown, and warnings.
- Paper-first execution: generated strategies and signals default to simulated execution; live trading is not a default capability.
- Permission safety: model agents should have scoped permissions, with read/backtest/paper permissions separated from any future live trading permission.
- Audit trail: model suggestions, strategy edits, backtest runs, paper fills, and risk overrides should be logged for later review.
