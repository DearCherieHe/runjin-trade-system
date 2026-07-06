# RunJin Trade System

润金交易系统 is a live-first Streamlit research cockpit for a long-term stock observation desk and a short-term paper-trading lab.

## Quick Start

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

The app starts in `Live auto` mode: it tries current market sources first, then falls back to bundled files under `data/sample` when public data endpoints are blocked, rate-limited, or empty.

In this Codex desktop environment, the bundled Python can run the data generator and validator:

```bash
/Users/serena/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/build_sample_data.py
/Users/serena/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/validate_demo.py
```

The bundled Python does not include Streamlit, so install `requirements.txt` in the Python environment you want to use for the UI.

## Data Modes

The sidebar supports:

- `Live auto`: try live/near-live sources first, then fall back to bundled sample data.
- `Sample only`: use offline demo data only.
- `Live strict`: raise an error if live sources fail.

Current live adapters:

- US equities: `yfinance`
- Market universe: US stocks, A-shares, Hong Kong stocks, and Singapore stocks with a USD 300M equivalent market-cap floor
- Crypto: Binance public REST klines, then `yfinance` crypto fallback
- Quarterly fundamentals: `yfinance`
- FinanceMCP-style research radar: sample-first catalog, optional external CSV via `FINANCE_MCP_RESEARCH_CSV`, optional MCP/HTTP endpoint placeholder via `FINANCE_MCP_HTTP_URL`
- Optional China market candidates: `finshare`, `opendatatools`, `tushare`
- Broker placeholder: Tiger OpenAPI / `tigeropen`

## Market Universe

The app separates the broad tradable universe from the hand-scored long-term watchlist.

Universe rule:

- Include US stocks, A-shares, Hong Kong stocks, and Singapore stocks.
- Exclude symbols with market cap below USD 300M equivalent.
- Select the top 3,000 symbols per market group by USD market cap after the micro-cap filter.
- Preserve `market_group`, `market`, `exchange`, `currency`, `market_cap_usd`, `market_rank`, and Yahoo-compatible ticker fields for filtering and chart selection.

Baseline exchange listing CSVs are committed under `data/listings/` and configured in `configs/live_sources.yaml` under `free_sources.market_universe.markets.*.listing_csv`.
Each CSV can use English or common Chinese column names for ticker, company/name, exchange, currency, market cap, and USD market cap.
If a full provider export is not available yet, the committed baseline CSVs keep `Live auto`, `Live strict`, and Streamlit Cloud startup stable. If all configured CSVs are removed or unavailable, the app still falls back to a small seed universe and marks the source status accordingly.

Validate or optionally refresh the configured listing CSVs:

```bash
python3 scripts/refresh_market_listings.py
python3 scripts/refresh_market_listings.py --yfinance-us
```

Check source availability:

```bash
python3 scripts/check_live_sources.py
```

See `docs/live_data_strategy.md` for the paid data-source upgrade path.

## K-line Replay

The Stock Detail and K-line Lab pages support:

- `1H / 1D / 1W / 1M / 1Q / 1Y` periods
- historical `Replay as of` date selection
- local OHLCV resampling for weekly/monthly/quarterly/yearly bars
- dynamic `1W / 1M / 1Q / YTD / 1Y / ALL` range buttons and dark range sliders on time-series charts
- KDJ, RSI, MACD, Bollinger Bands, moving averages, volatility, and relative strength
- coverage notes so live/free-source limits are visible

## Finance MCP Radar

The `Finance MCP Radar` page turns FinanceMCP-style capabilities into a research-only layer:

- market news, macro calendar, money flow, index/constituent context, fundamentals, valuation, China-market candidates, and crypto context
- source/status visibility in the Dashboard data-source table
- sample-first operation so Streamlit starts without API keys
- optional external research CSV path through `FINANCE_MCP_RESEARCH_CSV`
- optional MCP/HTTP service pointer through `FINANCE_MCP_HTTP_URL`

## Backtest Lab

The `Backtest Lab` page uses `backtesting.py` as the execution engine instead of a homegrown backtester.

- input strategy specs as editable YAML, not arbitrary Python code
- supported templates: `sma_crossover`, `rsi_mean_reversion`, `bollinger_reversion`, `macd_trend`
- portfolio templates: `equal_weight_rebalance`, `momentum_top_n`, `inverse_volatility`
- supported assets: US stock OHLCV and crypto OHLCV from the current data mode
- outputs: return, max drawdown, Sharpe, win rate, trade count, equity curve, drawdown overlay, statistics table, trade log, rebalance log, and latest weights
- V0.1 is research-only: no leverage, no broker orders, no exchange keys, no automatic live trading

## Boundaries

- No real brokerage integration.
- No exchange API keys.
- No leverage.
- Local sample forecasts are disabled while the app is running in live-data mode.
