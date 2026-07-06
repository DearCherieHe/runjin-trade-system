# RunJin Trade System

润金交易系统 is a live-data Streamlit research cockpit for a long-term stock observation desk and a short-term paper-trading lab.

## Quick Start

```bash
python3 -m pip install -r requirements.txt
streamlit run app.py
```

The app starts in live-data mode and requires network access for current market data. Bundled files under `data/sample` remain only for local validation and development checks.

In this Codex desktop environment, the bundled Python can run the data generator and validator:

```bash
/Users/serena/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/build_sample_data.py
/Users/serena/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/validate_demo.py
```

The bundled Python does not include Streamlit, so install `requirements.txt` in the Python environment you want to use for the UI.

## Data Mode

The sidebar now uses `Live data` only. Live source failures are shown directly in the UI instead of silently falling back to old sample data.

Current live adapters:

- US equities: `yfinance`
- Crypto: Binance public REST klines, then `yfinance` crypto fallback
- Quarterly fundamentals: `yfinance`
- Optional China market candidates: `finshare`, `opendatatools`, `tushare`
- Broker placeholder: Tiger OpenAPI / `tigeropen`

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
- coverage notes so live/free-source limits are visible

## Boundaries

- No real brokerage integration.
- No exchange API keys.
- No leverage.
- Local sample forecasts are disabled while the app is running in live-data mode.
