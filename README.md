# RunJin Trade System

润金交易系统 is an offline-first Streamlit demo for a long-term stock observation desk and a short-term paper-trading lab.

## Quick Start

```bash
python3 scripts/build_sample_data.py
python3 -m pip install -r requirements.txt
streamlit run app.py
```

If dependencies are already installed, the app runs fully from `data/sample` and does not need network access.

In this Codex desktop environment, the bundled Python can run the data generator and validator:

```bash
/Users/serena/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/build_sample_data.py
/Users/serena/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/validate_demo.py
```

The bundled Python does not include Streamlit, so install `requirements.txt` in the Python environment you want to use for the UI.

## Data Modes

The sidebar supports:

- `Live auto`: try live/near-live sources first, then fall back to sample data.
- `Sample only`: fully offline demo mode.
- `Live strict`: raise errors if live sources fail.

Current live adapters:

- US equities: `yfinance`
- Crypto: Binance public REST klines
- Optional China market candidates: `finshare`, `opendatatools`, `tushare`
- Broker placeholder: Tiger OpenAPI / `tigeropen`

Check source availability:

```bash
python3 scripts/check_live_sources.py
```

See `docs/live_data_strategy.md` for the paid data-source upgrade path.

## Boundaries

- No real brokerage integration.
- No exchange API keys.
- No leverage.
- Kronos-style forecast is a research-only sample adapter in V0.1.
