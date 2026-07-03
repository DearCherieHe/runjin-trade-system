# Trade Lab

Offline-first Streamlit demo for a long-term stock observation desk and a short-term paper-trading lab.

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

## Boundaries

- No real brokerage integration.
- No exchange API keys.
- No leverage.
- Kronos-style forecast is a research-only sample adapter in V0.1.
