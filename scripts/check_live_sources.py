import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_sources.loaders import get_data_source_status, load_crypto_prices, load_financials, load_prices


def main():
    print("Checking live/near-live sources...")
    try:
        stocks = load_prices(data_mode="live")
        print(f"US equities rows: {len(stocks):,}")
    except Exception as exc:
        print(f"US equities failed: {exc}")

    try:
        crypto = load_crypto_prices(data_mode="live")
        print(f"Crypto rows: {len(crypto):,}")
    except Exception as exc:
        print(f"Crypto failed: {exc}")

    try:
        financials = load_financials(data_mode="live")
        print(f"Financial rows: {len(financials):,}")
    except Exception as exc:
        print(f"Financials failed: {exc}")

    print("Source status:")
    for layer, status in get_data_source_status().items():
        print(f"- {layer}: {status}")


if __name__ == "__main__":
    main()
