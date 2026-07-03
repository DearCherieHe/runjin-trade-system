from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main():
    try:
        import yfinance as yf
    except ImportError:
        print("yfinance is not installed. Offline sample data remains available.")
        return

    tickers = ["NVDA", "TSLA", "SNDK", "PLTR", "AMD", "AVGO", "TSM", "ARM", "SOUN", "CRWV"]
    try:
        data = yf.download(tickers, period="1y", group_by="ticker", auto_adjust=False, progress=False)
    except Exception as exc:
        print(f"Live refresh failed: {exc}. Offline sample data remains available.")
        return
    out = ROOT / "data" / "sample" / "yfinance_latest_snapshot.csv"
    data.to_csv(out)
    print(f"Wrote optional live snapshot to {out}")


if __name__ == "__main__":
    main()
