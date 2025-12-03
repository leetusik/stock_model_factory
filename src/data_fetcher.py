import csv
from pathlib import Path
from typing import Any, Dict, List

from pykrx import stock

# Define paths relative to project root
# This assumes the structure:
# project_root/
#   src/
#     data_fetcher.py
#   01_data_raw/
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DATA_RAW_DIR = PROJECT_ROOT / "01_data_raw"


def get_market_tickers(market: str) -> List[Dict[str, str]]:
    """
    Fetch all tickers and their names for a given market.

    Args:
        market: "KOSPI" or "KOSDAQ"

    Returns:
        List of dicts containing ticker, name, and market.
    """
    tickers = stock.get_market_ticker_list(market=market)
    results = []
    for ticker in tickers:
        name = stock.get_market_ticker_name(ticker)
        results.append({"ticker": ticker, "name": name, "market": market})
    return results


def fetch_daily_ohlcv(
    ticker: str, start_date: str, end_date: str
) -> List[Dict[str, Any]]:
    """
    Fetch daily OHLCV data for a specific ticker from pykrx.

    Args:
        ticker: Stock ticker code (e.g., "005930")
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format

    Returns:
        List of daily price dictionaries.
    """
    try:
        df = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)

        daily_prices = []
        # pykrx returns index as Date
        for date, row in df.iterrows():
            daily_prices.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "open": float(row["시가"]),
                    "high": float(row["고가"]),
                    "low": float(row["저가"]),
                    "close": float(row["종가"]),
                    "volume": int(row["거래량"]),
                }
            )

        return daily_prices
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        return []


def save_to_csv(data: List[Dict[str, Any]], filepath: Path) -> None:
    """
    Save list of dicts to CSV.
    """
    if not data:
        return

    fieldnames = ["date", "open", "high", "low", "close", "volume"]

    # Create directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def fetch_data_pipeline(start_date: str, end_date: str):
    """
    Main pipeline to fetch and save stock data.
    """
    print(f"Saving data to: {DATA_RAW_DIR}")
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_stocks = []
    for market in ["KOSPI", "KOSDAQ"]:
        print(f"Fetching {market} ticker list...")
        all_stocks.extend(get_market_tickers(market))

    print(f"Found {len(all_stocks)} stocks total.")

    for stock_info in all_stocks:
        ticker = stock_info["ticker"]
        name = stock_info["name"]
        market = stock_info["market"]

        # Sanitize name for filename (remove slashes etc)
        safe_name = name.replace("/", "_").replace("\\", "_")

        # Format: ticker_name_market.csv
        filename = f"{ticker}_{safe_name}_{market}.csv"
        filepath = DATA_RAW_DIR / filename

        print(f"Processing {ticker} ({name})...")
        prices = fetch_daily_ohlcv(ticker, start_date, end_date)

        if prices:
            save_to_csv(prices, filepath)
        else:
            print(f"No data found for {ticker}")


if __name__ == "__main__":
    # Default usage for testing
    import sys

    if len(sys.argv) >= 3:
        fetch_data_pipeline(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python data_fetcher.py YYYYMMDD YYYYMMDD")
