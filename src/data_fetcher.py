import csv
import os
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import OpenDartReader
from dotenv import load_dotenv
from pykrx import stock

# Define paths relative to project root
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
DATA_RAW_DIR = PROJECT_ROOT / "01_data_raw"
METADATA_FILE = DATA_RAW_DIR / "stock_metadata.csv"


class StockDataFetcher:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("DART_API_KEY")
        if not self.api_key:
            print("Warning: DART_API_KEY not found in environment variables.")
            self.dart = None
        else:
            self.dart = OpenDartReader(self.api_key)

        self.induty_dict = self._load_industry_codes()

    def _load_industry_codes(self) -> Dict[str, str]:
        """
        Load industry codes from Excel file.
        Returns a flat dictionary mapping specific industry code to description.
        """
        excel_path = PROJECT_ROOT / "induty_code.xlsx"
        if not excel_path.exists():
            print(f"Warning: {excel_path} not found.")
            return {}

        try:
            df = pd.read_excel(excel_path)
            code_map = {}

            for _, row in df.iterrows():
                code = str(row["산업코드"])
                if not code.isalpha():
                    normalized_code = code.lstrip("0")
                    if not normalized_code:
                        normalized_code = "0"

                    code_map[normalized_code] = row["산업내용"]

            return code_map
        except Exception as e:
            print(f"Error loading industry codes: {e}")
            return {}

    def get_industry_info(self, ticker: str) -> Optional[Dict[str, str]]:
        """
        Fetch industry info from DART for a given ticker.
        Returns None if status is not '000' or API fails, to enable strict filtering.
        """

        if not self.dart:
            return None

        try:
            # Note: This makes an API call per ticker.
            time.sleep(0.1)  # Small delay to be gentle on API limits
            company_info = self.dart.company(ticker)

            if company_info and company_info.get("status") == "000":
                ind_code = company_info.get("induty_code", "")

                # Try to map code to name using loaded excel data
                ind_name = self.induty_dict.get(ind_code.lstrip("0"), "")
                if not ind_name:
                    ind_name = self.induty_dict.get(ind_code, "")

                result = {"industry_code": ind_code, "industry_name": ind_name}

                return result
        except Exception as e:
            print(f"\nError fetching DART info for {ticker}: {e}")
            pass

        return None

    def get_market_tickers(self, market: str) -> List[Dict[str, str]]:
        """
        Fetch all tickers and their names for a given market.
        Only returns tickers that successfully fetch DART info (status '000')
        and that do not contain the word "스팩" in their name.
        """
        tickers = stock.get_market_ticker_list(market=market)
        results = []

        print(f"Fetching metadata for {len(tickers)} {market} stocks...")

        for i, ticker in enumerate(tickers):
            name = stock.get_market_ticker_name(ticker)

            # Skip stocks whose names contain "스팩"
            if "스팩" in name:
                continue

            # Strict filtering: only proceed if we get valid industry info
            ind_info = self.get_industry_info(ticker)

            if ind_info:
                info = {"ticker": ticker, "name": name, "market": market, **ind_info}
                results.append(info)
            # Else: silently skip or log if verbose (skipped to match "not even save it")

            # Progress indicator
            if i % 50 == 0:
                print(
                    f"Processed {i}/{len(tickers)}: {name} ({len(results)} valid so far)"
                )

        return results

    def fetch_daily_ohlcv(
        self, ticker: str, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """
        Fetch daily OHLCV data for a specific ticker from pykrx.
        """
        try:
            df = stock.get_market_ohlcv_by_date(start_date, end_date, ticker)

            daily_prices = []
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
            print(f"Error fetching prices for {ticker}: {e}")
            return []

    def save_prices_to_csv(self, data: List[Dict[str, Any]], filepath: Path) -> None:
        """
        Save daily price list to CSV.
        """
        if not data:
            return

        fieldnames = ["date", "open", "high", "low", "close", "volume"]
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)

    def save_metadata_to_csv(self, data: List[Dict[str, str]], filepath: Path) -> None:
        """
        Save stock metadata to a separate CSV file.
        """
        if not data:
            return

        fieldnames = ["ticker", "name", "market", "industry_code", "industry_name"]
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)


def fetch_data_pipeline(start_date: str, end_date: str):
    """
    Main pipeline to fetch and save stock data.
    """
    fetcher = StockDataFetcher()

    print(f"Saving data to: {DATA_RAW_DIR}")
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    all_valid_stocks = []

    # 1. Fetch Metadata (Tickers + Industry Codes) with strict filtering
    for market in ["KOSPI", "KOSDAQ"]:
        print(f"Fetching {market} ticker list...")
        market_stocks = fetcher.get_market_tickers(market)
        all_valid_stocks.extend(market_stocks)

    # Save Metadata Registry (Only valid stocks)
    print(f"Saving metadata registry to {METADATA_FILE}")
    fetcher.save_metadata_to_csv(all_valid_stocks, METADATA_FILE)

    print(f"Found {len(all_valid_stocks)} valid stocks total.")

    # 2. Fetch Daily Prices for valid stocks only
    for stock_info in all_valid_stocks:
        ticker = stock_info["ticker"]
        name = stock_info["name"]
        market = stock_info["market"]

        safe_name = name.replace("/", "_").replace("\\", "_")
        filename = f"{ticker}_{safe_name}_{market}.csv"
        filepath = DATA_RAW_DIR / filename

        print(f"Processing prices for {ticker} ({name})...")
        prices = fetcher.fetch_daily_ohlcv(ticker, start_date, end_date)

        if prices:
            fetcher.save_prices_to_csv(prices, filepath)
        else:
            print(f"No price data found for {ticker}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) >= 3:
        fetch_data_pipeline(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python data_fetcher.py YYYYMMDD YYYYMMDD")
