import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

# Import the Miner Class we just created
from src.miner import VCPMiner


def clean_column_names(df):
    """
    Normalize column names to standard ['Open', 'High', 'Low', 'Close', 'Volume'].
    Handles Uppercase, Lowercase, and Korean.
    """
    # 1. Convert all existing columns to Lowercase first for easier matching
    df.columns = [c.lower() for c in df.columns]

    # 2. Map Lowercase/Korean inputs to Standard Output
    col_map = {
        # Standard Lowercase -> Target
        "date": "Date",
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
        "adj close": "Close",
        # Korean -> Target
        "일자": "Date",
        "시가": "Open",
        "고가": "High",
        "저가": "Low",
        "종가": "Close",
        "거래량": "Volume",
    }

    # 3. Rename columns
    # We use a loop to rename only the ones that exist in the map
    new_names = {}
    for col in df.columns:
        if col in col_map:
            new_names[col] = col_map[col]

    df = df.rename(columns=new_names)

    # 4. Check if we have the Big 5
    required = ["Open", "High", "Low", "Close", "Volume"]
    if not all(col in df.columns for col in required):
        # Determine what's missing for debugging
        missing = [col for col in required if col not in df.columns]
        # print(f"Missing columns: {missing}") # Uncomment to debug
        return None

    return df[required]


def main():
    # 1. Setup Directories
    RAW_DIR = Path("01_data_raw_2")
    PROCESSED_DIR = Path("02_data_processed_2")
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = list(RAW_DIR.glob("*.csv"))
    if not csv_files:
        print(f"Error: No CSV files found in {RAW_DIR}")
        return

    print(f"Found {len(csv_files)} stock files. Initializing Miner...")

    # 2. Initialize VCP Miner
    # Using 120-day window and pivot lookback as decided
    miner = VCPMiner(input_window=120, pivot_lookback=120)

    all_X = []
    all_y = []
    all_meta = []

    # 3. Mining Loop
    print("Starting extraction...")
    for file_path in tqdm(csv_files):
        try:
            # Read Data
            df = pd.read_csv(file_path)

            # Identify Date Column
            date_col = None
            for col in df.columns:
                if "date" in col.lower() or "일자" in col:
                    date_col = col
                    break

            if date_col:
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col).sort_index()

            # Clean Headers
            df = clean_column_names(df)
            if df is None:
                continue

            # Extract Patterns
            X, y, meta = miner.process_stock(df, ticker_name=file_path.stem)

            if len(X) > 0:
                all_X.append(X)
                all_y.append(y)
                all_meta.extend(meta)

        except Exception as e:
            # Silent error handling to keep the loop going
            continue

    # 4. Save Results
    if all_X:
        final_X = np.concatenate(all_X)
        final_y = np.concatenate(all_y)

        # Save Metadata (Useful for debugging "Why did it pick this?")
        meta_df = pd.DataFrame(all_meta)
        meta_df.to_csv(PROCESSED_DIR / "metadata.csv", index=False)

        # Save Tensors
        np.save(PROCESSED_DIR / "train_X.npy", final_X)
        np.save(PROCESSED_DIR / "train_y.npy", final_y)

        print("\n" + "=" * 30)
        print(f"MINING COMPLETE")
        print(f"Total Samples: {len(final_y)}")
        print(f"Success Rate:  {(final_y.mean()*100):.2f}%")
        print(f"Saved to:      {PROCESSED_DIR}")
        print("=" * 30)
    else:
        print("\nNo patterns found. Check your CSV format or loosen constraints.")


if __name__ == "__main__":
    main()
