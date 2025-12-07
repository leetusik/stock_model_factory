import pandas as pd
import numpy as np


class VCPMiner:
    def __init__(
        self,
        input_window=120,  # 6 months context
        future_window=20,  # 1 month result
        pivot_lookback=120,  # 6 months ceiling
        vol_contraction_ratio=0.7,
        target_gain=0.20,  # +20%
        stop_loss=-0.10,  # -10%
        max_gap_allowed=0.05,
    ):  # Skip if gap > 5%

        self.input_window = input_window
        self.future_window = future_window
        self.pivot_lookback = pivot_lookback
        self.vol_ratio = vol_contraction_ratio
        self.target = target_gain
        self.stop = stop_loss
        self.max_gap = max_gap_allowed

    def process_stock(self, df, ticker_name="Unknown"):
        """
        Input: DataFrame with 'Open', 'High', 'Low', 'Close', 'Volume'
        Output: X (Array), y (Array), metadata (List)
        """
        # 1. DATA CLEANING & PRE-CALCS
        if len(df) < 300:
            return [], [], []  # Not enough history

        df = df.copy()

        # Calculate Volatility
        df["Vol_Short"] = df["Close"].rolling(window=10).std()
        df["Vol_Long"] = df["Close"].rolling(window=60).std()

        # Calculate Moving Averages (Trend Template)
        df["SMA_50"] = df["Close"].rolling(window=50).mean()
        df["SMA_150"] = df["Close"].rolling(window=150).mean()
        df["SMA_200"] = df["Close"].rolling(window=200).mean()

        # Calculate Pivot (Static Ceiling)
        # Shift(1) because we want the High of the PAST window, not including today
        df["Pivot_Ref"] = df["High"].rolling(window=self.pivot_lookback).max().shift(1)

        # OPTIMIZATION: Vectorized "Days Since Pivot"
        # We calculate how many days ago the max high occurred
        high_values = df["High"].values
        shape = (high_values.shape[0] - self.pivot_lookback + 1, self.pivot_lookback)
        strides = (high_values.strides[0], high_values.strides[0])
        windows = np.lib.stride_tricks.as_strided(
            high_values, shape=shape, strides=strides
        )

        # Argmax gives index 0..119. We want "Days Ago" (119..0)
        days_ago = self.pivot_lookback - 1 - np.argmax(windows, axis=1)

        # Pad and align with DataFrame
        pad = np.full(self.pivot_lookback - 1, np.nan)
        pivot_days_col = np.concatenate((pad, days_ago))
        df["Pivot_Days_Ago"] = pd.Series(pivot_days_col, index=df.index).shift(1)

        X_list, y_list, meta_list = [], [], []

        # 2. MINING LOOP (Event Driven)
        start_idx = max(250, self.input_window, self.pivot_lookback)

        for i in range(start_idx, len(df) - self.future_window):
            pivot = df["Pivot_Ref"].iloc[i]

            # Sanity Checks
            if pd.isna(pivot) or pivot == 0:
                continue

            today_high = df["High"].iloc[i]
            today_prev_high = df["High"].iloc[i - 1]

            # --- EVENT: BREAKOUT TRIGGER ---
            # Today crossed Pivot AND Yesterday was below Pivot
            if today_high >= pivot and today_prev_high < pivot:

                # --- FILTERS (The Funnel) ---

                # 1. Pivot Freshness (Don't buy a pivot made yesterday)
                if df["Pivot_Days_Ago"].iloc[i] < 5:
                    continue

                # 2. Trend Alignment (Stage 2)
                sma50 = df["SMA_50"].iloc[i]
                sma150 = df["SMA_150"].iloc[i]
                sma200 = df["SMA_200"].iloc[i]

                # Strict Order: Pivot > 50 > 150 > 200
                if not (pivot > sma50 and sma50 > sma150 and sma150 > sma200):
                    continue

                # 3. Volatility Contraction (VCP)
                v_short = df["Vol_Short"].iloc[i - 1]
                v_long = df["Vol_Long"].iloc[i - 1]
                if v_short > v_long * self.vol_ratio:
                    continue

                # 4. Proximity to Highs (Not a bottom play)
                year_high = df["High"].iloc[i - 250 : i].max()
                if pivot < year_high * 0.75:
                    continue

                # 5. Not Extended from Lows
                year_low = df["Low"].iloc[i - 250 : i].min()
                if pivot < year_low * 1.3:
                    continue

                # --- EXTRACTION & NORMALIZATION ---

                # Slice the sequence (History + Today)
                seq = df.iloc[i - self.input_window + 1 : i + 1].copy()
                norm_seq = np.zeros((self.input_window, 5))

                # Vectorized Normalization (Price - Pivot) / Pivot
                cols = ["Open", "High", "Low", "Close"]
                norm_seq[:, :4] = (seq[cols].values - pivot) / pivot

                # MASKING THE LAST CANDLE (Simulation of Entry)
                last_open = seq["Open"].iloc[-1]

                if last_open > pivot:
                    # CASE A: GAP UP ENTRY
                    gap_pct = (last_open - pivot) / pivot

                    # If gap is huge (> 5%), skip it (Too risky)
                    if gap_pct > self.max_gap:
                        continue

                    # If gap is okay, we enter at Open.
                    # Set the whole candle to the gap price (Flat line)
                    # This tells the model: "We are entering 2% higher than pivot"
                    norm_seq[-1, :] = gap_pct
                else:
                    # CASE B: STANDARD BREAKOUT ENTRY
                    # We enter exactly at Pivot (0.0)
                    # Cap High and Close at 0.0
                    norm_seq[-1, 1] = 0.0
                    norm_seq[-1, 3] = 0.0
                    # Ensure Open/Low are not physically impossible (e.g. > 0.0)
                    norm_seq[-1, 0] = min(norm_seq[-1, 0], 0.0)
                    norm_seq[-1, 2] = min(norm_seq[-1, 2], 0.0)

                # Volume Normalization
                v_max = seq["Volume"].max()
                norm_seq[:, 4] = seq["Volume"] / v_max if v_max > 0 else 0

                # --- LABELING ---
                label = 0
                future = df.iloc[i + 1 : i + 1 + self.future_window]
                take_profit = pivot * (1 + self.target)
                stop_loss = pivot * (1 + self.stop)

                for _, row in future.iterrows():
                    if row["Low"] <= stop_loss:
                        label = 0
                        break
                    if row["High"] >= take_profit:
                        label = 1
                        break

                X_list.append(norm_seq)
                y_list.append(label)
                meta_list.append(
                    {
                        "ticker": ticker_name,
                        "date": str(df.index[i].date()),
                        "pivot": round(pivot, 2),
                        "label": label,
                        "gap_pct": (
                            round((last_open - pivot) / pivot, 4)
                            if last_open > pivot
                            else 0
                        ),
                    }
                )

        return np.array(X_list), np.array(y_list), meta_list
