import torch
import numpy as np
from torch.utils.data import DataLoader, Subset
from pathlib import Path

# Import your model architecture
from src.model import VCPNet, VCPDataset

# Configuration
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = "models/vcp_cnn_best_2.pth"
DATA_DIR = Path("02_data_processed_2")


def evaluate():
    print(f"--- STRICT EVALUATION (UNSEEN DATA ONLY) ---")

    # 1. Load Full Data
    try:
        full_dataset = VCPDataset(DATA_DIR / "train_X.npy", DATA_DIR / "train_y.npy")
    except FileNotFoundError:
        print("Error: Data not found.")
        return

    # 2. ISOLATE THE VALIDATION SET (Must match train.py logic exactly)
    total_len = len(full_dataset)
    train_size = int(0.8 * total_len)

    # We only want indices from 80% to 100%
    val_indices = list(range(train_size, total_len))

    # Create the Subset
    test_dataset = Subset(full_dataset, val_indices)
    loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    print(f"Total History: {total_len} samples")
    print(f"Evaluating ONLY on Validation Set (Last 20%): {len(test_dataset)} samples")

    # 3. Load Model
    model = VCPNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    all_probs = []
    all_targets = []

    print("Running inference...", end=" ")
    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            probs = model(X).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(y.numpy())
    print("Done.\n")

    all_probs = np.array(all_probs).flatten()
    all_targets = np.array(all_targets).flatten()

    # 4. Analysis
    print(
        f"{'Threshold':<10} | {'Trades':<8} | {'Win Rate':<10} | {'Exp. Value':<12} | {'Status'}"
    )
    print("-" * 65)

    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90]

    for th in thresholds:
        predicted_buys = (all_probs > th).astype(int)

        tp = np.sum((predicted_buys == 1) & (all_targets == 1))
        fp = np.sum((predicted_buys == 1) & (all_targets == 0))
        total_trades = tp + fp

        if total_trades > 0:
            win_rate = tp / total_trades
            # EV Calculation: Win (+20%) vs Loss (-10%)
            ev = (win_rate * 0.20) - ((1 - win_rate) * 0.10)
            status = "✅ PROFIT" if ev > 0 else "❌ LOSS"
            print(
                f"{th:<10.2f} | {total_trades:<8} | {win_rate*100:6.2f}%    | {ev*100:6.2f}%      | {status}"
            )
        else:
            print(f"{th:<10.2f} | 0        | N/A        | N/A          | NO TRADES")

    baseline = (np.sum(all_targets) / len(all_targets)) * 100
    print("-" * 65)
    print(f"Baseline Win Rate (In this period): {baseline:.2f}%")


if __name__ == "__main__":
    evaluate()
