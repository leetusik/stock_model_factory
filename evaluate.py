import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from src.model import VCPNet, VCPDataset
from pathlib import Path

# --- Configuration ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = "models/vcp_cnn_best_1.pth"
DATA_DIR = Path("02_data_processed_2")


def evaluate():
    print(f"--- Evaluating Model: {MODEL_PATH} ---")
    print(f"--- Device: {DEVICE} ---")

    # 1. Load Data (The Test Subject)
    # We load the FULL dataset here to simulate a lifetime of trading
    try:
        dataset = VCPDataset(DATA_DIR / "train_X.npy", DATA_DIR / "train_y.npy")
        loader = DataLoader(dataset, batch_size=1024, shuffle=False)
        print(f"Loaded {len(dataset)} samples for evaluation.")
    except FileNotFoundError:
        print("Error: Data not found. Run build_dataset.py first.")
        return

    # 2. Load Model (The Brain)
    model = VCPNet().to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print("Model weights loaded successfully.")
    except FileNotFoundError:
        print(f"Error: {MODEL_PATH} not found. Run train.py first.")
        return

    # 3. Inference (The Exam)
    # We turn off 'Training Mode' (no dropout, no gradient calc)
    model.eval()

    all_probs = []
    all_targets = []

    print("Running inference...", end=" ")
    with torch.no_grad():  # Saves memory, speeds up calculation
        for X, y in loader:
            X = X.to(DEVICE)
            # Model outputs raw log-odds, Sigmoid converts to 0.0~1.0 probability
            probs = model(X).cpu().numpy()
            all_probs.extend(probs)
            all_targets.extend(y.numpy())
    print("Done.\n")

    all_probs = np.array(all_probs).flatten()
    all_targets = np.array(all_targets).flatten()

    # 4. Threshold Analysis (The Business Logic)
    # We simulate trading at different confidence levels

    print(
        f"{'Threshold':<10} | {'Trades':<8} | {'Win Rate':<10} | {'Exp. Value':<12} | {'Status'}"
    )
    print("-" * 65)

    thresholds = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]

    for th in thresholds:
        # DECISION: Buy if Probability > Threshold
        predicted_buys = (all_probs > th).astype(int)

        # LOGIC:
        # TP (True Positive) = We bought AND it went up (Win)
        # FP (False Positive) = We bought AND it crashed (Loss)
        tp = np.sum((predicted_buys == 1) & (all_targets == 1))
        fp = np.sum((predicted_buys == 1) & (all_targets == 0))

        total_trades = tp + fp

        if total_trades > 0:
            win_rate = tp / total_trades

            # EXPECTED VALUE (EV) CALCULATION
            # Assumption: Target Win = +20%, Stop Loss = -10%
            ev = (win_rate * 0.20) - ((1 - win_rate) * 0.10)

            status = "✅ PROFIT" if ev > 0 else "❌ LOSS"
            print(
                f"{th:<10.2f} | {total_trades:<8} | {win_rate*100:6.2f}%    | {ev*100:6.2f}%      | {status}"
            )
        else:
            print(f"{th:<10.2f} | 0        | N/A        | N/A          | NO TRADES")

    print("-" * 65)
    print(
        f"Baseline Win Rate (Random Guessing): {(np.sum(all_targets)/len(all_targets))*100:.2f}%"
    )


if __name__ == "__main__":
    evaluate()
