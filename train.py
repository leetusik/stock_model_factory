import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler, random_split
from pathlib import Path
from tqdm import tqdm
import numpy as np

# Import from your src
from src.model import VCPNet, VCPDataset

# Configuration
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# If you are on Mac M3, use "mps"
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")


def train():
    print(f"Using Device: {DEVICE}")

    # 1. Load Data
    data_dir = Path("02_data_processed_2")
    full_dataset = VCPDataset(data_dir / "train_X.npy", data_dir / "train_y.npy")

    # 2. Split Train (80%) / Val (20%)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    print(f"Data Loaded. Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # 3. Handle Imbalance (Crucial Step)
    # We need to calculate weights ONLY for the training set
    # Access indices of the subset to get labels
    y_train_indices = train_dataset.indices
    y_train = full_dataset.y[y_train_indices].numpy().flatten()

    class_counts = np.bincount(y_train.astype(int))
    # Weight = Total / (Count * Number of Classes)
    # This gives higher weight to the minority class (1)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train.astype(int)]

    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True,
    )

    # 4. Data Loaders
    # Note: Shuffle must be False when using Sampler
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    # Validation doesn't need rebalancing, we want to see real-world accuracy
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # 5. Initialize Model
    model = VCPNet().to(DEVICE)
    criterion = nn.BCELoss()  # Binary Cross Entropy
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5
    )  # L2 Regularization

    # 6. Training Loop
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        # --- TRAIN ---
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        loop = tqdm(train_loader, leave=False)
        for X, y in loop:
            X, y = X.to(DEVICE), y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calc Accuracy
            predicted = (outputs > 0.5).float()
            total += y.size(0)
            correct += (predicted == y).sum().item()

            loop.set_description(f"Epoch [{epoch+1}/{EPOCHS}]")
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct / total

        # --- VALIDATE ---
        model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item()

                predicted = (outputs > 0.5).float()
                total += y.size(0)
                correct += (predicted == y).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total

        print(
            f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f} (Acc: {train_acc:.1f}%) | Val Loss={avg_val_loss:.4f} (Acc: {val_acc:.1f}%)"
        )

        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            Path("models").mkdir(exist_ok=True)
            torch.save(model.state_dict(), "models/vcp_cnn_best_1.pth")
            print("  >>> Model Saved!")


if __name__ == "__main__":
    train()
