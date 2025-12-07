import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np


# ==========================================
# 1. THE DATASET LOADER
# ==========================================
class VCPDataset(Dataset):
    def __init__(self, X_path, y_path):
        # Load the saved Numpy arrays
        self.X = np.load(X_path).astype(np.float32)
        self.y = np.load(y_path).astype(np.float32)

        # Convert to PyTorch Tensors
        # Shape: [N, 120, 5] -> PyTorch likes [N, Channels, Length] -> [N, 5, 120]
        self.X = torch.from_numpy(self.X).permute(0, 2, 1)
        self.y = torch.from_numpy(self.y).unsqueeze(1)  # [N, 1]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==========================================
# 2. THE MODEL ARCHITECTURE (Multi-Scale CNN)
# ==========================================
class VCPNet(nn.Module):
    def __init__(self):
        super(VCPNet, self).__init__()

        # We use 3 parallel branches to look at different time scales

        # Branch A: Micro-Structure (Last 3 days) - The "Handle"
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Branch B: Medium-Structure (Last 15 days) - The "Pullback"
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=16, kernel_size=15, padding=7),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Branch C: Macro-Structure (Trend)
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels=5, out_channels=16, kernel_size=30, padding=15),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        # Combined Feature Processing
        # After pooling, length 120 becomes 60.
        # 3 branches * 16 channels = 48 channels.
        self.flatten_dim = 48 * 60

        self.fc_layers = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),  # Crucial for your dataset size (Prevents overfitting)
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid(),  # Output probability 0.0 ~ 1.0
        )

    def forward(self, x):
        # x shape: [Batch, 5, 120]

        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)

        # Concatenate features from all branches
        # Shape becomes [Batch, 48, 60]
        combined = torch.cat([out1, out2, out3], dim=1)

        # Flatten
        combined = combined.view(combined.size(0), -1)

        # Solve
        return self.fc_layers(combined)
