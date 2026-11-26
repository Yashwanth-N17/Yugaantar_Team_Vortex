"""Training pipeline for Multiverse of Monsoon architecture using hackdata/india_climate_train.nc.

Strictly adheres to:
- Using ONLY hackdata/india_climate_train.nc and hackdata/stats.json
- Chronological train/validation splits (no temporal leakage)
- 3-day sliding input windows [B, 3, 6, 32, 36] and next-day target [B, 6, 32, 36]
- Static geography [2, 32, 36] and calendar [B, 2]
- Residual learning: next_state = current_state + delta
- Saving trained model weights to weights.pt and static geography to static.pt
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from monsoon_r1.contract import CHANNELS, NC, NLAT, NLON, T_IN
from monsoon_r1.models.multiverse_model import MultiverseOfMonsoonModel


class ClimateDataset(Dataset):
    """PyTorch Dataset loading daily climate data from NetCDF/HDF5."""

    def __init__(
        self,
        nc_path: Path,
        stats_path: Path,
        start_idx: int = 0,
        end_idx: Optional[int] = None,
        seq_len: int = T_IN,
        start_date: dt.date = dt.date(2005, 1, 1),
    ) -> None:
        super().__init__()
        self.nc_path = Path(nc_path)
        self.seq_len = seq_len
        self.start_date = start_date

        with h5py.File(self.nc_path, "r") as f:
            total_days = f["data"].shape[0]
            if end_idx is None or end_idx > total_days:
                end_idx = total_days
            self.start_idx = start_idx
            self.end_idx = end_idx

            # Load slices into memory for fast training
            # shape: [N_days, 6, 32, 36]
            self.data = torch.from_numpy(f["data"][start_idx:end_idx]).float()
            # shape: [2, 32, 36]
            self.static = torch.from_numpy(f["static"][:]).float()

        # Load normalization constants
        with open(stats_path, "r") as f:
            stats = json.load(f)
        self.means = torch.tensor(stats["mean"], dtype=torch.float32).view(1, NC, 1, 1)
        self.stds = torch.tensor(stats["std"], dtype=torch.float32).view(1, NC, 1, 1)

        # Number of sliding window samples
        self.num_samples = len(self.data) - self.seq_len

    def __len__(self) -> int:
        return max(0, self.num_samples)

    def get_doy_sincos(self, day_index: int) -> torch.Tensor:
        """Compute [sin, cos] calendar coordinates for a given day index."""
        current_date = self.start_date + dt.timedelta(days=self.start_idx + day_index)
        # Day of year (1-based: 1 to 366)
        doy = (current_date - dt.date(current_date.year, 1, 1)).days + 1
        angle = 2.0 * math.pi * float(doy) / 365.0
        return torch.tensor([math.sin(angle), math.cos(angle)], dtype=torch.float32)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (input_window [T, 6, 32, 36], target_next_state [6, 32, 36], calendar [2])."""
        # Input sequence: T consecutive daily states
        window = self.data[idx : idx + self.seq_len]  # [3, 6, 32, 36]
        # Target: next day state
        target = self.data[idx + self.seq_len]        # [6, 32, 36]
        # Calendar doy_sincos for the LAST day of the input window (idx + seq_len - 1)
        calendar = self.get_doy_sincos(idx + self.seq_len - 1)  # [2]

        return window, target, calendar


def train_model(
    nc_path: str = "hackdata/india_climate_train.nc",
    stats_path: str = "hackdata/stats.json",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-3,
    train_split_ratio: float = 0.85,
    device_name: str = "cpu",
    output_weights: str = "weights.pt",
    output_static: str = "static.pt",
) -> MultiverseOfMonsoonModel:
    """Train the MultiverseOfMonsoonModel on the provided climate dataset."""
    device = torch.device(device_name)
    print(f"--- Training Multiverse of Monsoon on {device} ---")
    print(f"Dataset path: {nc_path}")
    print(f"Stats path:   {stats_path}")

    # Load dataset dimensions
    with h5py.File(nc_path, "r") as f:
        total_days = f["data"].shape[0]
        static_geo = torch.from_numpy(f["static"][:]).float().to(device)

    # Save static geography tensor for serve.py bundle
    torch.save(static_geo, output_static)
    np.save("static.npy", static_geo.cpu().numpy())
    print(f"Saved bundled static geography to {output_static} and static.npy")

    train_days = int(total_days * train_split_ratio)
    val_days = total_days - train_days
    print(f"Chronological split: {train_days} training days, {val_days} validation days")

    train_dataset = ClimateDataset(
        nc_path=Path(nc_path),
        stats_path=Path(stats_path),
        start_idx=0,
        end_idx=train_days,
    )
    val_dataset = ClimateDataset(
        nc_path=Path(nc_path),
        stats_path=Path(stats_path),
        start_idx=train_days,
        end_idx=total_days,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Instantiate model with channel standard deviation scales from stats.json
    with open(stats_path, "r") as f:
        stats = json.load(f)
    channel_scales = stats["std"]

    model = MultiverseOfMonsoonModel(channel_scales=channel_scales).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Weighted loss: normalize channels by 1 / (std^2) so all variables contribute equally
    channel_weights = (1.0 / (torch.tensor(channel_scales, device=device) ** 2)).view(1, NC, 1, 1)

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0.0
        num_train_batches = 0

        for windows, targets, calendars in train_loader:
            windows = windows.to(device)      # [B, 3, 6, 32, 36]
            targets = targets.to(device)      # [B, 6, 32, 36]
            calendars = calendars.to(device)  # [B, 2]

            optimizer.zero_grad()

            # Predict next climate state through complete 6-block pipeline
            preds = model(windows, static_geo, calendars)

            # Standardized MSE loss
            diff = (preds - targets) ** 2
            loss = (diff * channel_weights).mean()

            # Autograd backward pass
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_train_loss += loss.item()
            num_train_batches += 1

        scheduler.step()
        avg_train_loss = total_train_loss / max(1, num_train_batches)

        # Validation loop
        model.eval()
        total_val_loss = 0.0
        num_val_batches = 0
        channel_errors = torch.zeros(NC, device=device)

        with torch.no_grad():
            for windows, targets, calendars in val_loader:
                windows = windows.to(device)
                targets = targets.to(device)
                calendars = calendars.to(device)

                preds = model(windows, static_geo, calendars)
                diff = (preds - targets) ** 2
                loss = (diff * channel_weights).mean()

                # Track unnormalized RMSE per channel
                channel_rmse = torch.sqrt(diff.mean(dim=(0, 2, 3)))
                channel_errors += channel_rmse

                total_val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = total_val_loss / max(1, num_val_batches)
        avg_channel_rmse = (channel_errors / max(1, num_val_batches)).cpu().tolist()

        print(
            f"Epoch {epoch:2d}/{epochs:2d} | "
            f"Train Loss: {avg_train_loss:.5f} | "
            f"Val Loss: {avg_val_loss:.5f} | "
            f"t2m RMSE: {avg_channel_rmse[0]:.2f}K | "
            f"tp RMSE: {avg_channel_rmse[1]:.2f}mm"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_weights)
            print(f"  --> Saved new best checkpoint to {output_weights}")

    print("--- Training Complete ---")
    print(f"Final model weights saved to {output_weights}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multiverse of Monsoon model")
    parser.add_argument("--nc", type=str, default="hackdata/india_climate_train.nc")
    parser.add_argument("--stats", type=str, default="hackdata/stats.json")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", type=str, default="weights.pt")
    args = parser.parse_args()

    train_model(
        nc_path=args.nc,
        stats_path=args.stats,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_weights=args.out,
    )
