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
import time
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
    train_split_ratio: Optional[float] = None,
    device_name: str = "cpu",
    output_weights: str = "weights.pt",
    output_static: str = "static.pt",
    static_out: Optional[str] = None,
    train_end_year: int = 2018,
    val_end_year: int = 2020,
    seed: int = 42,
) -> MultiverseOfMonsoonModel:
    """Train MultiverseOfMonsoonModel using normalised-delta loss.

    Key improvements over the initial run:
    1. Block 1 normalises physical inputs (zero-mean, unit-std per channel) so that
       msl (~101325 Pa) cannot overpower tp (~2.5 mm/day) in the first convolution.
    2. Loss is computed on NORMALISED DELTAS (next_state - current_state) / std, 
       forcing the model to learn genuine atmospheric movement rather than the 
       persistence shortcut (delta ~ 0).
    3. A small absolute-state term (5%) prevents unbounded drift over rollouts.
    4. Strict chronological 3-way split: Training (2005-2018), Validation (2019-2020),
       and Holdout (2021) which is completely isolated from training and checkpoint selection.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    output_static = output_static or "static.pt"
    device = torch.device(device_name)
    print(f"--- Training Multiverse of Monsoon (normalised-delta mode) on {device} ---")
    print(f"Dataset: {nc_path}  |  Stats: {stats_path}")

    with h5py.File(nc_path, "r") as f:
        total_days = f["data"].shape[0]
        times = f["time"][:]
        static_geo = torch.from_numpy(f["static"][:]).float().to(device)

    torch.save(static_geo, output_static)
    np.save("static.npy", static_geo.cpu().numpy())
    print(f"Saved static geography to {output_static}")

    # Chronological 3-way split using NetCDF time coordinates (days since 2005-01-01)
    base_date = dt.date(2005, 1, 1)
    all_dates = [base_date + dt.timedelta(days=int(t)) for t in times]

    if train_split_ratio is not None:
        train_end_idx = int(total_days * train_split_ratio)
        val_end_idx = total_days
    else:
        train_end_idx = next((i for i, d in enumerate(all_dates) if d.year > train_end_year), total_days)
        val_end_idx   = next((i for i, d in enumerate(all_dates) if d.year > val_end_year), total_days)

    train_days   = train_end_idx
    val_days     = val_end_idx - train_end_idx
    holdout_days = total_days - val_end_idx

    print(
        f"Chronological 3-way split:\n"
        f"  Training:   indices 0 to {train_end_idx - 1} ({train_days} days, {all_dates[0]} to {all_dates[train_end_idx - 1]})\n"
        f"  Validation: indices {train_end_idx} to {val_end_idx - 1} ({val_days} days, {all_dates[train_end_idx]} to {all_dates[val_end_idx - 1]})\n"
        f"  Holdout:    indices {val_end_idx} to {total_days - 1} ({holdout_days} days, {all_dates[val_end_idx]} to {all_dates[-1]}) [EXCLUDED FROM TRAINING]"
    )

    # Slice raw timeline before generating windows inside ClimateDataset
    train_dataset = ClimateDataset(nc_path=Path(nc_path), stats_path=Path(stats_path),
                                    start_idx=0, end_idx=train_end_idx)
    val_dataset   = ClimateDataset(nc_path=Path(nc_path), stats_path=Path(stats_path),
                                    start_idx=train_end_idx, end_idx=val_end_idx)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    with open(stats_path, "r") as f:
        stats = json.load(f)
    channel_means  = stats["mean"]
    channel_scales = stats["std"]

    # Pass both means and stds so Block 1 can standardise inputs internally
    model = MultiverseOfMonsoonModel(
        channel_scales=channel_scales,
        channel_means=channel_means,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    std_t   = torch.tensor(channel_scales, device=device, dtype=torch.float32)
    inv_var = (1.0 / (std_t ** 2)).view(1, NC, 1, 1)

    best_val_loss = float("inf")
    t0_train = time.perf_counter()

    for epoch in range(1, epochs + 1):
        t0_epoch = time.perf_counter()
        model.train()
        total_train_loss = 0.0
        n_batches = 0

        for windows, targets, calendars in train_loader:
            windows   = windows.to(device)
            targets   = targets.to(device)
            calendars = calendars.to(device)

            optimizer.zero_grad()

            current    = windows[:, -1]
            true_delta = targets - current
            pred_next  = model(windows, static_geo, calendars)
            pred_delta = pred_next - current

            # Normalise deltas by per-channel std: forces learning of dynamics
            norm_true = true_delta / (std_t.view(1, NC, 1, 1) + 1e-6)
            norm_pred = pred_delta / (std_t.view(1, NC, 1, 1) + 1e-6)
            loss_delta = ((norm_pred - norm_true) ** 2).mean()

            # Small absolute-state term (5%) prevents drift
            loss_state = (inv_var * (pred_next - targets) ** 2).mean()
            loss = 0.95 * loss_delta + 0.05 * loss_state

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_train_loss = total_train_loss / max(1, n_batches)

        model.eval()
        total_val_loss = 0.0
        n_val = 0
        channel_rmse = torch.zeros(NC, device=device)

        with torch.no_grad():
            for windows, targets, calendars in val_loader:
                windows   = windows.to(device)
                targets   = targets.to(device)
                calendars = calendars.to(device)

                current    = windows[:, -1]
                true_delta = targets - current
                pred_next  = model(windows, static_geo, calendars)
                pred_delta = pred_next - current

                norm_true = true_delta / (std_t.view(1, NC, 1, 1) + 1e-6)
                norm_pred = pred_delta / (std_t.view(1, NC, 1, 1) + 1e-6)
                loss_delta = ((norm_pred - norm_true) ** 2).mean()
                loss_state = (inv_var * (pred_next - targets) ** 2).mean()
                loss = 0.95 * loss_delta + 0.05 * loss_state

                per_ch = torch.sqrt(((pred_next - targets) ** 2).mean(dim=(0, 2, 3)))
                channel_rmse += per_ch
                total_val_loss += loss.item()
                n_val += 1

        avg_val_loss = total_val_loss / max(1, n_val)
        avg_rmse = (channel_rmse / max(1, n_val)).cpu().tolist()

        current_lr = optimizer.param_groups[0]["lr"]
        epoch_sec = time.perf_counter() - t0_epoch
        assert math.isfinite(avg_train_loss), "NaN/Inf in training loss"
        assert math.isfinite(avg_val_loss), "NaN/Inf in validation loss"

        print(
            f"Epoch {epoch:2d}/{epochs:2d} ({epoch_sec:.1f}s) | "
            f"LR: {current_lr:.6f} | "
            f"Train Loss: {avg_train_loss:.5f} | "
            f"Val Loss: {avg_val_loss:.5f} | "
            f"t2m RMSE: {avg_rmse[0]:.3f} K | "
            f"tp RMSE: {avg_rmse[1]:.3f} mm | "
            f"msl RMSE: {avg_rmse[4]:.1f} Pa"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), output_weights)
            print(f"  --> Saved new best checkpoint to {output_weights}")

    total_sec = time.perf_counter() - t0_train
    print(f"--- Training Complete in {total_sec:.1f}s. Best val loss: {best_val_loss:.5f} ---")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Multiverse of Monsoon model")
    parser.add_argument("--nc",         type=str,   default="hackdata/india_climate_train.nc")
    parser.add_argument("--stats",      type=str,   default="hackdata/stats.json")
    parser.add_argument("--epochs",     type=int,   default=5)
    parser.add_argument("--batch-size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--out",        type=str,   default="weights.pt")
    args = parser.parse_args()

    train_model(
        nc_path=args.nc,
        stats_path=args.stats,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_weights=args.out,
    )
