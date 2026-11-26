"""Multiverse of Monsoon — Model Training Entry Point.

This script trains the 6-block Multiverse of Monsoon climate forecasting model
strictly using the dataset in hackdata/ (india_climate_train.nc and stats.json).

Usage:
    python train.py --epochs 3 --batch-size 32 --lr 0.001 --out weights.pt
"""

import argparse
import sys
from pathlib import Path

from monsoon_r1.train import train_model


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train the Multiverse of Monsoon 6-Block Climate Simulation Model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--nc",
        type=str,
        default="hackdata/india_climate_train.nc",
        help="Path to training NetCDF dataset",
    )
    parser.add_argument(
        "--stats",
        type=str,
        default="hackdata/stats.json",
        help="Path to channel normalization statistics JSON",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="AdamW learning rate",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="weights.pt",
        help="Output path for trained model checkpoint",
    )
    parser.add_argument(
        "--static-pt",
        type=str,
        default="static.pt",
        help="Output path for extracted static geography tensor",
    )
    args = parser.parse_args()

    train_model(
        nc_path=args.nc,
        stats_path=args.stats,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        output_weights=args.out,
        output_static=args.static_pt,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
