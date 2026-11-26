"""Validation and evaluation test suite for the trained Multiverse of Monsoon model.

Tests:
- Verification of trained checkpoint (weights.pt) and bundled static geography (static.pt).
- 1-step climate state prediction accuracy and physical validity.
- 10-step autoregressive forecast rollout.
- Full compliance with the Round 1 physical validity gate:
  - Shape [10, 6, 32, 36]
  - No NaN or Inf
  - Non-negative rain (tp >= 0.0)
  - Temperature in [-40, +60] C (233.15 - 333.15 K)
- Counterfactual perturbation response (non-zero localized shock response).
- Null perturbation invariance.
"""

from __future__ import annotations

import json
from pathlib import Path
import unittest
import h5py
import numpy as np
import torch

from monsoon_r1.contract import (
    CHANNELS,
    NC,
    NLAT,
    NLON,
    SCORE_STEPS,
    T_IN,
    doy_sincos,
)
from monsoon_r1.models.multiverse_model import MultiverseOfMonsoonModel
from monsoon_r1.scoring import NULL_TOL, validity


class TestTrainedModelEvaluation(unittest.TestCase):
    """Test suite evaluating the trained Multiverse of Monsoon model against real climate data."""

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        self.weights_path = Path("weights.pt")
        self.static_path = Path("static.pt")
        self.nc_path = Path("hackdata/india_climate_train.nc")
        self.stats_path = Path("hackdata/stats.json")

        self.assertTrue(self.weights_path.exists(), "weights.pt checkpoint must exist")
        self.assertTrue(self.static_path.exists(), "static.pt must exist")
        self.assertTrue(self.nc_path.exists(), "hackdata/india_climate_train.nc must exist")

        with open(self.stats_path, "r") as f:
            stats = json.load(f)
        self.channel_scales = stats["std"]
        self.channel_means = stats["mean"]

        # Load model and weights — must pass channel_means so Block1 buffers match the checkpoint
        self.model = MultiverseOfMonsoonModel(
            channel_scales=self.channel_scales,
            channel_means=self.channel_means,
        ).to(self.device)
        state_dict = torch.load(self.weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # Load static geography
        self.static_geo = torch.load(self.static_path, map_location=self.device)

    def test_01_checkpoint_and_parameters_loaded(self) -> None:
        """Verify weights loaded correctly into all blocks and parameters are finite."""
        self.assertEqual(self.static_geo.shape, (2, NLAT, NLON))
        self.assertTrue(torch.isfinite(self.static_geo).all().item())

        for name, param in self.model.named_parameters():
            self.assertTrue(
                torch.isfinite(param).all().item(),
                f"Parameter {name} contains non-finite values",
            )

    def test_02_real_data_one_step_prediction(self) -> None:
        """Evaluate 1-step prediction on real held-out climate sequence from hackdata."""
        with h5py.File(self.nc_path, "r") as f:
            # Pick a sample window from the validation period (e.g. index 5500)
            sample_idx = 5500
            window_np = f["data"][sample_idx : sample_idx + T_IN]  # [3, 6, 32, 36]
            target_np = f["data"][sample_idx + T_IN]               # [6, 32, 36]

        window = torch.from_numpy(window_np).unsqueeze(0).to(self.device)  # [1, 3, 6, 32, 36]
        target = torch.from_numpy(target_np).unsqueeze(0).to(self.device)  # [1, 6, 32, 36]
        cal = torch.tensor(doy_sincos(180), device=self.device).unsqueeze(0)  # [1, 2]

        with torch.no_grad():
            pred = self.model(window, self.static_geo, cal)

        self.assertEqual(pred.shape, (1, 6, NLAT, NLON))
        self.assertTrue(torch.isfinite(pred).all().item())

        # Physical checks
        self.assertTrue(torch.all(pred[:, 1] >= 0.0).item(), "Precipitation must not be negative")
        t2m = pred[:, 0]
        self.assertTrue(torch.all((t2m >= 233.15) & (t2m <= 333.15)).item(), "Temperature out of physical bounds")

        # Error check: 1-step RMSE must be well within reasonable limits
        t2m_rmse = torch.sqrt(((pred[:, 0] - target[:, 0]) ** 2).mean()).item()
        tp_rmse = torch.sqrt(((pred[:, 1] - target[:, 1]) ** 2).mean()).item()

        self.assertLess(t2m_rmse, 10.0, f"t2m RMSE {t2m_rmse:.2f}K exceeds acceptable threshold")
        self.assertLess(tp_rmse, 25.0, f"tp RMSE {tp_rmse:.2f}mm exceeds acceptable threshold")

    def test_03_ten_day_rollout_physical_validity_gate(self) -> None:
        """Verify that a 10-day autoregressive rollout on real data passes the validity gate."""
        with h5py.File(self.nc_path, "r") as f:
            sample_idx = 6000
            window_np = f["data"][sample_idx : sample_idx + T_IN]  # [3, 6, 32, 36]

        window = torch.from_numpy(window_np).to(self.device)
        cal = doy_sincos(200)

        with torch.no_grad():
            rollout = self.model.rollout(
                init_states=window,
                geography=self.static_geo,
                doy_sincos=cal,
                steps=SCORE_STEPS,
            )

        self.assertEqual(rollout.shape, (SCORE_STEPS, 6, NLAT, NLON))

        # Check against monsoon_r1.scoring.validity
        rollout_np = rollout.cpu().numpy()
        violations = validity(rollout_np, steps=SCORE_STEPS)
        self.assertEqual(violations, [], f"Validity gate failed with violations: {violations}")

    def test_04_counterfactual_shock_response(self) -> None:
        """Test response to a localized physical shock in Bay of Bengal on real data."""
        with h5py.File(self.nc_path, "r") as f:
            sample_idx = 5800
            window_np = f["data"][sample_idx : sample_idx + T_IN]

        window = torch.from_numpy(window_np).to(self.device)
        cal = doy_sincos(190)

        with torch.no_grad():
            baseline = self.model.rollout(
                init_states=window,
                geography=self.static_geo,
                doy_sincos=cal,
                steps=SCORE_STEPS,
            )

            # Apply warm ocean shock: +3.0 K temperature in Bay of Bengal
            shock = torch.zeros(6, NLAT, NLON, device=self.device)
            shock[0, 2:14, 17:29] = 3.0

            perturbed = self.model.rollout(
                init_states=window,
                geography=self.static_geo,
                doy_sincos=cal,
                steps=SCORE_STEPS,
                perturbation=shock,
            )

        # Check non-zero response
        diff = (perturbed - baseline).abs()
        max_diff = diff.max().item()
        self.assertGreater(max_diff, 1e-4, "Model did not respond to perturbation shock")

        # Ensure perturbed rollout also passes validity
        violations = validity(perturbed.cpu().numpy(), steps=SCORE_STEPS)
        self.assertEqual(violations, [], f"Perturbed rollout validity failed: {violations}")

    def test_05_null_perturbation_invariance(self) -> None:
        """Test that zero perturbation shock yields deterministic invariant rollout."""
        with h5py.File(self.nc_path, "r") as f:
            sample_idx = 5700
            window_np = f["data"][sample_idx : sample_idx + T_IN]

        window = torch.from_numpy(window_np).to(self.device)
        cal = doy_sincos(160)

        with torch.no_grad():
            unperturbed = self.model.rollout(
                init_states=window,
                geography=self.static_geo,
                doy_sincos=cal,
                steps=SCORE_STEPS,
                perturbation=None,
            )
            null_shock = torch.zeros(6, NLAT, NLON, device=self.device)
            perturbed_null = self.model.rollout(
                init_states=window,
                geography=self.static_geo,
                doy_sincos=cal,
                steps=SCORE_STEPS,
                perturbation=null_shock,
            )

        max_err = (perturbed_null - unperturbed).abs().max().item()
        self.assertLess(max_err, 1e-5, f"Null shock divergence was {max_err}")


if __name__ == "__main__":
    unittest.main()
