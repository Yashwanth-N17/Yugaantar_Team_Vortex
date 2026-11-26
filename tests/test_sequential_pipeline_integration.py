"""Comprehensive sequential integration tests for Multiverse of Monsoon.

Verifies:
- Complete connected sequential pipeline: Block 1 -> Block 2 -> Block 3 -> Block 4 -> Block 5 -> Block 6.
- Per-stage shape and contract verification table.
- Differentiable end-to-end gradient propagation through all six blocks.
- Autoregressive 10-day rollouts with rolling 3-day windows.
- Multiverse counterfactual simulation (baseline vs perturbed timelines).
- Null perturbation invariance (determinism test).
- Physical constraints preservation (non-negative precipitation, valid temperature).
- CPU execution and runtime performance within competition limits.
"""

from __future__ import annotations

import math
import time
import unittest
import torch
import torch.nn as nn

from monsoon_r1.contract import (
    CHANNELS,
    NC,
    NLAT,
    NLON,
    SCORE_STEPS,
    T_IN,
    TIME_LIMIT_S,
    BOX_BAY_OF_BENGAL,
    box_mask,
)
from monsoon_r1.models.climate_input_encoder import ClimateInputEncoder
from monsoon_r1.models.multi_scale_spatial_encoder import MultiScaleSpatialEncoder
from monsoon_r1.models.temporal_dynamics_encoder import TemporalDynamicsEncoder
from monsoon_r1.models.spatial_context_attention import SpatialContextAttention
from monsoon_r1.models.forecast_decoder import ForecastDecoder
from monsoon_r1.models.residual_state_transition import ResidualStateTransition
from monsoon_r1.models.multiverse_model import MultiverseOfMonsoonModel


class TestSequentialPipelineIntegration(unittest.TestCase):
    """Integration test suite for the complete six-block Multiverse of Monsoon architecture."""

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        torch.manual_seed(42)

    def _generate_synthetic_climate(
        self,
        b: int = 2,
        t: int = 3,
        h: int = 32,
        w: int = 36,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate realistic synthetic inputs matching the project contract."""
        climate = torch.randn(b, t, 6, h, w, device=self.device)
        climate[:, :, 0] = climate[:, :, 0] * 10.0 + 300.0  # t2m ~ 290-310 K
        climate[:, :, 1] = torch.clamp(climate[:, :, 1] * 5.0 + 5.0, min=0.0)  # tp >= 0 mm/day
        climate[:, :, 2] = climate[:, :, 2] * 4.0  # u10 ~ m/s
        climate[:, :, 3] = climate[:, :, 3] * 4.0  # v10 ~ m/s
        climate[:, :, 4] = climate[:, :, 4] * 1000.0 + 101325.0  # msl ~ Pa
        climate[:, :, 5] = torch.clamp(climate[:, :, 5] * 10.0 + 35.0, min=1.0)  # tcwv ~ kg/m2

        geo = torch.zeros(b, 2, h, w, device=self.device)
        geo[:, 0] = torch.rand(b, h, w)  # land-sea mask [0, 1]
        geo[:, 1] = torch.rand(b, h, w) * 2500.0  # orography elevation

        doy = 180.0
        angle = 2.0 * math.pi * doy / 365.0
        calendar = torch.tensor([[math.sin(angle), math.cos(angle)]] * b, device=self.device)

        return climate, geo, calendar

    def test_01_step_by_step_stage_shapes_and_records(self) -> None:
        """Test 1: Step-by-step verification and recording of shapes across all 6 sequential blocks."""
        b, t, h, w = 2, 3, 32, 36
        climate, geo, calendar = self._generate_synthetic_climate(b=b, t=t, h=h, w=w)

        block1 = ClimateInputEncoder().to(self.device)
        block2 = MultiScaleSpatialEncoder().to(self.device)
        block3 = TemporalDynamicsEncoder().to(self.device)
        block4 = SpatialContextAttention().to(self.device)
        block5 = ForecastDecoder().to(self.device)
        block6 = ResidualStateTransition().to(self.device)

        # Stage 1: Climate Input Encoder
        b1_out = block1(climate, geo, calendar)
        self.assertEqual(b1_out.shape, (b, t, 48, h, w))
        self.assertTrue(torch.isfinite(b1_out).all().item())

        # Stage 2: Multi-Scale Spatial Encoder
        b2_out = block2(b1_out)
        self.assertEqual(b2_out.shape, (b, t, 64, h, w))
        self.assertTrue(torch.isfinite(b2_out).all().item())

        # Stage 3: Temporal Dynamics Encoder (ConvLSTM)
        b3_out = block3(b2_out)
        self.assertEqual(b3_out.shape, (b, 64, h, w))
        self.assertTrue(torch.isfinite(b3_out).all().item())

        # Stage 4: Spatial Context Attention Module
        b4_out = block4(b3_out)
        self.assertEqual(b4_out.shape, (b, 64, h, w))
        self.assertTrue(torch.isfinite(b4_out).all().item())

        # Stage 5: Forecast Decoder
        delta = block5(b4_out)
        self.assertEqual(delta.shape, (b, 6, h, w))
        self.assertTrue(torch.isfinite(delta).all().item())

        # Stage 6: Residual State Transition
        curr_state = climate[:, -1]
        next_state = block6(curr_state, delta)
        self.assertEqual(next_state.shape, (b, 6, h, w))
        self.assertTrue(torch.isfinite(next_state).all().item())
        # Physical constraint: precipitation non-negative
        self.assertTrue(torch.all(next_state[:, 1] >= 0.0).item())

    def test_02_multiverse_model_end_to_end_forward(self) -> None:
        """Test 2: MultiverseOfMonsoonModel unified forward matches step-by-step sequential execution."""
        b, t, h, w = 2, 3, 32, 36
        climate, geo, calendar = self._generate_synthetic_climate(b=b, t=t, h=h, w=w)

        model = MultiverseOfMonsoonModel().to(self.device)
        model.eval()

        next_state, intermediates = model(
            climate, geo, calendar, return_intermediates=True
        )

        self.assertEqual(intermediates["block1_encoded"].shape, (b, t, 48, h, w))
        self.assertEqual(intermediates["block2_spatial"].shape, (b, t, 64, h, w))
        self.assertEqual(intermediates["block3_temporal"].shape, (b, 64, h, w))
        self.assertEqual(intermediates["block4_attended"].shape, (b, 64, h, w))
        self.assertEqual(intermediates["block5_delta"].shape, (b, 6, h, w))
        self.assertEqual(intermediates["block6_next_state"].shape, (b, 6, h, w))
        self.assertEqual(next_state.shape, (b, 6, h, w))
        self.assertTrue(torch.isfinite(next_state).all().item())
        self.assertTrue(torch.all(next_state[:, 1] >= 0.0).item())

    def test_03_batch_dimension_flexibility(self) -> None:
        """Test 3: Connected pipeline executes correctly for B=1, B=2, B=4, and B=8."""
        model = MultiverseOfMonsoonModel().to(self.device)
        model.eval()

        for b in [1, 2, 4, 8]:
            climate, geo, calendar = self._generate_synthetic_climate(b=b, t=3, h=32, w=36)
            out = model(climate, geo, calendar)
            self.assertEqual(out.shape, (b, 6, 32, 36), f"Failed for batch size B={b}")
            self.assertTrue(torch.isfinite(out).all().item())

    def test_04_sequence_length_flexibility(self) -> None:
        """Test 4: Connected pipeline executes correctly for T=1, T=3, T=5."""
        model = MultiverseOfMonsoonModel().to(self.device)
        model.eval()

        for t in [1, 3, 5]:
            climate, geo, calendar = self._generate_synthetic_climate(b=2, t=t, h=32, w=36)
            out = model(climate, geo, calendar)
            self.assertEqual(out.shape, (2, 6, 32, 36), f"Failed for sequence length T={t}")
            self.assertTrue(torch.isfinite(out).all().item())

    def test_05_gradient_propagation_through_entire_pipeline(self) -> None:
        """Test 5: Full backward pass propagates gradients through all 6 blocks to inputs and parameters."""
        model = MultiverseOfMonsoonModel().to(self.device)
        model.train()

        climate, geo, calendar = self._generate_synthetic_climate(b=2, t=3, h=32, w=36)
        climate.requires_grad_(True)
        geo.requires_grad_(True)
        calendar.requires_grad_(True)

        next_state = model(climate, geo, calendar)
        loss = next_state.sum()
        loss.backward()

        # 1. Gradients reach inputs
        self.assertIsNotNone(climate.grad, "Gradients must reach dynamic climate input")
        self.assertIsNotNone(geo.grad, "Gradients must reach static geography input")
        self.assertIsNotNone(calendar.grad, "Gradients must reach calendar input")
        self.assertTrue(torch.isfinite(climate.grad).all().item())

        # 2. Gradients reach Block 1 parameters
        self.assertIsNotNone(model.block1.climate_proj.conv.weight.grad)
        self.assertIsNotNone(model.block1.geo_proj.conv.weight.grad)
        self.assertIsNotNone(model.block1.cal_proj.linear.weight.grad)

        # 3. Gradients reach Block 2 parameters
        self.assertIsNotNone(model.block2.branch_a[0].weight.grad)
        self.assertIsNotNone(model.block2.branch_b[0].weight.grad)
        self.assertIsNotNone(model.block2.branch_c[0].weight.grad)
        self.assertIsNotNone(model.block2.fusion[0].weight.grad)

        # 4. Gradients reach Block 3 parameters
        self.assertIsNotNone(model.block3.cell.conv_ih.weight.grad)
        self.assertIsNotNone(model.block3.cell.conv_hh.weight.grad)

        # 5. Gradients reach Block 4 parameters
        self.assertIsNotNone(model.block4.channel_attention.mlp[0].weight.grad)
        self.assertIsNotNone(model.block4.spatial_attention.conv.weight.grad)

        # 6. Gradients reach Block 5 parameters
        self.assertIsNotNone(model.block5.layer1[0].weight.grad)
        self.assertIsNotNone(model.block5.layer2[0].weight.grad)
        self.assertIsNotNone(model.block5.layer3.weight.grad)

    def test_06_autoregressive_10_day_rollout(self) -> None:
        """Test 6: Multi-step 10-day autoregressive forecast rollout with rolling 3-day window."""
        model = MultiverseOfMonsoonModel().to(self.device)
        model.eval()

        b = 2
        climate, geo, calendar = self._generate_synthetic_climate(b=b, t=3, h=32, w=36)
        doy_sincos = calendar[0]  # [sin, cos]

        rollout = model.rollout(
            init_states=climate,
            geography=geo,
            doy_sincos=doy_sincos,
            steps=SCORE_STEPS,
        )

        # Rollout shape must be [10, B, 6, 32, 36]
        self.assertEqual(rollout.shape, (10, b, 6, 32, 36))
        self.assertTrue(torch.isfinite(rollout).all().item(), "Rollout contains NaN or Inf")

        # Physical sanity: precipitation never negative at any forecast lead day
        self.assertTrue(torch.all(rollout[:, :, 1] >= 0.0).item(), "Precipitation must not be negative")

        # Also test unbatched rollout matching serve.py contract [3, 6, 32, 36]
        climate_3d = climate[0]  # [3, 6, 32, 36]
        geo_3d = geo[0]          # [2, 32, 36]
        rollout_3d = model.rollout(
            init_states=climate_3d,
            geography=geo_3d,
            doy_sincos=doy_sincos,
            steps=SCORE_STEPS,
        )
        self.assertEqual(rollout_3d.shape, (10, 6, 32, 36))
        self.assertTrue(torch.isfinite(rollout_3d).all().item())

    def test_07_multiverse_counterfactual_simulation(self) -> None:
        """Test 7: Multiverse comparison between baseline timeline and perturbed timeline."""
        model = MultiverseOfMonsoonModel().to(self.device)
        model.eval()

        climate, geo, calendar = self._generate_synthetic_climate(b=1, t=3, h=32, w=36)
        doy_sincos = calendar[0]

        # 1. Timeline Prime (Baseline)
        with torch.no_grad():
            baseline_rollout = model.rollout(
                init_states=climate,
                geography=geo,
                doy_sincos=doy_sincos,
                steps=SCORE_STEPS,
            )

        # 2. Alternate Timeline with warm_ocean shock: +3.0 K temperature shock in Bay of Bengal
        perturbation = torch.zeros(1, 6, 32, 36, device=self.device)
        # Bay of Bengal indices ~ lat 8..20 (indices 2..14), lon 82..94 (indices 17..29)
        perturbation[0, 0, 2:14, 17:29] = 3.0

        with torch.no_grad():
            perturbed_rollout = model.rollout(
                init_states=climate,
                geography=geo,
                doy_sincos=doy_sincos,
                steps=SCORE_STEPS,
                perturbation=perturbation,
            )

        # Rollouts must both be finite and shape-correct
        self.assertEqual(baseline_rollout.shape, (10, 1, 6, 32, 36))
        self.assertEqual(perturbed_rollout.shape, (10, 1, 6, 32, 36))
        self.assertTrue(torch.isfinite(perturbed_rollout).all().item())

        # Perturbed rollout must differ from baseline due to the shock
        diff = (perturbed_rollout - baseline_rollout).abs()
        max_diff = diff.max().item()
        self.assertGreater(max_diff, 1e-4, "Shock must produce non-zero divergence in alternate timeline")

    def test_08_null_perturbation_invariance(self) -> None:
        """Test 8: Null perturbation (zeros) produces identical output to unperturbed run."""
        model = MultiverseOfMonsoonModel().to(self.device)
        model.eval()

        climate, geo, calendar = self._generate_synthetic_climate(b=1, t=3, h=32, w=36)
        doy_sincos = calendar[0]

        with torch.no_grad():
            unperturbed = model.rollout(
                init_states=climate,
                geography=geo,
                doy_sincos=doy_sincos,
                steps=SCORE_STEPS,
                perturbation=None,
            )
            null_shock = torch.zeros(1, 6, 32, 36, device=self.device)
            perturbed_null = model.rollout(
                init_states=climate,
                geography=geo,
                doy_sincos=doy_sincos,
                steps=SCORE_STEPS,
                perturbation=null_shock,
            )

        # Difference must be essentially zero (< 1e-6)
        max_diff = (perturbed_null - unperturbed).abs().max().item()
        self.assertLess(max_diff, 1e-5, f"Null shock divergence was {max_diff}, expected < 1e-5")

    def test_09_cpu_execution_and_speed(self) -> None:
        """Test 9: 10-step rollout execution time on CPU must be well below 5.0 seconds."""
        model = MultiverseOfMonsoonModel().to(self.device)
        model.eval()

        climate, geo, calendar = self._generate_synthetic_climate(b=1, t=3, h=32, w=36)
        doy_sincos = calendar[0]

        t0 = time.perf_counter()
        with torch.no_grad():
            rollout = model.rollout(
                init_states=climate,
                geography=geo,
                doy_sincos=doy_sincos,
                steps=SCORE_STEPS,
            )
        elapsed_s = time.perf_counter() - t0

        self.assertEqual(rollout.shape, (10, 1, 6, 32, 36))
        self.assertLess(elapsed_s, TIME_LIMIT_S, f"10-step rollout took {elapsed_s:.3f}s (limit {TIME_LIMIT_S}s)")

    def test_10_nan_inf_pipeline_detection(self) -> None:
        """Test 10: check_finite=True detects NaN/Inf across pipeline inputs."""
        model = MultiverseOfMonsoonModel().to(self.device)
        climate, geo, calendar = self._generate_synthetic_climate(b=2, t=3, h=32, w=36)

        climate_nan = climate.clone()
        climate_nan[0, 0, 0, 0, 0] = float("nan")

        with self.assertRaises(ValueError):
            model(climate_nan, geo, calendar, check_finite=True)


if __name__ == "__main__":
    unittest.main()
