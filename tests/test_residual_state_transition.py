"""Unit and integration tests for Block 6: Residual State Transition.

Tests:
- Test A: Tensor shape tests ([1, 6, 32, 36] + [1, 6, 32, 36] -> [1, 6, 32, 36]) and aliases
- Test B: Batch compatibility (B=1, B=2, B=4, B=8)
- Test C: Residual correctness (Zero delta produces exact state, known delta updates each channel correctly)
- Test D: Precipitation tests (Non-negative clamp, zero rain preservation, positive rain unaltered, other channels unaffected)
- Test E: Numerical stability (No NaN, No Inf across realistic meteorological input ranges)
- Test F: Gradient flow (Gradients propagate through transition to both current_state and predicted_delta)
- Test G: CPU compatibility (Operates strictly on CPU without CUDA)
- Test H: Input validation (Channel count, shape mismatch, 3D/5D shapes, B=0, H/W < 1, non-tensor)
- Test I: Integration with Block 5 (Block 5 output passed into Block 6, finite predictions, gradient backprop)
- Test J: Full pipeline integration (Block 1 -> Block 2 -> Block 3 -> Block 4 -> Block 5 -> Block 6)
- Test K: Autoregressive rollout simulation (Multi-step rolling window update)
- Test L: Perturbation compatibility (Physical perturbation applied to initial climate state)
- Test M: Channel scaling and normalization flexibility (from_default_scales, from_stats_dict, normalized lower bounds)
- Test N: Regression tests verifying Blocks 1-5 continue to pass all checks
"""

from __future__ import annotations

import math
import unittest
import torch
import torch.nn as nn

from monsoon_r1.contract import CHANNELS, NC, NLAT, NLON
from monsoon_r1.models.climate_input_encoder import ClimateInputEncoder
from monsoon_r1.models.multi_scale_spatial_encoder import MultiScaleSpatialEncoder
from monsoon_r1.models.temporal_dynamics_encoder import TemporalDynamicsEncoder
from monsoon_r1.models.spatial_context_attention import SpatialContextAttention
from monsoon_r1.models.forecast_decoder import ForecastDecoder
from monsoon_r1.models.residual_state_transition import (
    Block6ResidualStateTransition,
    ClimateResidualStateTransition,
    ResidualStateTransition,
)


class TestResidualStateTransition(unittest.TestCase):
    """Comprehensive test suite for Block 6: ResidualStateTransition."""

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        torch.manual_seed(42)

    def _sample_climate_state(
        self,
        b: int = 1,
        c: int = NC,
        h: int = NLAT,
        w: int = NLON,
    ) -> torch.Tensor:
        """Create sample physical climate state with realistic values."""
        state = torch.zeros(b, c, h, w, device=self.device)
        state[:, 0] = torch.rand(b, h, w) * 20.0 + 290.0  # t2m ~ 290 - 310 K
        state[:, 1] = torch.rand(b, h, w) * 50.0          # tp ~ 0 - 50 mm/day
        state[:, 2] = torch.randn(b, h, w) * 5.0          # u10 ~ -15 to +15 m/s
        state[:, 3] = torch.randn(b, h, w) * 5.0          # v10 ~ -15 to +15 m/s
        state[:, 4] = torch.rand(b, h, w) * 2000.0 + 100000.0  # msl ~ 100,000 - 102,000 Pa
        state[:, 5] = torch.rand(b, h, w) * 40.0 + 10.0   # tcwv ~ 10 - 50 kg/m²
        return state

    def _sample_delta(
        self,
        b: int = 1,
        c: int = NC,
        h: int = NLAT,
        w: int = NLON,
        scale: float = 1.0,
    ) -> torch.Tensor:
        """Create sample predicted deltas."""
        return torch.randn(b, c, h, w, device=self.device) * scale

    def test_a_tensor_shapes(self) -> None:
        """Test A: Input [1, 6, 32, 36] + [1, 6, 32, 36] must produce [1, 6, 32, 36]."""
        transition = ResidualStateTransition().to(self.device)
        curr = self._sample_climate_state(b=1, c=6, h=32, w=36)
        delta = self._sample_delta(b=1, c=6, h=32, w=36)

        out = transition(curr, delta)

        expected_shape = (1, 6, 32, 36)
        self.assertEqual(out.shape, expected_shape)
        self.assertEqual(out.dtype, torch.float32)

        # Verify aliases produce identical results
        alias1 = ClimateResidualStateTransition().to(self.device)
        out_alias1 = alias1(curr, delta)
        torch.testing.assert_close(out, out_alias1)

        alias2 = Block6ResidualStateTransition().to(self.device)
        out_alias2 = alias2(curr, delta)
        torch.testing.assert_close(out, out_alias2)

    def test_b_batch_compatibility(self) -> None:
        """Test B: Verify output batch dimension is preserved for B=1, B=2, B=4, B=8."""
        transition = ResidualStateTransition().to(self.device)

        for b in [1, 2, 4, 8]:
            curr = self._sample_climate_state(b=b, c=6, h=32, w=36)
            delta = self._sample_delta(b=b, c=6, h=32, w=36)
            out = transition(curr, delta)
            self.assertEqual(out.shape, (b, 6, 32, 36), f"Failed for batch size B={b}")

    def test_c_residual_correctness(self) -> None:
        """Test C: Controlled tensors verify zero delta and known deltas across all channels."""
        transition = ResidualStateTransition().to(self.device)
        b, c, h, w = 2, 6, 32, 36

        # 1. Zero delta produces the exact same state (x + 0 = x)
        curr = self._sample_climate_state(b=b, c=c, h=h, w=w)
        zero_delta = torch.zeros(b, c, h, w, device=self.device)
        out_zero = transition(curr, zero_delta)
        torch.testing.assert_close(out_zero, curr, msg="Zero delta must return identical state")

        # 2. Known delta produces mathematically correct result (x + delta)
        known_delta = torch.zeros(b, c, h, w, device=self.device)
        # Channel 0 (t2m): +2.5 K
        known_delta[:, 0] = 2.5
        # Channel 1 (tp): +10.0 mm/day
        known_delta[:, 1] = 10.0
        # Channel 2 (u10): -1.2 m/s
        known_delta[:, 2] = -1.2
        # Channel 3 (v10): +0.8 m/s
        known_delta[:, 3] = 0.8
        # Channel 4 (msl): -150.0 Pa
        known_delta[:, 4] = -150.0
        # Channel 5 (tcwv): +3.4 kg/m²
        known_delta[:, 5] = 3.4

        out_known = transition(curr, known_delta)

        # Verify each channel individually
        torch.testing.assert_close(out_known[:, 0], curr[:, 0] + 2.5)
        torch.testing.assert_close(out_known[:, 1], curr[:, 1] + 10.0)
        torch.testing.assert_close(out_known[:, 2], curr[:, 2] - 1.2)
        torch.testing.assert_close(out_known[:, 3], curr[:, 3] + 0.8)
        torch.testing.assert_close(out_known[:, 4], curr[:, 4] - 150.0)
        torch.testing.assert_close(out_known[:, 5], curr[:, 5] + 3.4)

    def test_d_precipitation_handling(self) -> None:
        """Test D: Physical constraints on precipitation (non-negativity, zero rain, positive rain)."""
        transition = ResidualStateTransition(clamp_precipitation=True, precipitation_min=0.0).to(self.device)
        b, c, h, w = 2, 6, 32, 36

        curr = self._sample_climate_state(b=b, c=c, h=h, w=w)
        # Force initial precipitation to known controlled values
        curr[:, 1] = 15.0  # 15 mm/day

        # Case 1: Large negative delta that would result in negative precipitation
        large_negative_delta = torch.zeros(b, c, h, w, device=self.device)
        large_negative_delta[:, 1] = -50.0  # 15 - 50 = -35 mm/day -> must be clamped to 0.0

        out_neg = transition(curr, large_negative_delta)
        self.assertTrue(torch.all(out_neg[:, 1] >= 0.0), "Precipitation must never be negative")
        self.assertTrue(torch.all(out_neg[:, 1] == 0.0), "Excessive negative rain delta must clamp to 0.0")

        # Verify other channels are NOT modified by the precipitation clamp
        for ch in [0, 2, 3, 4, 5]:
            torch.testing.assert_close(out_neg[:, ch], curr[:, ch])

        # Case 2: Zero precipitation remains valid (0 + 0 = 0)
        curr_zero_rain = curr.clone()
        curr_zero_rain[:, 1] = 0.0
        zero_delta = torch.zeros(b, c, h, w, device=self.device)
        out_zero_rain = transition(curr_zero_rain, zero_delta)
        self.assertTrue(torch.all(out_zero_rain[:, 1] == 0.0))

        # Case 3: Positive precipitation is unaltered (15 + 10 = 25 mm/day)
        pos_delta = torch.zeros(b, c, h, w, device=self.device)
        pos_delta[:, 1] = 10.0
        out_pos = transition(curr, pos_delta)
        torch.testing.assert_close(out_pos[:, 1], torch.full_like(curr[:, 1], 25.0))

        # Case 4: Disabling clamp allows negative values if explicitly requested
        unclamped_transition = ResidualStateTransition(clamp_precipitation=False).to(self.device)
        out_unclamped = unclamped_transition(curr, large_negative_delta)
        self.assertTrue(torch.all(out_unclamped[:, 1] == -35.0))

    def test_e_numerical_stability(self) -> None:
        """Test E: Numerical stability (no NaN, no Inf) across various input distributions."""
        transition = ResidualStateTransition().to(self.device)

        # 1. Realistic meteorological ranges
        curr_real = self._sample_climate_state(b=4, c=6, h=32, w=36)
        delta_real = self._sample_delta(b=4, c=6, h=32, w=36, scale=2.0)
        out_real = transition(curr_real, delta_real)
        self.assertTrue(torch.isfinite(out_real).all(), "Output must be completely finite for realistic inputs")

        # 2. Zero inputs
        curr_zeros = torch.zeros(2, 6, 32, 36, device=self.device)
        delta_zeros = torch.zeros(2, 6, 32, 36, device=self.device)
        out_zeros = transition(curr_zeros, delta_zeros)
        self.assertTrue(torch.isfinite(out_zeros).all())
        torch.testing.assert_close(out_zeros, curr_zeros)

        # 3. Large input ranges
        curr_large = torch.full((2, 6, 32, 36), 1e5, device=self.device)
        delta_large = torch.full((2, 6, 32, 36), 1e4, device=self.device)
        out_large = transition(curr_large, delta_large)
        self.assertTrue(torch.isfinite(out_large).all())

        # 4. Extreme negative inputs for non-precipitation channels
        curr_neg = torch.full((2, 6, 32, 36), -500.0, device=self.device)
        delta_neg = torch.full((2, 6, 32, 36), -100.0, device=self.device)
        out_neg = transition(curr_neg, delta_neg)
        self.assertTrue(torch.isfinite(out_neg).all())
        self.assertTrue(torch.all(out_neg[:, 1] >= 0.0), "Precipitation must remain >= 0 even with negative inputs")

    def test_f_gradient_flow(self) -> None:
        """Test F: Gradients propagate correctly through transition to both state and delta."""
        transition = ResidualStateTransition().to(self.device)
        b, c, h, w = 2, 6, 32, 36

        curr = self._sample_climate_state(b=b, c=c, h=h, w=w)
        curr.requires_grad_(True)

        delta = self._sample_delta(b=b, c=c, h=h, w=w)
        delta.requires_grad_(True)

        out = transition(curr, delta)

        # Verify tensors are not detached
        self.assertTrue(out.requires_grad, "Output must require grad when inputs require grad")

        # Compute loss and backpropagate
        loss = out.sum()
        loss.backward()

        self.assertIsNotNone(curr.grad, "Gradients must propagate to current_state")
        self.assertIsNotNone(delta.grad, "Gradients must propagate to predicted_delta")
        self.assertTrue(torch.isfinite(curr.grad).all(), "Gradients for current_state must be finite")
        self.assertTrue(torch.isfinite(delta.grad).all(), "Gradients for predicted_delta must be finite")

        # For channels 0, 2, 3, 4, 5, gradient w.r.t sum() must be 1.0 everywhere
        for ch in [0, 2, 3, 4, 5]:
            torch.testing.assert_close(curr.grad[:, ch], torch.ones(b, h, w, device=self.device))
            torch.testing.assert_close(delta.grad[:, ch], torch.ones(b, h, w, device=self.device))

        # Test subgradient behavior at precipitation clamp boundary
        curr_p = torch.tensor([[[[10.0]], [[-10.0]]]], requires_grad=True)
        delta_p = torch.tensor([[[[-20.0]], [[5.0]]]], requires_grad=True)
        # Make 4-channel shape [1, 2, 1, 1] for a minimal test
        mini_trans = ResidualStateTransition(num_channels=2, precipitation_idx=0)
        out_mini = mini_trans(curr_p, delta_p)
        # Element 0: 10 + (-20) = -10 -> clamped to 0 -> gradient should be 0.0
        # Element 1: -10 + 5 = -5 -> clamped to 0 -> gradient should be 0.0
        loss_mini = out_mini.sum()
        loss_mini.backward()
        self.assertEqual(curr_p.grad[0, 0, 0, 0].item(), 0.0)

    def test_g_cpu_compatibility(self) -> None:
        """Test G: Operates strictly on CPU without CUDA."""
        transition = ResidualStateTransition()
        curr = torch.randn(1, 6, 32, 36, device="cpu")
        delta = torch.randn(1, 6, 32, 36, device="cpu")

        out = transition(curr, delta)
        self.assertEqual(out.device.type, "cpu")

    def test_h_input_validation(self) -> None:
        """Test H: Input validation handles all malformed inputs with appropriate exceptions."""
        transition = ResidualStateTransition(num_channels=6)

        valid_curr = torch.randn(1, 6, 32, 36)
        valid_delta = torch.randn(1, 6, 32, 36)

        # 1. Non-tensor inputs -> TypeError
        with self.assertRaises(TypeError):
            transition("not_a_tensor", valid_delta)
        with self.assertRaises(TypeError):
            transition(valid_curr, [1.0, 2.0, 3.0])

        # 2. Incorrect dimensionality (3D, 5D) -> ValueError
        with self.assertRaises(ValueError):
            transition(torch.randn(6, 32, 36), valid_delta)
        with self.assertRaises(ValueError):
            transition(valid_curr, torch.randn(1, 1, 6, 32, 36))

        # 3. Shape mismatch between current state and delta -> ValueError
        with self.assertRaises(ValueError):
            transition(torch.randn(1, 6, 32, 36), torch.randn(2, 6, 32, 36))
        with self.assertRaises(ValueError):
            transition(torch.randn(1, 6, 32, 36), torch.randn(1, 6, 16, 18))

        # 4. Incorrect channel count -> ValueError
        with self.assertRaises(ValueError):
            transition(torch.randn(1, 5, 32, 36), torch.randn(1, 5, 32, 36))
        with self.assertRaises(ValueError):
            transition(torch.randn(1, 7, 32, 36), torch.randn(1, 7, 32, 36))

        # 5. Invalid batch size (B=0) -> ValueError
        with self.assertRaises(ValueError):
            transition(torch.empty(0, 6, 32, 36), torch.empty(0, 6, 32, 36))

        # 6. Invalid spatial dimensions -> ValueError
        with self.assertRaises(ValueError):
            transition(torch.empty(1, 6, 0, 36), torch.empty(1, 6, 0, 36))

        # 7. Device / dtype mismatch -> ValueError
        with self.assertRaises(ValueError):
            transition(valid_curr.float(), valid_delta.double())

        # 8. Invalid initialization parameters -> ValueError
        with self.assertRaises(ValueError):
            ResidualStateTransition(num_channels=0)
        with self.assertRaises(ValueError):
            ResidualStateTransition(num_channels=6, precipitation_idx=6)
        with self.assertRaises(ValueError):
            ResidualStateTransition(num_channels=6, precipitation_idx=-1)

    def test_i_integration_with_block_5(self) -> None:
        """Test I: Output of Block 5 (ForecastDecoder) passes directly into Block 6."""
        block4 = SpatialContextAttention().to(self.device)
        block5 = ForecastDecoder().to(self.device)
        block6 = ResidualStateTransition().to(self.device)

        b, c, h, w = 2, 64, 32, 36

        # Block 4 input feature representation
        x4 = torch.randn(b, c, h, w, device=self.device, requires_grad=True)

        # 1. Run Block 4 -> [B, 64, 32, 36]
        feat4 = block4(x4)
        self.assertEqual(feat4.shape, (b, 64, h, w))

        # 2. Run Block 5 -> [B, 6, 32, 36]
        deltas = block5(feat4)
        self.assertEqual(deltas.shape, (b, 6, h, w))
        self.assertEqual(block5.CHANNELS, CHANNELS)

        # 3. Run Block 6 with current climate state
        curr_state = self._sample_climate_state(b=b, c=6, h=h, w=w)
        next_state = block6(curr_state, deltas)

        # Verify output
        self.assertEqual(next_state.shape, (b, 6, h, w))
        self.assertTrue(torch.isfinite(next_state).all())
        self.assertTrue(torch.all(next_state[:, 1] >= 0.0), "Precipitation must be non-negative")

        # 4. Backward pass through Block 6 -> Block 5 -> Block 4
        loss = next_state.sum()
        loss.backward()

        self.assertIsNotNone(block5.layer3.weight.grad, "Gradients must reach Block 5 layer 3")
        self.assertIsNotNone(block5.layer2[0].weight.grad, "Gradients must reach Block 5 layer 2")
        self.assertIsNotNone(block5.layer1[0].weight.grad, "Gradients must reach Block 5 layer 1")
        self.assertIsNotNone(x4.grad, "Gradients must reach Block 4 input")

    def test_j_full_pipeline_integration(self) -> None:
        """Test J: End-to-end integration across all 6 blocks (Block 1 through Block 6)."""
        block1 = ClimateInputEncoder().to(self.device)
        block2 = MultiScaleSpatialEncoder().to(self.device)
        block3 = TemporalDynamicsEncoder().to(self.device)
        block4 = SpatialContextAttention().to(self.device)
        block5 = ForecastDecoder().to(self.device)
        block6 = ResidualStateTransition().to(self.device)

        b, t, h, w = 2, 3, 32, 36

        # 1. Raw multi-day climate sequence [B, T, 6, H, W]
        climate_seq = torch.randn(b, t, 6, h, w, device=self.device, requires_grad=True)
        climate_scaled = climate_seq * 5.0 + 300.0
        # Ensure precipitation in input is positive
        with torch.no_grad():
            climate_scaled[:, :, 1] = torch.rand(b, t, h, w) * 20.0

        geo = torch.zeros(b, 2, h, w, device=self.device)
        geo[:, 0] = torch.rand(b, h, w)           # land_sea_mask
        geo[:, 1] = torch.rand(b, h, w) * 2000.0  # orography_m

        doy = 180.0
        angle = 2.0 * math.pi * doy / 365.0
        cal = torch.tensor([[math.sin(angle), math.cos(angle)]] * b, device=self.device)

        # 2. Block 1: [B, T, 6, H, W] -> [B, T, 48, H, W]
        out1 = block1(climate_scaled, geo, cal)
        self.assertEqual(out1.shape, (b, t, 48, h, w))

        # 3. Block 2: [B, T, 48, H, W] -> [B, T, 64, H, W]
        out2 = block2(out1)
        self.assertEqual(out2.shape, (b, t, 64, h, w))

        # 4. Block 3: [B, T, 64, H, W] -> [B, 64, H, W]
        out3 = block3(out2)
        self.assertEqual(out3.shape, (b, 64, h, w))

        # 5. Block 4: [B, 64, H, W] -> [B, 64, H, W]
        out4 = block4(out3)
        self.assertEqual(out4.shape, (b, 64, h, w))

        # 6. Block 5: [B, 64, H, W] -> [B, 6, H, W] (predicted deltas)
        pred_delta = block5(out4)
        self.assertEqual(pred_delta.shape, (b, 6, h, w))

        # 7. Block 6: Current state is the latest day in sequence (Day 3 = climate_scaled[:, -1])
        curr_state = climate_scaled[:, -1]
        next_state = block6(curr_state, pred_delta)
        self.assertEqual(next_state.shape, (b, 6, h, w))
        self.assertTrue(torch.all(next_state[:, 1] >= 0.0), "Predicted precipitation must be non-negative")

        # 8. Full end-to-end backpropagation through all 6 blocks
        loss = next_state.mean()
        loss.backward()

        # Check gradients reach all 6 blocks
        self.assertIsNotNone(block5.layer3.weight.grad, "Block 5 layer 3 has grad")
        self.assertIsNotNone(block4.spatial_attention.conv.weight.grad, "Block 4 spatial conv has grad")
        self.assertIsNotNone(block3.cell.conv_ih.weight.grad, "Block 3 conv_ih has grad")
        self.assertIsNotNone(block2.fusion[0].weight.grad, "Block 2 fusion conv has grad")
        self.assertIsNotNone(block1.climate_proj.conv.weight.grad, "Block 1 climate conv has grad")
        self.assertIsNotNone(climate_seq.grad, "Raw climate input sequence has grad")

    def test_k_autoregressive_rollout_simulation(self) -> None:
        """Test K: Simulate a multi-step rolling window autoregressive rollout."""
        block1 = ClimateInputEncoder().to(self.device)
        block2 = MultiScaleSpatialEncoder().to(self.device)
        block3 = TemporalDynamicsEncoder().to(self.device)
        block4 = SpatialContextAttention().to(self.device)
        block5 = ForecastDecoder().to(self.device)
        block6 = ResidualStateTransition().to(self.device)

        b, t, h, w = 1, 3, 32, 36
        num_steps = 3

        # Initial rolling window [Day 1, Day 2, Day 3]
        window = self._sample_climate_state(b=b, c=6, h=h, w=w).unsqueeze(1).repeat(1, t, 1, 1, 1)
        geo = torch.zeros(b, 2, h, w, device=self.device)
        cal = torch.tensor([[0.5, 0.866]] * b, device=self.device)

        rollout = []
        for step in range(num_steps):
            # 1. Blocks 1-5 predict delta for next day
            out1 = block1(window, geo, cal)
            out2 = block2(out1)
            out3 = block3(out2)
            out4 = block4(out3)
            delta = block5(out4)

            # 2. Block 6 produces next state from current state (latest state in window)
            curr = window[:, -1]
            next_s = block6(curr, delta)

            # 3. Verify validity
            self.assertEqual(next_s.shape, (b, 6, h, w))
            self.assertTrue(torch.isfinite(next_s).all())
            self.assertTrue(torch.all(next_s[:, 1] >= 0.0))

            rollout.append(next_s)

            # 4. Slide window forward: drop oldest day, append new state
            window = torch.cat([window[:, 1:], next_s.unsqueeze(1)], dim=1)

        # Verify rollout shape
        rollout_tensor = torch.stack(rollout, dim=0)  # [steps, B, 6, H, W]
        self.assertEqual(rollout_tensor.shape, (num_steps, b, 6, h, w))

    def test_l_perturbation_compatibility(self) -> None:
        """Test L: Physical perturbation applied to initial climate state."""
        transition = ResidualStateTransition().to(self.device)
        b, c, h, w = 1, 6, 32, 36

        init_state = self._sample_climate_state(b=b, c=c, h=h, w=w)

        # Create physical perturbation delta (e.g. +3.0 K temperature shock in Bay of Bengal)
        perturbation = torch.zeros(b, c, h, w, device=self.device)
        perturbation[:, 0, 10:20, 15:25] = 3.0  # +3 K temperature anomaly

        # 1. Batched application via apply_perturbation
        perturbed_state = transition.apply_perturbation(init_state, perturbation)
        self.assertEqual(perturbed_state.shape, (b, 6, h, w))
        # Verify perturbed region has higher temperature
        diff = perturbed_state - init_state
        torch.testing.assert_close(diff[:, 0, 10:20, 15:25], torch.full((b, 10, 10), 3.0))

        # 2. Unbatched 3D shape [6, 32, 36] (wire format for serve.py)
        init_3d = init_state.squeeze(0)
        pert_3d = perturbation.squeeze(0)
        perturbed_3d = transition.apply_perturbation(init_3d, pert_3d)
        self.assertEqual(perturbed_3d.shape, (6, h, w))
        torch.testing.assert_close(perturbed_3d, perturbed_state.squeeze(0))

    def test_m_scaling_and_normalization_flexibility(self) -> None:
        """Test M: Channel scaling buffer and normalized precipitation bounds."""
        # 1. Custom scales via constructor
        scales = [8.0, 8.0, 3.5, 2.5, 450.0, 20.0]
        trans_scaled = ResidualStateTransition(channel_scales=scales).to(self.device)

        curr = torch.zeros(1, 6, 32, 36, device=self.device)
        norm_delta = torch.ones(1, 6, 32, 36, device=self.device)

        out_scaled = trans_scaled(curr, norm_delta)
        # Expected: delta is multiplied by scale before addition
        for ch, sc in enumerate(scales):
            torch.testing.assert_close(out_scaled[:, ch], torch.full((1, 32, 36), sc))

        # 2. from_default_scales factory method
        from monsoon_r1.scoring import STD_DEFAULT
        trans_def = ResidualStateTransition.from_default_scales()
        self.assertIsNotNone(trans_def.channel_scales)
        torch.testing.assert_close(
            trans_def.channel_scales.squeeze(),
            torch.tensor(STD_DEFAULT, dtype=torch.float32)
        )

        # 3. from_stats_dict factory method
        stats_dict = {
            "channels": list(CHANNELS),
            "mean": [300.0, 5.0, 1.0, 0.5, 101325.0, 30.0],
            "std": [8.0, 8.0, 3.5, 2.5, 450.0, 20.0],
        }
        trans_stats = ResidualStateTransition.from_stats_dict(stats_dict, state_is_normalized=True)
        self.assertIsNotNone(trans_stats.channel_means)
        self.assertIsNotNone(trans_stats.channel_scales)
        # Effective min in normalized space: (0.0 - mean) / std = -5.0 / 8.0 = -0.625
        eff_min = trans_stats._get_effective_precipitation_min()
        self.assertAlmostEqual(eff_min, -0.625, places=5)

        # 4. Normalized state clamp check
        norm_curr = torch.zeros(1, 6, 32, 36)
        norm_delta_neg = torch.full((1, 6, 32, 36), -2.0)  # -2.0 is below -0.625
        out_norm = trans_stats(norm_curr, norm_delta_neg)
        self.assertTrue(torch.all(out_norm[:, 1] >= eff_min))
        self.assertTrue(torch.all(out_norm[:, 1] == eff_min))

    def test_n_regression_blocks_1_to_5(self) -> None:
        """Test N: Regression check ensuring Blocks 1-5 continue to behave identically."""
        block1 = ClimateInputEncoder().to(self.device)
        block2 = MultiScaleSpatialEncoder().to(self.device)
        block3 = TemporalDynamicsEncoder().to(self.device)
        block4 = SpatialContextAttention().to(self.device)
        block5 = ForecastDecoder().to(self.device)

        b, t, h, w = 1, 3, 32, 36
        climate = torch.randn(b, t, 6, h, w, device=self.device)
        geo = torch.randn(b, 2, h, w, device=self.device)
        cal = torch.randn(b, 2, device=self.device)

        out1 = block1(climate, geo, cal)
        self.assertEqual(out1.shape, (b, t, 48, h, w))

        out2 = block2(out1)
        self.assertEqual(out2.shape, (b, t, 64, h, w))

        out3 = block3(out2)
        self.assertEqual(out3.shape, (b, 64, h, w))

        out4 = block4(out3)
        self.assertEqual(out4.shape, (b, 64, h, w))

        out5 = block5(out4)
        self.assertEqual(out5.shape, (b, 6, h, w))

    def test_o_nan_and_inf_detection(self) -> None:
        """Test O: Proper detection and handling of NaN and Inf in Block 6."""
        transition = ResidualStateTransition().to(self.device)
        curr = self._sample_climate_state(b=2, c=6, h=32, w=36)
        delta = self._sample_delta(b=2, c=6, h=32, w=36)

        # 1. NaN in current_state with check_finite=True
        curr_nan = curr.clone()
        curr_nan[0, 0, 10, 10] = float("nan")
        with self.assertRaises(ValueError):
            transition(curr_nan, delta, check_finite=True)

        # NaN propagation with check_finite=False
        out_nan = transition(curr_nan, delta, check_finite=False)
        self.assertTrue(torch.isnan(out_nan).any().item())

        # 2. NaN in predicted_delta with check_finite=True
        delta_nan = delta.clone()
        delta_nan[0, 1, 10, 10] = float("nan")
        with self.assertRaises(ValueError):
            transition(curr, delta_nan, check_finite=True)

        # 3. Inf in current_state with check_finite=True
        curr_inf = curr.clone()
        curr_inf[0, 4, 10, 10] = float("inf")
        with self.assertRaises(ValueError):
            transition(curr_inf, delta, check_finite=True)

    def test_p_missing_inputs(self) -> None:
        """Test P: Handling of None / missing inputs."""
        transition = ResidualStateTransition().to(self.device)
        curr = self._sample_climate_state(b=1, c=6, h=32, w=36)
        delta = self._sample_delta(b=1, c=6, h=32, w=36)

        with self.assertRaises(TypeError):
            transition(None, delta)
        with self.assertRaises(TypeError):
            transition(curr, None)

    def test_q_deterministic_behavior(self) -> None:
        """Test Q: ResidualStateTransition is strictly deterministic."""
        transition = ResidualStateTransition().to(self.device)
        curr = self._sample_climate_state(b=2, c=6, h=32, w=36)
        delta = self._sample_delta(b=2, c=6, h=32, w=36)

        out1 = transition(curr, delta)
        out2 = transition(curr, delta)
        torch.testing.assert_close(out1, out2)

    def test_r_channel_specific_clamping_isolation(self) -> None:
        """Test R: Only precipitation is clamped; negative values are preserved on wind and temperature."""
        transition = ResidualStateTransition(clamp_precipitation=True).to(self.device)
        b, c, h, w = 1, 6, 32, 36

        curr = torch.zeros(b, c, h, w, device=self.device)
        curr[:, 0] = 300.0  # t2m
        curr[:, 1] = 5.0    # tp
        curr[:, 2] = 0.0    # u10
        curr[:, 3] = 0.0    # v10

        # Large negative delta
        delta = torch.zeros(b, c, h, w, device=self.device)
        delta[:, 0] = -50.0  # t2m drops by 50K -> 250K (valid physical cold)
        delta[:, 1] = -20.0  # tp drops by 20mm -> 5 - 20 = -15, must clamp to 0.0
        delta[:, 2] = -10.0  # u10 westward wind -10 m/s (must remain negative)
        delta[:, 3] = -12.0  # v10 southward wind -12 m/s (must remain negative)

        out = transition(curr, delta)

        # Precipitation must be clamped to exactly 0.0
        torch.testing.assert_close(out[:, 1], torch.zeros(b, h, w, device=self.device))
        # Wind must NOT be clamped to 0.0
        torch.testing.assert_close(out[:, 2], torch.full((b, h, w), -10.0, device=self.device))
        torch.testing.assert_close(out[:, 3], torch.full((b, h, w), -12.0, device=self.device))
        # Temperature must be 250.0
        torch.testing.assert_close(out[:, 0], torch.full((b, h, w), 250.0, device=self.device))


if __name__ == "__main__":
    unittest.main()
