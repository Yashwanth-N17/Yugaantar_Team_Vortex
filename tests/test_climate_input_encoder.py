"""Unit and integration tests for Block 1: ClimateInputEncoder.

Verifies:
- Test 1: Output shape [B, 3, 48, 32, 36]
- Test 2: Batch handling (B=1, B=4)
- Test 3: Sequence handling (T=1, T=3, T=5)
- Test 4: Spatial preservation (H=32, W=36)
- Test 5: Channel composition (32 climate + 8 geographic + 8 calendar = 48 total)
- Test 6: Finite values (no NaN or Inf on valid inputs)
- Test 7: Gradient propagation (gradients reach all learnable parameters)
- Test 8: CPU compatibility (all operations fully supported on CPU)
- Test 9: Input validation (error handling for invalid tensor shapes, channels, dimensions)
- Test 10: Calendar handling (broadcast, per-day sequence, and preceding-day derivation with wraparound)
"""

from __future__ import annotations

import math
import unittest
import pytest
import torch

from monsoon_r1.models.climate_input_encoder import (
    CalendarProjection,
    ClimateInputEncoder,
    ClimateProjection,
    GeographicProjection,
)


class TestClimateInputEncoder(unittest.TestCase):
    """Test suite for Block 1: ClimateInputEncoder and its subcomponents."""

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        torch.manual_seed(42)

    def _generate_synthetic_inputs(
        self,
        b: int = 2,
        t: int = 3,
        h: int = 32,
        w: int = 36,
        cal_seq: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Helper to generate realistic synthetic inputs."""
        # Realistic values: t2m ~ 300K, tp ~ 10mm, u10 ~ 2m/s, v10 ~ -3m/s, msl ~ 101325Pa, tcwv ~ 40kg/m2
        climate = torch.randn(b, t, 6, h, w, device=self.device)
        climate[:, :, 0] = climate[:, :, 0] * 10.0 + 300.0  # t2m
        climate[:, :, 1] = torch.clamp(climate[:, :, 1] * 5.0 + 5.0, min=0.0)  # tp
        climate[:, :, 4] = climate[:, :, 4] * 1000.0 + 101325.0  # msl

        # Static geography: land_sea_mask in [0, 1], orography_m in [0, 3000]
        geo = torch.zeros(b, 2, h, w, device=self.device)
        geo[:, 0] = torch.rand(b, h, w)  # mask
        geo[:, 1] = torch.rand(b, h, w) * 2500.0  # elevation

        # Calendar: doy 180 (June) -> sincos
        doy = 180.0
        angle = 2.0 * math.pi * doy / 365.0
        if cal_seq:
            calendar = torch.zeros(b, t, 2, device=self.device)
            for i in range(t):
                a = 2.0 * math.pi * (doy - (t - 1 - i)) / 365.0
                calendar[:, i, 0] = math.sin(a)
                calendar[:, i, 1] = math.cos(a)
        else:
            calendar = torch.tensor([[math.sin(angle), math.cos(angle)]] * b, device=self.device)

        return climate, geo, calendar

    def test_01_output_shape(self) -> None:
        """Test 1: Output shape must be [B, 3, 48, 32, 36] for input [B, 3, 6, 32, 36]."""
        encoder = ClimateInputEncoder().to(self.device)
        b, t, h, w = 2, 3, 32, 36
        climate, geo, calendar = self._generate_synthetic_inputs(b=b, t=t, h=h, w=w)

        out = encoder(climate, geo, calendar)

        expected_shape = (b, 3, 48, 32, 36)
        self.assertEqual(out.shape, expected_shape)

        # Under torch.no_grad(), requires_grad should be False
        with torch.no_grad():
            out_eval = encoder(climate, geo, calendar)
            self.assertEqual(out_eval.shape, expected_shape)
            self.assertFalse(out_eval.requires_grad)

    def test_02_batch_handling(self) -> None:
        """Test 2: Module must support different batch sizes, specifically B=1, B=2, B=4, and B=8."""
        encoder = ClimateInputEncoder().to(self.device)

        for b in [1, 2, 4, 8]:
            climate, geo, calendar = self._generate_synthetic_inputs(b=b, t=3, h=32, w=36)
            out = encoder(climate, geo, calendar)
            self.assertEqual(out.shape, (b, 3, 48, 32, 36), f"Failed for batch size B={b}")

    def test_03_sequence_handling(self) -> None:
        """Test 3: Module must support configurable sequence lengths, e.g. T=1, T=3, T=5."""
        encoder = ClimateInputEncoder().to(self.device)

        for t in [1, 2, 3, 5, 10]:
            climate, geo, calendar = self._generate_synthetic_inputs(b=2, t=t, h=32, w=36)
            out = encoder(climate, geo, calendar)
            self.assertEqual(out.shape, (2, t, 48, 32, 36), f"Failed for sequence length T={t}")

    def test_04_spatial_preservation(self) -> None:
        """Test 4: Output must preserve spatial dimensions Height = 32, Width = 36."""
        encoder = ClimateInputEncoder().to(self.device)

        climate, geo, calendar = self._generate_synthetic_inputs(b=2, t=3, h=32, w=36)
        out = encoder(climate, geo, calendar)

        self.assertEqual(out.size(-2), 32, "Height dimension was not preserved")
        self.assertEqual(out.size(-1), 36, "Width dimension was not preserved")

        # Also test on non-standard spatial grid
        climate_alt, geo_alt, cal_alt = self._generate_synthetic_inputs(b=1, t=2, h=16, w=20)
        out_alt = encoder(climate_alt, geo_alt, cal_alt)
        self.assertEqual(out_alt.shape, (1, 2, 48, 16, 20))

    def test_05_channel_composition(self) -> None:
        """Test 5: Channel composition must be exactly 32 climate, 8 geo, 8 calendar = 48 total."""
        encoder = ClimateInputEncoder().to(self.device)
        climate, geo, calendar = self._generate_synthetic_inputs(b=2, t=3, h=32, w=36)

        out = encoder(climate, geo, calendar)

        self.assertEqual(out.size(2), 48, "Total channel count must be 48")

        # Verify component sub-projections match exact slices
        climate_part = out[:, :, 0:32, :, :]
        geo_part = out[:, :, 32:40, :, :]
        cal_part = out[:, :, 40:48, :, :]

        self.assertEqual(climate_part.shape, (2, 3, 32, 32, 36))
        self.assertEqual(geo_part.shape, (2, 3, 8, 32, 36))
        self.assertEqual(cal_part.shape, (2, 3, 8, 32, 36))

        # Direct submodule outputs must equal the concatenated slices
        c_direct = encoder.climate_proj(climate)
        g_direct = encoder.geo_proj(geo, seq_len=3)
        k_direct = encoder.cal_proj(calendar, seq_len=3, height=32, width=36)

        torch.testing.assert_close(climate_part, c_direct)
        torch.testing.assert_close(geo_part, g_direct)
        torch.testing.assert_close(cal_part, k_direct)

    def test_06_finite_values(self) -> None:
        """Test 6: Verify output contains no NaN or infinity values when valid finite inputs are supplied."""
        encoder = ClimateInputEncoder().to(self.device)
        climate, geo, calendar = self._generate_synthetic_inputs(b=4, t=3, h=32, w=36)

        out = encoder(climate, geo, calendar)

        self.assertTrue(torch.isfinite(out).all().item(), "Output tensor contains NaN or Inf values")
        self.assertFalse(torch.isnan(out).any().item(), "Output contains NaN")
        self.assertFalse(torch.isinf(out).any().item(), "Output contains Inf")

    def test_07_gradient_propagation(self) -> None:
        """Test 7: Verify forward and backward pass propagates gradients to all learnable parameters."""
        encoder = ClimateInputEncoder().to(self.device)
        climate, geo, calendar = self._generate_synthetic_inputs(b=2, t=3, h=32, w=36)

        # Enable grad on climate input to verify end-to-end grad flow
        climate.requires_grad_(True)
        geo.requires_grad_(True)
        calendar.requires_grad_(True)

        out = encoder(climate, geo, calendar)
        loss = out.sum()
        loss.backward()

        # Check gradients in climate projection
        self.assertIsNotNone(encoder.climate_proj.conv.weight.grad)
        self.assertIsNotNone(encoder.climate_proj.conv.bias.grad)
        self.assertFalse(torch.all(encoder.climate_proj.conv.weight.grad == 0))

        # Check gradients in geographic projection
        self.assertIsNotNone(encoder.geo_proj.conv.weight.grad)
        self.assertIsNotNone(encoder.geo_proj.conv.bias.grad)
        self.assertFalse(torch.all(encoder.geo_proj.conv.weight.grad == 0))

        # Check gradients in calendar projection
        self.assertIsNotNone(encoder.cal_proj.linear.weight.grad)
        self.assertIsNotNone(encoder.cal_proj.linear.bias.grad)
        self.assertFalse(torch.all(encoder.cal_proj.linear.weight.grad == 0))

        # Check gradients on inputs
        self.assertIsNotNone(climate.grad)
        self.assertIsNotNone(geo.grad)
        self.assertIsNotNone(calendar.grad)

    def test_08_cpu_compatibility(self) -> None:
        """Test 8: Ensure the entire module and subcomponents operate purely on CPU."""
        encoder = ClimateInputEncoder().to("cpu")
        climate, geo, calendar = self._generate_synthetic_inputs(b=1, t=3, h=32, w=36)

        # Confirm all tensors are on CPU
        self.assertEqual(climate.device.type, "cpu")
        self.assertEqual(geo.device.type, "cpu")
        self.assertEqual(calendar.device.type, "cpu")

        out = encoder(climate, geo, calendar)
        self.assertEqual(out.device.type, "cpu")

    def test_09_input_validation(self) -> None:
        """Test 9: Test appropriate error handling for invalid tensor shapes, channels, and dimensions."""
        encoder = ClimateInputEncoder().to(self.device)
        climate, geo, calendar = self._generate_synthetic_inputs(b=2, t=3, h=32, w=36)

        # 1. Non-tensor input
        with self.assertRaises(TypeError):
            encoder("not_a_tensor", geo, calendar)
        with self.assertRaises(TypeError):
            encoder(climate, "not_a_tensor", calendar)
        with self.assertRaises(TypeError):
            encoder(climate, geo, "not_a_tensor")

        # 2. Invalid climate tensor dimensions (4D instead of 5D)
        with self.assertRaises(ValueError):
            encoder(climate[:, 0], geo, calendar)

        # 3. Invalid climate channel count (e.g. 5 instead of 6)
        with self.assertRaises(ValueError):
            encoder(climate[:, :, :5], geo, calendar)

        # 4. Invalid geographic channel count (e.g. 1 instead of 2)
        with self.assertRaises(ValueError):
            encoder(climate, geo[:, :1], calendar)

        # 5. Invalid calendar dimensions (1D tensor)
        with self.assertRaises(ValueError):
            encoder(climate, geo, torch.tensor([0.5, 0.5]))

        # 6. Batch size mismatch
        climate_b3, _, _ = self._generate_synthetic_inputs(b=3, t=3, h=32, w=36)
        with self.assertRaises(ValueError):
            encoder(climate_b3, geo, calendar)

        # 7. Spatial dimension mismatch between climate and geography
        geo_mismatch = torch.randn(2, 2, 16, 18)
        with self.assertRaises(ValueError):
            encoder(climate, geo_mismatch, calendar)

        # 8. Calendar sequence length mismatch for 3D calendar
        cal_wrong_t = torch.randn(2, 5, 2)  # T=5 while climate T=3
        with self.assertRaises(ValueError):
            encoder(climate, geo, cal_wrong_t)

    def test_10_calendar_handling_and_broadcasting(self) -> None:
        """Test 10: Test calendar projection, spatial broadcasting, per-day sequence, and preceding-day derivation."""
        # 1. Spatial broadcasting test: verify identical calendar representation across all (h, w)
        encoder = ClimateInputEncoder(derive_preceding_days=False).to(self.device)
        climate, geo, calendar = self._generate_synthetic_inputs(b=2, t=3, h=32, w=36)

        out = encoder(climate, geo, calendar)
        cal_features = out[:, :, 40:48, :, :]  # [B, T, 8, H, W]

        # For any cell (h, w), it should equal cell (0, 0)
        ref_cell = cal_features[:, :, :, 0:1, 0:1]
        diff = (cal_features - ref_cell).abs().max()
        self.assertAlmostEqual(diff.item(), 0.0, places=6, msg="Calendar feature is not spatially uniform")

        # 2. Per-day 3D calendar sequence test: [B, T, 2]
        climate, geo, cal_seq = self._generate_synthetic_inputs(b=2, t=3, h=32, w=36, cal_seq=True)
        out_seq = encoder(climate, geo, cal_seq)
        self.assertEqual(out_seq.shape, (2, 3, 48, 32, 36))

        # 3. Preceding-day derivation test with derive_preceding_days=True
        encoder_derived = ClimateInputEncoder(derive_preceding_days=True).to(self.device)
        out_derived = encoder_derived(climate, geo, calendar)
        self.assertEqual(out_derived.shape, (2, 3, 48, 32, 36))

        # 4. Thorough test of trigonometric preceding-day derivation and wraparound at year boundaries
        cal_proj = CalendarProjection(derive_preceding_days=True)

        # Test at DOY = 1 (start of year): preceding 2 days should be DOY 364 and DOY 365
        alpha = 2.0 * math.pi / 365.0
        doy1_sincos = torch.tensor([[math.sin(1.0 * alpha), math.cos(1.0 * alpha)]])  # [1, 2]
        derived = cal_proj._derive_preceding_sincos(doy1_sincos, seq_len=3)  # [1, 3, 2]

        # Day t=2 (index 2) must match DOY 1
        torch.testing.assert_close(derived[0, 2], doy1_sincos[0], atol=1e-6, rtol=1e-5)

        # Day t=1 (index 1) must match DOY 365 (or DOY 0 on circle)
        expected_doy365 = torch.tensor([math.sin(365.0 * alpha), math.cos(365.0 * alpha)])
        torch.testing.assert_close(derived[0, 1], expected_doy365, atol=1e-6, rtol=1e-5)

        # Day t=0 (index 0) must match DOY 364
        expected_doy364 = torch.tensor([math.sin(364.0 * alpha), math.cos(364.0 * alpha)])
        torch.testing.assert_close(derived[0, 0], expected_doy364, atol=1e-6, rtol=1e-5)

        # Verify unit circle radius condition: sin^2 + cos^2 == 1 for all derived days
        radii_sq = derived[:, :, 0] ** 2 + derived[:, :, 1] ** 2
        torch.testing.assert_close(radii_sq, torch.ones_like(radii_sq), atol=1e-6, rtol=1e-5)

    def test_11_activation_option(self) -> None:
        """Verify that optional activation function works as expected."""
        encoder_silu = ClimateInputEncoder(activation=torch.nn.SiLU()).to(self.device)
        climate, geo, calendar = self._generate_synthetic_inputs(b=2, t=3, h=32, w=36)
        out = encoder_silu(climate, geo, calendar)
        self.assertEqual(out.shape, (2, 3, 48, 32, 36))
        self.assertTrue(torch.isfinite(out).all().item())

    def test_12_nan_and_inf_inputs(self) -> None:
        """Test 12: Proper detection and handling of NaN and Inf inputs."""
        encoder = ClimateInputEncoder().to(self.device)
        climate, geo, calendar = self._generate_synthetic_inputs(b=2, t=3, h=32, w=36)

        # 1. NaN in dynamic climate input
        climate_nan = climate.clone()
        climate_nan[0, 0, 0, 10, 10] = float("nan")

        # When check_finite=True, must raise ValueError
        with self.assertRaises(ValueError):
            encoder(climate_nan, geo, calendar, check_finite=True)

        # When check_finite=False, NaN propagates without crashing
        out_nan = encoder(climate_nan, geo, calendar, check_finite=False)
        self.assertTrue(torch.isnan(out_nan).any().item())

        # 2. Inf in dynamic climate input
        climate_inf = climate.clone()
        climate_inf[0, 0, 0, 10, 10] = float("inf")
        with self.assertRaises(ValueError):
            encoder(climate_inf, geo, calendar, check_finite=True)

        # 3. NaN in static geographic input
        geo_nan = geo.clone()
        geo_nan[0, 0, 5, 5] = float("nan")
        with self.assertRaises(ValueError):
            encoder(climate, geo_nan, calendar, check_finite=True)

        # 4. NaN in calendar input
        cal_nan = calendar.clone()
        cal_nan[0, 0] = float("nan")
        with self.assertRaises(ValueError):
            encoder(climate, geo, cal_nan, check_finite=True)

    def test_13_missing_inputs(self) -> None:
        """Test 13: Proper exception handling when required inputs are missing or None."""
        encoder = ClimateInputEncoder().to(self.device)
        climate, geo, calendar = self._generate_synthetic_inputs(b=2, t=3, h=32, w=36)

        with self.assertRaises(TypeError):
            encoder(None, geo, calendar)
        with self.assertRaises(TypeError):
            encoder(climate, None, calendar)
        with self.assertRaises(TypeError):
            encoder(climate, geo, None)

    def test_14_deterministic_behavior(self) -> None:
        """Test 14: Verifies deterministic output across multiple invocations."""
        encoder = ClimateInputEncoder().to(self.device)
        encoder.eval()
        climate, geo, calendar = self._generate_synthetic_inputs(b=2, t=3, h=32, w=36)

        with torch.no_grad():
            out1 = encoder(climate, geo, calendar)
            out2 = encoder(climate, geo, calendar)

        torch.testing.assert_close(out1, out2, msg="Encoder forward pass must be deterministic")

    def test_15_flexible_broadcasting(self) -> None:
        """Test 15: Auto-broadcasting of static geography and calendar across batch dimensions."""
        encoder = ClimateInputEncoder().to(self.device)

        # Batch size 4 climate sequence
        climate = torch.randn(4, 3, 6, 32, 36, device=self.device)

        # Static geography with batch dimension 1
        geo_single = torch.randn(1, 2, 32, 36, device=self.device)

        # Calendar with batch dimension 1
        cal_single = torch.tensor([[0.5, 0.866]], device=self.device)

        out = encoder(climate, geo_single, cal_single)
        self.assertEqual(out.shape, (4, 3, 48, 32, 36))

        # Static features for all batch items should be identical
        geo_features = out[:, :, 32:40, :, :]
        torch.testing.assert_close(geo_features[0], geo_features[1])
        torch.testing.assert_close(geo_features[0], geo_features[2])
        torch.testing.assert_close(geo_features[0], geo_features[3])


if __name__ == "__main__":
    unittest.main()
