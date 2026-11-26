"""Unit and integration tests for Block 2: MultiScaleSpatialEncoder.

Tests:
- Test A: Basic shape [1, 3, 48, 32, 36] -> [1, 3, 64, 32, 36]
- Test B: Batch flexibility (B=1, B=4)
- Test C: Sequence flexibility (T=1, T=3, T=5)
- Test D: Spatial preservation (H=32, W=36)
- Test E: Branch correctness (16 channels each, 48 concat, 64 fusion)
- Test F: Finite outputs (no NaN / Inf)
- Test G: Gradient flow (gradients reach all three branches and fusion)
- Test H: CPU compatibility (all ops on CPU)
- Test I: Input validation (channels, dimensions, types)
- Test J: Integration with actual Block 1 (ClimateInputEncoder -> MultiScaleSpatialEncoder)
"""

from __future__ import annotations

import math
import unittest
try:
    import pytest
except ImportError:
    pytest = None
import torch

from monsoon_r1.models.climate_input_encoder import ClimateInputEncoder
from monsoon_r1.models.multi_scale_spatial_encoder import MultiScaleSpatialEncoder


class TestMultiScaleSpatialEncoder(unittest.TestCase):
    """Test suite for Block 2: MultiScaleSpatialEncoder."""

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        torch.manual_seed(42)

    def _sample_block1_output(
        self,
        b: int = 1,
        t: int = 3,
        c: int = 48,
        h: int = 32,
        w: int = 36,
    ) -> torch.Tensor:
        """Create sample 48-channel tensor matching Block 1 output."""
        return torch.randn(b, t, c, h, w, device=self.device)

    def test_a_basic_shape(self) -> None:
        """Test A: Input [1, 3, 48, 32, 36] must return [1, 3, 64, 32, 36]."""
        encoder = MultiScaleSpatialEncoder().to(self.device)
        x = self._sample_block1_output(b=1, t=3, c=48, h=32, w=36)

        out = encoder(x)

        expected_shape = (1, 3, 64, 32, 36)
        self.assertEqual(out.shape, expected_shape)

    def test_b_batch_flexibility(self) -> None:
        """Test B: Test batch sizes B=1, B=2, B=4, and B=8; batch dimension must remain unchanged."""
        encoder = MultiScaleSpatialEncoder().to(self.device)

        for b in [1, 2, 4, 8]:
            x = self._sample_block1_output(b=b, t=3, c=48, h=32, w=36)
            out = encoder(x)
            self.assertEqual(out.shape, (b, 3, 64, 32, 36), f"Failed for batch size B={b}")

    def test_c_sequence_flexibility(self) -> None:
        """Test C: Test sequence lengths T=1, T=3, T=5; sequence dimension must remain unchanged."""
        encoder = MultiScaleSpatialEncoder().to(self.device)

        for t in [1, 3, 5, 7]:
            x = self._sample_block1_output(b=2, t=t, c=48, h=32, w=36)
            out = encoder(x)
            self.assertEqual(out.shape, (2, t, 64, 32, 36), f"Failed for sequence length T={t}")

    def test_d_spatial_preservation(self) -> None:
        """Test D: Spatial dimensions Height=32 and Width=36 must be preserved."""
        encoder = MultiScaleSpatialEncoder().to(self.device)
        x = self._sample_block1_output(b=2, t=3, c=48, h=32, w=36)

        out = encoder(x)

        self.assertEqual(out.size(-2), 32, "Height must be 32")
        self.assertEqual(out.size(-1), 36, "Width must be 36")

        # Also test with alternative valid spatial grid
        x_alt = self._sample_block1_output(b=1, t=2, c=48, h=16, w=20)
        out_alt = encoder(x_alt)
        self.assertEqual(out_alt.shape, (1, 2, 64, 16, 20))

    def test_e_branch_correctness(self) -> None:
        """Test E: Verify each branch returns 16 channels, preserves spatial dims,

        concatenation has 48 channels, and fusion produces 64 channels.
        """
        encoder = MultiScaleSpatialEncoder().to(self.device)
        b, t, h, w = 2, 3, 32, 36
        x = self._sample_block1_output(b=b, t=t, c=48, h=h, w=w)

        # 1. Branch A: 3x3 local convolution
        feat_a = encoder.forward_branch_a(x)
        self.assertEqual(feat_a.shape, (b, t, 16, h, w), "Branch A must produce 16 channels")

        # 2. Branch B: 5x5 broader convolution
        feat_b = encoder.forward_branch_b(x)
        self.assertEqual(feat_b.shape, (b, t, 16, h, w), "Branch B must produce 16 channels")

        # 3. Branch C: 3x3 dilated (d=2) convolution
        feat_c = encoder.forward_branch_c(x)
        self.assertEqual(feat_c.shape, (b, t, 16, h, w), "Branch C must produce 16 channels")

        # 4. Multi-scale concatenation: 16 + 16 + 16 = 48 channels
        feat_cat = torch.cat([feat_a, feat_b, feat_c], dim=2)
        self.assertEqual(feat_cat.shape, (b, t, 48, h, w), "Concatenated features must have 48 channels")

        # 5. Fusion: 48 -> 64 channels
        fused = encoder.fuse(feat_cat)
        self.assertEqual(fused.shape, (b, t, 64, h, w), "Fusion must produce 64 channels")

        # Direct forward pass must match step-by-step
        out_direct = encoder(x)
        torch.testing.assert_close(out_direct, fused)

    def test_f_finite_outputs(self) -> None:
        """Test F: Verify output contains no NaN or Inf for valid inputs."""
        encoder = MultiScaleSpatialEncoder().to(self.device)
        x = self._sample_block1_output(b=4, t=3, c=48, h=32, w=36)

        out = encoder(x)

        self.assertTrue(torch.isfinite(out).all().item(), "Output must contain only finite numbers")
        self.assertFalse(torch.isnan(out).any().item(), "Output must not contain NaN")
        self.assertFalse(torch.isinf(out).any().item(), "Output must not contain Inf")

    def test_g_gradient_flow(self) -> None:
        """Test G: Verify training backward pass propagates gradients through all three branches and fusion layer."""
        encoder = MultiScaleSpatialEncoder().to(self.device)
        x = self._sample_block1_output(b=2, t=3, c=48, h=32, w=36)
        x.requires_grad_(True)

        out = encoder(x)
        loss = out.sum()
        loss.backward()

        # Check input gradients
        self.assertIsNotNone(x.grad, "Gradients must reach input tensor")
        self.assertFalse(torch.all(x.grad == 0), "Input gradients must not be all zero")

        # Branch A conv weights and biases
        conv_a = encoder.branch_a[0]
        self.assertIsNotNone(conv_a.weight.grad, "Gradients must reach Branch A conv weights")
        self.assertIsNotNone(conv_a.bias.grad, "Gradients must reach Branch A conv bias")
        self.assertFalse(torch.all(conv_a.weight.grad == 0))

        # Branch B conv weights and biases
        conv_b = encoder.branch_b[0]
        self.assertIsNotNone(conv_b.weight.grad, "Gradients must reach Branch B conv weights")
        self.assertIsNotNone(conv_b.bias.grad, "Gradients must reach Branch B conv bias")
        self.assertFalse(torch.all(conv_b.weight.grad == 0))

        # Branch C conv weights and biases
        conv_c = encoder.branch_c[0]
        self.assertIsNotNone(conv_c.weight.grad, "Gradients must reach Branch C conv weights")
        self.assertIsNotNone(conv_c.bias.grad, "Gradients must reach Branch C conv bias")
        self.assertFalse(torch.all(conv_c.weight.grad == 0))

        # Fusion conv weights and biases
        conv_fuse = encoder.fusion[0]
        self.assertIsNotNone(conv_fuse.weight.grad, "Gradients must reach Fusion conv weights")
        self.assertIsNotNone(conv_fuse.bias.grad, "Gradients must reach Fusion conv bias")
        self.assertFalse(torch.all(conv_fuse.weight.grad == 0))

    def test_h_cpu_compatibility(self) -> None:
        """Test H: Ensure module and all operations execute reliably on CPU without CUDA."""
        encoder = MultiScaleSpatialEncoder().to("cpu")
        x = self._sample_block1_output(b=2, t=3, c=48, h=32, w=36)

        self.assertEqual(x.device.type, "cpu")
        out = encoder(x)
        self.assertEqual(out.device.type, "cpu")

    def test_i_input_validation(self) -> None:
        """Test I: Test reasonable handling of incorrect channels, dimensions, and types."""
        encoder = MultiScaleSpatialEncoder().to(self.device)
        x = self._sample_block1_output(b=2, t=3, c=48, h=32, w=36)

        # 1. Non-tensor input
        with self.assertRaises(TypeError):
            encoder("invalid_string_input")

        # 2. Invalid dimensions (4D tensor instead of 5D [B, T, C, H, W])
        with self.assertRaises(ValueError):
            encoder(x[:, 0])

        # 3. Invalid channel count (e.g. 32 or 47 instead of 48)
        with self.assertRaises(ValueError):
            encoder(x[:, :, :32])
        with self.assertRaises(ValueError):
            encoder(x[:, :, :47])

        # 4. Tiny spatial grid (< 5x5) where 5x5 or dilated 3x3 cannot operate properly
        x_tiny = torch.randn(1, 1, 48, 3, 3)
        with self.assertRaises(ValueError):
            encoder(x_tiny)

        # 5. Invalid fuse input
        with self.assertRaises(ValueError):
            encoder.fuse(x[:, :, :32])

    def test_j_integration_with_block_1(self) -> None:
        """Test J: End-to-end integration test with the actual Block 1 implementation."""
        block1 = ClimateInputEncoder().to(self.device)
        block2 = MultiScaleSpatialEncoder().to(self.device)

        b, t, h, w = 3, 3, 32, 36

        # 1. Realistic raw inputs to Block 1
        climate = torch.randn(b, t, 6, h, w, device=self.device)
        climate[:, :, 0] = climate[:, :, 0] * 10.0 + 300.0  # t2m (K)
        climate[:, :, 1] = torch.clamp(climate[:, :, 1] * 5.0 + 5.0, min=0.0)  # tp (mm/day)
        climate[:, :, 4] = climate[:, :, 4] * 1000.0 + 101325.0  # msl (Pa)

        geo = torch.zeros(b, 2, h, w, device=self.device)
        geo[:, 0] = torch.rand(b, h, w)  # land_sea_mask
        geo[:, 1] = torch.rand(b, h, w) * 2000.0  # orography_m

        doy = 200.0
        angle = 2.0 * math.pi * doy / 365.0
        calendar = torch.tensor([[math.sin(angle), math.cos(angle)]] * b, device=self.device)

        # 2. Block 1 forward pass
        block1_out = block1(climate, geo, calendar)
        self.assertEqual(block1_out.shape, (b, t, 48, h, w), "Block 1 must output [B, T, 48, 32, 36]")

        # 3. Block 2 forward pass
        block2_out = block2(block1_out)
        self.assertEqual(block2_out.shape, (b, t, 64, h, w), "Block 2 must output [B, T, 64, 32, 36]")

        # 4. End-to-end backward pass through both blocks
        loss = block2_out.mean()
        loss.backward()

        # Gradients must propagate all the way through Block 2 back into Block 1 parameters
        self.assertIsNotNone(block2.fusion[0].weight.grad)
        self.assertIsNotNone(block2.branch_a[0].weight.grad)
        self.assertIsNotNone(block1.climate_proj.conv.weight.grad)
        self.assertIsNotNone(block1.geo_proj.conv.weight.grad)
        self.assertIsNotNone(block1.cal_proj.linear.weight.grad)

    def test_k_nan_and_inf_detection(self) -> None:
        """Test K: Proper detection and handling of NaN and Inf inputs in Block 2."""
        encoder = MultiScaleSpatialEncoder().to(self.device)
        x = self._sample_block1_output(b=2, t=3, c=48, h=32, w=36)

        # 1. NaN detection with check_finite=True
        x_nan = x.clone()
        x_nan[0, 0, 0, 10, 10] = float("nan")
        with self.assertRaises(ValueError):
            encoder(x_nan, check_finite=True)

        # NaN propagation with check_finite=False
        out_nan = encoder(x_nan, check_finite=False)
        self.assertTrue(torch.isnan(out_nan).any().item())

        # 2. Inf detection with check_finite=True
        x_inf = x.clone()
        x_inf[0, 0, 0, 10, 10] = float("inf")
        with self.assertRaises(ValueError):
            encoder(x_inf, check_finite=True)

    def test_l_missing_inputs(self) -> None:
        """Test L: Handling of None / missing input."""
        encoder = MultiScaleSpatialEncoder().to(self.device)
        with self.assertRaises(TypeError):
            encoder(None)

    def test_m_deterministic_behavior(self) -> None:
        """Test M: Forward pass is deterministic across multiple calls."""
        encoder = MultiScaleSpatialEncoder().to(self.device)
        encoder.eval()
        x = self._sample_block1_output(b=2, t=3, c=48, h=32, w=36)

        with torch.no_grad():
            out1 = encoder(x)
            out2 = encoder(x)

        torch.testing.assert_close(out1, out2)

    def test_n_branch_dilated_spatial_receptive_field(self) -> None:
        """Test N: Dilated Branch C preserves exact spatial grid size with dilation=2, padding=2."""
        encoder = MultiScaleSpatialEncoder().to(self.device)
        x = self._sample_block1_output(b=1, t=1, c=48, h=32, w=36)

        # Branch C has dilation 2, kernel 3, padding 2
        feat_c = encoder.forward_branch_c(x)
        self.assertEqual(feat_c.shape, (1, 1, 16, 32, 36))
        self.assertTrue(torch.isfinite(feat_c).all())


if __name__ == "__main__":
    unittest.main()
