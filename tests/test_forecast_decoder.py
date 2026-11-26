"""Unit and integration tests for Block 5: Forecast Decoder.

Tests:
- Test A: Basic shape [1, 64, 32, 36] -> [1, 6, 32, 36]
- Test B: Batch flexibility (B=1, B=4, B=8)
- Test C: Spatial preservation (H=32, W=36 and arbitrary spatial dimensions)
- Test D: Layer correctness (Layer 1: 64->64, Layer 2: 64->32, Layer 3: 32->6; spatial dimensions preserved)
- Test E: Channel order (t2m, tp, u10, v10, msl, tcwv) matching R1 contract
- Test F: Finite outputs (no NaN or Inf across multiple input distributions)
- Test G: Gradient flow (gradients propagate through all decoder layers and reach input tensor)
- Test H: CPU compatibility (operates strictly on CPU without CUDA)
- Test I: Input validation (channel count, dimensions, small spatial grids, empty inputs, non-tensors)
- Test J: Integration with actual Block 4 (SpatialContextAttention -> ForecastDecoder)
- Test K: Full pipeline integration (Block 1 -> Block 2 -> Block 3 -> Block 4 -> Block 5)
- Test L: Regression tests verifying Blocks 1-4 continue to pass all checks
"""

from __future__ import annotations

import math
import unittest
import torch
import torch.nn as nn

from monsoon_r1.contract import CHANNELS, NC, NLAT, NLON
from monsoon_r1.models.climate_input_encoder import ClimateInputEncoder
from monsoon_r1.models.multi_scale_spatial_encoder import MultiScaleSpatialEncoder
from monsoon_r1.models.temporal_dynamics_encoder import (
    ConvLSTM,
    TemporalDynamicsEncoder,
)
from monsoon_r1.models.spatial_context_attention import SpatialContextAttention
from monsoon_r1.models.forecast_decoder import (
    Block5ForecastDecoder,
    ClimateForecastDecoder,
    ForecastDecoder,
)


class TestForecastDecoder(unittest.TestCase):
    """Comprehensive test suite for Block 5: ForecastDecoder."""

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        torch.manual_seed(42)

    def _sample_block4_output(
        self,
        b: int = 1,
        c: int = 64,
        h: int = NLAT,
        w: int = NLON,
    ) -> torch.Tensor:
        """Create sample 64-channel feature map matching Block 4 output."""
        return torch.randn(b, c, h, w, device=self.device)

    def test_a_basic_shape(self) -> None:
        """Test A: Input [1, 64, 32, 36] must produce [1, 6, 32, 36]."""
        decoder = ForecastDecoder().to(self.device)
        x = self._sample_block4_output(b=1, c=64, h=32, w=36)

        out = decoder(x)

        expected_shape = (1, 6, 32, 36)
        self.assertEqual(out.shape, expected_shape)

        # Test aliases produce identical output
        alias1 = ClimateForecastDecoder().to(self.device)
        alias1.load_state_dict(decoder.state_dict())
        out_alias1 = alias1(x)
        torch.testing.assert_close(out, out_alias1)

        alias2 = Block5ForecastDecoder().to(self.device)
        alias2.load_state_dict(decoder.state_dict())
        out_alias2 = alias2(x)
        torch.testing.assert_close(out, out_alias2)

    def test_b_batch_flexibility(self) -> None:
        """Test B: Verify output batch dimension is preserved for B=1, B=2, B=4, and B=8."""
        decoder = ForecastDecoder().to(self.device)

        for b in [1, 2, 4, 8]:
            x = self._sample_block4_output(b=b, c=64, h=32, w=36)
            out = decoder(x)
            self.assertEqual(out.shape, (b, 6, 32, 36), f"Failed for batch size B={b}")

    def test_c_spatial_preservation(self) -> None:
        """Test C: Verify height and width are strictly preserved."""
        decoder = ForecastDecoder().to(self.device)

        # Standard grid size (32 x 36)
        x_std = self._sample_block4_output(b=2, c=64, h=32, w=36)
        out_std = decoder(x_std)
        self.assertEqual(out_std.shape[2], 32, "Height dimension 32 must be preserved")
        self.assertEqual(out_std.shape[3], 36, "Width dimension 36 must be preserved")

        # Non-standard arbitrary spatial dimensions
        for h, w in [(16, 16), (20, 24), (40, 48)]:
            x_arb = self._sample_block4_output(b=2, c=64, h=h, w=w)
            out_arb = decoder(x_arb)
            self.assertEqual(out_arb.shape, (2, 6, h, w), f"Spatial preservation failed for ({h}, {w})")

    def test_d_layer_correctness(self) -> None:
        """Test D: Verify channel dimensions and spatial preservation at each layer."""
        decoder = ForecastDecoder().to(self.device)
        b, h, w = 2, 32, 36
        x = self._sample_block4_output(b=b, c=64, h=h, w=w)

        # 1. Forward with return_intermediates=True
        out, intermediates = decoder(x, return_intermediates=True)

        feat1 = intermediates["layer1_refinement"]
        feat2 = intermediates["layer2_reduction"]
        feat3 = intermediates["layer3_projection"]

        # Layer 1: 64 -> 64 channels, spatial dimensions preserved
        self.assertEqual(feat1.shape, (b, 64, h, w), "Layer 1 must produce [B, 64, H, W]")

        # Layer 2: 64 -> 32 channels, spatial dimensions preserved
        self.assertEqual(feat2.shape, (b, 32, h, w), "Layer 2 must produce [B, 32, H, W]")

        # Layer 3: 32 -> 6 channels, spatial dimensions preserved
        self.assertEqual(feat3.shape, (b, 6, h, w), "Layer 3 must produce [B, 6, H, W]")
        torch.testing.assert_close(out, feat3)

        # 2. Test individual forward methods
        feat1_ind = decoder.forward_refinement(x)
        torch.testing.assert_close(feat1, feat1_ind)

        feat2_ind = decoder.forward_reduction(feat1_ind)
        torch.testing.assert_close(feat2, feat2_ind)

        feat3_ind = decoder.forward_projection(feat2_ind)
        torch.testing.assert_close(feat3, feat3_ind)

        # 3. Check convolution attributes and configurations
        # Layer 1: 3x3 conv, stride 1, padding 1
        l1_conv = decoder.layer1[0]
        self.assertEqual(l1_conv.in_channels, 64)
        self.assertEqual(l1_conv.out_channels, 64)
        self.assertEqual(l1_conv.kernel_size, (3, 3))
        self.assertEqual(l1_conv.stride, (1, 1))
        self.assertEqual(l1_conv.padding, (1, 1))

        # Layer 2: 3x3 conv, stride 1, padding 1
        l2_conv = decoder.layer2[0]
        self.assertEqual(l2_conv.in_channels, 64)
        self.assertEqual(l2_conv.out_channels, 32)
        self.assertEqual(l2_conv.kernel_size, (3, 3))
        self.assertEqual(l2_conv.stride, (1, 1))
        self.assertEqual(l2_conv.padding, (1, 1))

        # Layer 3: 1x1 conv, stride 1, padding 0
        l3_conv = decoder.layer3
        self.assertEqual(l3_conv.in_channels, 32)
        self.assertEqual(l3_conv.out_channels, 6)
        self.assertEqual(l3_conv.kernel_size, (1, 1))
        self.assertEqual(l3_conv.stride, (1, 1))
        self.assertEqual(l3_conv.padding, (0, 0))

    def test_e_channel_order(self) -> None:
        """Test E: Verify that output channels map strictly to the 6 target climate variables."""
        expected_channels = ("t2m", "tp", "u10", "v10", "msl", "tcwv")

        # Check against contract and class attributes
        self.assertEqual(ForecastDecoder.CHANNELS, expected_channels)
        self.assertEqual(ForecastDecoder.TARGET_VARIABLES, expected_channels)
        self.assertEqual(CHANNELS, expected_channels)
        self.assertEqual(ForecastDecoder.NUM_CHANNELS, 6)

        decoder = ForecastDecoder().to(self.device)
        self.assertEqual(decoder.out_channels, 6)

        x = self._sample_block4_output(b=1, c=64, h=32, w=36)
        out = decoder(x)

        # Verify indexing per variable matches specification
        for idx, var_name in enumerate(expected_channels):
            var_slice = out[:, idx, :, :]
            self.assertEqual(var_slice.shape, (1, 32, 36), f"Channel {idx} ({var_name}) shape mismatch")

    def test_f_finite_outputs(self) -> None:
        """Test F: Verify output contains no NaN or Inf across multiple input distributions."""
        decoder = ForecastDecoder().to(self.device)

        test_inputs = [
            self._sample_block4_output(b=2, c=64, h=32, w=36),
            torch.zeros(2, 64, 32, 36, device=self.device),
            torch.ones(2, 64, 32, 36, device=self.device) * 100.0,
            torch.ones(2, 64, 32, 36, device=self.device) * -50.0,
            torch.randn(2, 64, 32, 36, device=self.device) * 10.0,
        ]

        for i, x in enumerate(test_inputs):
            out = decoder(x)
            self.assertFalse(torch.isnan(out).any().item(), f"Input {i} produced NaN")
            self.assertFalse(torch.isinf(out).any().item(), f"Input {i} produced Inf")

    def test_g_gradient_flow(self) -> None:
        """Test G: Verify training forward/backward pass propagates gradients through all layers."""
        decoder = ForecastDecoder().to(self.device)
        x = self._sample_block4_output(b=2, c=64, h=32, w=36)
        x.requires_grad_(True)

        out = decoder(x)

        # Loss against simulated target delta
        target_delta = torch.randn_like(out)
        loss = nn.functional.mse_loss(out, target_delta)
        loss.backward()

        # Check gradients reach input tensor
        self.assertIsNotNone(x.grad, "Gradients must propagate back to input tensor x")
        self.assertTrue(torch.isfinite(x.grad).all().item(), "Input gradients must be finite")
        self.assertGreater(x.grad.abs().sum().item(), 0.0, "Input gradients must be non-zero")

        # Check gradients reach Layer 1 parameters
        self.assertIsNotNone(decoder.layer1[0].weight.grad, "Layer 1 weight grad must exist")
        self.assertIsNotNone(decoder.layer1[0].bias.grad, "Layer 1 bias grad must exist")
        self.assertTrue(torch.isfinite(decoder.layer1[0].weight.grad).all().item())

        # Check gradients reach Layer 2 parameters
        self.assertIsNotNone(decoder.layer2[0].weight.grad, "Layer 2 weight grad must exist")
        self.assertIsNotNone(decoder.layer2[0].bias.grad, "Layer 2 bias grad must exist")
        self.assertTrue(torch.isfinite(decoder.layer2[0].weight.grad).all().item())

        # Check gradients reach Layer 3 parameters
        self.assertIsNotNone(decoder.layer3.weight.grad, "Layer 3 weight grad must exist")
        self.assertIsNotNone(decoder.layer3.bias.grad, "Layer 3 bias grad must exist")
        self.assertTrue(torch.isfinite(decoder.layer3.weight.grad).all().item())

    def test_h_cpu_compatibility(self) -> None:
        """Test H: Ensure the decoder runs natively on CPU without requiring CUDA."""
        decoder = ForecastDecoder()
        # Verify all parameters reside on CPU
        for param in decoder.parameters():
            self.assertEqual(param.device.type, "cpu")

        x = torch.randn(1, 64, 32, 36, device="cpu")
        out = decoder(x)
        self.assertEqual(out.device.type, "cpu")
        self.assertEqual(out.shape, (1, 6, 32, 36))

    def test_i_input_validation(self) -> None:
        """Test I: Robust input validation and informative error messages."""
        decoder = ForecastDecoder().to(self.device)

        # Non-tensor input
        with self.assertRaises(TypeError):
            decoder("invalid_input")  # type: ignore

        with self.assertRaises(TypeError):
            decoder([1, 2, 3])  # type: ignore

        # Incorrect dimensionality (3D instead of 4D)
        with self.assertRaises(ValueError) as ctx:
            decoder(torch.randn(64, 32, 36))
        self.assertIn("Expected 4D tensor", str(ctx.exception))

        # Incorrect dimensionality (5D instead of 4D)
        with self.assertRaises(ValueError) as ctx:
            decoder(torch.randn(1, 3, 64, 32, 36))
        self.assertIn("Expected 4D tensor", str(ctx.exception))

        # Channel count mismatch (32 or 48 instead of 64)
        with self.assertRaises(ValueError) as ctx:
            decoder(torch.randn(1, 32, 32, 36))
        self.assertIn("Expected 64 channels", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            decoder(torch.randn(1, 48, 32, 36))
        self.assertIn("Expected 64 channels", str(ctx.exception))

        # Empty batch size (B=0)
        with self.assertRaises(ValueError) as ctx:
            decoder(torch.randn(0, 64, 32, 36))
        self.assertIn("Batch size B must be >= 1", str(ctx.exception))

        # Spatial dimensions too small (< 3x3)
        with self.assertRaises(ValueError) as ctx:
            decoder(torch.randn(1, 64, 2, 36))
        self.assertIn("at least 3x3", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            decoder(torch.randn(1, 64, 32, 2))
        self.assertIn("at least 3x3", str(ctx.exception))

        # Validation on constructor arguments
        with self.assertRaises(ValueError):
            ForecastDecoder(in_channels=0)
        with self.assertRaises(ValueError):
            ForecastDecoder(hidden_channels=-1)
        with self.assertRaises(ValueError):
            ForecastDecoder(reduced_channels=0)
        with self.assertRaises(ValueError):
            ForecastDecoder(out_channels=-6)

    def test_j_integration_with_block_4(self) -> None:
        """Test J: Integration with actual Block 4 implementation."""
        block4 = SpatialContextAttention().to(self.device)
        block5 = ForecastDecoder().to(self.device)

        b, c, h, w = 2, 64, 32, 36

        # Input to Block 4: [B, 64, 32, 36]
        block4_in = torch.randn(b, c, h, w, device=self.device, requires_grad=True)

        # Run Block 4
        block4_out = block4(block4_in)
        self.assertEqual(block4_out.shape, (b, 64, h, w), "Block 4 must output [B, 64, 32, 36]")

        # Pass Block 4 output directly into Block 5
        block5_out = block5(block4_out)
        self.assertEqual(block5_out.shape, (b, 6, h, w), "Block 5 must output [B, 6, 32, 36]")

        # Backward pass across Block 5 and Block 4
        loss = block5_out.sum()
        loss.backward()

        # Check gradients reach Block 5 layers
        self.assertIsNotNone(block5.layer3.weight.grad, "Gradients must reach Block 5 layer 3")
        self.assertIsNotNone(block5.layer2[0].weight.grad, "Gradients must reach Block 5 layer 2")
        self.assertIsNotNone(block5.layer1[0].weight.grad, "Gradients must reach Block 5 layer 1")

        # Check gradients reach Block 4 attention components
        self.assertIsNotNone(block4.spatial_attention.conv.weight.grad, "Gradients must reach Block 4 spatial conv")
        self.assertIsNotNone(block4.channel_attention.mlp[0].weight.grad, "Gradients must reach Block 4 channel MLP")

        # Check gradients reach the input tensor to Block 4
        self.assertIsNotNone(block4_in.grad, "Gradients must reach Block 4 input")

    def test_k_full_pipeline_integration(self) -> None:
        """Test K: Full pipeline integration across Block 1 -> Block 2 -> Block 3 -> Block 4 -> Block 5."""
        block1 = ClimateInputEncoder().to(self.device)
        block2 = MultiScaleSpatialEncoder().to(self.device)
        block3 = TemporalDynamicsEncoder().to(self.device)
        block4 = SpatialContextAttention().to(self.device)
        block5 = ForecastDecoder().to(self.device)

        b, t, h, w = 2, 3, 32, 36

        # 1. Realistic raw inputs to Block 1
        climate = torch.randn(b, t, 6, h, w, device=self.device, requires_grad=True)
        # Shift to realistic scales
        climate_scaled = climate * 5.0 + 300.0

        geo = torch.zeros(b, 2, h, w, device=self.device)
        geo[:, 0] = torch.rand(b, h, w)  # land_sea_mask
        geo[:, 1] = torch.rand(b, h, w) * 2000.0  # orography_m

        doy = 180.0
        angle = 2.0 * math.pi * doy / 365.0
        calendar = torch.tensor([[math.sin(angle), math.cos(angle)]] * b, device=self.device)

        # 2. Block 1: [B, T, 6, 32, 36] -> [B, T, 48, 32, 36]
        block1_out = block1(climate_scaled, geo, calendar)
        self.assertEqual(block1_out.shape, (b, t, 48, h, w), "Block 1 must output [B, T, 48, 32, 36]")

        # 3. Block 2: [B, T, 48, 32, 36] -> [B, T, 64, 32, 36]
        block2_out = block2(block1_out)
        self.assertEqual(block2_out.shape, (b, t, 64, h, w), "Block 2 must output [B, T, 64, 32, 36]")

        # 4. Block 3: [B, T, 64, 32, 36] -> [B, 64, 32, 36]
        block3_out = block3(block2_out)
        self.assertEqual(block3_out.shape, (b, 64, h, w), "Block 3 must output [B, 64, 32, 36]")

        # 5. Block 4: [B, 64, 32, 36] -> [B, 64, 32, 36]
        block4_out = block4(block3_out)
        self.assertEqual(block4_out.shape, (b, 64, h, w), "Block 4 must output [B, 64, 32, 36]")

        # 6. Block 5: [B, 64, 32, 36] -> [B, 6, 32, 36]
        block5_out = block5(block4_out)
        self.assertEqual(block5_out.shape, (b, 6, h, w), "Block 5 must output [B, 6, 32, 36]")

        # 7. End-to-end backward pass through all five blocks
        loss = block5_out.mean()
        loss.backward()

        # Verify gradients reach all blocks from Block 5 down to Block 1
        self.assertIsNotNone(block5.layer3.weight.grad, "Gradients must reach Block 5 layer 3")
        self.assertIsNotNone(block5.layer2[0].weight.grad, "Gradients must reach Block 5 layer 2")
        self.assertIsNotNone(block5.layer1[0].weight.grad, "Gradients must reach Block 5 layer 1")
        self.assertIsNotNone(block4.spatial_attention.conv.weight.grad, "Gradients must reach Block 4 spatial conv")
        self.assertIsNotNone(block4.channel_attention.mlp[0].weight.grad, "Gradients must reach Block 4 channel MLP")
        self.assertIsNotNone(block3.cell.conv_ih.weight.grad, "Gradients must reach Block 3 conv_ih")
        self.assertIsNotNone(block3.cell.conv_hh.weight.grad, "Gradients must reach Block 3 conv_hh")
        self.assertIsNotNone(block2.fusion[0].weight.grad, "Gradients must reach Block 2 fusion")
        self.assertIsNotNone(block1.climate_proj.conv.weight.grad, "Gradients must reach Block 1 climate conv")
        self.assertIsNotNone(climate.grad, "Gradients must reach raw input climate tensor")

    def test_l_regression_testing_blocks_1_to_4(self) -> None:
        """Test L: Regression check confirming Blocks 1, 2, 3, and 4 continue to behave identically."""
        block1 = ClimateInputEncoder().to(self.device)
        block2 = MultiScaleSpatialEncoder().to(self.device)
        block3 = ConvLSTM().to(self.device)
        block4 = SpatialContextAttention().to(self.device)

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

    def test_m_nan_and_inf_detection(self) -> None:
        """Test M: Proper detection and handling of NaN and Inf in Block 5."""
        decoder = ForecastDecoder().to(self.device)
        x = self._sample_block4_output(b=2, c=64, h=32, w=36)

        # 1. NaN in input with check_finite=True
        x_nan = x.clone()
        x_nan[0, 10, 5, 5] = float("nan")
        with self.assertRaises(ValueError):
            decoder(x_nan, check_finite=True)

        # NaN propagation with check_finite=False
        out_nan = decoder(x_nan, check_finite=False)
        self.assertTrue(torch.isnan(out_nan).any().item())

        # 2. Inf in input with check_finite=True
        x_inf = x.clone()
        x_inf[0, 10, 5, 5] = float("inf")
        with self.assertRaises(ValueError):
            decoder(x_inf, check_finite=True)

    def test_n_missing_inputs(self) -> None:
        """Test N: Handling of None / missing inputs."""
        decoder = ForecastDecoder().to(self.device)
        with self.assertRaises(TypeError):
            decoder(None)

    def test_o_deterministic_behavior(self) -> None:
        """Test O: Forward pass is deterministic across multiple calls."""
        decoder = ForecastDecoder().to(self.device)
        decoder.eval()
        x = self._sample_block4_output(b=2, c=64, h=32, w=36)

        with torch.no_grad():
            out1 = decoder(x)
            out2 = decoder(x)

        torch.testing.assert_close(out1, out2)

    def test_p_delta_representation_consistency(self) -> None:
        """Test P: Verify Layer 3 is linear and can predict both positive and negative deltas."""
        decoder = ForecastDecoder().to(self.device)
        x = self._sample_block4_output(b=2, c=64, h=32, w=36)

        out = decoder(x)
        # Check that output has shape [B, 6, H, W]
        self.assertEqual(out.shape, (2, 6, 32, 36))
        # Layer 3 should be nn.Conv2d without bounding activation
        self.assertIsInstance(decoder.layer3, nn.Conv2d)
        # Verify layer3 is indeed linear
        feat2 = decoder.layer2(decoder.layer1(x))
        out_proj = decoder.layer3(feat2)
        torch.testing.assert_close(out, out_proj)


if __name__ == "__main__":
    unittest.main()
