"""Unit and integration tests for Block 4: Spatial Context & Attention Module.

Tests:
- Test A: Basic shape [1, 64, 32, 36] -> [1, 64, 32, 36]
- Test B: Batch flexibility (B=1, B=4, B=8)
- Test C: Channel attention component verification
  - Average pooling produces a valid channel representation
  - Max pooling produces a valid channel representation
  - Shared MLP processes both representations
  - Channel attention weights have shape [B, 64, 1, 1]
  - Channel attention weights are finite
  - Sigmoid weights fall within [0, 1]
  - Channel reweighting preserves feature dimensions
- Test D: Spatial attention component verification
  - Average pooling across channels produces [B, 1, 32, 36]
  - Max pooling across channels produces [B, 1, 32, 36]
  - Concatenation produces [B, 2, 32, 36]
  - 7x7 convolution preserves spatial dimensions
  - Spatial attention weights have shape [B, 1, 32, 36]
  - Spatial attention weights are finite
  - Sigmoid weights fall within [0, 1]
  - Spatial reweighting preserves feature dimensions
- Test E: Final output shape [B, 64, 32, 36]
- Test F: Finite outputs (no NaN or Inf) across various input regimes
- Test G: Gradient flow (gradients reach Channel Attention, Shared MLP, Spatial Attention conv, input tensor)
- Test H: CPU compatibility (operates strictly on CPU without CUDA)
- Test I: Input validation (channel count, 4D tensor, spatial dimensions, empty inputs, non-tensor)
- Test J: Integration with actual Block 3 (TemporalDynamicsEncoder -> SpatialContextAttention)
- Test K: Full pipeline integration (Block 1 -> Block 2 -> Block 3 -> Block 4)
- Test L: Regression testing verification
"""

from __future__ import annotations

import math
import unittest
import torch

from monsoon_r1.models.climate_input_encoder import ClimateInputEncoder
from monsoon_r1.models.multi_scale_spatial_encoder import MultiScaleSpatialEncoder
from monsoon_r1.models.temporal_dynamics_encoder import (
    ConvLSTM,
    TemporalDynamicsEncoder,
)
from monsoon_r1.models.spatial_context_attention import (
    CBAM,
    ChannelAttention,
    SpatialAttention,
    SpatialContextAttention,
    SpatialContextAttentionModule,
)


class TestSpatialContextAttention(unittest.TestCase):
    """Test suite for Block 4: SpatialContextAttention."""

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        torch.manual_seed(42)

    def _sample_block3_output(
        self,
        b: int = 1,
        c: int = 64,
        h: int = 32,
        w: int = 36,
    ) -> torch.Tensor:
        """Create sample 64-channel tensor matching Block 3 output."""
        return torch.randn(b, c, h, w, device=self.device)

    def test_a_basic_shape(self) -> None:
        """Test A: Input [1, 64, 32, 36] must return [1, 64, 32, 36]."""
        block4 = SpatialContextAttention().to(self.device)
        x = self._sample_block3_output(b=1, c=64, h=32, w=36)

        out = block4(x)

        expected_shape = (1, 64, 32, 36)
        self.assertEqual(out.shape, expected_shape)

        # Also verify aliases
        alias1 = SpatialContextAttentionModule().to(self.device)
        alias1.load_state_dict(block4.state_dict())
        out_alias1 = alias1(x)
        torch.testing.assert_close(out, out_alias1)

        alias2 = CBAM().to(self.device)
        alias2.load_state_dict(block4.state_dict())
        out_alias2 = alias2(x)
        torch.testing.assert_close(out, out_alias2)

    def test_b_batch_flexibility(self) -> None:
        """Test B: Verify batch dimension is preserved for B=1, B=2, B=4, and B=8."""
        block4 = SpatialContextAttention().to(self.device)

        for b in [1, 2, 4, 8]:
            x = self._sample_block3_output(b=b, c=64, h=32, w=36)
            out = block4(x)
            self.assertEqual(out.shape, (b, 64, 32, 36), f"Failed for batch size B={b}")

    def test_c_channel_attention(self) -> None:
        """Test C: Channel attention component verification."""
        ca = ChannelAttention(in_channels=64, reduction_channels=16).to(self.device)
        b, c, h, w = 2, 64, 32, 36
        x = torch.randn(b, c, h, w, device=self.device)

        # 1. Verify spatial global average pooling representation
        avg_rep = ca.avg_pool(x).flatten(1)
        self.assertEqual(avg_rep.shape, (b, 64))
        torch.testing.assert_close(avg_rep, torch.mean(x, dim=(-2, -1)))

        # 2. Verify spatial global max pooling representation
        max_rep = ca.max_pool(x).flatten(1)
        self.assertEqual(max_rep.shape, (b, 64))
        torch.testing.assert_close(max_rep, torch.amax(x, dim=(-2, -1)))

        # 3. Verify shared MLP processes both representations
        mlp_avg = ca.mlp(avg_rep)
        mlp_max = ca.mlp(max_rep)
        self.assertEqual(mlp_avg.shape, (b, 64))
        self.assertEqual(mlp_max.shape, (b, 64))

        # Check bottleneck intermediate dimension (first layer output is 16)
        intermediate = ca.mlp[0](avg_rep)
        self.assertEqual(intermediate.shape, (b, 16))

        # 4. Verify channel attention weights shape
        weights = ca.compute_weights(x)
        self.assertEqual(weights.shape, (b, 64, 1, 1))

        # 5. Verify channel attention weights are finite
        self.assertTrue(torch.isfinite(weights).all())

        # 6. Verify sigmoid weights fall within [0, 1]
        self.assertTrue((weights >= 0.0).all())
        self.assertTrue((weights <= 1.0).all())

        # 7. Verify channel reweighting preserves feature dimensions
        out, w_out = ca(x, return_weights=True)
        self.assertEqual(out.shape, (b, 64, h, w))
        torch.testing.assert_close(w_out, weights)
        torch.testing.assert_close(out, x * weights)

    def test_d_spatial_attention(self) -> None:
        """Test D: Spatial attention component verification."""
        sa = SpatialAttention(kernel_size=7).to(self.device)
        b, c, h, w = 2, 64, 32, 36
        x = torch.randn(b, c, h, w, device=self.device)

        # 1. Average pooling across channels produces [B, 1, 32, 36]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        self.assertEqual(avg_out.shape, (b, 1, h, w))

        # 2. Max pooling across channels produces [B, 1, 32, 36]
        max_out = torch.max(x, dim=1, keepdim=True)[0]
        self.assertEqual(max_out.shape, (b, 1, h, w))

        # 3. Concatenation produces [B, 2, 32, 36]
        concat = torch.cat([avg_out, max_out], dim=1)
        self.assertEqual(concat.shape, (b, 2, h, w))

        # 4. 7x7 convolution preserves spatial dimensions
        conv_out = sa.conv(concat)
        self.assertEqual(conv_out.shape, (b, 1, h, w))

        # 5. Spatial attention weights have shape [B, 1, 32, 36]
        weights = sa.compute_weights(x)
        self.assertEqual(weights.shape, (b, 1, h, w))

        # 6. Spatial attention weights are finite
        self.assertTrue(torch.isfinite(weights).all())

        # 7. Sigmoid weights fall within [0, 1]
        self.assertTrue((weights >= 0.0).all())
        self.assertTrue((weights <= 1.0).all())

        # 8. Spatial reweighting preserves feature dimensions
        out, w_out = sa(x, return_weights=True)
        self.assertEqual(out.shape, (b, c, h, w))
        torch.testing.assert_close(w_out, weights)
        torch.testing.assert_close(out, x * weights)

    def test_e_final_output_shape(self) -> None:
        """Test E: Verify complete attention module returns [B, 64, 32, 36]."""
        block4 = SpatialContextAttention().to(self.device)

        for b in [1, 2, 5]:
            x = self._sample_block3_output(b=b, c=64, h=32, w=36)
            out = block4(x)
            self.assertEqual(out.shape, (b, 64, 32, 36))

            # Test with return_attention_maps=True
            out_with_maps, maps = block4(x, return_attention_maps=True)
            self.assertEqual(out_with_maps.shape, (b, 64, 32, 36))
            torch.testing.assert_close(out, out_with_maps)
            self.assertIn("channel_weights", maps)
            self.assertIn("spatial_weights", maps)
            self.assertEqual(maps["channel_weights"].shape, (b, 64, 1, 1))
            self.assertEqual(maps["spatial_weights"].shape, (b, 1, 32, 36))

    def test_f_finite_outputs(self) -> None:
        """Test F: Verify valid inputs produce outputs containing no NaN or Inf."""
        block4 = SpatialContextAttention().to(self.device)

        # 1. Standard normal inputs
        x_norm = self._sample_block3_output(b=2, c=64, h=32, w=36)
        out_norm = block4(x_norm)
        self.assertTrue(torch.isfinite(out_norm).all())
        self.assertFalse(torch.isnan(out_norm).any())

        # 2. All zeros input
        x_zeros = torch.zeros(2, 64, 32, 36, device=self.device)
        out_zeros = block4(x_zeros)
        self.assertTrue(torch.isfinite(out_zeros).all())
        torch.testing.assert_close(out_zeros, torch.zeros_like(out_zeros))

        # 3. Negative inputs
        x_neg = -torch.abs(torch.randn(2, 64, 32, 36, device=self.device))
        out_neg = block4(x_neg)
        self.assertTrue(torch.isfinite(out_neg).all())

        # 4. Large magnitude inputs
        x_large = torch.randn(2, 64, 32, 36, device=self.device) * 100.0
        out_large = block4(x_large)
        self.assertTrue(torch.isfinite(out_large).all())

    def test_g_gradient_flow(self) -> None:
        """Test G: Verify training-style forward and backward pass propagates gradients."""
        block4 = SpatialContextAttention().to(self.device)
        x = torch.randn(2, 64, 32, 36, device=self.device, requires_grad=True)

        out = block4(x)
        loss = out.sum()
        loss.backward()

        # 1. Gradients propagate back to input tensor
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertFalse((x.grad == 0).all())

        # 2. Gradients propagate to Channel Attention shared MLP
        mlp_linear1 = block4.channel_attention.mlp[0]
        mlp_linear2 = block4.channel_attention.mlp[2]
        self.assertIsNotNone(mlp_linear1.weight.grad)
        self.assertIsNotNone(mlp_linear1.bias.grad)
        self.assertIsNotNone(mlp_linear2.weight.grad)
        self.assertIsNotNone(mlp_linear2.bias.grad)
        self.assertTrue(torch.isfinite(mlp_linear1.weight.grad).all())
        self.assertTrue(torch.isfinite(mlp_linear2.weight.grad).all())

        # 3. Gradients propagate to Spatial Attention convolution
        conv_layer = block4.spatial_attention.conv
        self.assertIsNotNone(conv_layer.weight.grad)
        self.assertIsNotNone(conv_layer.bias.grad)
        self.assertTrue(torch.isfinite(conv_layer.weight.grad).all())

    def test_h_cpu_compatibility(self) -> None:
        """Test H: Ensure the module works on CPU without requiring CUDA."""
        block4 = SpatialContextAttention()
        self.assertEqual(next(block4.parameters()).device.type, "cpu")

        x = torch.randn(2, 64, 32, 36, device="cpu")
        out = block4(x)
        self.assertEqual(out.device.type, "cpu")

    def test_i_input_validation(self) -> None:
        """Test I: Test reasonable handling of invalid inputs."""
        block4 = SpatialContextAttention().to(self.device)
        x_valid = self._sample_block3_output(b=2, c=64, h=32, w=36)

        # 1. Non-tensor input
        with self.assertRaises(TypeError):
            block4("not_a_tensor")

        # 2. Incorrect tensor dimensions (3D instead of 4D)
        with self.assertRaises(ValueError):
            block4(x_valid[0])

        # 3. Incorrect tensor dimensions (5D instead of 4D)
        with self.assertRaises(ValueError):
            block4(x_valid.unsqueeze(1))

        # 4. Incorrect channel count (e.g. 48 or 32 instead of 64)
        with self.assertRaises(ValueError):
            block4(x_valid[:, :48])

        with self.assertRaises(ValueError):
            block4(torch.randn(2, 32, 32, 36))

        # 5. Invalid spatial dimensions (< 7x7 required for 7x7 conv)
        with self.assertRaises(ValueError):
            block4(torch.randn(2, 64, 5, 5))

        with self.assertRaises(ValueError):
            block4(torch.randn(2, 64, 32, 4))

        # 6. Empty batch (B=0)
        with self.assertRaises(ValueError):
            block4(torch.randn(0, 64, 32, 36))

        # 7. ChannelAttention subcomponent validation
        ca = ChannelAttention()
        with self.assertRaises(TypeError):
            ca("invalid")
        with self.assertRaises(ValueError):
            ca(torch.randn(2, 32, 32, 36))  # wrong channel count
        with self.assertRaises(ValueError):
            ca(torch.randn(0, 64, 32, 36))  # empty batch

        # 8. SpatialAttention subcomponent validation
        sa = SpatialAttention(kernel_size=7)
        with self.assertRaises(TypeError):
            sa("invalid")
        with self.assertRaises(ValueError):
            sa(torch.randn(2, 64, 4, 4))  # spatial size < kernel_size

    def test_j_integration_with_block_3(self) -> None:
        """Test J: Integration with actual Block 3 (TemporalDynamicsEncoder -> Block 4)."""
        block3 = TemporalDynamicsEncoder().to(self.device)
        block4 = SpatialContextAttention().to(self.device)

        b, t, c, h, w = 2, 3, 64, 32, 36

        # Input to Block 3: [B, T, 64, 32, 36]
        block3_in = torch.randn(b, t, c, h, w, device=self.device)

        # Run Block 3
        block3_out = block3(block3_in)
        self.assertEqual(block3_out.shape, (b, 64, h, w), "Block 3 must output [B, 64, 32, 36]")

        # Pass Block 3 output directly into Block 4
        block4_out = block4(block3_out)
        self.assertEqual(block4_out.shape, (b, 64, h, w), "Block 4 must output [B, 64, 32, 36]")

        # Backward pass through both Block 4 and Block 3
        loss = block4_out.sum()
        loss.backward()

        # Check gradients reach Block 4 attention components
        self.assertIsNotNone(block4.channel_attention.mlp[0].weight.grad)
        self.assertIsNotNone(block4.spatial_attention.conv.weight.grad)

        # Check gradients reach Block 3 ConvLSTMCell parameters
        self.assertIsNotNone(block3.cell.conv_ih.weight.grad)
        self.assertIsNotNone(block3.cell.conv_hh.weight.grad)

    def test_k_full_pipeline_integration(self) -> None:
        """Test K: Full pipeline integration across Block 1 -> Block 2 -> Block 3 -> Block 4."""
        block1 = ClimateInputEncoder().to(self.device)
        block2 = MultiScaleSpatialEncoder().to(self.device)
        block3 = TemporalDynamicsEncoder().to(self.device)
        block4 = SpatialContextAttention().to(self.device)

        b, t, h, w = 2, 3, 32, 36

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

        # 2. Block 1: [B, T, 6, 32, 36] + geo + cal -> [B, T, 48, 32, 36]
        block1_out = block1(climate, geo, calendar)
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

        # 6. End-to-end backward pass through all four blocks
        loss = block4_out.mean()
        loss.backward()

        # Verify gradients reach all blocks from Block 4 to Block 1
        self.assertIsNotNone(block4.spatial_attention.conv.weight.grad, "Gradients must reach Block 4 spatial conv")
        self.assertIsNotNone(block4.channel_attention.mlp[0].weight.grad, "Gradients must reach Block 4 channel MLP")
        self.assertIsNotNone(block3.cell.conv_ih.weight.grad, "Gradients must reach Block 3 conv_ih")
        self.assertIsNotNone(block3.cell.conv_hh.weight.grad, "Gradients must reach Block 3 conv_hh")
        self.assertIsNotNone(block2.fusion[0].weight.grad, "Gradients must reach Block 2 fusion")
        self.assertIsNotNone(block1.climate_proj.conv.weight.grad, "Gradients must reach Block 1 climate conv")
        self.assertIsNotNone(block1.geo_proj.conv.weight.grad, "Gradients must reach Block 1 geo conv")
        self.assertIsNotNone(block1.cal_proj.linear.weight.grad, "Gradients must reach Block 1 cal linear")

    def test_l_regression_testing_blocks_1_to_3(self) -> None:
        """Test L: Regression check confirming Blocks 1, 2, and 3 behave identically."""
        block1 = ClimateInputEncoder().to(self.device)
        block2 = MultiScaleSpatialEncoder().to(self.device)
        block3 = ConvLSTM().to(self.device)

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

    def test_m_nan_and_inf_detection(self) -> None:
        """Test M: Proper detection and handling of NaN and Inf in Block 4."""
        block4 = SpatialContextAttention().to(self.device)
        x = self._sample_block3_output(b=2, c=64, h=32, w=36)

        # 1. NaN in input with check_finite=True
        x_nan = x.clone()
        x_nan[0, 5, 10, 10] = float("nan")
        with self.assertRaises(ValueError):
            block4(x_nan, check_finite=True)

        # NaN propagation with check_finite=False
        out_nan = block4(x_nan, check_finite=False)
        self.assertTrue(torch.isnan(out_nan).any().item())

        # 2. Inf in input with check_finite=True
        x_inf = x.clone()
        x_inf[0, 5, 10, 10] = float("inf")
        with self.assertRaises(ValueError):
            block4(x_inf, check_finite=True)

    def test_n_missing_inputs(self) -> None:
        """Test N: Handling of None / missing inputs."""
        block4 = SpatialContextAttention().to(self.device)
        with self.assertRaises(TypeError):
            block4(None)

    def test_o_deterministic_behavior(self) -> None:
        """Test O: Forward pass is deterministic across multiple calls."""
        block4 = SpatialContextAttention().to(self.device)
        block4.eval()
        x = self._sample_block3_output(b=2, c=64, h=32, w=36)

        with torch.no_grad():
            out1 = block4(x)
            out2 = block4(x)

        torch.testing.assert_close(out1, out2)

    def test_p_attention_weights_bounds_and_effects(self) -> None:
        """Test P: Channel and spatial attention maps strictly in range [0, 1]."""
        block4 = SpatialContextAttention().to(self.device)
        x = self._sample_block3_output(b=2, c=64, h=32, w=36)

        out, maps = block4(x, return_attention_maps=True)
        c_weights = maps["channel_weights"]
        s_weights = maps["spatial_weights"]

        # Channel attention weights in [0, 1]
        self.assertTrue(torch.all(c_weights >= 0.0) and torch.all(c_weights <= 1.0))
        # Spatial attention weights in [0, 1]
        self.assertTrue(torch.all(s_weights >= 0.0) and torch.all(s_weights <= 1.0))
        # Output shape matches input shape
        self.assertEqual(out.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
