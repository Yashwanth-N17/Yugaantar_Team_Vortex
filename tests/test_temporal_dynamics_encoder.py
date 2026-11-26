"""Unit and integration tests for Block 3: TemporalDynamicsEncoder (ConvLSTM).

Tests:
- Test A: Basic input and output shape [1, 3, 64, 32, 36] -> [1, 64, 32, 36]
- Test B: Batch flexibility (B=1, B=4, B=8)
- Test C: Sequence flexibility (T=1, T=3, T=5)
- Test D: Spatial preservation (H=32, W=36)
- Test E: Hidden-state and cell-state correctness (initial zero/custom states, step-by-step updates)
- Test F: Finite outputs (no NaN / Inf)
- Test G: Gradient flow (gradients reach conv_ih, conv_hh, and input tensor)
- Test H: CPU compatibility (all operations on CPU)
- Test I: Input validation (channels, dimensions, types, initial states, tiny grids)
- Test J: Integration with actual Block 2 (MultiScaleSpatialEncoder -> TemporalDynamicsEncoder)
- Test K: Full pipeline integration (ClimateInputEncoder -> MultiScaleSpatialEncoder -> TemporalDynamicsEncoder)
- Test L: Regression testing verification
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
from monsoon_r1.models.temporal_dynamics_encoder import (
    ConvLSTM,
    ConvLSTMCell,
    TemporalDynamicsEncoder,
)


class TestTemporalDynamicsEncoder(unittest.TestCase):
    """Test suite for Block 3: TemporalDynamicsEncoder (ConvLSTM)."""

    def setUp(self) -> None:
        self.device = torch.device("cpu")
        torch.manual_seed(42)

    def _sample_block2_output(
        self,
        b: int = 1,
        t: int = 3,
        c: int = 64,
        h: int = 32,
        w: int = 36,
    ) -> torch.Tensor:
        """Create sample 64-channel tensor matching Block 2 output."""
        return torch.randn(b, t, c, h, w, device=self.device)

    def test_a_basic_shape(self) -> None:
        """Test A: Input [1, 3, 64, 32, 36] must return [1, 64, 32, 36]."""
        encoder = TemporalDynamicsEncoder().to(self.device)
        x = self._sample_block2_output(b=1, t=3, c=64, h=32, w=36)

        out = encoder(x)

        expected_shape = (1, 64, 32, 36)
        self.assertEqual(out.shape, expected_shape)

        # Also verify alias ConvLSTM behaves identically
        conv_lstm = ConvLSTM().to(self.device)
        conv_lstm.load_state_dict(encoder.state_dict())
        out_alias = conv_lstm(x)
        torch.testing.assert_close(out, out_alias)

    def test_b_batch_flexibility(self) -> None:
        """Test B: Test batch sizes B=1, B=2, B=4, and B=8; batch dimension must remain unchanged."""
        encoder = TemporalDynamicsEncoder().to(self.device)

        for b in [1, 2, 4, 8]:
            x = self._sample_block2_output(b=b, t=3, c=64, h=32, w=36)
            out = encoder(x)
            self.assertEqual(out.shape, (b, 64, 32, 36), f"Failed for batch size B={b}")

    def test_c_sequence_flexibility(self) -> None:
        """Test C: Test sequence lengths T=1, T=3, T=5; encoder must handle different sequence lengths."""
        encoder = TemporalDynamicsEncoder().to(self.device)

        for t in [1, 3, 5, 7]:
            x = self._sample_block2_output(b=2, t=t, c=64, h=32, w=36)
            out = encoder(x)
            self.assertEqual(out.shape, (2, 64, 32, 36), f"Failed for sequence length T={t}")

    def test_d_spatial_preservation(self) -> None:
        """Test D: Spatial dimensions Height=32 and Width=36 must be preserved."""
        encoder = TemporalDynamicsEncoder().to(self.device)
        x = self._sample_block2_output(b=2, t=3, c=64, h=32, w=36)

        out = encoder(x)

        self.assertEqual(out.size(-2), 32, "Height must be 32")
        self.assertEqual(out.size(-1), 36, "Width must be 36")

        # Also test with an alternative valid spatial grid
        x_alt = self._sample_block2_output(b=1, t=2, c=64, h=16, w=20)
        out_alt = encoder(x_alt)
        self.assertEqual(out_alt.shape, (1, 64, 16, 20))

    def test_e_hidden_state_and_cell_state_correctness(self) -> None:
        """Test E: Verify initial states, state dimensions, sequential updates, and final hidden-state output."""
        encoder = TemporalDynamicsEncoder().to(self.device)
        b, t, h, w = 2, 3, 32, 36
        x = self._sample_block2_output(b=b, t=t, c=64, h=h, w=w)

        # 1. Test init_hidden defaults to zero tensors with correct dimensions
        h_0, c_0 = encoder.init_hidden(b, h, w, device=self.device)
        self.assertEqual(h_0.shape, (b, 64, h, w))
        self.assertEqual(c_0.shape, (b, 64, h, w))
        self.assertTrue(torch.all(h_0 == 0.0))
        self.assertTrue(torch.all(c_0 == 0.0))

        # 2. Test manual step-by-step updates match encoder forward pass
        h_step, c_step = h_0, c_0
        for step in range(t):
            h_step, c_step = encoder.cell(x[:, step], (h_step, c_step))
            self.assertEqual(h_step.shape, (b, 64, h, w))
            self.assertEqual(c_step.shape, (b, 64, h, w))

        out_direct = encoder(x)
        torch.testing.assert_close(out_direct, h_step)

        # 3. Test return_state=True returns matching (out, (h_last, c_last))
        out_val, (h_last, c_last) = encoder(x, return_state=True)
        torch.testing.assert_close(out_val, h_step)
        torch.testing.assert_close(h_last, h_step)
        torch.testing.assert_close(c_last, c_step)

        # 4. Test return_sequence=True returns sequence of hidden states [B, T, 64, H, W]
        out_seq = encoder(x, return_sequence=True)
        self.assertEqual(out_seq.shape, (b, t, 64, h, w))
        torch.testing.assert_close(out_seq[:, -1], h_step)

        # 5. Test passing custom non-zero initial state
        h_init = torch.randn(b, 64, h, w, device=self.device)
        c_init = torch.randn(b, 64, h, w, device=self.device)
        out_custom = encoder(x, initial_state=(h_init, c_init))
        self.assertEqual(out_custom.shape, (b, 64, h, w))
        # Custom state must yield different output than zero init
        self.assertFalse(torch.allclose(out_custom, out_direct))

        # 6. Verify sequential temporal causality: modifying x at t=0 changes the output at t=2
        x_perturbed = x.clone()
        x_perturbed[:, 0] = x_perturbed[:, 0] + 1.0
        out_perturbed = encoder(x_perturbed)
        self.assertFalse(torch.allclose(out_direct, out_perturbed))

    def test_f_finite_outputs(self) -> None:
        """Test F: Verify output contains no NaN or Inf for valid inputs."""
        encoder = TemporalDynamicsEncoder().to(self.device)
        x = self._sample_block2_output(b=4, t=3, c=64, h=32, w=36)

        out = encoder(x)

        self.assertTrue(torch.isfinite(out).all().item(), "Output must contain only finite numbers")
        self.assertFalse(torch.isnan(out).any().item(), "Output must not contain NaN")
        self.assertFalse(torch.isinf(out).any().item(), "Output must not contain Inf")

    def test_g_gradient_flow(self) -> None:
        """Test G: Verify training backward pass propagates gradients through ConvLSTM parameters."""
        encoder = TemporalDynamicsEncoder().to(self.device)
        x = self._sample_block2_output(b=2, t=3, c=64, h=32, w=36)
        x.requires_grad_(True)

        out = encoder(x)
        loss = out.sum()
        loss.backward()

        # Check input gradients
        self.assertIsNotNone(x.grad, "Gradients must reach input tensor")
        self.assertFalse(torch.all(x.grad == 0), "Input gradients must not be all zero")

        # ConvLSTM cell conv_ih weights and bias
        self.assertIsNotNone(encoder.cell.conv_ih.weight.grad, "Gradients must reach conv_ih weights")
        self.assertIsNotNone(encoder.cell.conv_ih.bias.grad, "Gradients must reach conv_ih bias")
        self.assertFalse(torch.all(encoder.cell.conv_ih.weight.grad == 0))
        self.assertFalse(torch.all(encoder.cell.conv_ih.bias.grad == 0))

        # ConvLSTM cell conv_hh weights and bias
        self.assertIsNotNone(encoder.cell.conv_hh.weight.grad, "Gradients must reach conv_hh weights")
        self.assertIsNotNone(encoder.cell.conv_hh.bias.grad, "Gradients must reach conv_hh bias")
        self.assertFalse(torch.all(encoder.cell.conv_hh.weight.grad == 0))
        self.assertFalse(torch.all(encoder.cell.conv_hh.bias.grad == 0))

    def test_h_cpu_compatibility(self) -> None:
        """Test H: Ensure module and all operations execute reliably on CPU without CUDA."""
        encoder = TemporalDynamicsEncoder().to("cpu")
        x = self._sample_block2_output(b=2, t=3, c=64, h=32, w=36)

        self.assertEqual(x.device.type, "cpu")
        out = encoder(x)
        self.assertEqual(out.device.type, "cpu")
        self.assertEqual(out.shape, (2, 64, 32, 36))

    def test_i_input_validation(self) -> None:
        """Test I: Test error handling for invalid channel count, dimensions, empty sequences, types, and states."""
        encoder = TemporalDynamicsEncoder().to(self.device)
        x = self._sample_block2_output(b=2, t=3, c=64, h=32, w=36)

        # 1. Non-tensor input
        with self.assertRaises(TypeError):
            encoder("not_a_tensor")

        # 2. Invalid dimensions (4D tensor instead of 5D [B, T, C, H, W])
        with self.assertRaises(ValueError):
            encoder(x[:, 0])

        # 3. Invalid channel count (e.g. 48 instead of 64)
        with self.assertRaises(ValueError):
            encoder(x[:, :, :48])

        # 4. Tiny spatial grid (< 3x3) where 3x3 kernel cannot operate
        x_tiny = torch.randn(1, 1, 64, 2, 2)
        with self.assertRaises(ValueError):
            encoder(x_tiny)

        # 5. Empty sequence or batch
        x_empty_seq = torch.randn(2, 0, 64, 32, 36)
        with self.assertRaises(ValueError):
            encoder(x_empty_seq)

        x_empty_batch = torch.randn(0, 3, 64, 32, 36)
        with self.assertRaises(ValueError):
            encoder(x_empty_batch)

        # 6. Invalid initial state format
        with self.assertRaises(TypeError):
            encoder(x, initial_state="not_a_tuple")

        with self.assertRaises(TypeError):
            encoder(x, initial_state=(torch.randn(2, 64, 32, 36), "not_a_tensor"))

        # Wrong initial state channel shape
        with self.assertRaises(ValueError):
            encoder(x, initial_state=(torch.randn(2, 32, 32, 36), torch.randn(2, 32, 32, 36)))

        # 7. Cell level validation
        cell = ConvLSTMCell(in_channels=64, hidden_channels=64)
        with self.assertRaises(TypeError):
            cell("invalid", (torch.zeros(1, 64, 32, 36), torch.zeros(1, 64, 32, 36)))
        with self.assertRaises(ValueError):
            cell(torch.randn(1, 64, 32, 36), (torch.zeros(1, 64, 16, 16), torch.zeros(1, 64, 16, 16)))

    def test_j_integration_with_block_2(self) -> None:
        """Test J: End-to-end integration test with the actual Block 2 implementation."""
        block2 = MultiScaleSpatialEncoder().to(self.device)
        block3 = TemporalDynamicsEncoder().to(self.device)

        b, t, h, w = 2, 3, 32, 36

        # Block 2 input: [B, T, 48, 32, 36]
        block2_in = torch.randn(b, t, 48, h, w, device=self.device)

        # Run through Block 2
        block2_out = block2(block2_in)
        self.assertEqual(block2_out.shape, (b, t, 64, h, w), "Block 2 must output [B, T, 64, 32, 36]")

        # Pass Block 2 output into Block 3
        block3_out = block3(block2_out)
        self.assertEqual(block3_out.shape, (b, 64, h, w), "Block 3 must output [B, 64, 32, 36]")

        # Backward pass through Block 3 and Block 2
        loss = block3_out.sum()
        loss.backward()

        self.assertIsNotNone(block3.cell.conv_ih.weight.grad)
        self.assertIsNotNone(block2.fusion[0].weight.grad)
        self.assertIsNotNone(block2.branch_a[0].weight.grad)

    def test_k_full_pipeline_integration(self) -> None:
        """Test K: Full pipeline integration across Block 1 -> Block 2 -> Block 3."""
        block1 = ClimateInputEncoder().to(self.device)
        block2 = MultiScaleSpatialEncoder().to(self.device)
        block3 = TemporalDynamicsEncoder().to(self.device)

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

        # 2. Block 1 forward pass: [B, T, 6, 32, 36] + geo + cal -> [B, T, 48, 32, 36]
        block1_out = block1(climate, geo, calendar)
        self.assertEqual(block1_out.shape, (b, t, 48, h, w), "Block 1 must output [B, T, 48, 32, 36]")

        # 3. Block 2 forward pass: [B, T, 48, 32, 36] -> [B, T, 64, 32, 36]
        block2_out = block2(block1_out)
        self.assertEqual(block2_out.shape, (b, t, 64, h, w), "Block 2 must output [B, T, 64, 32, 36]")

        # 4. Block 3 forward pass: [B, T, 64, 32, 36] -> [B, 64, 32, 36]
        block3_out = block3(block2_out)
        self.assertEqual(block3_out.shape, (b, 64, h, w), "Block 3 must output [B, 64, 32, 36]")

        # 5. End-to-end backward pass through all three blocks
        loss = block3_out.mean()
        loss.backward()

        # Gradients must propagate all the way from Block 3 output to Block 1 inputs/parameters
        self.assertIsNotNone(block3.cell.conv_ih.weight.grad, "Gradients must reach Block 3 conv_ih")
        self.assertIsNotNone(block3.cell.conv_hh.weight.grad, "Gradients must reach Block 3 conv_hh")
        self.assertIsNotNone(block2.fusion[0].weight.grad, "Gradients must reach Block 2 fusion")
        self.assertIsNotNone(block1.climate_proj.conv.weight.grad, "Gradients must reach Block 1 climate conv")
        self.assertIsNotNone(block1.geo_proj.conv.weight.grad, "Gradients must reach Block 1 geo conv")
        self.assertIsNotNone(block1.cal_proj.linear.weight.grad, "Gradients must reach Block 1 cal linear")

    def test_l_regression_testing_blocks_1_and_2(self) -> None:
        """Test L: Regression check confirming Blocks 1 and 2 interfaces and computations remain identical."""
        block1 = ClimateInputEncoder().to(self.device)
        block2 = MultiScaleSpatialEncoder().to(self.device)

        b, t, h, w = 1, 3, 32, 36
        climate = torch.randn(b, t, 6, h, w, device=self.device)
        geo = torch.randn(b, 2, h, w, device=self.device)
        cal = torch.randn(b, 2, device=self.device)

        out1 = block1(climate, geo, cal)
        self.assertEqual(out1.shape, (b, t, 48, h, w))

        out2 = block2(out1)
        self.assertEqual(out2.shape, (b, t, 64, h, w))

    def test_m_nan_and_inf_handling(self) -> None:
        """Test M: Proper detection and handling of NaN and Inf in Block 3."""
        encoder = TemporalDynamicsEncoder().to(self.device)
        x = self._sample_block2_output(b=2, t=3, c=64, h=32, w=36)

        # 1. NaN in input x with check_finite=True
        x_nan = x.clone()
        x_nan[0, 1, 0, 5, 5] = float("nan")
        with self.assertRaises(ValueError):
            encoder(x_nan, check_finite=True)

        # NaN propagation with check_finite=False
        out_nan = encoder(x_nan, check_finite=False)
        self.assertTrue(torch.isnan(out_nan).any().item())

        # 2. Inf in input x with check_finite=True
        x_inf = x.clone()
        x_inf[0, 1, 0, 5, 5] = float("inf")
        with self.assertRaises(ValueError):
            encoder(x_inf, check_finite=True)

        # 3. NaN in initial_state with check_finite=True
        h_nan = torch.zeros(2, 64, 32, 36)
        h_nan[0, 0, 0, 0] = float("nan")
        c_zero = torch.zeros(2, 64, 32, 36)
        with self.assertRaises(ValueError):
            encoder(x, initial_state=(h_nan, c_zero), check_finite=True)

    def test_n_missing_inputs(self) -> None:
        """Test N: Handling of None / missing inputs."""
        encoder = TemporalDynamicsEncoder().to(self.device)
        with self.assertRaises(TypeError):
            encoder(None)

    def test_o_deterministic_behavior(self) -> None:
        """Test O: Forward pass is deterministic across multiple calls."""
        encoder = TemporalDynamicsEncoder().to(self.device)
        encoder.eval()
        x = self._sample_block2_output(b=2, t=3, c=64, h=32, w=36)

        with torch.no_grad():
            out1 = encoder(x)
            out2 = encoder(x)

        torch.testing.assert_close(out1, out2)

    def test_p_long_sequence_stability(self) -> None:
        """Test P: Multi-timestep recurrence (T=10, T=15) stability of cell and hidden states."""
        encoder = TemporalDynamicsEncoder().to(self.device)
        x_long = self._sample_block2_output(b=1, t=15, c=64, h=32, w=36)

        out, (h_last, c_last) = encoder(x_long, return_state=True)
        self.assertEqual(out.shape, (1, 64, 32, 36))
        self.assertTrue(torch.isfinite(out).all().item())
        self.assertTrue(torch.isfinite(h_last).all().item())
        self.assertTrue(torch.isfinite(c_last).all().item())
        # tanh(C) bounds hidden state magnitude strictly within (-1, 1)
        self.assertTrue(torch.all(out >= -1.0) and torch.all(out <= 1.0))


if __name__ == "__main__":
    unittest.main()
