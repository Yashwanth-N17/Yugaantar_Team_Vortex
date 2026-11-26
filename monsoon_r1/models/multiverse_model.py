"""Multiverse of Monsoon — Full Architecture Sequential Integration Model.

Sequentially connects all six blocks into an end-to-end differentiable climate forecasting
and counterfactual simulation pipeline:

    Climate Sequence [B, T, 6, 32, 36] + Static [B, 2, 32, 36] + Calendar [B, 2]
    │
    ▼
    Block 1: Climate Input Encoder        ───>  [B, T, 48, 32, 36]
    │
    ▼
    Block 2: Multi-Scale Spatial Encoder  ───>  [B, T, 64, 32, 36]
    │
    ▼
    Block 3: Temporal Dynamics (ConvLSTM) ───>  [B, 64, 32, 36]
    │
    ▼
    Block 4: Spatial Context Attention    ───>  [B, 64, 32, 36]
    │
    ▼
    Block 5: Forecast Decoder             ───>  [B, 6, 32, 36] (Deltas)
    │
    ▼
    Block 6: Residual State Transition    ───>  [B, 6, 32, 36] (Next Climate State)
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from monsoon_r1.contract import CHANNELS, NC, NLAT, NLON, SCORE_STEPS, T_IN
from .climate_input_encoder import ClimateInputEncoder
from .multi_scale_spatial_encoder import MultiScaleSpatialEncoder
from .temporal_dynamics_encoder import TemporalDynamicsEncoder
from .spatial_context_attention import SpatialContextAttention
from .forecast_decoder import ForecastDecoder
from .residual_state_transition import ResidualStateTransition


class MultiverseOfMonsoonModel(nn.Module):
    """Full Sequential Architecture: Multiverse of Monsoon.

    Integrates all six architectural blocks into a coherent, differentiable pipeline
    supporting single-step forecasting, autoregressive multi-step rollouts, and
    counterfactual perturbation simulations.

    Attributes:
        block1 (ClimateInputEncoder): Transforms dynamic, static, and calendar inputs.
        block2 (MultiScaleSpatialEncoder): Extracts multi-scale spatial patterns (3 branches).
        block3 (TemporalDynamicsEncoder): ConvLSTM learning atmospheric evolution across time.
        block4 (SpatialContextAttention): CBAM-inspired channel and spatial refinement.
        block5 (ForecastDecoder): Decodes refined latents into 6 climate change deltas.
        block6 (ResidualStateTransition): Residual update enforcing physical non-negativity.
    """

    CHANNELS: Tuple[str, ...] = CHANNELS
    NUM_CHANNELS: int = NC
    DEFAULT_HEIGHT: int = NLAT
    DEFAULT_WIDTH: int = NLON
    INPUT_WINDOW: int = T_IN

    def __init__(
        self,
        block1: Optional[ClimateInputEncoder] = None,
        block2: Optional[MultiScaleSpatialEncoder] = None,
        block3: Optional[TemporalDynamicsEncoder] = None,
        block4: Optional[SpatialContextAttention] = None,
        block5: Optional[ForecastDecoder] = None,
        block6: Optional[ResidualStateTransition] = None,
        channel_scales: Optional[Union[torch.Tensor, Sequence[float]]] = None,
        channel_means: Optional[Union[torch.Tensor, Sequence[float]]] = None,
        state_is_normalized: bool = False,
    ) -> None:
        """Initialize full 6-block MultiverseOfMonsoonModel.

        Args:
            channel_means: Per-channel means from stats.json used to normalise raw physical
                inputs inside Block 1 (ClimateInputEncoder), so all 6 channels (temperature,
                rain, wind, pressure, moisture) are on the same zero-mean scale before the
                first convolution.
            channel_scales: Per-channel standard deviations for Block 6 delta scaling and
                the weighted training loss.
        """
        super().__init__()

        self.block1 = block1 if block1 is not None else ClimateInputEncoder(
            input_mean=channel_means,
            input_std=channel_scales,
        )
        self.block2 = block2 if block2 is not None else MultiScaleSpatialEncoder()
        self.block3 = block3 if block3 is not None else TemporalDynamicsEncoder()
        self.block4 = block4 if block4 is not None else SpatialContextAttention()
        self.block5 = block5 if block5 is not None else ForecastDecoder()
        self.block6 = (
            block6
            if block6 is not None
            else ResidualStateTransition(
                channel_scales=channel_scales,
                channel_means=channel_means,
                state_is_normalized=state_is_normalized,
            )
        )

    def predict_delta(
        self,
        climate: torch.Tensor,
        geography: torch.Tensor,
        calendar: torch.Tensor,
        check_finite: bool = False,
    ) -> torch.Tensor:
        """Execute Blocks 1 through 5 to compute predicted climate deltas.

        Args:
            climate (torch.Tensor): Dynamic climate sequence [B, T, 6, H, W].
            geography (torch.Tensor): Static geography [B, 2, H, W] or [2, H, W].
            calendar (torch.Tensor): Calendar context [B, 2] or [2].
            check_finite (bool): Whether to enforce finite input verification.

        Returns:
            torch.Tensor: Predicted climate deltas [B, 6, H, W].
        """
        # Block 1: Climate Input Encoder -> [B, T, 48, H, W]
        b1_out = self.block1(climate, geography, calendar, check_finite=check_finite)

        # Block 2: Multi-Scale Spatial Encoder -> [B, T, 64, H, W]
        b2_out = self.block2(b1_out, check_finite=check_finite)

        # Block 3: Temporal Dynamics Encoder (ConvLSTM) -> [B, 64, H, W]
        b3_out = self.block3(b2_out, check_finite=check_finite)

        # Block 4: Spatial Context Attention -> [B, 64, H, W]
        b4_out = self.block4(b3_out, check_finite=check_finite)

        # Block 5: Forecast Decoder -> [B, 6, H, W] (deltas)
        delta = self.block5(b4_out, check_finite=check_finite)

        return delta

    def forward(
        self,
        climate: torch.Tensor,
        geography: torch.Tensor,
        calendar: torch.Tensor,
        current_state: Optional[torch.Tensor] = None,
        channel_scales: Optional[Union[torch.Tensor, Sequence[float]]] = None,
        return_intermediates: bool = False,
        check_finite: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Execute the complete 6-block pipeline for one step of forecast transition.

        Args:
            climate (torch.Tensor): Dynamic climate sequence [B, T, 6, H, W].
            geography (torch.Tensor): Static geography [B, 2, H, W] or [2, H, W].
            calendar (torch.Tensor): Calendar context [B, 2] or [2].
            current_state (torch.Tensor, optional): State to add delta to [B, 6, H, W].
                Defaults to climate[:, -1] (latest day in input window).
            channel_scales (Tensor or Sequence[float], optional): Optional per-channel scales.
            return_intermediates (bool): Whether to return a dict of per-block feature maps.
            check_finite (bool): Whether to check for NaN/Inf.

        Returns:
            torch.Tensor or (torch.Tensor, dict): Next climate state [B, 6, H, W],
                optionally with intermediate tensors for all blocks.
        """
        if current_state is None:
            # By default, residual delta is applied to the latest observed state
            if climate.dim() == 5:
                current_state = climate[:, -1]
            elif climate.dim() == 4:
                current_state = climate[-1].unsqueeze(0)
            else:
                raise ValueError(f"Unexpected climate tensor dimensions: {climate.dim()}D")

        # Block 1
        b1_out = self.block1(climate, geography, calendar, check_finite=check_finite)

        # Block 2
        b2_out = self.block2(b1_out, check_finite=check_finite)

        # Block 3
        b3_out = self.block3(b2_out, check_finite=check_finite)

        # Block 4
        b4_out = self.block4(b3_out, check_finite=check_finite)

        # Block 5
        delta = self.block5(b4_out, check_finite=check_finite)

        # Block 6: Residual State Transition
        next_state = self.block6(
            current_state,
            delta,
            channel_scales=channel_scales,
            check_finite=check_finite,
        )

        if return_intermediates:
            intermediates = {
                "block1_encoded": b1_out,
                "block2_spatial": b2_out,
                "block3_temporal": b3_out,
                "block4_attended": b4_out,
                "block5_delta": delta,
                "block6_next_state": next_state,
            }
            return next_state, intermediates

        return next_state

    def apply_perturbation(
        self,
        initial_state: torch.Tensor,
        perturbation: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a physical perturbation to an initial state via Block 6."""
        return self.block6.apply_perturbation(initial_state, perturbation)

    def rollout(
        self,
        init_states: torch.Tensor,
        geography: torch.Tensor,
        doy_sincos: Union[torch.Tensor, Sequence[float]],
        steps: int = SCORE_STEPS,
        perturbation: Optional[torch.Tensor] = None,
        channel_scales: Optional[Union[torch.Tensor, Sequence[float]]] = None,
        check_finite: bool = False,
    ) -> torch.Tensor:
        """Autoregressive multi-step rollout for climate forecasting and counterfactuals.

        Args:
            init_states (torch.Tensor): Initial 3-day window [B, 3, 6, H, W] or [3, 6, H, W].
            geography (torch.Tensor): Static geography [B, 2, H, W] or [2, H, W].
            doy_sincos (Tensor or Sequence[float]): [sin, cos] of the last initial day.
            steps (int): Number of forecast steps to roll out (default: 10).
            perturbation (torch.Tensor, optional): Physical perturbation delta added to
                the last initial day before simulation starts.
            channel_scales (Tensor or Sequence[float], optional): Per-channel delta scales.
            check_finite (bool): Whether to enforce finite checks.

        Returns:
            torch.Tensor: Rollout predictions of shape [steps, B, 6, H, W]
                or [steps, 6, H, W] if unbatched.
        """
        unbatched = False
        if init_states.dim() == 4:
            init_states = init_states.unsqueeze(0)  # [1, 3, 6, H, W]
            unbatched = True

        b, t_in, c, h, w = init_states.shape
        window = init_states.clone()

        # Format calendar context
        if isinstance(doy_sincos, torch.Tensor):
            cal_tensor = doy_sincos.clone().float()
            if cal_tensor.dim() == 1:
                cal_tensor = cal_tensor.unsqueeze(0).expand(b, -1)
        else:
            cal_tensor = torch.tensor(list(doy_sincos), dtype=torch.float32, device=init_states.device)
            cal_tensor = cal_tensor.unsqueeze(0).expand(b, -1)

        # Apply perturbation to the last initial state if provided (counterfactual timeline)
        if perturbation is not None:
            if perturbation.dim() == 3:
                pert_b = perturbation.unsqueeze(0).expand(b, -1, -1, -1)
            else:
                pert_b = perturbation
            window[:, -1] = self.block6.apply_perturbation(window[:, -1], pert_b)

        predictions: List[torch.Tensor] = []
        alpha = 2.0 * math.pi / 365.0

        for step in range(steps):
            # Advance day of year by step + 1
            # Angle shift: delta_angle = (step + 1) * alpha
            delta_angle = (step + 1) * alpha
            cos_d = math.cos(delta_angle)
            sin_d = math.sin(delta_angle)

            sin_0 = cal_tensor[:, 0]
            cos_0 = cal_tensor[:, 1]
            sin_k = sin_0 * cos_d + cos_0 * sin_d
            cos_k = cos_0 * cos_d - sin_0 * sin_d
            curr_cal = torch.stack([sin_k, cos_k], dim=-1)

            # Predict next state using current rolling window
            next_state = self.forward(
                climate=window,
                geography=geography,
                calendar=curr_cal,
                channel_scales=channel_scales,
                check_finite=check_finite,
            )

            predictions.append(next_state)

            # Roll sliding window forward: drop oldest day, append new state
            window = torch.cat([window[:, 1:], next_state.unsqueeze(1)], dim=1)

        rollout_out = torch.stack(predictions, dim=0)  # [steps, B, 6, H, W]
        return rollout_out.squeeze(1) if unbatched else rollout_out


# Aliases
MultiverseModel = MultiverseOfMonsoonModel
MonsoonArchitecture = MultiverseOfMonsoonModel
