"""Block 6: Residual State Transition for Multiverse of Monsoon.

Converts the forecast decoder's (Block 5) predicted climate deltas into the next
atmospheric climate state by applying a representation-consistent residual update:

    Next State = Current State + Predicted Delta

The transition module enforces physical constraints (specifically non-negative
precipitation on channel index 1), maintains consistent units and normalization,
supports optional per-channel delta scaling, and integrates directly into
autoregressive multi-day rollouts.

Channel ordering (strictly matching monsoon_r1.contract.CHANNELS):
  0: t2m  - Temperature at 2 metres (K)
  1: tp   - Total precipitation (mm/day) [physically non-negative: tp >= 0]
  2: u10  - 10-metre zonal wind (m/s)
  3: v10  - 10-metre meridional wind (m/s)
  4: msl  - Mean sea-level pressure (Pa)
  5: tcwv - Total column water vapour (kg/m²)

Input shapes:
  - current_state:   [B, 6, 32, 36]
  - predicted_delta: [B, 6, 32, 36]
Output shape:
  - next_state:      [B, 6, 32, 36]
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from monsoon_r1.contract import CHANNELS, NC, NLAT, NLON


class ResidualStateTransition(nn.Module):
    """Block 6: Residual State Transition Module.

    Combines the current atmospheric state with the forecast decoder's predicted
    state deltas to compute the subsequent climate state:

        y_{t+1} = y_t + Delta y

    Attributes:
        num_channels (int): Number of climate channels (default: 6).
        precipitation_idx (int): Channel index for precipitation (default: 1).
        clamp_precipitation (bool): Whether to enforce non-negative precipitation (default: True).
        precipitation_min (float): Minimum threshold for precipitation (default: 0.0 mm/day).
        state_is_normalized (bool): Whether states are in normalized space (default: False).
        channel_scales (torch.Tensor, optional): Registered buffer of shape [1, 6, 1, 1] for delta scaling.
        channel_means (torch.Tensor, optional): Registered buffer of shape [1, 6, 1, 1] for normalized states.
    """

    CHANNELS: Tuple[str, ...] = CHANNELS
    TARGET_VARIABLES: Tuple[str, ...] = CHANNELS
    NUM_CHANNELS: int = NC
    DEFAULT_HEIGHT: int = NLAT
    DEFAULT_WIDTH: int = NLON
    PRECIPITATION_INDEX: int = 1

    def __init__(
        self,
        num_channels: int = NC,
        precipitation_idx: int = 1,
        clamp_precipitation: bool = True,
        precipitation_min: float = 0.0,
        channel_scales: Optional[Union[torch.Tensor, Sequence[float]]] = None,
        channel_means: Optional[Union[torch.Tensor, Sequence[float]]] = None,
        state_is_normalized: bool = False,
    ) -> None:
        """Initialize ResidualStateTransition.

        Args:
            num_channels (int): Number of climate channels (default: 6).
            precipitation_idx (int): Channel index for precipitation (default: 1).
            clamp_precipitation (bool): Whether to enforce non-negative precipitation (default: True).
            precipitation_min (float): Minimum physical threshold for precipitation (default: 0.0).
            channel_scales (Tensor or Sequence[float], optional): Per-channel standard deviations
                or scale factors for converting normalized deltas to physical units.
            channel_means (Tensor or Sequence[float], optional): Per-channel climatological means.
                Used if state_is_normalized=True to compute mathematically exact normalized lower bound.
            state_is_normalized (bool): Whether states are processed in normalized space (default: False).
        """
        super().__init__()
        if num_channels <= 0:
            raise ValueError(f"num_channels must be positive, got {num_channels}")
        if precipitation_idx < 0 or precipitation_idx >= num_channels:
            raise ValueError(
                f"precipitation_idx must be in range [0, {num_channels - 1}], got {precipitation_idx}"
            )

        self.num_channels = num_channels
        self.precipitation_idx = precipitation_idx
        self.clamp_precipitation = clamp_precipitation
        self.precipitation_min = float(precipitation_min)
        self.state_is_normalized = state_is_normalized

        if channel_scales is not None:
            scales_t = self._format_channel_tensor(channel_scales, "channel_scales")
            self.register_buffer("channel_scales", scales_t)
        else:
            self.register_buffer("channel_scales", None)

        if channel_means is not None:
            means_t = self._format_channel_tensor(channel_means, "channel_means")
            self.register_buffer("channel_means", means_t)
        else:
            self.register_buffer("channel_means", None)

    def _format_channel_tensor(
        self,
        data: Union[torch.Tensor, Sequence[float]],
        param_name: str,
    ) -> torch.Tensor:
        """Format 1D or broadcastable channel scales/means into [1, C, 1, 1] tensor."""
        if isinstance(data, torch.Tensor):
            t = data.detach().clone().float()
        else:
            try:
                t = torch.tensor(list(data), dtype=torch.float32)
            except Exception as e:
                raise TypeError(f"{param_name} must be a Tensor or sequence of numbers: {e}") from e

        if t.numel() != self.num_channels:
            raise ValueError(
                f"{param_name} must contain exactly {self.num_channels} values, got {t.numel()}"
            )
        return t.view(1, self.num_channels, 1, 1)

    @classmethod
    def from_default_scales(
        cls,
        clamp_precipitation: bool = True,
        state_is_normalized: bool = False,
    ) -> ResidualStateTransition:
        """Instantiate with standard deviation scales from monsoon_r1.scoring.STD_DEFAULT.

        STD_DEFAULT = [8.0, 8.0, 3.5, 2.5, 450.0, 20.0]
        """
        from monsoon_r1.scoring import STD_DEFAULT

        return cls(
            num_channels=NC,
            precipitation_idx=1,
            clamp_precipitation=clamp_precipitation,
            channel_scales=STD_DEFAULT,
            state_is_normalized=state_is_normalized,
        )

    @classmethod
    def from_stats_dict(
        cls,
        stats: dict,
        clamp_precipitation: bool = True,
        state_is_normalized: bool = False,
    ) -> ResidualStateTransition:
        """Instantiate from channel statistics dictionary (e.g. from stats.json)."""
        scales = stats.get("std", None)
        means = stats.get("mean", None)
        return cls(
            num_channels=NC,
            precipitation_idx=1,
            clamp_precipitation=clamp_precipitation,
            channel_scales=scales,
            channel_means=means,
            state_is_normalized=state_is_normalized,
        )

    def _validate_inputs(
        self,
        current_state: torch.Tensor,
        predicted_delta: torch.Tensor,
        check_finite: bool = False,
    ) -> Tuple[int, int, int, int]:
        """Validate input tensors shape, type, and dimensions.

        Args:
            current_state (torch.Tensor): Current climate state [B, C, H, W].
            predicted_delta (torch.Tensor): Predicted climate delta [B, C, H, W].
            check_finite (bool): If True, validates no NaN or Inf values.

        Returns:
            Tuple[int, int, int, int]: (B, C, H, W) dimensions.

        Raises:
            TypeError: If either input is not a torch.Tensor.
            ValueError: If either tensor is not 4D, shapes mismatch, channels mismatch,
                batch size < 1, spatial dimensions < 1, device/dtype mismatch,
                or contains NaN/Inf (when check_finite=True).
        """
        if not isinstance(current_state, torch.Tensor):
            raise TypeError(
                f"current_state must be a torch.Tensor, got {type(current_state).__name__}"
            )
        if not isinstance(predicted_delta, torch.Tensor):
            raise TypeError(
                f"predicted_delta must be a torch.Tensor, got {type(predicted_delta).__name__}"
            )

        if check_finite:
            if not torch.isfinite(current_state).all():
                raise ValueError("current_state contains NaN or Inf values")
            if not torch.isfinite(predicted_delta).all():
                raise ValueError("predicted_delta contains NaN or Inf values")

        if current_state.dim() != 4:
            raise ValueError(
                f"Expected 4D current_state [B, C, H, W], got {current_state.dim()}D with shape {list(current_state.shape)}"
            )
        if predicted_delta.dim() != 4:
            raise ValueError(
                f"Expected 4D predicted_delta [B, C, H, W], got {predicted_delta.dim()}D with shape {list(predicted_delta.shape)}"
            )

        if current_state.shape != predicted_delta.shape:
            raise ValueError(
                f"Shape mismatch: current_state {list(current_state.shape)} != predicted_delta {list(predicted_delta.shape)}"
            )

        b, c, h, w = current_state.shape
        if b < 1:
            raise ValueError(f"Batch size B must be >= 1, got B={b}")
        if c != self.num_channels:
            raise ValueError(
                f"Expected {self.num_channels} channels at dim 1, got {c} in shape {list(current_state.shape)}"
            )
        if h < 1 or w < 1:
            raise ValueError(
                f"Spatial dimensions ({h}, {w}) must be >= 1, got ({h}, {w})"
            )
        if current_state.device != predicted_delta.device:
            raise ValueError(
                f"Device mismatch: current_state is on {current_state.device} but predicted_delta is on {predicted_delta.device}"
            )
        if current_state.dtype != predicted_delta.dtype:
            raise ValueError(
                f"Dtype mismatch: current_state has dtype {current_state.dtype} but predicted_delta has dtype {predicted_delta.dtype}"
            )
        return b, c, h, w

    def _get_effective_precipitation_min(
        self,
        current_scales: Optional[torch.Tensor] = None,
    ) -> float:
        """Compute the mathematically equivalent minimum threshold for precipitation.

        If states are normalized, uses (0.0 - mean) / std to ensure that the
        physical precipitation will never be negative upon denormalization.
        If states are in physical space, uses precipitation_min (default 0.0 mm/day).
        """
        if self.state_is_normalized:
            scales = current_scales if current_scales is not None else self.channel_scales
            if self.channel_means is not None and scales is not None:
                mean_p = float(self.channel_means[0, self.precipitation_idx, 0, 0])
                std_p = float(scales[0, self.precipitation_idx, 0, 0])
                if std_p > 0:
                    return -mean_p / std_p
            return self.precipitation_min
        return self.precipitation_min

    def forward(
        self,
        current_state: torch.Tensor,
        predicted_delta: torch.Tensor,
        channel_scales: Optional[Union[torch.Tensor, Sequence[float]]] = None,
        check_finite: bool = False,
    ) -> torch.Tensor:
        """Forward pass of Block 6: Residual State Transition.

        Applies predicted atmospheric changes to current climate state:
            next_state = current_state + delta

        Enforces non-negativity constraint on precipitation channel (index 1)
        in an autograd-friendly, out-of-place manner.

        Args:
            current_state (torch.Tensor): Current climate state [B, 6, H, W].
            predicted_delta (torch.Tensor): Predicted climate delta [B, 6, H, W].
            channel_scales (Tensor or Sequence[float], optional): Optional per-channel
                scale factors to override or supplement buffer scales.
            check_finite (bool): If True, validates no NaN or Inf in inputs. Default: False.

        Returns:
            torch.Tensor: Next climate state of shape [B, 6, H, W].
        """
        self._validate_inputs(current_state, predicted_delta, check_finite=check_finite)

        # 1. Determine active channel scaling factors
        active_scales = None
        if channel_scales is not None:
            active_scales = self._format_channel_tensor(channel_scales, "channel_scales").to(
                device=predicted_delta.device, dtype=predicted_delta.dtype
            )
        elif self.channel_scales is not None:
            active_scales = self.channel_scales

        if active_scales is not None:
            effective_delta = predicted_delta * active_scales
        else:
            effective_delta = predicted_delta

        # 2. Residual state transition
        next_state = current_state + effective_delta

        # 3. Precipitation physical constraint (non-negative precipitation)
        if self.clamp_precipitation:
            p_idx = self.precipitation_idx
            min_val = self._get_effective_precipitation_min(active_scales)
            p_curr = next_state[:, p_idx : p_idx + 1, :, :]
            p_clamped = torch.clamp(p_curr, min=min_val)

            if p_idx == 0:
                next_state = torch.cat([p_clamped, next_state[:, 1:, :, :]], dim=1)
            elif p_idx == self.num_channels - 1:
                next_state = torch.cat([next_state[:, :p_idx, :, :], p_clamped], dim=1)
            else:
                next_state = torch.cat(
                    [
                        next_state[:, :p_idx, :, :],
                        p_clamped,
                        next_state[:, p_idx + 1 :, :, :],
                    ],
                    dim=1,
                )

        return next_state

    def apply_perturbation(
        self,
        initial_state: torch.Tensor,
        perturbation: torch.Tensor,
    ) -> torch.Tensor:
        """Apply a physical perturbation to the latest initial climate state.

        Supports both batched [B, 6, H, W] and unbatched [6, H, W] tensors
        matching the competition specification:
            last = last + perturbation

        Args:
            initial_state (torch.Tensor): Initial climate state [B, 6, H, W] or [6, H, W].
            perturbation (torch.Tensor): Physical perturbation delta [B, 6, H, W] or [6, H, W].

        Returns:
            torch.Tensor: Perturbed climate state with identical shape and physical validity.
        """
        unbatched = False
        if initial_state.dim() == 3:
            initial_state = initial_state.unsqueeze(0)
            unbatched = True
        if perturbation.dim() == 3:
            perturbation = perturbation.unsqueeze(0)

        # Physical perturbations are expressed directly in physical units (e.g. +3.0 K);
        # override any registered channel scale standard deviations with unit scales.
        unit_scales = torch.ones(
            self.num_channels, device=perturbation.device, dtype=perturbation.dtype
        )
        out = self.forward(initial_state, perturbation, channel_scales=unit_scales)
        return out.squeeze(0) if unbatched else out


# Aliases for flexible imports
ClimateResidualStateTransition = ResidualStateTransition
Block6ResidualStateTransition = ResidualStateTransition
