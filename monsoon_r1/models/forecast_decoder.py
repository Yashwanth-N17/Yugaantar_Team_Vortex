"""Block 5: Forecast Decoder for Multiverse of Monsoon.

Transforms the refined 64-channel spatial feature representation produced by Block 4
(Spatial Context & Attention Module) into six spatial prediction maps corresponding
to the six core climate variables.

The decoder predicts atmospheric state changes (deltas) for the next time step.
The predicted deltas are intended for consumption by Block 6 (Residual State Transition),
which combines them with the current climate state to produce the next state.

Architecture:
- Layer 1 (Feature Refinement):
    Conv2d: 64 -> 64 channels, kernel 3x3, stride 1, padding 1 + Activation (ReLU)
    Refines and deepens the attended feature representation before decoding.
- Layer 2 (Feature Reduction):
    Conv2d: 64 -> 32 channels, kernel 3x3, stride 1, padding 1 + Activation (ReLU)
    Compresses feature channels while retaining spatial structure.
- Layer 3 (Climate Output Projection):
    Conv2d: 32 -> 6 channels, kernel 1x1, stride 1, padding 0 (linear projection)
    Projects latent features onto the 6 climate variable maps.

Input shape:  [B, 64, 32, 36] (from Block 4)
Output shape: [B, 6, 32, 36] (prediction maps for 6 climate variables)

The 6 output channels correspond strictly to:
  0: t2m  - Temperature at 2 metres (K)
  1: tp   - Total precipitation (mm/day)
  2: u10  - 10-metre zonal wind (m/s)
  3: v10  - 10-metre meridional wind (m/s)
  4: msl  - Mean sea-level pressure (Pa)
  5: tcwv - Total column water vapour (kg/m²)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from monsoon_r1.contract import CHANNELS, NC, NLAT, NLON


class ForecastDecoder(nn.Module):
    """Block 5: Forecast Decoder.

    Decodes refined spatial-temporal features from Block 4 into 6 spatial
    prediction maps (climate-state changes / deltas).

    Attributes:
        in_channels (int): Input feature channels from Block 4 (default: 64).
        hidden_channels (int): Intermediate channels for Layer 1 (default: 64).
        reduced_channels (int): Intermediate channels for Layer 2 (default: 32).
        out_channels (int): Output climate variable channels (default: 6).
        bias (bool): Whether convolutions include additive bias (default: True).
        layer1 (nn.Sequential): Layer 1 (Feature refinement: 64 -> 64, 3x3 conv + ReLU).
        layer2 (nn.Sequential): Layer 2 (Feature reduction: 64 -> 32, 3x3 conv + ReLU).
        layer3 (nn.Conv2d): Layer 3 (Output projection: 32 -> 6, 1x1 conv).
        refinement (nn.Sequential): Alias for layer1.
        reduction (nn.Sequential): Alias for layer2.
        projection (nn.Conv2d): Alias for layer3.
    """

    CHANNELS: Tuple[str, ...] = CHANNELS
    TARGET_VARIABLES: Tuple[str, ...] = CHANNELS
    NUM_CHANNELS: int = NC
    DEFAULT_HEIGHT: int = NLAT
    DEFAULT_WIDTH: int = NLON

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 64,
        reduced_channels: int = 32,
        out_channels: int = 6,
        activation: Optional[nn.Module] = None,
        bias: bool = True,
    ) -> None:
        """Initialize ForecastDecoder.

        Args:
            in_channels (int): Number of input channels from Block 4 (default: 64).
            hidden_channels (int): Feature refinement channels in Layer 1 (default: 64).
            reduced_channels (int): Feature reduction channels in Layer 2 (default: 32).
            out_channels (int): Number of output climate channels (default: 6).
            activation (nn.Module, optional): Activation function for Layers 1 and 2.
                Defaults to nn.ReLU(inplace=False).
            bias (bool): Whether convolutional layers include bias (default: True).
        """
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if hidden_channels <= 0:
            raise ValueError(f"hidden_channels must be positive, got {hidden_channels}")
        if reduced_channels <= 0:
            raise ValueError(f"reduced_channels must be positive, got {reduced_channels}")
        if out_channels <= 0:
            raise ValueError(f"out_channels must be positive, got {out_channels}")

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.reduced_channels = reduced_channels
        self.out_channels = out_channels
        self.bias = bias

        act_fn = activation if activation is not None else nn.ReLU(inplace=False)

        # Layer 1 — Feature refinement (64 -> 64, 3x3 conv, stride 1, padding 1 + activation)
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.in_channels,
                out_channels=self.hidden_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=self.bias,
            ),
            self._clone_activation(act_fn),
        )

        # Layer 2 — Feature reduction (64 -> 32, 3x3 conv, stride 1, padding 1 + activation)
        self.layer2 = nn.Sequential(
            nn.Conv2d(
                in_channels=self.hidden_channels,
                out_channels=self.reduced_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=self.bias,
            ),
            self._clone_activation(act_fn),
        )

        # Layer 3 — Climate output projection (32 -> 6, 1x1 conv, stride 1, padding 0)
        # Linear projection without activation so output deltas can cover the full dynamic range
        self.layer3 = nn.Conv2d(
            in_channels=self.reduced_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=self.bias,
        )

        # Structural aliases for semantic clarity
        self.refinement = self.layer1
        self.reduction = self.layer2
        self.projection = self.layer3

        self.reset_parameters()

    @staticmethod
    def _clone_activation(act: nn.Module) -> nn.Module:
        """Helper to create a fresh instance of activation module if needed."""
        cls = act.__class__
        try:
            return cls()
        except TypeError:
            return act

    def reset_parameters(self) -> None:
        """Initialize convolution parameters with Kaiming uniform."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _validate_input(self, x: torch.Tensor, check_finite: bool = False) -> Tuple[int, int, int, int]:
        """Validate input tensor shape, type, and dimensions.

        Args:
            x (torch.Tensor): Input tensor.
            check_finite (bool): If True, validates no NaN or Inf values.

        Returns:
            tuple[int, int, int, int]: Extracted (B, C, H, W) dimensions.

        Raises:
            TypeError: If input is not a torch.Tensor.
            ValueError: If input tensor does not have 4 dimensions, has empty batch,
                has mismatched channel count, spatial dimensions are smaller
                than the 3x3 convolution kernel, or contains NaN/Inf (when check_finite=True).
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
        if check_finite and not torch.isfinite(x).all():
            raise ValueError("Input tensor x contains NaN or Inf values")
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor [B, C, H, W], got {x.dim()}D tensor with shape {list(x.shape)}"
            )

        b, c, h, w = x.shape
        if b < 1:
            raise ValueError(f"Batch size B must be >= 1, got B={b}")
        if c != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} channels at dim 1, got {c} in shape {list(x.shape)}"
            )
        if h < 3 or w < 3:
            raise ValueError(
                f"Spatial dimensions ({h}, {w}) must be at least 3x3 for 3x3 convolutions, "
                f"got ({h}, {w})"
            )
        return b, c, h, w

    def forward_refinement(self, x: torch.Tensor) -> torch.Tensor:
        """Execute Layer 1 (Feature Refinement) only.

        Args:
            x (torch.Tensor): Feature map [B, 64, H, W].

        Returns:
            torch.Tensor: Refined feature map [B, 64, H, W].
        """
        self._validate_input(x)
        return self.layer1(x)

    def forward_reduction(self, x: torch.Tensor) -> torch.Tensor:
        """Execute Layer 2 (Feature Reduction) on refined features.

        Args:
            x (torch.Tensor): Refined feature map [B, 64, H, W].

        Returns:
            torch.Tensor: Reduced feature map [B, 32, H, W].
        """
        if x.dim() != 4 or x.size(1) != self.hidden_channels:
            raise ValueError(
                f"Expected 4D tensor with {self.hidden_channels} channels, got shape {list(x.shape)}"
            )
        return self.layer2(x)

    def forward_projection(self, x: torch.Tensor) -> torch.Tensor:
        """Execute Layer 3 (Climate Output Projection) on reduced features.

        Args:
            x (torch.Tensor): Reduced feature map [B, 32, H, W].

        Returns:
            torch.Tensor: Climate prediction maps [B, 6, H, W].
        """
        if x.dim() != 4 or x.size(1) != self.reduced_channels:
            raise ValueError(
                f"Expected 4D tensor with {self.reduced_channels} channels, got shape {list(x.shape)}"
            )
        return self.layer3(x)

    def forward(
        self,
        x: torch.Tensor,
        return_intermediates: bool = False,
        check_finite: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass of Block 5: Forecast Decoder.

        Passes input features through Layer 1 (refinement), Layer 2 (reduction),
        and Layer 3 (climate projection).

        Args:
            x (torch.Tensor): Input feature tensor of shape [B, 64, H, W] from Block 4.
            return_intermediates (bool): If True, returns a tuple of
                (predictions, dict_of_intermediate_layers). Default: False.
            check_finite (bool): If True, validates no NaN or Inf in inputs. Default: False.

        Returns:
            torch.Tensor or tuple[torch.Tensor, dict[str, torch.Tensor]]:
                - If return_intermediates is False (default):
                    Output tensor of shape [B, 6, H, W] containing predicted deltas
                    ordered strictly as (t2m, tp, u10, v10, msl, tcwv).
                - If return_intermediates is True:
                    (predictions, {
                        "layer1_refinement": feat1,  # [B, 64, H, W]
                        "layer2_reduction": feat2,   # [B, 32, H, W]
                        "layer3_projection": out,    # [B, 6, H, W]
                    })
        """
        self._validate_input(x, check_finite=check_finite)

        # Layer 1 — Feature refinement [B, 64, H, W] -> [B, 64, H, W]
        feat1 = self.layer1(x)

        # Layer 2 — Feature reduction [B, 64, H, W] -> [B, 32, H, W]
        feat2 = self.layer2(feat1)

        # Layer 3 — Climate output projection [B, 32, H, W] -> [B, 6, H, W]
        out = self.layer3(feat2)

        if return_intermediates:
            intermediates = {
                "layer1_refinement": feat1,
                "layer2_reduction": feat2,
                "layer3_projection": out,
            }
            return out, intermediates

        return out


# Aliases for flexible imports
ClimateForecastDecoder = ForecastDecoder
Block5ForecastDecoder = ForecastDecoder
