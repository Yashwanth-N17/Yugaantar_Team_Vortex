"""Block 2: Multi-Scale Spatial Encoder for Multiverse of Monsoon.

Transforms the 48-channel feature representation from Block 1 into a 64-channel
spatial feature representation by extracting multi-scale patterns across neighbouring
grid cells.

Architecture:
- Branch A (Local): 3x3 conv (padding 1, stride 1), 48 -> 16 channels + activation
- Branch B (Broader): 5x5 conv (padding 2, stride 1), 48 -> 16 channels + activation
- Branch C (Dilated): 3x3 conv (dilation 2, padding 2, stride 1), 48 -> 16 channels + activation
- Fusion: Channel-wise concat (16 + 16 + 16 = 48) -> 1x1 conv (48 -> 64 channels) + activation

Output shape: [B, T, 64, H, W]
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class MultiScaleSpatialEncoder(nn.Module):
    """Block 2: Multi-Scale Spatial Encoder.

    Processes each day's 48-channel representation from Block 1 (Climate Input Encoder)
    through three parallel convolutional branches at different spatial scales,
    concatenates the multi-scale representations, and fuses them into 64 channels
    via a 1x1 convolution.

    Attributes:
        in_channels (int): Input feature channels from Block 1 (default: 48).
        branch_channels (int): Output channels for each branch (default: 16).
        out_channels (int): Final fused output feature channels (default: 64).
        branch_a (nn.Sequential): 3x3 standard convolution for local features.
        branch_b (nn.Sequential): 5x5 standard convolution for broader neighborhood.
        branch_c (nn.Sequential): 3x3 dilated convolution (dilation=2) for wide context.
        fusion (nn.Sequential): 1x1 convolution mapping 48 -> 64 channels.
    """

    def __init__(
        self,
        in_channels: int = 48,
        branch_channels: int = 16,
        out_channels: int = 64,
        activation: Optional[nn.Module] = None,
        bias: bool = True,
    ) -> None:
        """Initialize MultiScaleSpatialEncoder.

        Args:
            in_channels (int): Number of input channels (default: 48).
            branch_channels (int): Number of output channels per branch (default: 16).
            out_channels (int): Number of fused output channels (default: 64).
            activation (nn.Module, optional): Activation function applied after
                each convolution. Defaults to nn.ReLU().
            bias (bool): Whether convolutions include additive bias (default: True).
        """
        super().__init__()
        self.in_channels = in_channels
        self.branch_channels = branch_channels
        self.out_channels = out_channels
        self.total_branch_channels = branch_channels * 3

        # Default activation is ReLU as specified in architecture design
        act_fn = activation if activation is not None else nn.ReLU()

        # Branch A: Local Spatial Features (3x3 conv, padding 1, stride 1)
        self.branch_a = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=branch_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=bias,
            ),
            self._clone_activation(act_fn),
        )

        # Branch B: Broader Spatial Features (5x5 conv, padding 2, stride 1)
        self.branch_b = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=branch_channels,
                kernel_size=5,
                stride=1,
                padding=2,
                bias=bias,
            ),
            self._clone_activation(act_fn),
        )

        # Branch C: Dilated Spatial Features (3x3 conv, dilation 2, padding 2, stride 1)
        self.branch_c = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=branch_channels,
                kernel_size=3,
                stride=1,
                padding=2,
                dilation=2,
                bias=bias,
            ),
            self._clone_activation(act_fn),
        )

        # Multi-Scale Fusion: 1x1 conv mapping 48 (16+16+16) -> 64 channels
        self.fusion = nn.Sequential(
            nn.Conv2d(
                in_channels=self.total_branch_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=bias,
            ),
            self._clone_activation(act_fn),
        )

    @staticmethod
    def _clone_activation(act: nn.Module) -> nn.Module:
        """Helper to create fresh instance of activation module if needed."""
        cls = act.__class__
        try:
            return cls()
        except TypeError:
            return act

    def _validate_input(self, x: torch.Tensor, check_finite: bool = False) -> tuple[int, int, int, int, int]:
        """Validate input tensor and return shape dimensions."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
        if check_finite and not torch.isfinite(x).all():
            raise ValueError("Input tensor x contains NaN or Inf values")
        if x.dim() != 5:
            raise ValueError(
                f"Expected 5D tensor [B, T, C, H, W], got {x.dim()}D tensor with shape {list(x.shape)}"
            )
        b, t, c, h, w = x.shape
        if b < 1 or t < 1:
            raise ValueError(f"Batch size and sequence length must be >= 1, got B={b}, T={t}")
        if c != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels at dim 2, got {c} in shape {list(x.shape)}"
            )
        if h < 5 or w < 5:
            raise ValueError(
                f"Grid spatial dimensions ({h}, {w}) must be at least 5x5 for 5x5 and dilated 3x3 convolutions"
            )
        return b, t, c, h, w

    def forward_branch_a(self, x: torch.Tensor) -> torch.Tensor:
        """Run Branch A only on 5D input [B, T, C, H, W].

        Returns:
            torch.Tensor: Feature map of shape [B, T, 16, H, W].
        """
        b, t, c, h, w = self._validate_input(x)
        flat_x = x.contiguous().view(b * t, c, h, w)
        out = self.branch_a(flat_x)
        return out.view(b, t, self.branch_channels, h, w)

    def forward_branch_b(self, x: torch.Tensor) -> torch.Tensor:
        """Run Branch B only on 5D input [B, T, C, H, W].

        Returns:
            torch.Tensor: Feature map of shape [B, T, 16, H, W].
        """
        b, t, c, h, w = self._validate_input(x)
        flat_x = x.contiguous().view(b * t, c, h, w)
        out = self.branch_b(flat_x)
        return out.view(b, t, self.branch_channels, h, w)

    def forward_branch_c(self, x: torch.Tensor) -> torch.Tensor:
        """Run Branch C only on 5D input [B, T, C, H, W].

        Returns:
            torch.Tensor: Feature map of shape [B, T, 16, H, W].
        """
        b, t, c, h, w = self._validate_input(x)
        flat_x = x.contiguous().view(b * t, c, h, w)
        out = self.branch_c(flat_x)
        return out.view(b, t, self.branch_channels, h, w)

    def fuse(self, concatenated_features: torch.Tensor) -> torch.Tensor:
        """Apply fusion layer to concatenated branch features.

        Args:
            concatenated_features (torch.Tensor): Feature map of shape [B, T, 48, H, W].

        Returns:
            torch.Tensor: Fused feature map of shape [B, T, 64, H, W].
        """
        if not isinstance(concatenated_features, torch.Tensor):
            raise TypeError(f"concatenated_features must be a torch.Tensor, got {type(concatenated_features).__name__}")
        if concatenated_features.dim() != 5:
            raise ValueError(f"Expected 5D tensor [B, T, 48, H, W], got shape {list(concatenated_features.shape)}")
        if concatenated_features.size(2) != self.total_branch_channels:
            raise ValueError(
                f"Expected {self.total_branch_channels} channels, got {concatenated_features.size(2)}"
            )

        b, t, c, h, w = concatenated_features.shape
        flat = concatenated_features.contiguous().view(b * t, c, h, w)
        out = self.fusion(flat)
        return out.view(b, t, self.out_channels, h, w)

    def forward(self, x: torch.Tensor, check_finite: bool = False) -> torch.Tensor:
        """Forward pass of MultiScaleSpatialEncoder.

        Args:
            x (torch.Tensor): Input tensor from Block 1 of shape [B, T, 48, H, W].
            check_finite (bool): If True, validates that input contains no NaN or Inf values.

        Returns:
            torch.Tensor: Multi-scale spatial representation of shape [B, T, 64, H, W].
        """
        b, t, c, h, w = self._validate_input(x, check_finite=check_finite)

        # Flatten B and T to process daily spatial slices: [B * T, 48, H, W]
        flat_x = x.contiguous().view(b * t, c, h, w)

        # 1. Branch A: Local spatial features (3x3 conv) -> [B*T, 16, H, W]
        feat_a = self.branch_a(flat_x)

        # 2. Branch B: Broader spatial features (5x5 conv) -> [B*T, 16, H, W]
        feat_b = self.branch_b(flat_x)

        # 3. Branch C: Dilated spatial features (3x3 conv, d=2) -> [B*T, 16, H, W]
        feat_c = self.branch_c(flat_x)

        # 4. Multi-Scale Concatenation along channel dimension: [B*T, 48, H, W]
        feat_cat = torch.cat([feat_a, feat_b, feat_c], dim=1)

        # 5. Fusion: 1x1 conv mapping 48 -> 64 channels: [B*T, 64, H, W]
        feat_fused = self.fusion(feat_cat)

        # Reshape back to [B, T, 64, H, W]
        return feat_fused.view(b, t, self.out_channels, h, w)
