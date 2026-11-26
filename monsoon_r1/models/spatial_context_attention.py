"""Block 4: Spatial Context & Attention Module for Multiverse of Monsoon.

Refines the 64-channel spatial feature representation produced by Block 3
(Temporal Dynamics Encoder / ConvLSTM) using a lightweight CBAM-inspired
attention mechanism.

Architecture:
1. Channel Attention:
   - Global Average Pooling across spatial dimensions: [B, 64, 32, 36] -> [B, 64, 1, 1]
   - Global Max Pooling across spatial dimensions:     [B, 64, 32, 36] -> [B, 64, 1, 1]
   - Shared MLP: 64 -> 16 -> ReLU -> 16 -> 64
   - Sum of average and max MLP outputs + Sigmoid activation
   - Channel weights [B, 64, 1, 1] reweight the original feature map.

2. Spatial Attention:
   - Channel Average Pooling across channel dimension: [B, 64, 32, 36] -> [B, 1, 32, 36]
   - Channel Max Pooling across channel dimension:     [B, 64, 32, 36] -> [B, 1, 32, 36]
   - Concatenation: 1 + 1 = 2 channels -> [B, 2, 32, 36]
   - 7x7 Convolution (padding 3, stride 1): 2 -> 1 channel + Sigmoid
   - Spatial weights [B, 1, 32, 36] reweight the channel-attended feature map.

Input shape:  [B, 64, 32, 36]
Output shape: [B, 64, 32, 36] (shape strictly preserved)
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn


class ChannelAttention(nn.Module):
    """Subcomponent: Channel Attention Module.

    Learns which feature channels are more useful for subsequent forecasting
    by aggregating spatial context via global average pooling and global max pooling,
    processing both through a shared lightweight MLP, and combining them
    with a sigmoid gate.

    Attributes:
        in_channels (int): Number of input channels (default: 64).
        reduction_channels (int): Number of reduced bottleneck channels (default: 16).
        bias (bool): Whether linear layers include additive bias (default: True).
        avg_pool (nn.AdaptiveAvgPool2d): Global spatial average pooling to [B, C, 1, 1].
        max_pool (nn.AdaptiveMaxPool2d): Global spatial max pooling to [B, C, 1, 1].
        mlp (nn.Sequential): Shared bottleneck MLP (in_channels -> reduction_channels -> in_channels).
    """

    def __init__(
        self,
        in_channels: int = 64,
        reduction_channels: int = 16,
        bias: bool = True,
    ) -> None:
        """Initialize ChannelAttention.

        Args:
            in_channels (int): Number of input feature channels (default: 64).
            reduction_channels (int): Number of bottleneck channels in shared MLP (default: 16).
            bias (bool): Whether MLP linear layers have learnable biases (default: True).
        """
        super().__init__()
        if in_channels <= 0:
            raise ValueError(f"in_channels must be positive, got {in_channels}")
        if reduction_channels <= 0:
            raise ValueError(f"reduction_channels must be positive, got {reduction_channels}")

        self.in_channels = in_channels
        self.reduction_channels = reduction_channels
        self.bias = bias

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared lightweight MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.in_channels, self.reduction_channels, bias=self.bias),
            nn.ReLU(inplace=False),
            nn.Linear(self.reduction_channels, self.in_channels, bias=self.bias),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize MLP parameters with Kaiming uniform."""
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _validate_input(self, x: torch.Tensor) -> Tuple[int, int, int, int]:
        """Validate input tensor for channel attention."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
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
        if h < 1 or w < 1:
            raise ValueError(f"Spatial dimensions must be >= 1, got ({h}, {w})")
        return b, c, h, w

    def compute_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Compute channel attention weights [B, C, 1, 1].

        Args:
            x (torch.Tensor): Feature map [B, in_channels, H, W].

        Returns:
            torch.Tensor: Channel attention weights in range [0, 1] with shape [B, in_channels, 1, 1].
        """
        b, c, _, _ = self._validate_input(x)

        # Global pooling across spatial dimensions: [B, C, 1, 1] -> [B, C]
        avg_rep = self.avg_pool(x).flatten(1)
        max_rep = self.max_pool(x).flatten(1)

        # Forward through shared MLP
        mlp_avg = self.mlp(avg_rep)
        mlp_max = self.mlp(max_rep)

        # Combine outputs and apply Sigmoid
        combined = mlp_avg + mlp_max
        weights = torch.sigmoid(combined).view(b, c, 1, 1)
        return weights

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for ChannelAttention.

        Args:
            x (torch.Tensor): Input feature tensor of shape [B, 64, H, W].
            return_weights (bool): If True, returns (reweighted_x, weights). Default: False.

        Returns:
            torch.Tensor or tuple[torch.Tensor, torch.Tensor]: Reweighted tensor of shape
                [B, 64, H, W], or (reweighted_x, weights) if return_weights is True.
        """
        weights = self.compute_weights(x)
        out = x * weights
        if return_weights:
            return out, weights
        return out


class SpatialAttention(nn.Module):
    """Subcomponent: Spatial Attention Module.

    Learns which spatial regions across the grid are more informative for
    forecasting by computing channel-wise average and max statistics,
    concatenating them into a 2-channel spatial summary map, and applying
    a 7x7 convolution followed by sigmoid gating.

    Attributes:
        kernel_size (int): Size of the 2D convolution kernel (default: 7).
        padding (int): Padding to preserve spatial dimensions (default: 3).
        bias (bool): Whether convolution includes additive bias (default: True).
        conv (nn.Conv2d): 7x7 convolution mapping 2 channels -> 1 channel.
    """

    def __init__(
        self,
        kernel_size: int = 7,
        bias: bool = True,
    ) -> None:
        """Initialize SpatialAttention.

        Args:
            kernel_size (int): Size of convolution kernel (default: 7, must be odd).
            bias (bool): Whether convolution includes additive bias (default: True).
        """
        super().__init__()
        if kernel_size % 2 == 0 or kernel_size < 1:
            raise ValueError(f"kernel_size must be a positive odd integer, got {kernel_size}")

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        # 2 channels (avg-pooled map + max-pooled map) -> 1 channel
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=self.bias,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize convolution parameters with Kaiming uniform."""
        nn.init.kaiming_uniform_(self.conv.weight, a=math.sqrt(5))
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)

    def _validate_input(self, x: torch.Tensor) -> Tuple[int, int, int, int]:
        """Validate input tensor for spatial attention."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
        if x.dim() != 4:
            raise ValueError(
                f"Expected 4D tensor [B, C, H, W], got {x.dim()}D tensor with shape {list(x.shape)}"
            )
        b, c, h, w = x.shape
        if b < 1:
            raise ValueError(f"Batch size B must be >= 1, got B={b}")
        if c < 1:
            raise ValueError(f"Channel dimension C must be >= 1, got C={c}")
        if h < self.kernel_size or w < self.kernel_size:
            raise ValueError(
                f"Spatial dimensions ({h}, {w}) must be at least kernel size ({self.kernel_size}, {self.kernel_size})"
            )
        return b, c, h, w

    def compute_weights(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spatial attention weights [B, 1, H, W].

        Args:
            x (torch.Tensor): Feature map [B, C, H, W].

        Returns:
            torch.Tensor: Spatial attention weights in range [0, 1] with shape [B, 1, H, W].
        """
        self._validate_input(x)

        # Channel-wise pooling across dimension 1
        avg_out = torch.mean(x, dim=1, keepdim=True)  # [B, 1, H, W]
        max_out = torch.max(x, dim=1, keepdim=True)[0]  # [B, 1, H, W]

        # Concatenate along channel dimension -> [B, 2, H, W]
        concat_maps = torch.cat([avg_out, max_out], dim=1)

        # 7x7 convolution and Sigmoid gating -> [B, 1, H, W]
        weights = torch.sigmoid(self.conv(concat_maps))
        return weights

    def forward(
        self,
        x: torch.Tensor,
        return_weights: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for SpatialAttention.

        Args:
            x (torch.Tensor): Input feature tensor of shape [B, C, H, W].
            return_weights (bool): If True, returns (reweighted_x, weights). Default: False.

        Returns:
            torch.Tensor or tuple[torch.Tensor, torch.Tensor]: Reweighted tensor of shape
                [B, C, H, W], or (reweighted_x, weights) if return_weights is True.
        """
        weights = self.compute_weights(x)
        out = x * weights
        if return_weights:
            return out, weights
        return out


class SpatialContextAttention(nn.Module):
    """Block 4: Spatial Context & Attention Module.

    Refines the spatial feature representation produced by Block 3 (ConvLSTM)
    using sequential Channel Attention and Spatial Attention (CBAM-inspired).

    Pipeline:
    1. Channel Attention: Reweights feature channels using global average and
       max pooling followed by a shared lightweight MLP.
    2. Spatial Attention: Reweights spatial locations using channel average and
       max pooling followed by a 7x7 convolution.

    Preserves tensor shape: [B, 64, 32, 36] -> [B, 64, 32, 36].

    Attributes:
        in_channels (int): Input feature channels from Block 3 (default: 64).
        reduction_channels (int): Bottleneck channels in shared MLP (default: 16).
        spatial_kernel_size (int): Convolution kernel size in spatial attention (default: 7).
        channel_attention (ChannelAttention): Modular Channel Attention subcomponent.
        spatial_attention (SpatialAttention): Modular Spatial Attention subcomponent.
    """

    def __init__(
        self,
        in_channels: int = 64,
        reduction_channels: int = 16,
        spatial_kernel_size: int = 7,
        bias: bool = True,
    ) -> None:
        """Initialize SpatialContextAttention.

        Args:
            in_channels (int): Number of input feature channels (default: 64).
            reduction_channels (int): Number of bottleneck channels in shared MLP (default: 16).
            spatial_kernel_size (int): Kernel size for spatial attention convolution (default: 7).
            bias (bool): Whether linear and convolutional layers include additive bias (default: True).
        """
        super().__init__()
        self.in_channels = in_channels
        self.reduction_channels = reduction_channels
        self.spatial_kernel_size = spatial_kernel_size
        self.bias = bias

        self.channel_attention = ChannelAttention(
            in_channels=self.in_channels,
            reduction_channels=self.reduction_channels,
            bias=self.bias,
        )

        self.spatial_attention = SpatialAttention(
            kernel_size=self.spatial_kernel_size,
            bias=self.bias,
        )

    def _validate_input(self, x: torch.Tensor, check_finite: bool = False) -> Tuple[int, int, int, int]:
        """Validate input tensor dimensions, types, and channel count."""
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
        if h < self.spatial_kernel_size or w < self.spatial_kernel_size:
            raise ValueError(
                f"Spatial dimensions ({h}, {w}) must be at least kernel size "
                f"({self.spatial_kernel_size}, {self.spatial_kernel_size})"
            )
        return b, c, h, w

    def forward(
        self,
        x: torch.Tensor,
        return_attention_maps: bool = False,
        check_finite: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """Forward pass of SpatialContextAttention.

        Applies channel attention followed by spatial attention to refine
        the feature representation while strictly preserving the tensor shape.

        Args:
            x (torch.Tensor): Input spatial feature tensor [B, 64, H, W] from Block 3.
            return_attention_maps (bool): If True, returns a tuple (refined_x, attention_maps_dict).
                Default: False.
            check_finite (bool): If True, validates that input contains no NaN or Inf values.

        Returns:
            torch.Tensor or tuple[torch.Tensor, dict[str, torch.Tensor]]:
                - If return_attention_maps is False (default): [B, 64, H, W]
                - If return_attention_maps is True:
                  (refined_x, {"channel_weights": [B, 64, 1, 1], "spatial_weights": [B, 1, H, W]})
        """
        self._validate_input(x, check_finite=check_finite)

        # 1. Apply Channel Attention
        x_channel, channel_weights = self.channel_attention(x, return_weights=True)

        # 2. Apply Spatial Attention to channel-attended features
        x_out, spatial_weights = self.spatial_attention(x_channel, return_weights=True)

        if return_attention_maps:
            attention_maps = {
                "channel_weights": channel_weights,
                "spatial_weights": spatial_weights,
            }
            return x_out, attention_maps

        return x_out


# Aliases for flexible imports
SpatialContextAttentionModule = SpatialContextAttention
CBAM = SpatialContextAttention
