"""Block 1: Climate Input Encoder for Multiverse of Monsoon.

Transforms raw climate data, static geographic information, and calendar context
into a learned 48-feature channel representation per grid cell across a sequence of days.

Architecture:
- Subcomponent A: ClimateProjection (6 -> 32 channels via 1x1 conv)
- Subcomponent B: GeographicProjection (2 -> 8 channels via 1x1 conv, broadcast across sequence T)
- Subcomponent C: CalendarProjection (2 -> 8 channels via linear projection, broadcast across spatial grid HxW and sequence T)
- Subcomponent D: Channel Concatenation (32 + 8 + 8 = 48 channels)

Output shape: [B, T, 48, H, W]
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class ClimateProjection(nn.Module):
    """Subcomponent A: Climate Projection Layer.

    Projects the 6 dynamic climate variables (t2m, tp, u10, v10, msl, tcwv)
    into 32 learned feature channels using a 1x1 2D convolution.
    
    This operates strictly on each grid cell individually across the input
    sequence and does NOT perform spatial neighborhood extraction (which belongs
    to Block 2: Multi-Scale Spatial Encoder).

    Attributes:
        in_channels (int): Number of climate channels (default: 6).
        out_channels (int): Number of projected feature channels (default: 32).
        conv (nn.Conv2d): 1x1 convolution mapping in_channels to out_channels.
        activation (nn.Module): Optional activation layer.
    """

    def __init__(
        self,
        in_channels: int = 6,
        out_channels: int = 32,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for climate projection.

        Args:
            x (torch.Tensor): Dynamic climate tensor of shape [B, T, in_channels, H, W].

        Returns:
            torch.Tensor: Projected climate features of shape [B, T, out_channels, H, W].
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
        if x.dim() != 5:
            raise ValueError(f"Expected 5D tensor [B, T, C, H, W], got shape {list(x.shape)}")
        if x.size(2) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} climate channels at dim 2, got {x.size(2)} in shape {list(x.shape)}"
            )

        b, t, c, h, w = x.shape
        # Reshape [B, T, C, H, W] -> [B * T, C, H, W] for 2D 1x1 conv
        flat_x = x.contiguous().view(b * t, c, h, w)
        out = self.conv(flat_x)
        out = self.activation(out)
        # Reshape back to [B, T, out_channels, H, W]
        return out.view(b, t, self.out_channels, h, w)


class GeographicProjection(nn.Module):
    """Subcomponent B: Geographic Projection Layer.

    Projects the 2 static geographic variables (land_sea_mask, orography_m)
    into 8 learned feature channels using a 1x1 2D convolution.
    
    The resulting spatial representation is broadcast across all T time steps
    of the sequence, ensuring the static geographic context is identically
    available for each day.

    Attributes:
        in_channels (int): Number of static geographic channels (default: 2).
        out_channels (int): Number of projected geographic channels (default: 8).
        conv (nn.Conv2d): 1x1 convolution mapping in_channels to out_channels.
        activation (nn.Module): Optional activation layer.
    """

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 8,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=True)
        self.activation = activation if activation is not None else nn.Identity()

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Forward pass for geographic projection.

        Args:
            x (torch.Tensor): Geographic tensor of shape [B, in_channels, H, W]
                or [B, 1, in_channels, H, W].
            seq_len (int): Desired sequence length T to broadcast across.

        Returns:
            torch.Tensor: Projected geographic features broadcast across T,
                of shape [B, T, out_channels, H, W].
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
        if seq_len < 1:
            raise ValueError(f"seq_len must be positive, got {seq_len}")

        # Squeeze optional trivial time dim: [B, 1, C, H, W] -> [B, C, H, W]
        if x.dim() == 5:
            if x.size(1) != 1:
                raise ValueError(
                    f"Geographic tensor with 5 dims must have time dimension of 1, got shape {list(x.shape)}"
                )
            x = x.squeeze(1)

        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W], got shape {list(x.shape)}")
        if x.size(1) != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} geographic channels at dim 1, got {x.size(1)} in shape {list(x.shape)}"
            )

        b, c, h, w = x.shape
        proj = self.conv(x)  # [B, out_channels, H, W]
        proj = self.activation(proj)

        # Broadcast across time steps: [B, out_channels, H, W] -> [B, T, out_channels, H, W]
        proj_seq = proj.unsqueeze(1).expand(b, seq_len, self.out_channels, h, w)
        return proj_seq


class CalendarProjection(nn.Module):
    """Subcomponent C: Calendar Projection Layer.

    Projects the 2 doy_sincos values [sin(2*pi*doy/365), cos(2*pi*doy/365)]
    into 8 learned calendar feature values using a learnable linear layer.
    
    The learned features are broadcast across the spatial grid [H, W] and
    across the sequence dimension [T].

    Calendar conventions supported:
    1. Single day calendar of shape [B, 2] (representing the last input day):
       - If `derive_preceding_days=False` (default): The learned calendar features
         are broadcast across all T steps and all HxW cells.
       - If `derive_preceding_days=True`: The preceding (T-1) days' [sin, cos]
         coordinates are derived backwards along the 365-day unit circle using
         exact trigonometric angle subtraction formulas, projected per day,
         and broadcast across HxW cells.
    2. Per-day calendar sequence of shape [B, T, 2]:
       - Each day's [sin, cos] values are projected to 8 features and broadcast
         across HxW cells.

    Attributes:
        in_features (int): Number of calendar values (default: 2).
        out_features (int): Number of projected features (default: 8).
        derive_preceding_days (bool): Whether to derive preceding days from last day.
        linear (nn.Linear): Linear layer mapping in_features to out_features.
        activation (nn.Module): Optional activation layer.
    """

    def __init__(
        self,
        in_features: int = 2,
        out_features: int = 8,
        derive_preceding_days: bool = False,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.derive_preceding_days = derive_preceding_days
        self.linear = nn.Linear(in_features, out_features, bias=True)
        self.activation = activation if activation is not None else nn.Identity()

    def _derive_preceding_sincos(self, cal_last: torch.Tensor, seq_len: int) -> torch.Tensor:
        """Derive preceding days' [sin, cos] from the last day's sincos.

        According to the project contract, doy_sincos = [sin(2*pi*doy/365), cos(2*pi*doy/365)].
        For sequence length T where the last day is index T-1:
        Offset for step t is dt = t - (T - 1) in days (e.g. -2, -1, 0 for T=3).
        Angle shift is delta = dt * (2*pi / 365.0).
        Using angle addition:
          sin(theta + delta) = sin(theta)*cos(delta) + cos(theta)*sin(delta)
          cos(theta + delta) = cos(theta)*cos(delta) - sin(theta)*sin(delta)

        This is exact, differentiable, handles circular year wraparound naturally,
        and avoids arbitrary heuristics.
        """
        b = cal_last.size(0)
        sin_last = cal_last[:, 0]  # [B]
        cos_last = cal_last[:, 1]  # [B]

        derived_days = []
        alpha = 2.0 * math.pi / 365.0

        for t in range(seq_len):
            dt = t - (seq_len - 1)
            delta = dt * alpha
            cos_delta = math.cos(delta)
            sin_delta = math.sin(delta)

            # Convert scalars to tensors on the same device and dtype as cal_last
            cos_d_tensor = torch.tensor(cos_delta, device=cal_last.device, dtype=cal_last.dtype)
            sin_d_tensor = torch.tensor(sin_delta, device=cal_last.device, dtype=cal_last.dtype)

            sin_t = sin_last * cos_d_tensor + cos_last * sin_d_tensor
            cos_t = cos_last * cos_d_tensor - sin_last * sin_d_tensor
            day_t = torch.stack([sin_t, cos_t], dim=-1)  # [B, 2]
            derived_days.append(day_t)

        return torch.stack(derived_days, dim=1)  # [B, T, 2]

    def forward(
        self,
        x: torch.Tensor,
        seq_len: int,
        height: int,
        width: int,
    ) -> torch.Tensor:
        """Forward pass for calendar projection and broadcast.

        Args:
            x (torch.Tensor): Calendar tensor of shape [B, 2] or [B, T, 2].
            seq_len (int): Sequence length T.
            height (int): Grid height H (e.g. 32).
            width (int): Grid width W (e.g. 36).

        Returns:
            torch.Tensor: Calendar features broadcast across [B, T, 8, H, W].
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
        if seq_len < 1:
            raise ValueError(f"seq_len must be positive, got {seq_len}")
        if height < 1 or width < 1:
            raise ValueError(f"height and width must be positive, got ({height}, {width})")

        if x.dim() == 2:
            # Shape [B, 2]
            if x.size(1) != self.in_features:
                raise ValueError(
                    f"Expected {self.in_features} calendar features at dim 1, got {x.size(1)} in shape {list(x.shape)}"
                )
            b = x.size(0)

            if self.derive_preceding_days and seq_len > 1:
                # Derive per-day [B, T, 2]
                x_seq = self._derive_preceding_sincos(x, seq_len)  # [B, T, 2]
                proj = self.linear(x_seq)  # [B, T, out_features]
                proj = self.activation(proj)
                # Reshape to [B, T, out_features, 1, 1] and broadcast to [B, T, out_features, H, W]
                proj = proj.unsqueeze(-1).unsqueeze(-1).expand(b, seq_len, self.out_features, height, width)
                return proj
            else:
                # Broadcast single day projection across all T steps
                proj = self.linear(x)  # [B, out_features]
                proj = self.activation(proj)
                # [B, out_features] -> [B, 1, out_features, 1, 1] -> [B, T, out_features, H, W]
                proj = proj.view(b, 1, self.out_features, 1, 1).expand(b, seq_len, self.out_features, height, width)
                return proj

        elif x.dim() == 3:
            # Shape [B, T_in, 2]
            b, t_in, d = x.shape
            if d != self.in_features:
                raise ValueError(
                    f"Expected {self.in_features} calendar features at dim 2, got {d} in shape {list(x.shape)}"
                )
            if t_in != seq_len:
                raise ValueError(
                    f"Calendar sequence length {t_in} does not match expected sequence length {seq_len}"
                )

            proj = self.linear(x)  # [B, T, out_features]
            proj = self.activation(proj)
            # [B, T, out_features] -> [B, T, out_features, 1, 1] -> [B, T, out_features, H, W]
            proj = proj.unsqueeze(-1).unsqueeze(-1).expand(b, seq_len, self.out_features, height, width)
            return proj

        else:
            raise ValueError(f"Expected 2D [B, 2] or 3D [B, T, 2] calendar tensor, got shape {list(x.shape)}")


class ClimateInputEncoder(nn.Module):
    """Block 1: Climate Input Encoder.

    Combines dynamic climate variables, static geographic features, and calendar
    context into a single learned representation of 48 channels per grid cell
    for each day in the sequence.

    Pipeline:
    1. Climate Projection: [B, T, 6, H, W] -> [B, T, 32, H, W] (1x1 conv)
    2. Geographic Projection: [B, 2, H, W] -> [B, T, 8, H, W] (1x1 conv + broadcast)
    3. Calendar Projection: [B, 2] or [B, T, 2] -> [B, T, 8, H, W] (linear + broadcast)
    4. Concatenation: Channel-wise concat -> [B, T, 48, H, W]

    Total channels: 32 (climate) + 8 (geographic) + 8 (calendar) = 48.

    Attributes:
        climate_proj (ClimateProjection): Subcomponent A projection module.
        geo_proj (GeographicProjection): Subcomponent B projection module.
        cal_proj (CalendarProjection): Subcomponent C projection module.
    """

    def __init__(
        self,
        climate_channels: int = 6,
        climate_features: int = 32,
        geo_channels: int = 2,
        geo_features: int = 8,
        cal_channels: int = 2,
        cal_features: int = 8,
        derive_preceding_days: bool = False,
        activation: Optional[nn.Module] = None,
    ) -> None:
        """Initialize ClimateInputEncoder.

        Args:
            climate_channels (int): Input dynamic climate channels (default: 6).
            climate_features (int): Learned climate feature channels (default: 32).
            geo_channels (int): Input static geographic channels (default: 2).
            geo_features (int): Learned geographic feature channels (default: 8).
            cal_channels (int): Input calendar feature count (default: 2).
            cal_features (int): Learned calendar feature channels (default: 8).
            derive_preceding_days (bool): Whether to derive preceding days from last
                day doy_sincos (default: False).
            activation (nn.Module, optional): Optional activation applied after
                each projection. Defaults to None (linear projections).
        """
        super().__init__()
        self.climate_channels = climate_channels
        self.climate_features = climate_features
        self.geo_channels = geo_channels
        self.geo_features = geo_features
        self.cal_channels = cal_channels
        self.cal_features = cal_features
        self.total_features = climate_features + geo_features + cal_features

        self.climate_proj = ClimateProjection(
            in_channels=climate_channels,
            out_channels=climate_features,
            activation=activation,
        )
        self.geo_proj = GeographicProjection(
            in_channels=geo_channels,
            out_channels=geo_features,
            activation=activation,
        )
        self.cal_proj = CalendarProjection(
            in_features=cal_channels,
            out_features=cal_features,
            derive_preceding_days=derive_preceding_days,
            activation=activation,
        )

    def forward(
        self,
        climate: torch.Tensor,
        geography: torch.Tensor,
        calendar: torch.Tensor,
        check_finite: bool = False,
    ) -> torch.Tensor:
        """Forward pass of ClimateInputEncoder.

        Args:
            climate (torch.Tensor): Dynamic climate tensor of shape [B, T, 6, H, W].
            geography (torch.Tensor): Static geographic tensor of shape [B, 2, H, W],
                [B, 1, 2, H, W], or [1, 2, H, W] / [2, H, W] (auto-broadcast to batch size B).
            calendar (torch.Tensor): Calendar tensor of shape [B, 2], [B, T, 2],
                or [2] / [1, 2] (auto-broadcast to batch size B).
            check_finite (bool): If True, checks that inputs contain no NaN or Inf values
                and raises ValueError if any are found. Default: False.

        Returns:
            torch.Tensor: Encoded spatiotemporal feature tensor of shape
                [B, T, 48, H, W].
        """
        # Validate inputs
        if not isinstance(climate, torch.Tensor):
            raise TypeError(f"climate must be a torch.Tensor, got {type(climate).__name__}")
        if not isinstance(geography, torch.Tensor):
            raise TypeError(f"geography must be a torch.Tensor, got {type(geography).__name__}")
        if not isinstance(calendar, torch.Tensor):
            raise TypeError(f"calendar must be a torch.Tensor, got {type(calendar).__name__}")

        if check_finite:
            if not torch.isfinite(climate).all():
                raise ValueError("climate tensor contains NaN or Inf values")
            if not torch.isfinite(geography).all():
                raise ValueError("geography tensor contains NaN or Inf values")
            if not torch.isfinite(calendar).all():
                raise ValueError("calendar tensor contains NaN or Inf values")

        if climate.dim() != 5:
            raise ValueError(f"climate must be a 5D tensor [B, T, C, H, W], got shape {list(climate.shape)}")
        if climate.size(2) != self.climate_channels:
            raise ValueError(
                f"climate channel count must be {self.climate_channels}, got {climate.size(2)}"
            )

        b, t, _, h, w = climate.shape

        # Validate calendar dimension
        if calendar.dim() not in (2, 3):
            raise ValueError(
                f"Expected 2D [B, 2] or 3D [B, T, 2] calendar tensor, got shape {list(calendar.shape)}"
            )

        # Support unbatched 3D geography [C, H, W] -> [1, C, H, W]
        if geography.dim() == 3:
            geography = geography.unsqueeze(0)

        # Support single-batch static geography auto-broadcasting
        if geography.dim() == 4 and geography.size(0) == 1 and b > 1:
            geography = geography.expand(b, -1, -1, -1)
        elif geography.size(0) != b:
            raise ValueError(
                f"geography batch size {geography.size(0)} does not match climate batch size {b}"
            )

        # Support single-batch calendar auto-broadcasting
        if calendar.size(0) == 1 and b > 1:
            if calendar.dim() == 2:
                calendar = calendar.expand(b, -1)
            elif calendar.dim() == 3:
                calendar = calendar.expand(b, -1, -1)
        elif calendar.size(0) != b:
            raise ValueError(
                f"calendar batch size {calendar.size(0)} does not match climate batch size {b}"
            )

        # Validate spatial alignment
        geo_h = geography.size(-2)
        geo_w = geography.size(-1)
        if geo_h != h or geo_w != w:
            raise ValueError(
                f"geography spatial dimensions ({geo_h}, {geo_w}) do not match climate ({h}, {w})"
            )

        # 1. Project dynamic climate variables -> [B, T, 32, H, W]
        c_feat = self.climate_proj(climate)

        # 2. Project static geographic variables -> [B, T, 8, H, W]
        g_feat = self.geo_proj(geography, seq_len=t)

        # 3. Project calendar context -> [B, T, 8, H, W]
        k_feat = self.cal_proj(calendar, seq_len=t, height=h, width=w)

        # 4. Concatenate along channel dimension (dim=2) -> [B, T, 48, H, W]
        out = torch.cat([c_feat, g_feat, k_feat], dim=2)
        return out
