"""Block 3: Temporal Dynamics Encoder (ConvLSTM) for Multiverse of Monsoon.

Learns how spatial climate patterns evolve across the input days by processing
the sequence of spatial feature maps produced by Block 2 (Multi-Scale Spatial Encoder).

Architecture:
- ConvLSTMCell: A single-step 2D Convolutional LSTM cell operating on 2D spatial feature maps.
- TemporalDynamicsEncoder (ConvLSTM): Processes a sequence of spatial feature maps
  sequentially, updating hidden and cell states over time, and returns the final
  temporal hidden-state representation.

Input shape:  [B, T, 64, H, W]
Output shape: [B, 64, H, W] (final hidden state)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Subcomponent: 2D Convolutional LSTM Cell.

    Applies convolutional operations across both the current input feature map
    and the previous recurrent hidden state to compute LSTM gating signals
    (input, forget, cell candidate, output), updating the cell state and hidden state
    while preserving 2D spatial dimensions.

    Gating Equations:
        i_t = sigmoid(W_xi * X_t + W_hi * H_{t-1} + b_i)
        f_t = sigmoid(W_xf * X_t + W_hf * H_{t-1} + b_f)
        g_t = tanh(W_xg * X_t + W_hg * H_{t-1} + b_g)
        o_t = sigmoid(W_xo * X_t + W_ho * H_{t-1} + b_o)
        C_t = f_t * C_{t-1} + i_t * g_t
        H_t = o_t * tanh(C_t)

    Attributes:
        in_channels (int): Number of input feature channels (default: 64).
        hidden_channels (int): Number of hidden state channels (default: 64).
        kernel_size (tuple[int, int]): Size of the 2D convolution kernel (default: (3, 3)).
        padding (tuple[int, int]): Padding added to both sides of input (default: (1, 1)).
        bias (bool): Whether convolutional layers use learnable bias (default: True).
        conv_ih (nn.Conv2d): Input-to-hidden convolution producing all 4 gates (4 * hidden_channels).
        conv_hh (nn.Conv2d): Hidden-to-hidden convolution producing all 4 gates (4 * hidden_channels).
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 64,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        bias: bool = True,
    ) -> None:
        """Initialize ConvLSTMCell.

        Args:
            in_channels (int): Number of input channels (default: 64).
            hidden_channels (int): Number of hidden state channels (default: 64).
            kernel_size (int or tuple[int, int]): Convolution kernel size (default: 3).
            bias (bool): Whether convolutions include additive bias (default: True).
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.bias = bias

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        if self.kernel_size[0] % 2 == 0 or self.kernel_size[1] % 2 == 0:
            raise ValueError(f"kernel_size components must be odd for symmetric padding, got {self.kernel_size}")

        self.padding = (self.kernel_size[0] // 2, self.kernel_size[1] // 2)

        # Input-to-hidden convolution: in_channels -> 4 * hidden_channels
        self.conv_ih = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=4 * self.hidden_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=self.bias,
        )

        # Hidden-to-hidden convolution: hidden_channels -> 4 * hidden_channels
        self.conv_hh = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=4 * self.hidden_channels,
            kernel_size=self.kernel_size,
            stride=1,
            padding=self.padding,
            bias=self.bias,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize module parameters with standard recurrent neural network schemes."""
        nn.init.kaiming_uniform_(self.conv_ih.weight, a=math.sqrt(5))
        nn.init.orthogonal_(self.conv_hh.weight)

        if self.bias:
            nn.init.zeros_(self.conv_ih.bias)
            nn.init.zeros_(self.conv_hh.bias)
            # Initialize forget gate bias to 1.0 to help preserve long-term dependencies
            with torch.no_grad():
                self.conv_ih.bias[self.hidden_channels : 2 * self.hidden_channels].fill_(1.0)

    def forward(
        self,
        x: torch.Tensor,
        state: Tuple[torch.Tensor, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Execute a single-step forward pass of the ConvLSTM cell.

        Args:
            x (torch.Tensor): Current step spatial feature map of shape [B, in_channels, H, W].
            state (tuple[torch.Tensor, torch.Tensor]): Tuple of (h_prev, c_prev),
                each of shape [B, hidden_channels, H, W].

        Returns:
            tuple[torch.Tensor, torch.Tensor]: Tuple of (h_next, c_next),
                each of shape [B, hidden_channels, H, W].
        """
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B, C, H, W] for x, got {x.dim()}D tensor with shape {list(x.shape)}")

        if not (isinstance(state, (tuple, list)) and len(state) == 2):
            raise TypeError("state must be a tuple or list of two tensors: (h_prev, c_prev)")

        h_prev, c_prev = state
        if not (isinstance(h_prev, torch.Tensor) and isinstance(c_prev, torch.Tensor)):
            raise TypeError("Both h_prev and c_prev in state must be torch.Tensor")
        if h_prev.dim() != 4 or c_prev.dim() != 4:
            raise ValueError(
                f"Expected 4D tensors for h_prev and c_prev [B, hidden_channels, H, W], "
                f"got h_prev shape {list(h_prev.shape)} and c_prev shape {list(c_prev.shape)}"
            )
        if h_prev.shape != c_prev.shape:
            raise ValueError(f"h_prev shape {list(h_prev.shape)} must match c_prev shape {list(c_prev.shape)}")

        b, c, h, w = x.shape
        if c != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} channels for x at dim 1, got {c} in shape {list(x.shape)}")
        if h_prev.size(0) != b or h_prev.size(1) != self.hidden_channels:
            raise ValueError(
                f"Expected state shape [{b}, {self.hidden_channels}, {h}, {w}], got {list(h_prev.shape)}"
            )
        if h_prev.size(2) != h or h_prev.size(3) != w:
            raise ValueError(
                f"Spatial dimensions of state ({h_prev.size(2)}, {h_prev.size(3)}) must match "
                f"input spatial dimensions ({h}, {w})"
            )

        # Compute all 4 gates via combined convolutions
        gates = self.conv_ih(x) + self.conv_hh(h_prev)
        i_gate, f_gate, g_gate, o_gate = torch.chunk(gates, 4, dim=1)

        # Activations
        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        g = torch.tanh(g_gate)
        o = torch.sigmoid(o_gate)

        # Cell state update
        c_next = f * c_prev + i * g

        # Hidden state update
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class TemporalDynamicsEncoder(nn.Module):
    """Block 3: Temporal Dynamics Encoder (ConvLSTM).

    Processes the sequential spatial feature maps generated by Block 2
    (Multi-Scale Spatial Encoder) across consecutive input days and outputs
    a refined representation of the evolving atmospheric system.

    Unlike standard flat LSTMs, the ConvLSTM retains spatial grid structure
    throughout recurrent state transitions by replacing matrix multiplications
    with 2D convolutions.

    Attributes:
        in_channels (int): Input feature channels from Block 2 (default: 64).
        hidden_channels (int): Hidden state channels in ConvLSTM (default: 64).
        kernel_size (tuple[int, int]): Size of convolution kernel in ConvLSTM (default: (3, 3)).
        bias (bool): Whether convolutions include learnable bias (default: True).
        return_sequence (bool): Whether forward returns the full sequence of hidden
            states [B, T, hidden_channels, H, W] instead of the final hidden state [B, hidden_channels, H, W].
        cell (ConvLSTMCell): The underlying modular ConvLSTM cell.
    """

    def __init__(
        self,
        in_channels: int = 64,
        hidden_channels: int = 64,
        kernel_size: Union[int, Tuple[int, int]] = 3,
        bias: bool = True,
        return_sequence: bool = False,
    ) -> None:
        """Initialize TemporalDynamicsEncoder.

        Args:
            in_channels (int): Number of input feature channels (default: 64).
            hidden_channels (int): Number of hidden state channels (default: 64).
            kernel_size (int or tuple[int, int]): Convolution kernel size (default: 3).
            bias (bool): Whether convolutions include additive bias (default: True).
            return_sequence (bool): Default return mode. If True, returns [B, T, C, H, W];
                if False (default), returns the final hidden state [B, C, H, W].
        """
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.bias = bias
        self.return_sequence = return_sequence

        if isinstance(kernel_size, int):
            self.kernel_size = (kernel_size, kernel_size)
        else:
            self.kernel_size = kernel_size

        self.cell = ConvLSTMCell(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            kernel_size=self.kernel_size,
            bias=self.bias,
        )

    def init_hidden(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize zero-valued hidden and cell states for the ConvLSTM.

        Args:
            batch_size (int): Batch size dimension B.
            height (int): Grid height dimension H.
            width (int): Grid width dimension W.
            device (torch.device, optional): Device for the initialized tensors.
            dtype (torch.dtype, optional): Data type for the initialized tensors (default: torch.float32).

        Returns:
            tuple[torch.Tensor, torch.Tensor]: (h_0, c_0), each of shape [B, hidden_channels, H, W].
        """
        if dtype is None:
            dtype = torch.float32
        h_0 = torch.zeros(batch_size, self.hidden_channels, height, width, device=device, dtype=dtype)
        c_0 = torch.zeros(batch_size, self.hidden_channels, height, width, device=device, dtype=dtype)
        return h_0, c_0

    def _validate_input(self, x: torch.Tensor, check_finite: bool = False) -> Tuple[int, int, int, int, int]:
        """Validate input tensor dimensions, types, and channel count."""
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
        if check_finite and not torch.isfinite(x).all():
            raise ValueError("Input tensor x contains NaN or Inf values")
        if x.dim() != 5:
            raise ValueError(
                f"Expected 5D tensor [B, T, C, H, W], got {x.dim()}D tensor with shape {list(x.shape)}"
            )

        b, t, c, h, w = x.shape
        if b < 1:
            raise ValueError(f"Batch size B must be >= 1, got B={b}")
        if t < 1:
            raise ValueError(f"Sequence length T must be >= 1, got T={t}")
        if c != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels at dim 2, got {c} in shape {list(x.shape)}"
            )
        if h < self.kernel_size[0] or w < self.kernel_size[1]:
            raise ValueError(
                f"Spatial dimensions ({h}, {w}) must be at least kernel size "
                f"({self.kernel_size[0]}, {self.kernel_size[1]})"
            )
        return b, t, c, h, w

    def _validate_initial_state(
        self,
        initial_state: Tuple[torch.Tensor, torch.Tensor],
        b: int,
        h: int,
        w: int,
        check_finite: bool = False,
    ) -> None:
        """Validate custom initial hidden and cell state tuple."""
        if not (isinstance(initial_state, (tuple, list)) and len(initial_state) == 2):
            raise TypeError("initial_state must be a tuple or list of two tensors: (h_0, c_0)")

        h_0, c_0 = initial_state
        if not (isinstance(h_0, torch.Tensor) and isinstance(c_0, torch.Tensor)):
            raise TypeError("Both h_0 and c_0 in initial_state must be torch.Tensor")
        if check_finite:
            if not torch.isfinite(h_0).all() or not torch.isfinite(c_0).all():
                raise ValueError("initial_state contains NaN or Inf values")

        expected_shape = (b, self.hidden_channels, h, w)
        if tuple(h_0.shape) != expected_shape:
            raise ValueError(f"Expected h_0 shape {list(expected_shape)}, got {list(h_0.shape)}")
        if tuple(c_0.shape) != expected_shape:
            raise ValueError(f"Expected c_0 shape {list(expected_shape)}, got {list(c_0.shape)}")

    def forward(
        self,
        x: torch.Tensor,
        initial_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_sequence: Optional[bool] = None,
        return_state: bool = False,
        check_finite: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass of TemporalDynamicsEncoder.

        Processes the input sequence of spatial feature maps sequentially across
        time steps T, updating hidden and cell states at each step.

        Args:
            x (torch.Tensor): Sequential spatial feature tensor of shape [B, T, 64, H, W]
                produced by Block 2.
            initial_state (tuple[torch.Tensor, torch.Tensor], optional): Initial recurrent
                state (h_0, c_0). If None, initialized to all zeros.
            return_sequence (bool, optional): If True, returns all intermediate hidden
                states [B, T, 64, H, W]. If False, returns only the final hidden state
                [B, 64, H, W]. Defaults to self.return_sequence (False).
            return_state (bool): If True, also returns the final tuple of (h_last, c_last).
                Default: False.

        Returns:
            torch.Tensor or tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
                - If return_state is False (default):
                    - If return_sequence is False (default): [B, 64, H, W] (final hidden state)
                    - If return_sequence is True: [B, T, 64, H, W] (full sequence)
                - If return_state is True:
                    - (output, (h_last, c_last))
        """
        b, t, c, h, w = self._validate_input(x, check_finite=check_finite)

        if initial_state is None:
            h_state, c_state = self.init_hidden(b, h, w, device=x.device, dtype=x.dtype)
        else:
            self._validate_initial_state(initial_state, b, h, w, check_finite=check_finite)
            h_state, c_state = initial_state[0], initial_state[1]

        ret_seq = return_sequence if return_sequence is not None else self.return_sequence

        if ret_seq:
            hidden_states = []
            for step in range(t):
                x_t = x[:, step]
                h_state, c_state = self.cell(x_t, (h_state, c_state))
                hidden_states.append(h_state)
            out = torch.stack(hidden_states, dim=1)
        else:
            for step in range(t):
                x_t = x[:, step]
                h_state, c_state = self.cell(x_t, (h_state, c_state))
            out = h_state

        if return_state:
            return out, (h_state, c_state)
        return out


# Alias ConvLSTM to TemporalDynamicsEncoder for flexible imports
ConvLSTM = TemporalDynamicsEncoder
