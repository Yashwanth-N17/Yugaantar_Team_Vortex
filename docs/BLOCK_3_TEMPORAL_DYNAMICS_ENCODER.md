# Block 3 — Temporal Dynamics Encoder (ConvLSTM)

## 1. Overview and Purpose

The **Temporal Dynamics Encoder** (`TemporalDynamicsEncoder` / `ConvLSTM`) is Block 3 of the **Multiverse of Monsoon** AI climate forecasting and counterfactual simulation engine.

In our pipeline:
- **Block 1 (Climate Input Encoder)** transforms raw multi-variable meteorological inputs, static geography, and calendar signals at each individual grid cell independently into 48 feature channels.
- **Block 2 (Multi-Scale Spatial Encoder)** learns spatial patterns (local gradients, regional storms, and macro-scale teleconnections) using multi-scale 2D convolutions, yielding 64 spatial feature channels per grid cell for each day.
- **Block 3 (Temporal Dynamics Encoder)** learns **how these spatial weather patterns evolve over time across the consecutive input days**.

```
                                  +------------------------------------+
                                  |     Input from Block 2             |
                                  |     [B, T, 64, 32, 36]             |
                                  +-----------------+------------------+
                                                    |
                                                    v
                 +----------------------------------------------------------------------+
                 |             BLOCK 3: TEMPORAL DYNAMICS ENCODER (ConvLSTM)            |
                 |                                                                      |
                 |  Time Step t=0 (Day 1):                                              |
                 |    X_0 [B, 64, 32, 36] + (H_0=0, C_0=0) ---> ConvLSTMCell            |
                 |                                          ---> (H_1, C_1)             |
                 |                                                                      |
                 |  Time Step t=1 (Day 2):                                              |
                 |    X_1 [B, 64, 32, 36] + (H_1, C_1)     ---> ConvLSTMCell            |
                 |                                          ---> (H_2, C_2)             |
                 |                                                                      |
                 |  Time Step t=2 (Day 3):                                              |
                 |    X_2 [B, 64, 32, 36] + (H_2, C_2)     ---> ConvLSTMCell            |
                 |                                          ---> (H_3, C_3)             |
                 +----------------------------------+-----------------------------------+
                                                    |
                                                    v (Final Hidden State H_T)
                                  +------------------------------------+
                                  |     Output to Block 4 (Attention)  |
                                  |     [B, 64, 32, 36]                |
                                  +------------------------------------+
```

---

## 2. Why ConvLSTM? (Beginner-Friendly Explanation)

### The Problem with Standard LSTMs for Climate
A standard Long Short-Term Memory (LSTM) network is designed for 1D sequences (like sentences or financial time series). To feed a 2D weather map into a standard LSTM, you would have to flatten the grid ($32 \times 36 = 1152$ locations) into a single long 1D vector.
When you flatten a map:
- A grid cell in Mumbai loses its spatial relationship to its neighbouring cell in Pune.
- Spatial adjacency, wind vectors, pressure gradients, and regional storm tracks are destroyed.
- Matrix multiplications become huge ($1152 \times 64 = 73,728$ values per step), making it computationally heavy and prone to overfitting.

### The Solution: Convolutional LSTM (ConvLSTM)
ConvLSTM (first introduced by Shi et al., 2015) replaces the matrix multiplications in a standard LSTM with **2D convolutions**:
1. **Preserves 2D Topography**: The hidden state and cell memory remain 2D feature maps ($32 \times 36$) at every step.
2. **Local & Spatial Temporal Transitions**: Gates and state transitions are computed using spatial convolution filters ($3 \times 3$), allowing the model to track how atmospheric features (such as low-pressure monsoon depressions) shift spatially across geographic coordinates over consecutive days.
3. **Parameter Efficiency**: Instead of millions of fully connected weights, ConvLSTM shares compact convolutional filter kernels across the entire grid.

---

## 3. Hidden State vs. Cell State: How Memory Works

In ConvLSTM, two recurrent tensors travel from one day to the next:

### 1. The Cell State ($C_t$): Long-Term System Memory
- **What it is**: The persistent "conveyor belt" of the climate system.
- **Physical intuition**: Atmospheric memory has inertia. For example, ground soil moisture, sea-surface temperature anomalies, and total column water vapour do not vanish overnight; they carry forward steadily.
- **How it updates**:
  $$C_t = f_t \odot C_{t-1} + i_t \odot g_t$$
  - **Forget Gate ($f_t \in [0, 1]$)**: Decides which past spatial patterns to retain or let fade (e.g., dissipating localized storms).
  - **Input Gate ($i_t \in [0, 1]$)**: Decides how strongly to incorporate new climate patterns observed today.
  - **Candidate Memory ($g_t \in [-1, 1]$)**: The newly extracted atmospheric tendencies from today's map and yesterday's circulation.

### 2. The Hidden State ($H_t$): Working / Short-Term Representation
- **What it is**: The current active representation of the climate state at day $t$.
- **Physical intuition**: The active weather pattern that directly influences the next day's forecast.
- **How it updates**:
  $$H_t = o_t \odot \tanh(C_t)$$
  - **Output Gate ($o_t \in [0, 1]$)**: Filters the long-term memory $C_t$ to expose only the active dynamics relevant for forecasting.

---

## 4. How Temporal Information Moves Across Days

When given a 3-day window of climate features ($T = 3$):
1. **Day 1 ($t = 0$)**:
   - Initial states $(H_0, C_0)$ start at zeros (or optional prior state).
   - Day 1 feature map $X_0$ initializes the atmospheric baseline.
   - Outputs state $(H_1, C_1)$ capturing Day 1 conditions.
2. **Day 2 ($t = 1$)**:
   - Takes Day 2 spatial features $X_1$ and recurrent state $(H_1, C_1)$.
   - Gates compare Day 2 features against Day 1 memory to detect **rates of change** (e.g. pressure drops, wind acceleration, moisture influx).
   - Updates to $(H_2, C_2)$.
3. **Day 3 ($t = 2$)**:
   - Ingests Day 3 spatial features $X_2$ combined with accumulated 2-day history $(H_2, C_2)$.
   - Captures the complete trend, velocity, and trajectory of the weather system over the 3-day window.
   - Produces final hidden representation $H_3$ of shape `[B, 64, 32, 36]`.

---

## 5. Specifications & Configuration

### ConvLSTM Configuration
| Parameter | Setting | Description |
|---|---|---|
| `in_channels` | 64 | Matches Block 2 output feature channels |
| `hidden_channels` | 64 | Hidden state spatial feature channels |
| `kernel_size` | 3 × 3 | Receptive field for temporal transitions |
| `padding` | 1 | Symmetric padding ensuring exact $32 \times 36$ preservation |
| `bias` | True | Additive bias for all 4 gates |
| Recurrent layers | 1 | Single-layer ConvLSTM as specified |
| Sequence processing | Batch-first | Input shape `[B, T, C, H, W]` |

### Tensor Shapes
- **Input Tensor**: `[B, T, 64, 32, 36]`
  - `B`: Batch size (e.g. 1, 4, 16)
  - `T`: Sequence length (e.g. 3 consecutive days, flexible for any $T \ge 1$)
  - `64`: Feature channels from Block 2
  - `32`: Latitude grid height (6.5°N to 37.5°N)
  - `36`: Longitude grid width (65.5°E to 100.5°E)
- **Output Tensor**: `[B, 64, 32, 36]`
  - Batch size `B` preserved.
  - Channel dimension is 64.
  - Spatial dimensions `32 × 36` strictly preserved.
  - Represents the final hidden state $H_T$, ready for Block 4 (Spatial & Channel Attention).

---

## 6. Project Architecture Integration

Block 3 cleanly connects into the pipeline:

```python
from monsoon_r1.models import (
    ClimateInputEncoder,       # Block 1
    MultiScaleSpatialEncoder,  # Block 2
    TemporalDynamicsEncoder,   # Block 3
)

# 1. Block 1: Encode climate variables, geography, calendar
block1 = ClimateInputEncoder()
feat_b1 = block1(climate, geo, calendar)  # -> [B, T, 48, 32, 36]

# 2. Block 2: Multi-scale spatial convolution & fusion
block2 = MultiScaleSpatialEncoder()
feat_b2 = block2(feat_b1)                 # -> [B, T, 64, 32, 36]

# 3. Block 3: ConvLSTM temporal evolution
block3 = TemporalDynamicsEncoder()
feat_b3 = block3(feat_b2)                 # -> [B, 64, 32, 36]
```

---

## 7. Files Created and Modified

### Created Files
1. `monsoon_r1/models/temporal_dynamics_encoder.py`:
   - Contains `ConvLSTMCell` and `TemporalDynamicsEncoder` (with alias `ConvLSTM`).
2. `tests/test_temporal_dynamics_encoder.py`:
   - Comprehensive test suite covering Tests A through L (12 distinct test classes/methods).
3. `docs/BLOCK_3_TEMPORAL_DYNAMICS_ENCODER.md`:
   - This architectural and conceptual documentation file.

### Modified Files
1. `monsoon_r1/models/__init__.py`:
   - Exported `ConvLSTMCell`, `ConvLSTM`, and `TemporalDynamicsEncoder` alongside existing Blocks 1 and 2 classes.

### Preserved Files (Untouched)
- `monsoon_r1/models/climate_input_encoder.py` (Block 1) — Not modified.
- `monsoon_r1/models/multi_scale_spatial_encoder.py` (Block 2) — Not modified.
- `tests/test_climate_input_encoder.py` — Not modified.
- `tests/test_multi_scale_spatial_encoder.py` — Not modified.

---

## 8. Validation and Test Results

All 33 automated tests passed across the entire project repository.

### Block 3 Test Breakdown (`tests/test_temporal_dynamics_encoder.py`)
| Test ID | Test Name | Purpose / Assertion | Result |
|---|---|---|---|
| **Test A** | `test_a_basic_shape` | Input `[1, 3, 64, 32, 36]` produces output `[1, 64, 32, 36]` | **PASSED** |
| **Test B** | `test_b_batch_flexibility` | Evaluated B=1, B=4, B=8; batch dimension strictly preserved | **PASSED** |
| **Test C** | `test_c_sequence_flexibility` | Evaluated T=1, T=3, T=5, T=7; variable sequence lengths supported | **PASSED** |
| **Test D** | `test_d_spatial_preservation` | Evaluated standard grid (32×36) and alternate grid (16×20); spatial dims preserved | **PASSED** |
| **Test E** | `test_e_hidden_state_and_cell_state_correctness` | Verified zero-initialization, custom state injection, step-by-step match, return_sequence, return_state, and temporal causality | **PASSED** |
| **Test F** | `test_f_finite_outputs` | Output contains 100% finite real numbers (no NaN, no Inf) | **PASSED** |
| **Test G** | `test_g_gradient_flow` | Backward pass propagates non-zero gradients to `conv_ih`, `conv_hh`, and input | **PASSED** |
| **Test H** | `test_h_cpu_compatibility` | Pure CPU execution confirmed (no CUDA requirement) | **PASSED** |
| **Test I** | `test_i_input_validation` | Validates rejection of non-tensor inputs, wrong dimensions, wrong channel counts, tiny grids (<3×3), empty sequences, and malformed initial states | **PASSED** |
| **Test J** | `test_j_integration_with_block_2` | Actual Block 2 output `[B, T, 64, 32, 36]` feeds directly into Block 3 `[B, 64, 32, 36]` with end-to-end backprop | **PASSED** |
| **Test K** | `test_k_full_pipeline_integration` | Full pipeline Block 1 → Block 2 → Block 3 executes cleanly and propagates gradients from Block 3 back to Block 1 inputs | **PASSED** |
| **Test L** | `test_l_regression_testing_blocks_1_and_2` | Regression check confirming Block 1 and Block 2 standalone interfaces and outputs remain fully identical | **PASSED** |

### Regression Test Results for Blocks 1 and 2
- `tests/test_climate_input_encoder.py`: 11 / 11 tests **PASSED**.
- `tests/test_multi_scale_spatial_encoder.py`: 10 / 10 tests **PASSED**.
- **Overall Suite**: 33 / 33 tests **PASSED** in ~2.0s on CPU.

---

## 9. Performance & Resource Constraints

The implementation complies with competition environment constraints (4 CPU cores, 6 GB RAM, no GPU, offline execution):
- **Parameter Count**: ~295K parameters (~1.18 MB in float32).
- **Latency**: Single forward pass for `[1, 3, 64, 32, 36]` runs in ~16 ms on CPU.
- **Memory Footprint**: Transient memory during forward pass is negligible (< 5 MB), well within the 6 GB RAM limit.
- **Dependencies**: Uses standard PyTorch (`torch`, `torch.nn`) without external dependencies or pretrained weights.

---

## 10. Unresolved Issues

None. All interfaces, tensor shapes, gradient flows, and regression tests are verified and fully operational.
