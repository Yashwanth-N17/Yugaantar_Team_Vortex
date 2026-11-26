# Block 6 — Residual State Transition

## 1. Overview and Purpose

The **Residual State Transition** (`ResidualStateTransition`) is Block 6 and the final architectural component of the **Multiverse of Monsoon** AI climate forecasting and counterfactual simulation engine.

In our modular 6-block architecture:
- **Block 1 (Climate Input Encoder)**: Projects physical dynamic climate variables, static geography, and calendar features into 48 learned feature channels per grid cell: `[B, T, 48, 32, 36]`.
- **Block 2 (Multi-Scale Spatial Encoder)**: Extracts spatial patterns across local, regional, and macro scales using multi-scale 2D convolutions, yielding 64 spatial feature channels: `[B, T, 64, 32, 36]`.
- **Block 3 (Temporal Dynamics Encoder / ConvLSTM)**: Integrates the 3-day sequence into a unified temporal hidden state: `[B, 64, 32, 36]`.
- **Block 4 (Spatial Context & Attention Module)**: Reweights feature channels and spatial regions using channel and spatial attention gates: `[B, 64, 32, 36]`.
- **Block 5 (Forecast Decoder)**: Transforms the 64-channel latent representation into **six atmospheric state change prediction maps (deltas)**: `[B, 6, 32, 36]`.
- **Block 6 (Residual State Transition)**: Combines the current atmospheric climate state with the predicted deltas from Block 5, enforces physical non-negativity for precipitation, maintains unit and normalization consistency, and returns the next complete climate state: `[B, 6, 32, 36]`.

```
                                  +------------------------------------+
                                  |     Current State [B, 6, 32, 36]   |
                                  |     (latest day in input window)   |
                                  +-----------------+------------------+
                                                    |
                                                    |
+------------------------------------+              |
|   Predicted Deltas [B, 6, 32, 36]  |              |
|   from Block 5 (ForecastDecoder)   |              |
+-----------------+------------------+              |
                  |                                 |
                  v                                 |
  [Optional: scale by std dev σ]                    |
                  |                                 |
                  v                                 v
+----------------------------------------------------------------------+
|               BLOCK 6: RESIDUAL STATE TRANSITION                     |
|                                                                      |
|  1. RESIDUAL ADDITION                                                |
|     next_state = current_state + effective_delta                     |
|                                                                      |
|  2. PHYSICAL PRECIPITATION CONSTRAINT                                |
|     next_state[:, 1, :, :] = clamp(next_state[:, 1, :, :], min=0.0)  |
|     (All other channels 0, 2, 3, 4, 5 preserved identically)         |
|                                                                      |
|  3. AUTOGRAD PRESERVATION                                            |
|     Out-of-place slicing & concatenation preserves backward graphs   |
+----------------------------------+-----------------------------------+
                                   |
                                   v
                 +------------------------------------+
                 |     Next State [B, 6, 32, 36]      |
                 |     (t2m, tp, u10, v10, msl, tcwv) |
                 |     Fed back to Rolling Window     |
                 +------------------------------------+
```

---

## 2. Why Residual State Transition is Needed

### The Physics of Weather Forecasting
Atmospheric variables exhibit high temporal autocorrelation from one day to the next. For instance:
- Temperature at a location changes by a few Kelvin ($\pm 1$ to $5$ K) against a baseline of ~300 K.
- Surface atmospheric pressure changes by hundreds of Pascals ($\pm 100$ to $500$ Pa) against a baseline of ~101,325 Pa.

Directly predicting the full state $\hat{y}_{t+1}$ with a neural network forces the model parameters to expend almost all their capacity memorizing static climatological baselines and spatial topography. By having Block 5 predict **residual changes (deltas)** $\Delta \hat{y}$ and applying Block 6 to perform the **residual state transition**:
$$\hat{y}_{t+1} = y_t + \Delta \hat{y}$$
the neural network focuses entirely on atmospheric dynamics, advection, and local convective shocks.

### Role of Block 6
Block 6 serves as the bridge between model predictions and physical reality:
1. **Mathematical Residual Integration**: Adds the predicted deltas to the current state.
2. **Channel-Scale Consistency**: Optionally scales normalized deltas by channel standard deviations ($\sigma$) when the neural network operates in normalized space.
3. **Physical Boundary Enforcement**: Prevents physically invalid negative rainfall values ($tp \ge 0$) without disturbing the gradients or modifying other meteorological channels.
4. **Autoregressive Feedback**: Produces the cleanly formatted next state tensor suitable for sliding the rolling 3-day input window forward.

---

## 3. Mathematical State Transition Formulation

### Basic Formulation (Physical Space)
When the model operates in physical units (the default convention of the competition contract):
$$\hat{y}_{t+1, c, i, j} = y_{t, c, i, j} + \Delta \hat{y}_{c, i, j} \quad \text{for } c \in \{0, 2, 3, 4, 5\}$$
$$\hat{y}_{t+1, 1, i, j} = \max\left(y_{t, 1, i, j} + \Delta \hat{y}_{1, i, j}, \, 0\right) \quad \text{for precipitation } (c=1)$$

### Scaled Formulation (Normalized Deltas to Physical States)
When Block 5 outputs normalized deltas $\Delta \hat{z}$ and the current state is in physical units:
$$\Delta \hat{y}_{\text{phys}} = \Delta \hat{z} \odot \boldsymbol{\sigma}$$
$$\hat{y}_{t+1} = y_{t, \text{phys}} + \Delta \hat{y}_{\text{phys}}$$
where $\boldsymbol{\sigma} = [\sigma_{t2m}, \sigma_{tp}, \sigma_{u10}, \sigma_{v10}, \sigma_{msl}, \sigma_{tcwv}]$ are the channel standard deviations (e.g., from `monsoon_r1.scoring.STD_DEFAULT` or `stats.json`).

### Normalized Space Formulation
When the entire model rollout operates in normalized space:
$$\hat{z}_{t+1} = z_t + \Delta \hat{z}$$
with normalized precipitation lower bound:
$$\hat{z}_{t+1, 1, i, j} = \max\left(z_{t, 1, i, j} + \Delta \hat{z}_{1, i, j}, \, \frac{-\mu_{tp}}{\sigma_{tp}}\right)$$
which guarantees that upon denormalization ($y = z \cdot \sigma + \mu$), the physical precipitation is strictly $\ge 0$.

---

## 4. Climate Channels and Prescribed Ordering

The tensor channels strictly follow `monsoon_r1.contract.CHANNELS`:

| Channel Index | Variable Code | Variable Name | Physical Units | Constraint Enforced |
|:---:|:---:|---|---|:---:|
| `0` | `t2m` | 2-metre Temperature | Kelvin (K) | None (unbounded delta) |
| `1` | `tp` | Total Daily Precipitation | mm / day | **Non-negative ($tp \ge 0$)** |
| `2` | `u10` | 10-metre Eastward Wind Component | m / s | None (unbounded delta) |
| `3` | `v10` | 10-metre Northward Wind Component | m / s | None (unbounded delta) |
| `4` | `msl` | Mean Sea-Level Pressure | Pascal (Pa) | None (unbounded delta) |
| `5` | `tcwv` | Total Column Water Vapour | kg / m² | None (unbounded delta) |

### Precipitation Handling Rules
- **Non-negativity**: Rainfall cannot be negative. If $y_{t, 1} + \Delta y_1 < 0$, it is clamped to 0.0.
- **Zero Rain**: Zero precipitation with zero delta produces exact zero ($0 + 0 = 0$).
- **Positive Rain**: Positive rain predictions are completely unaltered ($15 + 10 = 25$ mm/day).
- **Isolation**: Clamping precipitation does not alter any other channel ($t2m, u10, v10, msl, tcwv$ remain exact additions).
- **No Arbitrary Thresholds**: No ad-hoc thresholds or arbitrary maximum clipping is applied.

---

## 5. Input and Output Specifications

### Input Specification
1. **`current_state`**:
   - Shape: `[B, 6, 32, 36]` (or `[B, 6, H, W]`).
   - Represents the latest known climate state (e.g. Day 3 in a 3-day sequence `climate[:, -1]`).
   - Type: `torch.Tensor` (float32).
2. **`predicted_delta`**:
   - Shape: `[B, 6, 32, 36]` (or `[B, 6, H, W]`).
   - Output from Block 5 (`ForecastDecoder`).
   - Type: `torch.Tensor` (float32).
3. **`channel_scales`** *(Optional)*:
   - Shape: `[6]` or `[1, 6, 1, 1]`.
   - Per-channel scaling standard deviations for converting normalized deltas to physical deltas.

### Output Specification
- Shape: `[B, 6, 32, 36]`.
- Batch size, height, and width preserved identically.
- Values: strictly finite (no `NaN`, no `Inf`).
- Channel ordering strictly preserved.

---

## 6. Autoregressive Rollout Workflow

Block 6 directly powers the 10-day autoregressive forecast rollout:

```python
# Rolling 3-day window: [B, 3, 6, 32, 36]
window = init_states.clone()

rollout = []
for step in range(10):
    # 1. Blocks 1-5 predict atmospheric change for next day
    feat1 = block1(window, geography, calendar)
    feat2 = block2(feat1)
    feat3 = block3(feat2)
    feat4 = block4(feat3)
    delta = block5(feat4)  # [B, 6, 32, 36]

    # 2. Block 6 advances current state (latest in window) to next state
    curr_state = window[:, -1]
    next_state = block6(curr_state, delta)  # [B, 6, 32, 36]

    rollout.append(next_state)

    # 3. Slide window forward (drop Day t-2, append Day t+1)
    window = torch.cat([window[:, 1:], next_state.unsqueeze(1)], dim=1)

# Full rollout: 10 daily prediction maps
forecast_rollout = torch.stack(rollout, dim=0)  # [10, B, 6, 32, 36]
```

---

## 7. Perturbation Compatibility

The competition contract allows physical climate shocks (perturbations of shape `[6, 32, 36]`) applied to the latest initial state `init_states[-1]`.

`ResidualStateTransition` provides the built-in helper:
```python
perturbed_state = block6.apply_perturbation(initial_state, perturbation)
```
This handles both unbatched `[6, 32, 36]` and batched `[B, 6, 32, 36]` tensors, applies the physical shock, and enforces physical precipitation validity.

---

## 8. Files Created and Modified

### Files Created
- `monsoon_r1/models/residual_state_transition.py`: Implementation of `ResidualStateTransition`, `ClimateResidualStateTransition`, and `Block6ResidualStateTransition`.
- `tests/test_residual_state_transition.py`: Test suite implementing Tests A through N.
- `docs/BLOCK_6_RESIDUAL_STATE_TRANSITION.md`: Complete documentation (this file).

### Files Modified
- `monsoon_r1/models/__init__.py`: Exported Block 6 classes (`ResidualStateTransition`, `ClimateResidualStateTransition`, `Block6ResidualStateTransition`). Blocks 1 through 5 implementations were left completely untouched.
- `Multiverse_of_Monsoon_Architecture_README.md`: Updated Section 7 with implementation status and verification.

---

## 9. Test Suite and Verification Results

The test suite was executed via `pytest -v`. All 71 tests passed (57 existing tests for Blocks 1–5 + 14 comprehensive tests for Block 6):

| Test ID | Test Name | Purpose / Verification | Result |
|:---:|---|---|:---:|
| **Test A** | `test_a_tensor_shapes` | Inputs `[1, 6, 32, 36]` + `[1, 6, 32, 36]` $\to$ `[1, 6, 32, 36]`. Aliases verified. | **PASSED** |
| **Test B** | `test_b_batch_compatibility` | Batch sizes $B \in \{1, 2, 4, 8\}$ verified with batch preservation. | **PASSED** |
| **Test C** | `test_c_residual_correctness` | Zero delta produces identical state ($x + 0 = x$). Known deltas update each channel correctly. | **PASSED** |
| **Test D** | `test_d_precipitation_handling` | Non-negative rain clamp ($tp \ge 0$), zero rain preservation, positive rain unaltered, other channels unaffected. | **PASSED** |
| **Test E** | `test_e_numerical_stability` | No NaN, no Inf across realistic meteorological ranges, zero inputs, and large inputs. | **PASSED** |
| **Test F** | `test_f_gradient_flow` | Gradients propagate to both `current_state` and `predicted_delta`. Subgradients at boundary verified. | **PASSED** |
| **Test G** | `test_g_cpu_compatibility` | Operates strictly on CPU without requiring CUDA. | **PASSED** |
| **Test H** | `test_h_input_validation` | Type checks, 3D/5D checks, shape mismatch, channel count mismatch, $B=0$, $H,W < 1$, device/dtype mismatch. | **PASSED** |
| **Test I** | `test_i_integration_with_block_5` | Block 4 $\to$ Block 5 $\to$ Block 6 integration. Gradients reach Block 5 and Block 4. | **PASSED** |
| **Test J** | `test_j_full_pipeline_integration` | End-to-end integration across all 6 blocks (Block 1 $\to$ Block 6). Gradients reach raw inputs. | **PASSED** |
| **Test K** | `test_k_autoregressive_rollout_simulation` | Multi-step rolling window autoregressive rollout simulation verified. | **PASSED** |
| **Test L** | `test_l_perturbation_compatibility` | Physical perturbations applied to batched and unbatched states. Perturbation divergence verified. | **PASSED** |
| **Test M** | `test_m_scaling_and_normalization_flexibility` | Factory methods `from_default_scales`, `from_stats_dict`, and normalized space precipitation lower bounds. | **PASSED** |
| **Test N** | `test_n_regression_blocks_1_to_5` | Regression check confirming Blocks 1, 2, 3, 4, and 5 continue to pass all checks. | **PASSED** |

### Overall Regression Summary
```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\Vijay M R\Yugaantar_Team_Vortex
configfile: pyproject.toml
collected 71 items

tests\test_climate_input_encoder.py ...........                          [ 15%]
tests\test_forecast_decoder.py ............                              [ 32%]
tests\test_multi_scale_spatial_encoder.py ..........                     [ 46%]
tests\test_residual_state_transition.py ..............                   [ 66%]
tests\test_spatial_context_attention.py ............                     [ 83%]
tests\test_temporal_dynamics_encoder.py ............                     [100%]

============================= 71 passed in 4.63s ==============================
```
