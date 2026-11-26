# Block 5 — Forecast Decoder

## 1. Overview and Purpose

The **Forecast Decoder** (`ForecastDecoder`) is Block 5 of the **Multiverse of Monsoon** AI climate forecasting and counterfactual simulation engine.

In our modular 6-block architecture:
- **Block 1 (Climate Input Encoder)**: Projects raw physical climate variables, static geography, and calendar features into 48 learned feature channels per grid cell: `[B, T, 48, 32, 36]`.
- **Block 2 (Multi-Scale Spatial Encoder)**: Extracts spatial patterns across local, regional, and macro scales using multi-scale 2D convolutions, yielding 64 spatial feature channels: `[B, T, 64, 32, 36]`.
- **Block 3 (Temporal Dynamics Encoder / ConvLSTM)**: Integrates the 3-day sequence into a unified temporal hidden state: `[B, 64, 32, 36]`.
- **Block 4 (Spatial Context & Attention Module)**: Reweights feature channels and spatial regions using channel and spatial attention gates: `[B, 64, 32, 36]`.
- **Block 5 (Forecast Decoder)**: Transforms the 64-channel latent representation into **six spatial prediction maps** corresponding to the six target climate variables: `[B, 6, 32, 36]`.
- **Block 6 (Residual State Transition)** *(Upcoming)*: Combines the predicted changes (deltas) with the current state to produce the next day's full physical climate state.

```
                                  +------------------------------------+
                                  |     Input from Block 4 (Attention) |
                                  |     [B, 64, 32, 36]                |
                                  +-----------------+------------------+
                                                    |
                                                    v
                 +----------------------------------------------------------------------+
                 |                     BLOCK 5: FORECAST DECODER                        |
                 |                                                                      |
                 |  LAYER 1: FEATURE REFINEMENT                                         |
                 |    X [B, 64, 32, 36]                                                 |
                 |    ---> Conv2d 3x3 (in=64, out=64, stride=1, pad=1)                  |
                 |    ---> ReLU Activation                                              |
                 |    ---> Refined Features: [B, 64, 32, 36]                            |
                 |                                                                      |
                 |  LAYER 2: FEATURE REDUCTION                                          |
                 |    ---> Conv2d 3x3 (in=64, out=32, stride=1, pad=1)                  |
                 |    ---> ReLU Activation                                              |
                 |    ---> Compressed Features: [B, 32, 32, 36]                         |
                 |                                                                      |
                 |  LAYER 3: CLIMATE OUTPUT PROJECTION                                  |
                 |    ---> Conv2d 1x1 (in=32, out=6, stride=1, pad=0)                   |
                 |    ---> Linear Projection (no non-linearity)                         |
                 |    ---> Climate Delta Prediction Maps: [B, 6, 32, 36]                |
                 +----------------------------------+-----------------------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  |   Predicted Climate Deltas [B, 6]  |
                                  |   (t2m, tp, u10, v10, msl, tcwv)   |
                                  |   Output to Block 6                |
                                  |   [B, 6, 32, 36]                   |
                                  +------------------------------------+
```

---

## 2. Why the Decoder is Needed (Beginner-Friendly Explanation)

### The Latent Representation vs. Physical Climate Reality
Blocks 1 through 4 compress, encode, and refine meteorological information into an abstract 64-dimensional feature space. In this latent space:
- Feature channel 12 might represent a complex non-linear blend of vertical moisture flux and mountain upslope wind.
- Feature channel 45 might capture cyclonic vorticity over the Bay of Bengal.
- These 64 channels are **abstract mathematical features**, not directly interpretable temperatures, wind speeds, or rainfall amounts.

### The Decoder's Role
The decoder acts as an "interpreter" that translates the high-dimensional latent representation back into physical meteorological variables.
Specifically:
1. **Gradual Dimensionality Reduction**: Jumping directly from 64 abstract features to 6 output channels in a single step would force extreme information loss and bottlenecking. The decoder uses a progressive 2-step reduction ($64 \to 64 \to 32 \to 6$) to distill features smoothly.
2. **Local Feature Refinement**: By applying a 3×3 convolution before channel compression, the decoder allows neighbouring grid cells to share context one final time, smoothing out edge artifacts and boundary discontinuities.
3. **Pointwise Projection to Climate Channels**: A final 1×1 convolution projects each 32-dimensional feature vector into exactly 6 output values per grid cell.

---

## 3. Layer Specifications

The decoder follows a lightweight, computationally efficient 3-layer convolutional architecture:

| Layer | Type | In Channels | Out Channels | Kernel Size | Stride | Padding | Activation | Purpose |
|---|---|---|---|---|---|---|---|---|
| **Layer 1** | Conv2d | 64 | 64 | 3 × 3 | 1 | 1 | ReLU | **Feature refinement**: Refines attended features before channel compression. |
| **Layer 2** | Conv2d | 64 | 32 | 3 × 3 | 1 | 1 | ReLU | **Feature reduction**: Halves channel dimensionality while preserving spatial context. |
| **Layer 3** | Conv2d | 32 | 6 | 1 × 1 | 1 | 0 | None (Linear) | **Climate projection**: Pointwise projection to 6 target variable prediction maps. |

### Why Layer 3 Has No Activation Function
Layers 1 and 2 employ standard non-linear activations (`ReLU`) to model non-linear interactions among climate features.
In contrast, **Layer 3 uses a linear projection without activation**:
- Predicted atmospheric deltas can be positive (e.g. rising temperature $\Delta T > 0$, wind accelerating eastward $\Delta u > 0$) or negative (e.g. dropping pressure $\Delta p < 0$, falling temperature $\Delta T < 0$).
- Using `ReLU` at Layer 3 would truncate all negative changes to 0, which would prevent the model from predicting pressure drops, cooling, or wind reversals.
- A linear projection allows predicted deltas to span the full real number spectrum $\mathbb{R}$.

---

## 4. Climate Output Channels and Prescribed Ordering

The output tensor has shape `[B, 6, 32, 36]`. The 6 channels correspond strictly to the official competition contract (`monsoon_r1.contract.CHANNELS`):

| Channel Index | Variable Code | Variable Name | Physical Units | Typical Dynamic Range |
|:---:|:---:|---|---|:---:|
| `0` | `t2m` | 2-metre Temperature | Kelvin (K) | 250 – 325 K |
| `1` | `tp` | Total Daily Precipitation | mm / day | 0 – 300+ mm/day |
| `2` | `u10` | 10-metre Eastward Wind Component | m / s | -30 – +30 m/s |
| `3` | `v10` | 10-metre Northward Wind Component | m / s | -30 – +30 m/s |
| `4` | `msl` | Mean Sea-Level Pressure | Pascal (Pa) | 98,000 – 104,000 Pa |
| `5` | `tcwv` | Total Column Water Vapour | kg / m² | 5 – 80 kg/m² |

### How Channel Order is Established and Preserved
- The channel order `("t2m", "tp", "u10", "v10", "msl", "tcwv")` is hardcoded as class attributes `ForecastDecoder.CHANNELS` and `ForecastDecoder.TARGET_VARIABLES` imported directly from `monsoon_r1.contract.CHANNELS`.
- When training the network, the loss function (e.g. MSE or MAE) computes errors between `out[:, c, :, :]` and target ground-truth deltas `target[:, c, :, :]` for each channel index $c \in \{0, 1, 2, 3, 4, 5\}$.
- Convolution channel 0 is explicitly optimized to predict `t2m`, channel 1 for `tp`, channel 2 for `u10`, and so forth.
- This 1-to-1 index alignment ensures seamless compatibility with Block 6 and evaluation scripts.

---

## 5. Input and Output Specifications

### Input Specification
- **Source**: Output of Block 4 (`SpatialContextAttention`).
- **Shape**: `[B, 64, 32, 36]`
  - `B`: Batch size ($\ge 1$).
  - `64`: Input feature channels.
  - `32`: Grid height ($NLAT = 32$, lat 6.5° to 37.5° N).
  - `36`: Grid width ($NLON = 36$, lon 65.5° to 100.5° E).
- **Validation**: Strict type checks (`torch.Tensor`), dimensionality checks (must be 4D), channel checks (must be 64), and spatial size checks ($H, W \ge 3$).

### Output Specification
- **Shape**: `[B, 6, 32, 36]`
  - `B`: Preserved batch size.
  - `6`: Exactly six climate variable prediction maps.
  - `32`: Height preserved identically.
  - `36`: Width preserved identically.
- **Data type**: `torch.float32`.
- **Values**: Finite real numbers (strictly verified: no `NaN`, no `Inf`).

---

## 6. Prediction Representation & Normalization Semantics

### Deltas vs. Direct State Prediction
In accordance with modern atmospheric deep learning benchmarks (such as GraphCast, ClimaX, and Pangu-Weather):
- Block 5 is designed to predict **atmospheric state changes (deltas)**:
  $$\Delta \hat{y} = \hat{y}_{t+1} - y_t$$
  rather than directly predicting the next full state $\hat{y}_{t+1}$.
- **Physical Rationale**: Atmospheric states from one day to the next have high autocorrelation (e.g. pressure over Delhi changes by ~200 Pa out of 101,325 Pa). Predicting full state values forces the network to spend capacity memorizing baseline climatology. Predicting deltas focuses gradient updates directly on evolving weather shocks, frontal movements, and diurnal changes.

### Normalization and Representation Assumptions
- **Inside the Neural Network (Blocks 1–5)**: All representations exist in normalized latent space.
- **Decoder Output**: The output tensor $\Delta \hat{y} \in \mathbb{R}^{B \times 6 \times 32 \times 36}$ represents **normalized predicted deltas**.
- **Interface to Block 6**:
  - Block 5 does **NOT** add the predictions to the current physical state.
  - Block 5 does **NOT** enforce physical clamps (e.g. clipping precipitation $\ge 0$).
  - In Block 6, the normalized deltas will be converted to physical units using the channel standard deviations $\sigma$ from `stats.json`:
    $$\Delta \hat{y}_{\text{phys}} = \Delta \hat{y} \odot \sigma$$
    and then added to the physical current state:
    $$\hat{y}_{t+1, \text{phys}} = y_{t, \text{phys}} + \Delta \hat{y}_{\text{phys}}$$
    followed by physical boundary validations (e.g. non-negative precipitation).
  - This design strictly maintains separation of concerns and prevents unnormalized physical values from being mixed with normalized latent deltas.

---

## 7. Performance & Computational Efficiency

The evaluation environment enforces strict resource constraints:
- **CPU only** (4 CPU cores, 6 GB RAM).
- **No GPU / CUDA**.
- **Inference time limit**: Under 5.0 seconds for a 10-day rollout.

### Parameter and FLOP Analysis for Block 5
- **Layer 1** ($64 \to 64$, $3 \times 3$): $64 \times 64 \times 3 \times 3 + 64 = 36,928$ parameters.
- **Layer 2** ($64 \to 32$, $3 \times 3$): $64 \times 32 \times 3 \times 3 + 32 = 18,464$ parameters.
- **Layer 3** ($32 \to 6$, $1 \times 1$): $32 \times 6 \times 1 \times 1 + 6 = 198$ parameters.
- **Total Parameters**: **55,590 parameters** (~217 KB in float32).
- **Execution Time**: Under 2 milliseconds on a single modern CPU core for batch size 1.
- Fully compatible with offline CPU execution.

---

## 8. Files Created and Modified

### Files Created
- `monsoon_r1/models/forecast_decoder.py`: Modular implementation of Block 5 (`ForecastDecoder`, with aliases `ClimateForecastDecoder` and `Block5ForecastDecoder`).
- `tests/test_forecast_decoder.py`: Comprehensive test suite implementing Tests A through L.
- `docs/BLOCK_5_FORECAST_DECODER.md`: Complete documentation (this file).

### Files Modified
- `monsoon_r1/models/__init__.py`: Exported `ForecastDecoder`, `ClimateForecastDecoder`, and `Block5ForecastDecoder`. Blocks 1 through 4 exports and implementations were left completely untouched and preserved.

---

## 9. Test Suite and Verification Results

The test suite was executed via `pytest -v`. All 57 tests passed (including all 45 existing tests for Blocks 1–4 and 12 new comprehensive tests for Block 5):

| Test ID | Test Name | Purpose / Verification | Result |
|:---:|---|---|:---:|
| **Test A** | `test_a_basic_shape` | Input `[1, 64, 32, 36]` produces output `[1, 6, 32, 36]`. Aliases verified. | **PASSED** |
| **Test B** | `test_b_batch_flexibility` | Batch sizes $B \in \{1, 4, 8\}$ verified with batch preservation. | **PASSED** |
| **Test C** | `test_c_spatial_preservation` | Grid preservation ($32 \times 36$) and arbitrary grids ($16 \times 16$, $20 \times 24$, $40 \times 48$). | **PASSED** |
| **Test D** | `test_d_layer_correctness` | Layer 1: $64 \to 64$, Layer 2: $64 \to 32$, Layer 3: $32 \to 6$. Intermediates verified. | **PASSED** |
| **Test E** | `test_e_channel_order` | Output channels match `(t2m, tp, u10, v10, msl, tcwv)` from contract. | **PASSED** |
| **Test F** | `test_f_finite_outputs` | Output contains no `NaN` or `Inf` across zero, large, negative, and random inputs. | **PASSED** |
| **Test G** | `test_g_gradient_flow` | Backward pass propagates non-zero finite gradients through all layers and input. | **PASSED** |
| **Test H** | `test_h_cpu_compatibility` | Operates natively on CPU without requiring CUDA. | **PASSED** |
| **Test I** | `test_i_input_validation` | Validates error handling for non-tensors, 3D/5D shapes, wrong channels, $B=0$, $H,W < 3$. | **PASSED** |
| **Test J** | `test_j_integration_with_block_4` | End-to-end integration: Block 4 (`SpatialContextAttention`) $\to$ Block 5. Gradients verified. | **PASSED** |
| **Test K** | `test_k_full_pipeline_integration` | Full pipeline: Block 1 $\to$ Block 2 $\to$ Block 3 $\to$ Block 4 $\to$ Block 5. Backward pass verified. | **PASSED** |
| **Test L** | `test_l_regression_testing_blocks_1_to_4`| Regression check confirming Blocks 1, 2, 3, and 4 continue to operate identically. | **PASSED** |

### Regression Summary
```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\Vijay M R\Yugaantar_Team_Vortex
collected 57 items

tests/test_climate_input_encoder.py ...........                          [ 19%]
tests/test_forecast_decoder.py ............                             [ 40%]
tests/test_multi_scale_spatial_encoder.py ..........                     [ 57%]
tests/test_spatial_context_attention.py ............                     [ 78%]
tests/test_temporal_dynamics_encoder.py ............                     [100%]

============================= 57 passed in 4.13s ==============================
```

---

## 10. Limitations and Unresolved Issues

- **Block 6 Dependency**: Block 5 produces predicted deltas $\Delta \hat{y}$; actual physical state advancement and residual addition are deferred to Block 6 (Residual State Transition).
- **Climatology vs. Delta Training**: While Block 5 architecture supports unbounded real-valued delta predictions, the final numerical accuracy depends on the upcoming training loop and dataset preparation.
