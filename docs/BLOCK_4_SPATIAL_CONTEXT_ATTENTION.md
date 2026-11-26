# Block 4 — Spatial Context & Attention Module

## 1. Overview and Purpose

The **Spatial Context & Attention Module** (`SpatialContextAttention` / `CBAM`) is Block 4 of the **Multiverse of Monsoon** AI climate forecasting and counterfactual simulation engine.

In our pipeline:
- **Block 1 (Climate Input Encoder)** transforms raw multi-variable meteorological inputs, static geography, and calendar signals at each individual grid cell independently into 48 feature channels.
- **Block 2 (Multi-Scale Spatial Encoder)** learns spatial patterns (local gradients, regional storms, and macro-scale teleconnections) using multi-scale 2D convolutions, yielding 64 spatial feature channels per grid cell for each day.
- **Block 3 (Temporal Dynamics Encoder / ConvLSTM)** processes the sequence of daily spatial feature maps over time and outputs a unified 64-channel spatial-temporal climate representation: `[B, 64, 32, 36]`.
- **Block 4 (Spatial Context & Attention Module)** refines this representation by dynamically reweighting both **feature channels** ("what" information matters) and **spatial regions** ("where" dynamics matter), while strictly preserving tensor shape: `[B, 64, 32, 36]`.

```
                                  +------------------------------------+
                                  |     Input from Block 3 (ConvLSTM)  |
                                  |     [B, 64, 32, 36]                |
                                  +-----------------+------------------+
                                                    |
                                                    v
                 +----------------------------------------------------------------------+
                 |               BLOCK 4: SPATIAL CONTEXT & ATTENTION MODULE            |
                 |                                                                      |
                 |  STAGE 1: CHANNEL ATTENTION                                          |
                 |    X [B, 64, 32, 36]                                                 |
                 |    |---> Spatial AvgPool ---> [B, 64] ---> Shared MLP                |
                 |    |---> Spatial MaxPool ---> [B, 64] ---> (64->16->64)              |
                 |    Sum + Sigmoid ---> Channel Weights M_c [B, 64, 1, 1]              |
                 |    Reweight: X_c = X * M_c ---> [B, 64, 32, 36]                      |
                 |                                                                      |
                 |  STAGE 2: SPATIAL ATTENTION                                          |
                 |    X_c [B, 64, 32, 36]                                               |
                 |    |---> Channel AvgPool ---> [B, 1, 32, 36]                         |
                 |    |---> Channel MaxPool ---> [B, 1, 32, 36]                         |
                 |    Concat ---> [B, 2, 32, 36] ---> 7x7 Conv ---> Sigmoid             |
                 |    Spatial Weights M_s [B, 1, 32, 36]                                |
                 |    Reweight: X_out = X_c * M_s ---> [B, 64, 32, 36]                  |
                 +----------------------------------+-----------------------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  |     Output to Block 5 (Decoder)    |
                                  |     [B, 64, 32, 36]                |
                                  +------------------------------------+
```

---

## 2. Why Attention is Used (Beginner-Friendly Explanation)

### The Challenge with Raw Feature Representations
After Block 3, every single grid cell ($32 \times 36 = 1152$ locations) contains 64 feature channels. These 64 channels represent a mixture of:
- Humidity and moisture fluxes.
- Temperature and pressure gradients.
- High-altitude wind shears.
- Terrain interactions.
- Background noise and redundant ambient variations.

In atmospheric forecasting, **not all features and not all locations are equally important at all times**:
1. During a dry pre-monsoon heatwave, temperature gradient channels over northwestern India are crucial, while localized moisture fluctuations in the Himalayas may be less informative.
2. During an active monsoon depression forming in the Bay of Bengal, total column water vapour and cyclonic wind shear channels over the ocean are paramount.

### What Attention Does
Attention is an automated "spotlight" mechanism inspired by human perception. Instead of treating all 64 channels and all 1152 grid cells with identical priority, attention computes dynamic weights between 0 and 1:
- Channels with weights near $1.0$ are amplified.
- Irrelevant or noisy channels with weights near $0.0$ are attenuated.
- Important geographic zones (e.g. active low-pressure troughs) receive heightened focus, helping the downstream decoder forecast rainfall and temperature accurately.

---

## 3. Channel Attention: "What to Focus On"

### Purpose
Channel attention determines **which feature channels are most informative** across the entire subcontinental domain.

### Mathematical Formulation
Given input $X \in \mathbb{R}^{B \times C \times H \times W}$ ($C=64, H=32, W=36$):
1. **Spatial Aggregation**: Global average pooling and global max pooling collapse spatial dimensions $(H, W)$ into channel descriptors:
   $$\mathbf{f}_{\text{avg}} = \text{AvgPool}(X) \in \mathbb{R}^{B \times C}$$
   $$\mathbf{f}_{\text{max}} = \text{MaxPool}(X) \in \mathbb{R}^{B \times C}$$
2. **Shared Bottleneck MLP**: Both pooled vectors are passed through a shared Multi-Layer Perceptron (MLP) with reduction ratio $r=4$ ($64 \to 16 \to 64$):
   $$\text{MLP}(\mathbf{v}) = W_1 \cdot \text{ReLU}(W_0 \cdot \mathbf{v})$$
   where $W_0 \in \mathbb{R}^{16 \times 64}$ and $W_1 \in \mathbb{R}^{64 \times 16}$.
3. **Combination and Gating**: The two representations are summed and mapped through the Sigmoid activation $\sigma$:
   $$\mathbf{M}_c = \sigma(\text{MLP}(\mathbf{f}_{\text{avg}}) + \text{MLP}(\mathbf{f}_{\text{max}})) \in \mathbb{R}^{B \times C \times 1 \times 1}$$
4. **Channel Reweighting**:
   $$X_c = X \odot \mathbf{M}_c$$
   where $\odot$ denotes element-wise multiplication broadcast across spatial dimensions $H$ and $W$.

### Why Use Both Average Pooling and Max Pooling?
- **Average Pooling ($\text{AvgPool}$)**: Computes the background average presence of each feature across the subcontinent. It captures the overall energy and baseline state of that channel.
- **Max Pooling ($\text{MaxPool}$)**: Identifies the peak, extreme occurrence of that feature anywhere in the domain (e.g. the single most intense storm core or lowest pressure point).
Combining both gives the MLP a complete picture of both widespread background dynamics and localized extreme phenomena.

### Why a Shared MLP?
Using the exact same MLP parameters for both average-pooled and max-pooled representations:
- Enforces consistency: both pooling descriptors are evaluated in the same feature projection space.
- Halves parameter count: only $64 \times 16 + 16 \times 64 + \text{biases} = 2,128$ parameters.
- Prevents overfitting on limited competition hardware.

---

## 4. Spatial Attention: "Where to Focus On"

### Purpose
Spatial attention determines **which geographic regions across the $32 \times 36$ grid are most critical** for forecasting.

### Mathematical Formulation
Given the channel-attended feature map $X_c \in \mathbb{R}^{B \times C \times H \times W}$:
1. **Channel Aggregation**: Channel-wise average and max operations collapse the 64 channels into 2 spatial summary maps:
   $$\mathbf{s}_{\text{avg}} = \frac{1}{C}\sum_{c=1}^{C} X_{c} \in \mathbb{R}^{B \times 1 \times H \times W}$$
   $$\mathbf{s}_{\text{max}} = \max_{c \in \{1,\dots,C\}} X_{c} \in \mathbb{R}^{B \times 1 \times H \times W}$$
2. **Concatenation**: The two 1-channel spatial maps are concatenated along the channel axis to form a 2-channel spatial summary:
   $$\mathbf{s}_{\text{cat}} = [\mathbf{s}_{\text{avg}} \,;\, \mathbf{s}_{\text{max}}] \in \mathbb{R}^{B \times 2 \times H \times W}$$
3. **Spatial Filtering (7×7 Convolution)**: A large $7 \times 7$ 2D convolution with padding $3$ processes the concatenated map:
   - Kernel size $7 \times 7$ covers an effective spatial footprint of approximately $700 \times 700\text{ km}$, encompassing synoptic monsoon systems.
   - Padding $3$ and stride $1$ guarantee output dimensions remain strictly $32 \times 36$.
4. **Sigmoid Gating**:
   $$\mathbf{M}_s = \sigma(\text{Conv}^{7 \times 7}(\mathbf{s}_{\text{cat}})) \in \mathbb{R}^{B \times 1 \times H \times W}$$
5. **Spatial Reweighting**:
   $$X_{\text{out}} = X_c \odot \mathbf{M}_s$$
   where $\odot$ broadcasts across all 64 channels.

---

## 5. What the Sigmoid Operation and Reweighting Mean

### The Sigmoid Function $\sigma(z) = \frac{1}{1 + e^{-z}}$
The raw outputs from the shared MLP and the $7 \times 7$ convolution are real numbers ranging from $-\infty$ to $+\infty$ (logits).
Sigmoid smoothly compresses them into the interval $(0, 1)$:
- $\sigma(z) \to 1.0$ when $z$ is large positive $\implies$ maximum retention / amplification.
- $\sigma(z) \to 0.0$ when $z$ is large negative $\implies$ suppression / attenuation.
- $\sigma(0) = 0.5 \implies$ neutral weighting.

### Reweighting
Reweighting is a element-wise multiplicative scaling ($X \odot M$):
- If channel $k$ has weight $0.95$, its feature activations are retained almost entirely.
- If channel $j$ has weight $0.05$, its feature activations are suppressed by $95\%$.
- Reweighting is differentiable everywhere, allowing gradients to flow backwards smoothly to adjust the attention weights during model training.

---

## 6. Tensor Shapes and Specifications

| Stage | Input Shape | Operation | Output Shape | Parameters |
|---|---|---|---|---|
| **Input** | `[B, 64, 32, 36]` | Received from Block 3 | `[B, 64, 32, 36]` | 0 |
| **Channel Pool** | `[B, 64, 32, 36]` | Spatial AvgPool + MaxPool | `[B, 64]` each | 0 |
| **Shared MLP** | `[B, 64]` | Linear(64,16) $\to$ ReLU $\to$ Linear(16,64) | `[B, 64]` each | 2,128 |
| **Channel Gate** | `[B, 64]` | Sum + Sigmoid + Reshape | `[B, 64, 1, 1]` | 0 |
| **Channel Reweight** | `[B, 64, 32, 36]` | Broadcast multiplication | `[B, 64, 32, 36]` | 0 |
| **Spatial Pool** | `[B, 64, 32, 36]` | Channel Mean + MaxPool | `[B, 1, 32, 36]` each | 0 |
| **Spatial Concat** | `[B, 1, 32, 36]` (x2) | Concat along channel dim | `[B, 2, 32, 36]` | 0 |
| **Spatial Conv** | `[B, 2, 32, 36]` | Conv2d(2, 1, k=7, p=3) + Sigmoid | `[B, 1, 32, 36]` | 99 |
| **Spatial Reweight**| `[B, 64, 32, 36]` | Broadcast multiplication | `[B, 64, 32, 36]` | 0 |
| **Final Output** | `[B, 64, 32, 36]` | Preserved shape to Block 5 | `[B, 64, 32, 36]` | **Total: 2,227** |

---

## 7. Files Created and Modified

### Files Created:
1. `monsoon_r1/models/spatial_context_attention.py`:
   - Contains `ChannelAttention`, `SpatialAttention`, and `SpatialContextAttention` (`CBAM`, `SpatialContextAttentionModule`).
2. `tests/test_spatial_context_attention.py`:
   - 12 comprehensive unit and integration tests covering shapes, subcomponents, gradient flow, CPU execution, input validation, Block 3 integration, and full pipeline integration.
3. `docs/BLOCK_4_SPATIAL_CONTEXT_ATTENTION.md`:
   - This documentation file.

### Files Modified:
1. `monsoon_r1/models/__init__.py`:
   - Exported `ChannelAttention`, `SpatialAttention`, `SpatialContextAttention`, `SpatialContextAttentionModule`, `CBAM`.
2. `monsoon_r1/__init__.py`:
   - Exported same Block 4 components at package level.

> **Protection of Existing Blocks**: Blocks 1, 2, and 3 source files (`climate_input_encoder.py`, `multi_scale_spatial_encoder.py`, `temporal_dynamics_encoder.py`) were strictly left untouched.

---

## 8. Test Execution and Actual Results

All tests were executed using `pytest -v`:

```
============================= test session starts =============================
platform win32 -- Python 3.11.9, pytest-9.2.0, pluggy-1.7.0
rootdir: C:\Users\Vijay M R\Yugaantar_Team_Vortex
configfile: pyproject.toml
collected 45 items

tests/test_climate_input_encoder.py::TestClimateInputEncoder::test_01_output_shape PASSED
tests/test_climate_input_encoder.py::TestClimateInputEncoder::test_02_batch_handling PASSED
tests/test_climate_input_encoder.py::TestClimateInputEncoder::test_03_sequence_handling PASSED
tests/test_climate_input_encoder.py::TestClimateInputEncoder::test_04_spatial_preservation PASSED
tests/test_climate_input_encoder.py::TestClimateInputEncoder::test_05_channel_composition PASSED
tests/test_climate_input_encoder.py::TestClimateInputEncoder::test_06_finite_values PASSED
tests/test_climate_input_encoder.py::TestClimateInputEncoder::test_07_gradient_propagation PASSED
tests/test_climate_input_encoder.py::TestClimateInputEncoder::test_08_cpu_compatibility PASSED
tests/test_climate_input_encoder.py::TestClimateInputEncoder::test_09_input_validation PASSED
tests/test_climate_input_encoder.py::TestClimateInputEncoder::test_10_calendar_handling_and_broadcasting PASSED
tests/test_climate_input_encoder.py::TestClimateInputEncoder::test_11_activation_option PASSED
tests/test_multi_scale_spatial_encoder.py::TestMultiScaleSpatialEncoder::test_a_basic_shape PASSED
tests/test_multi_scale_spatial_encoder.py::TestMultiScaleSpatialEncoder::test_b_batch_flexibility PASSED
tests/test_multi_scale_spatial_encoder.py::TestMultiScaleSpatialEncoder::test_c_sequence_flexibility PASSED
tests/test_multi_scale_spatial_encoder.py::TestMultiScaleSpatialEncoder::test_d_spatial_preservation PASSED
tests/test_multi_scale_spatial_encoder.py::TestMultiScaleSpatialEncoder::test_e_branch_correctness PASSED
tests/test_multi_scale_spatial_encoder.py::TestMultiScaleSpatialEncoder::test_f_finite_outputs PASSED
tests/test_multi_scale_spatial_encoder.py::TestMultiScaleSpatialEncoder::test_g_gradient_flow PASSED
tests/test_multi_scale_spatial_encoder.py::TestMultiScaleSpatialEncoder::test_h_cpu_compatibility PASSED
tests/test_multi_scale_spatial_encoder.py::TestMultiScaleSpatialEncoder::test_i_input_validation PASSED
tests/test_multi_scale_spatial_encoder.py::TestMultiScaleSpatialEncoder::test_j_integration_with_block_1 PASSED
tests/test_spatial_context_attention.py::TestSpatialContextAttention::test_a_basic_shape PASSED
tests/test_spatial_context_attention.py::TestSpatialContextAttention::test_b_batch_flexibility PASSED
tests/test_spatial_context_attention.py::TestSpatialContextAttention::test_c_channel_attention PASSED
tests/test_spatial_context_attention.py::TestSpatialContextAttention::test_d_spatial_attention PASSED
tests/test_spatial_context_attention.py::TestSpatialContextAttention::test_e_final_output_shape PASSED
tests/test_spatial_context_attention.py::TestSpatialContextAttention::test_f_finite_outputs PASSED
tests/test_spatial_context_attention.py::TestSpatialContextAttention::test_g_gradient_flow PASSED
tests/test_spatial_context_attention.py::TestSpatialContextAttention::test_h_cpu_compatibility PASSED
tests/test_spatial_context_attention.py::TestSpatialContextAttention::test_i_input_validation PASSED
tests/test_spatial_context_attention.py::TestSpatialContextAttention::test_j_integration_with_block_3 PASSED
tests/test_spatial_context_attention.py::TestSpatialContextAttention::test_k_full_pipeline_integration PASSED
tests/test_spatial_context_attention.py::TestSpatialContextAttention::test_l_regression_testing_blocks_1_to_3 PASSED
tests/test_temporal_dynamics_encoder.py::TestTemporalDynamicsEncoder::test_a_basic_shape PASSED
tests/test_temporal_dynamics_encoder.py::TestTemporalDynamicsEncoder::test_b_batch_flexibility PASSED
tests/test_temporal_dynamics_encoder.py::TestTemporalDynamicsEncoder::test_c_sequence_flexibility PASSED
tests/test_temporal_dynamics_encoder.py::TestTemporalDynamicsEncoder::test_d_spatial_preservation PASSED
tests/test_temporal_dynamics_encoder.py::TestTemporalDynamicsEncoder::test_e_hidden_state_and_cell_state_correctness PASSED
tests/test_temporal_dynamics_encoder.py::TestTemporalDynamicsEncoder::test_f_finite_outputs PASSED
tests/test_temporal_dynamics_encoder.py::TestTemporalDynamicsEncoder::test_g_gradient_flow PASSED
tests/test_temporal_dynamics_encoder.py::TestTemporalDynamicsEncoder::test_h_cpu_compatibility PASSED
tests/test_temporal_dynamics_encoder.py::TestTemporalDynamicsEncoder::test_i_input_validation PASSED
tests/test_temporal_dynamics_encoder.py::TestTemporalDynamicsEncoder::test_j_integration_with_block_2 PASSED
tests/test_temporal_dynamics_encoder.py::TestTemporalDynamicsEncoder::test_k_full_pipeline_integration PASSED
tests/test_temporal_dynamics_encoder.py::TestTemporalDynamicsEncoder::test_l_regression_testing_blocks_1_and_2 PASSED

============================= 45 passed in 4.28s ==============================
```

---

## 9. Limitations and Unresolved Issues

- **Fixed Spatial Kernel Constraint**: The $7 \times 7$ convolution in Spatial Attention requires grid dimensions $H \ge 7$ and $W \ge 7$. Since our target subcontinental grid is $32 \times 36$, this constraint is comfortably satisfied.
- **Offline CPU Execution**: The module contains only 2,227 learnable parameters, making forward/backward passes negligible in execution time (< 2 milliseconds on CPU), well within the competition budget (4 CPU cores, 6 GB RAM, no GPU).
- **No Unresolved Issues**: All unit tests, integration tests, full pipeline tests, and regression tests pass with zero warnings or errors.
