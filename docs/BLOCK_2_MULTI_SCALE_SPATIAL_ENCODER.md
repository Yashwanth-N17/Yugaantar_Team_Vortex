# Block 2 — Multi-Scale Spatial Encoder

## 1. Overview and Purpose

The **Multi-Scale Spatial Encoder** (`MultiScaleSpatialEncoder`) is Block 2 of the **Multiverse of Monsoon** AI climate forecasting engine.

While Block 1 processes climate variables, geography, and calendar features at each individual grid cell independently, **Block 2 learns spatial relationships and weather patterns across neighbouring grid cells at multiple spatial scales**.

Atmospheric systems occur simultaneously across diverse spatial scales:
- **Local convective cells and thunderstorms**: micro-scale / mesoscale ($< 100\text{ km}$, 1–2 grid cells).
- **Monsoonal depressions and fronts**: synoptic scale ($200\text{--}500\text{ km}$, 3–5 grid cells).
- **Subcontinental monsoon troughs and jet streams**: macro-scale ($> 1000\text{ km}$, across entire regions of India).

Block 2 captures all these spatial interactions in parallel through three specialized convolutional branches.

```
                                  +------------------------------------+
                                  |     Input from Block 1             |
                                  |     [B, T, 48, 32, 36]             |
                                  +-----------------+------------------+
                                                    |
                      +-----------------------------+-----------------------------+
                      |                             |                             |
                      v                             v                             v
           +--------------------+        +--------------------+        +--------------------+
           |     Branch A       |        |     Branch B       |        |     Branch C       |
           |   3x3 Conv (d=1)   |        |   5x5 Conv (d=1)   |        |   3x3 Conv (d=2)   |
           |   pad=1, stride=1  |        |   pad=2, stride=1  |        |   pad=2, stride=1  |
           |   48 -> 16 chan    |        |   48 -> 16 chan    |        |   48 -> 16 chan    |
           +----------+---------+        +----------+---------+        +----------+---------+
                      |                             |                             |
                      | [B, T, 16, 32, 36]          | [B, T, 16, 32, 36]          | [B, T, 16, 32, 36]
                      +-----------------------------+-----------------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  | Channel Concatenation (16+16+16)   |
                                  | [B, T, 48, 32, 36]                 |
                                  +-----------------+------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  | Multi-Scale Fusion (1x1 Conv)      |
                                  | 48 -> 64 channels + ReLU           |
                                  +-----------------+------------------+
                                                    |
                                                    v
                                  +------------------------------------+
                                  | Output to Block 3 (ConvLSTM)       |
                                  | [B, T, 64, 32, 36]                 |
                                  +------------------------------------+
```

---

## 2. Input and Output Tensor Shapes

- **Input Tensor**: `[B, T, 48, 32, 36]` (produced by Block 1)
  - `B`: Batch size (e.g. 1, 4, 16)
  - `T`: Sequence length (e.g. 3 daily states)
  - `48`: Feature channels (32 climate + 8 geographic + 8 calendar)
  - `32`: Grid height (latitudes 6.5°N to 37.5°N)
  - `36`: Grid width (longitudes 65.5°E to 100.5°E)
- **Output Tensor**: `[B, T, 64, 32, 36]`
  - Batch size `B`, sequence length `T`, and spatial dimensions `32 × 36` are strictly preserved.
  - Channels are expanded to `64` rich multi-scale spatial features.

---

## 3. The Three Convolutional Branches

### Branch A: Local Spatial Features (3×3 Convolution)
- **Configuration**: `kernel_size=3, padding=1, stride=1, in_channels=48, out_channels=16`
- **Receptive Field**: $3 \times 3$ grid cells ($\approx 300 \times 300\text{ km}$).
- **What it learns**: Immediate local gradients, such as localized temperature boundaries, steep local pressure gradients between adjacent cells, and micro-scale moisture pooling.

### Branch B: Broader Spatial Features (5×5 Convolution)
- **Configuration**: `kernel_size=5, padding=2, stride=1, in_channels=48, out_channels=16`
- **Receptive Field**: $5 \times 5$ grid cells ($\approx 500 \times 500\text{ km}$).
- **What it learns**: Intermediate mesoscale interactions, including cyclonic vortex rotation over the Bay of Bengal or Arabian Sea, regional mountain-blocking effects along the Western Ghats and Himalayas, and regional wind convergence zones.

### Branch C: Dilated Spatial Features (3×3 Dilated Convolution, Dilation=2)
- **Configuration**: `kernel_size=3, dilation=2, padding=2, stride=1, in_channels=48, out_channels=16`
- **Receptive Field**: Effective $5 \times 5$ grid footprint with a dilated kernel.
- **What it learns**: Wider-range atmospheric teleconnections and spatial context without increasing parameter count or computational cost compared to a dense $5 \times 5$ kernel. Captures how moisture transported from oceanic cells influences inland pressure fields multiple grid steps away.

All three branches use symmetric padding, guaranteeing that spatial dimensions ($32 \times 36$) remain unchanged.

---

## 4. Multi-Scale Feature Fusion

1. **Concatenation**: The outputs of Branch A, Branch B, and Branch C are concatenated along the channel dimension:
   $$16 \text{ (local)} + 16 \text{ (broader)} + 16 \text{ (dilated)} = 48 \text{ channels}$$
2. **1×1 Convolutional Fusion**: An `nn.Conv2d(48, 64, kernel_size=1)` combines the multi-scale representations at each grid cell, learning cross-scale relationships (e.g. how local rain is modulated by broader monsoon winds).
3. **Activation**: Followed by `nn.ReLU()` (or configurable activation) to introduce non-linearity into the fused representation.
4. **Final Output**: `[B, T, 64, 32, 36]`, ready for Block 3 (ConvLSTM Temporal Dynamics Encoder).

---

## 5. Files Created and Modified

- **Created**:
  - [`monsoon_r1/models/multi_scale_spatial_encoder.py`](file:///C:/Users/Vijay%20M%20R/Yugaantar_Team_Vortex/monsoon_r1/models/multi_scale_spatial_encoder.py): Implementation of `MultiScaleSpatialEncoder`.
  - [`tests/test_multi_scale_spatial_encoder.py`](file:///C:/Users/Vijay%20M%20R/Yugaantar_Team_Vortex/tests/test_multi_scale_spatial_encoder.py): Automated test suite covering Tests A through J.
  - [`docs/BLOCK_2_MULTI_SCALE_SPATIAL_ENCODER.md`](file:///C:/Users/Vijay%20M%20R/Yugaantar_Team_Vortex/docs/BLOCK_2_MULTI_SCALE_SPATIAL_ENCODER.md): Architectural documentation.
- **Modified**:
  - [`monsoon_r1/models/__init__.py`](file:///C:/Users/Vijay%20M%20R/Yugaantar_Team_Vortex/monsoon_r1/models/__init__.py): Added `MultiScaleSpatialEncoder` to exports.
  - [`monsoon_r1/__init__.py`](file:///C:/Users/Vijay%20M%20R/Yugaantar_Team_Vortex/monsoon_r1/__init__.py): Added `MultiScaleSpatialEncoder` to package exports.

---

## 6. How to Run Tests

Using Python's standard `unittest`:
```powershell
python -m unittest tests/test_multi_scale_spatial_encoder.py -v
```

Using `pytest`:
```powershell
pytest tests/test_multi_scale_spatial_encoder.py -v
```

To run all project tests (Blocks 1 and 2):
```powershell
pytest -v
```

---

## 7. Actual Test Results

```
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

10 passed in 1.21s
```

All 21 tests across Block 1 and Block 2 pass successfully.

---

## 8. Limitations and Notes

- **Offline / CPU efficiency**: The module is lightweight (~40k parameters), uses 2D spatial convolutions flattened across time steps, executes in $\approx 25\text{ ms}$ on standard CPU, and consumes minimal memory well within the 6 GB RAM limit.
- **Strict Boundary**: Block 2 does not perform temporal learning or attention; temporal modeling is reserved for Block 3 (ConvLSTM).
