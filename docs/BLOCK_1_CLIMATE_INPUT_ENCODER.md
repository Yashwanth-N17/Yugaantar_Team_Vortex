# Block 1 — Climate Input Encoder

## 1. Overview and Purpose

The **Climate Input Encoder** (`ClimateInputEncoder`) is Block 1 of the **Multiverse of Monsoon** AI climate forecasting engine. Its objective is to transform raw physical climate variables, static geographic properties, and calendar temporal context into a unified learned representation containing **48 feature channels per grid cell** for each time step.

```
+--------------------------------------------------------------------------------------------------+
|                                    BLOCK 1: CLIMATE INPUT ENCODER                                |
|                                                                                                  |
|  Dynamic Climate [B, T, 6, 32, 36]  ---> [ Conv2d 1x1 (6->32) ]              ---> [B, T, 32, 32, 36]  \
|                                                                                                       \ Channel Concat
|  Static Geography [B, 2, 32, 36]    ---> [ Conv2d 1x1 (2->8) + Broadcast T ] ---> [B, T,  8, 32, 36] ----> [B, T, 48, 32, 36]
|                                                                                                       /
|  Calendar doy_sincos [B, 2]         ---> [ Linear (2->8) + Broadcast Grid ]  ---> [B, T,  8, 32, 36]  /
+--------------------------------------------------------------------------------------------------+
```

Block 1 sits immediately prior to Block 2 (Multi-Scale Spatial Encoder). It deliberately operates **at the single-cell level without spatial mixing or spatial neighborhood extraction** (no 3×3, 5×5, or dilated spatial convolutions). Spatial neighborhood learning belongs strictly to Block 2.

---

## 2. Input Specifications

| Input Name | Variable Description | Expected Tensor Shape | Physical Units |
|---|---|---|---|
| **Dynamic Climate** | `[t2m, tp, u10, v10, msl, tcwv]` (6 variables) | `[B, T, 6, 32, 36]` | K, mm/day, m/s, m/s, Pa, kg/m² |
| **Static Geography** | `[land_sea_mask, orography_m]` (2 variables) | `[B, 2, 32, 36]` or `[B, 1, 2, 32, 36]` | binary / normalized mask, metres |
| **Calendar** | `doy_sincos` = `[sin(2π·doy/365), cos(2π·doy/365)]` | `[B, 2]` (or `[B, T, 2]`) | dimensionless unit circle coordinates |

- **B**: Batch size (e.g. 1, 4, 16)
- **T**: Sequence length (default: 3 days, configurable for any $T \ge 1$)
- **Grid**: Height $H = 32$, Width $W = 36$ (covers the India evaluation region: lat 6.5°–37.5°, lon 65.5°–100.5°)

---

## 3. Subcomponents and Architecture

### Subcomponent A: Climate Projection (`ClimateProjection`)
- **Input**: `[B, T, 6, 32, 36]`
- **Layer**: 2D 1×1 Convolution (`nn.Conv2d(6, 32, kernel_size=1, bias=True)`)
- **Transformation**: Reshapes `[B, T, 6, H, W]` to `[B*T, 6, H, W]`, performs 1×1 convolution across channels, and restores sequence dimension: `[B, T, 32, H, W]`.
- **Design rationale**: A 1×1 convolution forms learned cross-variable combinations (e.g., interaction between moisture `tcwv`, temperature `t2m`, and horizontal wind velocity `u10`, `v10`) at each individual grid cell independently. It does not perform spatial neighborhood filtering.

### Subcomponent B: Geographic Projection (`GeographicProjection`)
- **Input**: `[B, 2, 32, 36]`
- **Layer**: 2D 1×1 Convolution (`nn.Conv2d(2, 8, kernel_size=1, bias=True)`)
- **Transformation**: Maps the 2 static channels (`land_sea_mask`, `orography_m`) into 8 learned geographic channels, then unsqueezes and broadcasts across sequence length $T$ using `.expand(B, T, 8, H, W)`.
- **Design rationale**: Land-sea contrast and elevation strongly modulate monsoonal precipitation and thermal circulation. This projection provides the identical spatial boundary condition at every daily step of the sequence.

### Subcomponent C: Calendar Projection (`CalendarProjection`)
- **Input**: `[B, 2]` (or `[B, T, 2]`)
- **Layer**: Linear projection (`nn.Linear(2, 8, bias=True)`)
- **Transformation**: Maps the 2 `doy_sincos` coordinates into 8 learned annual-cycle features. The features are unsqueezed and broadcast across the spatial grid $(H=32, W=36)$ and across sequence length $T$: `[B, T, 8, 32, 36]`.
- **Calendar Handling & Conventions**:
  1. **Broadcast Last-Day Context (Default, `derive_preceding_days=False`)**: The API contract sends `doy_sincos` for the *last* input day ($t=T-1$). In this mode, the learned calendar embedding is broadcast identically across all $T$ sequence steps and spatial cells.
  2. **Trigonometric Preceding-Day Derivation (`derive_preceding_days=True`)**: When enabled, the preceding $T-1$ days' sincos coordinates are derived backwards from the last day using exact angle subtraction formulas:
     $$\sin(\theta + \Delta) = \sin(\theta)\cos(\Delta) + \cos(\theta)\sin(\Delta)$$
     $$\cos(\theta + \Delta) = \cos(\theta)\cos(\Delta) - \sin(\theta)\sin(\Delta)$$
     where $\Delta = (t - (T-1)) \cdot \frac{2\pi}{365}$. This formulation is differentiable, naturally handles circular year wraparound (e.g. Day 1 stepping back to Day 365 and 364), and avoids leap-year heuristics.
  3. **Direct Per-Day Sequence Input**: If a 3D tensor `[B, T, 2]` is provided, each day's calendar coordinates are projected per-day and broadcast spatially.

### Subcomponent D: Feature Concatenation
The three representations are concatenated along the channel dimension (`dim=2`):
- Climate features: 32 channels
- Geographic features: 8 channels
- Calendar features: 8 channels
- **Total output channels**: $32 + 8 + 8 = 48$ channels per grid cell.
- **Final Output Tensor**: `[B, T, 48, 32, 36]`

---

## 4. Activation Function Justification

By default, `ClimateInputEncoder` applies linear projections (`activation=None`, implemented via `nn.Identity()`). An optional non-linear activation (such as `nn.SiLU()` or `nn.GELU()`) can be passed via the `activation` constructor argument.

**Design Justification**:
1. **Preserving Directional Sign**: Variables like wind components ($u_{10}$ and $v_{10}$) have zero-mean symmetric distributions representing eastward/westward and northward/southward flow. Using non-linearities like standard ReLU at the initial embedding layer would truncate half the physical velocity domain to zero prior to spatial modeling.
2. **Modular Non-Linearity in Block 2**: Block 2 (Multi-Scale Spatial Encoder) applies non-linear activations following multi-scale spatial convolution and channel fusion. Keeping Block 1's projection linear preserves the full dynamic range of the physical inputs into the latent space.

---

## 5. Running Tests

The automated test suite covers all 10 required test criteria plus optional activation verification.

To run tests with Python's built-in `unittest`:
```bash
python -m unittest tests/test_climate_input_encoder.py -v
```

To run tests with `pytest`:
```bash
pytest tests/test_climate_input_encoder.py -v
```
