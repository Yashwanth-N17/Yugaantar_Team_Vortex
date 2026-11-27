# 🌌 Multiverse of Monsoon: Earth-System Digital Twin & Climate Simulation Engine

[![Tests](https://img.shields.io/badge/tests-110%20passed-brightgreen.svg)]()
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.15%20(CPU)-orange.svg)]()

> **Track 2: "Multiverse of Monsoon"**  
> *AI-Powered Digital Twin of India's Climate (Inspired by ISRO's Bharatiya Antariksh Hackathon PS-5)*  
> Built by **Team Vortex**

---

## 1. Executive Summary

Weather apps forecast: *"Will it rain tomorrow?"*  
The **Multiverse of Monsoon Climate Digital Twin** answers:  
***"How does India's atmospheric state evolve over 10 days, and what happens in an alternate universe if we inject localized physical shocks into the climate system?"***

### Core Capabilities:
1. **Timeline Prime (10-Day Deterministic Simulation):**  
   From a 3-day history of India's atmospheric variables ($T_{2m}$, $TP$, $U_{10}$, $V_{10}$, $MSL$, $TCWV$), the engine rolls weather forward autoregressively day by day **without exploding, decaying, or blurring into a flat climatological average**.
2. **Multiverse Branching (Counterfactual Shock Propagation):**  
   Accepts localized physical perturbations (e.g., $+3^\circ\text{C}$ marine heatwave in the Bay of Bengal, $-5\text{ m/s}$ weaker Arabian Sea monsoon cross-equatorial flow) and models how anomalies non-linearly ripple across the subcontinent.
3. **Monsoon Impact Co-Pilot & Policy Advisory Extension:**  
   Translates raw atmospheric dynamics into actionable socio-economic metrics: **Kharif crop moisture stress**, **extreme flash flood risk**, **coastal wind hazard**, and **automated State Disaster Management (SDMA) policy advisories**.

---

## 2. Six-Block Physics-Aware Architecture

The engine is engineered as a sequential, modular 6-block spatiotemporal deep learning pipeline operating on an Indian domain grid of $32 \times 36$ ($1^\circ$ resolution, centres $6.5^\circ\text{N}–37.5^\circ\text{N}, 65.5^\circ\text{E}–100.5^\circ\text{E}$):

```
Climate Input [B, T=3, 6, 32, 36] + Geography [B, 2, 32, 36] + Calendar [B, 2]
    │
    ▼
┌────────────────────────────────────────────────────────┐
│ Block 1: Climate Input Encoder                         │  --> [B, T=3, 48, 32, 36]
│ Multi-modal fusion of climate, land-sea/orography, doy │
└────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────┐
│ Block 2: Multi-Scale Spatial Encoder                   │  --> [B, T=3, 64, 32, 36]
│ Parallel 3x3, 5x5, dilated-3x3 (d=2) convolutions      │
└────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────┐
│ Block 3: Temporal Dynamics Encoder (ConvLSTM)          │  --> [B, 64, 32, 36]
│ Spatiotemporal recurrent state transition (H_t, C_t)   │
└────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────┐
│ Block 4: Spatial Context / Attention Module (CBAM)     │  --> [B, 64, 32, 36]
│ Channel & spatial attention with shared MLP & 7x7 conv │
└────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────┐
│ Block 5: Forecast Decoder                              │  --> [B, 6, 32, 36]
│ Multi-layer conv decoding to atmospheric deltas (ΔS)   │
└────────────────────────────────────────────────────────┘
    │
    ▼
┌────────────────────────────────────────────────────────┐
│ Block 6: Residual State Transition                     │  --> [B, 6, 32, 36]
│ Physical residual addition + non-negative rain clamp   │
└────────────────────────────────────────────────────────┘
```

### Prevention of Drift and Blur:
- **Residual Delta Learning:** The model predicts step deltas $\Delta S = \text{Decoder}(H_T)$, executing $S_{t+1} = S_t + \Delta S \cdot \sigma$, preventing numerical collapse into blurry spatial averages.
- **Strict Physical Validity Constraints:** Precipitation is strictly constrained to $TP \ge 0.0\text{ mm/day}$ through out-of-place gradient-friendly clipping. Temperatures are bounded to the physical ERA5 range $[-40^\circ\text{C}, +60^\circ\text{C}]$.

---

## 3. Dataset & Preprocessing

The model is trained strictly using the provided Indian climate dataset:
- **`hackdata/india_climate_train.nc`**: 122.8 MB NetCDF dataset containing 6,209 daily ERA5 observations (2005–2021). Available for download from the official team kit release: [yug-monsoon-kit Releases (v1.0.0)](https://github.com/OfficialAbhinavSingh/yug-monsoon-kit/releases/tag/v1.0.0) (`india_climate_train.nc`).
  - SHA-256 Checksum: `f37bdbd0bd44e6fe951adf36c9a08620075f9d17de391b04c97d61f06797903b` (verified).
- **Channels (in exact order):**
  1. `t2m`: 2-meter air temperature (Kelvin)
  2. `tp`: Total daily precipitation (mm/day)
  3. `u10`: 10m zonal wind component (m/s)
  4. `v10`: 10m meridional wind component (m/s)
  5. `msl`: Mean sea level pressure (Pa)
  6. `tcwv`: Total column water vapor (kg/m²)
- **Static Geography:** Land-sea mask (`lsm`) and surface geopotential height (`z`), bundled locally as `static.pt` ($[2, 32, 36]$).

---

## 4. Quick Start & Execution

### 1. Model Training
To train the model from scratch on `hackdata/india_climate_train.nc`:
```bash
python train.py --epochs 3 --batch-size 32 --lr 0.001 --out weights.pt
```

### 2. Offline Inference Server
The official offline inference server is located at the repository root:
```bash
python serve.py --port 8000
```
- Binds to `0.0.0.0:8000`
- Zero external internet access required
- Grader `/health` and `/simulate` endpoints fully compliant
- 10-day rollouts execute in **~140 ms** on CPU (well below the 5.0 s limit)

### 3. Schema & API Compliance Verification
Run the official competition validator against the running server:
```bash
python -m monsoon_r1.schema_validator http://localhost:8000
```
```
health: OK
  [JSON lists (canonical)] no perturbation  OK   0.44s
  [JSON lists (canonical)] warm_ocean       OK   0.35s
  [JSON lists (canonical)] zeros            OK   0.39s
  [base64 (also tolerated)] no perturbation  OK   0.39s
  [base64 (also tolerated)] warm_ocean       OK   0.34s
  [base64 (also tolerated)] zeros            OK   0.35s
RESULT: all checks passed
```

### 4. Running the Complete Test Suite
Execute the recursive 110-test unit, integration, and physical evaluation suite:
```bash
pytest -v
```
**Results:** `110 passed in 10.02s` (100% pass rate).

---

## 5. Interactive Map Dashboard & Impact Extension

Launch the standalone interactive dashboard:
```bash
python -m dashboard.app --port 8501
```
Open **`http://localhost:8501`** in any web browser.

### Features:
- **Interactive Multi-Channel Map:** Canvas-based rendering of India's $32 \times 36$ grid across temperature, precipitation, wind speed, pressure, and moisture.
- **Timeline Prime vs. Multiverse Branching:**
  - Inject localized shocks:
    - *Bay of Bengal Marine Heatwave:* $+3^\circ\text{C}$ ocean shock.
    - *Arabian Sea Wind Shear:* $-5\text{ m/s}$ weaker monsoon flow.
    - *Pre-Monsoon Land Heat Dome:* $+4^\circ\text{C}$ northern India heatwave.
  - Interactive difference map showing causal anomaly dispersion over days 1 to 10.
- **Scientific Impact Co-Pilot:**
  - Regional Kharif Crop Stress Index across Indo-Gangetic Plain and Deccan Plateau.
  - Extreme Flash Flood Cell Warning ($> 150\text{ mm}$ accumulated precipitation).
  - Clean Wind Energy Density estimation ($\frac{1}{2}\rho v^3$).
  - Automated policy advisories for State Disaster Management Authorities (SDMA).

---

## 6. Repository Layout

```
├── serve.py                         # Official offline inference server (FastAPI/uvicorn, port 8000)
├── train.py                         # Model training script
├── Multiverse_of_Monsoon_Brief.pdf  # Official 2-page brief PDF (model architecture, failure modes & dashboard)
├── weights.pt                       # Trained PyTorch model checkpoint (1.57 MB)
├── static.pt                        # Bundled static geography tensor [2, 32, 36]
├── README.md                        # Comprehensive documentation and user guide
├── pyproject.toml                   # Project metadata and dependencies
├── scripts/                         # Offline evaluation and development tooling
│   ├── eval.py                      # Comprehensive holdout evaluation & benchmark suite
│   ├── generate_brief_pdf.py        # Reproducible 2-page brief PDF generator
│   └── verify_server.py             # Automated API contract & schema validation runner
├── hackdata/
│   ├── india_climate_train.nc       # ERA5 daily climate observations (2005–2021)
│   ├── static.npy                   # Static geography NumPy array
│   ├── stats.json                   # Normalization means and standard deviations
│   └── SHA256SUMS.txt               # Verification checksum
├── monsoon_r1/
│   ├── contract.py                  # API definitions, constants, and encoders
│   ├── scoring.py                   # Validity gate, metrics, and shock definitions
│   ├── train.py                     # Training loops and dataset loaders
│   └── models/                      # The 6 core architecture blocks
│       ├── climate_input_encoder.py
│       ├── multi_scale_spatial_encoder.py
│       ├── temporal_dynamics_encoder.py
│       ├── spatial_context_attention.py
│       ├── forecast_decoder.py
│       ├── residual_state_transition.py
│       └── multiverse_model.py
├── dashboard/                       # Interactive dashboard application
│   ├── app.py                       # Dashboard server (port 8501)
│   ├── impact.py                    # Socio-economic impact calculation engine
│   ├── static/                      # Bundled geographic polygons and fonts
│   └── templates/
│       └── index.html               # Single-page interactive map interface
└── tests/                           # 110-test unit, regression, & evaluation suite
```

---

## 7. Submission Checklist & Rule Compliance

| Hackathon Requirement | Status | Verification |
| :--- | :---: | :--- |
| **`serve.py` at repo root** | ✅ | Implemented, listens on `0.0.0.0:8000`, loads weights once at startup |
| **`train.py` at repo root** | ✅ | Implemented, trains on `hackdata/india_climate_train.nc` |
| **Model weights in repo** | ✅ | `weights.pt` (1.57 MB) stored directly inside repo root (< 2 GB) |
| **Static geography bundled** | ✅ | `static.pt` included in repo root, `static.npy` in `hackdata/` |
| **Offline execution** | ✅ | No internet / external API dependencies at runtime |
| **API Contract (/health, /simulate)** | ✅ | Passes `monsoon_r1.schema_validator` with 0 failures |
| **Latency limit (< 5.0 s / 10 steps)** | ✅ | ~135 ms on CPU (~37x faster than limit) |
| **Physical validity gate** | ✅ | Verified $TP \ge 0$, bounded $T_{2m}$, no NaN/Inf |
| **Causal shocks & null invariance** | ✅ | Shocks propagate coherently; zero shock changes nothing |
| **Interactive Dashboard & Impact Extension**| ✅ | Separate dashboard in `dashboard/` coupled live to `/simulate` |
| **Official 2-Page Brief PDF** | ✅ | `Multiverse_of_Monsoon_Brief.pdf` (strict 2 pages) in repository root |
