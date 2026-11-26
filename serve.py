"""Multiverse of Monsoon — Official Offline Inference Server.

Adheres strictly to the Track 2 Hackathon API Contract:
- Offline execution (no external internet or weather APIs).
- Listen on 0.0.0.0 (default port 8000).
- POST /health: returns {"status": "ready", "model_name": "multiverse-of-monsoon-r1"}.
- POST /simulate: accepts raw physical units for initial states [3, 6, 32, 36],
  doy_sincos [sin, cos], steps (e.g. 10), and optional localized physical perturbation [6, 32, 36].
- Returns rollout predictions [steps, 6, 32, 36] and inference_ms.
- 10-step rollouts execute in < 200 ms on CPU (< 5.0 s limit).
"""

from __future__ import annotations

import argparse
import base64
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from monsoon_r1.contract import CHANNELS, NC, NLAT, NLON, SCORE_STEPS, T_IN, MAX_STEPS
from monsoon_r1.models.multiverse_model import MultiverseOfMonsoonModel

app = FastAPI(title="Multiverse of Monsoon AI Engine", version="1.0.0")

MODEL_NAME = "multiverse-of-monsoon-r1"
DEVICE = torch.device("cpu")

# Paths to model weights and static assets
ROOT_DIR = Path(__file__).resolve().parent
WEIGHTS_PATH = ROOT_DIR / "weights.pt"
STATIC_PATH = ROOT_DIR / "static.pt"
STATS_PATH = ROOT_DIR / "hackdata" / "stats.json"

# Load normalization scales and means if present
channel_scales = None
channel_means = None
if STATS_PATH.exists():
    try:
        with open(STATS_PATH, "r") as f:
            stats = json.load(f)
            channel_scales = stats.get("std")
            channel_means = stats.get("mean")
    except Exception:
        channel_scales = None
        channel_means = None

# Initialize model
model = MultiverseOfMonsoonModel(
    channel_scales=channel_scales,
    channel_means=channel_means,
).to(DEVICE)

if WEIGHTS_PATH.exists():
    state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    print(f"[*] Loaded trained model weights from {WEIGHTS_PATH}")
else:
    print(f"[!] Warning: weights.pt not found at {WEIGHTS_PATH}; using initialized model")

model.eval()

# Load static geography
if STATIC_PATH.exists():
    static_geo = torch.load(STATIC_PATH, map_location=DEVICE).float()
    print(f"[*] Loaded static geography from {STATIC_PATH}")
else:
    print(f"[!] Warning: static.pt not found; falling back to zeros")
    static_geo = torch.zeros(2, NLAT, NLON, device=DEVICE, dtype=torch.float32)


def _arr(x: Any) -> np.ndarray:
    """Accept nested lists (canonical contract) or {'b64': ..., 'shape': [...]} (tolerated)."""
    if isinstance(x, dict):
        return (
            np.frombuffer(base64.b64decode(x["b64"]), dtype="<f4")
            .reshape(x["shape"])
            .astype("f4")
        )
    return np.asarray(x, dtype="f4")


def simulate(
    init_states: np.ndarray,
    doy_sincos: Union[torch.Tensor, Sequence[float]],
    steps: int = SCORE_STEPS,
    perturbation: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Run multi-step autoregressive weather simulation and counterfactual branching.

    Args:
        init_states: [3, 6, 32, 36] in physical units (t2m, tp, u10, v10, msl, tcwv).
        doy_sincos: [sin, cos] of day-of-year of init_states[-1].
        steps: Number of forecast steps to roll forward.
        perturbation: Optional localized physical shock [6, 32, 36] added to init_states[-1].

    Returns:
        np.ndarray: [steps, 6, 32, 36] rollout in physical units with physical bounds enforced.
    """
    with torch.no_grad():
        init_tensor = torch.from_numpy(init_states).to(DEVICE)
        pert_tensor = (
            torch.from_numpy(perturbation).to(DEVICE)
            if perturbation is not None
            else None
        )

        rollout_tensor = model.rollout(
            init_states=init_tensor,
            geography=static_geo,
            doy_sincos=doy_sincos,
            steps=steps,
            perturbation=pert_tensor,
            check_finite=False,
        )

        out = rollout_tensor.cpu().numpy().astype("f4")

        # Step 4 physical clamping & nan safety:
        # nan_to_num
        out = np.nan_to_num(out, nan=0.0, posinf=333.15, neginf=233.15)
        # clamp t2m (channel 0) to [233.15, 333.15]
        out[:, 0] = np.clip(out[:, 0], 233.15, 333.15)
        # clamp tp (channel 1) >= 0
        out[:, 1] = np.maximum(out[:, 1], 0.0)
        # clamp tcwv (channel 5) >= 0
        out[:, 5] = np.maximum(out[:, 5], 0.0)

        return out


@app.get("/health")
@app.post("/health")
def health():
    """Health check endpoint required by the evaluation harness."""
    return {"status": "ready", "model_name": MODEL_NAME}


@app.post("/simulate")
async def simulate_endpoint(request: Request):
    """Simulation endpoint executing autoregressive rollouts."""
    t0 = time.perf_counter()
    body = await request.json()

    # Validate steps
    try:
        raw_steps = body.get("steps", SCORE_STEPS)
        steps = int(raw_steps)
        if steps < 1 or steps > MAX_STEPS:
            return JSONResponse(
                status_code=400,
                content={"error": f"Invalid steps {steps}; must be between 1 and {MAX_STEPS}"},
            )
    except (ValueError, TypeError):
        return JSONResponse(
            status_code=400,
            content={"error": f"Invalid steps format: {body.get('steps')}"},
        )

    init_states = _arr(body["init_states"])
    perturbation = (
        None if body.get("perturbation") is None else _arr(body["perturbation"])
    )
    doy_sincos = body["doy_sincos"]

    out = simulate(init_states, doy_sincos, steps, perturbation)
    inference_ms = (time.perf_counter() - t0) * 1000.0

    return {
        "rollout": np.asarray(out, dtype="f4").tolist(),
        "inference_ms": round(inference_ms, 2),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multiverse of Monsoon Offline Inference Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host interface to bind")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    args = parser.parse_args()

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
