"""Interactive Map Dashboard & Impact Co-Pilot Server.

Provides a live web dashboard to visualize 10-day forecasts, inject
counterfactual shocks (Multiverse branching), and view real-time socio-economic
impacts (crop stress, flood risk, wind energy, and policy co-pilot).

Usage:
    python -m dashboard.app --port 8501
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import h5py
import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

from monsoon_r1.contract import CHANNELS, NC, NLAT, NLON, SCORE_STEPS, T_IN, doy_sincos
from monsoon_r1.scoring import make_shock
from dashboard.impact import compute_impact_metrics

app = FastAPI(title="Multiverse of Monsoon Dashboard", version="1.0.0")

DASHBOARD_DIR = Path(__file__).resolve().parent
ROOT_DIR = DASHBOARD_DIR.parent
DATA_PATH = ROOT_DIR / "hackdata" / "india_climate_train.nc"
SERVE_URL = "http://127.0.0.1:8000"


def get_sample_input(sample_idx: int = 5900) -> tuple[np.ndarray, list[float]]:
    """Retrieve realistic 3-day initial window from dataset."""
    if DATA_PATH.exists():
        with h5py.File(DATA_PATH, "r") as f:
            total = f["data"].shape[0]
            idx = min(sample_idx, total - T_IN - SCORE_STEPS)
            window = f["data"][idx : idx + T_IN].astype("f4")
    else:
        # Realistic fallback baseline
        base = np.array([301.0, 5.0, 3.0, 2.0, 100800.0, 38.0], "f4")[:, None, None]
        window = np.repeat(base[None, ...], T_IN, axis=0)

    cal = doy_sincos(180)  # Mid-monsoon day of year
    return window, cal


def call_engine_simulate(
    init_states: np.ndarray,
    doy: list[float],
    steps: int,
    pert: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, float]:
    """Call the live inference engine /simulate endpoint."""
    payload = {
        "init_states": init_states.tolist(),
        "doy_sincos": doy,
        "steps": steps,
        "perturbation": pert.tolist() if pert is not None else None,
    }

    try:
        resp = requests.post(f"{SERVE_URL}/simulate", json=payload, timeout=10.0)
        if resp.status_code == 200:
            res = resp.json()
            return np.array(res["rollout"], dtype="f4"), float(res.get("inference_ms", 120.0))
    except Exception as e:
        print(f"[Dashboard] Engine call to {SERVE_URL} failed ({e}), using in-process fallback...")

    # Standalone in-process fallback if serve.py is not yet running
    from serve import simulate
    import time
    t0 = time.perf_counter()
    out = simulate(init_states, doy, steps, pert)
    lat_ms = (time.perf_counter() - t0) * 1000.0
    return out, lat_ms


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the interactive dashboard single-page interface."""
    html_file = DASHBOARD_DIR / "templates" / "index.html"
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"))


@app.post("/api/simulate")
async def api_simulate(request: Request):
    """Endpoint for frontend to trigger live simulations and multiverse shocks."""
    body = await request.json()
    mode = body.get("mode", "prime")
    steps = int(body.get("steps", SCORE_STEPS))

    init_states, cal = get_sample_input(sample_idx=5850)

    # 1. Baseline timeline (Timeline Prime)
    baseline_rollout, lat_ms = call_engine_simulate(init_states, cal, steps=steps, pert=None)

    # 2. Multiverse shock branch
    shock = None
    if mode in ("shock_bob", "diff"):
        shock = np.zeros((NC, NLAT, NLON), dtype="f4")
        shock[0, 2:14, 17:29] = 3.0  # +3°C warm ocean shock in Bay of Bengal
    elif mode == "shock_wind":
        shock = np.zeros((NC, NLAT, NLON), dtype="f4")
        shock[2, 10:25, 2:15] = -5.0  # -5 m/s weaker Arabian Sea monsoon flow
    elif mode == "shock_heatwave":
        shock = np.zeros((NC, NLAT, NLON), dtype="f4")
        shock[0, 8:16, 8:20] = 4.0   # +4°C heat dome in Northern India

    perturbed_rollout = None
    if shock is not None:
        perturbed_rollout, p_lat = call_engine_simulate(init_states, cal, steps=steps, pert=shock)
        lat_ms = max(lat_ms, p_lat)

    # Determine rollout to display
    if mode == "diff" and perturbed_rollout is not None:
        display_rollout = perturbed_rollout - baseline_rollout
    elif mode.startswith("shock_") and perturbed_rollout is not None:
        display_rollout = perturbed_rollout
    else:
        display_rollout = baseline_rollout

    impact = compute_impact_metrics(
        baseline_rollout=baseline_rollout,
        perturbed_rollout=perturbed_rollout,
    )

    return JSONResponse({
        "rollout": display_rollout.tolist(),
        "baseline": baseline_rollout.tolist(),
        "perturbed": perturbed_rollout.tolist() if perturbed_rollout is not None else None,
        "impact": impact,
        "inference_ms": lat_ms,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multiverse of Monsoon Interactive Dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Port to listen on")
    args = parser.parse_args()

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
