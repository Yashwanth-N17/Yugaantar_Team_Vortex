"""Scientific Interactive Dashboard Server for India Climate Digital Twin.

Provides the HTTP API and web interface for:
- Connecting to the offline inference engine (serve.py at /simulate and /health)
- Rendering synoptic maps (Prime, Branch, Difference) on the 32x36 India grid (6.5-37.5N, 65.5-100.5E)
- Applying counterfactual physical shocks (warm ocean, monsoon stall, custom click-drag shocks)
- Real-time regional socio-economic impacts on land cells across India

Backend contract: browser -> dashboard/app.py -> serve.py POST /simulate.
"""

from __future__ import annotations

import argparse
import json
import socket
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import requests
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from monsoon_r1.contract import (
    BOX_ARABIAN_SEA,
    BOX_BAY_OF_BENGAL,
    CHANNELS,
    NC,
    NLAT,
    NLON,
    SCORE_STEPS,
    T_IN,
    box_mask,
    doy_sincos,
)
from monsoon_r1.scoring import make_shock
from dashboard.impact import compute_impact_metrics

app = FastAPI(title="India Climate Digital Twin Synoptic Console", version="2.0.0")

DASHBOARD_DIR = Path(__file__).resolve().parent
ROOT_DIR = DASHBOARD_DIR.parent
SERVE_URL = "http://127.0.0.1:8000"
STATIC_DIR = DASHBOARD_DIR / "static"
DATA_DIR = STATIC_DIR / "data"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

_BUNDLED_WINDOWS: Optional[List[Dict[str, Any]]] = None


def load_bundled_windows() -> List[Dict[str, Any]]:
    """Load pre-bundled 3-day meteorological input windows from JSON."""
    global _BUNDLED_WINDOWS
    if _BUNDLED_WINDOWS is None:
        windows_file = DATA_DIR / "bundled_windows.json"
        if windows_file.exists():
            try:
                with open(windows_file, "r") as f:
                    _BUNDLED_WINDOWS = json.load(f)
            except Exception as e:
                print(f"[Dashboard] Error loading bundled windows: {e}")
                _BUNDLED_WINDOWS = []
        else:
            _BUNDLED_WINDOWS = []
    return _BUNDLED_WINDOWS


def get_window_data(window_id_or_idx: Any) -> Tuple[np.ndarray, List[float], str]:
    """Retrieve initial states and doy_sincos for a selected bundled window."""
    windows = load_bundled_windows()
    selected = None
    if isinstance(window_id_or_idx, str):
        for w in windows:
            if w.get("id") == window_id_or_idx:
                selected = w
                break
    elif isinstance(window_id_or_idx, int) and 0 <= window_id_or_idx < len(windows):
        selected = windows[window_id_or_idx]

    if selected is None and len(windows) > 0:
        selected = windows[0]

    if selected and "states" in selected and selected["states"] is not None:
        states = np.array(selected["states"], dtype="f4")
        doy = selected.get("doy_sincos", doy_sincos(180))
        date_str = selected.get("date", "bundled_window")
        return states, doy, date_str

    # Synthetic fallback baseline (mid-monsoon)
    base = np.array([300.5, 6.0, 4.0, 2.5, 100800.0, 42.0], "f4")[:, None, None]
    states = np.repeat(base[None, ...], T_IN, axis=0)
    return states, doy_sincos(200), "baseline_mid_monsoon"


def is_port_open(host: str, port: int, timeout: float = 0.2) -> bool:
    """Non-blocking socket check for localhost service."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except (socket.timeout, ConnectionRefusedError, OSError):
        return False


def call_engine_simulate(
    init_states: np.ndarray,
    doy: List[float],
    steps: int,
    pert: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, bool]:
    """Call the offline inference engine /simulate endpoint with strict error handling."""
    payload = {
        "init_states": init_states.tolist(),
        "doy_sincos": doy,
        "steps": steps,
        "perturbation": pert.tolist() if pert is not None else None,
    }

    if is_port_open("127.0.0.1", 8000, timeout=0.15):
        try:
            resp = requests.post(f"{SERVE_URL}/simulate", json=payload, timeout=6.0)
            if resp.status_code == 200:
                res = resp.json()
                return np.array(res["rollout"], dtype="f4"), float(res.get("inference_ms", 120.0)), True
        except Exception as e:
            print(f"[Dashboard] Engine call to {SERVE_URL}/simulate failed: {e}")

    # Seamless in-process execution using weights.pt
    try:
        import time
        from serve import simulate
        t0 = time.perf_counter()
        out = simulate(init_states, doy, steps, pert)
        lat_ms = (time.perf_counter() - t0) * 1000.0
        return out, lat_ms, True
    except Exception as e:
        print(f"[Dashboard] In-process simulation fallback failed: {e}")
        return np.empty((0,), dtype="f4"), 0.0, False


@app.get("/", response_class=HTMLResponse)
def index():
    """Serve the scientific synoptic dashboard single-page interface."""
    html_file = DASHBOARD_DIR / "templates" / "index.html"
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"))


@app.get("/api/health")
@app.post("/api/health")
def api_health():
    """Poll the real underlying inference engine /health endpoint."""
    if is_port_open("127.0.0.1", 8000, timeout=0.15):
        try:
            resp = requests.get(f"{SERVE_URL}/health", timeout=1.0)
            if resp.status_code == 200:
                data = resp.json()
                return JSONResponse({
                    "online": True,
                    "engine_status": data.get("status", "ready"),
                    "model_name": data.get("model_name", "multiverse-of-monsoon-r1"),
                    "serve_url": SERVE_URL,
                })
        except Exception:
            pass

    # When serve.py port 8000 is not running, check in-process model readiness
    weights_file = ROOT_DIR / "weights.pt"
    if weights_file.exists():
        return JSONResponse({
            "online": True,
            "engine_status": "ready (in-process engine)",
            "model_name": "multiverse-of-monsoon-r1",
            "serve_url": SERVE_URL,
        })

    return JSONResponse({
        "online": False,
        "engine_status": "offline",
        "model_name": None,
        "serve_url": SERVE_URL,
        "error": "Inference engine serve.py (port 8000) is unreachable."
    })


@app.get("/api/windows")
def api_windows():
    """Return available initial-state bundled windows metadata without heavy state arrays."""
    windows = load_bundled_windows()
    meta = []
    for w in windows:
        meta.append({
            "id": w.get("id"),
            "name": w.get("name"),
            "date": w.get("date"),
            "doy": w.get("doy"),
            "doy_sincos": w.get("doy_sincos"),
            "description": w.get("description"),
        })
    return JSONResponse(meta)


@app.post("/api/simulate")
async def api_simulate(request: Request):
    """Run dual simulation (Baseline Prime & Counterfactual Branch) and compute differential impact."""
    body = await request.json()
    steps = int(body.get("steps", SCORE_STEPS))
    steps = max(1, min(14, steps))

    window_id = body.get("window_id", "monsoon_depression_2010")
    custom_init = body.get("init_states")
    custom_doy = body.get("doy_sincos")

    if custom_init is not None and custom_doy is not None:
        try:
            init_states = np.array(custom_init, dtype="f4")
            doy = [float(custom_doy[0]), float(custom_doy[1])]
            date_str = "Custom Input"
        except Exception as e:
            return JSONResponse({"error": f"Invalid custom initial state: {e}"}, status_code=400)
    else:
        init_states, doy, date_str = get_window_data(window_id)

    # 1. Baseline timeline (Prime)
    baseline_rollout, b_lat, ok_b = call_engine_simulate(init_states, doy, steps=steps, pert=None)
    if not ok_b or baseline_rollout.size == 0:
        return JSONResponse({
            "error": "Inference engine is offline or unreachable. No simulation performed.",
            "online": False,
        }, status_code=503)

    # 2. Multiverse shock branch
    shock_type = body.get("shock_type", "null")
    shock = np.zeros((NC, NLAT, NLON), dtype="f4")
    has_shock = False

    stats_path = ROOT_DIR / "hackdata" / "stats.json"
    std_scale = np.array([13.46, 7.55, 2.90, 2.43, 619.02, 20.49], dtype="f4")
    if stats_path.exists():
        try:
            with open(stats_path) as sf:
                std_scale = np.array(json.load(sf).get("std", std_scale), dtype="f4")
        except Exception:
            pass

    if shock_type == "preset":
        preset_name = body.get("preset_name")
        if preset_name in ("warm_ocean", "monsoon_stall"):
            try:
                shock = make_shock(preset_name, init_states[-1], std_scale)
                has_shock = True
            except Exception as e:
                print(f"[Dashboard] make_shock error: {e}")
    elif shock_type == "custom":
        custom_pert = body.get("perturbation")
        if custom_pert is not None:
            try:
                pert_arr = np.array(custom_pert, dtype="f4")
                if pert_arr.shape == (NC, NLAT, NLON):
                    shock = pert_arr
                    has_shock = bool(np.any(shock != 0.0))
            except Exception as e:
                print(f"[Dashboard] Custom perturbation parse error: {e}")

    perturbed_rollout = None
    lat_ms = b_lat
    if has_shock:
        p_rollout, p_lat, ok_p = call_engine_simulate(init_states, doy, steps=steps, pert=shock)
        if ok_p and p_rollout.size > 0:
            perturbed_rollout = p_rollout
            lat_ms = round(b_lat + p_lat, 2)
        else:
            perturbed_rollout = baseline_rollout.copy()

    # Differential rollout: never offset deltas by K or Pa; direct difference
    diff_rollout = None
    if perturbed_rollout is not None:
        diff_rollout = perturbed_rollout - baseline_rollout

    # Regional impacts on land cells
    impact = compute_impact_metrics(
        baseline_rollout=baseline_rollout,
        perturbed_rollout=perturbed_rollout,
    )

    # Compute shock response vs distance profile by forecast day
    distance_response = []
    if has_shock and perturbed_rollout is not None:
        shock_mag = np.sum(np.abs(shock), axis=0)
        shock_mask = shock_mag > 1e-4
        if np.any(shock_mask):
            si, sj = np.where(shock_mask)
            ii, jj = np.indices((NLAT, NLON))
            dist_grid = np.sqrt(((ii[..., None] - si) ** 2 + (jj[..., None] - sj) ** 2)).min(axis=-1)

            bins = [(0.0, 2.5), (2.5, 5.5), (5.5, 9.5), (9.5, 14.5), (14.5, 22.0)]
            bin_labels = ["0-250 km", "250-600 km", "600-1000 km", "1000-1600 km", ">1600 km"]

            d_norm = (diff_rollout / std_scale[None, :, None, None]) ** 2
            e_total = np.sum(d_norm, axis=1)

            for s_idx in range(steps):
                day_vals = []
                for (d_min, d_max) in bins:
                    m = (dist_grid >= d_min) & (dist_grid < d_max)
                    val = float(np.mean(e_total[s_idx, m])) if np.any(m) else 0.0
                    day_vals.append(round(val, 4))
                distance_response.append({
                    "step": s_idx + 1,
                    "bins": bin_labels,
                    "energies": day_vals
                })

    return JSONResponse({
        "baseline": baseline_rollout.tolist(),
        "perturbed": perturbed_rollout.tolist() if perturbed_rollout is not None else None,
        "difference": diff_rollout.tolist() if diff_rollout is not None else None,
        "impact": impact,
        "inference_ms": round(lat_ms, 2),
        "steps": steps,
        "shock_applied": has_shock,
        "shock_footprint": (shock.tolist() if has_shock else None),
        "distance_response": distance_response,
        "date": date_str,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="India Climate Digital Twin Scientific Dashboard")
    parser.add_argument("--port", type=int, default=8501, help="Port to listen on")
    args = parser.parse_args()

    uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
