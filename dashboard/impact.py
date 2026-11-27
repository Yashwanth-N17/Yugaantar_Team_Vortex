"""Scientific Impact Calculator: Regional Climate & Disaster Impact Analysis.

Translates 10-day physical climate simulations (t2m, tp, u10, v10, msl, tcwv)
into rigorous socio-economic and regional impact metrics across India on land cells:
1. Indo-Gangetic Plain (~24-31°N, 74-88°E)
2. Western Ghats (~8-20°N, 73-77°E)
3. Deccan Plateau (~13-22°N, 74-82°E)
4. Odisha / Andhra Coast (~15-21°N, 80-87°E)
5. Gujarat / Rajasthan (~21-30°N, 68-76°E)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np

from monsoon_r1.contract import NLAT, NLON

# Grid specifications:
# LAT: 6.5 .. 37.5, step 1.0 (idx = round(lat - 6.5))
# LON: 65.5 .. 100.5, step 1.0 (idx = round(lon - 65.5))

# Regional coordinate bounds: (lat_min, lat_max, lon_min, lon_max)
REGIONAL_SPECS = {
    "Indo-Gangetic Plain": {
        "lat": (24.0, 31.0),
        "lon": (74.0, 88.0),
        "desc": "Primary agricultural belt (Rice/Wheat)"
    },
    "Western Ghats": {
        "lat": (8.0, 20.0),
        "lon": (73.0, 77.0),
        "desc": "Orograpic catchment & high runoff"
    },
    "Deccan Plateau": {
        "lat": (13.0, 22.0),
        "lon": (74.0, 82.0),
        "desc": "Rainfed agrarian & semi-arid core"
    },
    "Odisha / Andhra Coast": {
        "lat": (15.0, 21.0),
        "lon": (80.0, 87.0),
        "desc": "Cyclone-prone Bay of Bengal littoral"
    },
    "Gujarat / Rajasthan": {
        "lat": (21.0, 30.0),
        "lon": (68.0, 76.0),
        "desc": "Arid northwest transition margin"
    },
}

_LAND_MASK: Optional[np.ndarray] = None


def get_land_mask() -> np.ndarray:
    """Retrieve official binary land-sea mask from static assets."""
    global _LAND_MASK
    if _LAND_MASK is None:
        static_pt = Path(__file__).resolve().parent.parent / "static.pt"
        if static_pt.exists():
            try:
                import torch
                geo = torch.load(static_pt, map_location="cpu")
                _LAND_MASK = (geo[0].numpy() > 0.5)
            except Exception:
                pass
        if _LAND_MASK is None:
            static_npy = Path(__file__).resolve().parent.parent / "hackdata" / "static.npy"
            if not static_npy.exists():
                static_npy = Path(__file__).resolve().parent.parent / "static.npy"
            if static_npy.exists():
                try:
                    geo = np.load(static_npy)
                    _LAND_MASK = (geo[0] > 0.5)
                except Exception:
                    pass
        if _LAND_MASK is None:
            _LAND_MASK = np.ones((NLAT, NLON), dtype=bool)
    return _LAND_MASK


def get_region_cell_mask(lat_box: tuple[float, float], lon_box: tuple[float, float], land_only: bool = True) -> np.ndarray:
    """Generate boolean mask for a given region bounding box, restricted to land cells."""
    r0 = max(0, int(round(lat_box[0] - 6.5)))
    r1 = min(NLAT, int(round(lat_box[1] - 6.5)) + 1)
    c0 = max(0, int(round(lon_box[0] - 65.5)))
    c1 = min(NLON, int(round(lon_box[1] - 65.5)) + 1)

    mask = np.zeros((NLAT, NLON), dtype=bool)
    mask[r0:r1, c0:c1] = True
    if land_only:
        lsm = get_land_mask()
        land_in_box = mask & lsm
        if np.any(land_in_box):
            mask = land_in_box
    return mask


def compute_impact_metrics(
    baseline_rollout: np.ndarray,
    perturbed_rollout: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Compute scientific impact indicators from simulation rollouts.

    Args:
        baseline_rollout: [steps, 6, 32, 36] rollout in physical units.
        perturbed_rollout: Optional [steps, 6, 32, 36] counterfactual rollout.

    Returns:
        Dict containing regional impact indices, disaster warnings, and advisories.
    """
    steps, c, h, w = baseline_rollout.shape
    lsm = get_land_mask()

    # Channels: 0: t2m (K), 1: tp (mm/day), 2: u10 (m/s), 3: v10 (m/s), 4: msl (Pa), 5: tcwv (kg/m2)
    b_t2m_c = baseline_rollout[:, 0] - 273.15
    b_tp_mm = baseline_rollout[:, 1]
    b_wind = np.sqrt(baseline_rollout[:, 2] ** 2 + baseline_rollout[:, 3] ** 2)

    has_pert = perturbed_rollout is not None and perturbed_rollout.shape == baseline_rollout.shape
    if has_pert:
        p_t2m_c = perturbed_rollout[:, 0] - 273.15
        p_tp_mm = perturbed_rollout[:, 1]
        p_wind = np.sqrt(perturbed_rollout[:, 2] ** 2 + perturbed_rollout[:, 3] ** 2)

    # 1. 10m wind-derived power density (W/m2) = 0.5 * rho * v_10^3
    rho_air = 1.225
    b_wpd = 0.5 * rho_air * (b_wind ** 3)

    # 2. Regional Analysis
    regional_insights = []
    advisories: List[str] = []

    for name, spec in REGIONAL_SPECS.items():
        reg_mask = get_region_cell_mask(spec["lat"], spec["lon"], land_only=True)
        cell_count = int(np.sum(reg_mask))
        if cell_count == 0:
            reg_mask = get_region_cell_mask(spec["lat"], spec["lon"], land_only=False)
            cell_count = int(np.sum(reg_mask))

        # Accumulated rain (sum over steps, mean over land cells in region)
        b_acc_rain_grid = np.sum(b_tp_mm, axis=0)  # [32, 36]
        b_reg_acc_rain = float(np.mean(b_acc_rain_grid[reg_mask]))

        # Dry-spell days: count of days where rain < 1.0 mm/day
        b_dry_days_grid = np.sum(b_tp_mm < 1.0, axis=0)
        b_reg_dry_days = float(np.mean(b_dry_days_grid[reg_mask]))

        # Heatwave days (>40°C)
        b_hot_days_grid = np.sum(b_t2m_c > 40.0, axis=0)
        b_reg_hot_days = float(np.mean(b_hot_days_grid[reg_mask]))

        # 10m wind speed and derived power
        b_reg_wind = float(np.mean(b_wind[:, reg_mask]))
        b_reg_wpd = float(np.mean(b_wpd[:, reg_mask]))

        reg_data = {
            "region": name,
            "description": spec["desc"],
            "land_cells": cell_count,
            "accumulated_rain_mm": round(b_reg_acc_rain, 1),
            "dry_spell_days": round(b_reg_dry_days, 1),
            "days_above_40c": round(b_reg_hot_days, 1),
            "mean_wind_10m_ms": round(b_reg_wind, 1),
            "wind_power_density_10m_wm2": round(b_reg_wpd, 1),
        }

        # Branch deltas if counterfactual is active
        if has_pert:
            p_acc_rain_grid = np.sum(p_tp_mm, axis=0)
            p_reg_acc_rain = float(np.mean(p_acc_rain_grid[reg_mask]))
            p_dry_days_grid = np.sum(p_tp_mm < 1.0, axis=0)
            p_reg_dry_days = float(np.mean(p_dry_days_grid[reg_mask]))
            p_hot_days_grid = np.sum(p_t2m_c > 40.0, axis=0)
            p_reg_hot_days = float(np.mean(p_hot_days_grid[reg_mask]))
            p_wpd = 0.5 * rho_air * (p_wind ** 3)
            p_reg_wpd = float(np.mean(p_wpd[:, reg_mask]))

            reg_data["branch_accumulated_rain_mm"] = round(p_reg_acc_rain, 1)
            reg_data["delta_accumulated_rain_mm"] = round(p_reg_acc_rain - b_reg_acc_rain, 1)
            reg_data["delta_dry_spell_days"] = round(p_reg_dry_days - b_reg_dry_days, 1)
            reg_data["delta_days_above_40c"] = round(p_reg_hot_days - b_reg_hot_days, 1)
            reg_data["delta_wind_power_wm2"] = round(p_reg_wpd - b_reg_wpd, 1)

            # Targeted plain advisories tied to numbers
            d_rain = p_reg_acc_rain - b_reg_acc_rain
            d_hot = p_reg_hot_days - b_reg_hot_days
            if abs(d_rain) >= 15.0:
                direction = "surplus" if d_rain > 0 else "deficit"
                advisories.append(
                    f"{name}: Counterfactual shock shifts accumulated rainfall by {d_rain:+.1f} mm ({direction}). "
                    f"Baseline: {b_reg_acc_rain:.1f} mm; Branch: {p_reg_acc_rain:.1f} mm over {steps}-day forecast horizon."
                )
            if d_hot >= 1.0:
                advisories.append(
                    f"{name}: Heat stress risk elevated by {d_hot:+.1f} additional days exceeding 40.0°C (Branch: {p_reg_hot_days:.1f} days)."
                )
        else:
            if b_reg_acc_rain > 120.0:
                advisories.append(
                    f"{name}: Heavy cumulative precipitation of {b_reg_acc_rain:.1f} mm projected over {steps} days. Elevated catchment runoff risk."
                )
            elif b_reg_dry_days >= 7.0:
                advisories.append(
                    f"{name}: Extended dry spell observed with {b_reg_dry_days:.1f} days exhibiting <1.0 mm/day rainfall."
                )
            if b_reg_hot_days >= 3.0:
                advisories.append(
                    f"{name}: Sustained severe heat recorded with {b_reg_hot_days:.1f} days surpassing 40.0°C."
                )

        regional_insights.append(reg_data)

    # 3. Shock Attribution (if counterfactual rollout exists)
    shock_attribution = None
    if has_pert:
        delta_t2m = p_t2m_c - b_t2m_c
        delta_tp = p_tp_mm - b_tp_mm
        delta_wind = p_wind - b_wind
        delta_msl = (perturbed_rollout[:, 4] - baseline_rollout[:, 4]) / 100.0

        mean_dt = float(np.mean(delta_t2m[:, lsm]))
        max_dt = float(np.max(np.abs(delta_t2m)))
        mean_dr = float(np.mean(delta_tp[:, lsm]))
        max_dr = float(np.max(np.abs(delta_tp)))
        mean_dw = float(np.mean(delta_wind[:, lsm]))

        shock_attribution = {
            "mean_land_temp_shift_k": round(mean_dt, 3),
            "max_temp_shift_k": round(max_dt, 2),
            "mean_land_rain_shift_mm": round(mean_dr, 2),
            "max_rain_shift_mm": round(max_dr, 2),
            "mean_land_wind_shift_ms": round(mean_dw, 2),
            "active_steps": steps,
        }

    if not advisories:
        advisories.append(
            f"All 5 evaluated regions remain within standard climatological variance bounds across the {steps}-day rollout horizon."
        )

    return {
        "regional_insights": regional_insights,
        "shock_attribution": shock_attribution,
        "policy_advisories": advisories,
    }
