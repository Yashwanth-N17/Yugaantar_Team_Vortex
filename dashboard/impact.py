"""Scientific & Social-Impact Extension: Monsoon Climate & Disaster Co-Pilot.

Translates 10-day physical climate simulations (t2m, tp, u10, v10, msl, tcwv)
into actionable socio-economic and scientific impact metrics across India:
1. Agricultural Drought & Kharif Crop Stress Index
2. Extreme Precipitation & Flash Flood Risk Alerts
3. Coastal Cyclone & Wind Hazard Indices
4. Renewable Wind Energy Capacity Factors
5. Actionable Policy Co-Pilot Recommendations for State Disaster Management
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np


# Key agricultural and economic regional bounding boxes on India 32x36 grid:
# Latitudes 8°N - 39°N (idx 0 to 31, top to bottom: 39°N down to 8°N)
# Longitudes 68°E - 103°E (idx 0 to 35, left to right: 68°E up to 103°E)
REGIONS = {
    "Indo-Gangetic Plain (Paddy/Wheat)": {"lat": slice(8, 14), "lon": slice(8, 22)},
    "Deccan Plateau (Cotton/Pulses)": {"lat": slice(16, 24), "lon": slice(8, 18)},
    "Western Ghats (High Rainfall/Runoff)": {"lat": slice(18, 28), "lon": slice(4, 9)},
    "Bay of Bengal Coastal Zone": {"lat": slice(14, 25), "lon": slice(18, 28)},
    "Gujarat & Northwest Arid Zone": {"lat": slice(10, 18), "lon": slice(2, 8)},
}


def compute_impact_metrics(
    baseline_rollout: np.ndarray,
    perturbed_rollout: np.ndarray | None = None,
) -> Dict[str, Any]:
    """Compute comprehensive impact indicators from 10-day simulation rollouts.

    Args:
        baseline_rollout: [steps, 6, 32, 36] rollout in physical units.
        perturbed_rollout: Optional [steps, 6, 32, 36] counterfactual rollout.

    Returns:
        Dict containing regional impact indices, disaster warnings, and policy advisories.
    """
    steps, c, h, w = baseline_rollout.shape

    # Channels: 0: t2m (K), 1: tp (mm/day), 2: u10 (m/s), 3: v10 (m/s), 4: msl (Pa), 5: tcwv (kg/m2)
    t2m_c = baseline_rollout[:, 0] - 273.15  # Celsius
    tp_mm = baseline_rollout[:, 1]           # mm/day
    wind_spd = np.sqrt(baseline_rollout[:, 2] ** 2 + baseline_rollout[:, 3] ** 2)
    msl_hpa = baseline_rollout[:, 4] / 100.0  # hPa

    # 1. Extreme Flood and Inundation Alerts
    max_daily_rain = float(np.max(tp_mm))
    total_acc_rain = np.sum(tp_mm, axis=0)  # [32, 36]
    flood_risk_grid = (total_acc_rain > 150.0).astype(int) + (total_acc_rain > 300.0).astype(int)
    high_flood_cells = int(np.sum(total_acc_rain > 150.0))

    # 2. Agricultural Drought & Heat Stress Index (GDD & Rain Deficit)
    heatwave_mask = (t2m_c > 40.0).astype(int)
    heatwave_days = int(np.sum(heatwave_mask))
    dry_spell_grid = (tp_mm < 1.0).astype(int).sum(axis=0)  # days with < 1mm
    crop_stress_index = float(np.mean(dry_spell_grid) / steps * 100.0)

    # 3. Wind Energy Potential Index (normalized cubic wind flux ~ 0.5 * rho * v^3)
    wind_power_density = 0.5 * 1.225 * (wind_spd ** 3)  # W/m2
    mean_wind_power = float(np.mean(wind_power_density))

    # 4. Regional Impact Summary
    regional_insights = []
    for reg_name, box in REGIONS.items():
        reg_rain = float(np.mean(tp_mm[:, box["lat"], box["lon"]]))
        reg_temp = float(np.mean(t2m_c[:, box["lat"], box["lon"]]))
        reg_wind = float(np.mean(wind_spd[:, box["lat"], box["lon"]]))

        status = "Normal"
        if reg_rain > 25.0:
            status = "Flash Flood Warning"
        elif reg_rain < 2.0 and reg_temp > 35.0:
            status = "Drought & Heat Stress"
        elif reg_wind > 15.0:
            status = "Gale / High Wind Warning"

        regional_insights.append({
            "region": reg_name,
            "mean_rain_mm_day": round(reg_rain, 1),
            "mean_temp_c": round(reg_temp, 1),
            "mean_wind_ms": round(reg_wind, 1),
            "status": status,
        })

    # 5. Counterfactual Shock Attribution (if perturbed timeline exists)
    shock_attribution = None
    if perturbed_rollout is not None:
        p_t2m_c = perturbed_rollout[:, 0] - 273.15
        p_tp_mm = perturbed_rollout[:, 1]
        p_wind = np.sqrt(perturbed_rollout[:, 2] ** 2 + perturbed_rollout[:, 3] ** 2)

        temp_shift = float(np.mean(p_t2m_c - t2m_c))
        rain_shift_pct = float(
            (np.mean(p_tp_mm) - np.mean(tp_mm)) / (np.mean(tp_mm) + 1e-4) * 100.0
        )
        max_loc_shift = float(np.max(np.abs(p_tp_mm - tp_mm)))

        shock_attribution = {
            "mean_temp_shift_c": round(temp_shift, 3),
            "rain_anomaly_pct": round(rain_shift_pct, 2),
            "max_localized_rain_change_mm": round(max_loc_shift, 2),
            "multiverse_divergence": "Strong Non-Linear Propagation"
            if abs(rain_shift_pct) > 5.0
            else "Localized Coherent Perturbation",
        }

    # 6. Policy Co-Pilot Recommendations
    advisories: List[str] = []
    if high_flood_cells > 10:
        advisories.append(
            "ALERT: Substantial flood inundation risk (>150mm accumulated rainfall) detected in coastal/riverine zones. "
            "Recommend pre-positioning NDRF response units and issuing dam discharge advisories."
        )
    if heatwave_days > 20:
        advisories.append(
            "HEAT ACTION PLAN: Severe heat stress (>40°C) projected across inland zones. "
            "Trigger urban cool roof initiatives and issue daytime labor advisories."
        )
    if crop_stress_index > 50.0:
        advisories.append(
            "AGRI-ADVISORY: Stalled monsoon conditions detected with extended dry spell (>5 days). "
            "Advise farmers in rainfed regions to delay Kharif sowing or activate canal irrigation reserves."
        )
    if not advisories:
        advisories.append(
            "NORMAL CONDITIONS: Weather parameters within seasonal climatological bounds. "
            "Standard agricultural and reservoir management protocols apply."
        )

    return {
        "max_daily_rain_mm": round(max_daily_rain, 1),
        "high_flood_cells_count": high_flood_cells,
        "crop_stress_index_pct": round(crop_stress_index, 1),
        "mean_wind_power_w_m2": round(mean_wind_power, 1),
        "regional_insights": regional_insights,
        "shock_attribution": shock_attribution,
        "policy_advisories": advisories,
    }
