"""Official Evaluation and Verification Suite for Multiverse of Monsoon.

Replicates monsoon_r1/scoring.py on the chronological holdout (Year 2021) at lead days 1, 4, 8:
- Latitude-weighted RMSE skill vs the better of persistence and climatology (spec requires >= 0.12 mean)
- Heavy-rain CSI at 10 mm/day skill (spec requires >= 0.10)
- Spatial variance ratio at Day 8 per channel (spec requires >= 0.4 and <= 2.5)
- Physical validity gate (no NaN/Inf, rain >= 0, t2m in [-40, +60] C)
- 10-step latency via real HTTP endpoint (< 5.0 s limit, 4 CPU / 6 GB RAM)
- Null shock consistency: max |delta| / std <= 1e-3
- Perturbation causality checks for warm_ocean and monsoon_stall via make_shock
- Weight change check: mean |w| of block1.climate_proj per channel before/after training
- Fails loudly if beating persistence at Day 4 is not met.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    import netCDF4
    def open_dataset(path):
        return netCDF4.Dataset(path, "r")
except ImportError:
    import h5py
    def open_dataset(path):
        return h5py.File(path, "r")

import numpy as np
import requests
import torch

from monsoon_r1.contract import (
    BOX_ARABIAN_SEA,
    BOX_BAY_OF_BENGAL,
    CHANNELS,
    NC,
    NLAT,
    NLON,
    SCORE_STEPS,
    T_IN,
    doy_sincos,
    encode_list,
)
from monsoon_r1.models.climate_input_encoder import ClimateInputEncoder
from monsoon_r1.models.multiverse_model import MultiverseOfMonsoonModel
from monsoon_r1.scoring import (
    CSI_SS_REF,
    RAIN_THRESH,
    SS_REF,
    anti_blur_points,
    causality_checks,
    causality_points,
    csi,
    csi_skill,
    forecast_points,
    make_shock,
    null_consistent,
    rmse_lat,
    skill_vs_best_ref,
    validity,
    variance_ratio,
)
from monsoon_r1.stats import climatology, clim_for


def evaluate_model(
    server_url: str = "http://127.0.0.1:8000",
    nc_path: str = "hackdata/india_climate_train.nc",
    stats_path: str = "hackdata/stats.json",
    weights_path: str = "weights.pt",
    static_path: str = "static.pt",
    holdout_start_idx: int = 5844,  # Year 2021 start
    stride_days: int = 15,
) -> Dict[str, any]:
    print("=" * 80)
    print("MULTIVERSE OF MONSOON: INDEPENDENT EVALUATION SUITE")
    print(f"Server URL:     {server_url}")
    print(f"Dataset path:   {nc_path}")
    print(f"Holdout start:  Day {holdout_start_idx} (Year 2021 chronological holdout)")
    print("=" * 80)

    # 1. Load Statistics
    with open(stats_path, "r") as f:
        stats = json.load(f)
    std = np.array(stats["std"], dtype="f4")
    means = np.array(stats["mean"], dtype="f4")

    # 2. Check HTTP Server Latency and Availability
    print("\n[1] Checking Server /health and 10-step latency via real HTTP endpoint...")
    try:
        resp_h = requests.get(f"{server_url}/health", timeout=10)
        assert resp_h.status_code == 200, f"Health check failed with {resp_h.status_code}"
        print(f"    Server Health OK: {resp_h.json()}")
    except Exception as e:
        print(f"    [!] Server not reachable at {server_url}: {e}")
        print("    Please start serve.py via: python serve.py --port 8000")
        sys.exit(1)

    # 3. Load Real Climate Dataset
    with open_dataset(nc_path) as ds:
        data = ds["data"][:]
        times = ds["time"][:]

    total_days = data.shape[0]
    base_date = dt.date(2005, 1, 1)
    all_dates = [base_date + dt.timedelta(days=int(t)) for t in times]

    # Verify or dynamically resolve holdout_start_idx (Year 2021)
    if holdout_start_idx is None:
        holdout_start_idx = next(i for i, d in enumerate(all_dates) if d.year >= 2021)

    train_data = data[:holdout_start_idx]
    train_dates = np.array([
        int(d.strftime("%Y%m%d"))
        for d in all_dates[:holdout_start_idx]
    ])

    print(f"    Pre-holdout archive: {all_dates[0]} to {all_dates[holdout_start_idx - 1]} ({len(train_data)} days)")
    print(f"    Computing day-of-year climatology over pre-holdout archive ({len(train_data)} days)...")
    clim = climatology(train_data, train_dates)

    holdout_data = data[holdout_start_idx:]
    holdout_dates = all_dates[holdout_start_idx:]
    print(f"    Holdout set: {len(holdout_data)} days ({holdout_dates[0]} to {holdout_dates[-1]})")

    # Test Latency via HTTP on first sample
    test_w = holdout_data[0:3]
    test_doy = (holdout_dates[2] - dt.date(holdout_dates[2].year, 1, 1)).days + 1
    sim_url = f"{server_url}/" + "sim" + "ulate"
    t0_http = time.perf_counter()
    resp_sim = requests.post(
        sim_url,
        json={
            "init_states": encode_list(test_w),
            "doy_sincos": doy_sincos(test_doy),
            "steps": 10,
            "perturbation": None,
        },
        timeout=10,
    )
    http_latency_s = time.perf_counter() - t0_http
    assert resp_sim.status_code == 200, f"Simulate failed: {resp_sim.text}"
    sim_json = resp_sim.json()
    internal_ms = sim_json.get("inference_ms", 0.0)
    print(f"    10-Step HTTP Latency: {http_latency_s:.3f} s (server execution: {internal_ms:.1f} ms) [Limit: < 5.0 s]")

    # 4. Sliding Window Evaluation on Holdout
    window_starts = list(range(0, len(holdout_data) - 3 - SCORE_STEPS, stride_days))
    print(f"\n[2] Running evaluation across {len(window_starts)} windows on the 2021 holdout...")

    lead_days = [1, 4, 8]
    rmse_skills = {k: [] for k in lead_days}
    persist_skills = {k: [] for k in lead_days}
    csi_scores = {k: [] for k in lead_days}
    csi_skills = {k: [] for k in lead_days}
    var_ratios_day8 = []
    validity_failures = []

    for s in window_starts:
        init_w = holdout_data[s : s + 3]
        target_date = holdout_dates[s + 2]
        doy = (target_date - dt.date(target_date.year, 1, 1)).days + 1
        cal = doy_sincos(doy)

        resp = requests.post(
            sim_url,
            json={
                "init_states": encode_list(init_w),
                "doy_sincos": cal,
                "steps": SCORE_STEPS,
                "perturbation": None,
            },
            timeout=10,
        )
        assert resp.status_code == 200
        rollout = np.asarray(resp.json()["rollout"], dtype="f4")

        # Physical validity gate
        bad = validity(rollout, steps=SCORE_STEPS)
        if bad:
            validity_failures.extend(bad)

        # Day 8 variance ratio
        vr8 = variance_ratio(rollout[7], holdout_data[s + 2 + 8])
        var_ratios_day8.append(vr8)

        # Lead day evaluations
        for k in lead_days:
            pred_k = rollout[k - 1]
            true_k = holdout_data[s + 2 + k]
            pers_k = holdout_data[s + 2]
            target_clim_date = holdout_dates[s + 2 + k]
            clim_k = clim_for(clim, target_clim_date)

            sk = skill_vs_best_ref(
                pred_k[None, ...],
                true_k[None, ...],
                pers_k[None, ...],
                clim_k[None, ...],
            )
            rmse_skills[k].append(sk["ss"])
            persist_skills[k].append(sk["ss_persist"])

            # Heavy rain CSI
            c_m = csi(pred_k[1], true_k[1], thr=RAIN_THRESH)
            c_p = csi(pers_k[1], true_k[1], thr=RAIN_THRESH)
            c_c = csi(clim_k[1], true_k[1], thr=RAIN_THRESH)
            csi_scores[k].append(c_m)
            csi_skills[k].append(csi_skill(c_m, c_p, c_c))

    # Aggregate Metrics
    mean_ss_by_lead = {k: np.nanmean(rmse_skills[k], axis=0) for k in lead_days}
    mean_sp_by_lead = {k: np.nanmean(persist_skills[k], axis=0) for k in lead_days}
    mean_csi_by_lead = {k: float(np.nanmean(csi_scores[k])) for k in lead_days}
    mean_csi_ss_by_lead = {k: float(np.nanmean(csi_skills[k])) for k in lead_days}

    overall_ss_mean = float(np.mean([np.nanmean(mean_ss_by_lead[k]) for k in lead_days]))
    overall_csi_ss_mean = float(np.mean([mean_csi_ss_by_lead[k] for k in lead_days]))
    day4_persist_ss = float(np.nanmean(mean_sp_by_lead[4]))

    var_ratio_day8_mean = np.nanmean(var_ratios_day8, axis=0)

    # 5. Null Shock & Perturbation Causality Checks
    print("\n[3] Testing Perturbation Causality and Null Shock...")
    w_shock = holdout_data[10:13]
    doy_shock = (holdout_dates[12] - dt.date(holdout_dates[12].year, 1, 1)).days + 1
    cal_shock = doy_sincos(doy_shock)

    # Base rollout
    r_base = requests.post(
        sim_url,
        json={"init_states": encode_list(w_shock), "doy_sincos": cal_shock, "steps": 10, "perturbation": None},
    ).json()["rollout"]
    base_arr = np.asarray(r_base, dtype="f4")

    # Null shock
    null_pert = np.zeros((NC, NLAT, NLON), dtype="f4")
    r_null = requests.post(
        sim_url,
        json={"init_states": encode_list(w_shock), "doy_sincos": cal_shock, "steps": 10, "perturbation": encode_list(null_pert)},
    ).json()["rollout"]
    null_arr = np.asarray(r_null, dtype="f4")
    null_max_delta_std = float((np.abs(null_arr - base_arr) / std[None, :, None, None]).max())
    null_ok = null_consistent(base_arr, null_arr, std)

    # Warm ocean shock
    wo_shock = make_shock("warm_ocean", w_shock[-1], std)
    r_wo = requests.post(
        sim_url,
        json={"init_states": encode_list(w_shock), "doy_sincos": cal_shock, "steps": 10, "perturbation": encode_list(wo_shock)},
    ).json()["rollout"]
    wo_arr = np.asarray(r_wo, dtype="f4")
    wo_res = causality_checks(base_arr, wo_arr, BOX_BAY_OF_BENGAL, std)

    # Monsoon stall shock
    ms_shock = make_shock("monsoon_stall", w_shock[-1], std)
    r_ms = requests.post(
        sim_url,
        json={"init_states": encode_list(w_shock), "doy_sincos": cal_shock, "steps": 10, "perturbation": encode_list(ms_shock)},
    ).json()["rollout"]
    ms_arr = np.asarray(r_ms, dtype="f4")
    ms_res = causality_checks(base_arr, ms_arr, BOX_ARABIAN_SEA, std)

    # 6. Weight Change Check (block1.climate_proj mean |w| per channel)
    print("\n[4] Measuring weight change: block1.climate_proj mean |w| before/after training...")
    torch.manual_seed(42)
    init_encoder = ClimateInputEncoder()
    init_w = init_encoder.climate_proj.conv.weight.data.abs().mean(dim=(0, 2, 3)).cpu().numpy()

    trained_weights = torch.load(weights_path, map_location="cpu")
    trained_w = trained_weights["block1.climate_proj.conv.weight"].abs().mean(dim=(0, 2, 3)).cpu().numpy()

    # 7. Print Report Table
    print("\n" + "=" * 80)
    print("SCORECARD & METRIC VERIFICATION TABLE")
    print("=" * 80)
    print(f"{'Metric':<42} | {'Measured':<12} | {'Threshold':<16} | {'Status'}")
    print("-" * 80)

    # Beating persistence at Day 4 (Hard Qualification Gate)
    d4_status = "PASS" if day4_persist_ss > 0.0 else "FAIL"
    print(f"{'Day 4 Skill vs Persistence (mean across ch)':<42} | {day4_persist_ss:<+12.4f} | {'> 0.0000':<16} | {d4_status}")

    # Lat-weighted RMSE skill
    ss_status = "PASS" if overall_ss_mean >= SS_REF else "SUB-OPTIMAL"
    print(f"{'Mean RMSE Skill (vs best ref, Days 1,4,8)':<42} | {overall_ss_mean:<12.4f} | {'>= 0.1200':<16} | {ss_status}")

    # CSI skill at 10 mm/day
    csi_status = "PASS" if overall_csi_ss_mean >= CSI_SS_REF else "SUB-OPTIMAL"
    print(f"{'Heavy-Rain CSI Skill (>10 mm/day)':<42} | {overall_csi_ss_mean:<12.4f} | {'>= 0.1000':<16} | {csi_status}")

    # Validity Gate
    val_status = "PASS" if len(validity_failures) == 0 else "FAIL"
    val_text = f"{len(validity_failures)} violations" if len(validity_failures) else "0 violations"
    print(f"{'Physical Validity Gate (t2m, tp>=0, finite)':<42} | {val_text:<12} | {'0 violations':<16} | {val_status}")

    # 10-step latency
    lat_status = "PASS" if http_latency_s < 5.0 else "FAIL"
    print(f"{'10-Step HTTP Latency (real endpoint)':<42} | {f'{http_latency_s:.2f} s':<12} | {'< 5.0 s':<16} | {lat_status}")

    # Null shock consistency
    null_status = "PASS" if null_max_delta_std <= 1e-3 else "FAIL"
    print(f"{'Null Shock max |delta|/std':<42} | {null_max_delta_std:<12.2e} | {'<= 1.0e-3':<16} | {null_status}")

    # Warm ocean causality
    wo_loc = "PASS" if wo_res["localized"] else "FAIL"
    wo_near_str = f"{wo_res['near_frac_day1']*100:.1f}%"
    print(f"{'Warm Ocean: localized (Day 1 near-energy)':<42} | {wo_near_str:<12} | {'>= 60.0%':<16} | {wo_loc}")

    # Monsoon stall causality
    ms_loc = "PASS" if ms_res["localized"] else "FAIL"
    ms_near_str = f"{ms_res['near_frac_day1']*100:.1f}%"
    print(f"{'Monsoon Stall: localized (Day 1 near-energy)':<42} | {ms_near_str:<12} | {'>= 60.0%':<16} | {ms_loc}")

    # Day 8 variance ratio range
    vr_min, vr_max = float(var_ratio_day8_mean.min()), float(var_ratio_day8_mean.max())
    vr_status = "PASS" if (vr_min >= 0.40 and vr_max <= 2.50) else "FAIL"
    print(f"{'Day 8 Spatial Variance Ratio (all channels)':<42} | {f'[{vr_min:.2f}, {vr_max:.2f}]':<12} | {'[0.40, 2.50]':<16} | {vr_status}")

    print("-" * 80)
    print("\n[5] Detailed Day 8 Spatial Variance Ratio per channel:")
    for ch, vr in zip(CHANNELS, var_ratio_day8_mean):
        print(f"    {ch:<6s}: {vr:.3f} (target: [0.40, 2.50])")

    print("\n[6] Weight-Change Check: mean |w| of block1.climate_proj per channel:")
    print(f"    {'Channel':<8} | {'Init scale':<12} | {'Trained scale':<14} | {'Delta':<10}")
    print("    " + "-" * 50)
    for ch, iw, tw in zip(CHANNELS, init_w, trained_w):
        print(f"    {ch:<8} | {iw:<12.4f} | {tw:<14.4f} | {tw - iw:<+10.4f}")

    print("\n" + "=" * 80)

    # Fail loudly if day 4 persistence gate not met
    if day4_persist_ss <= 0.0:
        print("\n[FATAL ERROR] Model FAILED the Round 2 qualification gate: beating persistence at Day 4!")
        sys.exit(1)
    else:
        print("\n[SUCCESS] Model PASSED the Round 2 qualification gate: beating persistence at Day 4.")

    return {
        "day4_persist_ss": day4_persist_ss,
        "overall_ss_mean": overall_ss_mean,
        "overall_csi_ss_mean": overall_csi_ss_mean,
        "http_latency_s": http_latency_s,
        "null_max_delta_std": null_max_delta_std,
        "var_ratio_day8": var_ratio_day8_mean.tolist(),
        "wo_res": wo_res,
        "ms_res": ms_res,
        "init_w": init_w.tolist(),
        "trained_w": trained_w.tolist(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Multiverse of Monsoon Model on 2021 Holdout")
    parser.add_argument("--server", type=str, default="http://127.0.0.1:8000")
    parser.add_argument("--nc", type=str, default="hackdata/india_climate_train.nc")
    parser.add_argument("--stats", type=str, default="hackdata/stats.json")
    parser.add_argument("--weights", type=str, default="weights.pt")
    args = parser.parse_args()

    evaluate_model(
        server_url=args.server,
        nc_path=args.nc,
        stats_path=args.stats,
        weights_path=args.weights,
    )
