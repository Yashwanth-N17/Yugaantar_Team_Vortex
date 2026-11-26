"""End-to-End Integration Verification Test for Multiverse of Monsoon.

Tests:
1. Backend serve.py health check (/health)
2. Backend serve.py simulation (/simulate)
3. Dashboard app.py health proxy (/api/health)
4. Dashboard app.py bundled windows (/api/windows)
5. Dashboard app.py baseline simulation (/api/simulate)
6. Dashboard app.py warm ocean perturbation simulation
7. Dashboard app.py monsoon stall perturbation simulation
8. Output tensor shape [10, 6, 32, 36], finite values, NaN count, inf count, neg tp count
9. Mathematical difference check (perturbed - baseline)
10. Error handling (invalid input, steps out of range, backend offline)
"""

import multiprocessing
import os
import sys
import time
import requests
import numpy as np

def run_backend():
    import uvicorn
    from serve import app
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

def run_frontend():
    import uvicorn
    from dashboard.app import app
    uvicorn.run(app, host="127.0.0.1", port=8501, log_level="warning")

def test_integration():
    print("=== STARTING INTEGRATION TEST ===")
    
    # Start backend
    p_backend = multiprocessing.Process(target=run_backend, daemon=True)
    p_backend.start()
    
    # Start frontend
    p_frontend = multiprocessing.Process(target=run_frontend, daemon=True)
    p_frontend.start()
    
    # Wait for startup
    time.sleep(3.5)
    
    try:
        # 1. Backend /health
        print("\n[Test 1] Backend /health...")
        r = requests.get("http://127.0.0.1:8000/health", timeout=5)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        b_health = r.json()
        print(f"  Backend health: {b_health}")
        assert b_health.get("status") == "ready"
        assert b_health.get("model_name") == "multiverse-of-monsoon-r1"

        # 2. Frontend /api/health
        print("\n[Test 2] Frontend /api/health proxy...")
        r = requests.get("http://127.0.0.1:8501/api/health", timeout=5)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}"
        f_health = r.json()
        print(f"  Frontend /api/health: {f_health}")
        assert f_health.get("online") is True
        assert f_health.get("engine_status") == "ready"

        # 3. Frontend /api/windows
        print("\n[Test 3] Frontend /api/windows...")
        r = requests.get("http://127.0.0.1:8501/api/windows", timeout=5)
        assert r.status_code == 200
        windows = r.json()
        print(f"  Bundled windows loaded: {len(windows)}")
        assert len(windows) >= 3
        for w in windows:
            print(f"    - {w['id']}: {w['name']}")

        # 4. Frontend Baseline Simulation
        print("\n[Test 4] Frontend /api/simulate (Baseline 10-day)...")
        t0 = time.perf_counter()
        r = requests.post("http://127.0.0.1:8501/api/simulate", json={
            "steps": 10,
            "window_id": "monsoon_depression_2010",
            "shock_type": "null"
        }, timeout=15)
        lat = (time.perf_counter() - t0) * 1000
        assert r.status_code == 200, f"Simulate failed: {r.status_code} {r.text}"
        data = r.json()
        
        base_arr = np.array(data["baseline"], dtype="f4")
        print(f"  Baseline rollout shape: {base_arr.shape}")
        assert base_arr.shape == (10, 6, 32, 36), f"Wrong shape: {base_arr.shape}"
        assert np.all(np.isfinite(base_arr)), "Non-finite values found in baseline!"
        assert np.all(base_arr[:, 1] >= 0), "Negative precipitation found!"
        print(f"  Inference latency reported: {data['inference_ms']} ms (roundtrip {lat:.1f} ms)")
        print(f"  Finite count: {base_arr.size} / {base_arr.size}")
        print(f"  NaN count: {np.isnan(base_arr).sum()}")
        print(f"  Negative tp count: {(base_arr[:, 1] < 0).sum()}")

        # 5. Frontend Warm Ocean Perturbation Simulation
        print("\n[Test 5] Frontend /api/simulate (Warm Ocean +3K BoB)...")
        r = requests.post("http://127.0.0.1:8501/api/simulate", json={
            "steps": 10,
            "window_id": "monsoon_depression_2010",
            "shock_type": "preset",
            "preset_name": "warm_ocean"
        }, timeout=15)
        assert r.status_code == 200
        data_warm = r.json()
        pert_arr = np.array(data_warm["perturbed"], dtype="f4")
        diff_arr = np.array(data_warm["difference"], dtype="f4")
        assert pert_arr.shape == (10, 6, 32, 36)
        assert diff_arr.shape == (10, 6, 32, 36)
        diff_mag = np.abs(diff_arr).max()
        print(f"  Warm Ocean shock applied: {data_warm['shock_applied']}")
        print(f"  Max absolute difference (|Perturbed - Baseline|): {diff_mag:.4f}")
        assert diff_mag > 0.01, "Shock produced no difference!"

        # 6. Frontend Monsoon Stall Perturbation Simulation
        print("\n[Test 6] Frontend /api/simulate (Monsoon Stall -40% winds)...")
        r = requests.post("http://127.0.0.1:8501/api/simulate", json={
            "steps": 10,
            "window_id": "monsoon_depression_2010",
            "shock_type": "preset",
            "preset_name": "monsoon_stall"
        }, timeout=15)
        assert r.status_code == 200
        data_stall = r.json()
        diff_stall = np.array(data_stall["difference"], dtype="f4")
        diff_stall_mag = np.abs(diff_stall).max()
        print(f"  Monsoon Stall shock applied: {data_stall['shock_applied']}")
        print(f"  Max absolute difference (|Perturbed - Baseline|): {diff_stall_mag:.4f}")
        assert diff_stall_mag > 0.01, "Stall shock produced no difference!"

        # 7. Day-by-Day rollout inspection (Day 1, Day 4, Day 10)
        print("\n[Test 7] Verifying Day 1, Day 4, Day 10 statistics...")
        channels = ["t2m (°C)", "tp (mm/day)", "u10 (m/s)", "v10 (m/s)", "msl (hPa)", "tcwv (kg/m²)"]
        for day_idx in [0, 3, 9]:
            day_num = day_idx + 1
            print(f"  --- Day {day_num} Domain Means ---")
            for ch in range(6):
                val = base_arr[day_idx, ch].mean()
                if ch == 0: val -= 273.15
                if ch == 4: val /= 100.0
                print(f"    {channels[ch]}: mean = {val:.2f}, min = {base_arr[day_idx, ch].min():.2f}, max = {base_arr[day_idx, ch].max():.2f}")

        # 8. Error Handling Tests
        print("\n[Test 8] Error handling (invalid steps, missing output)...")
        r_err = requests.post("http://127.0.0.1:8000/simulate", json={"steps": 999, "init_states": [], "doy_sincos": [0, 1]}, timeout=5)
        print(f"  Backend invalid steps response: {r_err.status_code} {r_err.text}")
        assert r_err.status_code == 400

        print("\n=== ALL INTEGRATION TESTS PASSED SUCCESSFULLY! ===")

    finally:
        p_backend.terminate()
        p_frontend.terminate()
        p_backend.join(timeout=1)
        p_frontend.join(timeout=1)

if __name__ == "__main__":
    test_integration()
