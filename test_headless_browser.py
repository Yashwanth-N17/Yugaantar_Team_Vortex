"""Headless Chrome Browser Test for Multiverse of Monsoon Dashboard.

Starts backend (serve.py :8000) and frontend (dashboard/app.py :8501),
then invokes headless Google Chrome to load http://127.0.0.1:8501,
execute the JavaScript simulation rollout, dump the live rendered DOM,
and capture a screenshot.
"""

import multiprocessing
import os
import subprocess
import time

def run_backend():
    import uvicorn
    from serve import app
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")

def run_frontend():
    import uvicorn
    from dashboard.app import app
    uvicorn.run(app, host="127.0.0.1", port=8501, log_level="warning")

def test_browser():
    print("[*] Starting backend and dashboard servers...")
    p_b = multiprocessing.Process(target=run_backend, daemon=True)
    p_b.start()
    p_f = multiprocessing.Process(target=run_frontend, daemon=True)
    p_f.start()

    time.sleep(4.0)

    chrome_path = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
    if not os.path.exists(chrome_path):
        chrome_path = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"

    print(f"[*] Found browser binary: {chrome_path}")
    screenshot_path = os.path.abspath("dashboard_screenshot.png")

    import tempfile
    user_data_dir = tempfile.mkdtemp(prefix="chrome_test_")

    cmd = [
        chrome_path,
        "--headless=new",
        "--no-sandbox",
        "--disable-gpu",
        "--disable-software-rasterizer",
        "--disable-dev-shm-usage",
        f"--user-data-dir={user_data_dir}",
        "--window-size=1400,900",
        f"--screenshot={screenshot_path}",
        "--dump-dom",
        "--virtual-time-budget=6000",
        "http://127.0.0.1:8501/"
    ]

    print("[*] Executing Headless Browser against http://127.0.0.1:8501/...")
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=25)
    
    dom_output = res.stdout
    print(f"[*] Headless Chrome finished with return code {res.returncode}")
    print(f"[*] Captured DOM length: {len(dom_output)} characters")
    
    # Check key live elements in rendered DOM
    checks = [
        ("Multiverse of Monsoon — Synoptic Climate Twin", "Page Title"),
        ("Simulation Control Panel", "Control Panel"),
        ("Model Verification Panel", "Verification Panel"),
        ("multiverse-of-monsoon-r1", "Model ID"),
        ("Connected", "Backend Status"),
        ("Grid Statistics", "Statistics Panel"),
        ("Rainfall Analytics", "Rainfall Panel"),
        ("Forecast Evolution", "Evolution Chart"),
        ("Timeline Prime (Baseline)", "Baseline Map Card"),
        ("Multiverse Branch (Perturbed)", "Branch Map Card"),
        ("Difference (Δ = Branch − Prime)", "Diff Map Card"),
    ]

    passed = 0
    for token, label in checks:
        if token in dom_output:
            print(f"  [PASS] {label}: Found '{token}'")
            passed += 1
        else:
            print(f"  [FAIL] {label}: Did not find '{token}'")

    if os.path.exists(screenshot_path):
        print(f"[*] Visual screenshot successfully captured: {screenshot_path} ({os.path.getsize(screenshot_path)} bytes)")

    import re
    print("\n--- DOM LIVE RENDERED VALUES IN CHROME BROWSER ---")
    for sec_id in ["verif-health", "verif-model", "verif-horizon", "verif-dims", "verif-finite", "verif-nan", "verif-negtp", "stat-prime-mean", "stat-prime-max", "rain-day-mean", "rain-10d-max", "rain-dist-trace"]:
        m = re.search(rf'id="{sec_id}"[^>]*>(.*?)<', dom_output)
        val = m.group(1).strip() if m else "NOT FOUND"
        print(f"  {sec_id}: {val}")

    # Terminate servers
    p_b.terminate()
    p_f.terminate()
    p_b.join(timeout=1)
    p_f.join(timeout=1)

    print(f"\n[*] Browser Verification: {passed}/{len(checks)} DOM checks passed.")
    assert passed == len(checks), "Some DOM checks failed!"

if __name__ == "__main__":
    test_browser()
