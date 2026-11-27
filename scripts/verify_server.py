"""Verify serve.py using the official monsoon_r1.schema_validator."""

import sys
import threading
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import uvicorn
from serve import app
from monsoon_r1 import schema_validator


def run_server():
    config = uvicorn.Config(app, host="127.0.0.1", port=8000, log_level="warning")
    server = uvicorn.Server(config)
    server.run()


if __name__ == "__main__":
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(2.0)

    code = schema_validator.main(["http://127.0.0.1:8000"])
    print(f"Schema validator exit code: {code}")
    exit(code)
