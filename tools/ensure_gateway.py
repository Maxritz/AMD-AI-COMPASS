#!/usr/bin/env python3
"""
Idempotent -- starts the AI-COMPASS Python-native memory gateway if not running.
Safe to call before any session. No external dependencies.
"""

import subprocess
import sys
import time
from http.client import HTTPConnection
from pathlib import Path

GATEWAY_PORT = 8420
GATEWAY_ADDR = f"127.0.0.1:{GATEWAY_PORT}"
MAX_RETRIES = 20
_THIS_DIR = Path(__file__).resolve().parent


def is_gateway_alive():
    try:
        conn = HTTPConnection("127.0.0.1", GATEWAY_PORT, timeout=2)
        conn.request("GET", "/health")
        ok = conn.getresponse().status == 200
        conn.close()
        return ok
    except OSError:
        return False


def start_gateway():
    gateway_script = _THIS_DIR / "memory_gateway.py"
    if not gateway_script.exists():
        print(f"[FAIL] gateway script not found: {gateway_script}", file=sys.stderr)
        return False
    try:
        subprocess.Popen(
            [sys.executable, str(gateway_script)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
        print(f"[FAIL] could not start gateway: {exc}", file=sys.stderr)
        return False
    for _ in range(MAX_RETRIES):
        time.sleep(0.5)
        if is_gateway_alive():
            return True
    return False


def main():
    if is_gateway_alive():
        print(f"[OK] memory gateway already running at http://{GATEWAY_ADDR}")
        return 0

    print("Starting AI-COMPASS memory gateway...")
    if start_gateway():
        print(f"[OK] gateway live at http://{GATEWAY_ADDR}")
        return 0

    print(f"[WARN] gateway did not respond within {MAX_RETRIES * 0.5:.0f}s")
    print("  Start manually: python tools/memory_gateway.py")
    return 1


if __name__ == "__main__":
    sys.exit(main())
