#!/usr/bin/env python3
"""
One-time bootstrap: initializes and starts the AI-COMPASS memory gateway.
Python-native, zero external dependencies. Runs as a background daemon.
"""

import subprocess
import sys
import time
from http.client import HTTPConnection
from pathlib import Path

GATEWAY_PORT = 8420
GATEWAY_ADDR = f"127.0.0.1:{GATEWAY_PORT}"
MAX_RETRIES = 30
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


def main():
    print("=" * 60)
    print(" AI-COMPASS  Memory Gateway Bootstrap")
    print("=" * 60)

    if is_gateway_alive():
        print(f"[OK] gateway already running at http://{GATEWAY_ADDR}")
        print("  Data: ~/.ai-compass/memory/memory.db")
        return 0

    gateway_script = _THIS_DIR / "memory_gateway.py"
    if not gateway_script.exists():
        print(f"[FAIL] gateway script not found: {gateway_script}")
        return 1

    print("Starting memory gateway (Python-native, no external deps)...")
    try:
        subprocess.Popen(
            [sys.executable, str(gateway_script), "--daemon"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
        print(f"[FAIL] could not start gateway daemon: {exc}")
        print(f"  Start manually: python {gateway_script}")
        return 1

    for i in range(MAX_RETRIES):
        time.sleep(0.5)
        if is_gateway_alive():
            print(f"[OK] gateway daemon started on http://{GATEWAY_ADDR}")
            print("  Storage:  ~/.ai-compass/memory/memory.db")
            print()
            print("Ready. Try: python aicompass.py memory status")
            return 0
        if i % 8 == 7:
            print(f"  waiting ({int((i + 1) * 0.5)}s)...")

    print(f"[FAIL] gateway did not respond on http://{GATEWAY_ADDR} within {MAX_RETRIES * 0.5:.0f}s")
    print(f"  Start manually: python tools/memory_gateway.py")
    return 1


if __name__ == "__main__":
    sys.exit(main())
