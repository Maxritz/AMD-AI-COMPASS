#!/usr/bin/env python3
"""
AI-COMPASS perf-analysis launcher.

One-shot: run llama-bench trace -> analyze -> (optional compare) -> memory.

Usage:
    python analyze_run.py --model model.gguf [--arch rdna4] [--compare baseline.csv]
                          [--no-bench] [--trace path.csv] [--no-memory]
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TOOLS = ROOT / "tools"
RUNS = ROOT / "benchmark_runs"


def _run(tool: str, *args: str) -> int:
    """Run a tools/ python module with args, capturing nothing."""
    return subprocess.call([sys.executable, str(TOOLS / tool), *args])


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", help="GGUF model path (benchmark only)")
    p.add_argument("--arch", default="rdna4",
                   choices=["rdna1", "rdna2", "rdna3", "rdna3_5", "rdna4"])
    p.add_argument("--compare", help="baseline trace CSV for before/after")
    p.add_argument("--trace", help="existing trace CSV (skip benchmark)")
    p.add_argument("--pp", type=int, default=256)
    p.add_argument("--tg", type=int, default=64)
    p.add_argument("--no-bench", action="store_true", help="skip benchmark step")
    p.add_argument("--no-memory", action="store_true", help="skip memory publish")
    args = p.parse_args()

    if not args.trace:
        if args.no_bench or not args.model:
            print("[FAIL] need --model (bench) or --trace (existing)")
            return 1
        RUNS.mkdir(exist_ok=True)
        rc = _run("run_benchmark.py", "--model", args.model,
                  "--pp", str(args.pp), "--tg", str(args.tg),
                  "--arch", args.arch, "--output", str(RUNS))
        if rc:
            return rc
        trace = str(RUNS / "hip_trace.csv")
    else:
        trace = args.trace

    # Analyze trace -> report/analysis.json (+ HTML)
    rc = _run("analyze.py", trace, "--arch", args.arch,
              "--memory" if not args.no_memory else "--no-memory")
    if rc:
        return rc

    if args.compare:
        rc = _run("analyze.py", trace, "--compare", args.compare,
                  "--arch", args.arch)
        if rc:
            return rc

    print("\n[OK] done. Report in analysis_output/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
