#!/usr/bin/env python3
"""
AI-COMPASS Benchmark Runner — end-to-end performance test

Records HIP kernel trace + GPU metrics for a model, then generates
a comprehensive performance report with optimization recommendations.

Usage:
    python tools/run_benchmark.py --model path/to/model.gguf [--pp 256] [--tg 64]
"""
import subprocess
import sys
import os
import argparse
import tempfile
import shutil
from datetime import datetime

AI_COMPASS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_llama_bench():
    """Find llama-bench or llama-cli in common locations."""
    candidates = [
        os.path.join(AI_COMPASS_ROOT, "build", "llama-bench.exe"),
        os.path.join(AI_COMPASS_ROOT, "..", "llama.cpp-ROCM-Test", "build-hip", "bin", "llama-bench.exe"),
        os.path.join(AI_COMPASS_ROOT, "..", "llama.cpp-ROCM-Test", "build-hip", "bin", "Release", "llama-bench.exe"),
        os.path.join(os.path.expanduser("~"), "Desktop", "llama", "llama.cpp-ROCM-Test", "build-hip", "bin", "llama-bench.exe"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None


def main():
    parser = argparse.ArgumentParser(description="AI-COMPASS Benchmark Runner")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--pp", type=int, default=256, help="Prompt processing tokens")
    parser.add_argument("--tg", type=int, default=64, help="Token generation count")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--compare", help="Baseline trace CSV for comparison")
    parser.add_argument("--llama-bench", help="Path to llama-bench executable")
    args = parser.parse_args()

    # Find llama-bench
    lb = args.llama_bench or find_llama_bench()
    if not lb:
        print("❌ llama-bench not found. Build it first:")
        print("   cd llma.cpp-ROCM-Test && mkdir build-hip && cd build-hip")
        print("   cmake .. -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release")
        print("   cmake --build . --config Release --target llama-bench")
        return 1

    if not os.path.exists(args.model):
        print(f"❌ Model not found: {args.model}")
        return 1

    # Setup output dir
    out_dir = args.output or os.path.join(AI_COMPASS_ROOT, "benchmark_runs",
                                          f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    trace_csv = os.path.join(out_dir, "hip_trace.csv")
    report_dir = os.path.join(out_dir, "report")
    os.makedirs(report_dir, exist_ok=True)

    print("=" * 60)
    print("🧭 AI-COMPASS Benchmark Runner")
    print("=" * 60)
    print(f"Model:    {args.model}")
    print(f"PP:       {args.pp}")
    print(f"TG:       {args.tg}")
    print(f"Output:   {out_dir}")
    print(f"llama-bench: {lb}")
    print()

    # Step 1: Run llama-bench with HIP tracer
    print("📡 Launching with HIP tracer...")
    env = os.environ.copy()
    env["HIP_TRACER_OUTPUT"] = trace_csv
    env["HIP_TRACER_DLL"] = os.path.join(AI_COMPASS_ROOT, "build", "ai_hip_tracer.dll")

    cmd = [lb, "-m", args.model, "-p", str(args.pp), "-n", str(args.tg), "-ngl", "99"]
    print(f"   {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr[-1000:]}")
    except subprocess.TimeoutExpired:
        print("⚠ Benchmark timed out (10 min)")
        return 1
    except FileNotFoundError:
        print(f"❌ Failed to launch {lb}")
        return 1

    # Step 2: Analyze the trace
    print(f"\n📊 Analyzing trace: {trace_csv}")
    if os.path.exists(trace_csv):
        from analyze import parse_trace, analyze_trace, generate_html_report

        records = parse_trace(trace_csv)
        if records:
            analysis = analyze_trace(records, cu_count=32)

            # Print summary
            s = analysis["summary"]
            print(f"   Kernels: {s['total_kernels']}  Time: {s['total_time_ms']} ms")
            print(f"   Avg: {s['avg_kernel_us']} µs  GPU busy: {s['estimated_gpu_busy_pct']}%")

            # Category breakdown
            print(f"\n   Category breakdown:")
            for cat, st in sorted(analysis["category_breakdown"].items(), key=lambda x: -x[1]["pct"]):
                print(f"     {cat:20s} {st['count']:5d}  {st['total_ms']:8.1f}ms  {st['pct']:5.1f}%  occ:{st['avg_occupancy_pct']:.0f}%")

            # Bottlenecks
            if analysis["bottlenecks"]:
                print(f"\n   🚨 Bottlenecks:")
                for b in analysis["bottlenecks"]:
                    print(f"      ⚠ {b}")

            # Optimization targets
            if analysis["optimization_targets"]:
                print(f"\n   🔧 Optimization Targets:")
                for t in analysis["optimization_targets"]:
                    print(f"      [{t['target']}] {t['suggestion']}")

            # Write JSON report
            import json
            with open(os.path.join(report_dir, "analysis.json"), "w") as f:
                json.dump(analysis, f, indent=2)

            # Write HTML report
            html_path = os.path.join(report_dir, "report.html")
            generate_html_report(analysis, html_path)
            print(f"\n   📄 Report: {html_path}")

            # Compare mode
            if args.compare and os.path.exists(args.compare):
                print(f"\n   🔄 Comparing vs baseline: {args.compare}")
                from analyze import compare_traces
                diff = compare_traces(args.compare, trace_csv)
                with open(os.path.join(report_dir, "comparison.json"), "w") as f:
                    json.dump(diff, f, indent=2)
                if diff["improvements"]:
                    print(f"   ✅ Improvements:")
                    for k, v in diff["improvements"].items():
                        print(f"      {k}: {v}")
                if diff["regressions"]:
                    print(f"   ❌ Regressions:")
                    for k, v in diff["regressions"].items():
                        print(f"      {k}: {v}")
        else:
            print("⚠ No kernel records found in trace")
    else:
        print(f"⚠ Trace CSV not found: {trace_csv}")

    print(f"\n✅ Done. Results in: {out_dir}")
    print(f"   Report: {os.path.join(report_dir, 'report.html')}")
    print(f"   Trace:  {trace_csv}")
    print(f"   To view in RCV: rocprof-compute-viewer.exe {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
