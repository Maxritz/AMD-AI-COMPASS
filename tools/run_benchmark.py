#!/usr/bin/env python3
"""
AI-COMPASS Benchmark Runner  end-to-end performance test

Records HIP kernel trace + GPU metrics for a model, then generates
a comprehensive performance report with optimization recommendations.

Usage:
    python tools/run_benchmark.py --model path/to/model.gguf [--pp 256] [--tg 64]
    python tools/run_benchmark.py --model path/to/model.gguf --arch rdna2
"""
import subprocess
import sys
import os
import argparse
import tempfile
import shutil
from datetime import datetime

# Import newly integrated core scripts
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect
import validate
import cpu_tune

AI_COMPASS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_rocprofv3():
    """Find rocprofv3 for Linux ROCm tracing (hip_tracer is Windows-only)."""
    import shutil
    return shutil.which("rocprofv3")


def convert_rocprof_csv_to_aicompass(rocpd_csv_path, output_csv_path):
    """Convert a rocprofv3 kernel_trace.csv to AI-COMPASS parse_trace-compatible format.

    rocprofv3 --sys-trace -f csv produces kernel_trace.csv with columns:
      Kind, Agent_Id, Queue_Id, Stream_Id, Thread_Id, Dispatch_Id, Kernel_Id,
      Kernel_Name, Correlation_Id, Start_Timestamp, End_Timestamp,
      LDS_Block_Size, Scratch_Size, VGPR_Count, Accum_VGPR_Count, SGGR_Count,
      Workgroup_Size_X/Y/Z, Grid_Size_X/Y/Z

    AI-COMPASS parse_trace expects:
      dispatch_id, kernel_name, grid_x, grid_y, grid_z, block_x, block_y,
      block_z, shared_mem, duration_us
    """
    import csv

    with open(rocpd_csv_path, newline="") as fin:
        reader = csv.DictReader(fin)
        rows = list(reader)

    with open(output_csv_path, "w", newline="") as fout:
        writer = csv.writer(fout)
        writer.writerow([
            "dispatch_id", "kernel_name", "grid_x", "grid_y", "grid_z",
            "block_x", "block_y", "block_z", "shared_mem", "duration_us",
        ])
        for row in rows:
            if row.get("Kind", "") != "KERNEL_DISPATCH":
                continue
            start = int(row["Start_Timestamp"])
            end = int(row["End_Timestamp"])
            duration_us = (end - start) / 1000.0  # timestamps are in ns
            writer.writerow([
                row.get("Dispatch_Id", ""),
                row.get("Kernel_Name", ""),
                row.get("Grid_Size_X", ""),
                row.get("Grid_Size_Y", ""),
                row.get("Grid_Size_Z", ""),
                row.get("Workgroup_Size_X", ""),
                row.get("Workgroup_Size_Y", ""),
                row.get("Workgroup_Size_Z", ""),
                row.get("LDS_Block_Size", ""),
                f"{duration_us:.3f}",
            ])
    return True


def find_llama_bench():
    """
    Find llama-bench using only generic, portable discovery:
      1. Check PATH (works on any OS if user has it installed)
      2. Check AI-COMPASS build directory
      3. Return None — user must pass --llama-bench explicitly
    Never hardcodes project names, drive letters, or user home paths.
    """
    import shutil

    # 1. Check if llama-bench is on PATH (cross-platform)
    found = shutil.which("llama-bench") or shutil.which("llama-bench.exe")
    if found:
        return found

    # 2. Check the AI-COMPASS own build directory (if user built it here)
    for subdir in ["build", os.path.join("build", "Release"), os.path.join("build", "bin")]:
        candidate = os.path.join(AI_COMPASS_ROOT, subdir, "llama-bench.exe")
        if os.path.exists(candidate):
            return candidate
        candidate_linux = os.path.join(AI_COMPASS_ROOT, subdir, "llama-bench")
        if os.path.exists(candidate_linux):
            return candidate_linux

    return None



def main():
    parser = argparse.ArgumentParser(description="AI-COMPASS Benchmark Runner")
    parser.add_argument("--model", required=True, help="Path to GGUF model")
    parser.add_argument("--pp", type=int, default=256, help="Prompt processing tokens")
    parser.add_argument("--tg", type=int, default=64, help="Token generation count")
    parser.add_argument("--output", default=None, help="Output directory")
    parser.add_argument("--compare", help="Baseline trace CSV for comparison")
    parser.add_argument("--llama-bench", help="Path to llama-bench executable")
    parser.add_argument("--arch", default=None,
                        choices=["rdna1", "rdna2", "rdna3", "rdna3_5", "rdna4"],
                        help="GPU architecture (default: auto-detect)")
    parser.add_argument("--ngl", type=int, default=99,
                        help="Number of GPU layers (default: 99 = all)")
    parser.add_argument("--no-tracer", action="store_true",
                        help="Skip HIP tracer (just run llama-bench)")
    parser.add_argument("--gpu-timing", action="store_true",
                        help="Enable real hipEvent-based per-kernel GPU timing (HIP_TRACER_GPU_TIMING=1) "
                             "instead of the estimated dispatch-interval fallback. Automatically also sets "
                             "GGML_CUDA_DISABLE_GRAPHS=1: recording timing events on a stream that enters "
                             "HIP graph capture crashes the traced process (hipErrorCapturedEvent), and "
                             "ggml/llama.cpp's HIP backend enters graph capture after a couple of stable "
                             "decode steps. This trades away graph-replay throughput for the duration of "
                             "the trace -- t/s numbers from a --gpu-timing run are NOT representative of "
                             "normal throughput, only the category/occupancy breakdown is the point.")
    parser.add_argument("--cpu-pin", action="store_true", default=True,
                        help="Apply optimal CPU core affinity pinning to reduce SMT/socket/NUMA overhead (default: True)")
    parser.add_argument("--no-cpu-pin", action="store_false", dest="cpu_pin",
                        help="Disable SMT/socket/NUMA CPU affinity pinning")
    args = parser.parse_args()

    # 1. Pre-flight system validation check
    print("[AICOMPASS] Running pre-flight system validation...")
    val_res = validate.run_validation()
    if not val_res["ready"]:
        print("  [WARNING] Pre-flight system checks returned errors. Benchmark may fail.")
    if val_res["warnings"]:
        print("  [AICOMPASS] Pre-flight Validation Warnings:")
        for w in val_res["warnings"]:
            print(f"    - [{w['check']}] {w['message']} (Fix: {w.get('fix', 'none')})")

    # 2. Hardware auto-detection
    arch = args.arch
    if not arch:
        print("[AICOMPASS] Detecting system GPU architecture...")
        try:
            sys_info = detect.detect_hardware()
            if sys_info.get("gpus"):
                gpu = sys_info["gpus"][0]
                gfx = gpu.get("gfx_target") or ""
                detected_arch = None
                if gfx.startswith("gfx120") or gfx.startswith("gfx121"):
                    detected_arch = "rdna4"
                elif gfx.startswith("gfx115"):
                    detected_arch = "rdna3_5"
                elif gfx.startswith("gfx110"):
                    detected_arch = "rdna3"
                elif gfx.startswith("gfx103"):
                    detected_arch = "rdna2"
                elif gfx.startswith("gfx101"):
                    detected_arch = "rdna1"
                
                if detected_arch:
                    arch = detected_arch
                    print(f"  Auto-detected AMD GPU: {gpu['name']} ({gfx}) -> Setting arch={arch}")
                else:
                    arch = "rdna4"
                    print(f"  Unknown AMD GPU gfx target: {gfx} -> Defaulting to arch=rdna4")
            else:
                arch = "rdna4"
                print("  No AMD GPU detected -> Defaulting to arch=rdna4")
        except Exception as e:
            arch = "rdna4"
            print(f"  Failed to auto-detect hardware ({e}) -> Defaulting to arch=rdna4")

    lb = args.llama_bench or find_llama_bench()
    if not lb:
        print(" llama-bench not found. Build it first:")
        print("   cd llma.cpp-ROCM-Test && mkdir build-hip && cd build-hip")
        print("   cmake .. -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release")
        print("   cmake --build . --config Release --target llama-bench")
        return 1

    if not os.path.exists(args.model):
        print(f" Model not found: {args.model}")
        return 1

    out_dir = args.output or os.path.join(AI_COMPASS_ROOT, "benchmark_runs",
                                          f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)

    trace_csv = os.path.join(out_dir, "hip_trace.csv")
    report_dir = os.path.join(out_dir, "report")
    os.makedirs(report_dir, exist_ok=True)

    print("=" * 60)
    print("[AICOMPASS] AI-COMPASS Benchmark Runner")
    print("=" * 60)
    print(f"Model:    {args.model}")
    print(f"Arch:     {arch.upper()}")
    print(f"PP:       {args.pp}")
    print(f"TG:       {args.tg}")
    print(f"Output:   {out_dir}")
    print(f"llama-bench: {lb}")
    print()

    print(" Launching with HIP tracer...")
    env = os.environ.copy()
    if args.gpu_timing and not args.no_tracer:
        env["HIP_TRACER_GPU_TIMING"] = "1"
        env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
        print("  [AICOMPASS] --gpu-timing: real per-kernel GPU timing enabled, HIP graph capture "
              "disabled for this run (required -- see --help). t/s numbers below will be lower than normal.")
    launcher = os.path.join(AI_COMPASS_ROOT, "..", "hip_tracer", "build", "hip_tracer_launcher.exe")
    rocprofv3 = find_rocprofv3()

    if args.no_tracer:
        print("  Running without tracer (--no-tracer).")
        cmd = [lb, "-m", args.model, "-p", str(args.pp), "-n", str(args.tg), "-ngl", str(args.ngl)]
    elif os.path.exists(launcher):
        cmd = [launcher, "--output", trace_csv, lb, "-m", args.model,
               "-p", str(args.pp), "-n", str(args.tg), "-ngl", str(args.ngl)]
    elif rocprofv3:
        # Linux ROCm fallback: use rocprofv3 for kernel tracing
        rocprof_out = os.path.join(out_dir, "rocprofv3_raw")
        os.makedirs(rocprof_out, exist_ok=True)
        print(f"  [AICOMPASS] hip_tracer launcher not found. Using rocprofv3 for kernel tracing.")
        env["HIP_TRACER_GPU_TIMING"] = "1"
        env["GGML_CUDA_DISABLE_GRAPHS"] = "1"
        cmd = [rocprofv3, "-s", "-f", "csv", "-d", rocprof_out, "--",
               lb, "-m", args.model, "-p", str(args.pp), "-n", str(args.tg), "-ngl", str(args.ngl)]
    else:
        print("  Running without tracer (no launcher found and rocprofv3 not available).")
        cmd = [lb, "-m", args.model, "-p", str(args.pp), "-n", str(args.tg), "-ngl", str(args.ngl)]
    print(f"   {' '.join(cmd)}")

    # Set up CPU affinity pinning if enabled
    affinity_mask = None
    if args.cpu_pin:
        try:
            tune_res = cpu_tune.tune_cpu(busy_threshold=15.0)
            if sys.platform == "win32" and "windows_affinity_mask" in tune_res:
                affinity_mask = tune_res["windows_affinity_mask"]
                print(f"  [AICOMPASS] CPU Pinning enabled: Will apply affinity mask {affinity_mask} (Hex: 0x{affinity_mask:X}) to benchmark process.")
            elif sys.platform != "win32" and "vllm_cpu_omp_threads_bind" in tune_res:
                env["VLLM_CPU_OMP_THREADS_BIND"] = tune_res["vllm_cpu_omp_threads_bind"]
                print(f"  [AICOMPASS] CPU Pinning enabled: Set VLLM_CPU_OMP_THREADS_BIND=\"{tune_res['vllm_cpu_omp_threads_bind']}\"")
        except Exception as te:
            print(f"  [AICOMPASS] CPU Pinning configuration failed: {te}")

    try:
        if sys.platform == "win32" and affinity_mask is not None:
            import ctypes
            import ctypes.wintypes
            
            proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
            kernel32.SetProcessAffinityMask.restype = ctypes.wintypes.BOOL
            
            success = kernel32.SetProcessAffinityMask(proc._handle, affinity_mask)
            if success:
                print(f"  [AICOMPASS] Successfully assigned processor affinity mask {affinity_mask} to process pid {proc.pid}")
            else:
                err_code = kernel32.GetLastError()
                print(f"  [AICOMPASS] Warning: SetProcessAffinityMask failed with code {err_code}")
                
            stdout, stderr = proc.communicate(timeout=600)
            result = subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)
        else:
            result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
            
        print(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
        if result.stderr:
            print(f"STDERR: {result.stderr[-1000:]}")
    except subprocess.TimeoutExpired:
        print(" Benchmark timed out (10 min)")
        return 1
    except FileNotFoundError:
        print(f" Failed to launch {lb}")
        return 1

    # Parse benchmark results from stdout
    bench_pp = None
    bench_tg = None
    for line in result.stdout.split('\n'):
        if 'pp512' in line or 'pp256' in line or f'pp{args.pp}' in line:
            parts = line.strip().split('|')
            if len(parts) >= 6:
                try:
                    bench_pp = float(parts[-1].strip().split()[0])
                except:
                    pass
        if 'tg128' in line or 'tg64' in line or f'tg{args.tg}' in line:
            parts = line.strip().split('|')
            if len(parts) >= 6:
                try:
                    bench_tg = float(parts[-1].strip().split()[0])
                except:
                    pass

    print(f"\n Analyzing trace: {trace_csv}")
    # If rocprofv3 was used, convert its kernel_trace.csv to AI-COMPASS format
    if rocprofv3 and not os.path.exists(trace_csv):
        import glob
        kernel_csvs = glob.glob(os.path.join(out_dir, "rocprofv3_raw", "**", "*_kernel_trace.csv"), recursive=True)
        if kernel_csvs:
            convert_rocprof_csv_to_aicompass(kernel_csvs[0], trace_csv)
            print(f"  [AICOMPASS] Converted rocprofv3 kernel trace to {trace_csv}")
    if os.path.exists(trace_csv):
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from analyze import parse_trace, analyze_trace, generate_html_report, compare_traces

        records = parse_trace(trace_csv)
        if records:
            from analyze import csv_has_measured_duration
            measured = csv_has_measured_duration(trace_csv)
            if not measured:
                print("   NOTE: hip_tracer.cpp logs async launches without GPU timers -- "
                      "timing figures below are estimated dispatch intervals, not measured kernel duration.")
            analysis = analyze_trace(records, arch=arch, duration_is_measured=measured)
            s = analysis["summary"]
            print(f"   Kernels: {s['total_kernels']}  Time: {s['total_time_ms']} ms")
            print(f"   Avg: {s['avg_kernel_us']} s  GPU busy: {s['estimated_gpu_busy_pct']}%")

            print(f"\n   Category breakdown:")
            for cat, st in sorted(analysis["category_breakdown"].items(), key=lambda x: -x[1]["pct"]):
                print(f"     {cat:20s} {st['count']:5d}  {st['total_ms']:8.1f}ms  {st['pct']:5.1f}%  occ:{st['avg_occupancy_pct']:.0f}%")

            if analysis["bottlenecks"]:
                print(f"\n    Bottlenecks:")
                for b in analysis["bottlenecks"]:
                    print(f"       {b}")

            if analysis["optimization_targets"]:
                print(f"\n    Optimization Targets:")
                for t in analysis["optimization_targets"]:
                    print(f"      [{t['target']}] {t['suggestion']}")

            import json
            with open(os.path.join(report_dir, "analysis.json"), "w") as f:
                json.dump(analysis, f, indent=2)

            html_path = os.path.join(report_dir, "report.html")
            generate_html_report(analysis, html_path)
            print(f"\n    Report: {html_path}")

            if args.compare and os.path.exists(args.compare):
                print(f"\n    Comparing vs baseline: {args.compare}")
                diff = compare_traces(args.compare, trace_csv, arch=arch)
                with open(os.path.join(report_dir, "comparison.json"), "w") as f:
                    json.dump(diff, f, indent=2)
                if diff["improvements"]:
                    print(f"    Improvements:")
                    for k, v in diff["improvements"].items():
                        print(f"      {k}: {v}")
                if diff["regressions"]:
                    print(f"    Regressions:")
                    for k, v in diff["regressions"].items():
                        print(f"      {k}: {v}")
        else:
            print(" No kernel records found in trace")
    else:
        print(f" Trace CSV not found: {trace_csv}")
        # Still show summary with benchmark numbers
        if bench_pp is not None or bench_tg is not None:
            print(f"\n    llama-bench results:")
            if bench_pp: print(f"      PP{args.pp}: {bench_pp:.1f} t/s")
            if bench_tg: print(f"      TG{args.tg}: {bench_tg:.1f} t/s")

    print(f"\n Done. Results in: {out_dir}")
    print(f"   Report: {os.path.join(report_dir, 'report.html')}")
    print(f"   Trace:  {trace_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

