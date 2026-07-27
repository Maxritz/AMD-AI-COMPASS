# 🧭 AI-COMPASS

**AI Compute Performance Analysis & Statistics Suite**

[![RDNA4](https://img.shields.io/badge/AMD-RDNA4_Focused-ED1C24?logo=amd)](https://github.com/Maxritz/AMD-AI-COMPASS)
[![Windows](https://img.shields.io/badge/OS-Windows_10|11-0078D6?logo=windows)](https://github.com/Maxritz/AMD-AI-COMPASS)
[![ROCm](https://img.shields.io/badge/ROCm-7.13+-6C1F8E?logo=amd)](https://rocm.docs.amd.com)
[![HIP](https://img.shields.io/badge/HIP-C++-00599C?logo=cplusplus)](https://github.com/ROCm/HIP)
[![GPUPerfAPI](https://img.shields.io/badge/GPUPerfAPI-4.0+-informational)](https://github.com/GPUOpen-Tools/gpu_performance_api)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![GPU Profiler](https://img.shields.io/badge/category-GPU_Profiler-blueviolet)](#)
[![Performance](https://img.shields.io/badge/category-Performance_Analysis-success)](#)

A unified RDNA1–4 AI compute performance toolset for Windows that traces every kernel dispatch, measures GPU utilization in real time, reads hardware performance counters, and generates actionable optimization reports — from model loading through prompt processing to token generation.

---

## What It Does

AI-COMPASS instruments your AI inference workload end-to-end and answers three questions:

1. **Where is the GPU spending its time?** — per-kernel breakdown (MMQ, MMVQ, Attention, MoE, RoPE, Norm, etc.) across prompt processing and token generation phases.
2. **What is limiting performance?** — occupancy bottlenecks, low GPU utilization, specific kernels dominating the critical path.
3. **Did my optimization help?** — before/after comparison with per-category regression detection.

### Example Output

```
📊 Summary: 1,247 kernels | 4,562 ms total | GPU busy: 87%

📈 Phases:
  Prompt Processing: 312 kernels, 1,234 ms (27%)
  Token Generation:  935 kernels, 3,328 ms (73%)

🏷️  Category Breakdown:
  MMQ         423 kernels   1923.4 ms   42.1%   occ:28.3%
  Attention   287 kernels    891.2 ms   19.5%   occ:45.1%
  MMVQ        198 kernels    612.5 ms   13.4%   occ:12.7%
  ...

🚨 Bottlenecks:
  ⚠ MMQ dominates at 42% with low occupancy (28%)
  ⚠ MMVQ has very low occupancy (12.7%) — small-k path bound

🔧 Optimization Targets:
  [MMQ]  K-tile doubling applied. Check MMQ_ITER_K for RDNA4 wave32.
  [MMVQ] Split-K heuristic may need tuning for RDNA4 small batches.
```

Reports are generated as both JSON (for scripting) and standalone HTML (for sharing).

---

## How It Helps

| Problem | AI-COMPASS Solution |
|---------|---------------------|
| "Why is inference slow?" | Per-kernel timing breakdown shows exact cost |
| "Is the GPU fully utilized?" | ADLX real-time GPU metrics (util %, clocks, power, temp) |
| "Which kernel should I optimize?" | Sorted by time %, bounded by occupancy analysis |
| "Did my patch help?" | `--compare` mode diffs before vs after runs |
| "Where is the bottleneck?" | Automatic bottleneck detection per category |
| "Can I visualize this?" | RCV integration for GPU trace visualization |

---

## Quick Start

### Prerequisites
- Windows 10/11 with AMD Radeon RX 5000+ GPU (RDNA1–4)
- ROCm 7.13+ (for HIP runtime) — [`E:\ROCM-7.13.0-Windows\`]
- Visual Studio 2022 + CMake + Ninja
- Vulkan SDK 1.4+ (for GPUPerfAPI counter access)

### Build

```bash
git clone https://github.com/Maxritz/AMD-AI-COMPASS.git
cd AI-COMPASS
mkdir build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

### Profile a Model

```bash
# Quick analysis from an existing HIP trace
python tools/analyze.py hip_trace.csv --output report/

# Full end-to-end benchmark
python tools/run_benchmark.py --model path/to/model.gguf --pp 256 --tg 64
```

### Compare Before/After Optimization

```bash
# Run baseline
python tools/run_benchmark.py --model model.gguf --pp 256 --tg 64 -o baseline/

# Apply optimization, rebuild, run again
python tools/run_benchmark.py --model model.gguf --pp 256 --tg 64 -o optimized/ \
  --compare baseline/hip_trace.csv
```

### Visualize with RCV

```bash
build/rocprof-compute-viewer.exe output_dir/
```

---

## Components

| Component | Purpose | RDNA4 | RDNA3 | RDNA2 | RDNA1 | Source |
|-----------|---------|-------|-------|-------|-------|--------|
| **HIP Tracer** | Kernel dispatch hooking via MinHook | ✅ | ✅ | ✅ | ✅ | Custom |
| **GPUPerfAPI** | GPU SQ/SPI/TCP/GL2C hardware counters | ✅ | ✅ | ✅ | ✅ | [AMD GPUOpen](https://github.com/GPUOpen-Tools/gpu_performance_api) |
| **ADLX** | Real-time GPU metrics (util %, clocks, power, temp, VRAM) | ✅ | ✅ | ✅ | ✅ | [AMD ADLX SDK](https://github.com/GPUOpen-LibrariesAndSDKs/ADLX) |
| **rocprof-compute-viewer** | GPU trace visualization (SQTT viewer) | ✅ | ✅ | ✅ | ✅ | [ROCm](https://github.com/ROCm/rocprof-compute-viewer) |
| **rocprofv3** | Compute profiling CLI | ✅ | ✅ | ✅ | ✅ | [ROCm](https://github.com/ROCm/rocm-systems) |
| **GEAK** | Agent-based workload optimization | ✅ | ⚠ | ⚠ | ❌ | Custom port |
| **Hyperloom** | Performance tuning framework | ✅ | ⚠ | ⚠ | ❌ | Custom port |
| **AQLProfile** | Low-level SQ/SX/TA/TD counter access | ✅ | ✅ | ✅ | ⚠ | [ROCm](https://github.com/ROCm/rocm-systems) |
| **Magpie** | Trace analysis | ✅ | ✅ | ✅ | ⚠ | Custom port |
| **intellikit** | AI toolkit | ✅ | ✅ | ✅ | ⚠ | Custom port |

✅ Full support  ⚠ Partial  ❌ Not supported

---

## GPU Architecture Support

Auto-detected via `hipGetDeviceProperties` with ADLX fallback:

| Arch | GFX IP | Example GPU |
|------|--------|-------------|
| RDNA1 | gfx1010–1012 | RX 5700 XT |
| RDNA2 | gfx1030–1035 | RX 6900 XT |
| RDNA3 | gfx1100–1103 | RX 7900 XTX |
| RDNA3.5 | gfx1150–1151 | RX 8800 XT |
| **RDNA4** | **gfx1200–1201** | **RX 9070 XT** ✅ primary target |

---

## CLI Reference

```
aicompass <command> [options]

Commands:
  trace    <app> [args]   Capture HIP kernel trace + GPU metrics
  profile  <app> [args]   Full profile with all plugins
  analyze  <dir>          Post-process trace data
  list                    List installed plugins
  help                    Show this help

Options:
  -o, --output <dir>     Output directory
  -i, --interval <ms>    Metrics poll interval (default: 100)
  -v, --verbose          Verbose output
  --counters             Enable hardware counter collection
```

### Python Analysis Tools

```
python tools/analyze.py <trace.csv> [options]

Options:
  -o, --output <dir>    Output directory for reports
  --compare <baseline>  Compare against a baseline trace
  --cu-count <n>        GPU CU count (default: 32)
  --html                Generate HTML report (default: on)
```

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   AI-COMPASS CLI                         │
│        trace | profile | analyze | visualize              │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │HIP Tracer│  │GPUPerfAPI│  │ADLX      │  │Plugins  │ │
│  │(MinHook) │  │(VK/DX12) │  │(Metrics) │  │GEAK/etc │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬────┘ │
│       │             │             │             │       │
│       └─────────────┴─────────────┴─────────────┘       │
│                         │                                │
│                  ┌──────▼──────┐                         │
│                  │  Analysis   │                         │
│                  │  Engine     │                         │
│                  │  (analyze.py)│                        │
│                  └──────┬──────┘                         │
│                         │                                │
│            ┌────────────▼────────────┐                   │
│            │  Output Formats         │                   │
│            │  HTML | JSON | RCV      │                   │
│            └─────────────────────────┘                   │
└──────────────────────────────────────────────────────────┘
```

---

## Credits & Acknowledgements

AI-COMPASS integrates and builds upon several open-source projects. We are deeply grateful to their creators:

### Core Dependencies
- **[GPU Performance API (GPUPerfAPI)](https://github.com/GPUOpen-Tools/gpu_performance_api)** — AMD GPUOpen. MIT license. Hardware performance counter access library for Windows and Linux. We use its Vulkan backend for SQ/SPI/TCP/GL2C counter collection.
- **[ADLX SDK](https://github.com/GPUOpen-LibrariesAndSDKs/ADLX)** — AMD GPUOpen. MIT license. AMD Display Library Next — provides real-time GPU metrics (utilization, clocks, power, temperature, VRAM).
- **[rocprof-compute-viewer](https://github.com/ROCm/rocprof-compute-viewer)** — ROCm. MIT license. GPU trace visualization tool for rocprofv3 thread trace data.
- **[rocprofiler-sdk / rocprofv3](https://github.com/ROCm/rocm-systems)** — ROCm. MIT license. ROCm profiler SDK and compute profiling CLI.
- **[ROCm](https://github.com/ROCm)** — AMD. Various licenses. HIP runtime, ROCr, compiler toolchain.
- **[MinHook](https://github.com/TsudaKageyu/minhook)** — Tsuda Kageyu. BSD 2-Clause. Minimalistic API hooking library used by the HIP tracer.
- **[Qt](https://www.qt.io/)** — The Qt Company. GPL/LGPL. UI framework used by rocprof-compute-viewer.
- **[Vulkan SDK](https://vulkan.lunarg.com/)** — LunarG / Khronos. Vulkan API headers and tools.

### AI Toolkit Ports
- **GEAK-RDNA** — Agent-based optimization framework, ported for RDNA4.
- **Hyperloom-RDNA** — Performance tuning framework, ported for RDNA4.
- **Magpie** — Trace analysis tool, ported for RDNA4.
- **intellikit** — AI toolkit integration, ported for RDNA4.
- **Apex** — Optimization toolkit, ported for RDNA4.
- **TraceLens** — Trace analysis visualization, ported for RDNA4.

### Our Work
All integration code — the HIP tracer DLL, GPUPerfAPI HIP adapter, ADLX metrics poller, plugin system, kernel classification engine, bottleneck detection, HTML report generator, and CLI framework — is original work developed for this project.

---

## License

MIT — See [LICENSE](LICENSE) for details. Third-party components retain their original licenses.
