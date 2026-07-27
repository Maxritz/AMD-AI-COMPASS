# AI-COMPASS

**AI Compute Performance Analysis & Statistics Suite**

[![RDNA4](https://img.shields.io/badge/AMD-RDNA4-ED1C24?logo=amd)](https://github.com/Maxritz/AMD-AI-COMPASS)
[![Windows](https://img.shields.io/badge/OS-Windows_10|11-0078D6?logo=windows)](https://github.com/Maxritz/AMD-AI-COMPASS)
[![ROCm](https://img.shields.io/badge/ROCm-7.13+-6C1F8E?logo=amd)](https://rocm.docs.amd.com)
[![HIP](https://img.shields.io/badge/HIP-C++-00599C?logo=cplusplus)](https://github.com/ROCm/HIP)
[![GPUPerfAPI](https://img.shields.io/badge/GPUPerfAPI-4.0+-informational)](https://github.com/GPUOpen-Tools/gpu_performance_api)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

A unified RDNA1–4 AI compute performance toolset for Windows that traces every kernel dispatch, measures GPU utilization in real time, reads hardware performance counters, and generates actionable optimization reports — from model loading through prompt processing to token generation.

---

## What It Does

AI-COMPASS instruments your AI inference workload end-to-end and answers three questions:

1. **Where is the GPU spending its time?** — per-kernel breakdown (MMQ, MMVQ, Attention, MoE, RoPE, Norm, etc.) across prompt processing and token generation phases.
2. **What is limiting performance?** — occupancy bottlenecks, low GPU utilization, specific kernels dominating the critical path.
3. **Did my optimization help?** — before/after comparison with per-category regression detection.

### Example Output

```
Summary: 4,956 kernels | 68.88 ms total | GPU busy: 47.7%

Phases:
  Prompt Processing: 990 kernels, 41.21 ms (59.8%)
  Token Generation:  3966 kernels, 27.67 ms (40.2%)

Category Breakdown:
  MMVQ         848 kernels   27.3 ms   39.6%   occ:27.6%
  Vector      1332 kernels   20.2 ms   29.3%   occ:58.8%
  Attention   2472 kernels   12.8 ms   18.5%   occ:51.7%
  Norm          96 kernels    4.8 ms    6.9%   occ:100%
  Quantize     192 kernels    3.9 ms    5.6%   occ:95.8%

Bottlenecks:
  MMVQ dominates at 39.6% with low occupancy (27.6%)
  MMVQ has low occupancy — likely occupancy-bound

Optimization Targets:
  [MMVQ] Check Split-K heuristic and small_k path for RDNA4.
  [Attention] Consider flash attention tuning.
```

Reports are generated as both JSON (for scripting) and standalone HTML (for sharing).

---

## How It Helps

| Problem | AI-COMPASS Solution |
|---------|---------------------|
| "Why is inference slow?" | Per-kernel timing breakdown shows exact cost |
| "Is the GPU fully utilized?" | ADLX real-time GPU metrics (util %, clocks, power, temp, VRAM) |
| "Which kernel should I optimize?" | Sorted by time %, bounded by occupancy analysis |
| "Did my patch help?" | `--compare` mode diffs before vs after runs — measured +7.6% TG on Gemma-4 12B |
| "Where is the bottleneck?" | Automatic bottleneck detection per category |
| "Can I visualize this?" | RCV integration for GPU trace visualization |

---

## Session Results (Jul 2026)

All benchmarks on **RX 9070 XT (gfx1201, 16GB VRAM, Wave32)** with **Ryzen 9 5900XT / 96GB DDR4**.

### RDNA4 Kernel Optimizations Committed

| Change | File | Gemma-4 12B TG | Qwen3.5-9B NVFP4 TG |
|--------|------|----------------|---------------------|
| Baseline | — | 58.3 t/s | 103.0 t/s |
| nwarps=4 for K-quant | mmvq.cu | 58.5 t/s | — |
| Thread mapping fix | mmvq.cu | **62.7 t/s (+7.6%)** | — |
| nwarps=2 for FP4 | mmvq.cu | — | **128.1 t/s (+24.4%)** |
| CPU AVX2+FMA enabled | CMake | +23% on CPU TG | — |
| **Best combined** | — | **62.7 t/s** | **128.1 t/s** |

### Quantization Format Comparison

| Model | Format | Size | PP256 | TG128 |
|-------|--------|------|-------|-------|
| Gemma-4 12B | Q4_K_M | 6.86GB | 3,648 t/s | 62.7 t/s |
| Gemma-4 12B | IQ4_XS | 6.22GB | 4,510 t/s | 68.4 t/s |
| Qwen3.5-9B | NVFP4 | 4.94GB | 4,810 t/s | 103.0 t/s |
| Qwen3-8B | Q4_K_M | 4.68GB | 5,854 t/s | 99.1 t/s |
| Qwen3.5-4B | NVFP4 | 2.36GB | 6,405 t/s | 149.7 t/s |
| Laguna XS 33B MoE | IQ4_XS | 16.84GB | 698 t/s | 69.0 t/s |
| Laguna XS 33B MoE | Q4_K_M | 18.88GB | 557 t/s | 51.6 t/s |

### ROCmFP4 Support (New)

ROCmFP4 (AMD-native FP4 format) is now buildable in the HIP backend:
- Add `GGML_TYPE_Q4_0_ROCMFP4` (type 100) and `Q4_0_ROCMFP4_FAST` (101)
- Full vec_dot, dequantize, MMVQ dispatch, nwarps=2 tuning
- **Status**: Builds clean, model loading crashes at runtime (HIP copy path needs completion)

### Large Model CPU Inference

| Model | Size | CPU TG | Bottleneck |
|-------|------|--------|------------|
| Laguna 117B Q4_K | 68GB | 4.3 t/s | DDR4 BW (~50 GB/s) |
| Laguna XS 33B Q4_K | 18.9GB | 13.0 t/s | DDR4 BW (~50 GB/s) |
| MiniCPM 1B Q8_0 | 1.1GB | 38.7 t/s | CPU compute |

---

## Quick Start

### Prerequisites
- Windows 10/11 with AMD Radeon RX 5000+ GPU (RDNA1–4)
- ROCm 7.13+ (for HIP runtime)
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

For the llama.cpp ROCm build with CPU optimizations:
```bash
cd llma.cpp-ROCM-Test
cmake -S . -B build-hip -G Ninja -DCMAKE_BUILD_TYPE=Release ^
  -DGGML_HIP=ON -DGGML_CUDA_FA_ALL_QUANTS=ON -DGGML_OPENMP=OFF ^
  -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_AVX_VNNI=ON
cmake --build build-hip --target llama-bench
```

### Profile a Model

```bash
# Quick analysis from an existing HIP trace
python tools/analyze.py hip_trace.csv --output report/

# Full end-to-end benchmark
python tools/run_benchmark.py --model path/to/model.gguf --pp 256 --tg 128
```

### Compare Before/After Optimization

```bash
# Run baseline
python tools/run_benchmark.py --model model.gguf --pp 256 --tg 128 -o baseline/

# Apply optimization, rebuild, run again
python tools/run_benchmark.py --model model.gguf --pp 256 --tg 128 -o optimized/ \
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
| **ROCmFP4** | Native AMD FP4 format (HIP backend, WIP) | ⚠ | ⚠ | ❌ | ❌ | [Ciru ROCmFPX](https://github.com/ciru-ai/ROCmFPX) |
| **GEAK** | Agent-based workload optimization | ✅ | ⚠ | ⚠ | ❌ | Custom port |
| **Hyperloom** | Performance tuning framework | ✅ | ⚠ | ⚠ | ❌ | Custom port |
| **AQLProfile** | Low-level SQ/SX/TA/TD counter access | ✅ | ✅ | ✅ | ⚠ | [ROCm](https://github.com/ROCm/rocm-systems) |
| **Magpie** | Trace analysis | ✅ | ✅ | ✅ | ⚠ | Custom port |
| **intellikit** | AI toolkit | ✅ | ✅ | ✅ | ⚠ | Custom port |

✅ Full support  ⚠ Partial/WIP  ❌ Not supported

---

## GPU Architecture Support

Auto-detected via `hipGetDeviceProperties` with ADLX fallback:

| Arch | GFX IP | Example GPU | Tested |
|------|--------|-------------|--------|
| RDNA1 | gfx1010–1012 | RX 5700 XT | ❌ |
| RDNA2 | gfx1030–1035 | RX 6900 XT | ❌ |
| RDNA3 | gfx1100–1103 | RX 7900 XTX | ❌ |
| RDNA3.5 | gfx1150–1151 | RX 8800 XT | ❌ |
| **RDNA4** | **gfx1200–1201** | **RX 9070 XT** | ✅ primary target |

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

## RDNA4 Optimization Guide

### Key Findings for RX 9070 XT

1. **MMVQ is memory-bandwidth bound** — K-quant types saturate VRAM bandwidth. TG improvement comes from reducing memory traffic (IQ4_XS, NVFP4) rather than increasing occupancy.
2. **nwarps=2 (64 threads) is the sweet spot for FP4** — avoids load imbalance from idle threads on small K tiles while providing enough parallelism.
3. **Thread mapping matters** — fixing `tid/16` to `tid/8` in MMVQ eliminated 2x start-position collision within warps, gaining +7.6% TG.
4. **IQ4_XS beats Q4_K_M** — simpler dequantization + less data = +17% TG on Gemma-4 12B.
5. **CPU AVX2 was OFF** — the HIP-targeted build had all CPU SIMD disabled. Enabling AVX2+FMA gave +23% on CPU TG.
6. **ROCmFP4 uses dp4a** — not native WMMA yet on RDNA4. A true WMMA FP4 kernel would bypass dp4a entirely.

---

## Credits & Acknowledgements

AI-COMPASS integrates and builds upon several open-source projects:

### Core Dependencies
- **[GPU Performance API (GPUPerfAPI)](https://github.com/GPUOpen-Tools/gpu_performance_api)** — AMD GPUOpen. MIT license. Hardware performance counter access library.
- **[ADLX SDK](https://github.com/GPUOpen-LibrariesAndSDKs/ADLX)** — AMD GPUOpen. MIT license. AMD Display Library Next.
- **[rocprof-compute-viewer](https://github.com/ROCm/rocprof-compute-viewer)** — ROCm. MIT license. GPU trace visualization.
- **[rocprofiler-sdk / rocprofv3](https://github.com/ROCm/rocm-systems)** — ROCm. MIT license. ROCm profiler SDK.
- **[ROCm](https://github.com/ROCm)** — AMD. HIP runtime, ROCr, compiler toolchain.
- **[ROCmFPX](https://github.com/ciru-ai/ROCmFPX)** — Ciru / Charlie. Custom ROCmFP4 Vulkan runtime.
- **[MinHook](https://github.com/TsudaKageyu/minhook)** — Tsuda Kageyu. BSD 2-Clause. API hooking library.
- **[Qt](https://www.qt.io/)** — The Qt Company. GPL/LGPL. UI framework for RCV.
- **[Vulkan SDK](https://vulkan.lunarg.com/)** — LunarG / Khronos. Vulkan API headers.

### AI Toolkit Ports
- **GEAK-RDNA** — Agent-based optimization framework, ported for RDNA4.
- **Hyperloom-RDNA** — Performance tuning framework, ported for RDNA4.
- **Magpie** — Trace analysis tool, ported for RDNA4.
- **intellikit** — AI toolkit integration, ported for RDNA4.
- **Apex** — Optimization toolkit, ported for RDNA4.
- **TraceLens** — Trace analysis visualization, ported for RDNA4.

### Our Work
All integration code — the HIP tracer DLL, GPUPerfAPI HIP adapter, ADLX metrics poller, plugin system, kernel classification engine, bottleneck detection, HTML report generator, MMVQ RDNA4 optimizations (nwarps tuning, thread mapping fix, FP4 support), and CLI framework — is original work developed for this project.

---

## License

MIT — See [LICENSE](LICENSE) for details. Third-party components retain their original licenses.
