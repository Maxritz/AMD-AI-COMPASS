---
title: RDNA4 on Windows 11 with HIP-ROCm 7.3 — deployment guide
kind: optimization
gens: [gfx1201, gfx1203, gfx1206, gfx1207]
dtypes: [fp16, bf16, fp8, int4]
regimes: [inference]
updated: 2026-07-25
platform: windows
---

# RDNA4 on Windows 11 with HIP-ROCm 7.3 — deployment guide

## Prerequisites

1. **Windows 11 24H2 or later** (required for full ROCm 7.3 support)
2. **AMD Radeon RX 9070 XT or newer RDNA4 GPU**
3. **HIP-ROCm 7.3 for Windows** (not available in official repos yet; download beta from AMD developer portal)
4. **Visual Studio 2022 17.10+** with C++ workload
5. **CMake 3.28+**, Python 3.11+

## Installation Steps

### Step 1: Install ROCm 7.3 Beta for Windows

```powershell
# Download ROCm 7.3 beta installer from AMD developer portal
# Or use winget (when available):
winget install --id AMD.ROCm.7.3

# Verify installation
hipConfig --show-version
```

Expected output:
```
HIP version: 7.3.0
ROCm version: 7.3.0
```

### Step 2: GPU Detection

On Windows, **rocm-smi is not available**. Use Windows device enumeration:

```powershell
# Option A: hipConfig
hipConfig --show-device
# Expected: "Radeon RX 9070 XT" or "gfx1201"

# Option B: Environment variable (for multi-GPU systems)
$env:HIP_VISIBLE_DEVICES="0"  # Select GPU 0
```

### Step 3: Build Dependencies

```powershell
# Install Python deps
pip install torch==2.5.0+cpu
pip install triton @ https://github.com/ROCmSoftwarePlatform/triton/releases/download/rocm-7.3.0-rc1/triton-3.3.0-cp311-cp311-win_amd64.whl

# Clone llama.cpp with HIP support
git clone https://github.com/gokceneraslan/llama.cpp.git
cd llama.cpp
git checkout origin/rocm-hip
```

### Step 4: Build llama.cpp with HIP Support

```powershell
# Set build environment
$env:CMAKE_PREFIX_PATH="C:\Program Files\AMD\ROCm\7.3"
$env:ROCM_PATH="C:\Program Files\AMD\ROCm\7.3"

cmake -B build -B build -DGPU_ARCH="gfx1201" ^
  -DLLAMA_HIPBLAS=ON ^
  -DLLAMA_NATIVE=OFF ^
  -DCMAKE_BUILD_TYPE=Release

cmake --build build --config Release -j 16
```

## Key Differences from Linux ROCm

| Aspect | Linux ROCm 7.x | Windows ROCm 7.3 |
|---|---|---|
| Device probe | `rocm-smi --showproductname` | `hipConfig --show-device` |
| Environment | `ROCM_VISIBLE_DEVICES` | `HIP_VISIBLE_DEVICES` |
| Profiler | `rocprof-compute` | `rocprof-compute` (limited) |
| Shell | bash | PowerShell/cmd |
| Path | `/opt/rocm` | `C:\Program Files\AMD\ROCm\7.3` |

## Triton FP8 WMMA Patch for Windows

vLLM/LLAMA-DX requires local patches for RDNA4 FP8 WMMA:

```python
# In your vLLM setup
# Disable AITER which blocks gfx1201
os.environ["VLLM_DISABLE_AITER"] = "1"

# Patch device detector
from vllm.backends.amd.hip.detect import detect_hip_device
# Add gfx1201 mapping
```

### Triton Kernel Patch Template

```python
# Add to llama.cpp custom ops or vLLM triton kernels
import triton
from triton.backends.amd import driver as amd_driver

if amd_driver.has_wmma_fp8():
    # Use 16x16x128 FP8 WMMA
    BLOCK_M, BLOCK_N, BLOCK_K = 16, 16, 128
else:
    # FP16 fallback
    BLOCK_M, BLOCK_N, BLOCK_K = 16, 16, 32
```

## Performance Tuning for RDNA4 on Windows

### Tile Size Recommendations

| Dtype | TILE_M | TILE_N | TILE_K | L2 Tiles |
|---|---|---|---|---|
| FP16 | 64 | 64 | 32 | 4 |
| BF16 | 64 | 64 | 32 | 4 |
| FP8 (WMMA) | 16 | 16 | 128 | 1 |
| INT4 (sparsity) | 128 | 128 | 128 | 1 |

### Launch Configuration

```python
# Optimal for RX 9070 XT (64 CUs, 128 lanes/CU = 8192 threads)
NUM_WARPS = 64  # 64 warps = 4096 threads per SM
BLOCKS_PER_SM = 2  # Double buffering
```

### Memory Bandwidth Strategy

GDDR6 @ 640 GB/s is the bottleneck. Optimize:
- **Attention KV cache**: Use sliding window for long sequences
- **PagedAttention**: Essential for long context with limited VRAM
- **Weight quantization**: Use Q4_0/Q4_1 to reduce bandwidth

## Troubleshooting

### Error: `hipErrorInvalidDeviceInfo` on window startup

```powershell
# Ensure HIP_VISIBLE_DEVICES is not set to invalid device
$env:HIP_VISIBLE_DEVICES=""  # Clear
# Or set to specific device
$env:HIP_VISIBLE_DEVICES="0"
```

### Error: `WMMA not supported`

- Requires gfx1201 or later
- Verify with `hipConfig --show-device`
- Triton compiled without AITER may block WMMA

### Poor performance

1. Check L2 hit rate: `rocprof-compute --stats`
2. Check memory BW: should be >70% of 640 GB/s for large models
3. Verify correct GPU arch: `-DGPU_ARCH=gfx1201` in CMake

## Next Steps for LLAMA-DX

1. [ ] Clone llama.cpp with ROCm HIP branch
2. [ ] Build with AMD GPU support
3. [ ] Apply Triton FP8 WMMA patches
4. [ ] Run `llama-bench` comparing FP16 vs FP8 WMMA paths
5. [ ] Profile with `rocprof-compute` for bandwidth bottlenecks
6. [ ] Tune tensor cores for Split-K GEMM (RDNA4 uses WMMA, not matrix cores)