---
title: RDNA2 / RX 6000 deployment (gfx1030)
kind: hardware
gens: [gfx1030, gfx1031, gfx1032]
updated: 2026-08-12
---

# Deploying AI workloads on RDNA2 (RX 6000)

## ROCm
- **Linux ROCm 6.x**: RX 6000 is officially supported (gfx1030). Install ROCm 6.2+ for llama.cpp/onnxruntime/vllm-rocm paths.
- **Windows**: no official consumer ROCm for RX 6000. Options:
  - WSL2 + ROCm 6.x (unofficial but workable; see community builds),
  - DirectML backend for llama.cpp (DirectX12, uses `DirectML.dll`),
  - Vulkan backend (llama.cpp Vulkan) — no ROCm needed, works on Windows + Linux.
- Prefer **Vulkan** on Windows for RX 6000: zero driver dance, WMMA absent anyway so the Vulkan path loses nothing over HIP.

## Verify GPU is seen
```bash
rocminfo | grep gfx103       # Linux
lspci | grep -i radeon       # Linux device present
vulkaninfo | grep -i "deviceName"   # Vulkan path
```

## Known limits
- No FP8 / WMMA / AI accelerators — kernels must be vector SIMD (INT8/FP16/FP32).
- BF16 emulated — use FP16/INT8.
- L2 4 MB — small-model decode fits, larger models are GDDR6-bound (~0.5-0.7 tok/s q4 7B on 512 GB/s).
