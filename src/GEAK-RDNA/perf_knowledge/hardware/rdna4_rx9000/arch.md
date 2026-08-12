---
title: RDNA4 / RX 9070 XT / RX 9000 Series (gfx1201) — architecture overview
kind: hardware
gens: [gfx1201, gfx1203, gfx1206, gfx1207]
dtypes: [fp64, fp32, bf16, fp16, fp8_e4m3, fp8_e5m2, int8, int4]
regimes: [both]
updated: 2026-07-25
sources:
   - https://docs.amd.com/v/u/en-US/rdna4-instruction-set-architecture
---

# RDNA4 / RX 9070 XT / RX 9000 Series (gfx1201) — architecture overview

> Target: **AMD Radeon RX 9070 XT (gfx1201)**, RDNA4, ISA **gfx1201**  
> Consumer/gaming AI inference GPU, NOT to be confused with datacenter CDNA4 (MI350X/MI355X).  
> Matrix math is WMMA (Wavefront Matrix Multiply-Accumulate), not dedicated matrix cores.

## TL;DR
> RDNA4 is the gaming-focused successor to RDNA3: **64 CUs** (RX 9070 XT), **16 GB GDDR6 @ 640 GB/s**, **32 2nd-gen AI Accelerators** (2 per CU), **WMMA matrix throughput ~194.8 TFLOPS FP16**, native **FP8 WMMA support** for FSR4 and inference, **8 MB L2 unified cache**, **8x increase in matrix ops** vs RDNA3. Peaks: FP16/BF16 **194.8 TFLOPS**, FP8 **~380 TFLOPS**, INT4 **~1,557 AI TOPS** (with sparsity).

## The one-screen cheat sheet
| Fact | Value (RX 9070 XT) | Why it matters |
|---|---|---|
| Wavefront | **64 lanes** | unchanged from RDNA3 |
| CUs (active) | **64** | consumer tier, matrix via WMMA not dedicated cores |
| SIMD/CU | **4** | occupancy per-SIMD |
| Wave slots | 8/SIMD → 32/CU | hard cap |
| VGPR | 512 ×4 B/SIMD, 16-granule | same as CDNA4 |
| LDS | **128 KiB/CU**, **32 banks** | smaller than CDNA4, was 64 banks in RDNA3 |
| L2 | **8 MB unified**, 4 shader engines | doubled bank count from RDNA3 |
| Infinity Cache | **80 MiB** | rdna3: 32 MiB → 80 MiB |
| GDDR6 | **16 GB**, **640 GB/s bus** (256-bit) | gaming memory, bandwidth vs hbm3e |
| WMMA peak | FP16 **~195 TF**, FP8 **~380 TF**, INT4 **~1.5k AI TOPS** | vector-first design, matrix as bolt-on |
| AI Accelerators | **32** (2nd-gen, 2 per CU) | FSR4/FSR5 path, FP8 acceleration |
| FP8 variant | **OCP** (E4M3FN/E5M2) | same as CDNA4, native WMMA path |
| Process | TSMC **N4P** | monolithic die |
| TDP | **300 W** | vs 1000-1400 W CDNA4 |
| Engine clock | up to ~3000 MHz | basis of peak math |

## Key differences from CDNA4 (MI350X/MI355X)
- **No dedicated matrix cores** — RDNA4 uses WMMA instructions that run in ALU slots (like Intel AMX, NVIDIA Tensor Cores as separate execution units, but AMD's approach shares ALU resources)
- **64 CUs vs 256** — fewer but modernized for gaming workloads
- **GDDR6 vs HBM3E** — 640 GB/s vs 8 TB/s, critical bottleneck for large models
- **FP8 WMMA support** — native, used by FSR4, not fully merged into vLLM/LLAMA-DX yet
- **32 AI Accelerators** (2nd-gen, 2 per CU) — newer than CDNA3's scalar accelerators
- **32 LDS banks** (vs CDNA4's 64) — affects kernel tile sizing

## WMMA (Wavefront Matrix Multiply-Accumulate)
RDNA4 introduces **WMMA** as the matrix math engine. Unlike CDNA's MFMA (Matrix Fused Multiply-Add) which runs on dedicated matrix cores, WMMA dispatches through wavefront ALUs. Key shapes:
- **FP16/BF16**: 16×16×32 (wavefront tiles)
- **FP8**: 16×16×128 via WMMA
- **INT4**: via INT4 instructions with sparsity support

WMMA is invoked via `s_wmma` and `v_wmma` instructions ( gfx12xx ISA). This is fundamentally different from CDNA4's `mfma` that runs on separate matrix cores.

## The levers
1. **Exploit FP8 WMMA** — requires disabling AITER in vLLM/LLAMA-DX, local patches
2. **Tuned Triton configs** — gfx1201 tile sizes differ from gfx1200/gfx1203/gfx1206/gfx1207
3. **Sparsity for INT4** — achieves 1557 AI TOPS peak with weight sparsity
4. **64 CUs, 8-multiple wavefronts** for grid sizing
5. **L2 bandwidth** — 8 MB unified cache benefits attention KV caches

## Pitfalls
- **Assuming CDNA4 codepaths work** — different ISA, no `mfma`; use `wmma` instead
- **Missing FP8 WMMA support** — upstream vLLM lacks gfx1201 FP8 path, requires patches
- **Cache sizes differ** — L2 is unified (read/write), not split like CDNA
- **Memory bandwidth bottleneck** — GDDR6 limits larger models vs HBM3E

## Windows 11 HIP-ROCm 7.3 Specifics
- Use `HIP_VISIBLE_DEVICES` instead of `ROCM_VISIBLE_DEVICES`
- Device enumeration via AMD GPU drivers, not rocm-smi
- WMMA intrinsics require `-target amdgcn-amd-amdhsa -mcpu gfx1201` or higher

## GPU Variants
| Model | gfx | CUs | Notes |
|-------|-----|-----|-------|
| RX 9070 XT | gfx1201 | 64 | flagship RDNA4 |
| RX 9070 | gfx1207 | 54 | 50% CU reduction vs 9070 XT |
| RX 9060 XT | gfx1206 | 40 | entry RDNA4 |
| RX 9000 XT | gfx1203 | varies | laptop/mobile |

## Verify
- `amd-smi` or Device Manager → gfx1201, 64 CU, 16 GB GDDR6
- `rocminfo` in WSL2 or ROCm Windows
- `rocprof-compute` for WMMA occupancy (128 KiB LDS limit)

## Sources
- AMD RDNA4 Instruction Set Architecture (2026): https://docs.amd.com/v/u/en-US/rdna4-instruction-set-architecture