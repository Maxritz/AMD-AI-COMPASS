---
title: RDNA2 / RX 6000 Series (gfx1030) - architecture overview
kind: hardware
gens: [gfx1030, gfx1031, gfx1032]
dtypes: [fp64, fp32, bf16, fp16, int8, int4]
regimes: [both]
updated: 2026-08-12
sources:
   - https://www.amd.com/en/products/graphics/amd-radeon-rx-6800-xt
---

# RDNA2 / RX 6000 Series (gfx1030) - architecture overview

> Target: **AMD Radeon RX 6000 series (gfx1030/gfx1031/gfx1032)**, RDNA2, ISA **gfx1030**  
> Consumer/gaming AI inference GPU, NOT to be confused with datacenter CDNA2 (MI250X).  
> Wave64 frontend (two Wave32 waves execute as one logical Wave64). No dedicated matrix cores.

## TL;DR
> RDNA2 is the gaming successor to RDNA1: **80 CUs** (RX 6900 XT), **16 GB GDDR6 @ 512 GB/s**, **Wave64** execution, no AI accelerators, no WMMA. Matrix math is scalar/vector SIMD only. FP16 native (2x FP32 rate), BF16 emulated (no native BF16 ALU) — avoid BF16 on RDNA2. FP8 has no hardware support. Peaks: FP32 **23.0 TFLOPS** (RX 6900 XT), FP16 **46.0 TFLOPS**.

## The one-screen cheat sheet
| Fact | Value (RX 6900 XT) | Why it matters |
|---|---|---|
| Wavefront | **64 lanes** (Wave64) | 2x Wave32 lanes, halves occupancy vs Wave32 |
| CUs (active) | **80** | matrix via SIMD only, no dedicated cores |
| SIMD/CU | **4** | occupancy per-SIMD |
| Wave slots | 8/SIMD = 32/CU | hard cap |
| VGPR | 512 x4 B/SIMD | per-SIMD register file |
| LDS | **64 KiB/CU**, 32 banks | half of RDNA3/4 |
| L2 | **4 MB**, 4 shader engines | small; memory-bound kernels hurt |
| Infinity Cache | **128 MiB** (RX 6900 XT) | mitigates 512 GB/s GDDR6 |
| GDDR6 | **16 GB**, **512 GB/s bus** (256-bit) | bandwidth is the ceiling for LLM decode |
| Peak FP32 | **23.0 TF** | vector throughput |
| Peak FP16 | **46.0 TF** (2x FP32) | native FP16 ALU |
| BF16 | **emulated** | no native BF16 — downcast cost |
| FP8 | **none** | unsupported, stay INT8/FP16 |
| WMMA | **none** | no matrix unit; no WMMA kernels |
| AI Accelerators | **0** | FSR4/FP8 path unavailable |
| FP8 variant | **n/a** | no FP8 path |
| Process | TSMC **7 nm** | older node, higher power per flop |
| TDP | **300 W** | |
| Engine clock | up to ~2250 MHz | basis of peak math |

## Key differences from RDNA4 (RX 9070 XT)
- **Wave64, not Wave32**: RDNA2 executes Wave64; RDNA4 is Wave32-native with 64-lane subgroups. Kernel block sizing tuned for RDNA2 uses 64-lane wave groups — see `KERNEL_DB_RDNA2` in `tools/analyze.py` (MMVQ block_y=4).
- **No WMMA, no FP8, no AI accelerators**: RDNA2 has zero matrix hardware. All GEMM work is vector SIMD. Do not propose WMMA/FP8/TMA kernels for gfx1030.
- **BF16 is emulated**: prefer FP16 or INT8 on RDNA2. BF16 kernels run through FP32 emulation — ~2-4x slower than FP16.
- **Smaller L2 (4 MB vs 8 MB)**: MMQ kernels are more L2-bound than on RDNA4.
- **ROCm**: Windows ROCm support is absent for consumer RX 6000; use Linux ROCm 6.x, or HIP on Windows via vendor builds. Check `windows_deployment.md`.

## The common RDNA2 SAD trap (memory-bound decode)
> LLM decode on RX 6000 is **bandwidth-bound** — 512 GB/s GDDR6 is the hard ceiling. A token-generation loop that fits in 4 MB L2 gets ~2x the effective bandwidth of streaming from GDDR6. For small models (<8B q4) keep weights in L2 (ring-buffer the KV cache), for larger models accept GDDR6-bound decode and do not over-optimize SIMD math that is idle waiting on memory.

## WMMA
RDNA2 has **no WMMA instructions**. See `wmma.md` (inherited placeholder documents why — all WMMA content is RDNA3+/not applicable).

## FP8
RDNA2 has **no FP8 hardware**. See `fp8_wmma.md` (placeholder; FP8 applies only to RDNA4/CDNA4).

## LDS
64 KiB/CU, 32 banks — see `lds.md`.

## Windows deployment
RDNA2 consumer cards have **no official Windows ROCm**; see `windows_deployment.md` for Linux ROCm 6.x instructions and Windows HIP caveats.

## Peak tables
See `peak_tables.md`.
