---
title: RDNA2 / RX 6000 WMMA status (gfx1030)
kind: hardware
gens: [gfx1030, gfx1031, gfx1032]
updated: 2026-08-12
---

# WMMA on RDNA2

**Not available.** RDNA2 has no matrix instructions (no WMMA, no MFMA-like matrix cores).

- All GEMM is vector SIMD: dot products via packed FP16/int8 vector FMA.
- Peak math is the FP32/FP16 SIMD table in `peak_tables.md`, NOT matrix TFLOPS.
- Optimization levers on RDNA2 GEMM: register tiling, LDS staging, vectorization (packed FP16x2 / int8x8), L2 reuse.
- WMMA-content from RDNA3/4 knowledge does not apply — ignore `wmma.md` hints for gfx1030.
