---
title: RDNA2 / RX 6000 peak performance tables (gfx1030)
kind: hardware
gens: [gfx1030, gfx1031, gfx1032]
updated: 2026-08-12
---

# RDNA2 peak performance tables

> Vectors-only: no matrix units. Peak = SIMD FP32/FP16/INT throughput.
> Per-CU FP32 = 4 SIMD x 32 lanes x 2 (FMA) x clock.

## RX 6000 family (gfx1030/gfx1031/gfx1032)

| SKU | CUs | Boost MHz | FP32 TF | FP16 TF | INT8 TOPS | GDDR6 GB | BW GB/s | L2 MB | TDP W |
|---|---|---|---|---|---|---|---|---|---|
| RX 6900 XT | 80 | 2250 | 23.0 | 46.0 | 92.0 | 16 | 512 | 4 | 300 |
| RX 6800 XT | 72 | 2250 | 20.7 | 41.5 | 83.0 | 16 | 512 | 4 | 300 |
| RX 6800 | 60 | 2105 | 16.2 | 32.3 | 64.6 | 16 | 512 | 4 | 250 |
| RX 6750 XT | 40 | 2600 | 13.3 | 26.6 | 53.2 | 12 | 432 | 3 | 250 |
| RX 6700 XT | 40 | 2424 | 12.4 | 24.8 | 49.6 | 12 | 384 | 3 | 230 |
| RX 6650 XT | 32 | 2635 | 10.8 | 21.6 | 43.2 | 8 | 280 | 3 | 180 |
| RX 6600 XT | 32 | 2359 | 9.6 | 19.3 | 38.6 | 8 | 256 | 3 | 160 |
| RX 6600 | 28 | 2044 | 8.9 | 17.8 | 35.6 | 8 | 224 | 3 | 132 |

## Notes
- FP16 = 2x FP32 (packed 2xFP16 vector). INT8 = 2x FP16 on packed int8x32.
- BF16: **emulated** via FP32 — no native BF16 ALU on RDNA2. Treat BF16 kernels as ~FP32-cost.
- FP8: none.
- L2 is per-shader-engine (4 SEs); effective usable ~3-4 MB.
- LLM decode ceiling is **GDDR6 bandwidth**, not math: at ~4.0 bit/weight (q4), a 7B model streams ~3.5 GB per token → ~0.7 tok/s on RX 6800 XT (512 GB/s) unless L2-cached.

## Roofline per SKU (decode, q4, L2-resident vs GDDR6-streaming)
| SKU | BW GB/s | q4 tok/s (GDDR6-bound) | q4 tok/s (L2-resident, ~2x) |
|---|---|---|---|
| RX 6900 XT | 512 | ~0.7 | ~1.4 |
| RX 6800 XT | 512 | ~0.7 | ~1.4 |
| RX 6700 XT | 384 | ~0.5 | ~1.0 |
| RX 6600 XT | 256 | ~0.35 | ~0.7 |
