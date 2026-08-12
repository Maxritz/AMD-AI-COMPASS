---
title: RDNA4 Peak Tables — FP16, BF16, FP8, INT4 Performance Numbers
kind: hardware
gens: [gfx1201, gfx1203, gfx1206, gfx1207]
dtypes: [fp16, bf16, fp8_e4m3, fp8_e5m2, int4]
regimes: [inference]
updated: 2026-07-25
hardware: rdna4_rx9000
---

# RDNA4 Peak Performance Tables

> Peak performance numbers for RDNA4 RX 9070 XT (gfx1201). Values assume ideal conditions and single-precision scaling. Real sustained performance is typically 70-85% of peak.

## RDNA4 RX 9070 XT (gfx1201) — Peak Metrics

### Compute Peaks

| Data Type | Peak FP32 | Peak TF32 | Peak FP16 | Peak BF16 | Peak FP8 | Peak INT4 (sparse) |
|---|---|---|---|---|---|---|
| Matrix ops | 146 GFLOPS | N/A | 194.8 TFLOPS | 194.8 TFLOPS | 389 TFLOPS | **1,557 AI TOPS** |
| Vector ops | 146 GFLOPS | N/A | 2.9 TFLOPS | 2.9 TFLOPS | N/A | 11.7 TOPS |

### Notes
- **FP32 peak**: 146 GFLOPS base clock rate
- **FP16/BF16 peak**: 194.8 TFLOPS via WMMA (16×16×32 tiles at 3000 MHz)
- **FP8 peak**: 389 TFLOPS (OCP E4M3/E5M2)
- **INT4 sparsity peak**: 1,557 AI TOPS with 2:1 weight sparsity
- **TF32**: Not supported (removed from gfx12xx)

## RDNA4 vs CDNA4 Comparison

| Metric | RDNA4 RX 9070 XT | CDNA4 MI355X | Ratio |
|---|---|---|---|
| CU Count | 64 | 256 | 1:4 |
| Matrix Peak FP16 | 195 TFLOPS | 2.5 PFLOPS | 1:12.8 |
| Matrix Peak FP8 | 389 TFLOPS | 5 PFLOPS | 1:12.8 |
| Memory | 16 GB GDDR6 | 288 GB HBM3E | 1:18 |
| Bandwidth | 640 GB/s | 8 TB/s | 1:12.5 |
| TDP | 300 W | 1400 W | 1:4.7 |

## RDNA4 GPU Variants

### RX 9070 XT (gfx1201)
| Spec | Value |
|---|---|
| Compute Units | 64 |
| Wavefront Lanes | 64 |
| VGPR/File | 512 × 4 B/SIMD |
| LDS | 128 KiB/CU, 32 banks |
| L2 Cache | 8 MB unified |
| Infinity Cache | 80 MiB |
| Memory | 16 GB GDDR6 |
| Bandwidth | 640 GB/s (256-bit) |
| AI Accelerators | 32 (2nd-gen) |
| TDP | 300 W |
| Peak FP16 | 194.8 TFLOPS |
| Peak FP8 WMMA | 389 TFLOPS |

### RX 9070 (gfx1207)
| Spec | Value |
|---|---|
| Compute Units | 54 |
| Peak FP16 | 164 TFLOPS |
| Peak FP8 WMMA | 326 TFLOPS |

### RX 9060 XT (gfx1206)
| Spec | Value |
|---|---|
| Compute Units | 40 |
| Peak FP16 | 121 TFLOPS |
| Peak FP8 WMMA | 241 TFLOPS |

### RX 9000 XT (gfx1203) - Mobile
| Spec | Value |
|---|---|
| Compute Units | varies by SKU |
| TDP | 115-175 W (laptop) |
| Peak FP16 | varies |

## Sustained Performance Estimates

Based on memory-bound workloads (LLM inference):

| Model Size | Approx. Memory | Expected BW Util | Effective FP16 | Est. Tokens/sec |
|---|---|---|---|---|
| 7B | 14 GB | 70% | 136 TFLOPS | ~180 |
| 14B | 28 GB | 50% | 97 TFLOPS | ~110 |
| 30B | 60 GB | 35% | 68 TFLOPS | ~45 |
| 70B | 140 GB | 15% | 29 TFLOPS | ~10 |

## Memory Layout for LLM Inference

| Data | Bytes | Notes |
|---|---|---|
| FP16 weights | 2× params | 7B = 14 GB, 30B = 60 GB |
| FP8 weights | 1× params | Half memory vs FP16 |
| KV Cache | 2× seq_len×hidden×2 | Attention bottleneck |
| Activations | 1-2× weights | For BF16/FP8 intermediate |

## RDNA4 WMMA Memory Requirements

For WMMA FP8 16×16×128 tiles:
- LDS: 128 KiB per CU → ~8192 bytes per tile
- VGPR: 64-128 per lane (512 max total)
- L2: 8 MB can hold ~10% of 30B model weights