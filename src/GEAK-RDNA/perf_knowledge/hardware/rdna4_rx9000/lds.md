---
title: RDNA4 LDS — Local Data Share Configuration
kind: hardware
gens: [gfx1201, gfx1203, gfx1206, gfx1207]
dtypes: [bf16, fp16, fp8]
regimes: [inference]
updated: 2026-07-25
hardware: rdna4_rx9000
---

# RDNA4 LDS — Local Data Share Configuration

## LDS Capacity

| Generation | LDS/CU | Banks | Read BW | Write BW |
|---|---|---|---|---|
| RDNA4 | **128 KiB** | **32** | 256 B/clk | 256 B/clk |
| CDNA4 | **160 KiB** | 64 | 512 B/clk | 512 B/clk |
| RDNA3 | 128 KiB | 64 | 512 B/clk | 512 B/clk |

## Key Differences from RDNA3/CDNA4

### Bank Count Change
RDNA4 reduces LDS banks from 64 to **32**, affecting bank conflict patterns:
- **RDNA3/CDNA4**: 64 banks, 1 bank per 128 bytes
- **RDNA4**: 32 banks, 1 bank per 256 bytes (different bank indexing)

Kernel rewrite may be needed for optimal bank padding.

### Read Bandwidth
RDNA4 maintains 256 B/clk read bandwidth (same as RDNA3), but:
- CDNA4 has 512 B/clk due to per-XCD L2 with higher bandwidth
- RDNA4 unified L2 provides different cache hierarchy

## LDS Usage Guidelines

### Optimal Tile Sizes for FP8 WMMA

| Tile | LDS Usage | Notes |
|---|---|---|
| 16×16×128 FP8 | ~8 KB | 1 tile A + 1 tile B |
| 32×32×64 FP8 | ~32 KB | 4 tiles A + 4 tiles B |
| 64×64×32 FP16 | ~64 KB | Near LDS limit per CU |

### Bank Conflict Avoidance

For 32 LDS banks:
```c
// Use stride of 32 or 64 for aligned access
__ldiv__ __syncthreads();  // Ensure warp synchronization

// Example: matrix A with stride padding
float4* lds_mat_a = shared_mem;
int row = lane_id / 32;
int col = lane_id % 32;
float4 val = lds_mat_a[row * 64 + col];  // 64 = 2x banks for padding
```

### LDS Bank Mapping (gfx12xx)

| Bank Bits | Address Offset | Maps To |
|---|---|---|
| [7:5] | bits 5-7 | Bank select (0-31) |
| [4:0] | bits 0-4 | Address within bank |

To avoid bank conflicts, ensure different threads access different bank+address combinations.

## LDS Tiling Strategy for RDNA4

### FP8 WMMA (16×16×128)
```
LDS Layout:
- A: [k][m] float8 (128×16 bytes = 2KB)
- B: [n][k] float8 (16×128 bytes = 2KB)
- C: [n][m] float (16×16×4 bytes = 1KB)
Total: ~5KB per tile
```

### BF16 GEMM (64×64×32)
```
LDS Layout:
- A tile: 64×64×2 = 8KB
- B tile: 32×64×2 = 4KB
- C accum: 64×64×4 = 16KB
Total: ~28KB (leaves 100KB for buffering)
```

## LDS Limits by Workload

| Workload | LDS Usage | Occupancy |
|---|---|---|
| 7B FP16 GEMM | ~128 KB | 100% |
| 30B FP8 WMMA | ~5 KB | 100% |
| Attention QKV | ~64 KB | 100% |
| Rotary Emb + QKV | ~80 KB | 100% |

## Programming Notes for HIP-ROCm 7.3 Windows

```hip
// LDS allocation in HIP
__shared__ half storage[128 * 1024 / 2];  // 128 KB for FP16

// Bank conflict example (avoid):
// Thread 0: lds[0]
// Thread 32: lds[32] -> different bank
// Thread 1: lds[1] -> different bank

// Bank conflict example (BAD):
// Thread 0: lds[0]
// Thread 1: lds[1] -> different bank, OK
// Thread 32: lds[32] -> SAME BANK as thread 0 (bank 0) + CLASH
```