---
title: RDNA2 / RX 6000 LDS notes (gfx1030)
kind: hardware
gens: [gfx1030, gfx1031, gfx1032]
updated: 2026-08-12
---

# RDNA2 LDS

- **64 KiB/CU**, **32 banks** (RDNA3/4 doubled to 128 KiB/32 and 128 KiB with more banks).
- LDS throughput: 32 banks x 4 B = 128 B/cycle read. Bank conflicts on 32-bit strides common.
- Wave64 executes as two Wave32 halves; LDS access is per-half, conflicts resolve per half.
- Practical shared-memory budget for kernels: **48-56 KiB** (leave room for compiler/scratch + swizzle).
- RDNA2 has no LDS-based thread-group matrix load special-casing (no WMMA); keep LDS for staging/transpose only.

## MMQ / MMVQ implications
- RDNA2 `KERNEL_DB_RDNA2` (tools/analyze.py) expects MMVQ with `block_y=4` (Wave64 prefers fewer threads), MMQ with ≥100 KiB smem for the LDS-resident path.
- With 64 KiB/CU LDS, two MMQ blocks per CU is tight — prefer one block + larger tile, or spill to L2 (4 MB).
