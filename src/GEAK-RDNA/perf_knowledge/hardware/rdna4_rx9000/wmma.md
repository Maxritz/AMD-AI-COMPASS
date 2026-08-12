---
title: RDNA4 WMMA — Wavefront Matrix Multiply-Accumulate
kind: hardware
gens: [gfx1201, gfx1203, gfx1206, gfx1207]
dtypes: [fp16, bf16, fp8_e4m3, fp8_e5m2, int4]
regimes: [both]
updated: 2026-07-25
related:
  - arch.md
  - ../matrix_core.md
  - fp8_wmma.md
---

# RDNA4 WMMA — Wavefront Matrix Multiply-Accumulate

> RDNA4 replaces CDNA's dedicated matrix cores (`mfma`) with **WMMA** (Wavefront Matrix Multiply-Accumulate). WMMA shares ALU resources with scalar/vector units, fundamentally unlike CDNA's hardware matrix cores.

## Key Distinctions from CDNA4 Matrix Cores

| Aspect | CDNA4 Matrix Cores | RDNA4 WMMA |
|---|---|---|
| Execution | Dedicated MFMA units (separate from SIMD/ALU) | WMMA dispatched through wavefront ALUs |
| Instructions | `v_mfma_*` | `s_wmma_*` / `v_wmma_*` |
| Utilization | Can saturate at GPU clock | Shares ALU bandwidth, lower peak math |
| TF32 | Removed | Not applicable (no dedicated matrix) |
| FP6/FP4 | Hardware MFMA support | Software emulation or INT4 path |

## WMMA Instruction Reference (gfx12xx ISA)

### Instruction Format
```asm
s_wmma_accumulate vdst, va, vb, vc        ; accumulate into vector register
s_wmma_load_a vdst, vsrc, tag, hints      ; load fragment A
s_wmma_load_b vdst, vsrc, tag, hints      ; load fragment B
s_wmma_fill vdst, value, tag, hints       ; fill fragment with scalar
```

### Supported Shapes

#### FP16 / BF16
- **16×16×32** (M×N×K) — one wavefront tile
- **32×32×32** — two wavefronts cooperating

#### FP8 (OCP: E4M3FN/E5M2)
- **16×16×128** (M×N×K) — optimized for inference
- **32×32×64**
- Requires `-mcpu gfx1201` or later

#### INT4
- **Through integer matrix ops** (not true WMMA, uses INT4 instructions with sparsity)
- `v_mul_lo_u32`, `v_dp4a` variant for INT4×INT4→INT32

## Triton Implementation Notes

RDNA4 WMMA support in Triton requires:
1. `-DTRITON_DISABLE_AITER=ON` (AITER's FP8 path blocks gfx1201)
2. Custom `triton_optimize.py` for gfx1201 tile sizing
3. Local patches to `lib/Target/AMDGPU/` for WMMA intrinsics

Example kernel attribute override:
```python
# In Triton kernel:
@triton.jit
def wmma_gemm(...):
    # gfx1201 WMMA shapes
    tile_m, tile_n, tile_k = 16, 16, 128  # FP8
```

## Kernel Porting from CDNA4

To port CDNA4 kernels using `mfma` to RDNA4 WMMA:

1. Replace `v_mfma_f32_f16_16x16x32` → `s_wmma_accumulate` with correct semantics
2. Update fragment loading: `v_mfma_load` → `s_wmma_load_a/b`
3. Adjust LDS bank padding (32 banks vs 64)
4. Use INT4 path for sparsity-optimized INT4 (vs MFMA FP4/FP6 on CDNA4)

## Performance Considerations

- WMMA shares issue bandwidth with other wavefront instructions
- Optimal tile sizes: 16×16×128 for FP8, larger tiles may stall
- LDS must be 32-byte aligned for optimal `s_wmma_load`
- VGPR pressure higher: limit fragments to fit 512 VGPR/SIMD limit

## Windows 11 HIP-ROCm 7.3 Usage

Compile with:
```bash
hipcc -target amdgcn-amd-amdhsa -mcpu gfx1201 \
  -D__WMMA_ENABLE__ -I/path/to/wmma/include \
  kernel.hip -o kernel_hip
```

Runtime detection:
```cpp
#if defined(__AMDGPU_ARCH__) && __AMDGPU_ARCH__ >= 1200
  // Use WMMA path
#else
  // Fallback to matmul via MIA
#endif
```