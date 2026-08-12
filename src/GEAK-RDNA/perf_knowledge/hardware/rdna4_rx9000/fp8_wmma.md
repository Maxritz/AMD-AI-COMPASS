---
title: RDNA4 FP8 WMMA — Native FP8 Matrix Computations
kind: hardware
gens: [gfx1201, gfx1203, gfx1206, gfx1207]
dtypes: [fp8_e4m3, fp8_e5m2]
regimes: [inference]
updated: 2026-07-25
related:
  - arch.md
  - wmma.md
---

# RDNA4 FP8 WMMA — Native FP8 Matrix Computations

> RDNA4 introduces **native FP8 WMMA** for FSR4 and inference acceleration. This is distinct from CDNA4's OCP FP8 which runs on dedicated matrix cores.

## FP8 Formats Supported

RDNA4 WMMA supports **OCP-standard FP8** (not FNUZ):
- **FP8 E4M3FN** — exponent biased 127, 4 exponent + 3 mantissa + 1 sign
- **FP8 E5M2** — exponent biased 130, 5 exponent + 2 mantissa + 1 sign

Both formats are industry standard (OCP FP8) for AI inference, used by FSR4 upscaling and now being patched into vLLM/LLAMA-DX for inference.

## WMMA FP8 Shapes

| Shape | Use Case | Throughput |
|---|---|---|
| 16×16×128 FP8 | GEMM, attention QKV | 380 TFLOPS peak |
| 32×32×64 FP8 | Convolution, dense layers | 760 GFLOPS/frame |
| 16×16×32 FP8 | Activation, residual | Fallback FP16 path |

## vLLM/LLAMA-DX Integration Status

Upstream vLLM **does not yet support gfx1201 FP8 WMMA**. Users are manually:
1. Adding `gfx1201` to device detector
2. Disabling `AITER` (which blocks gfx1201)
3. Injecting tuned Triton configs
4. Applying local patches to `src/experimental/region_handlers/hip_wmma_fp8.py`

### Required Patches (not upstreamed)

```diff
+ // arch.h snippet
+ case AMDGPU_ISA_GFX1201:
+   target_triple = "amdgcn-amd-amdhsa";
+   mcpu = "gfx1201";
+   enable_fp8_wmma = true;
```

### Triton Kernel Override Pattern

```python
# triton_wmma_fp8_gemm.py — patched for gfx1201
@triton.jit
def fp8_wmma_gemm_kernel(...):
    # gfx1201-specific tile sizing
    if DEVICE_PROP.is_gfx1201():
        TILE_M, TILE_N, TILE_K = 16, 16, 128
    else:
        TILE_M, TILE_N, TILE_K = 16, 16, 32  # fallback
```

## FP8 Quantization Strategy

**Symmetric per-tensor quantization** recommended:
```
scale = 127.0 / max(abs(x))
qfp8 = clamp(round(x * scale), -127, 127).to(int8).to(fp8)
```

For weight-only quantization (QLoRA-style):
- Use `fp8_e4m3` for weights
- Keep activations in FP16 or E5M2
- Scales stored in FP16 per-channel

## Kernel Template for RDNA4 FP8 GEMM

```hip
// wmma_fp8_gemm_gfx1201.hip
#include <hip/hip_runtime.h>
#include <hip/amdhip64.h>

template<int M, int N, int K>
__global__ void fp8_wmma_gemm_kernel(
    const __fp8_e* __restrict__ A,
    const __fp8_e* __restrict__ B,
    float* __restrict__ C,
    float* __restrict__ scale_a,
    float* __restrict__ scale_b,
    int m, int n, int k
) {
#if __AMDGPU_ARCH__ >= 1200
    // WMMA FP8 path
    wmma::fragment<wmma::matrix_a, 16, 16, 128, __fp8_e, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 128, __fp8_e, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 128, float> c_frag;
    
    // Load and compute (pseudocode)
    wmma::load_matrix_sync(a_frag, A + ... , scale_a);
    wmma::load_matrix_sync(b_frag, B + ..., scale_b);
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(C, c_frag, ... , wmma::mem_row_major);
#else
    // FP16 fallback
#endif
}
```

## Performance Expectations

On RX 9070 XT (gfx1201):
- Small models (7B-14B): **2.5-3.5×** speedup via FP8
- Large models (30B+): **bottlenecked by 640 GB/s GDDR6** bandwidth
- Compare to CDNA4 FP8: **lower throughput** due to shared ALU execution

## FP8 vs INT4 Sparsity

| Metric | FP8 WMMA | INT4 Sparsity |
|---|---|---|
| Peak throughput | ~380 TFLOPS | ~1557 AI TOPS (sparsity) |
| Accuracy | Near FP16 quality | Requires 2:1 weight sparsity |
| Kernel complexity | Simple FP8 math | Requires sparse index handling |
| Best use case | Mixed-weight FP8 | Heavy-weight INT4 with sparsity |