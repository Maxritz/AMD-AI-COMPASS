# Vulkan LLM Inference Engine — Compute Shader Specification
## Target: AMD RDNA4 (gfx1201, RX 9070 XT) + RDNA2 (gfx1031, RX 6700 XT)
### Vulkan SDK 1.4.357.0 — C:\VulkanSDK\1.4.357.0

---

## 0. Global Shader Conventions

### 0.1 Extension & Feature Requirements (Every Shader)

```glsl
#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int8    : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16   : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float32 : require
#extension GL_KHR_shader_subgroup_basic                    : require
#extension GL_KHR_shader_subgroup_arithmetic               : require
#extension GL_KHR_shader_subgroup_shuffle                  : require
#extension GL_KHR_shader_subgroup_ballot                   : require
#extension GL_EXT_subgroup_size_control                    : require
#extension GL_EXT_nonuniform_qualifier                     : require
#extension GL_EXT_scalar_block_layout                      : require
// Conditional (quantized shaders only):
#extension GL_KHR_shader_integer_dot_product               : require
#extension GL_KHR_cooperative_matrix                       : require
// When 16-bit storage needed:
#extension GL_EXT_shader_16bit_storage                     : require
// When 8-bit storage needed:
#extension GL_EXT_shader_8bit_storage                      : require
#extension GL_KHR_memory_scope_semantics                   : require
```

### 0.2 Dual-Compilation Strategy

Every shader is compiled **twice** into separate SPIR-V modules:

| Variant | `SUBGROUP_SIZE` | WG Limits | Target Architecture |
|---------|----------------|-----------|---------------------|
| `_rdna4` | 32 | WG ≤ 256, 8 wavefronts/WG max | gfx1201 (RX 9070 XT) |
| `_rdna2` | 64 | WG ≤ 256, 4 wavefronts/WG max | gfx1031 (RX 6700 XT) |

```cmake
# SPD compile commands:
# RDNA4: glslc -fshader-stage=compute --target-spv=spv1.4 -DSUBGROUP_SIZE=32 ...
# RDNA2: glslc -fshader-stage=compute --target-spv=spv1.4 -DSUBGROUP_SIZE=64 ...
```

Pipeline creation uses `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` with `requiredSubgroupSize = 32` or `64`, plus `VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT`.

### 0.3 Specialization Constants (Baked at Pipeline Creation)

```glsl
layout(constant_id = 0) const uint SUBGROUP_SIZE     = 32;  // set by SPD
layout(constant_id = 1) const uint WG_SIZE_X          = 64;  // set by SPD
layout(constant_id = 2) const uint D                  = 4096;
layout(constant_id = 3) const uint FFN_DIM            = 11008;
layout(constant_id = 4) const uint HEAD_DIM           = 128;
layout(constant_id = 5) const uint N_HEADS            = 32;
layout(constant_id = 6) const uint N_KV_HEADS         = 8;
layout(constant_id = 7) const uint VOCAB_SIZE         = 128000;
layout(constant_id = 8) const uint KV_PAGE_TOKENS     = 256;
layout(constant_id = 9) const uint PAGES_PER_LAYER    = 16;
layout(constant_id = 10) const uint QUANT_TYPE        = 0; // 0=FP16,1=Q8_0,2=Q4_K,3=Q6_K,4=IQ4_XS
layout(constant_id = 11) const uint WARP_PER_WG       = 2;  // WG_SIZE_X / SUBGROUP_SIZE
```

All model-dimension values are baked via specialization constants — zero runtime branches for dimension checks.

### 0.4 Descriptor Set Layout (Shared by All Shaders)

```
Set 0 — Static Weights (UPDATE_AFTER_BIND, PARTIALLY_BOUND, DESCRIPTOR_INDEXING)
  binding=0: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, count=MAX_LAYERS
    Element type: WeightBlock (see per-shader layout below)
    Accessed via: weight_array[nonuniformEXT(layer_idx)]

Set 1 — Layer I/O (Push Descriptors via vkCmdPushDescriptorSetKHR)
  binding=0: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER  — hidden_in (read)
  binding=1: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER  — hidden_out / scratch_out (write)
  binding=2: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER  — k_cache (read/write)
  binding=3: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER  — v_cache (read/write)
  binding=4: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER  — scratch (read/write, generic)

Set 2 — Static Tables (allocated once, never updated)
  binding=0: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER  — rope_freqs (precomputed cos/sin)
  binding=1: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER  — kv_page_table (uint32_t array)
```

### 0.5 Push Constant Layout (Common)

```glsl
layout(push_constant) uniform PushConstants {
    uint  layer_idx;          // current layer 0..31
    uint  kv_cache_pos;       // current decode token position
    uint  seq_len;            // total sequence length so far
    uint  n_heads;            // runtime override (0 = use spec const)
    uint  n_kv_heads;         // runtime override
    float attn_scale;         // 1/sqrt(head_dim), precomputed
    float rope_theta;         // RoPE base frequency
    float norm_eps;           // RMS norm epsilon (1e-6)
    uint  reserved[13];       // padding to 80 bytes total (safe for 256 limit)
} pc;
// Total: 80 bytes (well within 256-byte limit)
```

### 0.6 GLSL Common Functions (Included via `#include`)

```glsl
// --- common.glsl ---

#ifndef SUBGROUP_SIZE
#error "SUBGROUP_SIZE must be defined at compile time"
#endif

#define WARP_SIZE SUBGROUP_SIZE

// Subgroup reduction helpers — these compile to single-cycle shuffle on RDNA
float warpReduceSum(float val) {
    for (uint offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val += subgroupShuffleXor(val, offset);
    }
    return val;
}

float warpReduceMax(float val) {
    for (uint offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        val = max(val, subgroupShuffleXor(val, offset));
    }
    return val;
}

// FP16 → FP32 helper (explicit arithmetic types)
float f16tof32(float16_t v) { return float(v); }
float16_t f32tof16(float v) { return float16_t(v); }

// Coalesced global memory load — vec4 ensures 16-byte transactions on RDNA
float16_t4 loadVec4F16(restrict readonly float16_t src[4]) {
    return float16_t4(src[0], src[1], src[2], src[3]);
}

void storeVec4F16(restrict float16_t dst[4], float16_t4 v) {
    dst[0] = v.x; dst[1] = v.y; dst[2] = v.z; dst[3] = v.w;
}

// ========== Quantization Dequant Functions ==========

// Q8_0: block_size=32, d(fp16) + 32×q8
// dequant: weight[i] = d * qs[i]
float dequant_q8_0(uint block_start, uint elem_idx) {
    uint byte_off = block_start + elem_idx + 2; // +2 skip d
    float d = f16tof32(weight_block[block_start]);
    return d * float(int(weight_block[byte_off]));
}

// Q4_K: superblock=256, subblock=16
// Layout: 2×fp16(d,dmin) + 16×2-bit sc + 16×6-bit sc_min + 256×4bit qs
// d = weight_block[0..1], dmin = weight_block[2..3]
// sc[16] packed in 4 bytes (2 bits each)
// sc_min[16] packed in 12 bytes (6 bits each)
// qs[256/2] = 128 bytes (4 bits each)
float dequant_q4_k(uint block_start, uint elem_idx) {
    float d  = f16tof32(float16_t(weight_block[block_start], weight_block[block_start+1]));
    float dm = f16tof32(float16_t(weight_block[block_start+2], weight_block[block_start+3]));
    uint sub = elem_idx / 16;
    uint q_off = elem_idx / 2;
    uint q_byte = weight_block[block_start + 28 + q_off];
    uint q4 = (elem_idx & 1) != 0 ? (q_byte >> 4) : (q_byte & 0x0F);
    // 2-bit scale: sc = (sc_packed[sub/4] >> (2*(sub%4))) & 3
    uint sc_bits = weight_block[block_start + 4 + sub/4];
    uint sc_2bit = (sc_bits >> (2 * (sub % 4))) & 0x03;
    // 6-bit scale min: sc_min packed 12 bytes for 16 values
    uint bit_off = sub * 6;
    uint byte_idx = bit_off / 8;
    uint bit_shift = bit_off % 8;
    uint sc_min = 0;
    if (bit_shift <= 2) {
        sc_min = (weight_block[block_start + 8 + byte_idx] >> bit_shift);
    } else {
        sc_min = (weight_block[block_start + 8 + byte_idx] >> bit_shift)
               | (weight_block[block_start + 8 + byte_idx + 1] << (8 - bit_shift));
    }
    sc_min &= 0x3F;
    float scale = d * float(sc_2bit);
    float m = dm * float(int(sc_min) - 32);
    return scale * float(int(q4)) + m;
}

// FP16 — direct read, no dequant
float dequant_f16(uint offset) {
    return f16tof32(weight_block[offset]);
}
```

### 0.7 Cooperative Matrix Declaration (Quantized GEMM Shaders)

```glsl
// RDNA4 (Wave32): 16×16×16 blocks
// RDNA2 (Wave64): not available, fall back to dp4a
#if SUBGROUP_SIZE == 32 && defined(COOPMAT_ENABLED)
layout(constant_id = 100) const uint COOPMAT_M = 16;
layout(constant_id = 101) const uint COOPMAT_N = 16;
layout(constant_id = 102) const uint COOPMAT_K = 16;

coopmat<float16_t, gl_ScopeSubgroup, COOPMAT_M, COOPMAT_N, COOPMAT_K> coopAccum(float16_t a, float16_t b) {
    coopmat<float16_t, gl_ScopeSubgroup, COOPMAT_M, COOPMAT_N, COOPMAT_K> result;
    // ... VK_KHR_cooperative_matrix operations
    return result;
}
#endif
```

### 0.8 RDNA4 GPU Occupancy Reference Table

| WG Size | Subgroups/WG | Wave Slots Used | Max Concurrent WGs (32 CU) | Occupancy |
|---------|-------------|-----------------|---------------------------|-----------|
| 32 | 1 | 1 | 1024 | 100% (compute-bound) |
| 64 | 2 | 2 | 512 | 100% |
| 128 | 4 | 4 | 256 | 100% |
| 256 | 8 | 8 | 128 | 100% |

Key insight: RDNA4 has 32 CUs × 4 SIMD × 16 wave slots = **2,048 wave slots** total. But with Wave32, we have 1,024 SIMD lanes per CU, so effectively 1,024 wavefront slots. Actually:

RDNA4 architecture: 32 CUs, each CU has 2 WGPs. Each WGP has 4 SIMD32 units (2 compute units per WGP), each can run 16 wavefronts. So 32 CUs × 2 WGPs/CU × 4 SIMD/WGP × 16 waves/SIMD = **4,096 wave slots** in Wave32 mode. In Wave64 mode, it's 2,048 wave slots.

For RDNA2: 40 CUs, Wave64-only, 40 × 4 SIMD × 16 waves = **2,560 wave slots**. But in Wave64 mode each SIMD64 actually runs a VGPR allocator, and the wave slot count is per-SIMD32-like if wavefront is treated as 64-wide... Let me use the standard formula:

```
RDNA4 Wave32: 32 CUs × 2 WGPs × 2 SIMD32 × 16 waves = 2,048 wave slots
RDNA4 Wave64: 32 CUs × 2 WGPs × 1 SIMD64 × 16 waves = 1,024 wave slots
RDNA2 Wave64: 40 CUs × 2 WGPs × 1 SIMD64 × 16 waves = 1,280 wave slots
```

### 0.9 VGPR Budget

| Architecture | VGPRs/Thread | Max Active Wavefronts/CU |
|-------------|-------------|-------------------------|
| RDNA4 Wave32 | 128 VGPR total/CU, 32 per SIMD32 | 4 waves/SIMD at 32 VGPRs each |
| RDNA2 Wave64 | 128 VGPR total/CU, 64 per SIMD64 | 2 waves/CU at 64 VGPRs each |

Target VGPR usage: ≤ 64 per thread. At 64 VGPRs, RDNA4 Wave32 can run 2 waves/SIMD32 = 4 waves/WGP = 8 waves/CU. With Wave64, 64 VGPRs allows 1 wave/CU — occupancy collapses. This is why Wave32 dominates on RDNA4.

---

## 1. rms_norm.comp — RMS Normalization + Residual Add

### 1.1 Purpose
```
hidden_out = rms_norm_weight * (hidden_out / rms(hidden_out + hidden_in)) + hidden_in
```
When `hidden_in == hidden_out` (same buffer), this is in-place RMS norm with residual:
```
hidden = hidden + rms_weight * (hidden / rms(hidden))
```
When `hidden_in != hidden_out`: RMS-norm hidden_in, residue-add onto hidden_out.

### 1.2 Workgroup Configuration

| Parameter | RDNA4 | RDNA2 | Justification |
|-----------|-------|-------|---------------|
| WG size X | 256 | 128 | 256 elements processed per WG. RDNA4 uses 8 wavefronts for max occupancy. RDNA2 uses 2 wavefronts of 64. |
| WG count X | `ceil(D / 256)` or 1 | `ceil(D / 128)` or 1 | For D=4096: 16 WGs (RDNA4) or 32 WGs (RDNA2) |
| WG size Y | 1 | 1 | Batch=1 for decode |
| WG size Z | 1 | 1 | |

**Dispatch count: `ceil(D / WG_SIZE_X)` workgroups in X dimension.** For D=4096:
- RDNA4: 16 workgroups × 256 threads = 4,096 elements covered. Each thread processes 1 element.
- RDNA2: 32 workgroups × 128 threads = 4,096 elements covered.

This is NOT 1 workgroup — we dispatch `ceil(D/256)` workgroups so threads process exactly 1 element each, avoiding thread-local loops. This maximizes occupancy (16 WGs × 8 waves = 128 waves, ~6.25% occupancy on RDNA4 = good for latency hiding).

*Alternative (for very small D < 256)*: Single workgroup of size D, each thread processes 1 element.

### 1.3 Push Constant Usage
```
pc.layer_idx   → which layer's RMS norm weight
pc.norm_eps    → epsilon value
```
Hidden in/out buffers set via push descriptors, not push constants.

### 1.4 Descriptor Bindings Used

| Set | Binding | Type | Access | Buffer |
|-----|---------|------|--------|--------|
| 0 | 0 | Storage Buffer | Read | Weight array: `weight_array[pc.layer_idx].attn_norm` or `.ffn_norm` |
| 1 | 0 | Storage Buffer | Read | `hidden_in` (fp16, D elements) |
| 1 | 1 | Storage Buffer | Read+Write | `hidden_out` (fp16, D elements) — residual add in-place |
| 2 | — | — | — | Not used |

### 1.5 Shared Memory
**0 bytes.** No LDS needed. RMS computation is done purely via subgroup shuffle reduction.

### 1.6 GLSL Structure

```glsl
#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float32 : require
#extension GL_KHR_shader_subgroup_basic                    : require
#extension GL_KHR_shader_subgroup_arithmetic               : require
#extension GL_KHR_shader_subgroup_shuffle                  : require
#extension GL_EXT_subgroup_size_control                    : require
#extension GL_EXT_scalar_block_layout                      : require

#include "common.glsl"
#include "push_constants.glsl"

layout(local_size_x_id = 1, local_size_y = 1, local_size_z = 1) in;
// WG_SIZE_X baked via specialization constant

// Set 0: weight array (variable-size, descriptor-indexed)
layout(set=0, binding=0) readonly buffer WeightArray {
    // Each element: 9 sub-buffers (see architecture doc §2.1)
    // We access attn_norm_weight or ffn_norm_weight at runtime
    float16_t weights[];  // flat array, we index by layer+offset
} weight_bufs[];

// Set 1: I/O
layout(set=1, binding=0) readonly buffer Input {
    float16_t x[];
} hidden_in;

layout(set=1, binding=1) buffer Output {
    float16_t y[];
} hidden_out;

// === PER-THREAD STATE (VGPRs only, no LDS) ===
// Register allocation target: ≤ 16 VGPRs (very light shader)

void main() {
    uint idx = gl_GlobalInvocationID.x;  // 0..D-1
    if (idx >= D) return;

    // Step 1: Read fp16 value, promote to fp32
    float val = float(hidden_in.x[idx]);
    float res = float(hidden_out.y[idx]);  // existing residual value

    // Step 2: Add residual (in-place: reads current hidden_out, will write back)
    float combined = val + res;

    // Step 3: Square for RMS reduction — each thread contributes its square
    float sq = combined * combined;

    // Step 4: Subgroup-wide inclusive scan → reduce to sum of squares
    // Each subgroup independently reduces its portion
    float wg_sum_sq = subgroupAdd(sq);  // single-cycle Wave32 shuffle on RDNA4

    // Step 5: If multiple subgroups per WG, first thread of each subgroup
    // writes to LDS. Then final reduction within first subgroup.
    // With SUBGROUP_SIZE=32, WG_SIZE=256 → 8 subgroups.
    // SUBGROUP_SIZE=64, WG_SIZE=128 → 2 subgroups.

    uint num_subgroups = WG_SIZE_X / SUBGROUP_SIZE;
    float total_sum_sq;

    if (num_subgroups == 1) {
        total_sum_sq = wg_sum_sq;
    } else {
        // Multi-subgroup path: use shared memory to merge
        shared float lds_sums[8];  // max 8 subgroups for WG=256/32
        lds_sums[gl_SubgroupID] = wg_sum_sq;
        barrier();
        memoryBarrierShared();

        // First subgroup reduces from LDS
        if (gl_SubgroupID == 0) {
            float local_sum = 0.0;
            uint max_sg = min(num_subgroups, 8u);
            for (uint i = gl_SubgroupInvocationID; i < max_sg; i += SUBGROUP_SIZE) {
                local_sum += lds_sums[i];
            }
            total_sum_sq = subgroupAdd(local_sum);
            // Broadcast result to all threads in subgroup 0
            if (gl_SubgroupInvocationID == 0) {
                lds_sums[0] = total_sum_sq;
            }
        }
        barrier();
        memoryBarrierShared();
        total_sum_sq = lds_sums[0];
    }

    // Step 6: Compute inverse RMS
    float inv_rms = inversesqrt(total_sum_sq / float(D) + NORM_EPS);

    // Step 7: Read RMS weight, apply, write back
    // Weight is stored at layer offset in weight buffer
    float w = float(weight_bufs[nonuniformEXT(pc.layer_idx)].weights[D_offset + idx]);
    float normed = combined * inv_rms * w;

    // Step 8: Residual add back to output
    hidden_out.y[idx] = float16_t(res + normed);
}
```

### 1.7 RDNA4 Optimizations

- **Wave32 × 8 subgroups/WG**: 256-element workgroup gives 8 wavefronts. With 16 WGs (D=4096), that's 128 wavefronts across 32 CUs — ~4 waves/CU, occupancy ~25%. Sufficient for a bandwidth-bound norm kernel.
- **`subgroupAdd` is single-cycle**: The reduction inside a 32-lane warp costs 5 shuffle iterations (log2(32)). Total < 10 cycles for the reduction.
- **No LDS read for single-subgroup path**: When WG=SUBGROUP_SIZE (small D), the multi-subgroup path is elided by specialization constants — compile-time branch elimination.
- **coalesced memory access**: All threads access sequential fp16 elements. 256 threads × 2 bytes = 512-byte transaction → 4 cache lines on RDNA4 (128B L2 line).

### 1.8 RDNA2 Fallback

- **Wave64 × 2 subgroups/WG**: 128-element WG, 2 wavefronts. 32 WGs for D=4096 → 64 wavefronts over 40 CUs → ~1.6 waves/CU, occupancy ~10%.
- Wave64 subgroup shuffle is `subgroupShuffleXor` with 6 iterations (log2(64)).
- Higher VGPR usage (threads do same work but more VGPRs per wavefront due to 64 lanes vs 32).

### 1.9 Pipeline Barriers (After This Shader)

```c
VkBufferMemoryBarrier barrier = {
    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    .buffer = hidden_out_buffer,  // or norm_scratch_buffer
    .size = VK_WHOLE_SIZE,
};
vkCmdPipelineBarrier(cb,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
    0, 1, &barrier, 0, NULL, 0, NULL);
```

Transition: `hidden_out` goes from WRITE (this shader wrote normalized+residual values) to READ (next shader reads them).

### 1.10 Expected Occupancy

| Metric | RDNA4 (32 CU) | RDNA2 (40 CU) |
|--------|--------------|---------------|
| VGPRs/thread | ~12 | ~12 |
| LDS bytes | 32 (for 8-subgroup path) | 16 (for 2-subgroup path) |
| Active wavefronts | 128 (16 WGs × 8 waves) | 64 (32 WGs × 2 waves) |
| Max concurrent waves | 2,048 | 1,280 |
| Occupancy | ~6.25% | ~5.0% |
| Limiter | Global dispatch count (few WGs) | Same |

**This is expected for a bandwidth-bound norm kernel.** The limiting factor is not occupancy but memory bandwidth — reading D fp16 values = 8 KB per layer, far below the L2 cache size (12 MB RDNA4, 4 MB RDNA2). The entire hidden state fits in L2.

---

## 2. attn_qkv.comp — Fused Q+K+V Projection + RoPE + KV Cache Write

### 2.1 Purpose

Single dispatch computes Q, K, V projections from hidden state (RMS-normed), applies RoPE to Q and K, and writes K/V directly to KV cache at the current token position.

```
For a single token (decode, batch=1):
  hidden[d] → W_Q[d,d] → Q[n_heads, head_dim]  + RoPE
  hidden[d] → W_K[d, d_kv] → K[n_kv_heads, head_dim] + RoPE → KV cache[pos]
  hidden[d] → W_V[d, d_kv] → V[n_kv_heads, head_dim] → KV cache[pos]
```

**Why fused:** Q, K, V all read the SAME hidden state. Fusing eliminates 2/3 of the hidden state reads (from 3×8KB = 24KB down to 8KB) and 2/3 of pipeline barriers.

### 2.2 Workgroup Configuration (Decode, batch=1)

| Parameter | RDNA4 | RDNA2 | Justification |
|-----------|-------|-------|---------------|
| WG size X (total threads) | 64 | 128 | 2 subgroups (Wave32) or 2 subgroups (Wave64) |
| Subgroup size | 32 | 64 | |
| WG count X | `ceil(D / 64)` = 64 (for D=4096) | `ceil(D / 128)` = 32 | Each thread handles K elements via inner loop |
| WG size Y | 1 | 1 | |
| WG size Z | 1 | 1 | |

**Thread-to-work mapping:** Each thread in a subgroup is responsible for computing its slice of all Q/K/V output elements. With WG_SIZE=64 and SUBGROUP_SIZE=32 on RDNA4:
- Subgroup 0 (threads 0-31) computes Q outputs (rows 0, 2, 4... based on output row assignment)
- Subgroup 1 (threads 32-63) computes K+V outputs

*Alternative tile design (preferred for FP16 weights):*

Each subgroup computes one output row. A workgroup of 4 subgroups (128 threads RDNA4, 256 RDNA2) processes 4 output rows simultaneously. Each thread computes a partial dot product over a slice of the input.

**Final decode design (tile-based):**

| Parameter | RDNA4 | RDNA2 |
|-----------|-------|-------|
| WG size | 128 (4 subgroups × 32) | 256 (4 subgroups × 64) |
| WG count X | `ceil(N_OUT / 4)` where N_OUT = n_heads × head_dim + 2 × n_kv_heads × head_dim | Same logic |
| Subgroups per WG | 4 | 4 |
| Rows per WG | 4 (one per subgroup) | 4 |
| K (input dim per thread) | ceil(D / SUBGROUP_SIZE) | ceil(D / SUBGROUP_SIZE) |

For Qwen3.5-9B: Q output = 32 × 128 = 4096 elements; K output = 8 × 128 = 1024; V output = 1024.
Total output rows = 32 + 8 + 8 = 48 rows.

WG count = ceil(48 / 4) = 12 workgroups.

Each of 12 WGs computes 4 output rows. Each subgroup (1 row) runs an inner loop over D/SUBGROUP_SIZE steps.

### 2.3 Push Constant Usage

```
pc.layer_idx        → weight layer
pc.kv_cache_pos     → KV cache write position
pc.seq_len          → for RoPE position
pc.attn_scale       → not used here (used in attn_compute)
pc.rope_theta       → RoPE base frequency
pc.n_heads          → 32
pc.n_kv_heads       → 8
pc.head_dim         → 128 (runtime: use spec const)
```

### 2.4 Descriptor Bindings Used

| Set | Binding | Type | Access | Buffer |
|-----|---------|------|--------|--------|
| 0 | 0 | Storage Buffer | Read | Weight array: q_weight, k_weight, v_weight for this layer |
| 1 | 0 | Storage Buffer | Read | `hidden_in` (RMS-normed fp16, D elements) |
| 1 | 1 | Storage Buffer | Write | `hidden_out` — Q output region (n_heads × head_dim fp16) + K region + V region |
| 1 | 2 | Storage Buffer | Write | `k_cache` — at position `pc.kv_cache_pos` |
| 1 | 3 | Storage Buffer | Write | `v_cache` — at position `pc.kv_cache_pos` |
| 2 | 0 | Storage Buffer | Read | `rope_freqs` — precomputed cos/sin pairs for RoPE |
| 2 | 1 | Storage Buffer | Read | `kv_page_table` — for page-offset computation |

### 2.5 Shared Memory

```
ROWS_PER_WG * SUBGROUP_SIZE * sizeof(fp16) × 2 = intermediate results
  RDNA4: 4 × 32 × 2 × 2 = 512 bytes for partial sums (negligible)
  Also: 0 bytes (all reductions via subgroup shuffle, no cross-subgroup reduction needed)
```

Each subgroup independently computes one output row. No cross-subgroup reduction. LDS usage: **0 bytes** for the core kernel. Optional: LDS for weight prefetching (not needed for decode, batch=1).

### 2.6 GLSL Structure

```glsl
#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_int8   : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16  : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float32 : require
#extension GL_KHR_shader_subgroup_basic                   : require
#extension GL_KHR_shader_subgroup_arithmetic              : require
#extension GL_KHR_shader_subgroup_shuffle                 : require
#extension GL_EXT_subgroup_size_control                   : require
#extension GL_EXT_scalar_block_layout                     : require
#extension GL_EXT_nonuniform_qualifier                    : require
#extension GL_KHR_shader_integer_dot_product              : require
#extension GL_EXT_shader_8bit_storage                     : require
#extension GL_EXT_shader_16bit_storage                    : require

#include "common.glsl"
#include "push_constants.glsl"

layout(local_size_x_id = 1, local_size_y = 1, local_size_z = 1) in;

// Set 0: Weight buffers (descriptor-indexed by layer)
layout(set=0, binding=0) readonly buffer WeightBufs {
    uint8_t data[];  // raw bytes, cast as needed for quantized types
} weight_array[];

// Set 1: I/O
layout(set=1, binding=0) readonly buffer Input {
    float16_t x[];
} hidden_in;

layout(set=1, binding=1) buffer Output {
    float16_t y[];
} qkv_out;  // layout: [Q_rows | K_rows | V_rows] contiguous

layout(set=1, binding=2) buffer KCache {
    float16_t k[];
} k_cache;

layout(set=1, binding=3) buffer VCache {
    float16_t v[];
} v_cache;

// Set 2: RoPE tables
layout(set=2, binding=0) readonly buffer RopeFreqs {
    float cos_sin[];  // interleaved cos,sin for each (head_dim/2) pair
} rope_table;

// === Workgroup logic ===
// WG_SIZE = 128 (RDNA4) or 256 (RDNA2)
// SUBGROUPS_PER_WG = WG_SIZE / SUBGROUP_SIZE
// Each subgroup processes 1 output row
// Output rows are interleaved: Q rows 0..31, K rows 32..39, V rows 40..47

// Weight layout for this layer (offsets from architecture doc §2.1):
// Q_weight: layer_start + 0
// K_weight: layer_start + Q_size
// V_weight: layer_start + Q_size + K_size

shared float s_partial[SUBGROUPS_PER_WG][SUBGROUP_SIZE];  // optional, for fp16 accumulation

void main() {
    uint sg_id       = gl_SubgroupID;       // 0..3 (4 subgroups per WG)
    uint lane_id     = gl_SubgroupInvocationID; // 0..31 or 0..63
    uint wg_row_base = gl_WorkGroupID.x * SUBGROUPS_PER_WG; // which 4 rows this WG handles

    // Total output rows: N_HEADS (Q) + N_KV_HEADS (K) + N_KV_HEADS (V)
    uint total_rows = N_HEADS + N_KV_HEADS + N_KV_HEADS;
    uint row = wg_row_base + sg_id;
    if (row >= total_rows) return;

    // Determine: is this a Q, K, or V row?
    uint q_rows = N_HEADS;
    uint k_rows = N_KV_HEADS;
    uint v_rows = N_KV_HEADS;

    bool is_q = (row < q_rows);
    bool is_k = (row >= q_rows && row < q_rows + k_rows);
    bool is_v = (row >= q_rows + k_rows);

    uint head_idx;
    uint weight_offset;
    uint output_offset;

    if (is_q) {
        head_idx      = row;
        weight_offset = 0; // Q weight at layer start
        output_offset = head_idx * HEAD_DIM;
    } else if (is_k) {
        head_idx      = row - q_rows;
        weight_offset = Q_WEIGHT_SIZE; // defined by specialization
        output_offset = q_rows * HEAD_DIM + head_idx * HEAD_DIM;
    } else {
        head_idx      = row - q_rows - k_rows;
        weight_offset = Q_WEIGHT_SIZE + K_WEIGHT_SIZE;
        output_offset = q_rows * HEAD_DIM + k_rows * HEAD_DIM + head_idx * HEAD_DIM;
    }

    uint weight_row = head_idx * HEAD_DIM + 0; // actually: head_idx * head_dim for per-head weights
    // NOTE: For standard Llama, Q weight is [D, n_heads * head_dim], not per-head.
    // The row we compute is: output[head * head_dim + :] = input[D] dot weight[D, head * head_dim + :]
    // So weight_row_base = 0 for Q (flat), and we process across D

    // Weight row stride: for Q, K, V weights, each output element is a dot product
    // over D input elements. The weight is stored as [D_out, D_in] = [N_OUT, D].
    // Row = output element index (0..N_OUT-1)
    uint out_elem_start = row * HEAD_DIM;  // if per-element dispatch
    // Actually each subgroup computes one full head row (head_dim elements),
    // doing D dot products in an inner loop.

    // === Inner loop: dot product over D ===
    // Each thread accumulates partial sum for its slice of D
    float partial_sum = 0.0;

    // Thread processes every lane_id + k*SUBGROUP_SIZE element of D
    // Accumulates dot(input[i], weight[i][row]) across D
    for (uint k = lane_id; k < D; k += SUBGROUP_SIZE) {
        float16_t x_val = hidden_in.x[k];
        float16_t w_val = dequant_weight_at(weight_offset, row, k);
        partial_sum += float(x_val) * float(w_val);
    }

    // Subgroup reduction: sum partials to get full dot product
    float dot_result = subgroupAdd(partial_sum);

    // Now we have ONE output value: the dot product for this row.
    // But we need head_dim values per row (for K/V, 128 values; for Q, 128 values per head).
    // Wait — each subgroup computes ONE output element? That's wrong for decode.
    //
    // CORRECTED DESIGN:
    // For decode (batch=1, single token), each subgroup computes ONE FULL OUTPUT ROW
    // by doing head_dim dot products. Or more precisely:
    //
    // The Q weight is [D, n_heads * head_dim]. Each row = a specific element of Q output.
    // With 32 heads × 128 dim = 4096 Q output elements = 4096 weight rows.
    //
    // Each subgroup processes SUBGROUP_SIZE contiguous output elements (a "tile").
    // With 48 total output rows (Q+K+V = 32+8+8), each output row has HEAD_DIM elements.
    // So we need HEAD_DIM/SUBGROUP_SIZE steps per row, OR each subgroup handles
    // HEAD_DIM rows in one go.
    //
    // SIMPLER DESIGN for decode:
    // Each WORKGROUP processes HEAD_DIM output rows, each SUBGROUP processes
    // SUBGROUP_SIZE output elements. Inner loop over D/SUBGROUP_SIZE input chunks.
    //
    // With D=4096, HEAD_DIM=128, SUBGROUP_SIZE=32:
    //   - WG_SIZE = 128 (4 subgroups × 32)
    //   - Each workgroup handles 4 output rows
    //   - 48 total rows → 12 workgroups
    //   - Each subgroup: 128 inner loop iterations over D
    //
    // Actually that's correct — each subgroup does ONE output value per D loop iteration.
    // Each thread in the subgroup accumulates its slice, then reduce at end.
    // That gives us 1 output value per subgroup.
    //
    // But we need HEAD_DIM=128 values per row! So we need multiple subgroups per row.
    //
    // BETTER DESIGN: Split HEAD_DIM across subgroups.
    // Each subgroup computes SUBGROUP_SIZE elements of HEAD_DIM.
    // So subtitle: HEAD_DIM / SUBGROUP_SIZE subgroups per output row.
    //
    // For Q: 32 heads × 128 dim = 4096 outputs = 128 subgroups needed
    // For K: 8 heads × 128 dim = 1024 outputs = 32 subgroups
    // For V: 1024 outputs = 32 subgroups
    // Total: 192 subgroups = 6 WG of 32 subgroups (WG=256) or 12 WG of 16 subgroups (WG=128)
    //
    // But 256-thread WG with 32-wide subgroups = 8 subgroups/WG → not integer 32 subgroups per something.
    //
    // SIMPLEST CORRECT DESIGN for decode batch=1:
    // We dispatch ceil(NUM_OUTPUT_ELEMENTS / SUBGROUP_SIZE) workgroups.
    // Each WG has 1 subgroup (WG_SIZE = SUBGROUP_SIZE).
    // Each subgroup computes SUBGROUP_SIZE output elements.
    // Inner loop over D.
    //
    // NUM_OUTPUT_ELEMENTS = N_HEADS * HEAD_DIM + 2 * N_KV_HEADS * HEAD_DIM
    //   = 32*128 + 2*8*128 = 4096 + 2048 = 6144
    //
    // SUBGROUP_SIZE=32: 6144/32 = 192 workgroups of 32 threads each
    // SUBGROUP_SIZE=64: 6144/64 = 96 workgroups of 64 threads each
    //
    // Each thread in subgroup computes ONE output element dot product:
    // sum(input[i] * weight[row][i]) for i in thread's slice of D
    // D/SUBGROUP_SIZE iterations, then subgroupAdd.

    // Corrected implementation:
    uint global_row = gl_WorkGroupID.x;  // workgroup = output row group
    uint out_idx_base = global_row * SUBGROUP_SIZE + lane_id;

    // out_idx_base identifies which element of the flat output we compute
    // Flat output layout: [Q:0..4095 | K:4096..5119 | V:5120..6143]
    //   Q: head h, dim d → out = h * HEAD_DIM + d
    //   K: head h, dim d → out = Q_SIZE + h * HEAD_DIM + d
    //   V: head h, dim d → out = Q_SIZE + K_SIZE + h * HEAD_DIM + d

    uint q_size = N_HEADS * HEAD_DIM;
    uint k_size = N_KV_HEADS * HEAD_DIM;

    float accum = 0.0;
    if (out_idx_base < q_size + 2 * k_size) {
        // Determine weight row and output position
        for (uint k = 0; k < D; k += SUBGROUP_SIZE) {
            uint k_idx = k + lane_id;
            if (k_idx < D) {
                float16_t x_val = hidden_in.x[k_idx];
                float w_val = dequant_weight(weight_offset, out_idx_base, k_idx);
                accum += float(x_val) * w_val;
            }
        }
        float result = subgroupAdd(accum);

        // Store result
        if (lane_id == 0) {
            qkv_out.y[out_idx_base] = f32tof16(result);
        }

        // Apply RoPE if Q or K (not V)
        // ...
    }
}
```

### 2.7 Quantization Handling

```glsl
// Weight dequant: called inside inner loop
// weight_offset = offset within layer buffer to Q/K/V weight data
// row = output element index (0..N_OUT-1)
// col = input element index (0..D-1)
float dequant_weight(uint weight_offset, uint row, uint col) {
    // weight is stored as [D_out, D_in] = [row, col]
    // byte offset = weight_offset + row * D * bytes_per_element + col * bytes_per_element
    // For quantized: byte offset = weight_offset + row * row_stride + block_start(col)

    uint weight_base = 0; // actual base from spec const

#if QUANT_TYPE == 0  // FP16
    uint elem_idx = row * D + col;
    float16_t w = float16_t(weight_array[pc.layer_idx].data[weight_offset + elem_idx * 2],
                             weight_array[pc.layer_idx].data[weight_offset + elem_idx * 2 + 1]);
    return float(w);

#elif QUANT_TYPE == 1  // Q8_0
    uint block_idx = col / 32; // Q8_0 block size = 32
    uint elem_in_block = col % 32;
    uint block_row_stride = D / 32 * 34; // 34 bytes per block (2B d + 32B qs)
    uint block_start = weight_offset + row * block_row_stride + block_idx * 34;
    return dequant_q8_0(block_start, elem_in_block);

#elif QUANT_TYPE == 2  // Q4_K
    uint superblock = col / 256; // Q4_K superblock = 256
    uint elem_in_sb = col % 256;
    uint sb_row_stride = D / 256 * Q4K_BLOCK_BYTES; // Q4K_BLOCK_BYTES = 148
    uint sb_start = weight_offset + row * sb_row_stride + superblock * Q4K_BLOCK_BYTES;
    return dequant_q4_k(sb_start, elem_in_sb);

#elif QUANT_TYPE == 3  // Q6_K
    // Q6_K: superblock=256, similar layout, block_bytes=176
    uint superblock_q6 = col / 256;
    uint elem_in_sb_q6 = col % 256;
    uint sb_row_stride_q6 = D / 256 * Q6K_BLOCK_BYTES;
    uint sb_start_q6 = weight_offset + row * sb_row_stride_q6 + superblock_q6 * Q6K_BLOCK_BYTES;
    return dequant_q6_k(sb_start_q6, elem_in_sb_q6);

#elif QUANT_TYPE == 4  // IQ4_XS
    // IQ4_XS: superblock=256, block_bytes depends on implementation
    // ...

#endif
}

// Cooperative matrix path (RDNA4 INT8):
// When VK_KHR_cooperative_matrix is available, use cooperative_matrix
// for the dot product accumulation instead of the scalar loop.
//
// cooperative_matrix<float16_t, gl_ScopeSubgroup, 16, 16, 16> C;
// Load A (input fp16) tile, load B (dequantized weight) tile, matmul accumulate.
// Then each thread writes its portion of C to output.
```

### 2.8 RoPE Application

```glsl
// RoPE: Applied in-place on Q and K before write
// For each head, for each pair (2i, 2i+1) in head_dim:
//   cos = cos_table[pos * head_dim/2 + i]
//   sin = sin_table[pos * head_dim/2 + i]
//   x' = x*cos - y*sin   (even index)
//   y' = y*cos + x*sin   (odd index)

void apply_rope(uint out_idx, bool is_kv_position) {
    // Decode: only one token position. RoPE freq depends on position.
    // Determine if this output element is Q or K (V does NOT get RoPE)
    // ...

    uint head_dim_half = HEAD_DIM / 2;
    uint head = output_head_idx; // determined from out_idx
    uint dim_within_head = output_dim_idx % HEAD_DIM;
    uint pair_idx = dim_within_head / 2;
    bool is_even = (dim_within_head % 2) == 0;

    uint freq_idx = pair_idx;
    float cos_val = rope_table.cos_sin[2 * freq_idx + 0];
    float sin_val = rope_table.cos_sin[2 * freq_idx + 1];

    // For RoPE with position: freq = 1 / (theta^(2i/d))
    // Precomputed in rope_table for each (freq_idx, position)
    // Actually simpler: precompute cos(m*theta^-i) and sin(m*theta^-i) for all positions
    // Index: rope_table[position * HEAD_DIM/2 * 2 + pair_idx * 2 + {0|1}]
    // This can be 128K positions × 2 × 64 pairs × 4 bytes = 64 MB for full table
    // OR compute on-the-fly: cosval = cos(position * freq), sinval = sin(position * freq)
    // where freq = 1.0 / pow(rope_theta, 2.0 * float(pair_idx) / float(HEAD_DIM))
    //
    // For decode (single position), on-the-fly compute is cheap:
    float freq = 1.0 / pow(ROPE_THETA, 2.0 * float(pair_idx) / float(HEAD_DIM));
    float pos = float(pc.kv_cache_pos);
    float cos_val_c = cos(pos * freq);
    float sin_val_c = sin(pos * freq);

    float this_val = float(qkv_out.y[out_idx]);
    float other_val = float(qkv_out.y[out_idx + (is_even ? 1 : -1)]);

    float rotated;
    if (is_even) {
        rotated = this_val * cos_val_c - other_val * sin_val_c;
    } else {
        rotated = other_val * cos_val_c + this_val * sin_val_c;
    }

    qkv_out.y[out_idx] = float16_t(rotated);
}
```

### 2.9 KV Cache Write

```glsl
// After computing K and V elements and applying RoPE (to K):
// Write to paged KV cache at current position

void write_to_kv_cache(uint head_idx, uint dim_idx, float16_t k_val, float16_t v_val, bool is_k) {
    // Paged KV cache addressing (from architecture doc §2.2):
    // Each page: PAGE_TOKENS × HEAD_DIM × sizeof(fp16) for K
    //           PAGE_TOKENS × HEAD_DIM × sizeof(fp16) for V
    //
    // token position in page: kv_cache_pos % PAGE_TOKENS
    // page index: kv_cache_pos / PAGE_TOKENS
    // logical page → physical page: kv_page_table[page_idx]

    uint token_in_page = pc.kv_cache_pos % KV_PAGE_TOKENS;
    uint page_idx      = pc.kv_cache_pos / KV_PAGE_TOKENS;
    // Look up physical page from table
    uint phys_page = kv_page_table.page_ids[pc.layer_idx * PAGES_PER_LAYER + page_idx];

    // K cache offset within page:
    // head_stride = PAGE_TOKENS * HEAD_DIM * sizeof(fp16)
    // element_offset = head_idx * head_stride + token_in_page * HEAD_DIM + dim_idx
    uint head_stride = KV_PAGE_TOKENS * HEAD_DIM;
    uint elem_off = phys_page * PAGES_PER_LAYER * head_stride * N_KV_HEADS +  // layer base
                    head_idx * head_stride +                                     // head offset
                    token_in_page * HEAD_DIM + dim_idx;                          // element

    if (is_k) {
        k_cache.k[elem_off] = k_val;
    } else {
        v_cache.v[elem_off] = v_val;
    }
}
```

### 2.10 RDNA4-Specific Optimizations

- **Wave32 × 4 subgroups/WG**: 128-thread WG splits into 4 subgroups. Each subgroup processes 32 elements. With 192 WGs for 6144 total output elements, 768 wavefronts across 32 CUs → 24 waves/CU, occupancy ~1.2%. The limiting factor is memory bandwidth, not occupancy.
- **dp4a for Q4_K/Q8_0**: Use `GL_KHR_shader_integer_dot_product` with `dot4add_u8packed` to compute 4 dot products per instruction:
  ```glsl
  // Instead of scalar loop:
  uint packed_quants = uint(data[offset]);  // 4 × INT8 packed in uint32
  uint packed_input  = pack4x8_as_uint(input_vals); // 4 × uint8 from input
  acc += dot4add_u8packed(packed_quants, packed_input, acc);
  ```
  This achieves 4 MACs/cycle vs 1 MAC/cycle for scalar, a 4× speedup for the dequant+dot inner loop.
- **Cooperative matrix for FP16 weights**: When weights are FP16, use `cooperative_matrix` with 16×16×16 tiles. The input vector is broadcast across the tile, weight matrix tile is loaded once. On RDNA4, cooperative_matrix maps to WMMA instructions (16×16×16 FP16 → FP32 accumulate).
- **Vectorized loads**: Load 4 fp16 elements at a time using `float16_t4` for 8-byte coalesced transactions.

### 2.11 RDNA2 Fallback

- **Wave64 × 4 subgroups/WG**: 256-thread WG. 96 WGs for 6144 elements → 384 wavefronts over 40 CUs → ~9.6 waves/CU.
- **No cooperative_matrix**: Fall back entirely on scalar or dp4a path.
- **dp2a for RDNA2**: If dp4a isn't available, dp2a (2 dot products per instruction) is universally available on RDNA2 via `dot2add_u8packed`.
- **Higher VGPR pressure**: Wave64 needs more VGPRs per wavefront for the same computation. Target ≤ 48 VGPRs to maintain 2 concurrent waves/CU.

### 2.12 Pipeline Barriers (After)

| Resource | Transition | Reason |
|----------|-----------|--------|
| `qkv_out` (set 1 binding 1) | WRITE → READ | Next shader reads Q/K/V |
| `k_cache` (set 1 binding 2) | WRITE → READ | Attention compute reads KV cache |
| `v_cache` (set 1 binding 3) | WRITE → READ | Attention compute reads KV cache |

```
VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT → VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT
dstAccessMask = VK_ACCESS_SHADER_READ_BIT
3 buffer barriers: qkv_out, k_cache, v_cache
```

### 2.13 Expected Occupancy

| Metric | RDNA4 (decode) | RDNA2 (decode) |
|--------|---------------|---------------|
| WG threads | 128 | 256 |
| Subgroups/WG | 4 × Wave32 | 4 × Wave64 |
| Total WGs | ceil(6144/32) = 192 | ceil(6144/64) = 96 |
| VGPRs/thread | ~48 | ~48 |
| LDS bytes | 0 | 0 |
| Active wavefronts | 768 | 384 |
| Max concurrent | 2,048 | 1,280 |
| Occupancy | ~37.5% | ~30% |
| Limiter | VGPR pressure | VGPR pressure |

For prefill (batch=seq_len), the workgroup grid expands: WG count Y = seq_len. Prefill QKV shader would use larger tiles (256-thread WG, 8 subgroups) for better memory coalescing.

---

## 3. attn_compute.comp — Fused Attention: Q·K^T + Scale + Softmax + V

### 3.1 Purpose

For decode (single-token query):
```
For each head h (0..N_HEADS-1):
  Q_h = Q[h * HEAD_DIM : (h+1) * HEAD_DIM]          // [head_dim]
  K_cached = K_cache[h_kv(h)][0..seq_len-1]          // [seq_len, head_dim]
  V_cached = V_cache[h_kv(h)][0..seq_len-1]          // [seq_len, head_dim]

  scores = Q_h · K_cached^T                           // [seq_len] — dot product per position
  scores = scores * attn_scale                        // scale by 1/sqrt(head_dim)
  scores = softmax(scores)                            // [seq_len] normalized probabilities
  out_h  = scores · V_cached                          // [head_dim] weighted sum
```

GQA mapping: `h_kv(h) = h * N_KV_HEADS / N_HEADS`. For 32 Q-heads, 8 KV-heads: heads 0-3 → KV head 0, heads 4-7 → KV head 1, etc.

### 3.2 Workgroup Configuration

| Parameter | RDNA4 | RDNA2 | Justification |
|-----------|-------|-------|---------------|
| WG size X | 128 (4 × Wave32) | 128 (2 × Wave64) | Each workgroup processes 1 head. HEAD_DIM=128 is evenly divisible by 128 (1 element/thread) or 64 (2 elements/thread) |
| WG size Y | 1 | 1 | |
| WG size Z | 1 | 1 | |
| WG count X | N_HEADS (32) | N_HEADS (32) | One WG per Q head |
| WG count Y | 1 | 1 | Decode batch=1 |
| WG count Z | 1 | 1 | |

Each workgroup handles ONE attention head. Inside the WG:
- Each subgroup computes Q·K dot products for a subset of KV positions (split seq_len across subgroups)
- Reduce across subgroups for softmax
- Each thread handles 1 element of head_dim for accumulation

**Alternatively: warp-per-row design.** Each subgroup processes one KV position:

| Subgroup 0 | Subgroup 1 | Subgroup 2 | Subgroup 3 |
|------------|------------|------------|------------|
| KV pos 0* | KV pos 1* | KV pos 2* | KV pos 3* |

Each iteration processes SUBGROUPS_PER_WG KV positions. Inner loop iterates `seq_len / SUBGROUPS_PER_WG` times. Each iteration's QK score is stored to LDS, then a WG-wide softmax runs, then final accumulation.

### 3.3 Push Constant Usage

```
pc.layer_idx        → which layer's KV cache
pc.kv_cache_pos      → current token position (for KV cache upper bound)
pc.seq_len           → number of tokens to attend over
pc.attn_scale        → 1/sqrt(head_dim)
pc.n_heads           → 32
pc.n_kv_heads        → 8
pc.head_dim          → 128
```

### 3.4 Descriptor Bindings Used

| Set | Binding | Type | Access | Buffer |
|-----|---------|------|--------|--------|
| 0 | — | — | — | Not used (weights not needed) |
| 1 | 0 | Storage Buffer | Read | `hidden_in` — Q output from QKV shader |
| 1 | 1 | Storage Buffer | Write | `hidden_out` — attention output (n_heads × head_dim fp16) |
| 1 | 2 | Storage Buffer | Read | `k_cache` — full KV cache for this layer |
| 1 | 3 | Storage Buffer | Read | `v_cache` — full KV cache for this layer |
| 2 | 1 | Storage Buffer | Read | `kv_page_table` — page table for cache addressing |

### 3.5 Shared Memory

```
seq_len × sizeof(fp16) × 2 = scores + normalized probs
For max 4096 context:
  4096 × 2 × 2 = 16,384 bytes = 16 KB

Also: intermediate partial sums for softmax:
  1 × sizeof(float) × 4 = 16 bytes (max_score, sum_exp, etc.)
  SUBGROUPS_PER_WG × sizeof(float) = 4 × 4 = 16 bytes

Total LDS: ~16.4 KB — within 32 KB limit
```

If seq_len exceeds what fits in LDS (seq_len > 4096 → needs >32KB for scores), we split into tiles:
- Process seq_len in tiles of TILE_KV (e.g., 2048 positions)
- For each tile: compute QK scores → partial softmax (online softmax algorithm) → accumulate weighted V
- No change to output correctness

### 3.6 GLSL Structure

```glsl
#version 460
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float32 : require
#extension GL_KHR_shader_subgroup_basic                   : require
#extension GL_KHR_shader_subgroup_arithmetic              : require
#extension GL_KHR_shader_subgroup_shuffle                 : require
#extension GL_EXT_subgroup_size_control                   : require
#extension GL_EXT_scalar_block_layout                     : require

#include "common.glsl"
#include "push_constants.glsl"

layout(local_size_x_id = 1, local_size_y = 1, local_size_z = 1) in;

layout(set=1, binding=0) readonly buffer Input {
    float16_t q[];  // Q values for ALL heads (N_HEADS * HEAD_DIM)
} q_in;

layout(set=1, binding=1) buffer Output {
    float16_t y[];  // attention output (N_HEADS * HEAD_DIM)
} attn_out;

layout(set=1, binding=2) readonly buffer KCache {
    float16_t k[];  // paged KV cache K values
} k_cache;

layout(set=1, binding=3) readonly buffer VCache {
    float16_t v[];  // paged KV cache V values
} v_cache;

layout(set=2, binding=1) readonly buffer PageTable {
    uint page_ids[];  // page_ids[layer * PAGES_PER_LAYER + page_idx]
} kv_page_table;

// LDS allocation
// max_seq_len at runtime; bounded at compile time via spec constant MAX_SEQ_LEN
shared float s_scores[MAX_SEQ_LEN];  // spec constant: up to 4096 (16KB)
shared float s_probs[MAX_SEQ_LEN];
shared float s_max_val;
shared float s_sum_exp;

// Thread-to-element mapping
// WG_SIZE = HEAD_DIM (128 or close to it)
// At least HEAD_DIM threads per WG, each thread handles 1 element of head_dim
// Inner loop over seq_len positions

void main() {
    uint head_idx   = gl_WorkGroupID.x;  // which Q head (0..31)
    uint kv_head    = head_idx * N_KV_HEADS / N_HEADS; // GQA mapping
    uint thread_idx = gl_LocalInvocationID.x;
    uint head_dim   = HEAD_DIM;

    // Step 1: Load Q for this head
    // Q layout: [head_0_dim0..dim127 | head_1_dim0..dim127 | ...]
    // Each thread loads its Q element (one head_dim element)
    float my_q = 0.0;
    if (thread_idx < head_dim) {
        my_q = float(q_in.q[head_idx * head_dim + thread_idx]);
    }

    // Step 2: Compute Q·K for all KV positions
    // Online softmax: process one KV position at a time, maintain running max and sum
    // For decode (single query, batch=1), this is bandwidth-bound on KV cache reads.

    // Split seq_len across subgroups to increase parallelism
    uint kv_per_subgroup = (SEQ_LEN + SUBGROUPS_PER_WG - 1) / SUBGROUPS_PER_WG;
    uint kv_start = gl_SubgroupID * kv_per_subgroup;
    uint kv_end = min(kv_start + kv_per_subgroup, SEQ_LEN);

    float subgroup_max = -1e20f; // equivalent to -inf for fp32
    float subgroup_sum = 0.0;

    // Tiled processing: if seq_len > LDS capacity, use online softmax in tiles
    #define TILE_SIZE (MAX_SEQ_LEN)  // spec constant, = min(4096, floor(32KB / sizeof(float)))

    // Phase 1: Compute all QK scores, write to LDS
    for (uint kv_pos = thread_idx; kv_pos < SEQ_LEN; kv_pos += WG_SIZE_X) {
        float qk = 0.0;

        // Dot product Q·K[kv_pos] across head_dim
        // Access K from paged cache
        uint k_offset = kv_cache_element_offset(kv_head, kv_pos, thread_idx);

        // Thread processes one element of the dot product
        // This requires a reduction — each thread has only 1 element of Q and K
        // NEED: Q[thread_idx] * K[kv_pos][thread_idx] → reduce across all threads

        // For this to work, we need 128 threads per WG (matching head_dim)
        // Each thread computes Q[thread_idx] * K[kv_pos][thread_idx]
        // Then subgroup-shuffle-reduce to get the full dot product

        if (thread_idx < head_dim) {
            float16_t k_val = k_cache.k[k_offset];
            qk = my_q * float(k_val);  // partial product
        }

        // Reduce across all threads in WG to get full dot product
        float dot_result = subgroupAdd(qk);  // within-subgroup reduction

        // If multiple subgroups, need cross-subgroup reduction
        // For WG_SIZE=128 = 4×32 subgroups:
        if (SUBGROUPS_PER_WG > 1) {
            // First thread of each subgroup writes its subgroup sum to LDS
            if (gl_SubgroupInvocationID == 0) {
                s_scores[kv_pos * SUBGROUPS_PER_WG + gl_SubgroupID] = dot_result;
            }
            barrier();
            memoryBarrierShared();
            // Subgroup 0 reads and sums
            if (gl_SubgroupID == 0) {
                float cross_sum = 0.0;
                for (uint s = 0; s < SUBGROUPS_PER_WG; s++) {
                    cross_sum += s_scores[kv_pos * SUBGROUPS_PER_WG + s];
                }
                // Write final score
                s_scores[kv_pos] = cross_sum * ATTENTION_SCALE;
            }
        } else {
            // Single subgroup: no cross-subgroup needed
            if (gl_SubgroupInvocationID == 0) {
                s_scores[kv_pos] = dot_result * ATTENTION_SCALE;
            }
        }
    }

    barrier();
    memoryBarrierShared();

    // Phase 2: Softmax — find max, compute exp, normalize
    // All threads participate; scores are in LDS
    float local_max = -1e20f;
    for (uint i = thread_idx; i < SEQ_LEN; i += WG_SIZE_X) {
        local_max = max(local_max, s_scores[i]);
    }
    float wg_max = subgroupMax(local_max);
    if (SUBGROUPS_PER_WG > 1) {
        if (gl_SubgroupInvocationID == 0) {
            s_scores[MAX_SEQ_LEN + gl_SubgroupID] = wg_max;
        }
        barrier();
        memoryBarrierShared();
        if (gl_SubgroupID == 0) {
            float cross_max = s_scores[MAX_SEQ_LEN];
            for (uint s = 1; s < SUBGROUPS_PER_WG; s++) {
                cross_max = max(cross_max, s_scores[MAX_SEQ_LEN + s]);
            }
            if (gl_SubgroupInvocationID == 0) {
                s_max_val = cross_max;
            }
        }
        barrier();
        memoryBarrierShared();
        wg_max = s_max_val;
    }

    // Compute exp and sum
    float local_sum = 0.0;
    for (uint i = thread_idx; i < SEQ_LEN; i += WG_SIZE_X) {
        float prob = exp(s_scores[i] - wg_max);
        s_probs[i] = prob;
        local_sum += prob;
    }
    float wg_sum = subgroupAdd(local_sum);
    if (SUBGROUPS_PER_WG > 1) {
        if (gl_SubgroupInvocationID == 0) {
            s_scores[MAX_SEQ_LEN + gl_SubgroupID] = wg_sum;
        }
        barrier();
        memoryBarrierShared();
        if (gl_SubgroupID == 0) {
            float cross_sum = s_scores[MAX_SEQ_LEN];
            for (uint s = 1; s < SUBGROUPS_PER_WG; s++) {
                cross_sum += s_scores[MAX_SEQ_LEN + s];
            }
            if (gl_SubgroupInvocationID == 0) {
                s_sum_exp = cross_sum;
            }
        }
        barrier();
        memoryBarrierShared();
        wg_sum = s_sum_exp;
    }

    // Normalize in-place
    float inv_sum = 1.0 / wg_sum;
    for (uint i = thread_idx; i < SEQ_LEN; i += WG_SIZE_X) {
        s_probs[i] *= inv_sum;
    }
    barrier();
    memoryBarrierShared();

    // Phase 3: Weighted sum — accumulate scores * V
    float out_accum = 0.0;
    for (uint kv_pos = 0; kv_pos < SEQ_LEN; kv_pos++) {
        float score = s_probs[kv_pos];
        if (thread_idx < head_dim) {
            uint v_offset = kv_cache_element_offset(kv_head, kv_pos, thread_idx);
            float16_t v_val = v_cache.v[v_offset];
            out_accum += score * float(v_val);
        }
    }

    // Write output
    if (thread_idx < head_dim) {
        attn_out.y[head_idx * head_dim + thread_idx] = f32tof16(out_accum);
    }
}

// Helper: compute byte offset into paged KV cache for element (head, position, dim)
uint kv_cache_element_offset(uint kv_head, uint token_pos, uint dim_idx) {
    uint token_in_page = token_pos % KV_PAGE_TOKENS;
    uint page_idx      = token_pos / KV_PAGE_TOKENS;
    uint phys_page     = kv_page_table.page_ids[pc.layer_idx * PAGES_PER_LAYER + page_idx];
    // Check invalid page (0xFFFFFFFF)
    if (phys_page == 0xFFFFFFFFu) return 0;

    uint head_stride = KV_PAGE_TOKENS * HEAD_DIM;
    // physical layer offset + head offset + token offset + dim
    return phys_page * PAGES_PER_LAYER * head_stride * N_KV_HEADS
         + kv_head * head_stride
         + token_in_page * HEAD_DIM
         + dim_idx;
}
```

### 3.7 RDNA4 Optimizations

- **Wave32, 4 subgroups/WG**: 128 threads per WG maps 1:1 with HEAD_DIM=128. Each thread handles 1 Q element. This is register-light (~32 VGPRs/thread) achieving high occupancy.
- **LDS bandwidth**: 16KB reads (scores + probs) at ~128 bytes/cycle per CU = well within limits.
- **Subgroup shuffle for reductions**: `subgroupAdd` and `subgroupMax` are single-cycle on RDNA4 for Wave32.
- **FP16 throughout, FP32 accumulation**: Q, K, V stored as fp16 for memory efficiency. Dot products accumulate in fp32 for precision. Output stored as fp16.

### 3.8 RDNA2 Fallback

- **Wave64, 2 subgroups/WG**: 128 threads = 2 × Wave64. Each thread handles 1 Q element (head_dim=128). Subgroup reductions take log2(64) = 6 shuffle iterations.
- **Lower L2 cache**: 4MB L2 on RDNA2 vs 12MB on RDNA4. The KV cache is large (multiple MB for long contexts) — tiles of 4096 positions × head_dim × fp16 × n_kv_heads per layer may exceed L2. Tile-based access pattern helps by reusing K/V across multiple Q heads within a GQA group (4 Q heads share the same K/V).
- **No cooperative_matrix used**: Attention is not a matrix multiply; it's dot products + reductions. Subgroup operations suffice.

### 3.9 Pipeline Barriers (After)

```
hidden_out (attn output): SHADER_WRITE → SHADER_READ
Buffer barrier on hidden_out for next shader (attention output projection)
```

### 3.10 Expected Occupancy

| Metric | RDNA4 | RDNA2 |
|--------|-------|-------|
| WG threads | 128 | 128 |
| VGPRs/thread | ~40 (Q, K, V loads + softmax state) | ~40 |
| LDS bytes | 16,400 (scores + probs + temp) | 16,400 |
| LDS limit | 32,768 | 32,768 |
| Active WGs | 32 (one per head) × factor | 32 |
| Active wavefronts | 128 (32 × 4) | 64 (32 × 2) |
| Occupancy | ~6.25% | ~5% |
| Limiter | Dispatch count (few WGs) | Dispatch count |

**Interpretation:** Attention compute for decode is a small problem. 32 WGs × 128 threads is very low occupancy. But this is inherent to batched=1 attention. Amortized over the 32 heads, each WG runs the inner KV loop `seq_len × SUBGROUPS_PER_WG / WG_SIZE` iterations — for 4096 seq_len, that's ~128 iterations per thread of the KV scan loop. The long inner loop keeps the GPU busy despite low occupancy.

---

## 4. attn_output.comp — Attention Output Projection (W_O) + Residual

### 4.1 Purpose

```
hidden = hidden + W_O · attention_output
```
Projects the concatenated attention heads back to hidden dimension, fused with residual add.

```
attention_output = [head_0 | head_1 | ... | head_31]  // n_heads × head_dim = D
hidden = hidden + W_O[D × D] · attention_output[D]
```

Since Qwen3.5-9B uses n_heads × head_dim == D (32 × 128 = 4096), the attention output is already D-dimensional. W_O is a square [D × D] matrix.

### 4.2 Workgroup Configuration

| Parameter | RDNA4 | RDNA2 |
|-----------|-------|-------|
| WG size | 128 (4 × Wave32) | 256 (4 × Wave64) |
| WG count X | ceil(D / WG_SIZE) = 32 | ceil(D / WG_SIZE) = 16 |
| Rows per WG | WG_SIZE / SUBGROUP_SIZE = 4 | 4 |
| Elements per WG | WG_SIZE (one per thread) | WG_SIZE |

Each workgroup computes WG_SIZE output elements (one per thread). Each thread computes one output element = dot product of attention_output[D] with one row of W_O[D×D].

### 4.3 Descriptor Bindings

| Set | Binding | Type | Access | Buffer |
|-----|---------|------|--------|--------|
| 0 | 0 | Storage Buffer | Read | O_weight for this layer |
| 1 | 0 | Storage Buffer | Read | `hidden_in` — attention output (D elements fp16) |
| 1 | 1 | Storage Buffer | Read+Write | `hidden_out` — residual accumulated (D elements fp16) |

### 4.4 Shared Memory

**0 bytes** — no cross-thread reduction needed for output projection (each thread computes its own full output element via inner loop + subgroup reduction).

Wait — each thread needs the full dot product across D. With SUBGROUP_SIZE threads in a subgroup, each thread computes a partial sum:
```
partial[t] = sum_{k=t, t+S, t+2S, ...} attn_out[k] * W_O[row][k]
final = subgroupAdd(partial[t])
final → hidden_out[global_row] += final
```

No LDS needed — pure subgroup shuffle reduction.

### 4.5 GLSL Structure

```glsl
void main() {
    uint row = gl_GlobalInvocationID.x;  // 0..D-1
    if (row >= D) return;

    uint lane_id = gl_SubgroupInvocationID;

    // Double-buffer mode: compute attn_output dot product, then add to residual
    float accum = 0.0;
    for (uint k = lane_id; k < D; k += SUBGROUP_SIZE) {
        float16_t a = hidden_in.x[k];  // attention output element
        float16_t w = dequant_o_weight(row, k);
        accum += float(a) * float(w);
    }

    float result = subgroupAdd(accum);

    // Residual add (read-modify-write)
    float residual = float(hidden_out.y[row]);
    hidden_out.y[row] = f32tof16(residual + result);
}
```

### 4.6 Quantization

Uses the same `dequant_weight()` pattern as attn_qkv (see §2.7), specialized for the O_weight quantization type (which matches the model's weight quantization: Q4_K, Q6_K, Q8_0, or FP16).

The W_O weight is stored at `layer_weight_buffer + O_weight_offset`.

### 4.7 RDNA4 Optimizations

- **dp4a for INT8 quantized**: `dot4add_u8packed` achieves 4× throughput vs scalar for the inner dot product loop.
- **Cooperative matrix for FP16**: `cooperative_matrix<16, 16, 16>` maps to WMMA on RDNA4.
- **Vectorized load**: `float16_t4` loads from attention output (coalesced reads, all threads access sequential elements).

### 4.8 Pipeline Barriers (After)

```
hidden_out: SHADER_WRITE → SHADER_READ (for next layer or next operation)
```

### 4.9 Expected Occupancy

| Metric | RDNA4 | RDNA2 |
|--------|-------|-------|
| WG threads | 128 | 256 |
| VGPRs/thread | ~20 | ~20 |
| Active WGs | 32 | 16 |
| Active wavefronts | 128 | 64 |
| Occupancy | ~6.25% | ~5% |

Similar to QKV — low occupancy but D=4096 inner loop keeps ALU busy per thread.

---

## 5. ffn_gate_up.comp — Fused Gate + Up Projection + SiLU + Elementwise Multiply

### 5.1 Purpose

Single dispatch performs the FFN "gate + up" path:
```
gate = gate_weight[D × FFN_DIM] · hidden[D]    // gate projection
up   = up_weight[D × FFN_DIM] · hidden[D]      // up projection
ffn_hidden = SiLU(gate) * up                    // elementwise multiply
```

Fusing gate+up into one shader:
- Eliminates one hidden state read (both compute from same input)
- Eliminates one pipeline barrier
- SiLU is fused with the gate write (no separate activation pass)
- Elementwise multiply is fused (no separate multiply pass)

### 5.2 Workgroup Configuration

| Parameter | RDNA4 | RDNA2 |
|-----------|-------|-------|
| WG size | 128 (4 × Wave32) | 256 (4 × Wave64) |
| WG count X | ceil(FFN_DIM / WG_SIZE) | ceil(FFN_DIM / WG_SIZE) |
| Rows per WG | 4 (one per subgroup) | 4 |
| Outputs per WG | WG_SIZE (one per thread) | WG_SIZE |
| Gate+Up elements per WG | 2 × WG_SIZE | 2 × WG_SIZE |

For Qwen3.5-9B, FFN_DIM=11008:
- RDNA4: 128 threads/WG, ceil(11008/128) = 86 WGs
- RDNA2: 256 threads/WG, ceil(11008/256) = 43 WGs

Each subgroup computes one output element: gate[row] and up[row] for row in the workgroup's range.

### 5.3 GLSL Structure

```glsl
void main() {
    uint wg_row_base = gl_WorkGroupID.x * SUBGROUPS_PER_WG; // 0, 4, 8, ...
    uint sg_id       = gl_SubgroupID;
    uint lane_id     = gl_SubgroupInvocationID;
    uint out_row     = wg_row_base + sg_id;  // output row for this subgroup

    if (out_row >= FFN_DIM) return;

    // Each subgroup computes one output row for BOTH gate and up
    // Each thread computes partial dot product over D
    float gate_accum = 0.0;
    float up_accum   = 0.0;

    for (uint k = lane_id; k < D; k += SUBGROUP_SIZE) {
        float16_t x_val = hidden_in.x[k];
        float gate_w = dequant_weight(GATE_WEIGHT_OFFSET, out_row, k);
        float up_w   = dequant_weight(UP_WEIGHT_OFFSET, out_row, k);
        gate_accum += float(x_val) * gate_w;
        up_accum   += float(x_val) * up_w;
    }

    float gate_val = subgroupAdd(gate_accum);
    float up_val   = subgroupAdd(up_accum);

    // SiLU: gate_val * sigmoid(gate_val)
    // sigmoid(x) = 1 / (1 + exp(-x))
    // SiLU(x) = x * sigmoid(x)
    float sigmoid_gate = 1.0 / (1.0 + exp(-gate_val));
    float silu_gate = gate_val * sigmoid_gate;

    // Elementwise multiply: SiLU(gate) * up
    float result = silu_gate * up_val;

    // Write to output
    if (lane_id == 0) {
        // Output layout: [gate_act_0, ..., gate_act_FFN_DIM-1]
        // The "gate" and "up" intermediate values are consumed by the multiply and discarded
        // Only the fused result is written
        hidden_out.y[out_row] = f32tof16(result);
    }
}
```

### 5.4 Quantization

Both gate_weight and up_weight use the same quantization type. They are adjacent in the weight buffer:
```
GATE_WEIGHT_OFFSET = Q_offset + K_offset + V_offset + O_offset + attn_norm_size;
UP_WEIGHT_OFFSET   = GATE_WEIGHT_OFFSET + D * FFN_DIM * bytes_per_element;
```

Same dequant pattern as §2.7.

### 5.5 Shared Memory

**0 bytes** — pure subgroup shuffle reduction, no cross-subgroup needed.

### 5.6 RDNA4 Optimizations

- **Dual accumulation**: Both gate and up accumulators live in VGPRs simultaneously. With ~40 VGPRs/thread, this is well within budget.
- **Fused SiLU**: Computing sigmoid inline avoids a separate dispatch. `exp()` is a single-cycle approximate instruction on RDNA4 (via `v_exp_f32`).
- **Cooperative matrix for FP16 weights**: When available, process 16×16 tiles.

### 5.7 Pipeline Barriers (After)

```
hidden_out (ffn_scratch): SHADER_WRITE → SHADER_READ (for FFN down projection)
```

### 5.8 Expected Occupancy

| Metric | RDNA4 | RDNA2 |
|--------|-------|-------|
| WG threads | 128 | 256 |
| Total WGs | 86 | 43 |
| VGPRs/thread | ~36 | ~36 |
| Active wavefronts | 344 | 172 |
| Occupancy | ~16.8% | ~13.4% |

---

## 6. ffn_down.comp — Down Projection + Residual Add

### 6.1 Purpose

```
hidden = hidden_residual + down_weight[FFN_DIM × D] · ffn_hidden[FFN_DIM]
```

Fused residual add eliminates a separate dispatch:
- Reads `hidden_residual` (the hidden state BEFORE the FFN block)
- Computes `down * ffn_hidden` (matrix-vector multiply, FFN_DIM → D)
- Adds the result to the residual
- Writes back to hidden buffer

### 6.2 Workgroup Configuration

| Parameter | RDNA4 | RDNA2 |
|-----------|-------|-------|
| WG size | 128 (4 × Wave32) | 256 (4 × Wave64) |
| WG count X | ceil(D / WG_SIZE) = 32 | ceil(D / WG_SIZE) = 16 |
| Rows per WG | 4 | 4 |

For FFN_DIM=11008, D=4096:
- RDNA4: 32 WGs of 128 threads
- Each subgroup computes one output row (dot product ffn_hidden[FFN_DIM] · down_weight[row])

### 6.3 GLSL Structure

```glsl
void main() {
    uint wg_row_base = gl_WorkGroupID.x * SUBGROUPS_PER_WG;
    uint sg_id       = gl_SubgroupID;
    uint lane_id     = gl_SubgroupInvocationID;
    uint out_row     = wg_row_base + sg_id;

    if (out_row >= D) return;

    // Dot product: ffn_hidden[FFN_DIM] · down_weight[row][FFN_DIM]
    float accum = 0.0;
    for (uint k = lane_id; k < FFN_DIM; k += SUBGROUP_SIZE) {
        float16_t ffn_val = ffn_hidden_in.x[k];
        float w = dequant_weight(DOWN_WEIGHT_OFFSET, out_row, k);
        accum += float(ffn_val) * w;
    }

    float result = subgroupAdd(accum);

    // Residual add: hidden = hidden_residual + down_result
    float residual = float(hidden_residual.y[out_row]);
    hidden_out.y[out_row] = f32tof16(residual + result);
}
```

### 6.4 Shared Memory

**0 bytes** — pure subgroup shuffle reduction.

### 6.5 Note on Residual Source

The architecture uses double-buffering for hidden state. The residual source (`hidden_residual`) is the hidden state at the START of this layer (before attention and FFN). After attention output projection, the hidden buffer already contains the post-attention residual. The FFN down shader reads from the PRE-FFN hidden buffer (which was saved before the FFN RMS norm) and adds the down projection result.

In practice, the attention output writes to `hidden_buf[1]`. The FFN RMS norm reads `hidden_buf[1]`, normalizes it, and writes to `norm_scratch`. Then `ffn_gate_up` writes to `ffn_scratch`. Then `ffn_down` reads `ffn_scratch` AND reads `hidden_buf[0]` (the ORIGINAL hidden before this layer), computes down projection, adds to `hidden_buf[0]`'s value, and writes to `hidden_buf[0]`. This implements the standard Transformer residual:

```
hidden = hidden + attn(norm(hidden))    # post-attention
hidden = hidden + ffn(norm(hidden))     # post-FFN — ffn_down is fused with this residual
```

The residual add in `ffn_down` must add to the PRE-ATTENTION-hidden, NOT the post-attention-hidden. The hidden state layout in double-buffering ensures this: attention output writes to buffer X, ffn_down adds to buffer Y which has the original value. Buffer Y then becomes the new hidden state.

### 6.6 Pipeline Barriers (After)

```
hidden_out: SHADER_WRITE → SHADER_READ (ready for next layer's RMS norm)
```

### 6.7 Expected Occupancy

| Metric | RDNA4 | RDNA2 |
|--------|-------|-------|
| WG threads | 128 | 256 |
| Total WGs | 32 | 16 |
| VGPRs/thread | ~20 | ~20 |
| Active wavefronts | 128 | 64 |
| Occupancy | ~6.25% | ~5% |

Inner loop over FFN_DIM=11008: 11008 / 32 = 344 iterations per thread (RDNA4) or 11008 / 64 = 172 iterations (RDNA2). Long enough inner loops to hide memory latency.

---

## 7. lm_head.comp — Output Logits Projection

### 7.1 Purpose

Projects final hidden state to vocabulary logits:
```
logits[vocab_size] = lm_head_weight[vocab_size × D] · hidden[D]
```

### 7.2 Workgroup Configuration

| Parameter | RDNA4 | RDNA2 |
|-----------|-------|-------|
| WG size | 128 (4 × Wave32) | 256 (4 × Wave64) |
| WG count X | ceil(VOCAB_SIZE / WG_SIZE) | ceil(VOCAB_SIZE / WG_SIZE) |
| Output per WG | WG_SIZE (one logit per thread) | WG_SIZE |

For VOCAB_SIZE=128000:
- RDNA4: ceil(128000/128) = 1000 WGs
- RDNA2: ceil(128000/256) = 500 WGs

This is the largest dispatch in the pipeline — 1000 WGs for RDNA4.

### 7.3 GLSL Structure

```glsl
void main() {
    uint vocab_id = gl_GlobalInvocationID.x;  // 0..vocab_size-1
    if (vocab_id >= VOCAB_SIZE) return;

    uint lane_id = gl_SubgroupInvocationID;

    float accum = 0.0;
    for (uint k = lane_id; k < D; k += SUBGROUP_SIZE) {
        float16_t h = hidden_in.x[k];
        float w = dequant_weight(LM_HEAD_WEIGHT_OFFSET, vocab_id, k);
        accum += float(h) * w;
    }

    float logit = subgroupAdd(accum);

    // Write as fp32 to host-visible logits buffer (CPU reads for sampling)
    logits_out[gl_GlobalInvocationID.x] = logit;
}
```

### 7.4 Descriptor Bindings

| Set | Binding | Type | Access | Buffer |
|-----|---------|------|--------|--------|
| 0 | 0 | Storage Buffer | Read | LM head weight (shared across all layers, single buffer) |
| 1 | 0 | Storage Buffer | Read | `hidden_in` — final hidden state (D fp16) |
| 1 | 1 | Storage Buffer | Write | `logits_out` — fp32 logits (vocab_size elements) |

Note: The LM head weight is NOT the per-layer lm_head_buffer. It's the dedicated `lm_head_buffer` (separate from layer weights). This can be handled as a special layer_idx or via a separate descriptor.

### 7.5 Output Format

Logits are written as `float` (fp32), not fp16. This is critical for:
1. Precision: fp16 has limited range for softmax temperature scaling. fp32 is required.
2. CPU readability: The logits buffer is host-visible for sampling. fp32 avoids CPU-side conversion.

### 7.6 Pipeline Barriers (After)

```
logits_out: SHADER_WRITE → HOST_READ
srcStage: VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
dstStage: VK_PIPELINE_STAGE_HOST_BIT
```

This is the ONLY host barrier in the entire decode step. It ensures the CPU can safely read logits after GPU completion.

### 7.7 Expected Occupancy

| Metric | RDNA4 | RDNA2 |
|--------|-------|-------|
| WG threads | 128 | 256 |
| Total WGs | 1000 | 500 |
| VGPRs/thread | ~16 | ~16 |
| Active wavefronts | 4000 (but only 2048 slot) | 2000 (but only 1280 slot) |
| Occupancy bound | 100% (wave-slot limited) | 100% (wave-slot limited) |

**LM head is the best-occupancy shader in the pipeline.** 1000 WGs of 128 threads → 4000 wavefronts trying to run on 2048 slots → full occupancy. Each WG runs ~128 inner loop iterations (D=4096 / 32).

### 7.8 RDNA4 Optimizations

- **dmma (dot product with accumulate)**: For INT8/Q8_0 weights, use `dot4add_u8packed` for 4× inner loop throughput.
- **Cooperative matrix for FP16**: `cooperative_matrix<float32_t, 16, 16, 16>` for the large vocab_size matrix.
- **Output buffer is host-visible**: Write-combining optimization — use uncached write to avoid cache pollution for data CPU will immediately read.

---

## 8. token_embed.comp — Token Embedding Lookup

### 8.1 Purpose

Lookup token embeddings for prefill batch:
```
embedded[seq_len][D] = embedding_table[token_ids[seq_len]][D]
```

For decode (single token), this is a trivial lookup. For prefill (seq_len tokens), processes all tokens simultaneously.

### 8.2 Workgroup Configuration (Prefill)

| Parameter | RDNA4 | RDNA2 |
|-----------|-------|-------|
| WG size | 64 (2 × Wave32) | 128 (2 × Wave64) |
| WG count X | ceil(seq_len / WG_SIZE) | ceil(seq_len / WG_SIZE) |
| WG count Y | ceil(D / WG_SIZE_Y) | same |
| WG size Y (spec constant) | 8 | 8 |

Wait — embeddings are a simple lookup. 2D dispatch: X = token index, Y = D block.

Better: 1D dispatch with inner loop.

| Parameter | Value |
|-----------|-------|
| WG size | 128 (RDNA4), 256 (RDNA2) |
| WG count X | ceil(seq_len × D / WG_SIZE) |
| Elements per thread | 1 (or ceil((seq_len × D) / total_threads)) |

For seq_len=256, D=4096: total elements = 1,048,576.
- RDNA4: ceil(1,048,576 / 128) = 8192 WGs — high but acceptable for prefill (one-time cost).
- RDNA2: ceil(1,048,576 / 256) = 4096 WGs.

### 8.3 GLSL Structure

```glsl
void main() {
    uint global_idx = gl_GlobalInvocationID.x;
    uint total_elements = SEQ_LEN * D;
    if (global_idx >= total_elements) return;

    uint token_idx = global_idx / D;
    uint dim_idx   = global_idx % D;

    uint token_id = token_ids[token_idx];

    // Simple lookup: embedding_table[token_id * D + dim_idx]
    float16_t emb = embedding_table[token_id * D + dim_idx];
    hidden_out.y[global_idx] = emb;
}
```

### 8.4 Descriptor Bindings

| Set | Binding | Type | Access | Buffer |
|-----|---------|------|--------|--------|
| 0 | — | — | — | Separate embedding buffer |
| 1 | 0 | Storage Buffer | Read | `token_ids` (uint32_t array) |
| 1 | 1 | Storage Buffer | Write | `hidden_out` — embedded output |

Alternatively: Use a separate descriptor for the embedding table (similar to Set 0 weight array but for embeddings).

### 8.5 Shared Memory

**0 bytes.**

---

## 9. Quantization Block Format Reference

### 9.1 Q8_0

```
Block size: 32 elements
Bytes per block: 34 (2B fp16 d + 32B int8 qs)
Layout: [d_lo][d_hi][qs_0][qs_1]...[qs_31]
Dequant: x[i] = d * qs[i]
```

### 9.2 Q4_K

```
Superblock: 256 elements
Subblocks: 16 subblocks × 16 elements
Bytes per superblock:
  2 + 2 = 4 bytes: fp16 d, fp16 dmin
  4 bytes: 16 × 2-bit scales (packed, 1 byte per 4 scales)
  12 bytes: 16 × 6-bit mins (packed, 192 bits total)
  128 bytes: 256 × 4-bit quants (packed pairs, 4 bits each)
Total: 4 + 4 + 12 + 128 = 148 bytes per 256 elements
Bytes per element: 0.578125

Dequant: x[i] = d * sc[sub] * qs[i] + dmin * (min[sub] - 32)
  where sub = i / 16,
        sc[sub] = (scales_packed[sub/4] >> (2*(sub%4))) & 3,
        min[sub] from packed 6-bit array,
        qs[i] from nibble-packed array
```

### 9.3 Q6_K

```
Superblock: 256 elements
Subblocks: 16 subblocks × 16 elements
Bytes per superblock:
  2 bytes: fp16 d
  2 + 2 = 4 bytes: 16 × 2-bit scales for high bits
  12 bytes: 16 × 6-bit scales (packed)
  192 bytes: 256 × 6-bit quants (packed in 128-byte qh + 64-byte ql)
Total: 2 + 4 + 12 + 192 = 210 bytes per 256 elements
Bytes per element: 0.8203125
```

### 9.4 FP16

```
No block structure. Direct fp16 values.
Bytes per element: 2
```

---

## 10. Build System Integration

### 10.1 SPIR-V Compilation Matrix

```
Source: shaders/*.comp
Compile commands (per shader × per arch × per quant variant):

# RDNA4, FP16
glslc -fshader-stage=compute --target-spv=spv1.4 \
  -DSUBGROUP_SIZE=32 -DWARP_PER_WG=4 -DWG_SIZE_X=128 \
  -DQUANT_TYPE=0 -DENABLE_FP16 -DENABLE_COOPMAT \
  -o spv/attn_qkv_rdna4_fp16.spv shaders/attn_qkv.comp

# RDNA4, Q4_K
glslc -fshader-stage=compute --target-spv=spv1.4 \
  -DSUBGROUP_SIZE=32 -DWARP_PER_WG=4 -DWG_SIZE_X=128 \
  -DQUANT_TYPE=2 -DENABLE_INT8 -DENABLE_DP4A \
  -o spv/attn_qkv_rdna4_q4k.spv shaders/attn_qkv.comp

# RDNA2, FP16
glslc -fshader-stage=compute --target-spv=spv1.4 \
  -DSUBGROUP_SIZE=64 -DWARP_PER_WG=2 -DWG_SIZE_X=128 \
  -DQUANT_TYPE=0 -DENABLE_FP16 \
  -o spv/attn_qkv_rdna2_fp16.spv shaders/attn_qkv.comp

# ... repeat for all combinations
```

### 10.2 SPIR-V → Header Embedding

```cmake
# Convert compiled SPIR-V to C header for embedding
add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/spv_headers/attn_qkv_rdna4_fp16.h
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_SOURCE_DIR}/tools/spv_to_header.py
        --input ${CMAKE_BINARY_DIR}/spv/attn_qkv_rdna4_fp16.spv
        --output ${CMAKE_BINARY_DIR}/spv_headers/attn_qkv_rdna4_fp16.h
        --name attn_qkv_rdna4_fp16_spv
    DEPENDS ${CMAKE_BINARY_DIR}/spv/attn_qkv_rdna4_fp16.spv
)
```

### 10.3 CMake Shader Registry

```cmake
# Must match compile_shaders.ps1 for consistency (see AGENTS.md)
set(VULKAN_SHADER_SOURCES
    shaders/common.glsl
    shaders/push_constants.glsl

    shaders/rms_norm.comp
    shaders/attn_qkv.comp
    shaders/attn_compute.comp
    shaders/attn_output.comp
    shaders/ffn_gate_up.comp
    shaders/ffn_down.comp
    shaders/lm_head.comp
    shaders/token_embed.comp
)

# Generate SPV targets for each shader × arch × quant variant
# Total variants: 8 shaders × 2 archs × up to 5 quant types = ~80 SPV files
# But attn_compute and token_embed don't use weights → only 2 variants each
# And rms_norm doesn't use quantization → 2 variants
# Estimated total: ~50 SPV files
```

### 10.4 Pipeline Creation (Runtime)

```c
// At model load time, after quant type is known:
typedef enum {
    QUANT_FP16 = 0,
    QUANT_Q8_0 = 1,
    QUANT_Q4_K = 2,
    QUANT_Q6_K = 3,
    QUANT_IQ4_XS = 4,
} quant_type_t;

// Pipeline variant key
typedef struct {
    int op_type;     // OP_ATTN_QKV, OP_FFN_GATE_UP, etc.
    int quant_type;  // QUANT_FP16, etc.
    int arch;        // ARCH_RDNA4 or ARCH_RDNA2
} pipeline_key_t;

// Lookup table: pipeline_key → VkPipeline
// Built at LoadModel() using the correct SPV header for the quant type

VkSpecializationMapEntry spec_entries[] = {
    {0, offsetof(spec_constants, SUBGROUP_SIZE),  sizeof(uint32_t)},
    {1, offsetof(spec_constants, WG_SIZE_X),      sizeof(uint32_t)},
    {2, offsetof(spec_constants, D),              sizeof(uint32_t)},
    {3, offsetof(spec_constants, FFN_DIM),        sizeof(uint32_t)},
    {4, offsetof(spec_constants, HEAD_DIM),       sizeof(uint32_t)},
    {5, offsetof(spec_constants, N_HEADS),        sizeof(uint32_t)},
    {6, offsetof(spec_constants, N_KV_HEADS),     sizeof(uint32_t)},
    {7, offsetof(spec_constants, VOCAB_SIZE),     sizeof(uint32_t)},
    {8, offsetof(spec_constants, KV_PAGE_TOKENS), sizeof(uint32_t)},
    {9, offsetof(spec_constants, PAGES_PER_LAYER),sizeof(uint32_t)},
    {10, offsetof(spec_constants, QUANT_TYPE),    sizeof(uint32_t)},
    {11, offsetof(spec_constants, WARP_PER_WG),   sizeof(uint32_t)},
};
```

---

## 11. Shader Summary Matrix

| Shader | WGs (RDNA4) | WGs (RDNA2) | WG Threads (RDNA4) | WG Threads (RDNA2) | VGPRs | LDS (bytes) | Occupancy (RDNA4) | Key Operation |
|--------|------------|------------|--------------------|---------------------|-------|------------|--------------------|---------------|
| rms_norm | ceil(D/256)=16 | ceil(D/128)=32 | 256 | 128 | 12 | 32 | 6.25% | subgroupAdd (reduction) |
| attn_qkv | 192 | 96 | 128 | 256 | 48 | 0 | 37.5% | subgroupAdd + dp4a/coopmat |
| attn_compute | 32 | 32 | 128 | 128 | 40 | 16400 | 6.25% | subgroupAdd/Max + LDS softmax |
| attn_output | ceil(D/128)=32 | ceil(D/256)=16 | 128 | 256 | 20 | 0 | 6.25% | subgroupAdd + dp4a/coopmat |
| ffn_gate_up | ceil(FFN_DIM/128)=86 | ceil(FFN_DIM/256)=43 | 128 | 256 | 36 | 0 | 16.8% | subgroupAdd × 2 + SiLU + mul |
| ffn_down | ceil(D/128)=32 | ceil(D/256)=16 | 128 | 256 | 20 | 0 | 6.25% | subgroupAdd + dp4a/coopmat |
| lm_head | ceil(VOCAB/128)=1000 | ceil(VOCAB/256)=500 | 128 | 256 | 16 | 0 | 100% | subgroupAdd + dp4a/coopmat |
| token_embed | ceil(seq×D/128) | ceil(seq×D/256) | 128 | 256 | 8 | 0 | varies | Memory lookup (no compute) |

---

## 12. Dispatch Ordering + Barrier Map (Single Decode Step CB)

```
Layer 0 (hidden_buf[0] → hidden_buf[1]):
  1. rms_norm           (in: hidden_buf[0], out: norm_scratch)
     BARRIER: norm_scratch WRITE→READ
  2. attn_qkv           (in: norm_scratch, out: qkv_out + k_cache + v_cache)
     BARRIER: qkv_out WRITE→READ, k_cache WRITE→READ, v_cache WRITE→READ
  3. attn_compute       (in: qkv_out + k_cache + v_cache, out: attn_scratch)
     BARRIER: attn_scratch WRITE→READ
  4. attn_output        (in: attn_scratch, out: hidden_buf[1] residual)
     BARRIER: hidden_buf[1] WRITE→READ
  5. rms_norm           (in: hidden_buf[1], out: norm_scratch)
     BARRIER: norm_scratch WRITE→READ
  6. ffn_gate_up        (in: norm_scratch, out: ffn_scratch)
     BARRIER: ffn_scratch WRITE→READ
  7. ffn_down           (in: ffn_scratch + hidden_buf[0](residual), out: hidden_buf[0])
     BARRIER: hidden_buf[0] WRITE→READ

Layer 1 (hidden_buf[0] → hidden_buf[1]):
  ... same pattern, buffers swapped ...

... layers 2-31 ...

Final:
  225. lm_head          (in: hidden_buf[n_layers%2], out: logits_buf)
     BARRIER: logits_buf SHADER_WRITE → HOST_READ
```

### 12.1 Barrier Count Per Decode

Per layer: 7 barriers × 32 layers = 224 + 1 (lm_head) = **225 barriers total**.
Each barrier: `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT → VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT`.

Resources transitioning per layer:
- `norm_scratch`: 2 transitions (attn norm write→read, FFN norm write→read)
- `qkv_out`: 1 transition
- `attn_scratch`: 1 transition
- `ffn_scratch`: 1 transition
- `hidden_buf[X]`: 2 transitions (attn output → readable, FFN output → readable)
- `k_cache + v_cache`: 1 transition (write→read after QKV, no barrier needed after attention because KV cache is read-only for the rest of the step)

---

## 13. VGPR Allocation Strategy Per Shader

| Shader | Variables in VGPRs | Count | Target ≤ 64? |
|--------|-------------------|-------|-------------|
| rms_norm | val, res, sq, inv_rms, w (5 floats) | ~10 VGPRs | Yes |
| attn_qkv | x_val accumulator (fp32), weight dequant state (uint + floats), 4-8 VGPRs for loop variables | ~48 VGPRs | Yes |
| attn_compute | Q element, partial QK result, softmax state, V accumulator, loop vars | ~40 VGPRs | Yes |
| attn_output | accumulator, dequant state | ~20 VGPRs | Yes |
| ffn_gate_up | gate_accum (fp32), up_accum (fp32), x_val, gate_w, up_w, loop vars | ~36 VGPRs | Yes |
| ffn_down | accumulator, ffn_val, w | ~20 VGPRs | Yes |
| lm_head | accumulator, w, h | ~16 VGPRs | Yes |
| token_embed | token_id, global_idx | ~8 VGPRs | Yes |

All shaders target ≤ 64 VGPRs. At 64 VGPRs:
- RDNA4 Wave32: 2 concurrent waves per SIMD32 (64 VGPRs / 128 SIMD VGPRs)
- RDNA2 Wave64: 1 concurrent wave per CU (64 VGPRs / 64 SIMD VGPRs) — borderline but functional

---

## 14. RDNA4 Cooperative Matrix Usage (When Weights Are FP16)

```glsl
#if defined(COOPMAT_ENABLED) && SUBGROUP_SIZE == 32 && QUANT_TYPE == 0

// Use VK_KHR_cooperative_matrix for 16×16×16 GEMM tiles
// Effective only when both input and weight are FP16
// On RDNA4, this maps to WMMA instructions

// Accumulator: 16×16 matrix of fp32
coopmat<float32_t, gl_ScopeSubgroup, 16, 16, 16> C;

// Input tile: 16×16 of fp16 (broadcast from 1×16 vector?)
// For matrix-vector multiply, input is 1×D, weight is D×D
// We can treat it as subgroup-cooperative: each subgroup loads 16 elements
// of input, 16×16 weight tiles, accumulates 16×16 partial sums

// Initialize accumulator
for (int i = 0; i < C.length(); i++) {
    C[i] = float32_t(0.0);
}

// Tile loop
for (uint k = 0; k < D; k += 16) {
    // Load 16-element input vector slice
    coopmat<float16_t, gl_ScopeSubgroup, 16, 1, 16> A;
    // Load 16×16 weight tile
    coopmat<float16_t, gl_ScopeSubgroup, 16, 16, 16> B;

    // cooperative matrix multiply-accumulate
    C = coopmatAdd(C, coopmatMulAdd(A, B, C));
}

// Extract result from accumulator to output
// Each thread gets one element of the 16×16 result
#endif
```

For matrix-vector multiply (decode, batch=1), the cooperative matrix benefits are marginal because each 16×16 tile multiply only reuses the input vector once across 16 output rows. The real benefit is for prefill (batch=seq_len) where both matrices are large.

For decode, dp4a (integer dot product) is more impactful for quantized weights, and scalar fp16 accumulation is sufficient for fp16 weights.

---

## 15. Appendix: SPIR-V Validation Checklist

Before integration, each SPIR-V module must pass:

```
spirv-val --target-env vulkan1.4 <shader>.spv
```

Validation checks:
1. All required capabilities declared (Shader, GroupNonUniform, Int8, Float16, etc.)
2. No undefined references
3. Entry point matches pipeline creation
4. Local size matches specialization constants
5. Descriptor bindings match pipeline layout
6. No out-of-bounds access in static analysis
7. Cooperative matrix types valid for target (KHR-spec compliant)
8. Subgroup size constraint `requiredSubgroupSize` compatible with declared capabilities

---

## 16. Performance Projections

### 16.1 Decode Step Latency Budget (Qwen3.5-9B Q4_K, D=4096, FFN_DIM=11008)

| Operation | Memory (MB) | Bandwidth (GB/s) | Time (us) | Compute (TOPS) | Time (us) | Total (us) |
|-----------|-------------|-----------------|-----------|----------------|-----------|------------|
| rms_norm (×2) | 0.016 | 960 | 0.02 | negligible | 5 | 5 |
| QKV proj | 14.16 | 960 | 14.8 | 0.03 | 50 | 65 |
| Attention | 2.06 | 960 | 2.1 | 0.01 | 15 | 17 |
| O-proj | 9.44 | 960 | 9.8 | 0.02 | 30 | 40 |
| FFN gate+up | 50.72 | 960 | 52.8 | 0.09 | 70 | 123 |
| FFN down | 25.36 | 960 | 26.4 | 0.05 | 40 | 66 |
| Subtotal (×32 layers) | 3256 | | 3395 | | 6720 | 10115 |
| LM head | 295 | 960 | 307 | 0.52 | 500 | 807 |
| **Total** | | | **~3.7ms** | | **~7.2ms** | **~10.9ms** |

**Expected throughput: ~92 tok/s** (decode only, no prefill, no sampling overhead).

### 16.2 Bottleneck Analysis

Memory bandwidth is the dominant limiter:
- FFN gate+up reads 50.72 MB per layer × 32 layers = 1,623 MB per step
- QKV+O reads 23.60 MB per layer × 32 = 755 MB per step
- **Total weight reads per decode: ~2,378 MB**
- At 960 GB/s (RDNA4, 100% efficiency): 2.48ms
- At 70% efficiency (more realistic for 9B model on RDNA4): 3.55ms

Compute is secondary:
- ~7.2ms of compute for dequant + dot products + softmax + activation
- Compute is memory-latency-limited: ALU utilization ~40% due to memory stalls

**Realistic projection: 80-95 tok/s** for Qwen3.5-9B Q4_K on RX 9070 XT, consistent with HIP backend measured 103 tok/s on NVFP4 (which is slightly lighter than Q4_K).

### 16.3 Prefill Performance (256-token prompt, Qwen3.5-9B)

Prefill processes 256 tokens in parallel. For FFN gate+up, this is a [256, FFN_DIM] × [FFN_DIM, D] matrix × [D, FFN_DIM] matrix multiply.

Memory reads scale with BATCH × model_size_per_layer, but computation scales quadratically in seq_len for attention. Prefill is compute-bound below ~512 tokens, becoming memory-bound beyond that.

For 256-token prompt:
- QKV projection: 256 × 256-tile GEMM → compute-bound
- Attention: O(seq_len² × head_dim) = O(256² × 128) = ~8.4M FLOPs per head → very small
- FFN: 256 × 11008 × 4096 GEMM → ~23B FLOPs per layer × 32 = 736B FLOPs
- Total compute: ~750B FLOPs / 16 TFLOPS (FP16) = ~47ms

**Expected prefill latency for 256 tokens: ~50-60ms.**

---

*End of Compute Shader Specification — v1.0*
*Generated for Vulkan LLM Inference Engine targeting AMD RDNA4 (gfx1201)*
