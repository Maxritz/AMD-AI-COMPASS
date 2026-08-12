// ─── Specialization Constants ───────────────────────────────────────────────
#extension GL_KHR_shader_subgroup_arithmetic : require

layout(constant_id = 0)  const uint SPEC_SUBGROUP_SIZE  = 32;
layout(constant_id = 1)  const uint SPEC_WG_SIZE_X      = 64;
layout(constant_id = 2)  const uint SPEC_D              = 4096;
layout(constant_id = 3)  const uint SPEC_FFN_DIM        = 11008;
layout(constant_id = 4)  const uint SPEC_HEAD_DIM       = 128;
layout(constant_id = 5)  const uint SPEC_N_HEADS        = 32;
layout(constant_id = 6)  const uint SPEC_N_KV_HEADS     = 8;
layout(constant_id = 7)  const uint SPEC_VOCAB_SIZE     = 128000;
layout(constant_id = 8)  const uint SPEC_KV_PAGE_TOKENS = 256;
layout(constant_id = 9)  const uint SPEC_PAGES_PER_LAYER = 16;
layout(constant_id = 10) const uint SPEC_QUANT_TYPE     = 0;
layout(constant_id = 11) const uint SPEC_WARP_PER_WG    = 2;
layout(constant_id = 12) const uint SPEC_MAX_SEQ_LEN    = 4096;

#ifndef SUBGROUP_SIZE
#define SUBGROUP_SIZE SPEC_SUBGROUP_SIZE
#endif
#define WG_SIZE_X     SPEC_WG_SIZE_X
#define D             SPEC_D
#define FFN_DIM       SPEC_FFN_DIM
#define HEAD_DIM      SPEC_HEAD_DIM
#define N_HEADS       SPEC_N_HEADS
#define N_KV_HEADS    SPEC_N_KV_HEADS
#define VOCAB_SIZE    SPEC_VOCAB_SIZE
#define KV_PAGE_TOKENS SPEC_KV_PAGE_TOKENS
#define PAGES_PER_LAYER SPEC_PAGES_PER_LAYER
#define WARP_PER_WG   SPEC_WARP_PER_WG
#define MAX_SEQ_LEN   SPEC_MAX_SEQ_LEN

#define WARP_SIZE     SUBGROUP_SIZE

// ─── Quantization Type (set by preprocessor at compile time) ────────────────

#ifndef QUANT_TYPE
#ifdef QUANT_FP16
#define QUANT_TYPE 0
#elif defined(QUANT_Q8_0)
#define QUANT_TYPE 1
#elif defined(QUANT_Q4_K)
#define QUANT_TYPE 2
#elif defined(QUANT_Q6_K)
#define QUANT_TYPE 3
#elif defined(QUANT_IQ4_XS)
#define QUANT_TYPE 4
#else
#define QUANT_TYPE 0
#endif
#endif

// ─── Push Constants (must match C++ push_constants_t byte-for-byte) ────────

layout(push_constant, scalar) uniform PushConstants {
    uint  layer_idx;
    uint  kv_cache_pos;
    uint  seq_len;
    uint  head_dim;
    uint  num_heads;
    uint  num_kv_heads;
    // page_size_tokens/pages_per_layer/temperature removed -- see common.h's
    // push_constants_t for why (dead fields, freed to fit the 128-byte budget).
    float attn_scale;
    float rope_theta;
    float norm_eps;
    uint  norm_type;          // 0=attn_norm, 1=ffn_norm, 2=output_norm, 3=q_norm, 4=k_norm
    uint  rope_is_neox;       // 1=NEOX split-half pairing, 0=NORM interleaved pairing
    int   token_id;           // current input token for embedding
    uint  attn_norm_offset;   // byte offset to attn_norm weights
    uint  ffn_norm_offset;    // byte offset to ffn_norm weights
    uint  output_norm_offset; // byte offset to output_norm weights
    uint  q_norm_offset;      // byte offset to q_norm weights (qwen3 per-head)
    uint  k_norm_offset;      // byte offset to k_norm weights (qwen3 per-head)
    uint  q_offset;
    uint  k_offset;
    uint  v_offset;
    uint  o_offset;
    uint  gate_offset;
    uint  up_offset;
    uint  down_offset;
    // Optional QKV bias (qwen2/qwen3-family). has_qkv_bias=0 means the
    // *_bias_offset fields are unused -- must check the flag, not assume a
    // zero offset means "no bias" (offset 0 is a valid real offset). These 4
    // fields exist in common.h's push_constants_t but were missing here --
    // this mirror is hand-maintained, not auto-synced from the C++ struct.
    uint  has_qkv_bias;
    uint  q_bias_offset;
    uint  k_bias_offset;
    uint  v_bias_offset;
    // MiniCPM-family depth-scaling (see common.h push_constants_t); 1.0 = no-op
    // for every other architecture, so these multiplies are always safe to apply.
    float embedding_scale;
    float residual_scale;
    float logit_scale;
} pc;

// ─── Descriptor Set Layout Constants ────────────────────────────────────────

#define WEIGHT_SET 0
#define IO_SET     1
#define TABLE_SET  2

#define BINDING_WEIGHTS    0
#define BINDING_HIDDEN_IN  0
#define BINDING_HIDDEN_OUT 1
#define BINDING_K_CACHE    2
#define BINDING_V_CACHE    3
#define BINDING_SCRATCH    4
#define BINDING_ROPE_FREQS 0
#define BINDING_PAGE_TABLE 1

// ─── Subgroup Reduction Helpers ─────────────────────────────────────────────
// Use native subgroup arithmetic intrinsics (VK_KHR_shader_subgroup_arithmetic)
// for 1-instruction reductions instead of N/2-instruction XOR butterfly.

float warpReduceSum(float val) {
    return subgroupAdd(val);
}

float warpReduceMax(float val) {
    return subgroupMax(val);
}

uint warpReduceSumUint(uint val) {
    return subgroupAdd(val);
}

// ─── FP16 ↔ FP32 Conversion Helpers ────────────────────────────────────────

float f16tof32(float16_t v) {
    return float(v);
}

float16_t f32tof16(float v) {
    return float16_t(v);
}

float f16tof32_from_bytes(uint lo, uint hi) {
    uint raw = lo | (hi << 8u);
    vec2 v = unpackFloat2x16(raw);
    return v.x;
}

// ─── Set 0: Weight Buffer (descriptor-indexed variable-count array) ─────────

layout(set = WEIGHT_SET, binding = BINDING_WEIGHTS) readonly buffer WeightBufs {
    uint8_t data[];
} weight_array[];

// Reads one F16 scalar (2 raw bytes, uploaded by the host as QUANT_FP16 --
// see vk_model.cpp's f32_to_f16_bytes conversion for any GGUF F32 tensor,
// which includes norm weights and qkv bias) from the per-layer weight buffer.
float read_layer_f16(uint byte_offset) {
    return f16tof32_from_bytes(
        weight_array[nonuniformEXT(pc.layer_idx)].data[byte_offset],
        weight_array[nonuniformEXT(pc.layer_idx)].data[byte_offset + 1u]);
}

// ─── Quantization Dequant Functions ─────────────────────────────────────────
// All dequant functions operate on a raw byte buffer (uint8_t array).
// The weight buffer is accessed via weight_array[nonuniformEXT(layer_idx)].data[]
// which is a flat uint8_t array. The caller passes the byte offset into the buffer.

// ── Q8_0 ────────────────────────────────────────────────────────────────────
// Block size: 32 elements, bytes per block: 34 (2B fp16 d + 32B int8 qs)
// Layout: [d_lo][d_hi][qs_0][qs_1]...[qs_31]
// Dequant: value[i] = d * int(qs[i])

#ifdef GL_EXT_shader_explicit_arithmetic_types_int8

float dequant_q8_0(uint block_start, uint elem_idx) {
    uint byte_off = block_start + elem_idx + 2u;
    float d = f16tof32_from_bytes(
        weight_array[pc.layer_idx].data[block_start],
        weight_array[pc.layer_idx].data[block_start + 1u]
    );
    // Explicit sign extension: `int8_t(...)` cast does NOT reliably sign-extend
    // on this compiler/driver; quants >= 128 must become negative. (Reading them
    // as unsigned makes every negative weight large positive -> no cancellation
    // -> Q projection ~12888 instead of ~5.6.)
    int qs = int(weight_array[pc.layer_idx].data[byte_off]);
    if (qs >= 128) qs -= 256;
    return d * float(qs);
}

#endif

// ── Q4_K ────────────────────────────────────────────────────────────────────
// Superblock: 256 elements, 144 bytes (llama.cpp block_q4_K)
// Layout:
//   Offset   0..  1: d (fp16 global scale)
//   Offset   2..  3: dmin (fp16 global min)
//   Offset   4.. 15: scales[12] (8 scales + 8 mins, 6-bit each, packed)
//   Offset  16..143: qs[128] (256 x 4-bit quants, 2 per byte)
// 8 groups of 32 weights; group g has scale/min pair from get_scale_min_k4(g).
// Decode: y = d * sc * q - dmin * m,  q in [0,15]

#ifdef GL_EXT_shader_explicit_arithmetic_types_int8

float dequant_q4_k(uint block_start, uint elem_idx) {
    uint group = elem_idx / 64u;
    uint sub   = elem_idx & 63u;
    uint sc_idx = group * 2u + (sub >= 32u ? 1u : 0u);

    uint qs_byte = uint(weight_array[pc.layer_idx].data[block_start + 16u + group * 32u + (sub & 31u)]);
    uint q4      = (sub >= 32u) ? (qs_byte >> 4u) : (qs_byte & 0x0Fu);

    uint sc, m;
    if (sc_idx < 4u) {
        sc = uint(weight_array[pc.layer_idx].data[block_start + 4u + sc_idx]) & 0x3Fu;
        m  = uint(weight_array[pc.layer_idx].data[block_start + 8u + sc_idx]) & 0x3Fu;
    } else {
        sc = (uint(weight_array[pc.layer_idx].data[block_start + 8u + sc_idx]) & 0x0Fu)
           | ((uint(weight_array[pc.layer_idx].data[block_start + sc_idx]) >> 6u) << 4u);
        m  = (uint(weight_array[pc.layer_idx].data[block_start + 8u + sc_idx]) >> 4u)
           | ((uint(weight_array[pc.layer_idx].data[block_start + 4u + sc_idx]) >> 6u) << 4u);
    }

    float d  = f16tof32_from_bytes(weight_array[pc.layer_idx].data[block_start], weight_array[pc.layer_idx].data[block_start + 1u]);
    float dm = f16tof32_from_bytes(weight_array[pc.layer_idx].data[block_start + 2u], weight_array[pc.layer_idx].data[block_start + 3u]);

    return d * float(sc) * float(q4) - dm * float(m);
}

#endif

// ── Q6_K ────────────────────────────────────────────────────────────────────
// Superblock: 256 elements, 210 bytes (llama.cpp block_q6_K)
// Layout:
//   Offset   0..127: ql[128] (low 4 bits, 2 elements per byte)
//   Offset 128..191: qh[64]  (high 2 bits, 4 elements per byte)
//   Offset 192..207: scales[16] (int8, one per 16-element sub-block)
//   Offset 208..209: d (fp16 super-block scale)
// Decode: y = d * sc * q,  q in [-32, 31]

#ifdef GL_EXT_shader_explicit_arithmetic_types_int8

float dequant_q6_k(uint block_start, uint elem_idx) {
    // llama.cpp block_q6_K (256 elems, 210 bytes):
    //   ql[128] at 0, qh[64] at 128, scales[16] at 192, d(fp16) at 208
    // dequantize_row_q6_K: for each 128-half: 32 l's produce 4 elems
    uint h2      = elem_idx / 128u;
    uint l    = elem_idx % 128u;
    uint quad = l / 32u;       // 0..3 quarter within the half
    uint w    = l % 32u;       // 0..31
    uint is   = w / 16u;       // 0..1

    uint ql_byte = uint(weight_array[pc.layer_idx].data[block_start + h2 * 64u + ((quad == 1u || quad == 3u) ? (w + 32u) : w)]);
    uint qh_byte = uint(weight_array[pc.layer_idx].data[block_start + 128u + h2 * 32u + w]);

    uint low4  = ((quad == 2u || quad == 3u) ? (ql_byte >> 4u) : (ql_byte & 0x0Fu));
    uint high2 = (qh_byte >> (quad * 2u)) & 3u;

    int q6 = int(low4 | (high2 << 4u)) - 32;

    float d = f16tof32_from_bytes(weight_array[pc.layer_idx].data[block_start + 208u], weight_array[pc.layer_idx].data[block_start + 209u]);
    int sc_byte = int(weight_array[pc.layer_idx].data[block_start + 192u + h2 * 8u + is + quad * 2u]);
    if (sc_byte >= 128) sc_byte -= 256;  // explicit sign extension (int8_t cast is unreliable on RDNA4)
    float scale = float(sc_byte);

    return d * scale * float(q6);
}

#endif

// ── Q5_K ────────────────────────────────────────────────────────────────────
// Superblock: 256 elements, 176 bytes (llama.cpp block_q5_K)
// Layout:
//   Offset   0..  1: d (fp16 global scale)
//   Offset   2..  3: dmin (fp16 global min)
//   Offset   4.. 15: scales[12] (same packing as Q4_K)
//   Offset  16.. 47: qh[32] (high bits, one bit per element)
//   Offset  48..175: qs[128] (low 4 bits, 2 elements per byte)
// 8 groups of 32 weights; group g uses qh bit mask (1|2) << (2*g).
// Decode: y = d * sc * (low4 | high5th_bit*16) - dmin * m

#ifdef GL_EXT_shader_explicit_arithmetic_types_int8

float dequant_q5_k(uint block_start, uint elem_idx) {
    uint group = elem_idx / 64u;
    uint sub   = elem_idx & 63u;
    uint l     = sub & 31u;
    uint sc_idx = group * 2u + (sub >= 32u ? 1u : 0u);

    uint qs_byte = uint(weight_array[pc.layer_idx].data[block_start + 48u + group * 32u + l]);
    uint qh_byte = uint(weight_array[pc.layer_idx].data[block_start + 16u + l]);

    uint q = (sub >= 32u) ? (qs_byte >> 4u) : (qs_byte & 0x0Fu);
    uint bit = (sub >= 32u) ? (2u << (2u * group)) : (1u << (2u * group));
    if ((qh_byte & bit) != 0u) q |= 16u;

    uint sc, m;
    if (sc_idx < 4u) {
        sc = uint(weight_array[pc.layer_idx].data[block_start + 4u + sc_idx]) & 0x3Fu;
        m  = uint(weight_array[pc.layer_idx].data[block_start + 8u + sc_idx]) & 0x3Fu;
    } else {
        sc = (uint(weight_array[pc.layer_idx].data[block_start + 8u + sc_idx]) & 0x0Fu)
           | ((uint(weight_array[pc.layer_idx].data[block_start + sc_idx]) >> 6u) << 4u);
        m  = (uint(weight_array[pc.layer_idx].data[block_start + 8u + sc_idx]) >> 4u)
           | ((uint(weight_array[pc.layer_idx].data[block_start + 4u + sc_idx]) >> 6u) << 4u);
    }

    float d  = f16tof32_from_bytes(weight_array[pc.layer_idx].data[block_start], weight_array[pc.layer_idx].data[block_start + 1u]);
    float dm = f16tof32_from_bytes(weight_array[pc.layer_idx].data[block_start + 2u], weight_array[pc.layer_idx].data[block_start + 3u]);

    return d * float(sc) * float(q) - dm * float(m);
}

#endif

// ── IQ4_XS ──────────────────────────────────────────────────────────────────
// Superblock: 256 elements, 135 bytes (llama.cpp block_iq4_xs)
// Layout:
//   Offset   0..  1: d (fp16 scale)
//   Offset   2..129: qs[128] (256 x 4-bit, 2 per byte)
//   Offset 130..133: scales_l[4] (low 4 bits of 8 group scales)
//   Offset 134     : scales_h[1] (high 2 bits of 8 group scales)
// 8 groups of 32 elements. Group ib scale: 6-bit signed (ls-32).
// Decode: y = d * (ls - 32) * kvalues_iq4nl[q]

#ifdef GL_EXT_shader_explicit_arithmetic_types_int8

float dequant_iq4_xs(uint block_start, uint elem_idx) {
    uint ib = elem_idx / 32u;
    uint j  = elem_idx & 31u;

    uint ls = ((uint(weight_array[pc.layer_idx].data[block_start + 130u + ib / 2u]) >> (4u * (ib & 1u))) & 0x0Fu)
            | (((uint(weight_array[pc.layer_idx].data[block_start + 134u]) >> (2u * ib)) & 3u) << 4u);
    float dl = f16tof32_from_bytes(weight_array[pc.layer_idx].data[block_start], weight_array[pc.layer_idx].data[block_start + 1u])
             * float(int(ls) - 32);

    uint qs_byte = uint(weight_array[pc.layer_idx].data[block_start + 2u + ib * 16u + j / 2u]);
    uint qi      = ((j & 1u) != 0u) ? (qs_byte >> 4u) : (qs_byte & 0x0Fu);

    float kval;
    if      (qi == 0u)  kval = -127.0;
    else if (qi == 1u)  kval = -104.0;
    else if (qi == 2u)  kval = -83.0;
    else if (qi == 3u)  kval = -65.0;
    else if (qi == 4u)  kval = -49.0;
    else if (qi == 5u)  kval = -35.0;
    else if (qi == 6u)  kval = -23.0;
    else if (qi == 7u)  kval = -12.0;
    else if (qi == 8u)  kval = -3.0;
    else if (qi == 9u)  kval = 5.0;
    else if (qi == 10u) kval = 13.0;
    else if (qi == 11u) kval = 22.0;
    else if (qi == 12u) kval = 32.0;
    else if (qi == 13u) kval = 43.0;
    else if (qi == 14u) kval = 56.0;
    else                kval = 71.0;
    return dl * kval;
}

#endif

// ── FP16 Direct Read ────────────────────────────────────────────────────────

float dequant_f16(uint offset) {
    uint elem_off = offset * 2u;
    return f16tof32_from_bytes(weight_array[pc.layer_idx].data[elem_off], weight_array[pc.layer_idx].data[elem_off + 1u]);
}

// ── Unified Dequant (dispatch based on QUANT_TYPE) ──────────────────────────
// Row-major weight layout: weight[row][col] where row is output dimension,
// col is input dimension.
// For quantized weights, we compute the block offset:
//   Q8_0:  superblock = col / 32,  bytes_per_sb = 34
//   Q4_K:  superblock = col / 256, bytes_per_sb = 144
//   Q6_K:  superblock = col / 256, bytes_per_sb = 210

#if QUANT_TYPE == 0
// FP16: direct layout [D_out, D_in]
float dequant_weight(uint weight_base, uint row, uint col) {
    uint offset = weight_base + row * D + col;
    return dequant_f16(offset);
}
#elif QUANT_TYPE == 1
// Q8_0: blocks of 32 elements
float dequant_weight(uint weight_base, uint row, uint col) {
    uint block_idx       = col / 32u;
    uint elem_in_block   = col & 31u;
    uint row_stride      = (D / 32u) * 34u;  // 34 bytes per block
    uint block_start     = weight_base + row * row_stride + block_idx * 34u;
    return dequant_q8_0(block_start, elem_in_block);
}
#elif QUANT_TYPE == 2
// Q4_K: superblocks of 256 elements, 144 bytes each
float dequant_weight(uint weight_base, uint row, uint col) {
    uint superblock      = col / 256u;
    uint elem_in_sb      = col & 255u;
    uint row_stride      = (D / 256u) * 144u;
    uint sb_start        = weight_base + row * row_stride + superblock * 144u;
    return dequant_q4_k(sb_start, elem_in_sb);
}
#elif QUANT_TYPE == 3
// Q6_K: superblocks of 256 elements, 210 bytes each
float dequant_weight(uint weight_base, uint row, uint col) {
    uint superblock      = col / 256u;
    uint elem_in_sb      = col & 255u;
    uint row_stride      = (D / 256u) * 210u;
    uint sb_start        = weight_base + row * row_stride + superblock * 210u;
    return dequant_q6_k(sb_start, elem_in_sb);
}
#elif QUANT_TYPE == 4
// IQ4_XS: superblocks of 256 elements, 135 bytes each
float dequant_weight(uint weight_base, uint row, uint col) {
    uint superblock      = col / 256u;
    uint elem_in_sb      = col & 255u;
    uint row_stride      = (D / 256u) * 135u;
    uint sb_start        = weight_base + row * row_stride + superblock * 135u;
    return dequant_iq4_xs(sb_start, elem_in_sb);
}
#elif QUANT_TYPE == 5
// Q5_K: superblocks of 256 elements, 176 bytes each
float dequant_weight(uint weight_base, uint row, uint col) {
    uint superblock      = col / 256u;
    uint elem_in_sb      = col & 255u;
    uint row_stride      = (D / 256u) * 176u;
    uint sb_start        = weight_base + row * row_stride + superblock * 176u;
    return dequant_q5_k(sb_start, elem_in_sb);
}
#endif

// ─── Set 1: Layer I/O (push descriptors) ────────────────────────────────────

#ifndef CUSTOM_IO_BINDINGS
layout(set = IO_SET, binding = BINDING_HIDDEN_IN,  scalar) readonly buffer InputBuf  { float16_t x[]; } hidden_in;
#ifndef CUSTOM_OUTPUT_BUF
layout(set = IO_SET, binding = BINDING_HIDDEN_OUT, scalar)          buffer OutputBuf { float16_t y[]; } hidden_out;
#endif
layout(set = IO_SET, binding = BINDING_K_CACHE,    scalar) buffer KCacheBuf  { float16_t k[]; } k_cache;
layout(set = IO_SET, binding = BINDING_V_CACHE,    scalar) buffer VCacheBuf  { float16_t v[]; } v_cache;
layout(set = IO_SET, binding = BINDING_SCRATCH,    scalar)          buffer ScratchBuf { float16_t s[]; } scratch;
#endif

// ─── Set 2: Static Tables ───────────────────────────────────────────────────

layout(set = TABLE_SET, binding = BINDING_ROPE_FREQS, scalar) readonly buffer RopeBuf    { float cs[]; } rope_freqs;
layout(set = TABLE_SET, binding = BINDING_PAGE_TABLE,  scalar) readonly buffer PageTable  { uint ids[]; } kv_page_table;

// ─── KV Cache Page Table Helper ─────────────────────────────────────────────
// Compute byte offset into paged KV cache for element (kv_head, token_pos, dim_idx).
// Returns element index (not byte offset) — callers multiply by sizeof(fp16) = 2
// or use implicit indexing in float16_t arrays.

uint kv_cache_page_offset(uint kv_head, uint token_pos, uint dim_idx, bool is_v) {
    uint token_in_page = token_pos % KV_PAGE_TOKENS;
    uint page_idx      = token_pos / KV_PAGE_TOKENS;
    uint table_idx     = pc.layer_idx * PAGES_PER_LAYER + page_idx;
    uint phys_page     = kv_page_table.ids[table_idx];

    if (phys_page == 0xFFFFFFFFu) return 0u;

    // Layout per physical page (fp16 elements):
    //   [K: kv_heads*head_dim*tokens | V: kv_heads*head_dim*tokens]
    // Matches host vk_kv_cache_addr: page_stride = 2 * kv_heads*head_dim*tokens.
    uint page_elems  = KV_PAGE_TOKENS * N_KV_HEADS * HEAD_DIM;
    uint kv_stride   = page_elems * 2u;
    uint page_base   = phys_page * kv_stride;
    uint half_offset = is_v ? page_elems : 0u;
    uint token_base  = token_in_page * N_KV_HEADS * HEAD_DIM;
    uint head_offset = kv_head * HEAD_DIM;
    return page_base + half_offset + token_base + head_offset + dim_idx;
}

// ─── RoPE Frequency Computation (on-the-fly for single-token decode) ────────

void rope_cos_sin(uint pair_idx, float pos, out float cos_val, out float sin_val) {
    float freq = 1.0 / pow(pc.rope_theta, (2.0 * float(pair_idx)) / float(HEAD_DIM));
    float theta = pos * freq;
    cos_val = cos(theta);
    sin_val = sin(theta);
}

// ─── Activation Functions ────────────────────────────────────────────────────

float sigmoid_f32(float x) {
    return 1.0 / (1.0 + exp(-x));
}

float silu_f32(float x) {
    return x * sigmoid_f32(x);
}
