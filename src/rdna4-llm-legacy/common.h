#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include <string>
#include <unordered_map>
#include <chrono>

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

#if defined(_MSC_VER) && !defined(DEBUG_NO_BREAK)
#define RDNA4_DEBUG_BREAK() __debugbreak()
#else
#define RDNA4_DEBUG_BREAK() do { } while(0)
#endif

#ifndef NDEBUG
#define VK_CHECK(call) do {                                                     \
    VkResult _vk_result = (call);                                               \
    if (_vk_result != VK_SUCCESS) {                                             \
        fprintf(stderr, "[VK_ERROR] %s:%d: %s returned VkResult=%d\n",         \
                __FILE__, __LINE__, #call, (int)_vk_result);                    \
        RDNA4_DEBUG_BREAK();                                                    \
        exit(1);                                                                \
    }                                                                           \
} while(0)
#else
#define VK_CHECK(call) (call)
#endif

#define RDNA4_LOG(fmt, ...)    do { fprintf(stdout, "[RDNA4] " fmt "\n", ##__VA_ARGS__); } while(0)
#define RDNA4_ERROR(fmt, ...)  do { fprintf(stderr, "[RDNA4 ERROR] " fmt "\n", ##__VA_ARGS__); } while(0)

#define MAX_LAYERS    128
#define MAX_SEQ_LEN   4096
#define KV_PAGE_TOKENS 256

#define RDNA4_SUBGROUP_SIZE 32
#define RDNA2_SUBGROUP_SIZE 64

enum vk_quant_type_t : uint32_t {
    QUANT_FP16    = 0,
    QUANT_Q4_K    = 1,
    QUANT_IQ4_XS  = 2,
    QUANT_Q6_K    = 3,
    QUANT_Q8_0    = 4,
    QUANT_Q4_0    = 5,
    QUANT_Q5_K    = 6,
    QUANT_FP32    = 7,
    QUANT_BF16    = 8,
    QUANT_NVFP4   = 9,
    QUANT_COUNT   = 10,
};

inline uint32_t vk_quant_to_shader_id(vk_quant_type_t qt) {
    switch (qt) {
        case QUANT_FP16:   return 0;
        case QUANT_Q8_0:   return 1;
        case QUANT_Q4_K:   return 2;
        case QUANT_Q6_K:   return 3;
        case QUANT_IQ4_XS: return 4;
        case QUANT_Q5_K:   return 5;
        default:           return 0;
    }
}

inline float vk_quant_bytes_per_element(vk_quant_type_t qt) {
    switch (qt) {
        case QUANT_FP16:   return 2.0f;
        case QUANT_Q4_K:   return 144.0f / 256.0f;   // block_q4_K: 144 B / 256 elems
        case QUANT_IQ4_XS: return 135.0f / 256.0f;   // block_iq4_xs: 135 B / 256 elems
        case QUANT_Q6_K:   return 210.0f / 256.0f;   // block_q6_K: 210 B / 256 elems
        case QUANT_Q8_0:   return 34.0f / 32.0f;     // block_q8_0: 34 B / 32 elems
        case QUANT_Q4_0:   return 18.0f / 32.0f;     // block_q4_0: 18 B / 32 elems
        case QUANT_Q5_K:   return 176.0f / 256.0f;   // block_q5_K: 176 B / 256 elems
        case QUANT_FP32:   return 4.0f;
        case QUANT_BF16:   return 2.0f;
        case QUANT_NVFP4:  return 0.5f;
        default:           return 1.0f;
    }
}

inline const char* vk_quant_type_name(vk_quant_type_t qt) {
    switch (qt) {
        case QUANT_FP16:   return "FP16";
        case QUANT_Q4_K:   return "Q4_K";
        case QUANT_IQ4_XS: return "IQ4_XS";
        case QUANT_Q6_K:   return "Q6_K";
        case QUANT_Q8_0:   return "Q8_0";
        case QUANT_Q4_0:   return "Q4_0";
        case QUANT_Q5_K:   return "Q5_K";
        case QUANT_FP32:   return "FP32";
        case QUANT_BF16:   return "BF16";
        case QUANT_NVFP4:  return "NVFP4";
        default:           return "UNKNOWN";
    }
}

enum op_type_t : uint32_t {
    OP_RMS_NORM         = 0,
    OP_ATTN_QKV         = 1,
    OP_ATTN_COMPUTE      = 2,
    OP_ATTN_OUTPUT       = 3,
    OP_ROPE             = 4,
    OP_FFN_GATE_UP      = 5,
    OP_FFN_DOWN         = 6,
    OP_LM_HEAD          = 7,
    OP_EMBEDDING_LOOKUP = 8,
    OP_KV_CACHE_WRITE   = 9,
    OP_ADD              = 10,
    OP_MUL              = 11,
    OP_SILU             = 12,
    OP_QK_NORM          = 13,
    OP_COUNT            = 14,
};

inline const char* op_type_name(op_type_t op) {
    switch (op) {
        case OP_RMS_NORM:         return "rms_norm";
        case OP_ATTN_QKV:         return "attn_qkv";
        case OP_ATTN_COMPUTE:      return "attn_compute";
        case OP_ATTN_OUTPUT:       return "attn_output";
        case OP_ROPE:             return "rope";
        case OP_FFN_GATE_UP:      return "ffn_gate_up";
        case OP_FFN_DOWN:         return "ffn_down";
        case OP_LM_HEAD:          return "lm_head";
        case OP_EMBEDDING_LOOKUP: return "embedding_lookup";
        case OP_KV_CACHE_WRITE:   return "kv_cache_write";
        case OP_ADD:              return "add";
        case OP_MUL:              return "mul";
        case OP_SILU:             return "silu";
        case OP_QK_NORM:          return "qk_norm";
        default:                  return "unknown";
    }
}

#pragma pack(push, 1)
struct push_constants_t {
    uint32_t layer_idx;
    uint32_t kv_cache_pos;
    uint32_t seq_len;
    uint32_t head_dim;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    // page_size_tokens/pages_per_layer/temperature removed: page_size_tokens
    // duplicated the SPEC_KV_PAGE_TOKENS specialization constant (common.glsl)
    // and pages_per_layer/temperature were never read by any shader -- dead
    // push-constant fields, confirmed via grep before removal. Freed 12 bytes
    // to stay within the 128-byte push-constant budget after adding MiniCPM's
    // embedding_scale/residual_scale/logit_scale below.
    float    attn_scale;
    float    rope_theta;
    float    norm_eps;
    uint32_t norm_type;        // 0=attn_norm, 1=ffn_norm, 2=output_norm, 3=q_norm, 4=k_norm
    uint32_t rope_is_neox;     // 1=NEOX split-half pairing, 0=NORM interleaved pairing
    int32_t  token_id;         // current input token for embedding
    uint32_t attn_norm_offset; // byte offset to attn_norm weights in layer buffer
    uint32_t ffn_norm_offset;  // byte offset to ffn_norm weights in layer buffer
    uint32_t output_norm_offset; // byte offset to output_norm weights in lm_head buffer
    uint32_t q_norm_offset;    // byte offset to q_norm weights (qwen3 per-head)
    uint32_t k_norm_offset;    // byte offset to k_norm weights (qwen3 per-head)
    // Region base offsets, computed once on the host (vk_model_compute_offsets)
    // from each region's ACTUAL per-role quant type. GEMM shaders must read
    // these rather than recomputing bases from compile-time macros — a shader
    // has no way to know at compile time whether a mixed-quant GGUF gave its
    // region a different byte layout than its neighbors.
    uint32_t q_offset;
    uint32_t k_offset;
    uint32_t v_offset;
    uint32_t o_offset;
    uint32_t gate_offset;
    uint32_t up_offset;
    uint32_t down_offset;
    // Optional QKV bias (qwen2-style architectures). has_qkv_bias=0 means the
    // *_bias_offset fields are unused -- shaders must check the flag, not
    // assume a zero offset means "no bias" (offset 0 is a valid real offset).
    uint32_t has_qkv_bias;
    uint32_t q_bias_offset;
    uint32_t k_bias_offset;
    uint32_t v_bias_offset;
    // MiniCPM-family depth-scaling; 1.0 = no-op for every other architecture.
    float    embedding_scale;
    float    residual_scale;
    float    logit_scale;
};
#pragma pack(pop)

static_assert(sizeof(push_constants_t) <= 128, "push_constants_t must be <= 128 bytes");

struct model_arch_t {
    uint32_t d;
    uint32_t ffn_dim;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t vocab_size;
    uint32_t n_layers;
    float    rope_theta;
    float    norm_eps;
    vk_quant_type_t quant_type;
    bool     use_gqa;
    // true  = GGML_ROPE_TYPE_NEOX (split-half pairing: i <-> i+HEAD_DIM/2) -- qwen/qwen2/qwen3/gemma*/phi*/etc.
    // false = GGML_ROPE_TYPE_NORM (interleaved pairing: 2i <-> 2i+1)       -- llama/minicpm/baichuan/etc.
    bool     rope_is_neox;
    // MiniCPM-family depth-scaling (llama.cpp: llama_model_minicpm::load_arch_hparams,
    // applied via the shared Granite graph -- see models/granite.cpp). All default
    // to 1.0 (identity/no-op multiply) for every other architecture.
    float    embedding_scale; // multiplies token embedding output once, before layer 0
    float    residual_scale;  // multiplies attn_output/ffn_down result before residual add
    float    logit_scale;     // multiplies final lm_head logits (note: this is already
                               // the RECIPROCAL of GGUF's residual_scale/logit_scale key --
                               // llama.cpp does ggml_scale(cur, 1.0f/hparams.f_logit_scale))
};

struct vk_buffer_t {
    VkBuffer             buffer;
    VmaAllocation        allocation;
    VmaAllocationInfo    alloc_info;
    VkDeviceSize         size;
    VkDeviceAddress      device_address;
    void*                mapped_ptr;
    VkBufferUsageFlags   usage;
    bool                 is_host_visible;
    bool                 is_host_coherent;
};
