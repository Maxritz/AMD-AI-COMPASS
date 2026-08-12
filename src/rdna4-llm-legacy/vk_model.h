#pragma once

#include "common.h"
#include "vk_kv_cache.h"
#include "gguf_parser.h"

// Per-tensor metadata: the single source of truth for where a weight lives
// and how to decode it. Built once at load time by reading each tensor's
// ACTUAL type and byte size directly from the parsed GGUF -- never guessed
// from a "dominant type" heuristic, never re-derived by a shader from a
// compile-time macro. Every kernel receives these values via push constants.
struct tensor_meta_t {
    vk_quant_type_t quant;      // this tensor's real type, from the GGUF
    uint64_t        offset;     // byte offset within its owning buffer
    uint64_t        size_bytes; // real uploaded size, taken from the GGUF tensor entry
    uint32_t        rows;       // ne[1]: output dim (number of rows)
    uint32_t        cols;       // ne[0]: input dim / row length (K)
    bool            present;    // false if this optional tensor isn't in the GGUF
};

struct layer_tensors_t {
    tensor_meta_t q, k, v, o;
    tensor_meta_t q_bias, k_bias, v_bias; // optional: qwen2-style QKV bias (F32 in GGUF, uploaded FP16)
    tensor_meta_t q_norm, k_norm;         // optional: qwen3-style per-head norm
    tensor_meta_t attn_norm, ffn_norm;
    tensor_meta_t gate, up, down;
};

struct vk_model_t {
    model_arch_t     config;
    vk_buffer_t      weight_bufs[MAX_LAYERS];
    vk_buffer_t      embedding_buf;
    vk_buffer_t      lm_head_buf;
    layer_tensors_t  tensors[MAX_LAYERS];  // single source of truth for layer weights
    tensor_meta_t    token_embd;           // lives in embedding_buf
    tensor_meta_t    output_norm;          // lives in lm_head_buf
    tensor_meta_t    lm_head;              // lives in lm_head_buf (aliases token_embd bytes if tied)
    vk_kv_cache_t    kv_cache;
    bool             lm_head_tied;    // no output.weight/lm_head.weight: head shares token_embd
    bool             has_output_norm; // output_norm.weight present (final RMS norm)
    bool             has_qk_norm;     // attn_q_norm/attn_k_norm present (qwen3 per-head norm)
    uint32_t         loaded_layers;
};

bool vk_model_load(vk_device_t* dev, vk_model_t* model, const char* gguf_path);
void vk_model_unload(vk_device_t* dev, vk_model_t* model);
