#include "vk_model.h"
#include "vk_device.h"
#include "vk_kv_cache.h"
#include "gguf_parser.h"
#include <cstring>

static uint64_t align_up(uint64_t val, uint64_t alignment) {
    return (val + alignment - 1) & ~(alignment - 1);
}

static bool buffer_create_device_local(vk_device_t* dev, VkDeviceSize size,
                                        VkBufferUsageFlags usage, vk_buffer_t* out) {
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size        = size;
    bci.usage       = usage;
    // The transfer queue uploads weights; the compute queue reads them. Without
    // CONCURRENT sharing (or ownership transfers) cross-queue access is UB and
    // on RDNA4 the compute side reads stale/garbage data.
    bci.sharingMode = VK_SHARING_MODE_CONCURRENT;
    uint32_t fams[] = { dev->compute_qf_idx, dev->transfer_qf_idx };
    bci.queueFamilyIndexCount = (dev->transfer_qf_idx != dev->compute_qf_idx) ? 2u : 1u;
    bci.pQueueFamilyIndices   = fams;

    VmaAllocationCreateInfo aci = {};
    aci.usage         = VMA_MEMORY_USAGE_AUTO;
    aci.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkResult result = vmaCreateBuffer(dev->allocator, &bci, &aci,
                                       &out->buffer, &out->allocation, &out->alloc_info);
    if (result != VK_SUCCESS) return false;

    out->size             = size;
    out->device_address   = 0;
    out->mapped_ptr       = nullptr;
    out->is_host_visible  = false;
    out->is_host_coherent = false;
    out->usage            = usage;
    return true;
}

static bool upload_buffer(vk_device_t* dev, const uint8_t* src_data, uint64_t src_size,
                           VkBuffer dst_buf, uint64_t dst_offset, VkFence fence,
                           VkBuffer staging_buf, void* staging_ptr) {
    memcpy(staging_ptr, src_data, (size_t)src_size);

    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool        = dev->transfer_cmd_pool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;

    VkCommandBuffer cb;
    vkAllocateCommandBuffers(dev->device, &cai, &cb);

    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);

    VkBufferCopy region = {0, dst_offset, src_size};
    vkCmdCopyBuffer(cb, staging_buf, dst_buf, 1, &region);

    vkEndCommandBuffer(cb);

    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;

    vkResetFences(dev->device, 1, &fence);
    vkQueueSubmit(dev->transfer_queue, 1, &si, fence);
    vkWaitForFences(dev->device, 1, &fence, VK_TRUE, UINT64_MAX);

    vkFreeCommandBuffers(dev->device, dev->transfer_cmd_pool, 1, &cb);
    return true;
}

// Convert a F32 float array to FP16 bytes. Norm/bias weights are stored F32 in
// the GGUF but every shader reads them as FP16 -- convert once at load time.
static std::vector<uint8_t> f32_to_f16_bytes(const uint8_t* src, uint64_t bytes) {
    uint64_t n = bytes / 4;
    std::vector<uint8_t> out(n * 2);
    const float* f = (const float*)src;
    uint16_t* h = (uint16_t*)out.data();
    for (uint64_t i = 0; i < n; i++) {
        float v = f[i];
        uint32_t b;
        memcpy(&b, &v, 4);
        uint32_t sign = (b >> 16) & 0x8000u;
        uint32_t exp  = (b >> 23) & 0xffu;
        uint32_t man  = b & 0x7fffffu;
        uint16_t r;
        if (exp == 0xffu) {
            r = (uint16_t)(sign | 0x7c00u);          // inf/nan
        } else if (exp > 0x8eu) {
            r = (uint16_t)(sign | 0x7c00u);          // overflow -> inf
        } else if (exp < 0x71u) {
            r = (uint16_t)sign;                      // underflow -> 0
        } else {
            uint32_t e = exp - 127u + 15u;
            uint32_t m = man >> 13u;
            r = (uint16_t)(sign | (e << 10u) | m);
        }
        h[i] = r;
    }
    return out;
}

// Fill one tensor_meta_t by looking up its ACTUAL type/dims/size directly from
// the GGUF tensor list, and advance the per-layer byte cursor by its real
// uploaded size. This is the entire fix for the "dominant type" bug class:
// every tensor's offset comes from what it actually is, never from a formula
// that assumes every tensor in a layer shares one quant type.
static tensor_meta_t place_tensor(const gguf_file_t* gguf, const char* name, uint64_t& cursor) {
    tensor_meta_t m = {};
    int32_t idx = gguf_find_tensor(gguf, name);
    if (idx < 0) {
        m.present = false;
        return m;
    }
    const gguf_tensor_info_t* t = &gguf->tensors[idx];
    m.present = true;
    m.quant   = (vk_quant_type_t)gguf_tensor_to_vk_quant(t->type);
    m.cols    = (uint32_t)t->dims[0];
    m.rows    = (t->n_dims > 1) ? (uint32_t)t->dims[1] : 1u;

    // Norms and biases are F32 in the GGUF; every shader reads FP16, so the
    // uploaded (and therefore reserved) size is half the raw GGUF tensor size.
    uint64_t upload_bytes = t->size_bytes;
    if (m.quant == QUANT_FP32) {
        m.quant = QUANT_FP16;
        upload_bytes = (uint64_t)m.cols * m.rows * 2ull;
    }

    m.offset     = cursor;
    m.size_bytes = upload_bytes;
    cursor += upload_bytes;
    return m;
}

// Upload one tensor's raw bytes (converting F32->FP16 first if needed) to its
// already-computed offset in the destination buffer.
static bool upload_tensor(vk_device_t* dev, const gguf_file_t* gguf, const char* name,
                          const tensor_meta_t& meta, VkBuffer dst_buf,
                          VkFence fence, VkBuffer staging_buf, void* staging_ptr) {
    if (!meta.present) return true;
    int32_t idx = gguf_find_tensor(gguf, name);
    if (idx < 0) return true; // shouldn't happen: place_tensor already found it
    uint64_t data_size;
    const uint8_t* data = gguf_get_tensor_data(gguf, (uint32_t)idx, &data_size);
    if (!data || data_size == 0) return true;

    std::vector<uint8_t> converted;
    if (data_size == meta.size_bytes * 2 && meta.quant == QUANT_FP16) {
        // Raw tensor was F32 (double the uploaded FP16 size) -> convert.
        converted = f32_to_f16_bytes(data, data_size);
        data = converted.data();
        data_size = converted.size();
    }
    if (data_size != meta.size_bytes) {
        RDNA4_ERROR("Tensor %s: size mismatch (raw=%llu, expected upload=%llu)",
                    name, (unsigned long long)data_size, (unsigned long long)meta.size_bytes);
        return false;
    }
    return upload_buffer(dev, data, data_size, dst_buf, meta.offset, fence, staging_buf, staging_ptr);
}

bool vk_model_load(vk_device_t* dev, vk_model_t* model, const char* gguf_path) {
    memset(model, 0, sizeof(*model));

    gguf_file_t gguf;
    if (!gguf_parser_open(gguf_path, &gguf)) {
        RDNA4_ERROR("Failed to parse GGUF: %s", gguf_path);
        return false;
    }

    model->config.d          = gguf.config.d;
    model->config.ffn_dim    = gguf.config.ffn_dim;
    model->config.n_heads    = gguf.config.n_heads;
    model->config.n_kv_heads = gguf.config.n_kv_heads;
    model->config.head_dim   = gguf.config.head_dim;
    model->config.vocab_size = gguf.config.vocab_size;
    model->config.n_layers   = gguf.config.n_layers;
    model->config.rope_theta = gguf.config.rope_theta;
    model->config.norm_eps   = gguf.config.norm_eps;
    model->loaded_layers     = gguf.config.n_layers;

    // GGML's RoPE has two mutually-incompatible pairing conventions and the
    // GGUF architecture string is the only signal for which one a model needs
    // (llama.cpp picks this per-arch in llama_model_rope_type(), never from a
    // GGUF key). Getting this wrong silently corrupts positional encoding for
    // every token past position 0 -- confirmed via llama.cpp's own arch table:
    // llama/minicpm/baichuan/starcoder/internlm2/xverse/command-r/cohere2/olmo/
    // deepseek*/granite*/chatglm/etc. use NORM (interleaved 2i,2i+1); qwen*/
    // gemma*/phi*/stablelm/bitnet/falcon/gptneox/starcoder2/orion/nemotron/
    // exaone*/codeshell/openelm/olmo2/olmoe/plamo*/minicpm3/dbrx/dream/etc.
    // use NEOX (split-half i,i+n/2). Default to NORM (the historically-common
    // convention) for any unrecognized architecture string.
    static const char* const neox_archs[] = {
        "qwen", "qwen2", "qwen2moe", "qwen3", "qwen3moe",
        "gemma", "gemma2", "gemma3", "gemma3n", "gemma4",
        "phi2", "phi3", "phimoe", "stablelm", "bitnet",
        "falcon", "falcon_h1", "gptneox", "gpt-neox",
        "starcoder2", "orion", "nemotron", "exaone", "exaone4",
        "codeshell", "openelm", "olmo2", "olmoe",
        "plamo", "plamo2", "plamo3", "minicpm3", "dbrx",
        "dream", "llada-moe", "rnd1",
        "bert", "modern-bert", "nomic-bert", "nomic-bert-moe", "eurobert",
        "jina-bert-v2", "jina-bert-v3",
    };
    model->config.rope_is_neox = false;
    for (const char* a : neox_archs) {
        if (strcmp(gguf.config.architecture, a) == 0) {
            model->config.rope_is_neox = true;
            break;
        }
    }

    // MiniCPM-family depth-scaling (llama.cpp: llama_model_minicpm::load_arch_hparams
    // in src/models/minicpm.cpp, applied via the shared Granite graph builder --
    // src/models/granite.cpp -- since MiniCPM literally reuses `llama_model_granite::graph`).
    // These are backward-compatible DEFAULTS for GGUFs that don't carry explicit
    // <arch>.embedding_scale / .residual_scale / .logit_scale keys; this project's
    // test MiniCPM GGUFs are exactly that case. Every other architecture gets 1.0
    // (identity multiply), so applying these unconditionally in shaders is safe.
    model->config.embedding_scale = 1.0f;
    model->config.residual_scale  = 1.0f;
    model->config.logit_scale     = 1.0f;
    if (strcmp(gguf.config.architecture, "minicpm") == 0 ||
        strcmp(gguf.config.architecture, "minicpm3") == 0) {
        model->config.embedding_scale = 12.0f;
        model->config.residual_scale  = 1.4f / sqrtf((float)model->config.n_layers);
        // llama.cpp scales final logits by 1/hparams.f_logit_scale, where
        // f_logit_scale defaults to 256/n_embd -- so the multiplier actually
        // applied to logits is n_embd/256, not 256/n_embd.
        model->config.logit_scale = model->config.d
            ? ((float)model->config.d / 256.0f) : 1.0f;
    }

    model->has_qk_norm = gguf_find_tensor(&gguf, "blk.0.attn_q_norm.weight") >= 0;

    // ── Build the per-layer tensor-metadata table ──────────────────────────
    // Every offset comes from each tensor's OWN real size (from the GGUF),
    // never from a formula that assumes uniform quantization across a layer.
    char name_buf[160];
    uint64_t max_layer_bytes = 0;
    for (uint32_t l = 0; l < model->loaded_layers; l++) {
        uint64_t cursor = 0;
        layer_tensors_t& lt = model->tensors[l];

        auto n = [&](const char* suffix) -> const char* {
            snprintf(name_buf, sizeof(name_buf), "blk.%u.%s", l, suffix);
            return name_buf;
        };

        lt.q         = place_tensor(&gguf, n("attn_q.weight"), cursor);
        lt.k         = place_tensor(&gguf, n("attn_k.weight"), cursor);
        lt.v         = place_tensor(&gguf, n("attn_v.weight"), cursor);
        lt.o         = place_tensor(&gguf, n("attn_output.weight"), cursor);
        lt.q_bias    = place_tensor(&gguf, n("attn_q.bias"), cursor);
        lt.k_bias    = place_tensor(&gguf, n("attn_k.bias"), cursor);
        lt.v_bias    = place_tensor(&gguf, n("attn_v.bias"), cursor);
        lt.q_norm    = place_tensor(&gguf, n("attn_q_norm.weight"), cursor);
        lt.k_norm    = place_tensor(&gguf, n("attn_k_norm.weight"), cursor);
        lt.attn_norm = place_tensor(&gguf, n("attn_norm.weight"), cursor);
        lt.ffn_norm  = place_tensor(&gguf, n("ffn_norm.weight"), cursor);
        lt.gate      = place_tensor(&gguf, n("ffn_gate.weight"), cursor);
        lt.up        = place_tensor(&gguf, n("ffn_up.weight"), cursor);
        lt.down      = place_tensor(&gguf, n("ffn_down.weight"), cursor);

        if (cursor > max_layer_bytes) max_layer_bytes = cursor;
    }
    uint64_t layer_buf_size = align_up(max_layer_bytes, 256);

    // ── Global tensors: token embedding, LM head, output norm ──────────────
    uint64_t emb_cursor = 0;
    model->token_embd = place_tensor(&gguf, "token_embd.weight", emb_cursor);
    if (!model->token_embd.present) {
        RDNA4_ERROR("GGUF has no token_embd.weight");
        gguf_parser_close(&gguf);
        return false;
    }
    uint64_t emb_buf_size = align_up(emb_cursor, 256);

    uint64_t lm_cursor = 0;
    bool have_output_weight = (gguf_find_tensor(&gguf, "output.weight") >= 0) ||
                               (gguf_find_tensor(&gguf, "lm_head.weight") >= 0);
    if (have_output_weight) {
        model->lm_head_tied = false;
        const char* head_name = (gguf_find_tensor(&gguf, "output.weight") >= 0)
                                     ? "output.weight" : "lm_head.weight";
        model->lm_head = place_tensor(&gguf, head_name, lm_cursor);
    } else {
        model->lm_head_tied = true;
        // Tied head: mirror the embedding table into the lm_head buffer at
        // offset 0, so lm_head reads exactly what token_embd wrote.
        model->lm_head = model->token_embd;
        model->lm_head.offset = 0;
        lm_cursor = model->token_embd.size_bytes;
        RDNA4_LOG("LM head TIED to token embedding (no output.weight)");
    }

    model->has_output_norm = gguf_find_tensor(&gguf, "output_norm.weight") >= 0;
    if (model->has_output_norm) {
        model->output_norm = place_tensor(&gguf, "output_norm.weight", lm_cursor);
    }
    uint64_t lm_buf_size = align_up(lm_cursor, 256);

    RDNA4_LOG("Model config: arch=%s, layers=%u, d=%u, ffn_dim=%u, heads=%u(kv=%u), head_dim=%u, vocab=%u, rope_theta=%.1f, rope_neox=%d",
              gguf.config.architecture, model->config.n_layers, model->config.d,
              model->config.ffn_dim, model->config.n_heads, model->config.n_kv_heads,
              model->config.head_dim, model->config.vocab_size,
              (double)model->config.rope_theta, model->config.rope_is_neox ? 1 : 0);
    RDNA4_LOG("Per-layer buffer size: %.2f MB (bias=%d qk_norm=%d output_norm=%d tied_head=%d)",
              (double)layer_buf_size / (1024.0 * 1024.0),
              model->tensors[0].q_bias.present ? 1 : 0, model->has_qk_norm ? 1 : 0,
              model->has_output_norm ? 1 : 0, model->lm_head_tied ? 1 : 0);

    VkBufferUsageFlags weight_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                       VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    for (uint32_t l = 0; l < model->config.n_layers; l++) {
        if (!buffer_create_device_local(dev, layer_buf_size, weight_usage, &model->weight_bufs[l])) {
            RDNA4_ERROR("Failed to allocate weight buffer for layer %u", l);
            vk_model_unload(dev, model);
            gguf_parser_close(&gguf);
            return false;
        }
    }
    if (!buffer_create_device_local(dev, emb_buf_size, weight_usage, &model->embedding_buf)) {
        RDNA4_ERROR("Failed to allocate embedding buffer");
        vk_model_unload(dev, model);
        gguf_parser_close(&gguf);
        return false;
    }
    if (!buffer_create_device_local(dev, lm_buf_size, weight_usage, &model->lm_head_buf)) {
        RDNA4_ERROR("Failed to allocate LM head buffer");
        vk_model_unload(dev, model);
        gguf_parser_close(&gguf);
        return false;
    }

    // ── Staging buffer sized to the largest single tensor ──────────────────
    uint64_t max_tensor_size = 0;
    for (uint64_t i = 0; i < gguf.tensor_count; i++) {
        if (gguf.tensors[i].size_bytes > max_tensor_size) max_tensor_size = gguf.tensors[i].size_bytes;
    }

    VkBufferCreateInfo staging_bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    staging_bci.size  = max_tensor_size;
    staging_bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
    staging_bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo staging_aci = {};
    staging_aci.usage = VMA_MEMORY_USAGE_AUTO;
    staging_aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                        VMA_ALLOCATION_CREATE_MAPPED_BIT;
    staging_aci.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    VkBuffer staging_buf;
    VmaAllocation staging_alloc;
    VmaAllocationInfo staging_ai;
    if (vmaCreateBuffer(dev->allocator, &staging_bci, &staging_aci,
                         &staging_buf, &staging_alloc, &staging_ai) != VK_SUCCESS) {
        RDNA4_ERROR("Failed to create staging buffer");
        vk_model_unload(dev, model);
        gguf_parser_close(&gguf);
        return false;
    }

    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence upload_fence;
    vkCreateFence(dev->device, &fci, nullptr, &upload_fence);

    bool ok = true;
    for (uint32_t l = 0; l < model->config.n_layers && ok; l++) {
        layer_tensors_t& lt = model->tensors[l];
        VkBuffer dst = model->weight_bufs[l].buffer;
        auto n = [&](const char* suffix) -> const char* {
            snprintf(name_buf, sizeof(name_buf), "blk.%u.%s", l, suffix);
            return name_buf;
        };
        ok = ok && upload_tensor(dev, &gguf, n("attn_q.weight"), lt.q, dst, upload_fence, staging_buf, staging_ai.pMappedData);
        ok = ok && upload_tensor(dev, &gguf, n("attn_k.weight"), lt.k, dst, upload_fence, staging_buf, staging_ai.pMappedData);
        ok = ok && upload_tensor(dev, &gguf, n("attn_v.weight"), lt.v, dst, upload_fence, staging_buf, staging_ai.pMappedData);
        ok = ok && upload_tensor(dev, &gguf, n("attn_output.weight"), lt.o, dst, upload_fence, staging_buf, staging_ai.pMappedData);
        ok = ok && upload_tensor(dev, &gguf, n("attn_q.bias"), lt.q_bias, dst, upload_fence, staging_buf, staging_ai.pMappedData);
        ok = ok && upload_tensor(dev, &gguf, n("attn_k.bias"), lt.k_bias, dst, upload_fence, staging_buf, staging_ai.pMappedData);
        ok = ok && upload_tensor(dev, &gguf, n("attn_v.bias"), lt.v_bias, dst, upload_fence, staging_buf, staging_ai.pMappedData);
        ok = ok && upload_tensor(dev, &gguf, n("attn_q_norm.weight"), lt.q_norm, dst, upload_fence, staging_buf, staging_ai.pMappedData);
        ok = ok && upload_tensor(dev, &gguf, n("attn_k_norm.weight"), lt.k_norm, dst, upload_fence, staging_buf, staging_ai.pMappedData);
        ok = ok && upload_tensor(dev, &gguf, n("attn_norm.weight"), lt.attn_norm, dst, upload_fence, staging_buf, staging_ai.pMappedData);
        ok = ok && upload_tensor(dev, &gguf, n("ffn_norm.weight"), lt.ffn_norm, dst, upload_fence, staging_buf, staging_ai.pMappedData);
        ok = ok && upload_tensor(dev, &gguf, n("ffn_gate.weight"), lt.gate, dst, upload_fence, staging_buf, staging_ai.pMappedData);
        ok = ok && upload_tensor(dev, &gguf, n("ffn_up.weight"), lt.up, dst, upload_fence, staging_buf, staging_ai.pMappedData);
        ok = ok && upload_tensor(dev, &gguf, n("ffn_down.weight"), lt.down, dst, upload_fence, staging_buf, staging_ai.pMappedData);
    }

    ok = ok && upload_tensor(dev, &gguf, "token_embd.weight", model->token_embd,
                              model->embedding_buf.buffer, upload_fence, staging_buf, staging_ai.pMappedData);
    if (model->lm_head_tied) {
        // Tied head: mirror the embedding table into lm_head_buf directly.
        ok = ok && upload_tensor(dev, &gguf, "token_embd.weight", model->lm_head,
                                  model->lm_head_buf.buffer, upload_fence, staging_buf, staging_ai.pMappedData);
    } else {
        const char* head_name = (gguf_find_tensor(&gguf, "output.weight") >= 0) ? "output.weight" : "lm_head.weight";
        ok = ok && upload_tensor(dev, &gguf, head_name, model->lm_head,
                                  model->lm_head_buf.buffer, upload_fence, staging_buf, staging_ai.pMappedData);
    }
    if (model->has_output_norm) {
        ok = ok && upload_tensor(dev, &gguf, "output_norm.weight", model->output_norm,
                                  model->lm_head_buf.buffer, upload_fence, staging_buf, staging_ai.pMappedData);
    }

    vkDestroyFence(dev->device, upload_fence, nullptr);
    vmaDestroyBuffer(dev->allocator, staging_buf, staging_alloc);
    gguf_parser_close(&gguf);

    if (!ok) {
        RDNA4_ERROR("Model upload failed");
        vk_model_unload(dev, model);
        return false;
    }

    RDNA4_LOG("Model loaded: %u layers", model->loaded_layers);
    return true;
}

void vk_model_unload(vk_device_t* dev, vk_model_t* model) {
    for (uint32_t l = 0; l < MAX_LAYERS; l++) {
        if (model->weight_bufs[l].buffer) {
            vmaDestroyBuffer(dev->allocator, model->weight_bufs[l].buffer,
                             model->weight_bufs[l].allocation);
            memset(&model->weight_bufs[l], 0, sizeof(vk_buffer_t));
        }
    }
    if (model->embedding_buf.buffer) {
        vmaDestroyBuffer(dev->allocator, model->embedding_buf.buffer,
                         model->embedding_buf.allocation);
        memset(&model->embedding_buf, 0, sizeof(vk_buffer_t));
    }
    if (model->lm_head_buf.buffer) {
        vmaDestroyBuffer(dev->allocator, model->lm_head_buf.buffer,
                         model->lm_head_buf.allocation);
        memset(&model->lm_head_buf, 0, sizeof(vk_buffer_t));
    }
}
