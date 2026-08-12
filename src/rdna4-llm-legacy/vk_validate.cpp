#include "vk_validate.h"
#include "vk_descriptor.h"
#include "vk_timeline.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <cinttypes>

static float f16_to_f32(uint16_t v) {
    const uint32_t sign = ((uint32_t)v >> 15) & 1;
    const uint32_t exp  = ((uint32_t)v >> 10) & 0x1F;
    const uint32_t mant = ((uint32_t)v) & 0x3FF;
    uint32_t f32;
    if (exp == 0) {
        if (mant == 0) {
            f32 = sign << 31;
        } else {
            uint32_t m2 = mant;
            int e2 = -14;
            while ((m2 & 0x400) == 0) { m2 <<= 1; e2--; }
            m2 &= 0x3FF;
            f32 = (sign << 31) | ((uint32_t)(e2 + 127) << 23) | (m2 << 13);
        }
    } else if (exp == 31) {
        f32 = (sign << 31) | (0xFF << 23) | (mant << 13);
    } else {
        f32 = (sign << 31) | ((uint32_t)(exp - 15 + 127) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f32, sizeof(float));
    return result;
}

static bool staging_buffer_create(vk_device_t* dev, VkDeviceSize size, vk_buffer_t* out) {
    memset(out, 0, sizeof(*out));
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size        = size;
    bci.usage       = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci = {};
    aci.usage          = VMA_MEMORY_USAGE_AUTO;
    aci.requiredFlags  = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    aci.flags          = VMA_ALLOCATION_CREATE_MAPPED_BIT | VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;

    VkResult result = vmaCreateBuffer(dev->allocator, &bci, &aci,
                                       &out->buffer, &out->allocation, &out->alloc_info);
    if (result != VK_SUCCESS) return false;
    out->size            = size;
    out->mapped_ptr      = out->alloc_info.pMappedData;
    out->is_host_visible = true;
    out->is_host_coherent = true;
    return true;
}

static void staging_buffer_destroy(vk_device_t* dev, vk_buffer_t* buf) {
    if (buf->buffer) {
        vmaDestroyBuffer(dev->allocator, buf->buffer, buf->allocation);
        memset(buf, 0, sizeof(*buf));
    }
}

static VkCommandBuffer alloc_cb(vk_device_t* dev) {
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool        = dev->compute_cmd_pool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(dev->device, &cai, &cb));
    return cb;
}

static void free_cb(vk_device_t* dev, VkCommandBuffer cb) {
    vkFreeCommandBuffers(dev->device, dev->compute_cmd_pool, 1, &cb);
}

static void submit_and_wait(vk_device_t* dev, VkCommandBuffer cb) {
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VK_CHECK(vkQueueSubmit(dev->compute_queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(dev->compute_queue));
}

static void shader_barrier(VkCommandBuffer cb, VkBuffer buf,
                           VkAccessFlags src, VkAccessFlags dst) {
    VkBufferMemoryBarrier b = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    b.srcAccessMask = src;
    b.dstAccessMask = dst;
    b.buffer        = buf;
    b.size          = VK_WHOLE_SIZE;
    vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        0, 0, nullptr, 1, &b, 0, nullptr);
}

static VkPipeline find_pipeline_local(vk_session_t* session, int op_type, int quant_type) {
    for (int i = 0; i < session->pipeline_count; i++) {
        if (session->pipelines[i].op_type == op_type &&
            session->pipelines[i].quant_type == quant_type &&
            session->pipelines[i].pipeline) {
            return session->pipelines[i].pipeline;
        }
    }
    for (int i = 0; i < session->pipeline_count; i++) {
        if (session->pipelines[i].op_type == op_type &&
            session->pipelines[i].quant_type == QUANT_FP16 &&
            session->pipelines[i].pipeline) {
            return session->pipelines[i].pipeline;
        }
    }
    return VK_NULL_HANDLE;
}

std::vector<float> download_buffer(vk_device_t* dev, vk_buffer_t* buf, uint32_t num_elements) {
    if (!buf || !buf->buffer || num_elements == 0) return {};

    VkDeviceSize size = (VkDeviceSize)num_elements * 2ull; // fp16 = 2 bytes each
    vk_buffer_t staging;
    if (!staging_buffer_create(dev, size, &staging)) {
        fprintf(stderr, "[VALIDATE] Failed to create staging buffer\n");
        return {};
    }

    VkCommandBuffer cb = alloc_cb(dev);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));

    {
        VkBufferMemoryBarrier barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
        barrier.buffer = buf->buffer;
        barrier.size   = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cb,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            0, 0, nullptr, 1, &barrier, 0, nullptr);
    }

    VkBufferCopy copy = {};
    copy.srcOffset = 0;
    copy.dstOffset = 0;
    copy.size      = size;
    vkCmdCopyBuffer(cb, buf->buffer, staging.buffer, 1, &copy);

    {
        VkBufferMemoryBarrier barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        barrier.buffer = staging.buffer;
        barrier.size   = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cb,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            0, 0, nullptr, 1, &barrier, 0, nullptr);
    }

    VK_CHECK(vkEndCommandBuffer(cb));
    submit_and_wait(dev, cb);
    free_cb(dev, cb);

    std::vector<float> result(num_elements);
    const uint16_t* src = (const uint16_t*)staging.mapped_ptr;
    for (uint32_t i = 0; i < num_elements; i++) {
        result[i] = f16_to_f32(src[i]);
    }

    staging_buffer_destroy(dev, &staging);
    return result;
}

static std::vector<uint8_t> download_raw_bytes(vk_device_t* dev, vk_buffer_t* buf,
                                                uint64_t byte_offset, uint64_t byte_size) {
    if (!buf || !buf->buffer || byte_size == 0) return {};

    vk_buffer_t staging;
    if (!staging_buffer_create(dev, byte_size, &staging)) {
        fprintf(stderr, "[VALIDATE] Failed to create staging buffer for raw download\n");
        return {};
    }

    VkCommandBuffer cb = alloc_cb(dev);
    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));

    VkBufferCopy copy = {};
    copy.srcOffset = byte_offset;
    copy.dstOffset = 0;
    copy.size      = byte_size;
    vkCmdCopyBuffer(cb, buf->buffer, staging.buffer, 1, &copy);

    {
        VkBufferMemoryBarrier barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        barrier.buffer = staging.buffer;
        barrier.size   = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cb,
            VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_HOST_BIT,
            0, 0, nullptr, 1, &barrier, 0, nullptr);
    }

    VK_CHECK(vkEndCommandBuffer(cb));
    submit_and_wait(dev, cb);
    free_cb(dev, cb);

    std::vector<uint8_t> result(byte_size);
    memcpy(result.data(), staging.mapped_ptr, (size_t)byte_size);

    staging_buffer_destroy(dev, &staging);
    return result;
}

static std::vector<float> download_fp16_weights(vk_device_t* dev, vk_buffer_t* buf,
                                                  uint64_t byte_offset, uint32_t count) {
    uint64_t byte_size = (uint64_t)count * 2ull;
    auto raw = download_raw_bytes(dev, buf, byte_offset, byte_size);
    if (raw.size() < byte_size) return {};

    std::vector<float> result(count);
    for (uint32_t i = 0; i < count; i++) {
        uint16_t v = (uint16_t)raw[i * 2] | ((uint16_t)raw[i * 2 + 1] << 8);
        result[i] = f16_to_f32(v);
    }
    return result;
}

std::vector<float> cpu_rms_norm(const std::vector<float>& input,
                                 const std::vector<float>& norm_weight,
                                 float eps, uint32_t d) {
    std::vector<float> output(d);

    double sum_sq = 0.0;
    for (uint32_t i = 0; i < d; i++) {
        sum_sq += (double)input[i] * (double)input[i];
    }

    float rms = (float)sqrt(sum_sq / (double)d + (double)eps);
    float inv_rms = 1.0f / rms;

    for (uint32_t i = 0; i < d; i++) {
        output[i] = input[i] * inv_rms * norm_weight[i];
    }

    return output;
}

std::vector<float> cpu_dequant_q8_0(const uint8_t* packed_data,
                                     uint32_t n_rows, uint32_t n_cols) {
    const uint32_t blocks_per_row = n_cols / 32u;
    const uint32_t row_stride = blocks_per_row * 34u;

    std::vector<float> result(n_rows * n_cols);

    for (uint32_t r = 0; r < n_rows; r++) {
        const uint8_t* row_data = packed_data + (uint64_t)r * row_stride;
        for (uint32_t blk = 0; blk < blocks_per_row; blk++) {
            const uint8_t* blk_data = row_data + (uint64_t)blk * 34u;

            uint16_t d_raw = (uint16_t)blk_data[0] | ((uint16_t)blk_data[1] << 8);
            float d = f16_to_f32(d_raw);

            for (uint32_t e = 0; e < 32u; e++) {
                int8_t q = (int8_t)blk_data[2 + e];
                uint32_t col = blk * 32u + e;
                if (col < n_cols) {
                    result[r * n_cols + col] = d * (float)(int)q;
                }
            }
        }
    }

    return result;
}

// Dequantize raw FP16 weight bytes into row-major floats (n_rows x n_cols).
static std::vector<float> cpu_dequant_fp16(const uint8_t* packed_data,
                                            uint32_t n_rows, uint32_t n_cols) {
    std::vector<float> result(n_rows * n_cols);
    for (uint32_t r = 0; r < n_rows; r++) {
        for (uint32_t c = 0; c < n_cols; c++) {
            const uint8_t* p = packed_data + ((uint64_t)r * n_cols + c) * 2ull;
            uint16_t v = (uint16_t)p[0] | ((uint16_t)p[1] << 8);
            result[(uint64_t)r * n_cols + c] = f16_to_f32(v);
        }
    }
    return result;
}

// Dequantize raw Q6_K weight bytes (block_q6_K: 256 elems, 210 bytes:
// ql[128]@0, qh[64]@128, scales[16]@192, d@208) into row-major floats.
static std::vector<float> cpu_dequant_q6_k(const uint8_t* packed_data,
                                            uint32_t n_rows, uint32_t n_cols) {
    const uint32_t sb_per_row = n_cols / 256u;
    const uint32_t row_stride = sb_per_row * 210u;
    std::vector<float> result(n_rows * n_cols);
    for (uint32_t r = 0; r < n_rows; r++) {
        const uint8_t* row_data = packed_data + (uint64_t)r * row_stride;
        for (uint32_t sb = 0; sb < sb_per_row; sb++) {
            const uint8_t* b = row_data + (uint64_t)sb * 210u;
            float d = f16_to_f32((uint16_t)b[208] | ((uint16_t)b[209] << 8));
            for (uint32_t half = 0; half < 2; half++) {
                const uint8_t* ql = b + half * 64u;
                const uint8_t* qh = b + 128u + half * 32u;
                const int8_t* sc = (const int8_t*)(b + 192u + half * 8u);
                for (uint32_t l = 0; l < 32; l++) {
                    int is = (int)(l / 16);
                    int q1 = (int)(ql[l] & 0xF) | (((int)qh[l] >> 0) & 3) << 4;
                    int q2 = (int)(ql[l + 32] & 0xF) | (((int)qh[l] >> 2) & 3) << 4;
                    int q3 = (int)(ql[l] >> 4) | (((int)qh[l] >> 4) & 3) << 4;
                    int q4 = (int)(ql[l + 32] >> 4) | (((int)qh[l] >> 6) & 3) << 4;
                    uint32_t base = r * n_cols + sb * 256u + half * 128u;
                    result[base + l + 0]  = d * (float)sc[is + 0] * (float)(q1 - 32);
                    result[base + l + 32] = d * (float)sc[is + 2] * (float)(q2 - 32);
                    result[base + l + 64] = d * (float)sc[is + 4] * (float)(q3 - 32);
                    result[base + l + 96] = d * (float)sc[is + 6] * (float)(q4 - 32);
                }
            }
        }
    }
    return result;
}

// Dequantize raw Q4_K weight bytes (block_q4_K: 256 elems, 144 bytes) into floats.
static std::vector<float> cpu_dequant_q4_k(const uint8_t* packed_data,
                                            uint32_t n_rows, uint32_t n_cols) {
    const uint32_t sb_per_row = n_cols / 256u;
    const uint32_t row_stride = sb_per_row * 144u;
    std::vector<float> result(n_rows * n_cols);
    for (uint32_t r = 0; r < n_rows; r++) {
        const uint8_t* row_data = packed_data + (uint64_t)r * row_stride;
        for (uint32_t sb = 0; sb < sb_per_row; sb++) {
            const uint8_t* b = row_data + (uint64_t)sb * 144u;
            float d   = f16_to_f32((uint16_t)b[0] | ((uint16_t)b[1] << 8));
            float mn  = f16_to_f32((uint16_t)b[2] | ((uint16_t)b[3] << 8));
            const uint8_t* q = b + 16u;
            for (uint32_t j = 0; j < 256; j += 64) {
                for (uint32_t half = 0; half < 2; half++) {
                    int is = (int)(j / 64) * 2 + (int)half;
                    uint8_t sc_b, m_b;
                    if (is < 4) {
                        sc_b = b[4 + is] & 63;
                        m_b  = b[8 + is] & 63;
                    } else {
                        sc_b = (b[is + 4] & 0xF) | ((b[is - 4] >> 6) << 4);
                        m_b  = (b[is + 4] >> 4) | ((b[is] >> 6) << 4);
                    }
                    float dl = d * (float)sc_b;
                    float ml = mn * (float)m_b;
                    for (uint32_t l = 0; l < 32; l++) {
                        uint32_t col = sb * 256u + j + half * 32u + l;
                        result[r * n_cols + col] = dl * (float)(q[(j / 2) + half * 32u + l] & 0xF) - ml;
                        result[r * n_cols + col + 32] = dl * (float)(q[(j / 2) + half * 32u + l] >> 4) - ml;
                    }
                }
            }
        }
    }
    return result;
}

std::vector<float> cpu_matmul(const std::vector<float>& A,
                               const std::vector<float>& B,
                               uint32_t M, uint32_t N, uint32_t K) {
    std::vector<float> C(M * N, 0.0f);

    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t n = 0; n < N; n++) {
            double accum = 0.0;
            const float* a_row = A.data() + m * K;
            const float* b_row = B.data() + n * K;
            for (uint32_t k = 0; k < K; k++) {
                accum += (double)a_row[k] * (double)b_row[k];
            }
            C[m * N + n] = (float)accum;
        }
    }

    return C;
}

static void cpu_rope(std::vector<float>& vals, uint32_t n_heads, uint32_t head_dim,
                      float rope_theta, uint32_t position) {
    for (uint32_t h = 0; h < n_heads; h++) {
        for (uint32_t pair = 0; pair < head_dim / 2u; pair++) {
            float freq = 1.0f / powf(rope_theta, (2.0f * (float)pair) / (float)head_dim);
            float theta = (float)position * freq;
            float c = cosf(theta);
            float s = sinf(theta);

            uint32_t idx_even = h * head_dim + pair * 2u;
            uint32_t idx_odd  = idx_even + 1u;

            float v_even = vals[idx_even];
            float v_odd  = vals[idx_odd];

            vals[idx_even] = v_even * c - v_odd * s;
            vals[idx_odd]  = v_odd * c + v_even * s;
        }
    }
}

static void cpu_softmax_attention(std::vector<float>& probs, float attn_scale, uint32_t num_positions) {
    float max_val = -1.0e30f;
    for (uint32_t i = 0; i < num_positions; i++) {
        float s = probs[i] * attn_scale;
        probs[i] = s;
        if (s > max_val) max_val = s;
    }

    double sum_exp = 0.0;
    for (uint32_t i = 0; i < num_positions; i++) {
        float e = expf(probs[i] - max_val);
        probs[i] = e;
        sum_exp += (double)e;
    }

    float inv_sum = 1.0f / (float)sum_exp;
    for (uint32_t i = 0; i < num_positions; i++) {
        probs[i] *= inv_sum;
    }
}

struct ValidationMetrics {
    float max_ae;
    float mse;
    uint32_t worst_idx;
    float worst_expected;
    float worst_actual;
    bool   passed;
};

static ValidationMetrics compare_outputs(const std::vector<float>& expected,
                                          const std::vector<float>& actual,
                                          uint32_t count, float threshold) {
    ValidationMetrics m = {};
    m.max_ae = 0.0f;
    m.worst_idx = 0;
    m.worst_expected = 0.0f;
    m.worst_actual = 0.0f;
    m.passed = true;

    double sum_sq_err = 0.0;
    for (uint32_t i = 0; i < count; i++) {
        float exp = expected[i];
        float act = actual[i];
        float ae = fabsf(exp - act);
        sum_sq_err += (double)ae * (double)ae;

        bool any_nan = std::isnan(exp) || std::isnan(act);
        if (any_nan) {
            m.passed = false;
            m.max_ae = NAN;
            m.worst_idx = i;
            m.worst_expected = exp;
            m.worst_actual = act;
            fprintf(stderr, "[VALIDATE NaN] GPU output[%u] is NaN! expected=%.6f actual=%.6f\n",
                    i, exp, act);
            break;
        }

        if (ae > m.max_ae) {
            m.max_ae = ae;
            m.worst_idx = i;
            m.worst_expected = exp;
            m.worst_actual = act;
        }
    }

    m.mse = (float)(sum_sq_err / (double)count);

    if (!m.passed) return m;
    if (m.max_ae > threshold) m.passed = false;

    return m;
}

static bool check_for_nan(const std::vector<float>& data) {
    for (size_t i = 0; i < data.size(); i++) {
        if (std::isnan(data[i])) return true;
    }
    return false;
}

static void print_nan_indices(FILE* f, const char* buf_name, const std::vector<float>& data, uint32_t max_show) {
    uint32_t nan_count = 0;
    for (uint32_t i = 0; i < (uint32_t)data.size() && nan_count < max_show; i++) {
        if (std::isnan(data[i])) {
            fprintf(f, "  [NaN] %s[%u] = NaN\n", buf_name, i);
            nan_count++;
        }
    }
    uint32_t total = 0;
    for (uint32_t i = 0; i < (uint32_t)data.size(); i++) {
        if (std::isnan(data[i])) total++;
    }
    if (total > max_show) {
        fprintf(f, "  [NaN] ... and %u more NaN values in %s (total %u / %u)\n",
                total - max_show, buf_name, total, (uint32_t)data.size());
    }
    if (total == 0) {
        fprintf(f, "  [OK] %s: no NaN values (%u elements)\n", buf_name, (uint32_t)data.size());
    }
}

static void log_metrics(FILE* f, const char* op_name, const ValidationMetrics& m, uint32_t count) {
    const char* status = m.passed ? "PASS" : "FAIL";
    fprintf(f, "[%s] %s | count=%u | maxAE=%.6e | MSE=%.6e | worst_idx=%u | worst_exp=%.6f | worst_act=%.6f\n",
            status, op_name, count, m.max_ae, m.mse, m.worst_idx, m.worst_expected, m.worst_actual);

    if (!m.passed || std::isnan(m.max_ae)) {
        fprintf(stderr, "[VALIDATE ERROR] %s: FAIL (maxAE=%.6e, NaN=%d)\n",
                op_name, m.max_ae, std::isnan(m.max_ae) ? 1 : 0);
    } else if (m.max_ae > 1.0f) {
        fprintf(stderr, "[VALIDATE ERROR] %s: maxAE=%.6e > 1.0\n", op_name, m.max_ae);
    } else if (m.max_ae > 0.01f) {
        fprintf(stderr, "[VALIDATE WARNING] %s: maxAE=%.6e > 0.01\n", op_name, m.max_ae);
    } else {
        RDNA4_LOG("[VALIDATE] %s: maxAE=%.6e PASS", op_name, m.max_ae);
    }
}

// GQA mapping: for each query head, map to KV head
static uint32_t gqa_map(uint32_t head_idx, uint32_t n_heads, uint32_t n_kv_heads) {
    return (head_idx * n_kv_heads) / n_heads;
}

bool validate_layer(vk_device_t* dev, vk_model_t* model, vk_session_t* session,
                    uint32_t layer_idx, int32_t token_id,
                    const std::vector<float>& cpu_embedding,
                    FILE* log_file) {
    RDNA4_LOG("[VALIDATE] Layer %u validation starting...", layer_idx);

    uint32_t d         = model->config.d;
    uint32_t ffn_dim   = model->config.ffn_dim;
    uint32_t n_heads   = model->config.n_heads;
    uint32_t n_kv_heads = model->config.n_kv_heads;
    uint32_t head_dim  = model->config.head_dim;
    float    norm_eps  = model->config.norm_eps;
    float    rope_th   = model->config.rope_theta;
    float    attn_scale = 1.0f / sqrtf((float)head_dim);
    int      quant_type_local = (int)session->weight_quant;

    if (d == 0 || ffn_dim == 0 || n_heads == 0 || head_dim == 0) {
        fprintf(stderr, "[VALIDATE ERROR] Invalid model config\n");
        return false;
    }

    const layer_weight_offsets_t& off = model->weight_offsets[layer_idx];
    vk_buffer_t* weight_buf = &model->weight_bufs[layer_idx];

    // ── Find Pipelines ────────────────────────────────────────────────────
    // Per-role quant: a mixed-quant GGUF can give ffn_down (or attn_v, etc.) a
    // different type than the rest of the layer — pick each op's pipeline (and,
    // below, its CPU reference dequant) from that region's ACTUAL type.
    VkPipeline pipe_rms_norm    = find_pipeline_local(session, OP_RMS_NORM, QUANT_FP16);
    VkPipeline pipe_attn_qkv    = find_pipeline_local(session, OP_ATTN_QKV, (int)model->qkv_quant);
    VkPipeline pipe_attn_comp   = find_pipeline_local(session, OP_ATTN_COMPUTE, QUANT_FP16);
    VkPipeline pipe_attn_output = find_pipeline_local(session, OP_ATTN_OUTPUT, (int)model->o_quant);
    VkPipeline pipe_ffn_gate_up = find_pipeline_local(session, OP_FFN_GATE_UP, (int)model->gate_up_quant);
    VkPipeline pipe_ffn_down    = find_pipeline_local(session, OP_FFN_DOWN, (int)model->down_quant);

    if (!pipe_rms_norm || !pipe_attn_qkv || !pipe_attn_comp ||
        !pipe_attn_output || !pipe_ffn_gate_up || !pipe_ffn_down) {
        fprintf(stderr, "[VALIDATE ERROR] Missing pipelines\n");
        return false;
    }

    // ── Create per-step buffers for GPU results ───────────────────────────
    // We create our own buffers to avoid interfering with session's pipeline
    auto create_dev_buf = [&](VkDeviceSize sz) -> vk_buffer_t {
        vk_buffer_t buf = {};
        VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bci.size        = sz;
        bci.usage       = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VmaAllocationCreateInfo aci = {};
        aci.usage = VMA_MEMORY_USAGE_AUTO;
        aci.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        if (vmaCreateBuffer(dev->allocator, &bci, &aci,
                            &buf.buffer, &buf.allocation, &buf.alloc_info) == VK_SUCCESS) {
            buf.size = sz;
        }
        return buf;
    };
    auto destroy_dev_buf = [&](vk_buffer_t* b) {
        if (b->buffer) vmaDestroyBuffer(dev->allocator, b->buffer, b->allocation);
        memset(b, 0, sizeof(*b));
    };

    vk_buffer_t val_hidden0 = {0};
    vk_buffer_t val_norm    = {0};
    vk_buffer_t val_qkv     = {0};
    vk_buffer_t val_attn    = {0};
    vk_buffer_t val_hidden1 = {0};
    vk_buffer_t val_ffn     = {0};

    val_hidden0 = create_dev_buf(d * 2);
    val_norm    = create_dev_buf(d * 2);
    val_qkv     = create_dev_buf((n_heads * head_dim + 2 * n_kv_heads * head_dim) * 2);
    val_attn    = create_dev_buf(n_heads * head_dim * 2);
    val_hidden1 = create_dev_buf(d * 2);
    val_ffn     = create_dev_buf(ffn_dim * 2);

    push_constants_t pc = {};
    pc.head_dim         = head_dim;
    pc.num_heads        = n_heads;
    pc.num_kv_heads     = n_kv_heads;
    pc.attn_scale       = attn_scale;
    pc.rope_theta       = rope_th;
    pc.norm_eps         = norm_eps;
    // 1.0 = no-op multiply; this validator's CPU reference math (below) doesn't
    // model MiniCPM's embedding/residual/logit scaling, so keep both sides at
    // identity until this tool is rewritten (plan Step 7) to call the real path.
    pc.embedding_scale  = 1.0f;
    pc.residual_scale   = 1.0f;
    pc.logit_scale      = 1.0f;
    pc.layer_idx        = layer_idx;
    pc.kv_cache_pos     = 0;
    pc.seq_len          = 1;
    pc.norm_type        = 0;
    pc.attn_norm_offset = off.attn_norm_offset;
    pc.ffn_norm_offset  = off.ffn_norm_offset;
    pc.q_offset    = (uint32_t)off.q_offset;
    pc.k_offset    = (uint32_t)off.k_offset;
    pc.v_offset    = (uint32_t)off.v_offset;
    pc.o_offset    = (uint32_t)off.o_offset;
    pc.gate_offset = (uint32_t)off.gate_offset;
    pc.up_offset   = (uint32_t)off.up_offset;
    pc.down_offset = (uint32_t)off.down_offset;

    VkPipelineLayout layout   = session->shared_pipeline_layout;
    VkDescriptorSet  set0     = session->desc.set0_weight_set;
    VkDescriptorSet  set2     = session->desc.set2_table_set;
    VkDevice         device   = dev->device;

    auto bind_sets = [&](VkCommandBuffer cb) {
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
            layout, 0, 1, &set0, 0, nullptr);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
            layout, 2, 1, &set2, 0, nullptr);
    };

    auto dispatch_validate = [&](const char* step_name,
                                  VkPipeline pipe,
                                  vk_buffer_t* in_buf, vk_buffer_t* out_buf,
                                  vk_buffer_t* kcache, vk_buffer_t* vcache,
                                  vk_buffer_t* scratch_buf,
                                  uint32_t wg_x,
                                  std::vector<float>& cpu_expected,
                                  std::vector<float>& gpu_result,
                                  uint32_t output_count) -> bool {
        VkCommandBuffer cb = alloc_cb(dev);
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cb, &bi));

        bind_sets(cb);

        vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vk_descriptor_arena_push_set1(&session->desc, device, cb, layout,
                                       in_buf, out_buf, kcache, vcache, scratch_buf);

        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdDispatch(cb, wg_x, 1, 1);

        shader_barrier(cb, out_buf->buffer,
                       VK_ACCESS_SHADER_WRITE_BIT,
                       VK_ACCESS_TRANSFER_READ_BIT);

        // Copy result to staging for CPU readback
        VkDeviceSize copy_sz = (VkDeviceSize)output_count * 2ull;
        vk_buffer_t staging;
        if (!staging_buffer_create(dev, copy_sz, &staging)) {
            vkEndCommandBuffer(cb);
            free_cb(dev, cb);
            fprintf(stderr, "[VALIDATE] Staging alloc failed for %s\n", step_name);
            return false;
        }

        VkBufferCopy copy = {0, 0, copy_sz};
        vkCmdCopyBuffer(cb, out_buf->buffer, staging.buffer, 1, &copy);

        {
            VkBufferMemoryBarrier barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            barrier.buffer = staging.buffer;
            barrier.size   = VK_WHOLE_SIZE;
            vkCmdPipelineBarrier(cb,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT,
                0, 0, nullptr, 1, &barrier, 0, nullptr);
        }

        VK_CHECK(vkEndCommandBuffer(cb));
        submit_and_wait(dev, cb);
        free_cb(dev, cb);

        gpu_result.resize(output_count);
        const uint16_t* src = (const uint16_t*)staging.mapped_ptr;
        for (uint32_t i = 0; i < output_count; i++) {
            gpu_result[i] = f16_to_f32(src[i]);
        }

        staging_buffer_destroy(dev, &staging);

        bool step_ok = true;

        if (check_for_nan(gpu_result)) {
            fprintf(log_file, "[NaN DETECTED] %s: GPU output contains NaN!\n", step_name);
            print_nan_indices(log_file, step_name, gpu_result, 10);
            step_ok = false;
        }

        if (cpu_expected.size() >= output_count) {
            auto met = compare_outputs(cpu_expected, gpu_result, output_count, 0.05f);
            log_metrics(log_file, step_name, met, output_count);
            if (!met.passed) {
                step_ok = false;
                fprintf(log_file, "  ── First 5 GPU vs CPU ──\n");
                for (uint32_t i = 0; i < (output_count < 8u ? output_count : 8u); i++) {
                    fprintf(log_file, "  [%u] GPU=%.6f  CPU=%.6f  diff=%.6e\n",
                            i, gpu_result[i], cpu_expected[i],
                            (double)gpu_result[i] - cpu_expected[i]);
                }
            }
        } else {
            fprintf(log_file, "[INFO] %s: GPU output (first 8): ", step_name);
            for (uint32_t i = 0; i < (output_count < 8u ? output_count : 8u); i++) {
                fprintf(log_file, "%.4f ", gpu_result[i]);
            }
            fprintf(log_file, "\n");
        }

        return step_ok;
    };

    bool all_ok = true;

    // ── (a) Step: Write embedding into val_hidden0 ─────────────────────────
    {
        VkCommandBuffer cb = alloc_cb(dev);
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cb, &bi));

        // For F16 embedding the raw bytes are already fp16; for quantized embeddings
        // (Q8_0/Q6_K/Q4_K/...) run the token_embed shader so the GPU output is the
        // dequantized embedding (same path the real decode uses).
        uint32_t emb_q = (uint32_t)model->emb_quant;
        if (emb_q == QUANT_FP16 || emb_q == QUANT_BF16) {
            VkBufferCopy emb_copy = {};
            emb_copy.srcOffset = (uint64_t)token_id * (uint64_t)d * 2ull;
            emb_copy.dstOffset = 0;
            emb_copy.size      = (uint64_t)d * 2ull;
            vkCmdCopyBuffer(cb, model->embedding_buf.buffer, val_hidden0.buffer, 1, &emb_copy);

            shader_barrier(cb, val_hidden0.buffer,
                           VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        } else {
            VkPipeline pipe_emb = find_pipeline_local(session, OP_EMBEDDING_LOOKUP, (int)emb_q);
            if (pipe_emb) {
                session->push_constants.token_id = (uint32_t)token_id;
                vkCmdPushConstants(cb, session->shared_pipeline_layout,
                                   VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(push_constants_t),
                                   &session->push_constants);
                vk_descriptor_arena_push_set1(&session->desc, dev->device, cb,
                                              session->shared_pipeline_layout,
                                              &model->embedding_buf, &val_hidden0,
                                              &model->kv_cache.buffer, &model->kv_cache.buffer,
                                              &session->decode_state.ffn_scratch);
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_emb);
                vkCmdDispatch(cb, (d + 128 - 1) / 128, 1, 1);
                shader_barrier(cb, val_hidden0.buffer,
                               VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
            } else {
                fprintf(log_file, "[SKIP] no token_embed pipeline for quant type %d\n", emb_q);
                return false;
            }
        }

        VK_CHECK(vkEndCommandBuffer(cb));
        submit_and_wait(dev, cb);
        free_cb(dev, cb);

        auto gpu_emb = download_buffer(dev, &val_hidden0, d);
        auto met = compare_outputs(cpu_embedding, gpu_emb, d, 0.001f);
        log_metrics(log_file, "embedding", met, d);
        if (!met.passed) {
            all_ok = false;
            fprintf(log_file, "  ── First 8 embedding values ──\n");
            for (uint32_t i = 0; i < 8; i++) {
                fprintf(log_file, "  [%u] GPU=%.6f  CPU=%.6f\n", i, gpu_emb[i], cpu_embedding[i]);
            }
        }
    }

    // ── CPU: attn_norm weight download ─────────────────────────────────────
    auto attn_norm_weight = download_fp16_weights(dev, weight_buf,
                                                   off.attn_norm_offset, d);
    if (attn_norm_weight.size() < d) {
        fprintf(stderr, "[VALIDATE] Failed to download attn_norm weight\n");
        all_ok = false;
    }

    // ── (b) attn_norm GPU dispatch ─────────────────────────────────────────
    if (attn_norm_weight.size() >= d) {
        std::vector<float> cpu_norm_expected = cpu_rms_norm(cpu_embedding, attn_norm_weight, norm_eps, d);
        std::vector<float> gpu_norm;
        pc.norm_type = 0;
        all_ok = all_ok && dispatch_validate("attn_norm", pipe_rms_norm,
                &val_hidden0, &val_norm,
                nullptr, nullptr, nullptr,
                1,
                cpu_norm_expected, gpu_norm, d);
    }

    // ── CPU: QKV weight download + CPU computation ─────────────────────────
    uint32_t q_rows = n_heads * head_dim;
    uint32_t k_rows = n_kv_heads * head_dim;
    uint32_t v_rows = n_kv_heads * head_dim;
    uint32_t qkv_total = q_rows + k_rows + v_rows;

    int qkv_quant_local = (int)model->qkv_quant;
    uint64_t q_bytes = q_rows * (d / 32) * 34;
    uint64_t k_bytes = k_rows * (d / 32) * 34;
    uint64_t v_bytes = v_rows * (d / 32) * 34;
    if (qkv_quant_local == QUANT_FP16) {
        q_bytes = (uint64_t)q_rows * d * 2;
        k_bytes = (uint64_t)k_rows * d * 2;
        v_bytes = (uint64_t)v_rows * d * 2;
    } else if (qkv_quant_local == QUANT_Q6_K) {
        q_bytes = (uint64_t)q_rows * (d / 256) * 210;
        k_bytes = (uint64_t)k_rows * (d / 256) * 210;
        v_bytes = (uint64_t)v_rows * (d / 256) * 210;
    } else if (qkv_quant_local == QUANT_Q4_K) {
        q_bytes = (uint64_t)q_rows * (d / 256) * 144;
        k_bytes = (uint64_t)k_rows * (d / 256) * 144;
        v_bytes = (uint64_t)v_rows * (d / 256) * 144;
    }

    auto q_raw = download_raw_bytes(dev, weight_buf, off.q_offset, q_bytes);
    auto k_raw = download_raw_bytes(dev, weight_buf, off.k_offset, k_bytes);
    auto v_raw = download_raw_bytes(dev, weight_buf, off.v_offset, v_bytes);

    std::vector<float> cpu_qkv_expected(qkv_total, 0.0f);

    if (q_raw.size() >= q_bytes && k_raw.size() >= k_bytes && v_raw.size() >= v_bytes) {
        std::vector<float> q_deq, k_deq, v_deq;
        if (qkv_quant_local == QUANT_FP16) {
            q_deq = cpu_dequant_fp16(q_raw.data(), q_rows, d);
            k_deq = cpu_dequant_fp16(k_raw.data(), k_rows, d);
            v_deq = cpu_dequant_fp16(v_raw.data(), v_rows, d);
        } else if (qkv_quant_local == QUANT_Q6_K) {
            q_deq = cpu_dequant_q6_k(q_raw.data(), q_rows, d);
            k_deq = cpu_dequant_q6_k(k_raw.data(), k_rows, d);
            v_deq = cpu_dequant_q6_k(v_raw.data(), v_rows, d);
        } else if (qkv_quant_local == QUANT_Q4_K) {
            q_deq = cpu_dequant_q4_k(q_raw.data(), q_rows, d);
            k_deq = cpu_dequant_q4_k(k_raw.data(), k_rows, d);
            v_deq = cpu_dequant_q4_k(v_raw.data(), v_rows, d);
        } else {
            q_deq = cpu_dequant_q8_0(q_raw.data(), q_rows, d);
            k_deq = cpu_dequant_q8_0(k_raw.data(), k_rows, d);
            v_deq = cpu_dequant_q8_0(v_raw.data(), v_rows, d);
        }

        // Use the CPU attn_norm output as input for QKV
        std::vector<float> qkv_input = (attn_norm_weight.size() >= d)
            ? cpu_rms_norm(cpu_embedding, attn_norm_weight, norm_eps, d)
            : cpu_embedding;

        auto q_out = cpu_matmul(qkv_input, q_deq, 1, q_rows, d);
        auto k_out = cpu_matmul(qkv_input, k_deq, 1, k_rows, d);
        auto v_out = cpu_matmul(qkv_input, v_deq, 1, v_rows, d);

        cpu_rope(q_out, n_heads, head_dim, rope_th, pc.kv_cache_pos);
        cpu_rope(k_out, n_kv_heads, head_dim, rope_th, pc.kv_cache_pos);

        memcpy(cpu_qkv_expected.data(), q_out.data(), q_rows * sizeof(float));
        memcpy(cpu_qkv_expected.data() + q_rows, k_out.data(), k_rows * sizeof(float));
        memcpy(cpu_qkv_expected.data() + q_rows + k_rows, v_out.data(), v_rows * sizeof(float));
    }

    // ── (c) attn_qkv GPU dispatch ──────────────────────────────────────────
    {
        std::vector<float> gpu_qkv;
        pc.norm_type = 0;
        all_ok = all_ok && dispatch_validate("attn_qkv", pipe_attn_qkv,
                &val_norm, &val_qkv,
                &model->kv_cache.buffer, &model->kv_cache.buffer, nullptr,
                (qkv_total + 3) / 4,   // 1 subgroup (4/WG) per element
                cpu_qkv_expected, gpu_qkv, qkv_total);
    }

    // ── (d) attn_compute GPU dispatch ──────────────────────────────────────
    // CPU attention: for position 0 (self-attention on a single token)
    // Q · K^T = scalar (since seq_len=1), score = Q[0..hd-1] · K[0..hd-1]
    // softmax([score]) = [1.0]
    // attention[head] = 1.0 * V[0..hd-1] = V[0..hd-1]
    // So for single position, attention output = V values per head
    std::vector<float> cpu_attn_expected(n_heads * head_dim, 0.0f);
    for (uint32_t h = 0; h < n_heads; h++) {
        uint32_t kv_h = gqa_map(h, n_heads, n_kv_heads);
        uint32_t q_base = h * head_dim;
        uint32_t k_base = q_rows + kv_h * head_dim;
        uint32_t v_base = q_rows + k_rows + kv_h * head_dim;

        // Single position: softmax over one element = 1.0
        for (uint32_t j = 0; j < head_dim; j++) {
            cpu_attn_expected[q_base + j] = cpu_qkv_expected[v_base + j];
        }
    }

    {
        std::vector<float> gpu_attn;
        all_ok = all_ok && dispatch_validate("attn_compute", pipe_attn_comp,
                &val_qkv, &val_attn,
                &model->kv_cache.buffer, &model->kv_cache.buffer, nullptr,
                n_heads,
                cpu_attn_expected, gpu_attn, n_heads * head_dim);
    }

    // ── CPU: O weight download + attn_output CPU ───────────────────────────
    int o_quant_local = (int)model->o_quant;
    uint64_t o_bytes = d * (n_heads * head_dim / 32) * 34;
    if (o_quant_local == QUANT_FP16) {
        o_bytes = (uint64_t)d * (n_heads * head_dim) * 2;
    } else if (o_quant_local == QUANT_Q6_K) {
        o_bytes = (uint64_t)d * ((n_heads * head_dim) / 256) * 210;
    } else if (o_quant_local == QUANT_Q4_K) {
        o_bytes = (uint64_t)d * ((n_heads * head_dim) / 256) * 144;
    }
    auto o_raw = download_raw_bytes(dev, weight_buf, off.o_offset, o_bytes);

    std::vector<float> cpu_attn_output_expected(d);
    if (o_raw.size() >= o_bytes) {
        std::vector<float> o_deq;
        if (o_quant_local == QUANT_FP16) {
            o_deq = cpu_dequant_fp16(o_raw.data(), d, n_heads * head_dim);
        } else if (o_quant_local == QUANT_Q6_K) {
            o_deq = cpu_dequant_q6_k(o_raw.data(), d, n_heads * head_dim);
        } else if (o_quant_local == QUANT_Q4_K) {
            o_deq = cpu_dequant_q4_k(o_raw.data(), d, n_heads * head_dim);
        } else {
            o_deq = cpu_dequant_q8_0(o_raw.data(), d, n_heads * head_dim);
        }

        auto attn_out = cpu_matmul(cpu_attn_expected, o_deq, 1, d, n_heads * head_dim);

        // attn_output shader adds residual from hidden_out.y[out_row]
        // hidden_out = hidden_buf[1] which is fresh (should be 0, but could be stale)
        // For correct behavior: residual = cpu_embedding (the original hidden state)
        for (uint32_t i = 0; i < d; i++) {
            cpu_attn_output_expected[i] = cpu_embedding[i] + attn_out[i];
        }
    }

    // ── (e) attn_output GPU dispatch ───────────────────────────────────────
    {
        std::vector<float> gpu_attn_out;
        all_ok = all_ok && dispatch_validate("attn_output", pipe_attn_output,
                &val_attn, &val_hidden1,
                nullptr, nullptr, &val_hidden0,   // scratch = residual (layer input)
                (d + 3) / 4,   // 1 subgroup (4/WG) per output row
                cpu_attn_output_expected, gpu_attn_out, d);
    }

    // ── CPU: ffn_norm weight download ─────────────────────────────────────
    auto ffn_norm_weight = download_fp16_weights(dev, weight_buf,
                                                  off.ffn_norm_offset, d);
    std::vector<float> cpu_ffn_norm_expected;
    if (ffn_norm_weight.size() >= d) {
        cpu_ffn_norm_expected = cpu_rms_norm(cpu_attn_output_expected, ffn_norm_weight, norm_eps, d);
    }

    // ── ffn_norm GPU dispatch (separate step not in original pipeline, but we do it) ─
    // Actually the rms_norm with norm_type=1 re-uses norm_buf.
    // For validation, we dispatch it manually.
    {
        std::vector<float> gpu_ffn_norm;
        pc.norm_type = 1;
        all_ok = all_ok && dispatch_validate("ffn_norm", pipe_rms_norm,
                &val_hidden1, &val_norm,
                nullptr, nullptr, nullptr,
                1,
                cpu_ffn_norm_expected, gpu_ffn_norm, d);
    }

    // ── CPU: gate + up weight download + CPU computation ──────────────────
    int gate_up_quant_local = (int)model->gate_up_quant;
    uint64_t gate_bytes = ffn_dim * (d / 32) * 34;
    uint64_t up_bytes   = ffn_dim * (d / 32) * 34;
    if (gate_up_quant_local == QUANT_FP16) {
        gate_bytes = (uint64_t)ffn_dim * d * 2;
        up_bytes   = (uint64_t)ffn_dim * d * 2;
    } else if (gate_up_quant_local == QUANT_Q6_K) {
        gate_bytes = (uint64_t)ffn_dim * (d / 256) * 210;
        up_bytes   = (uint64_t)ffn_dim * (d / 256) * 210;
    } else if (gate_up_quant_local == QUANT_Q4_K) {
        gate_bytes = (uint64_t)ffn_dim * (d / 256) * 144;
        up_bytes   = (uint64_t)ffn_dim * (d / 256) * 144;
    }
    auto gate_raw = download_raw_bytes(dev, weight_buf, off.gate_offset, gate_bytes);
    auto up_raw   = download_raw_bytes(dev, weight_buf, off.up_offset, up_bytes);

    std::vector<float> cpu_ffn_gate_up_expected(ffn_dim, 0.0f);
    if (gate_raw.size() >= gate_bytes && up_raw.size() >= up_bytes) {
        std::vector<float> gate_deq, up_deq;
        if (gate_up_quant_local == QUANT_FP16) {
            gate_deq = cpu_dequant_fp16(gate_raw.data(), ffn_dim, d);
            up_deq   = cpu_dequant_fp16(up_raw.data(), ffn_dim, d);
        } else if (gate_up_quant_local == QUANT_Q6_K) {
            gate_deq = cpu_dequant_q6_k(gate_raw.data(), ffn_dim, d);
            up_deq   = cpu_dequant_q6_k(up_raw.data(), ffn_dim, d);
        } else if (gate_up_quant_local == QUANT_Q4_K) {
            gate_deq = cpu_dequant_q4_k(gate_raw.data(), ffn_dim, d);
            up_deq   = cpu_dequant_q4_k(up_raw.data(), ffn_dim, d);
        } else {
            gate_deq = cpu_dequant_q8_0(gate_raw.data(), ffn_dim, d);
            up_deq   = cpu_dequant_q8_0(up_raw.data(), ffn_dim, d);
        }

        auto gate_out = cpu_matmul(cpu_ffn_norm_expected, gate_deq, 1, ffn_dim, d);
        auto up_out   = cpu_matmul(cpu_ffn_norm_expected, up_deq, 1, ffn_dim, d);

        for (uint32_t i = 0; i < ffn_dim; i++) {
            float silu_val = gate_out[0 * ffn_dim + i];
            silu_val = silu_val / (1.0f + expf(-silu_val));
            cpu_ffn_gate_up_expected[i] = silu_val * up_out[0 * ffn_dim + i];
        }
    }

    // ── (f) ffn_gate_up GPU dispatch ──────────────────────────────────────
    {
        std::vector<float> gpu_ffn;
        pc.norm_type = 1;
        all_ok = all_ok && dispatch_validate("ffn_gate_up", pipe_ffn_gate_up,
                &val_norm, &val_ffn,
                nullptr, nullptr, &val_ffn,
                (ffn_dim + 3) / 4,   // 1 subgroup (4/WG) per output row
                cpu_ffn_gate_up_expected, gpu_ffn, ffn_dim);
    }

    // ── CPU: down weight download + CPU computation ───────────────────────
    int down_quant_local = (int)model->down_quant;
    uint64_t down_bytes = d * (ffn_dim / 32) * 34;
    if (down_quant_local == QUANT_FP16) {
        down_bytes = (uint64_t)d * ffn_dim * 2;
    } else if (down_quant_local == QUANT_Q6_K) {
        down_bytes = (uint64_t)d * (ffn_dim / 256) * 210;
    } else if (down_quant_local == QUANT_Q4_K) {
        down_bytes = (uint64_t)d * (ffn_dim / 256) * 144;
    }
    auto down_raw = download_raw_bytes(dev, weight_buf, off.down_offset, down_bytes);

    std::vector<float> cpu_ffn_down_expected(d);
    if (down_raw.size() >= down_bytes) {
        std::vector<float> down_deq;
        if (down_quant_local == QUANT_FP16) {
            down_deq = cpu_dequant_fp16(down_raw.data(), d, ffn_dim);
        } else if (down_quant_local == QUANT_Q6_K) {
            down_deq = cpu_dequant_q6_k(down_raw.data(), d, ffn_dim);
        } else if (down_quant_local == QUANT_Q4_K) {
            down_deq = cpu_dequant_q4_k(down_raw.data(), d, ffn_dim);
        } else {
            down_deq = cpu_dequant_q8_0(down_raw.data(), d, ffn_dim);
        }

        auto down_out = cpu_matmul(cpu_ffn_gate_up_expected, down_deq, 1, d, ffn_dim);

        // ffn_down adds residual from hidden_out.y[out_row] = post-attention hidden
        for (uint32_t i = 0; i < d; i++) {
            cpu_ffn_down_expected[i] = cpu_attn_output_expected[i] + down_out[i];
        }
    }

    // ── (g) ffn_down GPU dispatch ─────────────────────────────────────────
    {
        std::vector<float> gpu_ffn_down;
        all_ok = all_ok && dispatch_validate("ffn_down", pipe_ffn_down,
                &val_ffn, &val_hidden1,
                nullptr, nullptr, &val_ffn,
                (d + 3) / 4,   // 1 subgroup (4/WG) per output row
                cpu_ffn_down_expected, gpu_ffn_down, d);
    }

    // ── (h) Multi-token attention check (seq_len=2) ───────────────────────
    // Dispatch token 2's QKV at kv_cache_pos=1, then attention over 2 positions.
    if (attn_norm_weight.size() >= d) {
        // CPU reference for token 2 QKV (rope at position 1)
        std::vector<float> qkv_input2 = (attn_norm_weight.size() >= d)
            ? cpu_rms_norm(cpu_embedding, attn_norm_weight, norm_eps, d)
            : cpu_embedding;
        std::vector<float> qkv2_exp(qkv_total, 0.0f);
        if (q_raw.size() >= q_bytes && k_raw.size() >= k_bytes && v_raw.size() >= v_bytes) {
            std::vector<float> q2, k2, v2;
            if (qkv_quant_local == QUANT_FP16) {
                q2 = cpu_dequant_fp16(q_raw.data(), q_rows, d);
                k2 = cpu_dequant_fp16(k_raw.data(), k_rows, d);
                v2 = cpu_dequant_fp16(v_raw.data(), v_rows, d);
            } else if (qkv_quant_local == QUANT_Q6_K) {
                q2 = cpu_dequant_q6_k(q_raw.data(), q_rows, d);
                k2 = cpu_dequant_q6_k(k_raw.data(), k_rows, d);
                v2 = cpu_dequant_q6_k(v_raw.data(), v_rows, d);
            } else if (qkv_quant_local == QUANT_Q4_K) {
                q2 = cpu_dequant_q4_k(q_raw.data(), q_rows, d);
                k2 = cpu_dequant_q4_k(k_raw.data(), k_rows, d);
                v2 = cpu_dequant_q4_k(v_raw.data(), v_rows, d);
            } else {
                q2 = cpu_dequant_q8_0(q_raw.data(), q_rows, d);
                k2 = cpu_dequant_q8_0(k_raw.data(), k_rows, d);
                v2 = cpu_dequant_q8_0(v_raw.data(), v_rows, d);
            }
            auto q_out2 = cpu_matmul(qkv_input2, q2, 1, q_rows, d);
            auto k_out2 = cpu_matmul(qkv_input2, k2, 1, k_rows, d);
            auto v_out2 = cpu_matmul(qkv_input2, v2, 1, v_rows, d);
            cpu_rope(q_out2, n_heads, head_dim, rope_th, 1);   // position 1
            cpu_rope(k_out2, n_kv_heads, head_dim, rope_th, 1);
            memcpy(qkv2_exp.data(), q_out2.data(), q_rows * sizeof(float));
            memcpy(qkv2_exp.data() + q_rows, k_out2.data(), k_rows * sizeof(float));
            memcpy(qkv2_exp.data() + q_rows + k_rows, v_out2.data(), v_rows * sizeof(float));
        }

        // Re-run the attn_norm dispatch so val_norm holds the ATTN-normed input
        // again (the ffn_norm dispatch above clobbered it).
        {
            std::vector<float> gpu_norm_again, dummy_exp = cpu_embedding;
            pc.norm_type = 0;
            dispatch_validate("attn_norm_again", pipe_rms_norm,
                    &val_hidden0, &val_norm,
                    nullptr, nullptr, nullptr,
                    1,
                    dummy_exp, gpu_norm_again, d);
        }

        // GPU: dispatch token 2 QKV (writes KV cache at position 1)
        std::vector<float> gpu_qkv2;
        pc.kv_cache_pos = 1;
        pc.seq_len      = 2;
        dispatch_validate("attn_qkv_t2", pipe_attn_qkv,
                &val_norm, &val_qkv,
                &model->kv_cache.buffer, &model->kv_cache.buffer, nullptr,
                (qkv_total + 3) / 4,
                qkv2_exp, gpu_qkv2, qkv_total);

        // CPU reference for seq_len=2 attention (q2 from pos1, k/v from pos0 & pos1)
        std::vector<float> cpu_attn2_expected(n_heads * head_dim, 0.0f);
        for (uint32_t h = 0; h < n_heads; h++) {
            uint32_t kv_h = gqa_map(h, n_heads, n_kv_heads);
            std::vector<float> probs(2);
            float maxs = -1e30f;
            for (uint32_t p = 0; p < 2; p++) {
                float s = 0.0f;
                for (uint32_t j = 0; j < head_dim; j++) {
                    float qv = qkv2_exp[h * head_dim + j];
                    // K section starts at q_rows; token 0 uses pos-0 K, token 1 uses pos-1 K
                    uint32_t koff = q_rows + kv_h * head_dim + j;
                    float kv = (p == 0) ? cpu_qkv_expected[koff] : qkv2_exp[koff];
                    s += qv * kv;
                }
                s *= attn_scale;
                probs[p] = s;
                maxs = (s > maxs) ? s : maxs;
            }
            float sum = 0.0f;
            for (uint32_t p = 0; p < 2; p++) { probs[p] = expf(probs[p] - maxs); sum += probs[p]; }
            for (uint32_t j = 0; j < head_dim; j++) {
                float acc = 0.0f;
                for (uint32_t p = 0; p < 2; p++) {
                    uint32_t voff = q_rows + k_rows + kv_h * head_dim + j;
                    float vv = (p == 0) ? cpu_qkv_expected[voff] : qkv2_exp[voff];
                    acc += (probs[p] / sum) * vv;
                }
                cpu_attn2_expected[h * head_dim + j] = acc;
            }
        }

        std::vector<float> gpu_attn2;
        pc.seq_len = 2;
        dispatch_validate("attn_compute_t2", pipe_attn_comp,
                &val_qkv, &val_attn,
                &model->kv_cache.buffer, &model->kv_cache.buffer, nullptr,
                n_heads,
                cpu_attn2_expected, gpu_attn2, n_heads * head_dim);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────
    destroy_dev_buf(&val_hidden0);
    destroy_dev_buf(&val_norm);
    destroy_dev_buf(&val_qkv);
    destroy_dev_buf(&val_attn);
    destroy_dev_buf(&val_hidden1);
    destroy_dev_buf(&val_ffn);

    if (all_ok) {
        fprintf(log_file, "[DONE] Layer %u validation: ALL PASSED\n", layer_idx);
        RDNA4_LOG("[VALIDATE] Layer %u: ALL PASSED", layer_idx);
    } else {
        fprintf(log_file, "[DONE] Layer %u validation: SOME FAILURES\n", layer_idx);
        RDNA4_LOG("[VALIDATE] Layer %u: SOME FAILURES", layer_idx);
    }

    return all_ok;
}

static int find_nan_layer_from_decode(vk_device_t* dev, vk_model_t* model,
                                             vk_session_t* session,
                                             int32_t token_id, FILE* log_file) {
    uint32_t d = model->config.d;
    uint32_t n_layers = model->config.n_layers;
    uint32_t ffn_dim = model->config.ffn_dim;

    VkPipeline pipe_rms_norm    = find_pipeline_local(session, OP_RMS_NORM, QUANT_FP16);
    VkPipeline pipe_attn_qkv    = find_pipeline_local(session, OP_ATTN_QKV, (int)model->qkv_quant);
    VkPipeline pipe_attn_comp   = find_pipeline_local(session, OP_ATTN_COMPUTE, QUANT_FP16);
    VkPipeline pipe_attn_output = find_pipeline_local(session, OP_ATTN_OUTPUT, (int)model->o_quant);
    VkPipeline pipe_ffn_gate_up = find_pipeline_local(session, OP_FFN_GATE_UP, (int)model->gate_up_quant);
    VkPipeline pipe_ffn_down    = find_pipeline_local(session, OP_FFN_DOWN, (int)model->down_quant);

    if (!pipe_rms_norm || !pipe_attn_qkv || !pipe_attn_comp ||
        !pipe_attn_output || !pipe_ffn_gate_up || !pipe_ffn_down) {
        fprintf(log_file, "[NaN-SCAN] Missing pipelines, cannot scan\n");
        return -1;
    }

    push_constants_t pc = session->push_constants;
    pc.kv_cache_pos = 0;
    pc.seq_len      = 1;

    VkPipelineLayout layout = session->shared_pipeline_layout;
    VkDescriptorSet  set0   = session->desc.set0_weight_set;
    VkDescriptorSet  set2   = session->desc.set2_table_set;
    VkDevice         device = dev->device;

    vk_buffer_t* hidden_bufs = session->decode_state.hidden_buf;
    vk_buffer_t* norm_buf    = &session->decode_state.norm_scratch;
    vk_buffer_t* qkv_buf     = &session->decode_state.qkv_scratch;
    vk_buffer_t* attn_buf    = &session->decode_state.attn_scratch;
    vk_buffer_t* ffn_buf     = &session->decode_state.ffn_scratch;

    vk_buffer_t staging;
    if (!staging_buffer_create(dev, d * 2, &staging)) {
        fprintf(log_file, "[NaN-SCAN] Failed to create staging buffer\n");
        return -1;
    }

    auto bind_sets = [&](VkCommandBuffer cb) {
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
            layout, 0, 1, &set0, 0, nullptr);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
            layout, 2, 1, &set2, 0, nullptr);
    };

    fprintf(log_file, "\n=== NaN Layer Scan (decode pipeline, one layer at a time) ===\n");

    for (uint32_t L = 0; L < n_layers; L++) {
        VkCommandBuffer cb = alloc_cb(dev);
        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cb, &bi));

        bind_sets(cb);

        VkBufferCopy emb_copy = {};
        emb_copy.srcOffset = (uint64_t)token_id * (uint64_t)d * 2ull;
        emb_copy.dstOffset = 0;
        emb_copy.size      = (uint64_t)d * 2ull;
        vkCmdCopyBuffer(cb, model->embedding_buf.buffer, hidden_bufs[0].buffer, 1, &emb_copy);
        shader_barrier(cb, hidden_bufs[0].buffer,
                       VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        uint32_t toggle = 0;
        for (uint32_t l = 0; l <= L; l++) {
            pc.layer_idx = l;
            uint32_t in_toggle  = toggle;
            uint32_t out_toggle = 1 - toggle;

            pc.norm_type = 0;
            vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vk_descriptor_arena_push_set1(&session->desc, device, cb, layout,
                &hidden_bufs[in_toggle], norm_buf, nullptr, nullptr, nullptr);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_rms_norm);
            vkCmdDispatch(cb, 1, 1, 1);
            shader_barrier(cb, norm_buf->buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

            vk_descriptor_arena_push_set1(&session->desc, device, cb, layout,
                norm_buf, qkv_buf, &model->kv_cache.buffer, &model->kv_cache.buffer, nullptr);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_attn_qkv);
            vkCmdDispatch(cb, ((model->config.n_heads * model->config.head_dim + 2 * model->config.n_kv_heads * model->config.head_dim) + 3) / 4, 1, 1);
            shader_barrier(cb, qkv_buf->buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

            vk_descriptor_arena_push_set1(&session->desc, device, cb, layout,
                qkv_buf, attn_buf, &model->kv_cache.buffer, &model->kv_cache.buffer, nullptr);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_attn_comp);
            vkCmdDispatch(cb, model->config.n_heads, 1, 1);
            shader_barrier(cb, attn_buf->buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

            vk_descriptor_arena_push_set1(&session->desc, device, cb, layout,
                attn_buf, &hidden_bufs[out_toggle], nullptr, nullptr, nullptr);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_attn_output);
            vkCmdDispatch(cb, (d + 3) / 4, 1, 1);
            shader_barrier(cb, hidden_bufs[out_toggle].buffer,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

            pc.norm_type = 1;
            vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vk_descriptor_arena_push_set1(&session->desc, device, cb, layout,
                &hidden_bufs[out_toggle], norm_buf, nullptr, nullptr, nullptr);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_rms_norm);
            vkCmdDispatch(cb, 1, 1, 1);
            shader_barrier(cb, norm_buf->buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

            vk_descriptor_arena_push_set1(&session->desc, device, cb, layout,
                norm_buf, ffn_buf, nullptr, nullptr, nullptr);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_ffn_gate_up);
            vkCmdDispatch(cb, (ffn_dim + 3) / 4, 1, 1);
            shader_barrier(cb, ffn_buf->buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

            vk_descriptor_arena_push_set1(&session->desc, device, cb, layout,
                ffn_buf, &hidden_bufs[out_toggle], nullptr, nullptr, nullptr);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_ffn_down);
            vkCmdDispatch(cb, (d + 3) / 4, 1, 1);
            shader_barrier(cb, hidden_bufs[out_toggle].buffer,
                VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

            toggle = out_toggle;
        }

        VkBufferCopy copy = {0, 0, (uint64_t)d * 2ull};
        vkCmdCopyBuffer(cb, hidden_bufs[toggle].buffer, staging.buffer, 1, &copy);

        {
            VkBufferMemoryBarrier barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
            barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
            barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
            barrier.buffer = staging.buffer;
            barrier.size   = VK_WHOLE_SIZE;
            vkCmdPipelineBarrier(cb,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_HOST_BIT,
                0, 0, nullptr, 1, &barrier, 0, nullptr);
        }

        VK_CHECK(vkEndCommandBuffer(cb));
        submit_and_wait(dev, cb);
        free_cb(dev, cb);

        std::vector<float> hidden(d);
        const uint16_t* src = (const uint16_t*)staging.mapped_ptr;
        for (uint32_t i = 0; i < d; i++) {
            hidden[i] = f16_to_f32(src[i]);
        }

        bool has_nan = check_for_nan(hidden);
        if (has_nan) {
            fprintf(log_file, "[NaN-SCAN] *** NaN DETECTED after layer %u (of %u layers) ***\n", L, n_layers);
            print_nan_indices(log_file, "hidden_after_layer", hidden, 20);
            staging_buffer_destroy(dev, &staging);
            return (int)L;
        }

        fprintf(log_file, "[NaN-SCAN] Layer %u: hidden state clean (no NaN)\n", L);
    }

    staging_buffer_destroy(dev, &staging);
    return -1;
}

bool vk_validate_run(vk_device_t* dev, vk_model_t* model, vk_session_t* session,
                     int32_t token_id, uint32_t kv_pos, FILE* log_file) {
    fprintf(log_file, "=== GPU Validation Run ===\n");
    fprintf(log_file, "token_id=%d  kv_pos=%u\n", token_id, kv_pos);
    fprintf(log_file, "config: d=%u ffn=%u n_heads=%u n_kv=%u hd=%u L=%u\n",
            model->config.d, model->config.ffn_dim, model->config.n_heads,
            model->config.n_kv_heads, model->config.head_dim, model->config.n_layers);

    session->push_constants.token_id     = token_id;
    session->push_constants.kv_cache_pos = kv_pos;
    session->push_constants.seq_len      = 1;

    uint32_t d = model->config.d;
    uint32_t n_layers = model->config.n_layers;

    uint64_t embedding_offset;
    uint64_t embedding_bytes  = (uint64_t)d * 2ull;
    uint32_t emb_q = (uint32_t)model->emb_quant;
    if (emb_q == QUANT_Q6_K) {
        // block_q6_K: 256 elems, 210 bytes -> token row stride
        uint64_t row_stride = ((uint64_t)d / 256ull) * 210ull;
        embedding_offset = (uint64_t)token_id * row_stride;
        embedding_bytes  = row_stride;
    } else if (emb_q == QUANT_Q4_K) {
        uint64_t row_stride = ((uint64_t)d / 256ull) * 144ull;
        embedding_offset = (uint64_t)token_id * row_stride;
        embedding_bytes  = row_stride;
    } else if (emb_q == QUANT_Q8_0 || emb_q == QUANT_Q4_0) {
        uint64_t row_stride = ((uint64_t)d / 32ull) * 34ull;
        embedding_offset = (uint64_t)token_id * row_stride;
        embedding_bytes  = row_stride;
    } else {
        embedding_offset = (uint64_t)token_id * (uint64_t)d * 2ull;
    }

    auto emb_raw = download_raw_bytes(dev, &model->embedding_buf,
                                       embedding_offset, embedding_bytes);

    std::vector<float> cpu_embedding(d);
    if (emb_raw.size() >= embedding_bytes) {
        if (emb_q == QUANT_Q6_K) {
            // dequant block_q6_K (llama.cpp struct layout) into fp32:
            //   ql[128]@0, qh[64]@128, scales[16]@192, d(fp16)@208
            const uint8_t* eb = emb_raw.data();
            for (uint32_t sb = 0; sb < d / 256; sb++) {
                const uint8_t* b = eb + (size_t)sb * 210u;
                float d_sc = f16_to_f32((uint16_t)b[208] | ((uint16_t)b[209] << 8));
                for (uint32_t half = 0; half < 2; half++) {
                    const uint8_t* ql = b + half * 64u;
                    const uint8_t* qh = b + 128u + half * 32u;
                    const int8_t* sc = (const int8_t*)(b + 192u + half * 8u);
                    for (uint32_t l = 0; l < 32; l++) {
                        int is = (int)(l / 16);
                        int q1 = (int)(ql[l] & 0xF) | (((int)qh[l] >> 0) & 3) << 4;
                        int q2 = (int)(ql[l + 32] & 0xF) | (((int)qh[l] >> 2) & 3) << 4;
                        int q3 = (int)(ql[l] >> 4) | (((int)qh[l] >> 4) & 3) << 4;
                        int q4 = (int)(ql[l + 32] >> 4) | (((int)qh[l] >> 6) & 3) << 4;
                        uint32_t base = sb * 256u + half * 128u;
                        cpu_embedding[base + l + 0]  = d_sc * (float)sc[is + 0] * (float)(q1 - 32);
                        cpu_embedding[base + l + 32] = d_sc * (float)sc[is + 2] * (float)(q2 - 32);
                        cpu_embedding[base + l + 64] = d_sc * (float)sc[is + 4] * (float)(q3 - 32);
                        cpu_embedding[base + l + 96] = d_sc * (float)sc[is + 6] * (float)(q4 - 32);
                    }
                }
            }
        } else if (emb_q == QUANT_Q4_K) {
            const uint8_t* eb = emb_raw.data();
            for (uint32_t sb = 0; sb < d / 256; sb++) {
                const uint8_t* b = eb + (size_t)sb * 144u;
                float dq = f16_to_f32((uint16_t)b[0] | ((uint16_t)b[1] << 8));
                float mn = f16_to_f32((uint16_t)b[2] | ((uint16_t)b[3] << 8));
                for (uint32_t j = 0; j < 256; j += 64) {
                    for (uint32_t half = 0; half < 2; half++) {
                        int is = (int)(j / 64) * 2 + half;
                        uint8_t sc_b, m_b;
                        // get_scale_min_k4: is<4 -> scales[is]&63, scales[is+4]&63
                        if (is < 4) {
                            sc_b = b[4 + is] & 63;
                            m_b  = b[8 + is] & 63;
                        } else {
                            sc_b = (b[is + 4] & 0xF) | ((b[is - 4] >> 6) << 4);
                            m_b  = (b[is + 4] >> 4) | ((b[is] >> 6) << 4);
                        }
                        float dl = dq * (float)sc_b;
                        float ml = mn * (float)m_b;
                        const uint8_t* q = b + 16u + j / 2u;
                        for (uint32_t l = 0; l < 32; l++) {
                            uint32_t base = sb * 256u + j + half * 32u;
                            cpu_embedding[base + l] = dl * (float)(q[l + half * 32u] & 0xF) - ml;
                            cpu_embedding[base + l + 32] = dl * (float)(q[l + half * 32u] >> 4) - ml;
                        }
                    }
                }
            }
        } else if (emb_q == QUANT_Q8_0) {
            // dequant block_q8_0: 32 elems, 34 bytes (d fp16 + 32 int8)
            const uint8_t* eb = emb_raw.data();
            for (uint32_t blk = 0; blk < d / 32; blk++) {
                const uint8_t* b = eb + (size_t)blk * 34u;
                float d_sc = f16_to_f32((uint16_t)b[0] | ((uint16_t)b[1] << 8));
                for (uint32_t e = 0; e < 32; e++) {
                    int8_t q = (int8_t)b[2 + e];
                    cpu_embedding[blk * 32u + e] = d_sc * (float)(int)q;
                }
            }
        } else {
            for (uint32_t i = 0; i < d; i++) {
                uint16_t v = (uint16_t)emb_raw[i * 2] | ((uint16_t)emb_raw[i * 2 + 1] << 8);
                cpu_embedding[i] = f16_to_f32(v);
            }
        }
        fprintf(log_file, "Embedding row %d downloaded: first 4 = [%.4f, %.4f, %.4f, %.4f]\n",
                token_id,
                cpu_embedding[0], cpu_embedding[1], cpu_embedding[2], cpu_embedding[3]);
    } else {
        fprintf(stderr, "[VALIDATE ERROR] Failed to download embedding row %d\n", token_id);
        return false;
    }

    if (check_for_nan(cpu_embedding)) {
        fprintf(log_file, "[NaN] CPU embedding contains NaN!\n");
    }

    // ── Step A: Layer-by-layer decode scan to find first NaN layer ────────
    int first_nan_layer = find_nan_layer_from_decode(dev, model, session, token_id, log_file);

    // ── Step B: Per-layer validation (all ops, GPU vs CPU) ─────────────────
    bool all_layers_ok = true;
    fprintf(log_file, "\n=== Per-Layer Per-Op Validation ===\n");
    for (uint32_t l = 0; l < n_layers; l++) {
        fprintf(log_file, "\n----- Layer %u / %u -----\n", l, n_layers - 1);
        bool layer_ok = validate_layer(dev, model, session, l, token_id, cpu_embedding, log_file);
        if (!layer_ok) {
            all_layers_ok = false;
        }
    }

    // ── Step C: Full chained CPU reference (all layers, single token) ─────
    {
        fprintf(log_file, "\n=== Chained CPU Reference (5 layers) ===\n");
        uint32_t n_heads   = model->config.n_heads;
        uint32_t n_kv      = model->config.n_kv_heads;
        uint32_t head_dim  = model->config.head_dim;
        uint32_t ffn_dim   = model->config.ffn_dim;
        float    norm_eps  = model->config.norm_eps;
        float    rope_th   = model->config.rope_theta;

        std::vector<float> h = cpu_embedding;

        for (uint32_t l = 0; l < n_layers; l++) {
            vk_buffer_t* wb = &model->weight_bufs[l];
            const layer_weight_offsets_t& wo = model->weight_offsets[l];
            uint32_t q_rows = n_heads * head_dim;
            uint32_t k_rows = n_kv * head_dim;
            uint32_t v_rows = n_kv * head_dim;

            // Per-role quant: a mixed-quant GGUF can give ffn_down (or attn_v,
            // etc.) a different type than the rest of the layer's tensors.
            auto dq = [&](uint64_t off, uint32_t rows, uint32_t cols, int qt) {
                std::vector<float> M;
                if (qt == QUANT_FP16) {
                    auto raw = download_raw_bytes(dev, wb, off, (uint64_t)rows * cols * 2);
                    if (raw.size() >= (uint64_t)rows * cols * 2)
                        M = cpu_dequant_fp16(raw.data(), rows, cols);
                } else if (qt == QUANT_Q6_K) {
                    auto raw = download_raw_bytes(dev, wb, off, (uint64_t)rows * (cols / 256) * 210);
                    if (raw.size() >= (uint64_t)rows * (cols / 256) * 210)
                        M = cpu_dequant_q6_k(raw.data(), rows, cols);
                } else if (qt == QUANT_Q4_K) {
                    auto raw = download_raw_bytes(dev, wb, off, (uint64_t)rows * (cols / 256) * 144);
                    if (raw.size() >= (uint64_t)rows * (cols / 256) * 144)
                        M = cpu_dequant_q4_k(raw.data(), rows, cols);
                } else {
                    auto raw = download_raw_bytes(dev, wb, off, (uint64_t)rows * (cols / 32) * 34);
                    if (raw.size() >= (uint64_t)rows * (cols / 32) * 34)
                        M = cpu_dequant_q8_0(raw.data(), rows, cols);
                }
                return M;
            };

            auto an_w = download_fp16_weights(dev, wb, wo.attn_norm_offset, d);
            auto fn_w = download_fp16_weights(dev, wb, wo.ffn_norm_offset, d);
            auto q_w = dq(wo.q_offset, q_rows, d, (int)model->qkv_quant);
            auto k_w = dq(wo.k_offset, k_rows, d, (int)model->qkv_quant);
            auto v_w = dq(wo.v_offset, v_rows, d, (int)model->qkv_quant);
            auto o_w = dq(wo.o_offset, d, q_rows, (int)model->o_quant);
            auto g_w = dq(wo.gate_offset, ffn_dim, d, (int)model->gate_up_quant);
            auto u_w = dq(wo.up_offset, ffn_dim, d, (int)model->gate_up_quant);
            auto dn_w = dq(wo.down_offset, d, ffn_dim, (int)model->down_quant);

            std::vector<float> xn = cpu_rms_norm(h, an_w, norm_eps, d);
            auto q = cpu_matmul(xn, q_w, 1, q_rows, d);
            auto k = cpu_matmul(xn, k_w, 1, k_rows, d);
            auto v = cpu_matmul(xn, v_w, 1, v_rows, d);
            cpu_rope(q, n_heads, head_dim, rope_th, kv_pos);
            cpu_rope(k, n_kv, head_dim, rope_th, kv_pos);

            // seq_len=1: attention output = V per kv head, expanded to query heads
            std::vector<float> attn_out(q_rows);
            for (uint32_t hh = 0; hh < n_heads; hh++) {
                uint32_t kvh = gqa_map(hh, n_heads, n_kv);
                for (uint32_t j = 0; j < head_dim; j++)
                    attn_out[hh * head_dim + j] = v[kvh * head_dim + j];
            }
            auto proj = cpu_matmul(attn_out, o_w, 1, d, q_rows);
            for (uint32_t i = 0; i < d; i++) h[i] += proj[i];

            std::vector<float> hn = cpu_rms_norm(h, fn_w, norm_eps, d);
            auto gate = cpu_matmul(hn, g_w, 1, ffn_dim, d);
            auto up   = cpu_matmul(hn, u_w, 1, ffn_dim, d);
            std::vector<float> ff(ffn_dim);
            for (uint32_t i = 0; i < ffn_dim; i++)
                ff[i] = (gate[i] / (1.0f + expf(-gate[i]))) * up[i];
            auto dn = cpu_matmul(ff, dn_w, 1, d, ffn_dim);
            for (uint32_t i = 0; i < d; i++) h[i] += dn[i];
        }

        fprintf(log_file, "Chained CPU final hidden[0..7]=%.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f\n",
            h[0], h[1], h[2], h[3], h[4], h[5], h[6], h[7]);

        // lm_head logits (output.weight tied to token_embd)
        auto out_w = download_fp16_weights(dev, &model->lm_head_buf, 0, d);
        if (out_w.size() < d) {
            auto raw = download_raw_bytes(dev, &model->lm_head_buf, 0, (uint64_t)model->config.vocab_size * d * 2);
            if (raw.size() >= (uint64_t)model->config.vocab_size * d * 2)
                out_w = cpu_dequant_fp16(raw.data(), model->config.vocab_size, d);
        }
        if (out_w.size() >= (uint64_t)model->config.vocab_size * d) {
            for (uint32_t r = 0; r < 6; r++) {
                float logit = 0.0f;
                for (uint32_t k = 0; k < d; k++) logit += h[k] * out_w[(uint64_t)r * d + k];
                fprintf(log_file, "Chained CPU logit[%u]=%.4f\n", r, logit);
            }
        }
    }

    // ── Summary ───────────────────────────────────────────────────────────
    fprintf(log_file, "\n=== Validation Summary ===\n");
    if (first_nan_layer >= 0) {
        fprintf(log_file, "FIRST NaN LAYER: %d (from decode pipeline scan)\n", first_nan_layer);
    } else {
        fprintf(log_file, "NaN SCAN: No NaN detected in any layer's hidden state\n");
    }
    fprintf(log_file, "PER-OP VALIDATION: %s\n", all_layers_ok ? "ALL PASSED" : "SOME FAILURES");

    return all_layers_ok && (first_nan_layer < 0);
}
