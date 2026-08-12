#include "gguf_parser.h"
#include "common.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>

struct gguf_reader_t {
    const uint8_t* data;
    size_t         size;
    size_t         pos;
};

static uint32_t read_u32(gguf_reader_t* r) {
    uint32_t v;
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return v;
}

static int32_t read_i32(gguf_reader_t* r) {
    return (int32_t)read_u32(r);
}

static uint64_t read_u64(gguf_reader_t* r) {
    uint64_t v;
    memcpy(&v, r->data + r->pos, 8);
    r->pos += 8;
    return v;
}

static float read_f32(gguf_reader_t* r) {
    float v;
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return v;
}

static float read_f64(gguf_reader_t* r) {
    double v;
    memcpy(&v, r->data + r->pos, 8);
    r->pos += 8;
    return (float)v;
}

static bool read_bool(gguf_reader_t* r) {
    return r->data[r->pos++] != 0;
}

static uint64_t read_string(gguf_reader_t* r, const char** out_str) {
    uint64_t len = read_u64(r);
    *out_str = (const char*)(r->data + r->pos);
    r->pos += len;
    return len;
}

static void skip_value(gguf_reader_t* r, uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            r->pos += 1; break;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            r->pos += 2; break;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            r->pos += 4; break;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            r->pos += 8; break;
        case GGUF_TYPE_STRING: {
            uint64_t len = read_u64(r);
            r->pos += len;
            break;
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t array_type = read_u32(r);
            uint64_t array_len  = read_u64(r);
            for (uint64_t i = 0; i < array_len; i++) {
                skip_value(r, array_type);
            }
            break;
        }
        default:
            break;
    }
}

static void read_metadata_value(gguf_reader_t* r, uint32_t type, uint32_t* out_u32, float* out_f32, uint64_t* out_u64) {
    switch (type) {
        case GGUF_TYPE_UINT8:   *out_u32 = r->data[r->pos++]; break;
        case GGUF_TYPE_INT8:    *out_u32 = (uint32_t)(int8_t)r->data[r->pos++]; break;
        case GGUF_TYPE_UINT16:  *out_u32 = *(uint16_t*)(r->data + r->pos); r->pos += 2; break;
        case GGUF_TYPE_INT16:   *out_u32 = (uint32_t)(int16_t)(*(uint16_t*)(r->data + r->pos)); r->pos += 2; break;
        case GGUF_TYPE_UINT32:  *out_u32 = read_u32(r); break;
        case GGUF_TYPE_INT32:   *out_u32 = (uint32_t)read_i32(r); break;
        case GGUF_TYPE_FLOAT32: *out_f32 = read_f32(r); break;
        case GGUF_TYPE_UINT64:  *out_u64 = read_u64(r); break;
        case GGUF_TYPE_INT64:   *out_u64 = read_u64(r); break;
        case GGUF_TYPE_FLOAT64: *out_f32 = read_f64(r); break;
        case GGUF_TYPE_BOOL:    *out_u32 = read_bool(r) ? 1 : 0; break;
        default:
            skip_value(r, type);
            break;
    }
}

#define KEY_MATCHES(key_ptr, key_len, literal) \
    ((key_len) == (sizeof(literal) - 1) && memcmp((key_ptr), (literal), (key_len)) == 0)

bool gguf_parser_open(const char* path, gguf_file_t* out) {
    memset(out, 0, sizeof(*out));

    HANDLE hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ,
                                NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) {
        fprintf(stderr, "GGUF: Cannot open file: %s (error %lu)\n", path, GetLastError());
        return false;
    }

    LARGE_INTEGER li;
    if (!GetFileSizeEx(hFile, &li)) {
        CloseHandle(hFile);
        return false;
    }
    out->file_size = (size_t)li.QuadPart;

    HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMapping) {
        CloseHandle(hFile);
        return false;
    }

    uint8_t* ptr = (uint8_t*)MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!ptr) {
        CloseHandle(hMapping);
        CloseHandle(hFile);
        return false;
    }

    out->file_handle    = hFile;
    out->mapping_handle = hMapping;
    out->data           = ptr;

    gguf_reader_t r = {out->data, out->file_size, 0};

    uint32_t magic = read_u32(&r);
    if (magic != GGUF_MAGIC) {
        fprintf(stderr, "GGUF: Invalid magic: 0x%08X (expected 0x%08X)\n", magic, GGUF_MAGIC);
        gguf_parser_close(out);
        return false;
    }

    out->version           = read_u32(&r);
    out->tensor_count      = read_u64(&r);
    out->metadata_kv_count = read_u64(&r);

    if (out->version < 2 || out->version > 3) {
        fprintf(stderr, "GGUF: Unsupported version %u (expected 2 or 3)\n", out->version);
        gguf_parser_close(out);
        return false;
    }

    memset(&out->config, 0, sizeof(out->config));
    out->config.rope_theta = 10000.0f;
    out->config.norm_eps   = 1e-6f;

    for (uint64_t i = 0; i < out->metadata_kv_count; i++) {
        const char* key;
        uint64_t key_len = read_string(&r, &key);
        uint32_t val_type = read_u32(&r);

        bool consumed = false;

        if (key_len >= 22 && memcmp(key, "general.architecture", 21) == 0 && key[21] == '\0') {
        } else if (KEY_MATCHES(key, key_len, "general.architecture")) {
            const char* arch_val;
            uint64_t arch_len = read_string(&r, &arch_val);
            size_t copy_len = arch_len < sizeof(out->config.architecture) - 1
                            ? (size_t)arch_len : sizeof(out->config.architecture) - 1;
            memcpy(out->config.architecture, arch_val, copy_len);
            out->config.architecture[copy_len] = '\0';
            consumed = true;
        }

        if (!consumed && KEY_MATCHES(key, key_len, "llama.block_count")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_layers = v;
            consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "llama.embedding_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.d = v;
            consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "llama.feed_forward_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.ffn_dim = v;
            consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "llama.attention.head_count")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_heads = v;
            consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "llama.attention.head_count_kv")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_kv_heads = v;
            consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "llama.context_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.context_length = v;
            consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "llama.rope.theta")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            if (val_type == GGUF_TYPE_FLOAT32) out->config.rope_theta = f;
            else if (val_type == GGUF_TYPE_FLOAT64) out->config.rope_theta = f;
            else out->config.rope_theta = (float)v;
            consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "llama.attention.layer_norm_rms_epsilon")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            if (val_type == GGUF_TYPE_FLOAT32) out->config.norm_eps = f;
            else if (val_type == GGUF_TYPE_FLOAT64) out->config.norm_eps = f;
            consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "llama.vocab_size")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.vocab_size = v;
            consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "gemma.block_count")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_layers = v;
            consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "gemma.embedding_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.d = v;
            consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "gemma.attention.head_count")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_heads = v;
            consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "gemma.attention.head_count_kv")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_kv_heads = v;
            consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "gemma.context_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.context_length = v;
            consumed = true;
        }

        // ─── Qwen2 ─────────────────────────────────────────────────────────
        if (!consumed && KEY_MATCHES(key, key_len, "qwen2.block_count")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_layers = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "qwen2.embedding_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.d = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "qwen2.feed_forward_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.ffn_dim = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "qwen2.attention.head_count")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_heads = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "qwen2.attention.head_count_kv")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_kv_heads = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "qwen2.context_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.context_length = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "qwen2.rope.freq_base")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            if (val_type == GGUF_TYPE_FLOAT32) out->config.rope_theta = f;
            else out->config.rope_theta = (float)v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "qwen2.attention.layer_norm_rms_epsilon")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            if (val_type == GGUF_TYPE_FLOAT32) out->config.norm_eps = f;
            else out->config.norm_eps = (float)v; consumed = true;
        }

        // ─── Qwen3 ─────────────────────────────────────────────────────────
        if (!consumed && KEY_MATCHES(key, key_len, "qwen3.block_count")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_layers = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "qwen3.embedding_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.d = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "qwen3.feed_forward_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.ffn_dim = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "qwen3.attention.head_count")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_heads = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "qwen3.attention.head_count_kv")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_kv_heads = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "qwen3.context_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.context_length = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "qwen3.rope.freq_base")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            if (val_type == GGUF_TYPE_FLOAT32) out->config.rope_theta = f;
            else out->config.rope_theta = (float)v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "qwen3.attention.layer_norm_rms_epsilon")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            if (val_type == GGUF_TYPE_FLOAT32) out->config.norm_eps = f;
            else out->config.norm_eps = (float)v; consumed = true;
        }

        // ─── DSpark (Qwen3-based custom architecture) ─────────────────────
        if (!consumed && KEY_MATCHES(key, key_len, "dspark.block_count")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_layers = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "dspark.embedding_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.d = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "dspark.feed_forward_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.ffn_dim = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "dspark.attention.head_count")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_heads = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "dspark.attention.head_count_kv")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_kv_heads = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "dspark.context_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.context_length = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "dspark.rope.freq_base")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            if (val_type == GGUF_TYPE_FLOAT32) out->config.rope_theta = f;
            else out->config.rope_theta = (float)v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "dspark.attention.layer_norm_rms_epsilon")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            if (val_type == GGUF_TYPE_FLOAT32) out->config.norm_eps = f;
            else out->config.norm_eps = (float)v; consumed = true;
        }

        // ─── Mistral ───────────────────────────────────────────────────────
        if (!consumed && KEY_MATCHES(key, key_len, "mistral.block_count")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_layers = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "mistral.embedding_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.d = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "mistral.feed_forward_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.ffn_dim = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "mistral.attention.head_count")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_heads = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "mistral.attention.head_count_kv")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_kv_heads = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "mistral.context_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.context_length = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "mistral.rope.freq_base")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            if (val_type == GGUF_TYPE_FLOAT32) out->config.rope_theta = f;
            else out->config.rope_theta = (float)v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "mistral.attention.layer_norm_rms_epsilon")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            if (val_type == GGUF_TYPE_FLOAT32) out->config.norm_eps = f;
            else out->config.norm_eps = (float)v; consumed = true;
        }

        // ─── Phi3 ──────────────────────────────────────────────────────────
        if (!consumed && KEY_MATCHES(key, key_len, "phi3.block_count")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_layers = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "phi3.embedding_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.d = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "phi3.feed_forward_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.ffn_dim = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "phi3.attention.head_count")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_heads = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "phi3.attention.head_count_kv")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_kv_heads = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "phi3.context_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.context_length = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "phi3.rope.freq_base")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            if (val_type == GGUF_TYPE_FLOAT32) out->config.rope_theta = f;
            else out->config.rope_theta = (float)v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "phi3.attention.layer_norm_rms_epsilon")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            if (val_type == GGUF_TYPE_FLOAT32) out->config.norm_eps = f;
            else out->config.norm_eps = (float)v; consumed = true;
        }

        // ─── DeepSeek V2/V3 ────────────────────────────────────────────────
        if (!consumed && KEY_MATCHES(key, key_len, "deepseek2.block_count")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_layers = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "deepseek2.embedding_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.d = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "deepseek2.feed_forward_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.ffn_dim = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "deepseek2.attention.head_count")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_heads = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "deepseek2.attention.head_count_kv")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_kv_heads = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "deepseek2.context_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.context_length = v; consumed = true;
        }

        // ─── Command-R ─────────────────────────────────────────────────────
        if (!consumed && KEY_MATCHES(key, key_len, "command-r.block_count")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_layers = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "command-r.embedding_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.d = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "command-r.feed_forward_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.ffn_dim = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "command-r.attention.head_count")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_heads = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "command-r.attention.head_count_kv")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_kv_heads = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "command-r.context_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.context_length = v; consumed = true;
        }

        // ─── Laguna ────────────────────────────────────────────────────────
        if (!consumed && KEY_MATCHES(key, key_len, "laguna.block_count")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_layers = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "laguna.embedding_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.d = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "laguna.feed_forward_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.ffn_dim = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "laguna.attention.head_count")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_heads = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "laguna.attention.head_count_kv")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_kv_heads = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "laguna.context_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.context_length = v; consumed = true;
        }

        // ─── Gemma 4 ───────────────────────────────────────────────────────
        if (!consumed && KEY_MATCHES(key, key_len, "gemma4.block_count")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_layers = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "gemma4.embedding_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.d = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "gemma4.feed_forward_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.ffn_dim = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "gemma4.attention.head_count")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_heads = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "gemma4.attention.head_count_kv")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_kv_heads = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "gemma4.context_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.context_length = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "gemma4.rope.freq_base")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            if (val_type == GGUF_TYPE_FLOAT32) out->config.rope_theta = f;
            else out->config.rope_theta = (float)v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "gemma4.attention.layer_norm_rms_epsilon")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            if (val_type == GGUF_TYPE_FLOAT32) out->config.norm_eps = f;
            else out->config.norm_eps = (float)v; consumed = true;
        }

        // ─── Qwen 3.5 ─────────────────────────────────────────────────────
        if (!consumed && KEY_MATCHES(key, key_len, "qwen35.block_count")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_layers = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "qwen35.embedding_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.d = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "qwen35.feed_forward_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.ffn_dim = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "qwen35.attention.head_count")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_heads = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "qwen35.attention.head_count_kv")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.n_kv_heads = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "qwen35.context_length")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            out->config.context_length = v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "qwen35.rope.freq_base")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            if (val_type == GGUF_TYPE_FLOAT32) out->config.rope_theta = f;
            else out->config.rope_theta = (float)v; consumed = true;
        }
        if (!consumed && KEY_MATCHES(key, key_len, "qwen35.attention.layer_norm_rms_epsilon")) {
            uint32_t v; float f; uint64_t u64;
            read_metadata_value(&r, val_type, &v, &f, &u64);
            if (val_type == GGUF_TYPE_FLOAT32) out->config.norm_eps = f;
            else out->config.norm_eps = (float)v; consumed = true;
        }

        // ─── Generic: <arch>.attention.key_length / value_length ───────────
        // Qwen/Gemma store the true head_dim here (may differ from d/n_heads).
        {
            static const char kl_suffix[] = ".attention.key_length";
            static const char vl_suffix[] = ".attention.value_length";
            size_t kl_len = sizeof(kl_suffix) - 1;
            size_t vl_len = sizeof(vl_suffix) - 1;
            bool is_kl = (key_len >= kl_len &&
                          memcmp(key + key_len - kl_len, kl_suffix, kl_len) == 0);
            bool is_vl = (key_len >= vl_len &&
                          memcmp(key + key_len - vl_len, vl_suffix, vl_len) == 0);
            if (!consumed && (is_kl || is_vl)) {
                uint32_t v; float f; uint64_t u64;
                read_metadata_value(&r, val_type, &v, &f, &u64);
                out->config.head_dim = v;
                consumed = true;
            }
        }

        if (!consumed) {
            skip_value(&r, val_type);
        }
    }

    if (out->config.n_kv_heads == 0) {
        out->config.n_kv_heads = out->config.n_heads;
    }
    if (out->config.head_dim == 0 && out->config.n_heads > 0 && out->config.d > 0) {
        out->config.head_dim = out->config.d / out->config.n_heads;
    }

    if (out->tensor_count > 0) {
        out->tensors = (gguf_tensor_info_t*)calloc((size_t)out->tensor_count, sizeof(gguf_tensor_info_t));
        if (!out->tensors) {
            gguf_parser_close(out);
            return false;
        }

        for (uint64_t i = 0; i < out->tensor_count; i++) {
            gguf_tensor_info_t* t = &out->tensors[i];

            const char* name_ptr;
            uint64_t name_len = read_string(&r, &name_ptr);
            t->name = (char*)malloc((size_t)(name_len + 1));
            memcpy(t->name, name_ptr, (size_t)name_len);
            t->name[name_len] = '\0';

            t->n_dims = read_u32(&r);
            t->dims   = (uint64_t*)malloc((size_t)t->n_dims * sizeof(uint64_t));

            uint64_t total_elements = 1;
            for (uint32_t d = 0; d < t->n_dims; d++) {
                t->dims[d] = read_u64(&r);
                total_elements *= t->dims[d];
            }

            uint32_t raw_type = read_u32(&r);
            t->type = (gguf_tensor_type_t)raw_type;
            t->offset = read_u64(&r);

            float quant_ratio = 1.0f;
            switch (t->type) {
                case GGUF_TENSOR_F32:     quant_ratio = 4.0f; break;
                case GGUF_TENSOR_F16:     quant_ratio = 2.0f; break;
                case GGUF_TENSOR_BF16:    quant_ratio = 2.0f; break;
                case GGUF_TENSOR_Q4_0:    quant_ratio = 18.0f / 32.0f;    break; // 0.5625
                case GGUF_TENSOR_Q4_1:    quant_ratio = 20.0f / 32.0f;    break; // 0.625
                case GGUF_TENSOR_Q5_0:    quant_ratio = 22.0f / 32.0f;    break; // 0.6875
                case GGUF_TENSOR_Q5_1:    quant_ratio = 24.0f / 32.0f;    break; // 0.75
                case GGUF_TENSOR_Q8_0:    quant_ratio = 34.0f / 32.0f;    break; // 1.0625
                case GGUF_TENSOR_Q8_1:    quant_ratio = 40.0f / 32.0f;    break; // 1.25
                case GGUF_TENSOR_Q2_K:    quant_ratio = 84.0f / 256.0f;   break; // 0.328125
                case GGUF_TENSOR_Q3_K:    quant_ratio = 110.0f / 256.0f;  break; // 0.4296875
                case GGUF_TENSOR_Q4_K:    quant_ratio = 144.0f / 256.0f;  break; // 0.5625
                case GGUF_TENSOR_Q5_K:    quant_ratio = 176.0f / 256.0f;  break; // 0.6875
                case GGUF_TENSOR_Q6_K:    quant_ratio = 210.0f / 256.0f;  break; // 0.8203125
                case GGUF_TENSOR_Q8_K:    quant_ratio = 290.0f / 256.0f;  break; // 1.1328125
                case GGUF_TENSOR_IQ4_NL:  quant_ratio = 18.0f / 32.0f;    break; // 0.5625
                case GGUF_TENSOR_IQ4_XS:  quant_ratio = 135.0f / 256.0f;  break; // 0.52734375
                case GGUF_TENSOR_NVFP4:   quant_ratio = 0.5f;              break;
                case GGUF_TENSOR_Q1_0:    quant_ratio = 6.0f / 32.0f;      break; // 0.1875
                case GGUF_TENSOR_Q2_0:    quant_ratio = 10.0f / 32.0f;     break; // 0.3125
                default:
                    fprintf(stderr, "GGUF: Warning: unknown tensor type %u for '%s'\n",
                            raw_type, t->name);
                    quant_ratio = 4.0f;
                    break;
            }
            t->size_bytes = (uint64_t)((double)total_elements * (double)quant_ratio);
        }
    }

    // GGUF tensor offsets are relative to the DATA section, which starts right
    // after the tensor-info table. Per the GGUF spec, the data section is padded
    // to GGUF_DEFAULT_ALIGNMENT (32) by the writer, so tensor offsets only resolve
    // correctly against the ALIGNED position. (Reading at the raw r.pos made every
    // weight 11-24 bytes early -> garbage dequant / NaN on quantized models.)
    uint64_t aligned_data_start = (r.pos + GGUF_DEFAULT_ALIGNMENT - 1) &
                                  ~((uint64_t)GGUF_DEFAULT_ALIGNMENT - 1);
    out->data_start = aligned_data_start;

    if (out->data_start > out->file_size) {
        fprintf(stderr, "GGUF: data section start %llu beyond file size %llu\n",
                (unsigned long long)out->data_start, (unsigned long long)out->file_size);
        gguf_parser_close(out);
        return false;
    }

    if (out->config.vocab_size == 0) {
        int32_t emb_idx = gguf_find_tensor(out, "token_embd.weight");
        if (emb_idx >= 0 && out->tensors[emb_idx].n_dims >= 2) {
            // GGUF may store token_embd as [vocab, d] (llama) or [d, vocab] (qwen).
            // vocab is whichever dim is NOT the embedding size.
            uint64_t d0 = out->tensors[emb_idx].dims[0];
            uint64_t d1 = out->tensors[emb_idx].dims[1];
            uint64_t dd = out->config.d;
            out->config.vocab_size = (d0 != dd) ? (uint32_t)d0 : (uint32_t)d1;
        }
    }

    return true;
}

void gguf_parser_close(gguf_file_t* f) {
    if (!f) return;

    if (f->tensors) {
        for (uint64_t i = 0; i < f->tensor_count; i++) {
            free(f->tensors[i].name);
            f->tensors[i].name = nullptr;
            free(f->tensors[i].dims);
            f->tensors[i].dims = nullptr;
        }
        free(f->tensors);
        f->tensors = nullptr;
    }

    if (f->data && f->mapping_handle) {
        UnmapViewOfFile(f->data);
        f->data = nullptr;
    }

    if (f->mapping_handle) {
        CloseHandle((HANDLE)f->mapping_handle);
        f->mapping_handle = nullptr;
    }

    if (f->file_handle) {
        CloseHandle((HANDLE)f->file_handle);
        f->file_handle = nullptr;
    }

    memset(f, 0, sizeof(*f));
}

const uint8_t* gguf_get_tensor_data(const gguf_file_t* f, uint32_t tensor_idx, uint64_t* out_size) {
    if (!f || tensor_idx >= f->tensor_count) return nullptr;
    *out_size = f->tensors[tensor_idx].size_bytes;
    return f->data + f->data_start + f->tensors[tensor_idx].offset;
}

int32_t gguf_find_tensor(const gguf_file_t* f, const char* name) {
    if (!f || !f->tensors) return -1;
    for (uint64_t i = 0; i < f->tensor_count; i++) {
        if (strcmp(f->tensors[i].name, name) == 0) {
            return (int32_t)i;
        }
    }
    return -1;
}

uint32_t gguf_tensor_to_vk_quant(gguf_tensor_type_t tt) {
    switch (tt) {
        case GGUF_TENSOR_F16:     return QUANT_FP16;   // 0
        case GGUF_TENSOR_F32:     return QUANT_FP32;
        case GGUF_TENSOR_BF16:    return QUANT_BF16;
        case GGUF_TENSOR_Q4_K:    return QUANT_Q4_K;   // 1
        case GGUF_TENSOR_IQ4_XS:  return QUANT_IQ4_XS; // 2
        case GGUF_TENSOR_IQ4_NL:  return QUANT_IQ4_XS; // 2 (same family, treated as IQ4_XS layout is wrong; map to FP16 fallback instead)
        case GGUF_TENSOR_Q6_K:    return QUANT_Q6_K;   // 3
        case GGUF_TENSOR_Q8_0:    return QUANT_Q8_0;   // 4
        case GGUF_TENSOR_Q4_0:    return QUANT_Q4_0;   // 5
        case GGUF_TENSOR_Q5_K:    return QUANT_Q5_K;   // 6
        case GGUF_TENSOR_NVFP4:   return QUANT_NVFP4;
        default:
            fprintf(stderr, "GGUF: Unsupported quant type %u, falling back to FP16\n", (uint32_t)tt);
            return QUANT_FP16;
    }
}

#undef KEY_MATCHES
