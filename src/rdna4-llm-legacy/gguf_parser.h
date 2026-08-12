#pragma once

#include <cstdint>
#include <cstddef>

#define GGUF_MAGIC          0x46554747
#define GGUF_VERSION        3
#define GGUF_DEFAULT_ALIGNMENT 32

enum gguf_type_t : uint32_t {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
};

enum gguf_tensor_type_t : uint32_t {
    GGUF_TENSOR_F32     = 0,
    GGUF_TENSOR_F16     = 1,
    GGUF_TENSOR_Q4_0    = 2,
    GGUF_TENSOR_Q4_1    = 3,
    GGUF_TENSOR_Q5_0    = 6,
    GGUF_TENSOR_Q5_1    = 7,
    GGUF_TENSOR_Q8_0    = 8,
    GGUF_TENSOR_Q8_1    = 9,
    GGUF_TENSOR_Q2_K    = 10,
    GGUF_TENSOR_Q3_K    = 11,
    GGUF_TENSOR_Q4_K    = 12,
    GGUF_TENSOR_Q5_K    = 13,
    GGUF_TENSOR_Q6_K    = 14,
    GGUF_TENSOR_Q8_K    = 15,
    GGUF_TENSOR_IQ2_XXS = 16,
    GGUF_TENSOR_IQ2_XS  = 17,
    GGUF_TENSOR_IQ3_XXS = 18,
    GGUF_TENSOR_IQ1_S   = 19,
    GGUF_TENSOR_IQ4_NL  = 20,
    GGUF_TENSOR_IQ3_S   = 21,
    GGUF_TENSOR_IQ2_S   = 22,
    GGUF_TENSOR_IQ4_XS  = 23,
    GGUF_TENSOR_I8      = 24,
    GGUF_TENSOR_I16     = 25,
    GGUF_TENSOR_I32     = 26,
    GGUF_TENSOR_I64     = 27,
    GGUF_TENSOR_F64     = 28,
    GGUF_TENSOR_IQ1_M   = 29,
    GGUF_TENSOR_BF16    = 30,
    GGUF_TENSOR_TQ1_0   = 34,
    GGUF_TENSOR_TQ2_0   = 35,
    GGUF_TENSOR_MXFP4   = 39,
    GGUF_TENSOR_NVFP4   = 40,
    GGUF_TENSOR_Q1_0    = 41,
    GGUF_TENSOR_Q2_0    = 42,
};

struct gguf_tensor_info_t {
    char*            name;
    uint32_t         n_dims;
    uint64_t*        dims;
    gguf_tensor_type_t type;
    uint64_t         offset;
    uint64_t         size_bytes;
};

struct gguf_model_config_t {
    char     architecture[64];
    uint32_t d;
    uint32_t ffn_dim;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t vocab_size;
    uint32_t n_layers;
    float    rope_theta;
    float    norm_eps;
    uint32_t context_length;
};

struct gguf_file_t {
    void*              file_handle;
    void*              mapping_handle;
    uint8_t*           data;
    size_t             file_size;
    uint32_t           version;
    uint64_t           tensor_count;
    uint64_t           metadata_kv_count;
    uint64_t           data_start;
    gguf_tensor_info_t* tensors;
    gguf_model_config_t config;
};

bool             gguf_parser_open(const char* path, gguf_file_t* out);
void             gguf_parser_close(gguf_file_t* f);
const uint8_t*   gguf_get_tensor_data(const gguf_file_t* f, uint32_t tensor_idx, uint64_t* out_size);
int32_t          gguf_find_tensor(const gguf_file_t* f, const char* name);
uint32_t         gguf_tensor_to_vk_quant(gguf_tensor_type_t tt);
