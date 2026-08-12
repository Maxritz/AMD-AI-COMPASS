#pragma once

#include "common.h"
#include "vk_device.h"

struct vk_kv_cache_t {
    vk_buffer_t buffer;
    vk_buffer_t page_table;
    uint32_t    page_size_tokens;
    uint32_t    pages_per_layer;
    uint32_t    total_pages;
    uint32_t    n_layers;
    uint32_t    n_kv_heads;
    uint32_t    head_dim;
    uint64_t    page_stride_bytes;
    uint64_t    total_kv_bytes;
};

struct vk_kv_cache_page_addr_t {
    uint64_t k_addr;
    uint64_t v_addr;
    uint64_t page_size_kv_bytes;
};

bool vk_kv_cache_create(vk_device_t* dev, vk_kv_cache_t* cache,
                        uint32_t n_layers, uint32_t n_kv_heads, uint32_t head_dim,
                        uint32_t max_seq_len);
void vk_kv_cache_destroy(vk_device_t* dev, vk_kv_cache_t* cache);
vk_kv_cache_page_addr_t vk_kv_cache_addr(const vk_kv_cache_t* cache,
                                          uint32_t layer, uint32_t page, uint32_t token_in_page);
