#include "vk_kv_cache.h"
#include "vk_device.h"

extern PFN_vkVoidFunction vk_load_device_fn(VkDevice dev, const char* name);

bool vk_kv_cache_create(vk_device_t* dev, vk_kv_cache_t* cache,
                        uint32_t n_layers, uint32_t n_kv_heads, uint32_t head_dim,
                        uint32_t max_seq_len) {
    memset(cache, 0, sizeof(*cache));

    cache->page_size_tokens = KV_PAGE_TOKENS;
    cache->n_layers         = n_layers;
    cache->n_kv_heads       = n_kv_heads;
    cache->head_dim         = head_dim;
    cache->pages_per_layer  = (max_seq_len + KV_PAGE_TOKENS - 1) / KV_PAGE_TOKENS;

    uint64_t k_half_per_page = (uint64_t)n_kv_heads * head_dim * KV_PAGE_TOKENS * 2;
    uint64_t v_half_per_page = k_half_per_page;
    cache->page_stride_bytes = k_half_per_page + v_half_per_page;

    cache->total_pages = n_layers * cache->pages_per_layer;
    cache->total_kv_bytes = (uint64_t)cache->total_pages * cache->page_stride_bytes;

    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size  = cache->total_kv_bytes;
    bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci = {};
    aci.usage         = VMA_MEMORY_USAGE_AUTO;
    aci.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkResult result = vmaCreateBuffer(dev->allocator, &bci, &aci,
                                       &cache->buffer.buffer,
                                       &cache->buffer.allocation,
                                       &cache->buffer.alloc_info);
    if (result != VK_SUCCESS) {
        RDNA4_ERROR("Failed to allocate KV cache buffer (%llu MB)",
                    (unsigned long long)(cache->total_kv_bytes / (1024 * 1024)));
        return false;
    }
    cache->buffer.size            = cache->total_kv_bytes;
    cache->buffer.device_address  = 0;
    cache->buffer.mapped_ptr      = nullptr;
    cache->buffer.is_host_visible = false;
    cache->buffer.is_host_coherent = false;
    cache->buffer.device_address  = 0;  // BDA not needed for now

    uint64_t page_table_bytes = (uint64_t)cache->total_pages * sizeof(uint32_t);

    VkBufferCreateInfo pt_bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    pt_bci.size  = page_table_bytes;
    pt_bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    pt_bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo pt_aci = {};
    pt_aci.usage         = VMA_MEMORY_USAGE_AUTO;
    pt_aci.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    result = vmaCreateBuffer(dev->allocator, &pt_bci, &pt_aci,
                              &cache->page_table.buffer,
                              &cache->page_table.allocation,
                              &cache->page_table.alloc_info);
    if (result != VK_SUCCESS) {
        RDNA4_ERROR("Failed to allocate KV page table buffer");
        vmaDestroyBuffer(dev->allocator, cache->buffer.buffer, cache->buffer.allocation);
        return false;
    }
    cache->page_table.size            = page_table_bytes;
    cache->page_table.device_address  = 0;
    cache->page_table.mapped_ptr      = nullptr;
    cache->page_table.is_host_visible = true;
    cache->page_table.is_host_coherent = true;
    cache->page_table.device_address  = 0;

    // Identity page table: logical page i -> physical page i.
    uint32_t* pt = nullptr;
    if (vmaMapMemory(dev->allocator, cache->page_table.allocation, (void**)&pt) == VK_SUCCESS) {
        for (uint64_t i = 0; i < cache->total_pages; i++) pt[i] = (uint32_t)i;
        vmaUnmapMemory(dev->allocator, cache->page_table.allocation);
        cache->page_table.mapped_ptr = pt;
    }
    return true;
}

void vk_kv_cache_destroy(vk_device_t* dev, vk_kv_cache_t* cache) {
    if (cache->page_table.buffer) {
        vmaDestroyBuffer(dev->allocator, cache->page_table.buffer, cache->page_table.allocation);
    }
    if (cache->buffer.buffer) {
        vmaDestroyBuffer(dev->allocator, cache->buffer.buffer, cache->buffer.allocation);
    }
    memset(cache, 0, sizeof(*cache));
}

vk_kv_cache_page_addr_t vk_kv_cache_addr(const vk_kv_cache_t* cache,
                                          uint32_t layer, uint32_t page, uint32_t token_in_page) {
    vk_kv_cache_page_addr_t addr;
    addr.page_size_kv_bytes = cache->page_stride_bytes;

    uint64_t k_half_per_page = (uint64_t)cache->n_kv_heads * cache->head_dim * cache->page_size_tokens * 2;

    uint64_t layer_base = (uint64_t)layer * cache->pages_per_layer * cache->page_stride_bytes;
    uint64_t page_base  = (uint64_t)page * cache->page_stride_bytes;
    uint64_t token_base = (uint64_t)token_in_page * cache->n_kv_heads * cache->head_dim * 2;

    addr.k_addr = layer_base + page_base + token_base;
    addr.v_addr = layer_base + page_base + k_half_per_page + token_base;
    return addr;
}
