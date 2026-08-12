#pragma once

#include "vk_device.h"
#include "gguf_parser.h"

struct vk_descriptor_arena_t {
    VkDescriptorPool        pool;
    VkDescriptorSet         set0_weight_set;
    VkDescriptorSet         set2_table_set;
    VkDescriptorSetLayout   set0_layout;
    VkDescriptorSetLayout   set1_layout;
    VkDescriptorSetLayout   set2_layout;
    vk_buffer_t             rope_freqs_buf;
    VkBuffer                dummy_buf;   // 4-byte scratch buffer for unused set1 slots
    VmaAllocation           dummy_alloc;
};

bool vk_descriptor_arena_create(vk_device_t* dev, vk_descriptor_arena_t* arena,
                                 uint32_t n_layers, uint32_t head_dim,
                                 uint32_t max_seq_len, float rope_theta);
void vk_descriptor_arena_destroy(vk_device_t* dev, vk_descriptor_arena_t* arena);
void vk_descriptor_arena_update_weights(vk_device_t* dev, vk_descriptor_arena_t* arena,
                                         vk_buffer_t* weight_bufs, uint32_t n_layers,
                                         vk_buffer_t* lm_head_buf);
void vk_descriptor_arena_push_set1(vk_descriptor_arena_t* arena, VkDevice device, VkCommandBuffer cb, VkPipelineLayout layout,
                                    vk_buffer_t* hidden_in, vk_buffer_t* hidden_out,
                                    vk_buffer_t* k_cache, vk_buffer_t* v_cache,
                                    vk_buffer_t* scratch);
