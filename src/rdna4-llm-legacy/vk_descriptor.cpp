#include "vk_descriptor.h"
#include "vk_device.h"

bool vk_descriptor_arena_create(vk_device_t* dev, vk_descriptor_arena_t* arena,
                                 uint32_t n_layers, uint32_t head_dim,
                                 uint32_t max_seq_len, float rope_theta) {
    memset(arena, 0, sizeof(*arena));

    VkDescriptorPoolSize pool_sizes[3];
    pool_sizes[0] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, (n_layers + 1) * 2};
    pool_sizes[1] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32};
    pool_sizes[2] = {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8};

    VkDescriptorPoolCreateInfo dpci = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpci.flags         = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT;
    dpci.maxSets       = 4;
    dpci.poolSizeCount = 3;
    dpci.pPoolSizes    = pool_sizes;

    VK_CHECK(vkCreateDescriptorPool(dev->device, &dpci, nullptr, &arena->pool));

    VkDescriptorBindingFlags binding_flags =
        VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT |
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT |
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT;

    VkDescriptorSetLayoutBindingFlagsCreateInfo flags_ci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO
    };
    flags_ci.bindingCount  = 1;
    flags_ci.pBindingFlags = &binding_flags;

    VkDescriptorSetLayoutBinding set0_binding = {};
    set0_binding.binding         = 0;
    set0_binding.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    set0_binding.descriptorCount = n_layers + 1;  // +1 for lm_head weight
    set0_binding.stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo set0_ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set0_ci.pNext        = &flags_ci;
    set0_ci.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT;
    set0_ci.bindingCount = 1;
    set0_ci.pBindings    = &set0_binding;

    VK_CHECK(vkCreateDescriptorSetLayout(dev->device, &set0_ci, nullptr, &arena->set0_layout));

    VkDescriptorSetLayoutBinding set1_bindings[5];
    memset(set1_bindings, 0, sizeof(set1_bindings));
    for (int i = 0; i < 5; i++) {
        set1_bindings[i].binding         = i;
        set1_bindings[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        set1_bindings[i].descriptorCount = 1;
        set1_bindings[i].stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo set1_ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set1_ci.flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    set1_ci.bindingCount = 5;
    set1_ci.pBindings    = set1_bindings;

    VK_CHECK(vkCreateDescriptorSetLayout(dev->device, &set1_ci, nullptr, &arena->set1_layout));

    VkDescriptorSetLayoutBinding set2_bindings[2];
    set2_bindings[0] = {0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
    set2_bindings[1] = {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr};

    VkDescriptorSetLayoutCreateInfo set2_ci = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    set2_ci.bindingCount = 2;
    set2_ci.pBindings    = set2_bindings;

    VK_CHECK(vkCreateDescriptorSetLayout(dev->device, &set2_ci, nullptr, &arena->set2_layout));

    VkDescriptorSetVariableDescriptorCountAllocateInfo var_ci = {
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_VARIABLE_DESCRIPTOR_COUNT_ALLOCATE_INFO
    };
    uint32_t var_count = n_layers + 1;  // +1 for lm_head
    var_ci.descriptorSetCount = 1;
    var_ci.pDescriptorCounts  = &var_count;

    VkDescriptorSetAllocateInfo set0_ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set0_ai.pNext              = &var_ci;
    set0_ai.descriptorPool     = arena->pool;
    set0_ai.descriptorSetCount = 1;
    set0_ai.pSetLayouts        = &arena->set0_layout;

    VK_CHECK(vkAllocateDescriptorSets(dev->device, &set0_ai, &arena->set0_weight_set));

    VkDescriptorSetAllocateInfo set2_ai = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    set2_ai.descriptorPool     = arena->pool;
    set2_ai.descriptorSetCount = 1;
    set2_ai.pSetLayouts        = &arena->set2_layout;

    VK_CHECK(vkAllocateDescriptorSets(dev->device, &set2_ai, &arena->set2_table_set));

    uint32_t rope_count = max_seq_len * (head_dim / 2) * 2;

    VkBufferCreateInfo rope_bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    rope_bci.size  = (VkDeviceSize)rope_count * sizeof(float);
    rope_bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    rope_bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo rope_aci = {};
    rope_aci.usage = VMA_MEMORY_USAGE_AUTO;
    rope_aci.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    VkResult result = vmaCreateBuffer(dev->allocator, &rope_bci, &rope_aci,
                                       &arena->rope_freqs_buf.buffer,
                                       &arena->rope_freqs_buf.allocation,
                                       &arena->rope_freqs_buf.alloc_info);
    if (result != VK_SUCCESS) {
        RDNA4_ERROR("Failed to create rope freq buffer");
        return false;
    }
    arena->rope_freqs_buf.size = rope_bci.size;
    arena->rope_freqs_buf.device_address = 0;
    arena->rope_freqs_buf.mapped_ptr = nullptr;
    arena->rope_freqs_buf.is_host_visible = false;
    arena->rope_freqs_buf.is_host_coherent = false;

    // Dummy 4-byte buffer used to fill unused set1 slots (null descriptors are
    // illegal unless nullDescriptor feature is enabled).
    VkBufferCreateInfo dummy_bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    dummy_bci.size  = 4;
    dummy_bci.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    dummy_bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VmaAllocationCreateInfo dummy_aci = {};
    dummy_aci.usage = VMA_MEMORY_USAGE_AUTO;
    dummy_aci.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    result = vmaCreateBuffer(dev->allocator, &dummy_bci, &dummy_aci,
                              &arena->dummy_buf, &arena->dummy_alloc, nullptr);
    if (result != VK_SUCCESS) {
        RDNA4_ERROR("Failed to create dummy buffer");
        return false;
    }

    std::vector<float> rope_cpu(rope_count);
    for (uint32_t pos = 0; pos < max_seq_len; pos++) {
        float* cos_ptr = rope_cpu.data() + pos * (head_dim / 2) * 2;
        float* sin_ptr = cos_ptr + (head_dim / 2);
        for (uint32_t i = 0; i < head_dim / 2; i++) {
            float theta = 1.0f / powf(rope_theta, (float)(2 * i) / (float)head_dim);
            float freq  = (float)pos * theta;
            cos_ptr[i]  = cosf(freq);
            sin_ptr[i]  = sinf(freq);
        }
    }

    VkBufferCreateInfo staging_bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    staging_bci.size  = rope_bci.size;
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
    result = vmaCreateBuffer(dev->allocator, &staging_bci, &staging_aci,
                              &staging_buf, &staging_alloc, &staging_ai);
    if (result != VK_SUCCESS) {
        RDNA4_ERROR("Failed to create rope staging buffer");
        return false;
    }
    memcpy(staging_ai.pMappedData, rope_cpu.data(), (size_t)rope_bci.size);

    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool        = dev->transfer_cmd_pool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;

    VkCommandBuffer cb;
    vkAllocateCommandBuffers(dev->device, &cai, &cb);

    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);

    VkBufferCopy region = {0, 0, rope_bci.size};
    vkCmdCopyBuffer(cb, staging_buf, arena->rope_freqs_buf.buffer, 1, &region);

    vkEndCommandBuffer(cb);

    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    vkCreateFence(dev->device, &fci, nullptr, &fence);

    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    vkQueueSubmit(dev->transfer_queue, 1, &si, fence);
    vkWaitForFences(dev->device, 1, &fence, VK_TRUE, UINT64_MAX);

    vkDestroyFence(dev->device, fence, nullptr);
    vkFreeCommandBuffers(dev->device, dev->transfer_cmd_pool, 1, &cb);
    vmaDestroyBuffer(dev->allocator, staging_buf, staging_alloc);

    VkDescriptorBufferInfo rope_buf_info = {};
    rope_buf_info.buffer = arena->rope_freqs_buf.buffer;
    rope_buf_info.offset = 0;
    rope_buf_info.range  = VK_WHOLE_SIZE;

    VkWriteDescriptorSet rope_write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    rope_write.dstSet          = arena->set2_table_set;
    rope_write.dstBinding      = 0;
    rope_write.dstArrayElement = 0;
    rope_write.descriptorCount = 1;
    rope_write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    rope_write.pBufferInfo     = &rope_buf_info;

    vkUpdateDescriptorSets(dev->device, 1, &rope_write, 0, nullptr);

    return true;
}

void vk_descriptor_arena_destroy(vk_device_t* dev, vk_descriptor_arena_t* arena) {
    if (arena->dummy_buf) vmaDestroyBuffer(dev->allocator, arena->dummy_buf, arena->dummy_alloc);
    if (arena->rope_freqs_buf.buffer) {
        vmaDestroyBuffer(dev->allocator, arena->rope_freqs_buf.buffer, arena->rope_freqs_buf.allocation);
    }
    if (arena->set2_layout) vkDestroyDescriptorSetLayout(dev->device, arena->set2_layout, nullptr);
    if (arena->set1_layout) vkDestroyDescriptorSetLayout(dev->device, arena->set1_layout, nullptr);
    if (arena->set0_layout) vkDestroyDescriptorSetLayout(dev->device, arena->set0_layout, nullptr);
    if (arena->pool)        vkDestroyDescriptorPool(dev->device, arena->pool, nullptr);
    memset(arena, 0, sizeof(*arena));
}

void vk_descriptor_arena_update_weights(vk_device_t* dev, vk_descriptor_arena_t* arena,
                                         vk_buffer_t* weight_bufs, uint32_t n_layers,
                                         vk_buffer_t* lm_head_buf) {
    std::vector<VkDescriptorBufferInfo> buf_infos(n_layers + 1);
    for (uint32_t i = 0; i < n_layers; i++) {
        buf_infos[i].buffer = weight_bufs[i].buffer;
        buf_infos[i].offset = 0;
        buf_infos[i].range  = VK_WHOLE_SIZE;
    }
    // LM head weight at index n_layers
    buf_infos[n_layers].buffer = lm_head_buf ? lm_head_buf->buffer : VK_NULL_HANDLE;
    buf_infos[n_layers].offset = 0;
    buf_infos[n_layers].range  = VK_WHOLE_SIZE;

    VkWriteDescriptorSet write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    write.dstSet          = arena->set0_weight_set;
    write.dstBinding      = 0;
    write.dstArrayElement = 0;
    write.descriptorCount = n_layers + 1;
    write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    write.pBufferInfo     = buf_infos.data();

    vkUpdateDescriptorSets(dev->device, 1, &write, 0, nullptr);
}

void vk_descriptor_arena_push_set1(vk_descriptor_arena_t* arena, VkDevice device, VkCommandBuffer cb, VkPipelineLayout layout,
                                    vk_buffer_t* hidden_in, vk_buffer_t* hidden_out,
                                    vk_buffer_t* k_cache, vk_buffer_t* v_cache,
                                    vk_buffer_t* scratch) {
    static PFN_vkCmdPushDescriptorSetKHR fn_push = nullptr;
    if (!fn_push) {
        fn_push = (PFN_vkCmdPushDescriptorSetKHR)vkGetDeviceProcAddr(device, "vkCmdPushDescriptorSetKHR");
    }

    if (!fn_push) return;

    VkDescriptorBufferInfo buf_infos[5];
    memset(buf_infos, 0, sizeof(buf_infos));

    VkBuffer dummy = arena->dummy_buf;
    VkBuffer b0 = hidden_in  ? hidden_in->buffer  : dummy;
    VkBuffer b1 = hidden_out ? hidden_out->buffer : dummy;
    VkBuffer b2 = k_cache    ? k_cache->buffer    : dummy;
    VkBuffer b3 = v_cache    ? v_cache->buffer    : dummy;
    VkBuffer b4 = scratch    ? scratch->buffer    : dummy;

    buf_infos[0].buffer = b0;
    buf_infos[0].offset = 0;
    buf_infos[0].range  = VK_WHOLE_SIZE;
    buf_infos[1].buffer = b1;
    buf_infos[1].offset = 0;
    buf_infos[1].range  = VK_WHOLE_SIZE;
    buf_infos[2].buffer = b2;
    buf_infos[2].offset = 0;
    buf_infos[2].range  = VK_WHOLE_SIZE;
    buf_infos[3].buffer = b3;
    buf_infos[3].offset = 0;
    buf_infos[3].range  = VK_WHOLE_SIZE;
    buf_infos[4].buffer = b4;
    buf_infos[4].offset = 0;
    buf_infos[4].range  = VK_WHOLE_SIZE;

    VkWriteDescriptorSet writes[5];
    memset(writes, 0, sizeof(writes));
    for (int i = 0; i < 5; i++) {
        writes[i].sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstBinding      = i;
        writes[i].dstArrayElement = 0;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo     = &buf_infos[i];
    }

    fn_push(cb, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 1, 5, writes);
}
