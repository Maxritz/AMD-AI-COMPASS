#include "vk_session.h"
#include "vk_descriptor.h"
#include "vk_timeline.h"
#include "sample.h"

static VkPipelineLayout create_shared_pipeline_layout(VkDevice device,
                                                       VkDescriptorSetLayout set0,
                                                       VkDescriptorSetLayout set1,
                                                       VkDescriptorSetLayout set2) {
    VkDescriptorSetLayout set_layouts[3] = {set0, set1, set2};

    VkPushConstantRange pc_range = {};
    pc_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pc_range.offset     = 0;
    pc_range.size       = sizeof(push_constants_t);

    VkPipelineLayoutCreateInfo plci = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    plci.setLayoutCount         = 3;
    plci.pSetLayouts            = set_layouts;
    plci.pushConstantRangeCount = 1;
    plci.pPushConstantRanges    = &pc_range;

    VkPipelineLayout layout;
    VkResult result = vkCreatePipelineLayout(device, &plci, nullptr, &layout);
    if (result != VK_SUCCESS) {
        RDNA4_ERROR("Failed to create pipeline layout: %d", (int)result);
        return VK_NULL_HANDLE;
    }
    return layout;
}

static VkPipeline create_compute_pipeline(VkDevice device,
                                           const uint32_t* spirv_data, size_t spirv_size,
                                           VkPipelineLayout layout,
                                           uint32_t required_subgroup_size,
                                           const model_arch_t& config,
                                           VkPipelineCache cache) {
    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = spirv_size;
    smci.pCode    = spirv_data;

    VkShaderModule module;
    VkResult result = vkCreateShaderModule(device, &smci, nullptr, &module);
    if (result != VK_SUCCESS) return VK_NULL_HANDLE;

    VkPipelineShaderStageCreateInfo ssci = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    ssci.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    ssci.module = module;
    ssci.pName  = "main";

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_ci = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO
    };
    subgroup_ci.requiredSubgroupSize = required_subgroup_size;
    ssci.pNext = &subgroup_ci;

    // Specialization constants matching common.glsl layout
    // constant_id=0: SPEC_SUBGROUP_SIZE
    // constant_id=1: SPEC_WG_SIZE_X
    // constant_id=2: SPEC_D
    // constant_id=3: SPEC_FFN_DIM
    // constant_id=4: SPEC_HEAD_DIM
    // constant_id=5: SPEC_N_HEADS
    // constant_id=6: SPEC_N_KV_HEADS
    // constant_id=7: SPEC_VOCAB_SIZE
    // 16 specialization constants matching common.glsl + rms_norm.comp
    uint32_t spec_data[] = {
        required_subgroup_size,                    // 0:  SPEC_SUBGROUP_SIZE
        required_subgroup_size * 4,                // 1:  SPEC_WG_SIZE_X
        config.d,                                  // 2:  SPEC_D
        config.ffn_dim,                            // 3:  SPEC_FFN_DIM
        config.head_dim,                           // 4:  SPEC_HEAD_DIM
        config.n_heads,                            // 5:  SPEC_N_HEADS
        config.n_kv_heads,                         // 6:  SPEC_N_KV_HEADS
        config.vocab_size,                         // 7:  SPEC_VOCAB_SIZE
        256,                                       // 8:  SPEC_KV_PAGE_TOKENS (default)
        16,                                        // 9:  SPEC_PAGES_PER_LAYER (default)
        0,                                         // 10: SPEC_QUANT_TYPE (default)
        4,                                         // 11: SPEC_WARP_PER_WG (default)
        4096,                                      // 12: SPEC_MAX_SEQ_LEN (default)
    };
    int spec_count = 13;

    VkSpecializationMapEntry spec_entries[13] = {};
    for (int i = 0; i < spec_count; i++) {
        spec_entries[i].constantID = i;
        spec_entries[i].offset     = i * sizeof(uint32_t);
        spec_entries[i].size       = sizeof(uint32_t);
    }

    VkSpecializationInfo spec_info = {};
    spec_info.mapEntryCount = spec_count;
    spec_info.pMapEntries   = spec_entries;
    spec_info.dataSize      = sizeof(spec_data);
    spec_info.pData         = spec_data;
    ssci.pSpecializationInfo = &spec_info;

    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage  = ssci;
    cpci.layout = layout;

    VkPipeline pipeline;
    result = vkCreateComputePipelines(device, cache, 1, &cpci, nullptr, &pipeline);

    vkDestroyShaderModule(device, module, nullptr);

    if (result != VK_SUCCESS) return VK_NULL_HANDLE;
    return pipeline;
}

static void barrier_buf(VkCommandBuffer cb, VkBuffer buf,
                         VkAccessFlags src_access, VkAccessFlags dst_access) {
    (void)buf;
    // Device-wide memory barrier: on RDNA4, buffer-scoped barriers between
    // compute dispatches in the SAME command buffer do not reliably make the
    // producer's writes visible to the consumer (the layer-0 output was wrong
    // until this was made a full memory barrier). The stage flags follow the
    // access flags.
    VkMemoryBarrier barrier = {VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = src_access;
    barrier.dstAccessMask = dst_access;

    VkPipelineStageFlags src_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    if (src_access & (VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT))
        src_stage = VK_PIPELINE_STAGE_TRANSFER_BIT;

    VkPipelineStageFlags dst_stage = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    if (dst_access & (VK_ACCESS_HOST_READ_BIT | VK_ACCESS_HOST_WRITE_BIT))
        dst_stage = VK_PIPELINE_STAGE_HOST_BIT;

    vkCmdPipelineBarrier(cb,
        src_stage, dst_stage,
        0, 1, &barrier, 0, nullptr, 0, nullptr);
}

static void barrier_buf_host_read(VkCommandBuffer cb, VkBuffer buf) {
    VkBufferMemoryBarrier barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
    barrier.srcAccessMask       = VK_ACCESS_SHADER_WRITE_BIT;
    barrier.dstAccessMask       = VK_ACCESS_HOST_READ_BIT;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.buffer              = buf;
    barrier.offset              = 0;
    barrier.size                = VK_WHOLE_SIZE;

    vkCmdPipelineBarrier(cb,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_HOST_BIT,
        0, 0, nullptr, 1, &barrier, 0, nullptr);
}

static bool buffer_create(vk_device_t* dev, VkDeviceSize size,
                           VkBufferUsageFlags usage, VkMemoryPropertyFlags mem_flags,
                           bool host_visible, vk_buffer_t* out) {
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size        = size;
    bci.usage       = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo aci = {};
    aci.usage         = VMA_MEMORY_USAGE_AUTO;
    aci.requiredFlags = mem_flags;

    if (host_visible) {
        aci.flags |= VMA_ALLOCATION_CREATE_MAPPED_BIT |
                     VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
    }

    VkResult result = vmaCreateBuffer(dev->allocator, &bci, &aci,
                                       &out->buffer, &out->allocation, &out->alloc_info);
    if (result != VK_SUCCESS) return false;

    out->size            = size;
    out->device_address  = 0;
    out->mapped_ptr      = host_visible ? out->alloc_info.pMappedData : nullptr;
    out->is_host_visible = host_visible;
    out->is_host_coherent = (mem_flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) != 0;
    out->usage           = usage;
    return true;
}

bool vk_session_create(vk_device_t* dev, vk_model_t* model, vk_session_t* session) {
    memset(session, 0, sizeof(*session));
    session->device = dev;
    session->model  = model;
    // Representative types for pipeline selection: layer 0's actual tensor
    // type per role (real GGUF tensors are role-consistent across layers).
    session->weight_quant  = model->tensors[0].q.quant;
    session->lm_head_quant = model->lm_head.quant;
    session->emb_quant     = model->token_embd.quant;

    if (!vk_timeline_create(dev->device, &session->timeline)) {
        RDNA4_ERROR("Failed to create timeline semaphore");
        return false;
    }

    uint32_t n_layers = model->config.n_layers;
    uint32_t head_dim = model->config.head_dim;
    if (head_dim == 0) head_dim = 128;

    if (!vk_descriptor_arena_create(dev, &session->desc, n_layers,
                                     head_dim, MAX_SEQ_LEN, model->config.rope_theta)) {
        RDNA4_ERROR("Failed to create descriptor arena");
        vk_timeline_destroy(dev->device, &session->timeline);
        return false;
    }

    vk_descriptor_arena_update_weights(dev, &session->desc,
                                        model->weight_bufs, n_layers,
                                        &model->lm_head_buf);

    // Bind the KV page table into set2 binding 1 (was never written -> GPU fault).
    if (model->kv_cache.page_table.buffer) {
        VkDescriptorBufferInfo pt_info = {};
        pt_info.buffer = model->kv_cache.page_table.buffer;
        pt_info.offset = 0;
        pt_info.range  = VK_WHOLE_SIZE;
        VkWriteDescriptorSet pt_write = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        pt_write.dstSet          = session->desc.set2_table_set;
        pt_write.dstBinding      = 1;  // BINDING_PAGE_TABLE
        pt_write.dstArrayElement = 0;
        pt_write.descriptorCount = 1;
        pt_write.descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pt_write.pBufferInfo     = &pt_info;
        vkUpdateDescriptorSets(dev->device, 1, &pt_write, 0, nullptr);
    }

    session->shared_pipeline_layout = create_shared_pipeline_layout(
        dev->device,
        session->desc.set0_layout,
        session->desc.set1_layout,
        session->desc.set2_layout);

    if (!session->shared_pipeline_layout) {
        vk_descriptor_arena_destroy(dev, &session->desc);
        vk_timeline_destroy(dev->device, &session->timeline);
        return false;
    }

    VkFenceCreateInfo fci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(dev->device, &fci, nullptr, &session->transfer_fence);

    // Compute fence: signaled initially so the first wait (before any submit)
    // succeeds; reset before each submit.
    VkFenceCreateInfo cfci = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    cfci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(dev->device, &cfci, nullptr, &session->compute_fence);
    session->pending_signal_value = 0;
    session->compute_failed      = false;

    session->compute_cmd_pool = dev->compute_cmd_pool;

    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool        = session->compute_cmd_pool;
    cai.level              = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(dev->device, &cai, &session->decode_state.cb));

    uint32_t d         = model->config.d;
    uint32_t ffn_dim   = model->config.ffn_dim;
    uint32_t vocab_size = model->config.vocab_size;
    uint32_t n_kv_heads = model->config.n_kv_heads;

    VkBufferUsageFlags io_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                   VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    VkMemoryPropertyFlags dev_local = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    if (!buffer_create(dev, d * 2, io_usage, dev_local, false, &session->decode_state.hidden_buf[0])) return false;
    if (!buffer_create(dev, d * 2, io_usage, dev_local, false, &session->decode_state.hidden_buf[1])) return false;
    if (!buffer_create(dev, d * 2, io_usage, dev_local, false, &session->decode_state.norm_scratch)) return false;
    // QKV output = Q+K+V = (n_heads + 2*n_kv_heads)*head_dim fp16 elements.
    uint64_t qkv_elems = ((uint64_t)model->config.n_heads + 2ull * model->config.n_kv_heads) * head_dim;
    if (!buffer_create(dev, qkv_elems * 2, io_usage, dev_local, false, &session->decode_state.qkv_scratch)) return false;
    // Attention output = n_heads * head_dim fp16 elements (not just d).
    uint64_t attn_elems = (uint64_t)model->config.n_heads * head_dim;
    if (!buffer_create(dev, attn_elems * 2, io_usage, dev_local, false, &session->decode_state.attn_scratch)) return false;
    if (!buffer_create(dev, ffn_dim * 2, io_usage, dev_local, false, &session->decode_state.ffn_scratch)) return false;

    VkBufferUsageFlags logits_usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    if (!buffer_create(dev, vocab_size * sizeof(float), logits_usage,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        true, &session->decode_state.logits_buf)) {
        RDNA4_ERROR("Failed to create logits buffer");
        return false;
    }

    // TEMP DEBUG: host-visible copy of lm_head_input for NaN bisection.
    if (!buffer_create(dev, d * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        true, &session->decode_state.hidden_debug)) {
        RDNA4_ERROR("Failed to create hidden_debug buffer");
        return false;
    }

    session->decode_state.hidden_toggle = 0;
    session->decode_state.current_pos   = 0;
    session->decode_state.total_seq_len = 0;

    session->push_constants = {};
    session->push_constants.layer_idx        = 0;
    session->push_constants.kv_cache_pos     = 0;
    session->push_constants.seq_len          = 0;
    session->push_constants.head_dim         = model->config.head_dim;
    session->push_constants.num_heads        = model->config.n_heads;
    session->push_constants.num_kv_heads     = model->config.n_kv_heads;
    session->push_constants.attn_scale       = 1.0f / sqrtf((float)model->config.head_dim);
    session->push_constants.rope_theta       = model->config.rope_theta;
    session->push_constants.rope_is_neox     = model->config.rope_is_neox ? 1u : 0u;
    session->push_constants.norm_eps         = model->config.norm_eps;
    session->push_constants.embedding_scale  = model->config.embedding_scale;
    session->push_constants.residual_scale   = model->config.residual_scale;
    session->push_constants.logit_scale      = model->config.logit_scale;
    // Every layer's real per-tensor metadata is computed independently in
    // vk_model_load, but well-formed GGUFs are role-consistent layer-to-layer,
    // so layer 0's offsets are representative for the push constants set once
    // here. The qk_norm offsets get refreshed per-layer below where used.
    const layer_tensors_t& lt0 = model->tensors[0];
    session->push_constants.attn_norm_offset   = (uint32_t)lt0.attn_norm.offset;
    session->push_constants.ffn_norm_offset    = (uint32_t)lt0.ffn_norm.offset;
    session->push_constants.output_norm_offset = (uint32_t)model->output_norm.offset;
    session->push_constants.q_norm_offset      = (uint32_t)lt0.q_norm.offset;
    session->push_constants.k_norm_offset      = (uint32_t)lt0.k_norm.offset;
    session->push_constants.q_offset    = (uint32_t)lt0.q.offset;
    session->push_constants.k_offset    = (uint32_t)lt0.k.offset;
    session->push_constants.v_offset    = (uint32_t)lt0.v.offset;
    session->push_constants.o_offset    = (uint32_t)lt0.o.offset;
    session->push_constants.gate_offset = (uint32_t)lt0.gate.offset;
    session->push_constants.up_offset   = (uint32_t)lt0.up.offset;
    session->push_constants.down_offset = (uint32_t)lt0.down.offset;
    session->push_constants.has_qkv_bias  = lt0.q_bias.present ? 1u : 0u;
    session->push_constants.q_bias_offset = (uint32_t)lt0.q_bias.offset;
    session->push_constants.k_bias_offset = (uint32_t)lt0.k_bias.offset;
    session->push_constants.v_bias_offset = (uint32_t)lt0.v_bias.offset;

    RDNA4_LOG("Session created");
    return true;
}

void vk_session_destroy(vk_session_t* session) {
    if (!session || !session->device) return;
    VkDevice device = session->device->device;
    VmaAllocator allocator = session->device->allocator;

    vkDeviceWaitIdle(device);

    for (int i = 0; i < session->pipeline_count; i++) {
        if (session->pipelines[i].pipeline)
            vkDestroyPipeline(device, session->pipelines[i].pipeline, nullptr);
    }

    if (session->shared_pipeline_layout)
        vkDestroyPipelineLayout(device, session->shared_pipeline_layout, nullptr);

    if (session->decode_state.cb)
        vkFreeCommandBuffers(device, session->compute_cmd_pool, 1, &session->decode_state.cb);

    vmaDestroyBuffer(allocator, session->decode_state.hidden_buf[0].buffer,  session->decode_state.hidden_buf[0].allocation);
    vmaDestroyBuffer(allocator, session->decode_state.hidden_buf[1].buffer,  session->decode_state.hidden_buf[1].allocation);
    vmaDestroyBuffer(allocator, session->decode_state.norm_scratch.buffer,    session->decode_state.norm_scratch.allocation);
    vmaDestroyBuffer(allocator, session->decode_state.qkv_scratch.buffer,     session->decode_state.qkv_scratch.allocation);
    vmaDestroyBuffer(allocator, session->decode_state.attn_scratch.buffer,    session->decode_state.attn_scratch.allocation);
    vmaDestroyBuffer(allocator, session->decode_state.ffn_scratch.buffer,     session->decode_state.ffn_scratch.allocation);
    vmaDestroyBuffer(allocator, session->decode_state.logits_buf.buffer,      session->decode_state.logits_buf.allocation);

    vkDestroyFence(device, session->transfer_fence, nullptr);
    if (session->compute_fence)
        vkDestroyFence(device, session->compute_fence, nullptr);
    vk_descriptor_arena_destroy(session->device, &session->desc);
    vk_timeline_destroy(device, &session->timeline);

    memset(session, 0, sizeof(*session));
}

#include "shader_registry.h"

static const embedded_shader_t* lookup_shader(int op_type, int quant_type, uint32_t subgroup_size) {
    for (const embedded_shader_t* s = EMBEDDED_SHADERS; s->name != nullptr; s++) {
        if (s->op_type == op_type && s->quant_type == quant_type &&
            s->subgroup_size == subgroup_size) {
            return s;
        }
    }
    if (quant_type != QUANT_FP16) {
        return lookup_shader(op_type, QUANT_FP16, subgroup_size);
    }
    return nullptr;
}

bool vk_session_build_pipelines(vk_session_t* session) {
    vk_device_t* dev = session->device;
    uint32_t sg_size = dev->subgroup_size;

    int ops[] = {
        OP_RMS_NORM, OP_ATTN_QKV, OP_ATTN_COMPUTE, OP_ATTN_OUTPUT, OP_ROPE,
        OP_QK_NORM, OP_FFN_GATE_UP, OP_FFN_DOWN, OP_LM_HEAD, OP_EMBEDDING_LOOKUP
    };
    int quants[] = {
        QUANT_FP16, QUANT_Q4_K, QUANT_Q6_K, QUANT_Q8_0, QUANT_IQ4_XS
    };

    session->pipeline_count = 0;

    for (int oi = 0; oi < (int)(sizeof(ops) / sizeof(ops[0])); oi++) {
        int op = ops[oi];
        for (int qi = 0; qi < (int)(sizeof(quants) / sizeof(quants[0])); qi++) {
            int qt = quants[qi];

            if (op == OP_RMS_NORM && qt != QUANT_FP16) continue;
            if (op == OP_ATTN_COMPUTE && qt != QUANT_FP16) continue;
            if (op == OP_ROPE && qt != QUANT_FP16) continue;
            if (op == OP_QK_NORM && qt != QUANT_FP16) continue;

            const embedded_shader_t* shader = lookup_shader(op, qt, sg_size);
            if (!shader) {
                continue;
            }

            int idx = session->pipeline_count++;
            if (idx >= MAX_PIPELINE_VARIANTS) break;

            session->pipelines[idx].set_layouts[0] = session->desc.set0_layout;
            session->pipelines[idx].set_layouts[1] = session->desc.set1_layout;
            session->pipelines[idx].set_layouts[2] = session->desc.set2_layout;
            session->pipelines[idx].layout = session->shared_pipeline_layout;
            session->pipelines[idx].push_constant_size = sizeof(push_constants_t);
            session->pipelines[idx].workgroup_x = sg_size;
            session->pipelines[idx].workgroup_y = 1;
            session->pipelines[idx].workgroup_z = 1;
            session->pipelines[idx].op_type   = op;
            session->pipelines[idx].quant_type = qt;

            session->pipelines[idx].pipeline = create_compute_pipeline(
                dev->device,
                shader->data, shader->byte_size,
                session->shared_pipeline_layout,
                sg_size,
                session->model->config,
                dev->pipeline_cache);

            if (!session->pipelines[idx].pipeline) {
                session->pipelines[idx].pipeline = VK_NULL_HANDLE;
            }
        }
    }

    RDNA4_LOG("Pipelines created: %d", session->pipeline_count);
    return session->pipeline_count > 0;
}

static VkPipeline find_pipeline(vk_session_t* session, int op_type, int quant_type) {
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

void vk_session_build_decode_cb(vk_session_t* session) {
    vk_device_t* dev  = session->device;
    vk_model_t* model = session->model;

    VkCommandBuffer cb = session->decode_state.cb;
    vkResetCommandBuffer(cb, 0);

    VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));

    VkDescriptorSet sets_to_bind[2] = {
        session->desc.set0_weight_set,
        session->desc.set2_table_set
    };

    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
        session->shared_pipeline_layout, 0, 1, &sets_to_bind[0], 0, nullptr);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE,
        session->shared_pipeline_layout, 2, 1, &sets_to_bind[1], 0, nullptr);

    // Representative per-role type: layer 0's actual tensor type (real GGUF
    // tensors are role-consistent across layers).
    const layer_tensors_t& lt0 = model->tensors[0];
    VkPipeline pipeline_rms_norm = find_pipeline(session, OP_RMS_NORM, QUANT_FP16);
    VkPipeline pipeline_attn_qkv = find_pipeline(session, OP_ATTN_QKV, lt0.q.quant);
    VkPipeline pipeline_attn_compute = find_pipeline(session, OP_ATTN_COMPUTE, QUANT_FP16);
    VkPipeline pipeline_rope = find_pipeline(session, OP_ROPE, QUANT_FP16);
    VkPipeline pipeline_qk_norm = find_pipeline(session, OP_QK_NORM, QUANT_FP16);
    VkPipeline pipeline_attn_output = find_pipeline(session, OP_ATTN_OUTPUT, lt0.o.quant);
    VkPipeline pipeline_ffn_gate_up = find_pipeline(session, OP_FFN_GATE_UP, lt0.gate.quant);
    VkPipeline pipeline_ffn_down = find_pipeline(session, OP_FFN_DOWN, lt0.down.quant);
    VkPipeline pipeline_lm_head = find_pipeline(session, OP_LM_HEAD, session->lm_head_quant);
    VkPipeline pipeline_token_embed = find_pipeline(session, OP_EMBEDDING_LOOKUP, session->emb_quant);

    // All 5 set1 slots are part of the pipeline layout; bind REAL buffers for
    // every slot so no descriptor is null/dummy. Shaders ignore unused slots.
    vk_buffer_t* kv_buf   = &model->kv_cache.buffer;
    vk_buffer_t* ffn_scr  = &session->decode_state.ffn_scratch;

    VkPipeline default_pipeline = VK_NULL_HANDLE;
    for (int i = 0; i < session->pipeline_count; i++) {
        if (session->pipelines[i].pipeline) {
            default_pipeline = session->pipelines[i].pipeline;
            break;
        }
    }

    if (!pipeline_rms_norm)     pipeline_rms_norm    = default_pipeline;
    if (!pipeline_attn_qkv)     pipeline_attn_qkv    = default_pipeline;
    if (!pipeline_attn_compute) pipeline_attn_compute = default_pipeline;
    if (!pipeline_rope)         pipeline_rope         = default_pipeline;
    if (!pipeline_qk_norm)      pipeline_qk_norm      = default_pipeline;
    if (!pipeline_attn_output)  pipeline_attn_output = default_pipeline;
    if (!pipeline_ffn_gate_up)  pipeline_ffn_gate_up = default_pipeline;
    if (!pipeline_ffn_down)     pipeline_ffn_down    = default_pipeline;
    if (!pipeline_lm_head)      pipeline_lm_head     = default_pipeline;
    if (!pipeline_token_embed)  pipeline_token_embed = default_pipeline;

    if (!default_pipeline) {
        RDNA4_ERROR("No pipelines available for decode CB");
        vkEndCommandBuffer(cb);
        return;
    }

    uint32_t n_layers   = model->config.n_layers;
    uint32_t d          = model->config.d;
    uint32_t ffn_dim    = model->config.ffn_dim;

    vk_buffer_t* hidden_bufs = session->decode_state.hidden_buf;
    vk_buffer_t* norm_buf    = &session->decode_state.norm_scratch;
    vk_buffer_t* qkv_buf     = &session->decode_state.qkv_scratch;
    vk_buffer_t* attn_buf    = &session->decode_state.attn_scratch;
    vk_buffer_t* ffn_buf     = &session->decode_state.ffn_scratch;
    vk_buffer_t* logits_buf  = &session->decode_state.logits_buf;

    push_constants_t pc = session->push_constants;

    // === Token embedding: copy one row from embedding table to hidden_buf[0] ===
    // Done in a COMPUTE shader (token_embed): a vkCmdCopyBuffer followed by a
    // compute read is not reliably synchronized on RDNA4's compute queue.
    {
        uint32_t emb_spg = 128 / dev->subgroup_size;
        if (emb_spg == 0) emb_spg = 1;
        vkCmdPushConstants(cb, session->shared_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vk_descriptor_arena_push_set1(&session->desc, dev->device, cb, session->shared_pipeline_layout,
                                       &model->embedding_buf, &hidden_bufs[0],
                                       kv_buf, kv_buf, ffn_scr);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_token_embed);
        vkCmdDispatch(cb, (d + 128 - 1) / 128, 1, 1);
        barrier_buf(cb, hidden_bufs[0].buffer,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
    }

    // The embedding row always lands in hidden_buf[0], so layer 0 must always
    // read buffer 0. hidden_toggle is sticky session state from the previous
    // token's last layer — reset it for this token's layer loop.
    session->decode_state.hidden_toggle = 0;

    // Subgroups per workgroup (WG_SIZE_X = subgroup*4, so 4 on RDNA3/4).
    // The QKV/output/down/lm_head shaders compute ONE row per subgroup.
    uint32_t spg = 128 / dev->subgroup_size;
    if (spg == 0) spg = 1;

    for (uint32_t l = 0; l < n_layers; l++) {
        pc.layer_idx = l;
        uint32_t in_toggle  = session->decode_state.hidden_toggle;
        uint32_t out_toggle = 1 - in_toggle;

        vkCmdPushConstants(cb, session->shared_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

        pc.norm_type = 0;
        vkCmdPushConstants(cb, session->shared_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vk_descriptor_arena_push_set1(&session->desc, dev->device, cb, session->shared_pipeline_layout,
                                       &hidden_bufs[in_toggle], norm_buf,
                                       kv_buf, kv_buf, ffn_scr);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_rms_norm);
        vkCmdDispatch(cb, 1, 1, 1);
        barrier_buf(cb, norm_buf->buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        vk_descriptor_arena_push_set1(&session->desc, dev->device, cb, session->shared_pipeline_layout,
                                       norm_buf, qkv_buf,
                                       &model->kv_cache.buffer, &model->kv_cache.buffer,
                                       ffn_scr);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_attn_qkv);
        // QKV writes (n_heads + 2*n_kv_heads)*head_dim fp16 elements, one element
        // per subgroup. Dispatch must cover TOTAL_ELEMS.
        {
            uint64_t qkv_elems = ((uint64_t)model->config.n_heads + 2ull * model->config.n_kv_heads) * model->config.head_dim;
            vkCmdDispatch(cb, (uint32_t)((qkv_elems + spg - 1) / spg), 1, 1);
        }
        barrier_buf(cb, qkv_buf->buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        // The QKV dispatch also WRITES the KV cache (current token's K/V); make it
        // visible to the attention dispatch that reads it.
        barrier_buf(cb, model->kv_cache.buffer.buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        // ── Qwen3 per-head Q/K RMS norm (before RoPE in attn_qkv) ────────
        // Qwen3 applies separate per-head RMS norm to Q and K AFTER the projection
        // matmul but BEFORE RoPE, on the raw Q/K in qkv_buf. Dedicated shader
        // (qk_norm.comp) instead of the old two rms_norm.comp dispatches with
        // norm_type 3/4 -- rewritten as its own single-purpose shader (one
        // dispatch covering both Q and K heads, matching rope.comp's shape)
        // rather than another branch bolted onto the D-wide attn/ffn/output
        // norm shader.
        if (model->has_qk_norm) {
            pc.q_norm_offset = (uint32_t)model->tensors[l].q_norm.offset;
            pc.k_norm_offset = (uint32_t)model->tensors[l].k_norm.offset;
            vkCmdPushConstants(cb, session->shared_pipeline_layout,
                               VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vk_descriptor_arena_push_set1(&session->desc, dev->device, cb, session->shared_pipeline_layout,
                                           qkv_buf, qkv_buf,
                                           kv_buf, kv_buf, ffn_scr);
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_qk_norm);
            vkCmdDispatch(cb, model->config.n_heads + model->config.n_kv_heads, 1, 1);
            barrier_buf(cb, qkv_buf->buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        }

        // ── RoPE (NEOX split-half rotation) ──────────────────────────────
        // Applied AFTER qk_norm (Qwen3 normalizes raw pre-RoPE Q/K) and BEFORE
        // attn_compute reads Q/K. Rotates Q in-place in qkv_buf and K in-place
        // in the KV cache at the current position; each cached K position is
        // rotated exactly once, when it's written, so past positions read back
        // already-rotated. See shaders/rope.comp for why this must be a
        // separate pass rather than fused into attn_qkv.
        vkCmdPushConstants(cb, session->shared_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vk_descriptor_arena_push_set1(&session->desc, dev->device, cb, session->shared_pipeline_layout,
                                       qkv_buf, qkv_buf,
                                       kv_buf, kv_buf, ffn_scr);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_rope);
        vkCmdDispatch(cb, model->config.n_heads + model->config.n_kv_heads, 1, 1);
        barrier_buf(cb, qkv_buf->buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        barrier_buf(cb, model->kv_cache.buffer.buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        vk_descriptor_arena_push_set1(&session->desc, dev->device, cb, session->shared_pipeline_layout,
                                       qkv_buf, attn_buf,
                                       &model->kv_cache.buffer, &model->kv_cache.buffer,
                                       ffn_scr);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_attn_compute);
        vkCmdDispatch(cb, model->config.n_heads, 1, 1);
        barrier_buf(cb, attn_buf->buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        vk_descriptor_arena_push_set1(&session->desc, dev->device, cb, session->shared_pipeline_layout,
                                       attn_buf, &hidden_bufs[out_toggle],
                                       kv_buf, kv_buf, &hidden_bufs[in_toggle]);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_attn_output);
        vkCmdDispatch(cb, (d + spg - 1) / spg, 1, 1);
        barrier_buf(cb, hidden_bufs[out_toggle].buffer,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        vk_descriptor_arena_push_set1(&session->desc, dev->device, cb, session->shared_pipeline_layout,
                                       &hidden_bufs[out_toggle], norm_buf,
                                       kv_buf, kv_buf, ffn_scr);
        pc.norm_type = 1;
        vkCmdPushConstants(cb, session->shared_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_rms_norm);
        vkCmdDispatch(cb, 1, 1, 1);
        barrier_buf(cb, norm_buf->buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        vk_descriptor_arena_push_set1(&session->desc, dev->device, cb, session->shared_pipeline_layout,
                                       norm_buf, ffn_buf,
                                       kv_buf, kv_buf, ffn_scr);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_ffn_gate_up);
        // One subgroup computes one ffn output row (SUBGROUPS_PER_WG = WG/32 = 4),
        // so the dispatch needs ceil(FFN_DIM / 4) workgroups — NOT /64.
        {
            uint32_t subgroups_per_wg = 128 / dev->subgroup_size;
            if (subgroups_per_wg == 0) subgroups_per_wg = 1;
            vkCmdDispatch(cb, (ffn_dim + subgroups_per_wg - 1) / subgroups_per_wg, 1, 1);
        }
        barrier_buf(cb, ffn_buf->buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        vk_descriptor_arena_push_set1(&session->desc, dev->device, cb, session->shared_pipeline_layout,
                                       ffn_buf, &hidden_bufs[out_toggle],
                                       kv_buf, kv_buf, ffn_scr);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_ffn_down);
        vkCmdDispatch(cb, (d + spg - 1) / spg, 1, 1);
        barrier_buf(cb, hidden_bufs[out_toggle].buffer,
                    VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        session->decode_state.hidden_toggle = out_toggle;
    }

    uint32_t final_toggle = session->decode_state.hidden_toggle;
    vk_buffer_t* lm_head_input = &hidden_bufs[final_toggle];

    // ── Final output_norm (RMS) before the LM head ────────────────────────
    // Most llama-family GGUFs apply output_norm.weight to the last hidden state
    // before the logit projection. Read it from lm_head_buf (set0 index n_layers)
    // at output_norm_offset via rms_norm's norm_type==2 path.
    if (model->has_output_norm) {
        vk_buffer_t* norm_buf = &session->decode_state.norm_scratch;
        pc.layer_idx     = n_layers;  // lm_head buffer slot
        pc.norm_type     = 2;         // output_norm
        vkCmdPushConstants(cb, session->shared_pipeline_layout,
                           VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vk_descriptor_arena_push_set1(&session->desc, dev->device, cb, session->shared_pipeline_layout,
                                       lm_head_input, norm_buf,
                                       kv_buf, kv_buf, ffn_scr);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_rms_norm);
        vkCmdDispatch(cb, 1, 1, 1);
        barrier_buf(cb, norm_buf->buffer, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);

        // LM head now consumes the normalized hidden state.
        lm_head_input = norm_buf;
    }

    // TEMP DEBUG: copy lm_head_input to a host-visible buffer for NaN bisection.
    {
        VkBufferCopy dbg_copy = {0, 0, (uint64_t)model->config.d * 2ull};
        vkCmdCopyBuffer(cb, lm_head_input->buffer, session->decode_state.hidden_debug.buffer, 1, &dbg_copy);
        VkBufferMemoryBarrier dbg_barrier = {VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER};
        dbg_barrier.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        dbg_barrier.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        dbg_barrier.buffer = session->decode_state.hidden_debug.buffer;
        dbg_barrier.size   = VK_WHOLE_SIZE;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
                             0, 0, nullptr, 1, &dbg_barrier, 0, nullptr);
    }

    pc.layer_idx = n_layers;  // LM head weight is at array index n_layers

    vkCmdPushConstants(cb, session->shared_pipeline_layout,
                       VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    vk_descriptor_arena_push_set1(&session->desc, dev->device, cb, session->shared_pipeline_layout,
                                   lm_head_input, logits_buf,
                                   kv_buf, kv_buf, ffn_scr);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_lm_head);
    vkCmdDispatch(cb, (model->config.vocab_size + spg - 1) / spg, 1, 1);

    barrier_buf_host_read(cb, logits_buf->buffer);

    VK_CHECK(vkEndCommandBuffer(cb));
}

bool vk_session_submit(vk_session_t* session) {
    vk_device_t* dev      = session->device;

    if (session->compute_failed) return false;

    // Synchronous path: the host waits on a fence after every submit, so the
    // timeline semaphore is not needed for ordering.
    VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers    = &session->decode_state.cb;

    vkResetFences(dev->device, 1, &session->compute_fence);

    VkResult result = vkQueueSubmit(dev->compute_queue, 1, &si, session->compute_fence);
    if (result == VK_SUCCESS) {
        session->pending_signal_value = 1;
        return true;
    }

    session->compute_failed = true;
    session->pending_signal_value = 0;
    RDNA4_ERROR("vkQueueSubmit failed: %d", (int)result);
    return false;
}

bool vk_session_wait(vk_session_t* session, uint64_t timeout_ms) {
    vk_device_t* dev = session->device;

    if (session->compute_failed) return false;
    if (session->pending_signal_value == 0) return true;  // nothing pending

    VkResult result = vkWaitForFences(dev->device, 1, &session->compute_fence,
                                      VK_TRUE, timeout_ms * 1000000ull);
    if (result == VK_SUCCESS) {
        session->pending_signal_value = 0;
        return true;
    }
    if (result == VK_TIMEOUT) {
        session->compute_failed = true;
        session->pending_signal_value = 0;
        RDNA4_ERROR("GPU fence timeout after %llu ms", (unsigned long long)timeout_ms);
        return false;
    }
    session->compute_failed = true;
    session->pending_signal_value = 0;
    RDNA4_ERROR("vkWaitForFences failed: %d (device lost?)", (int)result);
    return false;
}

float* vk_session_poll_logits(vk_session_t* session) {
    if (session->compute_failed) return nullptr;
    // Only consider valid once the EXACT signal value of the last submit
    // has been reached — never a stale/previous value.
    if (session->pending_signal_value == 0) return nullptr;
    uint64_t val = vk_timeline_poll(session->device->device, &session->timeline);
    if (val >= session->pending_signal_value) {
        session->pending_signal_value = 0;
        return (float*)session->decode_state.logits_buf.mapped_ptr;
    }
    return nullptr;
}

uint64_t vk_session_timeline_value(vk_session_t* session) {
    return vk_timeline_poll(session->device->device, &session->timeline);
}
