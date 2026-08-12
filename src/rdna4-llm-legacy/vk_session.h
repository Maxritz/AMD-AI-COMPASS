#pragma once

#include "common.h"
#include "vk_device.h"
#include "vk_model.h"
#include "vk_timeline.h"
#include "vk_descriptor.h"

#define MAX_PIPELINE_VARIANTS (OP_COUNT * QUANT_COUNT)

struct vk_pipeline_t {
    VkPipeline            pipeline;
    VkPipelineLayout      layout;
    VkShaderModule        module;
    VkDescriptorSetLayout set_layouts[3];
    uint32_t              push_constant_size;
    uint32_t              workgroup_x, workgroup_y, workgroup_z;
    int                   op_type;
    int                   quant_type;
};

struct vk_session_t {
    vk_device_t*          device;
    vk_model_t*           model;
    vk_timeline_t         timeline;
    vk_descriptor_arena_t desc;
    VkFence               transfer_fence;
    VkFence               compute_fence;
    uint64_t              pending_signal_value;  // signal value of last submit (0 = none pending)
    bool                  compute_failed;        // sticky: last submit failed / device lost

    vk_pipeline_t  pipelines[MAX_PIPELINE_VARIANTS];
    int            pipeline_count;
    VkPipelineLayout shared_pipeline_layout;

    struct {
        VkCommandBuffer cb;
        vk_buffer_t     hidden_buf[2];
        uint32_t        hidden_toggle;
        vk_buffer_t     logits_buf;
        vk_buffer_t     norm_scratch;
        vk_buffer_t     qkv_scratch;
        vk_buffer_t     attn_scratch;
        vk_buffer_t     ffn_scratch;
        vk_buffer_t     embed_staging;  // host-visible for token embedding upload
        vk_buffer_t     hidden_debug;   // host-visible: copy of lm_head_input for NaN bisection
        uint32_t        current_pos;
        uint32_t        total_seq_len;
    } decode_state;

    VkCommandPool  compute_cmd_pool;
    push_constants_t push_constants;
    vk_quant_type_t   weight_quant;
    vk_quant_type_t   lm_head_quant;
    vk_quant_type_t   emb_quant;
    bool            pipelines_ready;
};

bool    vk_session_create(vk_device_t* dev, vk_model_t* model, vk_session_t* session);
void    vk_session_destroy(vk_session_t* session);
bool    vk_session_build_pipelines(vk_session_t* session);
void    vk_session_build_decode_cb(vk_session_t* session);
bool    vk_session_submit(vk_session_t* session);
bool    vk_session_wait(vk_session_t* session, uint64_t timeout_ms);
float*  vk_session_poll_logits(vk_session_t* session);
uint64_t vk_session_timeline_value(vk_session_t* session);
