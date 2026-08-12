#pragma once

#include "common.h"

struct vk_device_t {
    VkInstance                          instance;
    VkPhysicalDevice                    physical_device;
    VkDevice                            device;
    VmaAllocator                        allocator;
    VkQueue                             compute_queue;
    VkQueue                             transfer_queue;
    VkCommandPool                       compute_cmd_pool;
    VkCommandPool                       transfer_cmd_pool;
    uint32_t                            compute_qf_idx;
    uint32_t                            transfer_qf_idx;
    uint32_t                            subgroup_size;
    bool                                has_cooperative_matrix;
    bool                                has_descriptor_indexing;
    bool                                has_timeline_semaphore;
    bool                                has_push_descriptor;
    bool                                has_subgroup_size_control;
    bool                                has_bda;
    bool                                has_float8;
    bool                                has_bf16_coopmat;
    bool                                has_int8;
    bool                                has_fp16;
    bool                                has_int16;
    uint32_t                            max_push_descriptors;
    uint32_t                            max_shared_memory;
    uint32_t                            max_workgroup_invocations;
    VkPhysicalDeviceProperties          props;
    VkPhysicalDeviceMemoryProperties    mem_props;
    VkPhysicalDeviceVulkan12Properties  vulkan12_props;
    VkPhysicalDeviceVulkan13Properties  vulkan13_props;
    VkPhysicalDeviceVulkan14Properties  vulkan14_props;
    VkPhysicalDeviceSubgroupProperties  subgroup_props;
    VkPipelineCache                     pipeline_cache;
    VkDebugUtilsMessengerEXT            debug_messenger;
};

vk_device_t* vk_device_create(bool enable_validation);
void vk_device_destroy(vk_device_t* dev);

PFN_vkVoidFunction vk_load_instance_fn(VkInstance inst, const char* name);
PFN_vkVoidFunction vk_load_device_fn(VkDevice dev, const char* name);
