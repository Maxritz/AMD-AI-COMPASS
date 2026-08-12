#include "vk_device.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
    VkDebugUtilsMessageSeverityFlagBitsEXT      severity,
    VkDebugUtilsMessageTypeFlagsEXT             type,
    const VkDebugUtilsMessengerCallbackDataEXT* data,
    void*                                       user_data)
{
    (void)user_data;
    (void)type;
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        fprintf(stderr, "[VK_DEBUG] %s\n", data->pMessage);
    }
    return VK_FALSE;
}

static VkResult create_debug_messenger(VkInstance instance, VkDebugUtilsMessengerEXT* messenger) {
    PFN_vkCreateDebugUtilsMessengerEXT fn = (PFN_vkCreateDebugUtilsMessengerEXT)
        vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");
    if (!fn) return VK_ERROR_EXTENSION_NOT_PRESENT;

    VkDebugUtilsMessengerCreateInfoEXT ci = {};
    ci.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    ci.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    ci.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                     VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    ci.pfnUserCallback = debug_callback;
    return fn(instance, &ci, nullptr, messenger);
}

PFN_vkVoidFunction vk_load_instance_fn(VkInstance inst, const char* name) {
    return vkGetInstanceProcAddr(inst, name);
}

PFN_vkVoidFunction vk_load_device_fn(VkDevice dev, const char* name) {
    return vkGetDeviceProcAddr(dev, name);
}

vk_device_t* vk_device_create(bool enable_validation) {
    vk_device_t* dev = new vk_device_t();
    memset(dev, 0, sizeof(vk_device_t));
    dev->compute_qf_idx = UINT32_MAX;
    dev->transfer_qf_idx = UINT32_MAX;
    dev->subgroup_size = RDNA4_SUBGROUP_SIZE;
    dev->max_push_descriptors = 32;

    std::vector<const char*> all_inst_exts;
    if (enable_validation) {
        all_inst_exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    const char* inst_layers[] = { "VK_LAYER_KHRONOS_validation" };
    uint32_t inst_layer_count = enable_validation ? 1 : 0;

    uint32_t api_version = VK_API_VERSION_1_4;

    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "RDNA4-LLM";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "RDNA4-LLM";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = api_version;

    VkInstanceCreateInfo inst_ci = {};
    inst_ci.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_ci.pApplicationInfo = &app_info;
    inst_ci.enabledExtensionCount = (uint32_t)all_inst_exts.size();
    inst_ci.ppEnabledExtensionNames = all_inst_exts.data();
    inst_ci.enabledLayerCount = inst_layer_count;
    inst_ci.ppEnabledLayerNames = inst_layers;

    VK_CHECK(vkCreateInstance(&inst_ci, nullptr, &dev->instance));

    if (enable_validation) {
        create_debug_messenger(dev->instance, &dev->debug_messenger);
    }

    uint32_t gpu_count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(dev->instance, &gpu_count, nullptr));
    std::vector<VkPhysicalDevice> gpus(gpu_count);
    VK_CHECK(vkEnumeratePhysicalDevices(dev->instance, &gpu_count, gpus.data()));

    int selected_gpu = -1;
    for (uint32_t i = 0; i < gpu_count; i++) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(gpus[i], &props);
        if (props.vendorID == 0x1002) {
            selected_gpu = (int)i;
            break;
        }
    }
    if (selected_gpu < 0) {
        RDNA4_ERROR("No AMD GPU (vendor 0x1002) found");
        delete dev;
        return nullptr;
    }

    dev->physical_device = gpus[selected_gpu];
    vkGetPhysicalDeviceProperties(dev->physical_device, &dev->props);
    vkGetPhysicalDeviceMemoryProperties(dev->physical_device, &dev->mem_props);

    VkPhysicalDeviceVulkan11Properties vk11_props = {};
    vk11_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES;

    dev->vulkan12_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES;
    dev->vulkan13_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES;
    dev->vulkan14_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_PROPERTIES;
    dev->subgroup_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;

    VkPhysicalDeviceProperties2 props2 = {};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &vk11_props;
    vk11_props.pNext = &dev->vulkan12_props;
    dev->vulkan12_props.pNext = &dev->vulkan13_props;
    dev->vulkan13_props.pNext = &dev->vulkan14_props;
    dev->vulkan14_props.pNext = &dev->subgroup_props;
    vkGetPhysicalDeviceProperties2(dev->physical_device, &props2);

    uint32_t qf_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(dev->physical_device, &qf_count, nullptr);
    std::vector<VkQueueFamilyProperties> qf_props(qf_count);
    vkGetPhysicalDeviceQueueFamilyProperties(dev->physical_device, &qf_count, qf_props.data());

    RDNA4_LOG("Queue families:");
    for (uint32_t i = 0; i < qf_count; i++) {
        VkQueueFlags flags = qf_props[i].queueFlags;
        RDNA4_LOG("  [%u] compute=%d graphics=%d transfer=%d count=%u",
                  i, !!(flags & VK_QUEUE_COMPUTE_BIT), !!(flags & VK_QUEUE_GRAPHICS_BIT),
                  !!(flags & VK_QUEUE_TRANSFER_BIT), qf_props[i].queueCount);
    }

    // Pass 1: prefer a compute queue family WITHOUT graphics (async compute).
    // This keeps the 3D/graphics engine idle — compute runs on the CUs only.
    for (uint32_t i = 0; i < qf_count; i++) {
        VkQueueFlags flags = qf_props[i].queueFlags;
        bool has_compute   = (flags & VK_QUEUE_COMPUTE_BIT) != 0;
        bool has_graphics  = (flags & VK_QUEUE_GRAPHICS_BIT) != 0;
        bool has_transfer  = (flags & VK_QUEUE_TRANSFER_BIT) != 0;

        if (!has_graphics && has_compute && dev->compute_qf_idx == UINT32_MAX) {
            dev->compute_qf_idx = i;
        }
        if (has_transfer && !has_compute && !has_graphics && dev->transfer_qf_idx == UINT32_MAX) {
            dev->transfer_qf_idx = i;
        }
    }

    // Pass 2: fallback — any family with compute (may include graphics).
    if (dev->compute_qf_idx == UINT32_MAX) {
        for (uint32_t i = 0; i < qf_count; i++) {
            VkQueueFlags flags = qf_props[i].queueFlags;
            if (flags & VK_QUEUE_COMPUTE_BIT) {
                dev->compute_qf_idx = i;
                break;
            }
        }
    }
    if (dev->transfer_qf_idx == UINT32_MAX) {
        for (uint32_t i = 0; i < qf_count; i++) {
            if (i != dev->compute_qf_idx) {
                VkQueueFlags flags = qf_props[i].queueFlags;
                if ((flags & VK_QUEUE_TRANSFER_BIT) && !(flags & VK_QUEUE_COMPUTE_BIT)) {
                    dev->transfer_qf_idx = i;
                    break;
                }
            }
        }
        if (dev->transfer_qf_idx == UINT32_MAX) {
            dev->transfer_qf_idx = dev->compute_qf_idx;
        }
    }

    {
        VkQueueFlags cq = qf_props[dev->compute_qf_idx].queueFlags;
        VkQueueFlags tq = qf_props[dev->transfer_qf_idx].queueFlags;
        RDNA4_LOG("Compute queue family %u (graphics=%d, 3D engine OFF)",
                  dev->compute_qf_idx, !!(cq & VK_QUEUE_GRAPHICS_BIT));
        RDNA4_LOG("Transfer queue family %u (compute=%d)",
                  dev->transfer_qf_idx, !!(tq & VK_QUEUE_COMPUTE_BIT));
    }

    // === Query feature support first, then enable only what's available ===

    if (dev->props.deviceID == 0x7560 || dev->props.deviceID == 0x7550) {
        dev->subgroup_size = RDNA4_SUBGROUP_SIZE;
    } else if (dev->props.deviceID >= 0x73FF && dev->props.deviceID <= 0x7400) {
        dev->subgroup_size = RDNA2_SUBGROUP_SIZE;
    } else {
        dev->subgroup_size = dev->vulkan13_props.minSubgroupSize;
    }

    dev->max_shared_memory = dev->props.limits.maxComputeSharedMemorySize;
    dev->max_workgroup_invocations = dev->props.limits.maxComputeWorkGroupInvocations;

    VkPhysicalDevicePushDescriptorPropertiesKHR push_desc_props = {};
    push_desc_props.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR;
    VkPhysicalDeviceProperties2 push_props2 = {};
    push_props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    push_props2.pNext = &push_desc_props;
    vkGetPhysicalDeviceProperties2(dev->physical_device, &push_props2);
    dev->max_push_descriptors = push_desc_props.maxPushDescriptors;
    dev->has_push_descriptor = (dev->max_push_descriptors >= 6);

    float queue_priority = 1.0f;
    std::vector<VkDeviceQueueCreateInfo> queue_cis;

    VkDeviceQueueCreateInfo compute_qci = {};
    compute_qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    compute_qci.queueFamilyIndex = dev->compute_qf_idx;
    compute_qci.queueCount = 1;
    compute_qci.pQueuePriorities = &queue_priority;
    queue_cis.push_back(compute_qci);

    if (dev->transfer_qf_idx != dev->compute_qf_idx) {
        VkDeviceQueueCreateInfo transfer_qci = {};
        transfer_qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        transfer_qci.queueFamilyIndex = dev->transfer_qf_idx;
        transfer_qci.queueCount = 1;
        transfer_qci.pQueuePriorities = &queue_priority;
        queue_cis.push_back(transfer_qci);
    } else {
        dev->transfer_qf_idx = dev->compute_qf_idx;
    }

    std::vector<const char*> device_extensions = {
        VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME,
    };

    // Query available extensions and only add ones that exist
    uint32_t ext_count = 0;
    vkEnumerateDeviceExtensionProperties(dev->physical_device, nullptr, &ext_count, nullptr);
    std::vector<VkExtensionProperties> avail_exts(ext_count);
    vkEnumerateDeviceExtensionProperties(dev->physical_device, nullptr, &ext_count, avail_exts.data());

    auto has_ext = [&](const char* name) -> bool {
        for (auto& e : avail_exts) {
            if (strcmp(e.extensionName, name) == 0) return true;
        }
        return false;
    };

    // Add extensions only if available
    if (has_ext(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME))
        device_extensions.push_back(VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME);
    if (has_ext(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME))
        device_extensions.push_back(VK_EXT_DESCRIPTOR_INDEXING_EXTENSION_NAME);
    if (has_ext(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME))
        device_extensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    if (has_ext(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME))
        device_extensions.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    if (has_ext(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME))
        device_extensions.push_back(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
    if (has_ext(VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME))
        device_extensions.push_back(VK_EXT_SCALAR_BLOCK_LAYOUT_EXTENSION_NAME);
    if (has_ext(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME))
        device_extensions.push_back(VK_KHR_SHADER_BFLOAT16_EXTENSION_NAME);

    // Only enable features that are actually supported
    // Query features first, then set to true only if supported
    VkPhysicalDeviceFeatures2 query_f2 = {};
    query_f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;

    VkPhysicalDeviceVulkan12Features q12 = {};
    q12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    VkPhysicalDeviceVulkan13Features q13 = {};
    q13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    VkPhysicalDeviceVulkan14Features q14 = {};
    q14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
    VkPhysicalDeviceSubgroupSizeControlFeatures qsg = {};
    qsg.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES;
    VkPhysicalDevice16BitStorageFeatures q16 = {};
    q16.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES;
    VkPhysicalDevice8BitStorageFeatures q8 = {};
    q8.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES;
    VkPhysicalDeviceShaderIntegerDotProductFeatures qdot = {};
    qdot.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;

    query_f2.pNext = &q12;
    q12.pNext = &q13;
    q13.pNext = &q14;
    q14.pNext = &qsg;
    qsg.pNext = &q16;
    q16.pNext = &q8;
    q8.pNext = &qdot;
    vkGetPhysicalDeviceFeatures2(dev->physical_device, &query_f2);

    // Now build the creation-time feature structs (only enable what's supported)
    VkPhysicalDeviceFeatures2 device_features2 = {};
    device_features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    device_features2.features.shaderInt16 = qdot.shaderIntegerDotProduct ? VK_TRUE : VK_FALSE;

    VkPhysicalDeviceVulkan12Features vulkan12_features = {};
    vulkan12_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vulkan12_features.timelineSemaphore = q12.timelineSemaphore;
    vulkan12_features.bufferDeviceAddress = q12.bufferDeviceAddress;
    vulkan12_features.shaderInt8 = q12.shaderInt8;
    vulkan12_features.shaderFloat16 = q12.shaderFloat16;
    vulkan12_features.descriptorIndexing = q12.descriptorIndexing;
    vulkan12_features.scalarBlockLayout = q12.scalarBlockLayout;
    vulkan12_features.hostQueryReset = q12.hostQueryReset;
    vulkan12_features.runtimeDescriptorArray = q12.runtimeDescriptorArray;
    vulkan12_features.shaderStorageBufferArrayNonUniformIndexing = q12.shaderStorageBufferArrayNonUniformIndexing;
    vulkan12_features.descriptorBindingStorageBufferUpdateAfterBind = q12.descriptorBindingStorageBufferUpdateAfterBind;
    vulkan12_features.descriptorBindingPartiallyBound = q12.descriptorBindingPartiallyBound;
    vulkan12_features.descriptorBindingVariableDescriptorCount = q12.descriptorBindingVariableDescriptorCount;
    vulkan12_features.storageBuffer8BitAccess = q8.storageBuffer8BitAccess;
    vulkan12_features.uniformAndStorageBuffer8BitAccess = q8.uniformAndStorageBuffer8BitAccess;

    VkPhysicalDeviceVulkan11Features vulkan11_features = {};
    vulkan11_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    vulkan11_features.storageBuffer16BitAccess = q16.storageBuffer16BitAccess;
    vulkan11_features.uniformAndStorageBuffer16BitAccess = q16.uniformAndStorageBuffer16BitAccess;

    VkPhysicalDeviceVulkan13Features vulkan13_features = {};
    vulkan13_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    vulkan13_features.dynamicRendering = q13.dynamicRendering;
    vulkan13_features.synchronization2 = q13.synchronization2;
    vulkan13_features.maintenance4 = q13.maintenance4;
    vulkan13_features.subgroupSizeControl = qsg.subgroupSizeControl;
    vulkan13_features.computeFullSubgroups = qsg.computeFullSubgroups;

    VkPhysicalDeviceVulkan14Features vulkan14_features = {};
    vulkan14_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
    vulkan14_features.pushDescriptor = q14.pushDescriptor;

    VkPhysicalDeviceShaderIntegerDotProductFeatures int_dot_features = {};
    int_dot_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES;
    int_dot_features.shaderIntegerDotProduct = qdot.shaderIntegerDotProduct;

    device_features2.pNext = &vulkan11_features;
    vulkan11_features.pNext = &vulkan12_features;
    vulkan12_features.pNext = &vulkan13_features;
    vulkan13_features.pNext = &vulkan14_features;
    vulkan14_features.pNext = &int_dot_features;

    // Save feature availability for later use
    dev->has_descriptor_indexing = q12.descriptorIndexing;
    dev->has_timeline_semaphore = q12.timelineSemaphore;
    dev->has_bda = q12.bufferDeviceAddress;
    dev->has_subgroup_size_control = qsg.subgroupSizeControl;
    dev->has_cooperative_matrix = false;
    dev->has_float8 = false;
    dev->has_bf16_coopmat = false;
    dev->has_int8 = q12.shaderInt8;
    dev->has_fp16 = q12.shaderFloat16;
    dev->has_int16 = query_f2.features.shaderInt16;

    VkDeviceCreateInfo device_ci = {};
    device_ci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_ci.pNext = &device_features2;
    device_ci.queueCreateInfoCount = (uint32_t)queue_cis.size();
    device_ci.pQueueCreateInfos = queue_cis.data();
    device_ci.enabledExtensionCount = (uint32_t)device_extensions.size();
    device_ci.ppEnabledExtensionNames = device_extensions.data();
    device_ci.pEnabledFeatures = nullptr;

    VK_CHECK(vkCreateDevice(dev->physical_device, &device_ci, nullptr, &dev->device));

    vkGetDeviceQueue(dev->device, dev->compute_qf_idx, 0, &dev->compute_queue);
    if (dev->transfer_qf_idx != dev->compute_qf_idx) {
        vkGetDeviceQueue(dev->device, dev->transfer_qf_idx, 0, &dev->transfer_queue);
    } else {
        dev->transfer_queue = dev->compute_queue;
    }

    VmaVulkanFunctions vma_vk_fns = {};
    vma_vk_fns.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vma_vk_fns.vkGetDeviceProcAddr = vkGetDeviceProcAddr;
    vma_vk_fns.vkGetPhysicalDeviceProperties = vkGetPhysicalDeviceProperties;
    vma_vk_fns.vkGetPhysicalDeviceMemoryProperties = vkGetPhysicalDeviceMemoryProperties;
    vma_vk_fns.vkAllocateMemory = vkAllocateMemory;
    vma_vk_fns.vkFreeMemory = vkFreeMemory;
    vma_vk_fns.vkMapMemory = vkMapMemory;
    vma_vk_fns.vkUnmapMemory = vkUnmapMemory;
    vma_vk_fns.vkFlushMappedMemoryRanges = vkFlushMappedMemoryRanges;
    vma_vk_fns.vkInvalidateMappedMemoryRanges = vkInvalidateMappedMemoryRanges;
    vma_vk_fns.vkBindBufferMemory = vkBindBufferMemory;
    vma_vk_fns.vkBindImageMemory = vkBindImageMemory;
    vma_vk_fns.vkGetBufferMemoryRequirements = vkGetBufferMemoryRequirements;
    vma_vk_fns.vkGetImageMemoryRequirements = vkGetImageMemoryRequirements;
    vma_vk_fns.vkCreateBuffer = vkCreateBuffer;
    vma_vk_fns.vkDestroyBuffer = vkDestroyBuffer;
    vma_vk_fns.vkCreateImage = vkCreateImage;
    vma_vk_fns.vkDestroyImage = vkDestroyImage;
    vma_vk_fns.vkCmdCopyBuffer = vkCmdCopyBuffer;
    vma_vk_fns.vkGetBufferMemoryRequirements2KHR = vkGetBufferMemoryRequirements2;
    vma_vk_fns.vkGetImageMemoryRequirements2KHR = vkGetImageMemoryRequirements2;
    vma_vk_fns.vkBindBufferMemory2KHR = vkBindBufferMemory2;
    vma_vk_fns.vkBindImageMemory2KHR = vkBindImageMemory2;
    vma_vk_fns.vkGetPhysicalDeviceMemoryProperties2KHR = vkGetPhysicalDeviceMemoryProperties2;

    VmaAllocatorCreateInfo vma_ci = {};
    vma_ci.vulkanApiVersion = api_version;
    vma_ci.instance = dev->instance;
    vma_ci.physicalDevice = dev->physical_device;
    vma_ci.device = dev->device;
    vma_ci.pVulkanFunctions = &vma_vk_fns;
    vma_ci.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT |
                   VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;

    VK_CHECK(vmaCreateAllocator(&vma_ci, &dev->allocator));

    VkCommandPoolCreateInfo compute_cp_ci = {};
    compute_cp_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    compute_cp_ci.queueFamilyIndex = dev->compute_qf_idx;
    compute_cp_ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(dev->device, &compute_cp_ci, nullptr, &dev->compute_cmd_pool));

    VkCommandPoolCreateInfo transfer_cp_ci = {};
    transfer_cp_ci.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    transfer_cp_ci.queueFamilyIndex = dev->transfer_qf_idx;
    transfer_cp_ci.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;
    VK_CHECK(vkCreateCommandPool(dev->device, &transfer_cp_ci, nullptr, &dev->transfer_cmd_pool));

    VkPipelineCacheCreateInfo pcache_ci = {};
    pcache_ci.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    vkCreatePipelineCache(dev->device, &pcache_ci, nullptr, &dev->pipeline_cache);

    RDNA4_LOG("Device creation complete. Features:");
    RDNA4_LOG("  subgroup_size=%u", dev->subgroup_size);
    RDNA4_LOG("  descriptor_indexing=%d", dev->has_descriptor_indexing);
    RDNA4_LOG("  timeline_semaphore=%d", dev->has_timeline_semaphore);
    RDNA4_LOG("  push_descriptor=%d (max=%u)", dev->has_push_descriptor, dev->max_push_descriptors);
    RDNA4_LOG("  cooperative_matrix=%d", dev->has_cooperative_matrix);
    RDNA4_LOG("  bda=%d", dev->has_bda);
    RDNA4_LOG("  int8=%d, fp16=%d, int16=%d", dev->has_int8, dev->has_fp16, dev->has_int16);
    RDNA4_LOG("  max_shared_memory=%u", dev->max_shared_memory);
    RDNA4_LOG("  max_workgroup_invocations=%u", dev->max_workgroup_invocations);
    RDNA4_LOG("  VRAM=%.2f GB", dev->mem_props.memoryHeaps[0].size / (1024.0 * 1024.0 * 1024.0));

    return dev;
}

void vk_device_destroy(vk_device_t* dev) {
    if (!dev) return;

    VK_CHECK(vkDeviceWaitIdle(dev->device));

    if (dev->pipeline_cache != VK_NULL_HANDLE) {
        vkDestroyPipelineCache(dev->device, dev->pipeline_cache, nullptr);
    }

    if (dev->compute_cmd_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(dev->device, dev->compute_cmd_pool, nullptr);
    }
    if (dev->transfer_cmd_pool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(dev->device, dev->transfer_cmd_pool, nullptr);
    }

    if (dev->allocator != VK_NULL_HANDLE) {
        vmaDestroyAllocator(dev->allocator);
    }

    if (dev->device != VK_NULL_HANDLE) {
        vkDestroyDevice(dev->device, nullptr);
    }

    if (dev->debug_messenger != VK_NULL_HANDLE) {
        PFN_vkDestroyDebugUtilsMessengerEXT fn = (PFN_vkDestroyDebugUtilsMessengerEXT)
            vkGetInstanceProcAddr(dev->instance, "vkDestroyDebugUtilsMessengerEXT");
        if (fn) {
            fn(dev->instance, dev->debug_messenger, nullptr);
        }
    }

    if (dev->instance != VK_NULL_HANDLE) {
        vkDestroyInstance(dev->instance, nullptr);
    }

    delete dev;
}
