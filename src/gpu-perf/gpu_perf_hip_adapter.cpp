#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif

#include "gpu_perf_hip_adapter.h"
#include "aicompass/logger.h"

#include "gpu_performance_api/gpu_perf_api.h"
#include "gpu_performance_api/gpu_perf_api_vk.h"

#include <vulkan/vulkan.h>

namespace aicompass {

#define ARRAY_SIZE(x) (sizeof(x) / sizeof(x[0]))

static const char* VK_INSTANCE_EXTENSIONS[] = {
    VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME,
};

static const char* VK_DEVICE_EXTENSIONS[] = {
    VK_AMD_GPA_INTERFACE_EXTENSION_NAME,
    VK_AMD_SHADER_CORE_PROPERTIES_EXTENSION_NAME,
};

GpuPerfHipAdapter::GpuPerfHipAdapter() {}
GpuPerfHipAdapter::~GpuPerfHipAdapter() {
    stop_sampling();
    cleanup();
}

void GpuPerfHipAdapter::cleanup() {
    auto table = reinterpret_cast<GpaFunctionTable*>(gpa_func_table_);

    if (gpa_session_) {
        if (table) {
            table->GpaAbortSession((GpaSessionId)gpa_session_);
            table->GpaDeleteSession((GpaSessionId)gpa_session_);
        }
        gpa_session_ = nullptr;
    }

    if (gpa_context_ && table) {
        table->GpaCloseContext((GpaContextId)gpa_context_);
        gpa_context_ = nullptr;
    }

    if (table) {
        table->GpaDestroy();
        delete table;
        gpa_func_table_ = nullptr;
    }

    if (gpa_lib_) {
        FreeLibrary((HMODULE)gpa_lib_);
        gpa_lib_ = nullptr;
    }

    if (vk_device_) {
        vkDestroyDevice((VkDevice)vk_device_, nullptr);
        vk_device_ = nullptr;
    }

    if (vk_instance_) {
        vkDestroyInstance((VkInstance)vk_instance_, nullptr);
        vk_instance_ = nullptr;
    }

    vk_physical_device_ = nullptr;
    initialized_ = false;
}

bool GpuPerfHipAdapter::init(const GpuDeviceInfo& device_info) {
    device_ = device_info;
    AI_LOG_INFO("Initializing GPUPerfAPI HIP adapter for %s (%s)",
        device_info.name.c_str(), GpuDetector::arch_name(device_info.arch));

    if (!create_vulkan_context()) return false;
    if (!load_gpu_perf_api()) return false;
    if (!setup_gpu_perf_counters()) return false;

    initialized_ = true;
    AI_LOG_INFO("GPUPerfAPI HIP adapter initialized with %zu counters", counters_.size());
    return true;
}

bool GpuPerfHipAdapter::create_vulkan_context() {
    VkApplicationInfo app_info = {};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "AI-COMPASS";
    app_info.applicationVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.pEngineName = "AI-COMPASS";
    app_info.engineVersion = VK_MAKE_VERSION(0, 1, 0);
    app_info.apiVersion = VK_API_VERSION_1_3;

    VkInstanceCreateInfo inst_info = {};
    inst_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    inst_info.pApplicationInfo = &app_info;
    inst_info.enabledExtensionCount = (uint32_t)ARRAY_SIZE(VK_INSTANCE_EXTENSIONS);
    inst_info.ppEnabledExtensionNames = VK_INSTANCE_EXTENSIONS;

    VkInstance instance = VK_NULL_HANDLE;
    VkResult res = vkCreateInstance(&inst_info, nullptr, &instance);
    if (res != VK_SUCCESS || !instance) {
        AI_LOG_ERROR("Failed to create Vulkan instance (%d)", res);
        return false;
    }
    vk_instance_ = (void*)instance;

    uint32_t device_count = 0;
    vkEnumeratePhysicalDevices(instance, &device_count, nullptr);
    std::vector<VkPhysicalDevice> phys_devices(device_count);
    vkEnumeratePhysicalDevices(instance, &device_count, phys_devices.data());

    VkPhysicalDevice selected = VK_NULL_HANDLE;
    for (auto pd : phys_devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pd, &props);
        if (props.vendorID == 0x1002) {
            selected = pd;
            AI_LOG_INFO("Selected Vulkan device: %s (AMD)", props.deviceName);
            break;
        }
    }

    if (!selected) {
        AI_LOG_ERROR("No AMD Vulkan device found");
        vkDestroyInstance(instance, nullptr);
        vk_instance_ = nullptr;
        return false;
    }
    vk_physical_device_ = (void*)selected;

    float queue_priority = 1.0f;
    VkDeviceQueueCreateInfo q_info = {};
    q_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    q_info.queueFamilyIndex = 0;
    q_info.queueCount = 1;
    q_info.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo dev_info = {};
    dev_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    dev_info.queueCreateInfoCount = 1;
    dev_info.pQueueCreateInfos = &q_info;
    dev_info.enabledExtensionCount = (uint32_t)ARRAY_SIZE(VK_DEVICE_EXTENSIONS);
    dev_info.ppEnabledExtensionNames = VK_DEVICE_EXTENSIONS;

    VkDevice device = VK_NULL_HANDLE;
    res = vkCreateDevice(selected, &dev_info, nullptr, &device);
    if (res != VK_SUCCESS || !device) {
        AI_LOG_ERROR("Failed to create Vulkan device with GPA extension (%d)", res);
        vkDestroyInstance(instance, nullptr);
        vk_instance_ = nullptr;
        return false;
    }
    vk_device_ = (void*)device;

    AI_LOG_INFO("Vulkan context created for GPUPerfAPI");
    return true;
}

bool GpuPerfHipAdapter::load_gpu_perf_api() {
    gpa_lib_ = (void*)LoadLibraryW(L"GPUPerfAPIVK-x64.dll");
    if (!gpa_lib_) {
        gpa_lib_ = (void*)LoadLibraryW(L"third_party\\gpu_perf_api\\build\\output\\release_x64\\GPUPerfAPIVK-x64.dll");
    }
    if (!gpa_lib_) {
        AI_LOG_ERROR("Failed to load GPUPerfAPIVK-x64.dll");
        return false;
    }
    AI_LOG_INFO("GPUPerfAPI VK DLL loaded");

    auto GpaGetFuncTable = (GpaGetFuncTablePtrType)GetProcAddress((HMODULE)gpa_lib_, "GpaGetFuncTable");
    if (!GpaGetFuncTable) {
        AI_LOG_ERROR("GpaGetFuncTable not found in GPUPerfAPI DLL");
        return false;
    }

    auto table = std::make_unique<GpaFunctionTable>();
    table->major_version = GPA_FUNCTION_TABLE_MAJOR_VERSION_NUMBER;
    table->minor_version = GPA_FUNCTION_TABLE_MINOR_VERSION_NUMBER;

    GpaStatus status = GpaGetFuncTable((void*)table.get());
    if (status != kGpaStatusOk) {
        AI_LOG_ERROR("GpaGetFuncTable failed (%d)", status);
        return false;
    }

    gpa_func_table_ = (void*)table.release();

    auto gpa = reinterpret_cast<GpaFunctionTable*>(gpa_func_table_);

    status = gpa->GpaInitialize(kGpaInitializeDefaultBit);
    if (status != kGpaStatusOk) {
        AI_LOG_ERROR("GpaInitialize failed (%d)", status);
        return false;
    }

    GpaVkContextOpenInfo vk_info = {};
    vk_info.instance = (VkInstance)vk_instance_;
    vk_info.physical_device = (VkPhysicalDevice)vk_physical_device_;
    vk_info.device = (VkDevice)vk_device_;

    status = gpa->GpaOpenContext(&vk_info, kGpaOpenContextDefaultBit, (GpaContextId*)&gpa_context_);
    if (status != kGpaStatusOk) {
        AI_LOG_ERROR("GpaOpenContext failed (%d)", status);
        return false;
    }

    GpaContextSampleTypeFlags sample_types = 0;
    gpa->GpaGetSupportedSampleTypes((GpaContextId)gpa_context_, &sample_types);
    if (!(sample_types & kGpaContextSampleTypeDiscreteCounter)) {
        AI_LOG_ERROR("Discrete counters not supported");
        return false;
    }

    status = gpa->GpaCreateSession((GpaContextId)gpa_context_, kGpaSessionSampleTypeDiscreteCounter, (GpaSessionId*)&gpa_session_);
    if (status != kGpaStatusOk) {
        AI_LOG_ERROR("GpaCreateSession failed (%d)", status);
        return false;
    }

    AI_LOG_INFO("GPUPerfAPI session created");
    return true;
}

bool GpuPerfHipAdapter::setup_gpu_perf_counters() {
    auto gpa = reinterpret_cast<GpaFunctionTable*>(gpa_func_table_);
    if (!gpa || !gpa_session_) return false;

    GpaUInt32 num_counters = 0;
    gpa->GpaGetNumCounters((GpaSessionId)gpa_session_, &num_counters);
    AI_LOG_INFO("GPUPerfAPI: %u counters available", num_counters);

    for (GpaUInt32 i = 0; i < num_counters; i++) {
        const char* name = nullptr;
        gpa->GpaGetCounterName((GpaSessionId)gpa_session_, i, &name);
        if (!name) continue;

        std::string n(name);

        bool useful = false;
        if (n.find("SQ_VALU") != std::string::npos) useful = true;
        else if (n.find("SQ_SALU") != std::string::npos) useful = true;
        else if (n.find("SQ_LDS") != std::string::npos) useful = true;
        else if (n.find("SQ_VMEM") != std::string::npos) useful = true;
        else if (n.find("SQ_WAVES") != std::string::npos) useful = true;
        else if (n.find("SQ_BUSY") != std::string::npos) useful = true;
        else if (n.find("SQ_INST") != std::string::npos) useful = true;
        else if (n.find("SPI_WGP_ACTIVE") != std::string::npos) useful = true;
        else if (n.find("GRBM_GUI_ACTIVE") != std::string::npos) useful = true;
        else if (n.find("TCP_") != std::string::npos && (
            n.find("HIT") != std::string::npos || n.find("MISS") != std::string::npos)) useful = true;
        else if (n.find("GL2C_") != std::string::npos && (
            n.find("HIT") != std::string::npos || n.find("MISS") != std::string::npos)) useful = true;

        if (useful) {
            gpa->GpaEnableCounter((GpaSessionId)gpa_session_, i);

            PerfCounterConfig cfg;
            cfg.name = n;
            cfg.counter_index = i;

            if (n.find("SQ_") != std::string::npos) cfg.block = "SQ";
            else if (n.find("SPI_") != std::string::npos) cfg.block = "SPI";
            else if (n.find("GRBM_") != std::string::npos) cfg.block = "GRBM";
            else if (n.find("TCP_") != std::string::npos) cfg.block = "TCP";
            else if (n.find("GL2C_") != std::string::npos) cfg.block = "GL2C";

            counters_.push_back(cfg);
        }
    }

    AI_LOG_INFO("Enabled %zu counters for per-unit GPU utilization", counters_.size());
    return !counters_.empty();
}

bool GpuPerfHipAdapter::begin_capture() {
    if (!gpa_session_) return false;
    auto gpa = reinterpret_cast<GpaFunctionTable*>(gpa_func_table_);
    GpaStatus status = gpa->GpaBeginSession((GpaSessionId)gpa_session_);
    return status == kGpaStatusOk;
}

bool GpuPerfHipAdapter::end_capture(std::vector<HardwareCounterSample>& results) {
    if (!gpa_session_) return false;
    auto gpa = reinterpret_cast<GpaFunctionTable*>(gpa_func_table_);

    GpaStatus status = gpa->GpaEndSession((GpaSessionId)gpa_session_);
    if (status != kGpaStatusOk) {
        AI_LOG_ERROR("GpaEndSession failed (%d)", status);
        return false;
    }

    GpaUInt32 sample_count = 0;
    gpa->GpaGetSampleCount((GpaSessionId)gpa_session_, &sample_count);

    for (GpaUInt32 s = 0; s < sample_count; s++) {
        GpaUInt32 sample_id = 0;
        gpa->GpaGetSampleId((GpaSessionId)gpa_session_, s, &sample_id);

        size_t result_size = 0;
        gpa->GpaGetSampleResultSize((GpaSessionId)gpa_session_, sample_id, &result_size);

        if (result_size == 0) continue;

        std::vector<uint8_t> buffer(result_size);
        gpa->GpaGetSampleResult((GpaSessionId)gpa_session_, sample_id, result_size, buffer.data());

        GpaUInt32 num_enabled = 0;
        gpa->GpaGetNumEnabledCounters((GpaSessionId)gpa_session_, &num_enabled);

        GpaUInt64* values = reinterpret_cast<GpaUInt64*>(buffer.data());
        GpaUInt32 value_count = std::min(num_enabled, (GpaUInt32)counters_.size());

        for (GpaUInt32 v = 0; v < value_count; v++) {
            GpaUInt32 counter_idx = 0;
            gpa->GpaGetEnabledIndex((GpaSessionId)gpa_session_, v, &counter_idx);

            if (counter_idx >= counters_.size()) continue;

            HardwareCounterSample sample;
            sample.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::high_resolution_clock::now().time_since_epoch()).count();
            sample.counter_name = counters_[counter_idx].name;
            sample.value = values[v];
            sample.block = counters_[counter_idx].block;
            results.push_back(sample);
        }
    }

    return true;
}

bool GpuPerfHipAdapter::start_sampling(int interval_ms) {
    if (!initialized_) return false;
    interval_ms_ = interval_ms;
    sampling_ = true;
    sampler_thread_ = std::thread(&GpuPerfHipAdapter::sampling_loop, this);
    AI_LOG_INFO("GPU counter sampling started (interval: %dms)", interval_ms_);
    return true;
}

void GpuPerfHipAdapter::stop_sampling() {
    sampling_ = false;
    if (sampler_thread_.joinable()) sampler_thread_.join();
}

void GpuPerfHipAdapter::sampling_loop() {
    while (sampling_) {
        std::vector<HardwareCounterSample> samples;

        if (begin_capture()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_ / 2));
            end_capture(samples);
        }

        if (!samples.empty()) {
            AI_LOG_DEBUG("Captured %zu counter samples", samples.size());
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_ / 2));
    }
}

bool GpuPerfHipAdapter::read_counters(std::vector<HardwareCounterSample>& samples) {
    return end_capture(samples);
}

} // namespace aicompass
