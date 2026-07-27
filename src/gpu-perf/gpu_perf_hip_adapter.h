#pragma once
#include <string>
#include <vector>
#include <thread>
#include <atomic>
#include <functional>
#include <chrono>
#include <memory>

#include "aicompass/types.h"
#include "aicompass/gpu_info.h"

namespace aicompass {

struct PerfCounterConfig {
    std::string name;
    std::string block;
    uint32_t block_id;
    uint32_t counter_index;
    uint32_t event_id;
};

class GpuPerfHipAdapter {
public:
    using CounterCallback = std::function<void(const std::vector<HardwareCounterSample>&)>;

    GpuPerfHipAdapter();
    ~GpuPerfHipAdapter();

    bool init(const GpuDeviceInfo& device_info);
    bool start_sampling(int interval_ms = 200);
    void stop_sampling();
    bool is_sampling() const { return sampling_; }

    bool begin_capture();
    bool end_capture(std::vector<HardwareCounterSample>& results);

    const std::vector<PerfCounterConfig>& available_counters() const { return counters_; }
    bool is_initialized() const { return initialized_; }

private:
    bool create_vulkan_context();
    bool setup_gpu_perf_counters();
    bool load_gpu_perf_api();
    bool read_counters(std::vector<HardwareCounterSample>& samples);
    void cleanup();

    void sampling_loop();

    std::atomic<bool> initialized_{false};
    std::atomic<bool> sampling_{false};
    std::thread sampler_thread_;
    int interval_ms_{200};

    void* gpa_lib_ = nullptr;
    void* gpa_func_table_ = nullptr;
    void* gpa_context_ = nullptr;
    void* gpa_session_ = nullptr;
    std::vector<PerfCounterConfig> counters_;
    GpuDeviceInfo device_;

    void* vk_instance_ = nullptr;
    void* vk_physical_device_ = nullptr;
    void* vk_device_ = nullptr;
};

} // namespace aicompass
