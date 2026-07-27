#pragma once
#include <windows.h>
#include <string>
#include <thread>
#include <atomic>
#include <functional>
#include <chrono>
#include <vector>
#include <memory>

#include "aicompass/types.h"

namespace aicompass {

class AdlxGpuPoller {
public:
    using MetricCallback = std::function<void(const GpuMetricsSample&)>;

    AdlxGpuPoller() : running_(false), interval_ms_(100), initialized_(false) {}
    ~AdlxGpuPoller();

    bool init();
    void set_interval(int ms) { interval_ms_ = ms; }
    void set_callback(MetricCallback cb) { callback_ = std::move(cb); }
    bool start();
    void stop();
    bool is_running() const { return running_; }
    bool is_initialized() const { return initialized_; }
    const GpuMetricsSample& last_sample() const { return last_sample_; }

private:
    void poll_loop();
    bool load_adlx();
    bool read_metrics(GpuMetricsSample& sample);

    std::atomic<bool> running_;
    std::atomic<bool> initialized_;
    std::thread poll_thread_;
    int interval_ms_;
    MetricCallback callback_;
    GpuMetricsSample last_sample_;
    void* gpu_handle_ = nullptr; // IADLXGPUPtr stored as void* to avoid ADLX headers in header
};

} // namespace aicompass
