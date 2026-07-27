#pragma once
#include <windows.h>
#include <string>
#include <thread>
#include <atomic>
#include <functional>
#include <chrono>
#include <vector>

#include "aicompass/types.h"

namespace aicompass {

class AdlxGpuPoller {
public:
    using MetricCallback = std::function<void(const GpuMetricsSample&)>;

    AdlxGpuPoller() : running_(false), interval_ms_(100) {}
    ~AdlxGpuPoller() { stop(); }

    bool init();
    void set_interval(int ms) { interval_ms_ = ms; }
    void set_callback(MetricCallback cb) { callback_ = std::move(cb); }
    bool start();
    void stop();
    bool is_running() const { return running_; }

    const GpuMetricsSample& last_sample() const { return last_sample_; }

private:
    void poll_loop();
    bool load_adlx();
    bool read_metrics(GpuMetricsSample& sample);

    std::atomic<bool> running_;
    std::thread poll_thread_;
    int interval_ms_;
    MetricCallback callback_;
    GpuMetricsSample last_sample_;

    // ADLX handle
    HMODULE adlx_dll_ = nullptr;
    void* adlx_instance_ = nullptr;
    void* gpu_metrics_interface_ = nullptr;
    void* gpu_ = nullptr;
};

} // namespace aicompass
