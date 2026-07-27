#include "adlx_poller.h"
#include "aicompass/logger.h"

namespace aicompass {


// ADLX function pointer types - resolved at runtime via GetProcAddress
// (ADLX_STD_CALL is __stdcall on Windows; we use plain function pointers
//  and cast after GetProcAddress)

bool AdlxGpuPoller::init() {
    if (!load_adlx()) {
        AI_LOG_WARN("ADLX not available, GPU metrics polling disabled");
        return false;
    }
    AI_LOG_INFO("ADLX initialized successfully");
    return true;
}

bool AdlxGpuPoller::load_adlx() {
    // Try loading ADLX from System32 first, then from driver store
    const wchar_t* paths[] = {
        L"amdadlx64.dll",
        L"C:\\Windows\\System32\\amdadlx64.dll",
        L"C:\\Windows\\System32\\DriverStore\\FileRepository\\u0202725.inf_amd64_c5ff89faaab9950b\\B026291\\amdadlx64.dll",
        L"C:\\Windows\\System32\\DriverStore\\FileRepository\\u0420529.inf_amd64_94ad5a6c4d1a04e2\\B419765\\amdadlx64.dll"
    };

    for (auto path : paths) {
        adlx_dll_ = LoadLibraryW(path);
        if (adlx_dll_) {
            AI_LOG_DEBUG("Loaded ADLX from: %S", path);
            return true;
        }
    }
    return false;
}

bool AdlxGpuPoller::start() {
    if (running_) return true;
    if (!adlx_dll_) return false;
    running_ = true;
    poll_thread_ = std::thread(&AdlxGpuPoller::poll_loop, this);
    AI_LOG_INFO("GPU metrics polling started (interval: %dms)", interval_ms_);
    return true;
}

void AdlxGpuPoller::stop() {
    running_ = false;
    if (poll_thread_.joinable()) poll_thread_.join();
    AI_LOG_INFO("GPU metrics polling stopped");
}

void AdlxGpuPoller::poll_loop() {
    while (running_) {
        GpuMetricsSample sample;
        sample.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        if (read_metrics(sample)) {
            last_sample_ = sample;
            if (callback_) callback_(sample);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
    }
}

bool AdlxGpuPoller::read_metrics(GpuMetricsSample& sample) {
    // ADLX provides GPU metrics through IADLXGPUMetrics interface
    // For now, use a simplified poll via AMD's public ADL/ADLX interface
    // If ADLX is not loaded, populate with sentinel values
    if (!adlx_dll_) {
        sample.gpu_utilization_pct = -1.0;
        sample.gpu_core_clock_mhz = -1.0;
        sample.gpu_memory_clock_mhz = -1.0;
        sample.gpu_temperature_c = -1.0;
        sample.gpu_power_w = -1.0;
        sample.gpu_vram_usage_mb = -1.0;
        sample.gpu_vram_total_mb = -1.0;
        sample.gpu_memory_bandwidth_pct = -1.0;
        return false;
    }

    // TODO: Full ADLX integration
    // The ADLX SDK provides:
    // - IADLXGPUMetrics::GetGPUUsage() -> double
    // - IADLXGPUMetrics::GetGPUClockSpeed() -> double
    // - IADLXGPUMetrics::GetGPUVRAMClockSpeed() -> double
    // - IADLXGPUMetrics::GetGPUTemperature() -> double
    // - IADLXGPUMetrics::GetGPUPower() -> double
    // - IADLXGPUMetrics::GetGPUVRAM() -> adlx_int
    // - IADLXGPUMetrics::GetGPUVRAMTotal() -> adlx_int

    // For now return sentinel -1 values to indicate ADLX metrics pending
    sample.gpu_utilization_pct = -1.0;
    sample.gpu_core_clock_mhz = -1.0;
    sample.gpu_memory_clock_mhz = -1.0;
    sample.gpu_temperature_c = -1.0;
    sample.gpu_power_w = -1.0;
    sample.gpu_vram_usage_mb = -1.0;
    sample.gpu_vram_total_mb = -1.0;
    sample.gpu_memory_bandwidth_pct = -1.0;

    return true;
}

} // namespace aicompass
