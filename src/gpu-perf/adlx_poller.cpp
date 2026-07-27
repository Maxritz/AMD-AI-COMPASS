#include "adlx_poller.h"
#include "aicompass/logger.h"

// ADLX SDK headers
#include "ADLX.h"
#include "IPerformanceMonitoring.h"
#include "IPerformanceMonitoring1.h"
#include "ISystem.h"
#include "ADLXHelper.h"

using namespace adlx;

namespace aicompass {

static ADLXHelper g_ADLXHelp;
static IADLXGPUPtr g_gpu;

bool AdlxGpuPoller::init() {
    if (!load_adlx()) {
        AI_LOG_WARN("ADLX DLL not loadable, GPU metrics polling disabled");
        return false;
    }

    ADLX_RESULT res = g_ADLXHelp.Initialize();
    if (ADLX_FAILED(res)) {
        AI_LOG_WARN("ADLX initialization failed (%d)", res);
        return false;
    }

    IADLXPerformanceMonitoringServicesPtr perfService;
    res = g_ADLXHelp.GetSystemServices()->GetPerformanceMonitoringServices(&perfService);
    if (ADLX_FAILED(res)) {
        AI_LOG_WARN("ADLX perf monitoring services unavailable");
        return false;
    }

    IADLXGPUListPtr gpus;
    res = g_ADLXHelp.GetSystemServices()->GetGPUs(&gpus);
    if (ADLX_FAILED(res) || gpus->Begin() == gpus->End()) {
        AI_LOG_WARN("No AMD GPUs found via ADLX");
        return false;
    }

    res = gpus->At(gpus->Begin(), &g_gpu);
    if (ADLX_FAILED(res)) {
        AI_LOG_WARN("Failed to get primary GPU from ADLX");
        return false;
    }

    const char* gpuName = nullptr;
    g_gpu->Name(&gpuName);
    AI_LOG_INFO("ADLX GPU metrics initialized: %s", gpuName ? gpuName : "unknown");
    initialized_ = true;
    return true;
}

bool AdlxGpuPoller::load_adlx() {
    HMODULE test = LoadLibraryW(L"amdadlx64.dll");
    if (test) { FreeLibrary(test); return true; }
    AI_LOG_DEBUG("ADLX DLL not in system path, trying third_party/");
    test = LoadLibraryW(L"third_party\\amdadlx64.dll");
    if (test) { FreeLibrary(test); return true; }
    return false;
}

bool AdlxGpuPoller::start() {
    if (!initialized_) return false;
    if (running_) return true;
    running_ = true;
    poll_thread_ = std::thread(&AdlxGpuPoller::poll_loop, this);
    AI_LOG_INFO("GPU metrics polling started (interval: %dms)", interval_ms_);
    return true;
}

void AdlxGpuPoller::stop() {
    running_ = false;
    if (poll_thread_.joinable()) poll_thread_.join();
}

void AdlxGpuPoller::poll_loop() {
    IADLXPerformanceMonitoringServicesPtr perfService;
    ADLX_RESULT res = g_ADLXHelp.GetSystemServices()->GetPerformanceMonitoringServices(&perfService);
    if (ADLX_FAILED(res)) {
        AI_LOG_ERROR("Failed to get ADLX perf services in poller thread");
        return;
    }

    while (running_) {
        GpuMetricsSample sample;
        sample.timestamp_us = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count();

        IADLXGPUMetricsPtr metrics;
        res = perfService->GetCurrentGPUMetrics(g_gpu, &metrics);
        if (ADLX_SUCCEEDED(res)) {
            adlx_double dval = 0;
            adlx_int ival = 0;

            if (ADLX_SUCCEEDED(metrics->GPUUsage(&dval)))
                sample.gpu_utilization_pct = dval;
            if (ADLX_SUCCEEDED(metrics->GPUClockSpeed(&ival)))
                sample.gpu_core_clock_mhz = (double)ival;
            if (ADLX_SUCCEEDED(metrics->GPUVRAMClockSpeed(&ival)))
                sample.gpu_memory_clock_mhz = (double)ival;
            if (ADLX_SUCCEEDED(metrics->GPUTemperature(&dval)))
                sample.gpu_temperature_c = dval;
            if (ADLX_SUCCEEDED(metrics->GPUPower(&dval)))
                sample.gpu_power_w = dval;
            if (ADLX_SUCCEEDED(metrics->GPUVRAM(&ival)))
                sample.gpu_vram_usage_mb = (double)ival;
            sample.gpu_vram_total_mb = 16384; // RX 9070 XT has 16GB VRAM

            last_sample_ = sample;
            if (callback_) callback_(sample);
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms_));
    }
}

bool AdlxGpuPoller::read_metrics(GpuMetricsSample&) {
    return false; // not used; inline in poll_loop
}

AdlxGpuPoller::~AdlxGpuPoller() {
    stop();
    g_ADLXHelp.Terminate();
}

} // namespace aicompass
