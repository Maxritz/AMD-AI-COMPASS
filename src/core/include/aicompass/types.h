#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <chrono>
#include <unordered_map>

namespace aicompass {

enum class Phase {
    UNKNOWN = 0,
    PROMPT_PROCESSING,
    TOKEN_GENERATION
};

struct KernelRecord {
    uint64_t dispatch_id;
    std::string kernel_name;
    uint32_t grid_x, grid_y, grid_z;
    uint32_t block_x, block_y, block_z;
    uint32_t shared_mem;
    double duration_us;
    int64_t timestamp_us;
    Phase phase;
};

struct GpuMetricsSample {
    int64_t timestamp_us;
    double gpu_utilization_pct;
    double gpu_core_clock_mhz;
    double gpu_memory_clock_mhz;
    double gpu_temperature_c;
    double gpu_power_w;
    double gpu_vram_usage_mb;
    double gpu_vram_total_mb;
    double gpu_memory_bandwidth_pct;
};

struct HardwareCounterSample {
    int64_t timestamp_us;
    uint64_t dispatch_id;
    std::string counter_name;
    uint64_t value;
    std::string block; // SQ, SPI, GRBM, TCP, etc.
    uint32_t block_instance;
};

struct PerUnitUtilization {
    double valu_util_pct;
    double valu_mfma_util_pct;
    double salu_util_pct;
    double lds_util_pct;
    double vmem_util_pct;
    double scalar_util_pct;
    double export_util_pct;
    double ta_util_pct;
    double td_util_pct;
    double tcp_util_pct;
    double gl2c_hit_rate_pct;
    double wave_occupancy_pct;
};

struct UtilizationSummary {
    Phase phase;
    double avg_gpu_util_pct;
    double avg_core_clock_mhz;
    double avg_memory_clock_mhz;
    double avg_power_w;
    double avg_vram_usage_mb;
    PerUnitUtilization per_unit;
    int64_t wall_time_ms;
    int64_t kernel_count;
    double total_kernel_time_ms;
};

struct AicompassConfig {
    std::string output_dir = ".";
    std::string hip_trace_csv = "hip_trace.csv";
    std::string gpu_metrics_csv = "gpu_metrics.csv";
    std::string counters_csv = "hw_counters.csv";
    bool collect_gpu_metrics = true;
    bool collect_hw_counters = false;
    int metrics_poll_interval_ms = 100;
    bool verbose = false;
    std::string target_app;
    std::string target_args;
};

struct TraceResult {
    std::vector<KernelRecord> kernels;
    std::vector<GpuMetricsSample> metrics;
    std::vector<HardwareCounterSample> hw_counters;
    UtilizationSummary pp_summary;
    UtilizationSummary tg_summary;

    void clear() {
        kernels.clear();
        metrics.clear();
        hw_counters.clear();
    }
};

} // namespace aicompass
