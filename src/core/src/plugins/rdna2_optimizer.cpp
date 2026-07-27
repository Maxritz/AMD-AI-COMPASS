#include "aicompass/plugins/rdna2_optimizer.h"
#include "aicompass/logger.h"
#include "aicompass/gpu_info.h"
#include <algorithm>

namespace aicompass {

bool RDNA2OptimizerPlugin::init(const AicompassConfig& config) {
    auto& dev = GpuDetector::instance().primary_device();
    if (!dev.is_rdna2()) {
        AI_LOG_INFO("RDNA2 Optimizer: not applicable (detected %s)",
            GpuDetector::arch_name(dev.arch));
        return true; // Don't fail, just no-op for non-RDNA2
    }
    AI_LOG_INFO("RDNA2 Optimizer initialized for %s (gfx103x, Wave64)",
        dev.name.c_str());
    return true;
}

bool RDNA2OptimizerPlugin::run(PluginContext& ctx) {
    auto& dev = GpuDetector::instance().primary_device();
    if (!dev.is_rdna2()) {
        AI_LOG_INFO("RDNA2 Optimizer: skipped (not RDNA2 hardware)");
        return false;
    }

    AI_LOG_INFO("RDNA2 Optimizer: generating optimizations for %s", dev.name.c_str());
    auto opts = generate_optimizations();

    for (auto& o : opts) {
        AI_LOG_INFO("[RDNA2-Opt] [P%d] %s: %s", o.priority, o.area.c_str(), o.suggestion.c_str());
    }

    // Save optimizations to output directory
    std::string opt_path = ctx.config.output_dir + "/rdna2_optimizations.txt";
    FILE* f = nullptr;
    if (fopen_s(&f, opt_path.c_str(), "w") == 0 && f) {
        fprintf(f, "RDNA2 Optimization Report for %s\n", dev.name.c_str());
        fprintf(f, "Architecture: %s (%s, %d CUs, Wave%d)\n\n",
            dev.name.c_str(), GpuDetector::arch_name(dev.arch),
            dev.num_cu, dev.wave_size);
        fprintf(f, "Optimization Recommendations:\n");
        fprintf(f, "============================\n\n");
        for (auto& o : opts) {
            fprintf(f, "[Priority %d] %s\n", o.priority, o.area.c_str());
            fprintf(f, "  %s\n\n", o.suggestion.c_str());
        }
        fclose(f);
        AI_LOG_INFO("RDNA2 optimizations saved to %s", opt_path.c_str());
    }

    return true;
}

void RDNA2OptimizerPlugin::shutdown() {
    AI_LOG_INFO("RDNA2 Optimizer plugin shutdown");
}

std::vector<RDNA2Opt> RDNA2OptimizerPlugin::generate_optimizations() {
    std::vector<RDNA2Opt> opts;
    opts.push_back(wave64_mmvq_opt());
    opts.push_back(l2_cache_opt());
    opts.push_back(occupancy_opt());
    opts.push_back(wave64_attn_opt());
    opts.push_back(rocm73_opt());
    // Sort by priority descending
    std::sort(opts.begin(), opts.end(),
        [](auto& a, auto& b) { return a.priority > b.priority; });
    return opts;
}

RDNA2Opt RDNA2OptimizerPlugin::wave64_mmvq_opt() {
    return {
        "MMVQ Thread Mapping (Wave64)",
        "RDNA2 uses Wave64 (wavefront size = 64 threads). The default mmvq.cu "
        "thread mapping uses tid/16 which assumes Wave32. For RDNA2, change to "
        "tid/32 (or tid/(qk/qi) like the RDNA4 fix but with /32 instead of /16). "
        "In mmvq.cu: replace 'int start = (tid / 16) * qk/qi' with "
        "'int start = (tid / 32) * qk/qi' to avoid 2x start-position collision within warps.\n"
        "nwarps tuning: prefer nwarps=4 (256 threads) or nwarps=8 (512 threads) "
        "to fully occupy Wave64 SIMDs.",
        5
    };
}

RDNA2Opt RDNA2OptimizerPlugin::l2_cache_opt() {
    return {
        "L2 Cache Pressure (4 MB vs 12 MB on RDNA4)",
        "RDNA2 has 4 MB L2 cache (vs 12 MB on RDNA4). Large models "
        "(30B+) may see significant cache thrashing. Mitigations:\n"
        "  1. Use smaller K-tiles in MMQ (MMQ_ITER_K = 4 instead of 8)\n"
        "  2. Enable Split-K for MMVQ to reduce per-wave L2 footprint\n"
        "  3. Group KV-cache operations into fewer dispatch calls\n"
        "  4. Use Q4_K_M instead of Q4_0 to reduce working set size\n"
        "  5. Consider IQ4_XS for 4-bit with better cache utilization",
        5
    };
}

RDNA2Opt RDNA2OptimizerPlugin::occupancy_opt() {
    return {
        "Occupancy Tuning (40 CUs, Wave64)",
        "RX 6700 XT has 40 CUs x 4 SIMDs x 16 waves = 2560 wave slots.\n"
        "With Wave64, each wave consumes 2x the register/LDS of Wave32.\n"
        "Block size recommendations:\n"
        "  - MMVQ: 256 threads (4 waves/wg) or 128 (2 waves/wg)\n"
        "  - MMQ:  64-128 threads (avoid 32, too few for Wave64)\n"
        "  - Attention: 256 threads minimum\n"
        "  - Elementwise: 512 threads to saturate 40 CUs\n"
        "Check with: python tools/analyze.py trace.csv --arch rdna2",
        4
    };
}

RDNA2Opt RDNA2OptimizerPlugin::wave64_attn_opt() {
    return {
        "Flash Attention for Wave64",
        "Flash attention kernels compiled for Wave32 (RDNA3+) may perform poorly "
        "on RDNA2 Wave64. If using custom flash attention:\n"
        "  - Ensure tile sizes are multiples of 64 (not 32)\n"
        "  - Use __builtin_amdgcn_readfirstlane instead of __ballot for Wave64\n"
        "  - Group-size should be 64 or 128, not 32\n"
        "  - The upstream ROCm 7.3 compiler for gfx1031 handles Wave64 natively",
        3
    };
}

RDNA2Opt RDNA2OptimizerPlugin::rocm73_opt() {
    return {
        "ROCm 7.3 on Windows Compatibility",
        "ROCm 7.3 on Windows 11 with RDNA2 (gfx1031):\n"
        "  - Use --offload-arch=gfx1031 in HIP compilation\n"
        "  - ROCm 7.3 HIP runtime may not support some newer API calls "
        "(hipGetDeviceProperties with gcnArchName might not populate correctly)\n"
        "  - CMake: set -DAMDGPU_TARGETS=gfx1031 -DCMAKE_HIP_ARCHITECTURES=gfx1031\n"
        "  - If hipGetDeviceProperties fails for gcnArchName, manually set "
        "architecture via environment variable HSA_OVERRIDE_GFX_VERSION=10.3.0\n"
        "  - Test with: python tools/run_benchmark.py --model model.gguf --arch rdna2",
        3
    };
}

} // namespace aicompass
