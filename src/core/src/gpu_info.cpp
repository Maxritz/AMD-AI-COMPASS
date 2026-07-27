#include "aicompass/gpu_info.h"
#include "aicompass/logger.h"

namespace aicompass {

bool GpuDetector::detect() {
    devices_.clear();

    // Try HIP first (best GPU info)
    if (query_hip()) {
        auto& dev = primary_device();
        AI_LOG_INFO("GPU detected via HIP: %s (%s, %d CUs, Wave%d, %d MiB VRAM)",
            dev.name.c_str(),
            arch_name(dev.arch),
            dev.num_cu,
            dev.wave_size,
            (int)(dev.total_global_mem / (1024 * 1024)));
        return true;
    }

    // Fall back to ADLX
    if (query_adlx()) {
        AI_LOG_INFO("GPU detected via ADLX: %s (%s)",
            primary_device().name.c_str(),
            arch_name(primary_device().arch));
        return true;
    }

    AI_LOG_WARN("No AMD GPU detected");
    return false;
}

bool GpuDetector::query_hip() {
    // Try to load hip runtime and query devices
    HMODULE hip_mod = LoadLibraryW(L"amdhip64_7.dll");
    if (!hip_mod) return false;

    // Get hipGetDeviceCount and hipGetDeviceProperties
    using hipGetDeviceCount_t = int (*)(int*);
    using hipGetDeviceProperties_t = int (*)(void*, int);
    using hipGetDeviceName_t = const char* (*)(int);

    auto hipGetDeviceCount = (hipGetDeviceCount_t)GetProcAddress(hip_mod, "hipGetDeviceCount");
    auto hipGetDeviceProperties = (hipGetDeviceProperties_t)GetProcAddress(hip_mod, "hipGetDeviceProperties");

    if (!hipGetDeviceCount || !hipGetDeviceProperties) {
        FreeLibrary(hip_mod);
        return false;
    }

    int count = 0;
    if (hipGetDeviceCount(&count) != 0 || count == 0) {
        FreeLibrary(hip_mod);
        return false;
    }

    for (int i = 0; i < count; i++) {
        GpuDeviceInfo info;
        info.device_id = i;
        info.vendor = GpuVendor::AMD;

        // Read GCN architecture name from environment or hipDeviceProp
        // hipDeviceProp_t has gcnArchName in newer ROCm
        struct { char name[256]; size_t totalGlobalMem; int multiProcessorCount;
                 int clockRate; int memoryClockRate; int major; int minor;
                 int maxThreadsPerBlock; int pciBusID;
                 char gcnArchName[256]; int clockInstructionRate; } prop = {};

        if (hipGetDeviceProperties(&prop, i) == 0) {
            info.name = prop.name;
            info.total_global_mem = prop.totalGlobalMem;
            info.num_cu = prop.multiProcessorCount;
            info.clock_rate_khz = prop.clockRate;
            info.mem_clock_rate_khz = prop.memoryClockRate;
            info.compute_capability_major = prop.major;
            info.compute_capability_minor = prop.minor;
            info.max_threads_per_block = prop.maxThreadsPerBlock;
            info.pci_bus_id = prop.pciBusID;

            // gcnArchName gives us the gfx arch string
            std::string gfx = prop.gcnArchName;
            if (!gfx.empty()) {
                info.gfx_arch = gfx;
                info.arch = gfx_to_arch(gfx);
            }

            // Architecture-specific values
            info.num_wgp = info.num_cu / 2;  // Dual CU per WGP on all RDNA
            info.simd_per_cu = 4;
            info.max_waves_per_simd = 16;

            switch (info.arch) {
                case GpuArch::RDNA1:
                case GpuArch::RDNA2:
                    info.wave_size = 64;  // Wave64
                    break;
                case GpuArch::RDNA3:
                case GpuArch::RDNA3_5:
                case GpuArch::RDNA4:
                    info.wave_size = 32;  // Wave32
                    break;
                default:
                    info.wave_size = 64;
                    break;
            }
        }

        devices_.push_back(info);
    }

    FreeLibrary(hip_mod);
    return !devices_.empty();
}

bool GpuDetector::query_adlx() {
    // ADLX fallback: provides GPU name and basic info
    // TODO: Implement ADLX-based GPU info query
    return false;
}

} // namespace aicompass
