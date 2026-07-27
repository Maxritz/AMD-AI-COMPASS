#pragma once
#ifndef _WINDOWS_
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef WIN32_LEAN_AND_MEAN
#endif
#include <cstdint>
#include <string>
#include <vector>

namespace aicompass {

enum class GpuArch {
    UNKNOWN = 0,
    RDNA1,    // gfx1010-1012 (RX 5000 series)
    RDNA2,    // gfx1030-1035 (RX 6000 series)
    RDNA3,    // gfx1100-1103 (RX 7000 series)
    RDNA3_5,  // gfx1150-1151 (RX 8000 series)
    RDNA4     // gfx1200-1201 (RX 9000 series)
};

enum class GpuVendor {
    UNKNOWN = 0,
    AMD,
    NVIDIA,
    INTEL
};

struct GpuDeviceInfo {
    GpuVendor vendor = GpuVendor::UNKNOWN;
    GpuArch arch = GpuArch::UNKNOWN;
    std::string name;
    std::string gfx_arch;  // e.g., "gfx1201"
    uint32_t compute_capability_major = 0;
    uint32_t compute_capability_minor = 0;
    int device_id = -1;
    size_t total_global_mem = 0;
    int num_cu = 0;       // Compute Units (WGP on RDNA)
    int num_wgp = 0;      // Work Group Processors
    int simd_per_cu = 0;
    int max_waves_per_simd = 0;
    int max_threads_per_block = 0;
    int clock_rate_khz = 0;
    int mem_clock_rate_khz = 0;
    int pci_bus_id = 0;
    bool is_large_bar = false;

    bool is_amd() const { return vendor == GpuVendor::AMD; }
    bool is_rdna() const { return arch >= GpuArch::RDNA1 && arch <= GpuArch::RDNA4; }
    bool is_rdna4() const { return arch == GpuArch::RDNA4; }
};

class GpuDetector {
public:
    static GpuDetector& instance() {
        static GpuDetector inst;
        return inst;
    }

    bool detect();
    int device_count() const { return (int)devices_.size(); }
    const GpuDeviceInfo& primary_device() const { return devices_.empty() ? fallback_ : devices_[0]; }
    const std::vector<GpuDeviceInfo>& all_devices() const { return devices_; }

    static const char* arch_name(GpuArch a) {
        switch (a) {
            case GpuArch::RDNA1:  return "RDNA1";
            case GpuArch::RDNA2:  return "RDNA2";
            case GpuArch::RDNA3:  return "RDNA3";
            case GpuArch::RDNA3_5: return "RDNA3.5";
            case GpuArch::RDNA4:  return "RDNA4";
            default:              return "Unknown";
        }
    }

    static GpuArch gfx_to_arch(const std::string& gfx) {
        if (gfx.find("gfx120") != std::string::npos) return GpuArch::RDNA4;
        if (gfx.find("gfx115") != std::string::npos) return GpuArch::RDNA3_5;
        if (gfx.find("gfx110") != std::string::npos) return GpuArch::RDNA3;
        if (gfx.find("gfx103") != std::string::npos) return GpuArch::RDNA2;
        if (gfx.find("gfx101") != std::string::npos) return GpuArch::RDNA1;
        return GpuArch::UNKNOWN;
    }

private:
    GpuDetector() = default;
    bool query_hip();
    bool query_adlx();

    std::vector<GpuDeviceInfo> devices_;
    GpuDeviceInfo fallback_;
};

} // namespace aicompass
