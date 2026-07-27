#pragma once
#ifndef _WINDOWS_
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef WIN32_LEAN_AND_MEAN
#endif
#include "aicompass/plugin.h"
#include <vector>
#include <string>

namespace aicompass {

struct RDNA2Opt {
    std::string area;
    std::string suggestion;
    int priority; // 1-5, 5=highest
};

class RDNA2OptimizerPlugin : public Plugin {
public:
    std::string name() const override { return "RDNA2_OPTIMIZER"; }
    std::string description() const override {
        return "RDNA2 (RX 6000 series) specific optimization recommendations";
    }

    bool init(const AicompassConfig& config) override;
    bool run(PluginContext& ctx) override;
    void shutdown() override;

private:
    std::vector<RDNA2Opt> generate_optimizations();
    RDNA2Opt wave64_mmvq_opt();
    RDNA2Opt l2_cache_opt();
    RDNA2Opt occupancy_opt();
    RDNA2Opt wave64_attn_opt();
    RDNA2Opt rocm73_opt();
};

} // namespace aicompass
