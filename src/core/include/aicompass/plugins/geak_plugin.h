#pragma once
#ifndef _WINDOWS_
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef WIN32_LEAN_AND_MEAN
#endif
#include "aicompass/plugin.h"

namespace aicompass {

class GeakPlugin : public Plugin {
public:
    std::string name() const override { return "GEAK"; }
    std::string description() const override {
        return "GEAK agent-based optimization for RDNA4 workloads";
    }

    bool init(const AicompassConfig& config) override;
    bool run(PluginContext& ctx) override;
    void shutdown() override;

private:
    bool geak_available_ = false;
};

} // namespace aicompass
