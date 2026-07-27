#pragma once
#ifndef _WINDOWS_
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef WIN32_LEAN_AND_MEAN
#endif
#include "aicompass/plugin.h"

namespace aicompass {

class HyperloomPlugin : public Plugin {
public:
    std::string name() const override { return "Hyperloom"; }
    std::string description() const override {
        return "Hyperloom performance tuning for RDNA4";
    }

    bool init(const AicompassConfig& config) override;
    bool run(PluginContext& ctx) override;
    void shutdown() override;

private:
    bool hyperloom_available_ = false;
};

} // namespace aicompass
