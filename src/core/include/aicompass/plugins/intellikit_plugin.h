#pragma once
#ifndef _WINDOWS_
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#undef WIN32_LEAN_AND_MEAN
#endif
#include "aicompass/plugin.h"

namespace aicompass {

class IntellikitPlugin : public Plugin {
public:
    std::string name() const override { return "intellikit"; }
    std::string description() const override { return "IntelLiKit AI toolkit integration"; }
    bool init(const AicompassConfig& config) override;
    bool run(PluginContext& ctx) override;
    void shutdown() override;
private:
    bool available_ = false;
};

} // namespace aicompass
