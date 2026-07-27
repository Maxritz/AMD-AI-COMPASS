#include "aicompass/plugins/hyperloom_plugin.h"
#include "aicompass/logger.h"
#include "aicompass/gpu_info.h"

namespace aicompass {

bool HyperloomPlugin::init(const AicompassConfig& config) {
    std::string hyperloom_dir = "F:\\AMD-Ai\\Hyperloom-RDNA";
    hyperloom_available_ = GetFileAttributesA((hyperloom_dir + "\\hyperloom.py").c_str()) != INVALID_FILE_ATTRIBUTES;

    if (hyperloom_available_) {
        AI_LOG_INFO("Hyperloom plugin initialized");
    } else {
        AI_LOG_WARN("Hyperloom not found at %s", hyperloom_dir.c_str());
    }
    return true;
}

bool HyperloomPlugin::run(PluginContext& ctx) {
    if (!hyperloom_available_) return false;

    AI_LOG_INFO("Hyperloom: tuning recommendations based on trace data");

    std::string cmd = "python F:\\AMD-Ai\\Hyperloom-RDNA\\hyperloom.py";
    cmd += " --gpu " + GpuDetector::instance().primary_device().gfx_arch;
    cmd += " --trace " + ctx.config.output_dir;
    cmd += " --output " + ctx.config.output_dir + "/hyperloom_tuning.json";

    system(cmd.c_str());
    return true;
}

void HyperloomPlugin::shutdown() {
    AI_LOG_INFO("Hyperloom plugin shutdown");
}

} // namespace aicompass
