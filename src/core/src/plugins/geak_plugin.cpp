#include "aicompass/plugins/geak_plugin.h"
#include "aicompass/logger.h"
#include "aicompass/gpu_info.h"

namespace aicompass {

bool GeakPlugin::init(const AicompassConfig& config) {
    // Check if GEAK is available
    std::string geak_dir = "F:\\AMD-Ai\\GEAK-RDNA";
    geak_available_ = GetFileAttributesA((geak_dir + "\\geak_cli.py").c_str()) != INVALID_FILE_ATTRIBUTES;

    if (geak_available_) {
        AI_LOG_INFO("GEAK plugin initialized (RDNA4 agent mode)");
    } else {
        AI_LOG_WARN("GEAK not found at %s", geak_dir.c_str());
    }
    return true;
}

bool GeakPlugin::run(PluginContext& ctx) {
    if (!geak_available_) return false;

    auto& info = GpuDetector::instance().primary_device();
    AI_LOG_INFO("GEAK analyzing trace on %s (%s)",
        info.name.c_str(), GpuDetector::arch_name(info.arch));

    // Build GEAK command
    std::string cmd = "python F:\\AMD-Ai\\GEAK-RDNA\\geak_cli.py";
    cmd += " --arch " + info.gfx_arch;
    cmd += " --input " + ctx.config.output_dir;
    cmd += " --output " + ctx.config.output_dir + "/geak_report";

    AI_LOG_INFO("GEAK command: %s", cmd.c_str());
    system(cmd.c_str());

    return true;
}

void GeakPlugin::shutdown() {
    AI_LOG_INFO("GEAK plugin shutdown");
}

} // namespace aicompass
