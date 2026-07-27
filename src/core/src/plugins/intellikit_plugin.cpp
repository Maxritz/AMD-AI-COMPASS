#include "aicompass/plugins/intellikit_plugin.h"
#include "aicompass/logger.h"

namespace aicompass {

bool IntellikitPlugin::init(const AicompassConfig&) {
    available_ = GetFileAttributesA("F:\\AMD-Ai\\intellikit\\intellikit.py") != INVALID_FILE_ATTRIBUTES;
    if (available_) AI_LOG_INFO("intellikit plugin initialized");
    return true;
}

bool IntellikitPlugin::run(PluginContext& ctx) {
    if (!available_) return false;
    std::string cmd = "python F:\\AMD-Ai\\intellikit\\intellikit.py --analyze " + ctx.config.output_dir;
    system(cmd.c_str());
    return true;
}

void IntellikitPlugin::shutdown() {}

} // namespace aicompass
