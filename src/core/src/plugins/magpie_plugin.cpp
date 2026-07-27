#include "aicompass/plugins/magpie_plugin.h"
#include "aicompass/logger.h"

namespace aicompass {

bool MagpiePlugin::init(const AicompassConfig&) {
    available_ = GetFileAttributesA("F:\\AMD-Ai\\Magpie\\magpie.py") != INVALID_FILE_ATTRIBUTES;
    if (available_) AI_LOG_INFO("Magpie plugin initialized");
    return true;
}

bool MagpiePlugin::run(PluginContext& ctx) {
    if (!available_) return false;
    std::string cmd = "python F:\\AMD-Ai\\Magpie\\magpie.py --input " + ctx.config.output_dir;
    system(cmd.c_str());
    return true;
}

void MagpiePlugin::shutdown() {}

} // namespace aicompass
