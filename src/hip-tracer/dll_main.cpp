#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>

#include "aicompass/logger.h"

// Forward declare the HIP tracer initialization
extern "C" __declspec(dllexport) void init_hip_tracer();

BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID lpReserved) {
    switch (reason) {
    case DLL_PROCESS_ATTACH: {
        DisableThreadLibraryCalls(hModule);
        aicompass::Logger::instance().set_log_file("ai_hip_tracer.log");
        AI_LOG_INFO("AI-COMPASS HIP Tracer loaded");

        // Initialize ADLX GPU metrics poller
        // Initialize HIP API hooks (MinHook)
        init_hip_tracer();
        break;
    }
    case DLL_PROCESS_DETACH:
        AI_LOG_INFO("AI-COMPASS HIP Tracer unloaded");
        break;
    }
    return TRUE;
}
