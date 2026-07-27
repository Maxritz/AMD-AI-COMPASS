#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>

#include "aicompass/logger.h"

// Stub: real init is in the original hip_tracer.cpp (MinHook setup)
// Linked from F:\AMD-Ai\hip_tracer\build\hip_tracer.lib when available
extern "C" __declspec(dllimport) void init_hip_tracer();

BOOL APIENTRY DllMain(HMODULE hModule, DWORD reason, LPVOID lpReserved) {
    switch (reason) {
    case DLL_PROCESS_ATTACH: {
        DisableThreadLibraryCalls(hModule);
        aicompass::Logger::instance().set_log_file("ai_hip_tracer.log");
        AI_LOG_INFO("AI-COMPASS HIP Tracer loaded");
        AI_LOG_INFO("AI-COMPASS v0.1 — RDNA AI Compute Performance Toolset");
        break;
    }
    case DLL_PROCESS_DETACH:
        AI_LOG_INFO("AI-COMPASS HIP Tracer unloaded");
        break;
    }
    return TRUE;
}
