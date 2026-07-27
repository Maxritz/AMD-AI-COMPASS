#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>
#include <string>
#include <vector>
#include <map>

#include "aicompass/logger.h"
#include "aicompass/plugin.h"
#include "aicompass/pipeline.h"

using namespace aicompass;

static void print_banner() {
    printf(R"(
    ╔══════════════════════════════════════════════════════════╗
    ║             AI-COMPASS v0.1                             ║
    ║   AI Compute Performance Analysis & Statistics Suite    ║
    ║         RDNA4 AI Compute Toolset — Windows              ║
    ╚══════════════════════════════════════════════════════════╝

)");
}

static void print_usage(const char* prog) {
    printf("Usage: %s <command> [options]\n\n", prog);
    printf("Commands:\n");
    printf("  trace    <app> [args]   Capture HIP kernel trace + GPU metrics\n");
    printf("  profile  <app> [args]   Full profile with all plugins\n");
    printf("  analyze  <dir>          Post-process trace data\n");
    printf("  list                    List installed plugins\n");
    printf("  help                    Show this help\n");
    printf("\nOptions:\n");
    printf("  -o, --output <dir>     Output directory\n");
    printf("  -i, --interval <ms>    Metrics poll interval (default: 100)\n");
    printf("  -v, --verbose          Verbose output\n");
    printf("  --counters             Enable hardware counter collection\n");
    printf("\nExamples:\n");
    printf("  aicompass trace -o trace_out -- llama-cli -m model.gguf -p \"hello\"\n");
    printf("  aicompass profile --counters -- llama-bench -m model.gguf -n 256\n");
}

int main(int argc, char* argv[]) {
    print_banner();

    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    std::string command = argv[1];

    if (command == "help" || command == "--help" || command == "-h") {
        print_usage(argv[0]);
        return 0;
    }

    if (command == "list") {
        printf("Registered plugins (%zu):\n", PluginRegistry::instance().count());
        for (auto* p : PluginRegistry::instance().all()) {
            printf("  %s — %s\n", p->name().c_str(), p->description().c_str());
        }
        return 0;
    }

    AicompassConfig config;
    std::vector<std::string> app_args;

    // Parse common options
    for (int i = 2; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) config.output_dir = argv[++i];
        } else if (arg == "-i" || arg == "--interval") {
            if (i + 1 < argc) config.metrics_poll_interval_ms = atoi(argv[++i]);
        } else if (arg == "-v" || arg == "--verbose") {
            config.verbose = true;
            Logger::instance().set_verbose(true);
        } else if (arg == "--counters") {
            config.collect_hw_counters = true;
        } else if (arg == "--") {
            // Remaining args are the target app
            for (int j = i + 1; j < argc; j++)
                app_args.push_back(argv[j]);
            break;
        } else {
            app_args.push_back(arg);
        }
    }

    if (app_args.empty() && (command == "trace" || command == "profile")) {
        AI_LOG_ERROR("No target application specified");
        return 1;
    }

    config.target_app = app_args.empty() ? "" : app_args[0];
    for (size_t i = 1; i < app_args.size(); i++) {
        if (!config.target_args.empty()) config.target_args += " ";
        config.target_args += app_args[i];
    }

    AI_LOG_INFO("Command: %s", command.c_str());
    AI_LOG_INFO("Target:  %s %s", config.target_app.c_str(), config.target_args.c_str());
    AI_LOG_INFO("Output:  %s", config.output_dir.c_str());

    TraceResult result;
    Pipeline pipeline;

    if (command == "trace" || command == "profile") {
        pipeline.add_stage(PipelineStage::TRACE, [&](TraceResult& r) {
            AI_LOG_INFO("Starting HIP trace + GPU metrics collection...");
            // Execute target app with HIP tracer injected
            std::string cmdline = config.target_app;
            if (!config.target_args.empty())
                cmdline += " " + config.target_args;

            AI_LOG_INFO("Launching: %s", cmdline.c_str());

            STARTUPINFOW si = { sizeof(si) };
            PROCESS_INFORMATION pi;

            // Set HIP_TRACER_OUTPUT env for the child process
            std::string trace_csv = config.output_dir + "/hip_trace.csv";
            SetEnvironmentVariableA("HIP_TRACER_OUTPUT", trace_csv.c_str());

            // Convert to wide string
            int wlen = MultiByteToWideChar(CP_UTF8, 0, cmdline.c_str(), -1, nullptr, 0);
            std::wstring wcmdline(wlen, L'\0');
            MultiByteToWideChar(CP_UTF8, 0, cmdline.c_str(), -1, &wcmdline[0], wlen);

            // Launch with HIP tracer DLL injection
            // (Requires the DLL to be in search path or use CreateProcess with
            //  DLL directory set; for now, run directly with LD_PRELOAD equivalent)
            SetEnvironmentVariableW(L"HIP_TRACER_DLL", L"ai_hip_tracer.dll");

            BOOL ok = CreateProcessW(
                nullptr, &wcmdline[0], nullptr, nullptr, FALSE,
                CREATE_DEFAULT_ERROR_MODE, nullptr, nullptr, &si, &pi
            );

            if (!ok) {
                AI_LOG_ERROR("Failed to launch process (error %d)", GetLastError());
                return;
            }

            WaitForSingleObject(pi.hProcess, INFINITE);
            CloseHandle(pi.hProcess);
            CloseHandle(pi.hThread);

            AI_LOG_INFO("Trace complete. CSV output: %s", trace_csv.c_str());
        });

        if (command == "profile") {
            pipeline.add_stage(PipelineStage::ANALYZE, [&](TraceResult& r) {
                AI_LOG_INFO("Analyzing trace data...");
                // Post-process trace CSV into structured records
                // TODO: Parse CSV, compute per-phase summaries
            });

            pipeline.add_stage(PipelineStage::OUTPUT, [&](TraceResult& r) {
                AI_LOG_INFO("Generating output...");
                // Write rocprofv3-compatible output
                // TODO: Generate JSON directory for RCV
            });
        }

        pipeline.run(result);

        AI_LOG_INFO("Done. Results in: %s", config.output_dir.c_str());
    } else {
        AI_LOG_ERROR("Unknown command: %s", command.c_str());
        return 1;
    }

    // Shutdown all plugins
    for (auto* p : PluginRegistry::instance().all()) {
        p->shutdown();
    }

    return 0;
}
