#pragma once
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>
#include <cstdarg>
#include <string>
#include <mutex>

// windows.h pollutes namespace with ERROR macro
#ifdef ERROR
#undef ERROR
#endif

namespace aicompass {

class Logger {
public:
    enum Level { LOG_ERROR, LOG_WARN, LOG_INFO, LOG_DEBUG };

    static Logger& instance() {
        static Logger inst;
        return inst;
    }

    void set_level(Level l) { level_ = l; }
    void set_verbose(bool v) { verbose_ = v; }
    void set_log_file(const std::string& path) {
        if (file_) fclose(file_);
        fopen_s(&file_, path.c_str(), "w");
    }

    void log_va(Level l, const char* fmt, va_list args) {
        std::lock_guard<std::mutex> lock(mtx_);
        const char* prefix = "";
        switch (l) {
            case LOG_ERROR: prefix = "[ERROR] "; break;
            case LOG_WARN:  prefix = "[WARN]  "; break;
            case LOG_INFO:  prefix = "[INFO]  "; break;
            case LOG_DEBUG: prefix = "[DEBUG] "; break;
        }
        if (file_) {
            fprintf(file_, "%s", prefix);
            vfprintf(file_, fmt, args);
            fprintf(file_, "\n");
            fflush(file_);
        } else {
            printf("%s", prefix);
            vprintf(fmt, args);
            printf("\n");
            fflush(stdout);
        }
    }

    void info(const char* fmt, ...) {
        va_list args; va_start(args, fmt);
        log_va(LOG_INFO, fmt, args); va_end(args);
    }
    void warn(const char* fmt, ...) {
        va_list args; va_start(args, fmt);
        log_va(LOG_WARN, fmt, args); va_end(args);
    }
    void error(const char* fmt, ...) {
        va_list args; va_start(args, fmt);
        log_va(LOG_ERROR, fmt, args); va_end(args);
    }
    void debug(const char* fmt, ...) {
        va_list args; va_start(args, fmt);
        log_va(LOG_DEBUG, fmt, args); va_end(args);
    }

    ~Logger() { if (file_) fclose(file_); }

private:
    Logger() : level_(LOG_INFO), verbose_(false), file_(nullptr) {}
    Level level_;
    bool verbose_;
    FILE* file_;
    std::mutex mtx_;
};

} // namespace aicompass

#define AI_LOG_INFO(...)  aicompass::Logger::instance().info(__VA_ARGS__)
#define AI_LOG_WARN(...)  aicompass::Logger::instance().warn(__VA_ARGS__)
#define AI_LOG_ERROR(...) aicompass::Logger::instance().error(__VA_ARGS__)
#define AI_LOG_DEBUG(...) aicompass::Logger::instance().debug(__VA_ARGS__)
