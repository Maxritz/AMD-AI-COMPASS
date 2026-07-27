#pragma once
#include <cstdio>
#include <string>
#include <mutex>

namespace aicompass {

class Logger {
public:
    enum Level { ERROR, WARN, INFO, DEBUG };

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

    void log(Level l, const char* fmt, ...) {
        if (l > level_ && !verbose_) return;
        std::lock_guard<std::mutex> lock(mtx_);
        const char* prefix = "";
        switch (l) {
            case ERROR: prefix = "[ERROR] "; break;
            case WARN:  prefix = "[WARN]  "; break;
            case INFO:  prefix = "[INFO]  "; break;
            case DEBUG: prefix = "[DEBUG] "; break;
        }
        va_list args;
        va_start(args, fmt);
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
        va_end(args);
    }

    void info(const char* fmt, ...) {
        va_list args; va_start(args, fmt);
        log(INFO, fmt, args); va_end(args);
    }
    void warn(const char* fmt, ...) {
        va_list args; va_start(args, fmt);
        log(WARN, fmt, args); va_end(args);
    }
    void error(const char* fmt, ...) {
        va_list args; va_start(args, fmt);
        log(ERROR, fmt, args); va_end(args);
    }
    void debug(const char* fmt, ...) {
        va_list args; va_start(args, fmt);
        log(DEBUG, fmt, args); va_end(args);
    }

    ~Logger() { if (file_) fclose(file_); }

private:
    Logger() : level_(INFO), verbose_(false), file_(nullptr) {}
    Level level_;
    bool verbose_;
    FILE* file_;
    std::mutex mtx_;
};

#define AI_LOG_INFO(...)  aicompass::Logger::instance().info(__VA_ARGS__)
#define AI_LOG_WARN(...)  aicompass::Logger::instance().warn(__VA_ARGS__)
#define AI_LOG_ERROR(...) aicompass::Logger::instance().error(__VA_ARGS__)
#define AI_LOG_DEBUG(...) aicompass::Logger::instance().debug(__VA_ARGS__)

} // namespace aicompass
