#pragma once
#include <string>
#include <memory>
#include <vector>
#include "types.h"

namespace aicompass {

struct PluginContext {
    AicompassConfig config;
    TraceResult* result = nullptr;
};

class Plugin {
public:
    virtual ~Plugin() = default;
    virtual std::string name() const = 0;
    virtual std::string description() const = 0;
    virtual bool init(const AicompassConfig& config) = 0;
    virtual bool run(PluginContext& ctx) = 0;
    virtual void shutdown() = 0;
};

using PluginPtr = std::unique_ptr<Plugin>;

class PluginRegistry {
public:
    static PluginRegistry& instance() {
        static PluginRegistry reg;
        return reg;
    }

    void register_plugin(PluginPtr plugin) {
        plugins_.push_back(std::move(plugin));
    }

    Plugin* get(const std::string& name) {
        for (auto& p : plugins_)
            if (p->name() == name) return p.get();
        return nullptr;
    }

    std::vector<Plugin*> all() {
        std::vector<Plugin*> result;
        for (auto& p : plugins_) result.push_back(p.get());
        return result;
    }

    size_t count() const { return plugins_.size(); }

private:
    PluginRegistry() = default;
    std::vector<PluginPtr> plugins_;
};

} // namespace aicompass
