#pragma once
#include <functional>
#include "types.h"

namespace aicompass {

enum class PipelineStage {
    PRE_TRACE,
    TRACE,
    POST_TRACE,
    ANALYZE,
    OUTPUT
};

class Pipeline {
public:
    using StageCallback = std::function<void(TraceResult&)>;

    void add_stage(PipelineStage stage, StageCallback cb) {
        stages_.push_back({stage, std::move(cb)});
    }

    void run(TraceResult& result) {
        for (auto& [stage, cb] : stages_) {
            cb(result);
        }
    }

private:
    std::vector<std::pair<PipelineStage, StageCallback>> stages_;
};

} // namespace aicompass
