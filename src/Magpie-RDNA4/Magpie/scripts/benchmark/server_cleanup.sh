#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
# Teardown for inference servers started with:  setsid <cmd> ... &
#
# Do not add this to InferenceX benchmark_lib.sh — Magpie copies this file
# next to the other *.sh scripts in InferenceX/benchmarks/.
#
# Requires setsid on the server line so the PID is process-group leader and
# kill -TERM -$pid reaches vLLM Worker_TP / SGLang workers.

stop_benchmark_server_stack() {
    local root="${1:-}"
    [[ -z "$root" ]] && return 0
    if ! kill -0 "$root" 2>/dev/null; then
        return 0
    fi
    kill -TERM "-$root" 2>/dev/null || kill -TERM "$root" 2>/dev/null
    local i=0
    while kill -0 "$root" 2>/dev/null && [[ $i -lt 120 ]]; do
        sleep 0.25
        i=$((i + 1))
    done
    if kill -0 "$root" 2>/dev/null; then
        kill -KILL "$root" 2>/dev/null || true
    fi
}

# Magpie script traps use this name; InferenceX single_node may use either.
magpie_stop_benchmark_server_stack() {
    stop_benchmark_server_stack "$@"
}
