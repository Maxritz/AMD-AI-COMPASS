#!/usr/bin/env bash
###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
#
# Magpie generic benchmark body for Strata (a custom C++/HIP text engine).
#
# Strata is SERVER-LESS like xDiT: there is no OpenAI/vLLM endpoint, so this
# script drives the engine binary directly, parses the engine's own timing
# lines, runs a correctness gate, and writes an InferenceX-shaped result JSON
# that Magpie's ResultParser consumes.
#
# Resolved by benchmarker._get_benchmark_script as the Priority-3 generic script
# "strata_<runner>.sh" (runner_type for gfx1201 is "rdna4").
#
# Magpie contract (env in):
#   RESULT_DIR         output dir (Magpie sets to the workspace)
#   RESULT_FILENAME    result basename (Magpie sets to "inferencex_result")
#   RUNNER_TYPE        gpu runner (rdna4 for gfx1201)
#   MODEL              model path (accepted; the pack/gguf are what the engine loads)
#   CONC / ISL / OSL   workload knobs; ISL is unused (the prompt is a token file),
#                      OSL -> --max-new, CONC -> iterations
#   EXTRA_STRATA_ARGS  extra engine flags injected by the grid runner
#
# Engine wiring (set these; paths may be Windows "H:\..." or POSIX):
#   STRATA_EXE         the engine binary                     (default: strata.exe)
#   STRATA_PACK        the Strata pack directory             (required)
#   STRATA_NATIVE      the native GGUF shard                 (required)
#   STRATA_PLE        the PLE table GGUF                    (default: STRATA_NATIVE)
#   STRATA_PROFILE     --expert-profile (optional)
#   STRATA_TOKENS      a pre-tokenized prompt id file        (required)
#
# Tunables (the optimizer may set these):
#   STRATA_SPEC (2) STRATA_PREFILL (128) STRATA_CACHE (0) STRATA_MAX_NEW (OSL)
#   STRATA_ITERS (CONC) STRATA_MAX_CONTEXT (4096) STRATA_GATE (1)
#
# Correctness gate (STRATA_GATE=1): the engine is run twice with an identical
# configuration and the two greedy streams must be IDENTICAL. Strata's own
# documentation notes that a different batch shape (a 2-token window vs a
# 4-token window) rounds differently, so exact WINDOW-SIZE invariance is not an
# invariant the engine promises and is not used here. Determinism of an
# identical run IS an invariant it must hold, and it is what catches a race or a
# corrupted pass. Set STRATA_GATE=0 to skip it (the gate then reports skipped),
# but a scriptable run with a non-passing gate fails the benchmark by contract.
#
set -euo pipefail

# ROCm toolchain (a caller may need hipcc/rocprofv3 on PATH) --------------------
if ! command -v hipcc &>/dev/null && [ -x /opt/rocm/bin/hipcc ]; then
    export PATH="/opt/rocm/bin:${PATH}"
fi

# The engine is a Windows binary; the harness can run it under MSYS (Git Bash)
# or WSL, and a caller may give either path form. `to_windows` turns a POSIX
# path into the form the engine's arguments need (C:/..., H:/...); `to_posix`
# turns a Windows path into the form the running shell needs to exec it
# (/c/..., /mnt/c/...). A path already in the target form is passed through, and
# on a native POSIX host both are no-ops.
to_windows() {
    case "$1" in
        /mnt/[a-zA-Z]/*) if command -v wslpath &>/dev/null; then wslpath -w "$1"; else printf '%s' "$1"; fi ;;
        /[a-zA-Z]/*)     printf '%s' "$1" | sed -E 's#^/([a-zA-Z])/#\1:/#' ;;
        *)               printf '%s' "$1" ;;
    esac
}
to_posix() {
    case "$1" in
        [a-zA-Z]:[\\/]*) printf '/%s/%s' "$(printf '%s' "${1:0:1}" | tr 'A-Z' 'a-z')" \
                                          "$(printf '%s' "${1:2}" | tr '\\' '/')" ;;
        *)               printf '%s' "$1" ;;
    esac
}

RESULT_DIR="${RESULT_DIR:?RESULT_DIR must be set by Magpie}"
RESULT_FILENAME="${RESULT_FILENAME:-inferencex_result}"
OUTPUT_FILE="${RESULT_DIR}/${RESULT_FILENAME}.json"
mkdir -p "${RESULT_DIR}"

STRATA_EXE="${STRATA_EXE:-strata.exe}"
STRATA_PACK="${STRATA_PACK:?STRATA_PACK (the pack directory) must be set}"
STRATA_NATIVE="${STRATA_NATIVE:?STRATA_NATIVE (the native GGUF shard) must be set}"
STRATA_PLE="${STRATA_PLE:-${STRATA_NATIVE}}"
STRATA_PROFILE="${STRATA_PROFILE:-}"
STRATA_TOKENS="${STRATA_TOKENS:?STRATA_TOKENS (a pre-tokenized prompt id file) must be set}"

SPEC="${STRATA_SPEC:-2}"
PREFILL="${STRATA_PREFILL:-128}"
CACHE="${STRATA_CACHE:-0}"
MAX_NEW="${STRATA_MAX_NEW:-${OSL:-128}}"
ITERS="${STRATA_ITERS:-${CONC:-8}}"
MAX_CONTEXT="${STRATA_MAX_CONTEXT:-4096}"
GATE="${STRATA_GATE:-1}"
EXTRA_ARGS="${EXTRA_STRATA_ARGS:-${STRATA_EXTRA_ARGS:-}}"

echo "[strata] runner=${RUNNER_TYPE:-unknown} spec=${SPEC} prefill=${PREFILL} max_new=${MAX_NEW} iters=${ITERS} cache=${CACHE}"

# One engine run. $1 = spec (window size), $2 = log path.
run_once() {
    local spec="$1" log="$2"
    local -a args=(
        --pack "$(to_windows "${STRATA_PACK}")"
        --native "$(to_windows "${STRATA_NATIVE}")"
        --ple-gguf "$(to_windows "${STRATA_PLE}")"
        --spec "$spec" --suffix-draft 0 --prefill "$PREFILL"
        --max-new "$MAX_NEW" --max-context "$MAX_CONTEXT"
        --tokens-file "$(to_windows "${STRATA_TOKENS}")"
    )
    [ -n "${STRATA_PROFILE}" ] && args+=(--expert-profile "$(to_windows "${STRATA_PROFILE}")")
    [ "${CACHE}" != "0" ] && args+=(--expert-cache "${CACHE}")
    # shellcheck disable=SC2086
    "$(to_posix "${STRATA_EXE}")" "${args[@]}" ${EXTRA_ARGS} >"${log}" 2>&1
}

# ---- correctness gate: an identical run twice must give the same stream ------
GATE_JSON='{"passed": true, "skipped": true, "reason": "STRATA_GATE=0"}'
if [ "${GATE}" = "1" ]; then
    echo "[strata][gate] determinism: the same run twice must give the same stream"
    run_once "${SPEC}" "${RESULT_DIR}/strata_gate_a.log"
    run_once "${SPEC}" "${RESULT_DIR}/strata_gate_b.log"
    if ! diff <(grep '^output' "${RESULT_DIR}/strata_gate_a.log") \
              <(grep '^output' "${RESULT_DIR}/strata_gate_b.log") >/dev/null 2>&1; then
        echo "[strata][gate] FAIL: two identical runs produced different streams" >&2
        GATE_JSON='{"passed": false, "metric": "determinism", "reason": "repeat run differs"}'
    else
        echo "[strata][gate] PASS: identical across repeat runs"
        GATE_JSON='{"passed": true, "metric": "determinism", "reason": "identical across repeat runs"}'
    fi
fi

# ---- timing runs ------------------------------------------------------------
WALL_START=$(date +%s.%N 2>/dev/null || date +%s)
i=0
while [ "${i}" -lt "${ITERS}" ]; do
    run_once "${SPEC}" "${RESULT_DIR}/strata_iter_${i}.log"
    i=$((i + 1))
done
WALL_END=$(date +%s.%N 2>/dev/null || date +%s)

# ---- parse + write the InferenceX-shaped result -----------------------------
export STRATA_RESULT_DIR="${RESULT_DIR}" STRATA_ITERS_DONE="${ITERS}"
export STRATA_WALL_START="${WALL_START}" STRATA_WALL_END="${WALL_END}"
export STRATA_MAX_NEW STRATA_GATE_JSON="${GATE_JSON}" STRATA_MODEL="${MODEL:-}" STRATA_PRECISION="${PRECISION:-}"
# A working interpreter for the result file: prefer python3, but a Windows
# "python3" is sometimes only the Store alias, so fall back to python.
PY="${PYTHON:-python3}"
if ! "${PY}" -c "pass" >/dev/null 2>&1; then
    command -v python >/dev/null 2>&1 && PY=python
fi
"${PY}" - "${OUTPUT_FILE}" <<'PYEOF'
import glob, json, os, re, statistics, sys

out_path = sys.argv[1]
rdir = os.environ["STRATA_RESULT_DIR"]
iters = int(os.environ["STRATA_ITERS_DONE"])
try:
    wall = float(os.environ["STRATA_WALL_END"]) - float(os.environ["STRATA_WALL_START"])
except ValueError:
    wall = 0.0

decode_re = re.compile(r"^decode\s+\d+\s+tokens in\s+[\d.]+\s+ms\s+->\s+([\d.]+)\s+tok/s", re.M)
ttft_re = re.compile(r"time to first token ([\d.]+) ms", re.M)
e2e_re = re.compile(r"^decode\s+\d+\s+tokens in\s+([\d.]+)\s+ms", re.M)
produced_re = re.compile(r"^output\s*:\s*(.*)$", re.M)

decode_rates, ttfts, e2es, produced = [], [], [], 0
for log in sorted(glob.glob(os.path.join(rdir, "strata_iter_*.log"))):
    text = open(log, encoding="utf-8", errors="replace").read()
    m = decode_re.search(text)
    if m:
        decode_rates.append(float(m.group(1)))
    m = ttft_re.search(text)
    if m:
        ttfts.append(float(m.group(1)))
    m = e2e_re.search(text)
    if m:
        e2es.append(float(m.group(1)))
    m = produced_re.search(text)
    if m:
        produced += len(m.group(1).split())

mean_tok_s = statistics.fmean(decode_rates) if decode_rates else 0.0
mean_ttft = statistics.fmean(ttfts) if ttfts else 0.0
mean_e2e = statistics.fmean(e2es) if e2es else 0.0

result = {
    "framework": "strata",
    "model": os.environ.get("STRATA_MODEL", ""),
    "workload_kind": "scriptable",
    "throughput_unit": "tok/s",
    "output_throughput": round(mean_tok_s, 6),
    "request_throughput": round(iters / wall, 6) if wall > 0 else 0.0,
    "total_token_throughput": round(mean_tok_s, 6),
    "completed": iters,
    "num_prompts": iters,
    "duration": round(wall, 3),
    "total_output_tokens": produced,
    "mean_ttft_ms": round(mean_ttft, 3),
    "mean_tpot_ms": round(1000.0 / mean_tok_s, 4) if mean_tok_s > 0 else 0.0,
    "mean_e2el_ms": round(mean_e2e, 3),
    "latency_s": round(mean_e2e / 1000.0, 4),
    "precision": os.environ.get("STRATA_PRECISION", ""),
    "quality_gate": json.loads(os.environ.get("STRATA_GATE_JSON", '{"passed": true}')),
}

with open(out_path, "w", encoding="utf-8") as fh:
    json.dump(result, fh, indent=2)
print(f"[strata] throughput={mean_tok_s:.4f} tok/s ttft={mean_ttft:.2f} ms completed={iters} -> {out_path}")
PYEOF

echo "=== Magpie Strata bench complete ==="
