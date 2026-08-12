"""
Session anchor knowledge (2026-08-07) for the AMD-AI-COMPASS context-memory + sgl_kernel ROCm/RDNA4 port work.

Captured into BOTH knowledge stores:
  1. TencentDB Agent Memory gateway (SQLite, http://127.0.0.1:8420).
  2. graphify knowledge graph (graphify-out/graph.json).

------------------------------------------------------------------
CONTEXT-MEMORY SERVER (AI-COMPASS Memory Gateway)
------------------------------------------------------------------
Repo path: tools/memory_gateway.py
A zero-dependency, stdlib-only HTTP server (Python http.server + SQLite, no async, no 3rd-party deps).

Start commands:
  python tools/memory_gateway.py              # foreground, blocks
  python tools/memory_gateway.py --daemon     # detached (Popen, start_new_session=True)
  python tools/ensure_gateway.py              # idempotent auto-start + liveness check

Endpoint: http://127.0.0.1:8420  (default port DEFAULT_PORT=8420, configurable via --port)
Storage:  SQLite DB at  ~/.ai-compass/memory/memory.db  (DATA_DIR; WAL journal, busy_timeout=3s)
PID file:  ~/.ai-compass/memory/gateway.pid
Version:   ai-compass-memory/0.1.0  (string embedded in MemoryGatewayHandler /health)

Lifecycle:
  - start_server() checks _is_running() (PID file + Win OpenProcess/UNIX kill(pid,0)).
  - On first start: _init_schema() creates tables {conversations, atomic_memories,
    scenarios, core}, then binds _ThreadingHTTPServer(("127.0.0.1", port), handler).
  - Handler.log_message is silenced (no access log to stdout).
  - _is_running() on Windows uses ctypes OpenProcess(0x0400=PROCESS_QUERY_INFORMATION).

Auth:
  - Health endpoint GET /health is auth-free (returns status, version, vectorStore,
    services.pipelineWorker, scenarios count).
  - Per memory_hub.MemoryHub, every /v2 route SHOULD send
        Authorization: Bearer <TDAI_MEMORY_API_KEY>
        X-Tdai-Service-Id: <TDAI_MEMORY_SERVICE_ID>
    but the gateway's _auth_ok() returns True unconditionally -> standalone mode
    accepts any non-empty Bearer (and tolerates no-auth).

Knowledge layers (L0..L3):
  L0 conversation: POST /v2/conversation/add  {session_id, messages:[{role,content,ts}]}
                   + POST /v2/conversation/search {query, limit, session_id?}
  L1 atomic:      POST /v2/atomic/search       {query, limit, type?}
  L2 scenario:    POST /v2/scenario/{ls,read,write}  (write is UPSERT on path PK;
                  summary <=500 chars, content <=100000 chars)
  L3 core:        POST /v2/core/{read,write}       (single-row persona)

Distillation pipeline:
  - After every /v2/conversation/add, a daemon thread runs _run_pipeline() which
    selects UNFROZEN conversations (frozen=0), runs _distill_conversation() (splits
    on '.', indexes sentences <=500 chars with a word-count score min(1, words/20)),
    then sets frozen=1 so each L0 row is distilled exactly once.
  - This is a HEURISTIC (keyword chunking) -- the real TencentDB pipeline uses an LLM;
    the docstring at _distill_conversation states this explicitly.

Client: tools/memory_hub.py  (MemoryHub class; stdlib urllib HTTP client)
  Commands: status | capture --text/--json-report | search -q | recap -q |
            scenarios {ls|read|write} | persona {read|write}
  Env overrides: TDAI_MEMORY_ENDPOINT, TDAI_MEMORY_API_KEY, TDAI_MEMORY_SERVICE_ID,
                 TDAI_MEMORY_TIMEOUT.
  Defaults: endpoint http://127.0.0.1:8420, service-id "default", timeout 10s.
  Envelope: {"code":0,"data":{...}} ; non-zero raises MemoryHubError.

Round-trip verification (2026-08-07):
  capture -> /v2/conversation/add -> async pipeline -> L1 atomic_memories.
  search -q "nwarps RDNA4" returned 1 new (score 0.85) + 3 pre-existing L1 hits.
  scenarios ls returned 19 scene blocks (the sgl_kernel/RDNA4 port guides).

------------------------------------------------------------------
sgl_kernel ROCm PORT -> RDNA4 gfx1201 (Windows ROCm 7.13)
------------------------------------------------------------------
Target GPU: AMD Radeon RX 9070 XT, gfx1101/gfx1201 (RDNA4), driver 2.0.395.
ROCm:      7.13.0 at E:\ROCM-7.13.0-Windows  (HIP_PATH set; hipify-clang.exe at
           E:\ROCM-7.13.0-Windows\bin\hipify-clang.exe).
Torch:     Windows wheel (ship BOTH c10/cuda/* and c10/hip/* headers + CUDAMacros,
           but MISSING ATen/cuda/CUDAStream.h, ATen/cuda/CUDAGuard.h, and the
           cmake-generated c10/cuda/impl/cuda_cmake_macros.h).

Compile flags ($common, port-test/compile_port.ps1):
  -std=c++17 -O2 -c -DUSE_ROCM -DSGLANG_RDNA4 -DENABLE_BF16 -DENABLE_FP8
  -DHIP_FP8_TYPE_E4M3 -DOPERATOR_NAMESPACE=sgl_kernel
  -include ".../hip-compat/cuda_runtime_api.h"   # force-include shim
  -Wno-ignored-attributes
  -I$incAot -I$incAot/hip-compat -I$incCsrc -I$incTorch -I$incPy -I$hipInc
  NOTE: -DC10_CUDA_NO_CMAKE_CONFIGURE_FILE was REMOVED (it suppressed the macros
        shim and re-triggered the CUDAStream/Exceptions redefinition cascade).

Compat shims under aot/include/hip-compat/ + redirect shims:
  - c10/cuda/impl/cuda_cmake_macros.h : empty TORCH_CUDA_CPP_API / _CU_API / _API
    / _CXX17_API / _LIB_EXPORT / _LIB_API / _GLOBAL_CUDA_API (does NOT redefine
    C10_CUDA_API -- left to c10/cuda/CUDAMacros.h). Fixes CUDAContextLight.h:73.
  - hip-compat/cuda_runtime_api.h     : cuda->hip runtime mapping + appended
        namespace at::cuda { using ::c10::cuda::CUDAStream;
                              using ::c10::cuda::CUDAGuard;
                              inline CUDAStream getCurrentCUDAStream(int=-1); }
    Fixes bare CUDAStream/CUDAGuard/getCurrentCUDAStream in ATen/cuda/CUDAEvent.h
    (wheel lacks ATen/cuda/CUDAStream.h & CUDAGuard.h).
  - hip-compat/cuda_bf16.h            : __nv_bfloat16 = __hip_bfloat16.
  - c10/hip/HIPCachingAllocator.h     : #include <c10/cuda/CUDACachingAllocator.h>
    (kills "redefinition of 'FreeMemoryCallback'").
  - ATen/hip/Exceptions.h             : #include <ATen/cuda/Exceptions.h>
    (kills "redefinition of 'CuDNNError'", _hipsolver_backend_suggestion, and the
     TORCH_DSA_KERNEL_ARGS redefine warning).

Why the redirect shims are cycle-safe: the HIP twins include their CUDA twin and
the CUDA twin does NOT include the HIP twin, and the colliding classes
(FreeMemoryCallback, CuDNNError) both live in `namespace c10` in each twin, so the
using-alias preserves the namespace and avoids duplicate-class definitions.

Known inconsistency (not yet patched): aot/setup_rocm.py:119 still injects
-DC10_CUDA_NO_CMAKE_CONFIGURE_FILE, so the *installed* (non port-test) build would
break the same way; only port-test/compile_port.ps1 is fixed.

hipify-clang (dry-run, no --inplace, on moe_topk_sigmoid_kernels.cu with the same
-I/-D flag set) reported ZERO "unmapped" diagnostics for cuda->hip runtime maps,
i.e. the common API surface translates cleanly. What hipify CANNOT auto-translate:
  - cudaGraphAddDependencies / cudaGraphExec*WithFlags extras, cudaCtxSet/JIT options,
    cudaPointerSetAttributes (struct layout differs), cudaFuncSetAttribute subset,
    driver cuGraph node attrs.
  - __nv_bfloat16 (vs __hip_bfloat16), __nv_fp8/__nv_fp8x2 (vs __hip_*),
    cudaComplex/cuComplex field diffs, cudaTypedefs.h cutlass-path when SGLANG_RDNA4
    is absent for that TU.
  - <cub/...>, <cuComplex.h>, <thrust/...>, <nvtarget>, <sm_80a.h> intrinsics,
    PTX inline asm (ld.global / @ptx literals), and macro bodies that hide cuda
    tokens inside #defines the regex misses.

Remaining port-test task (not done this session): a clean `compile_port.ps1 -Clean`
rebuild with all shims live, then the link.exe pass (port-test/build.cmd) and a
functional smoke (python -c "import sglang" or a topk kernel dispatch) to confirm
the .pyd loads on RDNA4.
"""

__knowledge_domain__ = "context-memory + sgl_kernel ROCm/RDNA4 port"
__session__ = "2026-08-07"
__status__ = "gateway-running; graphify-updated; compile-shims-applied; link/smoke-pending"

# Layer -> endpoint
L0_CONVERSATION_ADD = "/v2/conversation/add"
L0_CONVERSATION_SEARCH = "/v2/conversation/search"
L1_ATOMIC_SEARCH = "/v2/atomic/search"
L2_SCENARIO_LS = "/v2/scenario/ls"
L2_SCENARIO_READ = "/v2/scenario/read"
L2_SCENARIO_WRITE = "/v2/scenario/write"
L3_CORE_READ = "/v2/core/read"
L3_CORE_WRITE = "/v2/core/write"
HEALTH = "/health"

# Gateway defaults (mirrors tools/memory_gateway.py)
DEFAULT_PORT = 8420
DATA_DIR_HOME = "~/.ai-compass/memory"
DB_FILENAME = "memory.db"
