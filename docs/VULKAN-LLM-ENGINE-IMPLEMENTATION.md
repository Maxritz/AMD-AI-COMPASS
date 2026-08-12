# Pure Vulkan LLM Inference Engine — Implementation Design

## AMD RDNA4 (gfx1201) + RDNA2 | Complete Build & Runtime Spec

---

---

# PART 1: CMake Build System

---

## 1.1 File Structure

```
F:\AMD-Ai\AMD-AI-COMPASS\
├── CMakeLists.txt                    (existing — we add rdna4-llm as a sub-project)
├── src/
│   └── rdna4-llm/                    <-- NEW: Vulkan LLM engine
│       ├── CMakeLists.txt            <-- NEW: standalone build for the engine
│       ├── main.cpp
│       ├── vk_device.h
│       ├── vk_device.cpp
│       ├── vk_buffer.h
│       ├── vk_buffer.cpp
│       ├── vk_pipeline.h
│       ├── vk_pipeline.cpp
│       ├── vk_session.h
│       ├── vk_session.cpp
│       ├── vk_model.h
│       ├── vk_model.cpp
│       ├── vk_model_load.cpp          (GGUF loader + weight upload)
│       ├── vk_kv_cache.h
│       ├── vk_kv_cache.cpp
│       ├── vk_timeline.h
│       ├── vk_timeline.cpp
│       ├── vk_descriptor.h
│       ├── vk_descriptor.cpp
│       ├── gguf_parser.h
│       ├── gguf_parser.cpp
│       ├── gguf_types.h
│       ├── sample.h
│       ├── sample.cpp
│       ├── common.h                    (shared macros, limits, quant enum)
│       ├── shaders/
│       │   ├── rms_norm.comp
│       │   ├── attn_qkv_fp16.comp
│       │   ├── attn_qkv_q4_k.comp
│       │   ├── attn_qkv_q6_k.comp
│       │   ├── attn_qkv_q8_0.comp
│       │   ├── attn_qkv_iq4_xs.comp
│       │   ├── attn_compute.comp
│       │   ├── attn_output_fp16.comp
│       │   ├── attn_output_q4_k.comp
│       │   ├── attn_output_q6_k.comp
│       │   ├── attn_output_q8_0.comp
│       │   ├── attn_output_iq4_xs.comp
│       │   ├── ffn_gate_up_fp16.comp
│       │   ├── ffn_gate_up_q4_k.comp
│       │   ├── ffn_gate_up_q6_k.comp
│       │   ├── ffn_gate_up_q8_0.comp
│       │   ├── ffn_gate_up_iq4_xs.comp
│       │   ├── ffn_down_fp16.comp
│       │   ├── ffn_down_q4_k.comp
│       │   ├── ffn_down_q6_k.comp
│       │   ├── ffn_down_q8_0.comp
│       │   ├── ffn_down_iq4_xs.comp
│       │   ├── lm_head_fp16.comp
│       │   ├── lm_head_q4_k.comp
│       │   ├── lm_head_q6_k.comp
│       │   ├── lm_head_q8_0.comp
│       │   ├── lm_head_iq4_xs.comp
│       │   ├── token_embed.comp
│       │   └── common.glsl
│       └── scripts/
│           ├── compile_shaders.ps1              (offline / CI shader compilation)
│           └── embed_spirv.py                    (SPIR-V → C header embedder)
└── third_party/
    └── vma/                                     (VulkanMemoryAllocator — user provides)
        ├── vk_mem_alloc.h
        └── vk_mem_alloc.cpp
```

**Build output**:
```
<build_dir>/
├── rdna4-llm.exe
├── spirv/                                       (intermediate SPIR-V binaries)
│   ├── rms_norm_w32.spv
│   ├── rms_norm_w64.spv
│   ├── attn_qkv_fp16_w32.spv
│   ├── attn_qkv_fp16_w64.spv
│   ├── attn_qkv_q4_k_w32.spv
│   ├── attn_qkv_q4_k_w64.spv
│   │   ... (all shader × wave variants × quant types)
├── generated/                                    (generated C headers from SPIR-V)
│   ├── rms_norm_w32.h
│   ├── rms_norm_w64.h
│   │   ...
│   └── shader_registry.h                        (auto-generated index of all shaders)
├── pipeline_cache.bin                            (runtime pipeline cache)
└── tmp/                                          (scratch for embedding)
```

---

## 1.2 CMakeLists.txt — Full Implementation

```cmake
# src/rdna4-llm/CMakeLists.txt
cmake_minimum_required(VERSION 3.20)

# ─── Vulkan SDK auto-detection ───────────────────────────────────────
if(NOT DEFINED VULKAN_SDK_ROOT)
    if(DEFINED ENV{VULKAN_SDK})
        set(VULKAN_SDK_ROOT "$ENV{VULKAN_SDK}" CACHE PATH "Vulkan SDK root")
        message(STATUS "[rdna4-llm] Vulkan SDK from VULKAN_SDK env: ${VULKAN_SDK_ROOT}")
    else()
        file(GLOB _vk_candidates "C:/VulkanSDK/*")
        list(SORT _vk_candidates ORDER DESCENDING)
        list(GET _vk_candidates 0 _vk_first)
        if(_vk_first AND EXISTS "${_vk_first}/Include")
            set(VULKAN_SDK_ROOT "${_vk_first}" CACHE PATH "Vulkan SDK root")
            message(STATUS "[rdna4-llm] Vulkan SDK auto-detected: ${VULKAN_SDK_ROOT}")
        else()
            message(FATAL_ERROR "[rdna4-llm] Vulkan SDK not found. Set VULKAN_SDK env var or install to C:/VulkanSDK/")
        endif()
    endif()
endif()

set(VULKAN_INCLUDE_DIR "${VULKAN_SDK_ROOT}/Include")
set(VULKAN_LIB_DIR      "${VULKAN_SDK_ROOT}/Lib")
set(GLSLC_EXE           "${VULKAN_SDK_ROOT}/Bin/glslc.exe")
set(SPIRV_OPT_EXE       "${VULKAN_SDK_ROOT}/Bin/spirv-opt.exe")

# Verify glslc exists
if(NOT EXISTS "${GLSLC_EXE}")
    message(FATAL_ERROR "[rdna4-llm] glslc.exe not found at ${GLSLC_EXE}")
endif()

# ─── VMA detection ────────────────────────────────────────────────────
# Priority: 1) bundled under third_party/vma/  2) download via FetchContent
set(VMA_SRC "")
set(VMA_INCLUDE_DIR "")

if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/vma/vk_mem_alloc.h")
    set(VMA_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/vma")
    if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/vma/vk_mem_alloc.cpp")
        set(VMA_SRC "${CMAKE_SOURCE_DIR}/third_party/vma/vk_mem_alloc.cpp")
    else()
        # Header-only mode: define VMA_IMPLEMENTATION in exactly one .cpp
        set(VMA_SRC "")
    endif()
    message(STATUS "[rdna4-llm] VMA found (bundled): ${VMA_INCLUDE_DIR}")
else()
    message(WARNING "[rdna4-llm] VMA not found at third_party/vma/. "
        "Download from https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator "
        "and place vk_mem_alloc.h under third_party/vma/")
    message(FATAL_ERROR "[rdna4-llm] VMA is required. Aborting.")
endif()

# ─── Build configuration ─────────────────────────────────────────────
set(RDNA4_LLM_OUTPUT_DIR "${CMAKE_BINARY_DIR}/rdna4-llm")
set(RDNA4_LLM_SPIRV_DIR  "${RDNA4_LLM_OUTPUT_DIR}/spirv")
set(RDNA4_LLM_GEN_DIR    "${RDNA4_LLM_OUTPUT_DIR}/generated")
file(MAKE_DIRECTORY "${RDNA4_LLM_SPIRV_DIR}")
file(MAKE_DIRECTORY "${RDNA4_LLM_GEN_DIR}")

# MSVC release flags
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /arch:AVX2 /GL /GS- /fp:fast")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /O2 /arch:AVX2 /GL /Zi")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /Zi /RTC1")
    add_compile_options("$<$<CONFIG:Release>:/MT>" "$<$<CONFIG:Debug>:/MTd>" "$<$<CONFIG:RelWithDebInfo>:/MT>")
endif()

# ─── Shader definition database ──────────────────────────────────────
# Structure: list of <base_name> <wave_sizes> <defines>
# wave_sizes is a semicolon-separated list: "32;64"
# defines is a space-separated string of preprocessor defines

set(RDNA4_SHADERS "")

# Helper macro to register a shader
macro(register_shader NAME WAVE_SIZES EXTRA_DEFINES)
    set(_shader_entry "${NAME}|${WAVE_SIZES}|${EXTRA_DEFINES}")
    list(APPEND RDNA4_SHADERS "${_shader_entry}")
endmacro()

register_shader("rms_norm"          "32;64" "")
register_shader("attn_qkv_fp16"     "32;64" "-DQUANT_FP16=1")
register_shader("attn_qkv_q4_k"     "32;64" "-DQUANT_Q4_K=1")
register_shader("attn_qkv_q6_k"     "32;64" "-DQUANT_Q6_K=1")
register_shader("attn_qkv_q8_0"     "32;64" "-DQUANT_Q8_0=1")
register_shader("attn_qkv_iq4_xs"   "32;64" "-DQUANT_IQ4_XS=1")
register_shader("attn_compute"      "32;64" "")
register_shader("attn_output_fp16"  "32;64" "-DQUANT_FP16=1")
register_shader("attn_output_q4_k"  "32;64" "-DQUANT_Q4_K=1")
register_shader("attn_output_q6_k"  "32;64" "-DQUANT_Q6_K=1")
register_shader("attn_output_q8_0"  "32;64" "-DQUANT_Q8_0=1")
register_shader("attn_output_iq4_xs""32;64" "-DQUANT_IQ4_XS=1")
register_shader("ffn_gate_up_fp16"  "32;64" "-DQUANT_FP16=1")
register_shader("ffn_gate_up_q4_k"  "32;64" "-DQUANT_Q4_K=1")
register_shader("ffn_gate_up_q6_k"  "32;64" "-DQUANT_Q6_K=1")
register_shader("ffn_gate_up_q8_0"  "32;64" "-DQUANT_Q8_0=1")
register_shader("ffn_gate_up_iq4_xs""32;64" "-DQUANT_IQ4_XS=1")
register_shader("ffn_down_fp16"     "32;64" "-DQUANT_FP16=1")
register_shader("ffn_down_q4_k"     "32;64" "-DQUANT_Q4_K=1")
register_shader("ffn_down_q6_k"     "32;64" "-DQUANT_Q6_K=1")
register_shader("ffn_down_q8_0"     "32;64" "-DQUANT_Q8_0=1")
register_shader("ffn_down_iq4_xs"   "32;64" "-DQUANT_IQ4_XS=1")
register_shader("lm_head_fp16"      "32;64" "-DQUANT_FP16=1")
register_shader("lm_head_q4_k"      "32;64" "-DQUANT_Q4_K=1")
register_shader("lm_head_q6_k"      "32;64" "-DQUANT_Q6_K=1")
register_shader("lm_head_q8_0"      "32;64" "-DQUANT_Q8_0=1")
register_shader("lm_head_iq4_xs"    "32;64" "-DQUANT_IQ4_XS=1")
register_shader("token_embed"       "32;64" "")

# ─── Shader compilation targets ──────────────────────────────────────
set(RDNA4_SPIRV_FILES "")
set(RDNA4_HEADER_FILES "")
set(RDNA4_SHADER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")

foreach(_entry ${RDNA4_SHADERS})
    string(REPLACE "|" ";" _parts "${_entry}")
    list(GET _parts 0 _name)
    list(GET _parts 1 _waves)
    list(GET _parts 2 _defines)

    foreach(_w ${_waves})
        set(_spv_output "${RDNA4_LLM_SPIRV_DIR}/${_name}_w${_w}.spv")
        set(_h_output   "${RDNA4_LLM_GEN_DIR}/${_name}_w${_w}.h")
        set(_src         "${RDNA4_SHADER_INCLUDE_DIR}/${_name}.comp")

        if(NOT EXISTS "${_src}")
            message(WARNING "[rdna4-llm] Shader source not found: ${_src}")
            continue()
        endif()

        # Step 1: GLSL → SPIR-V via glslc
        add_custom_command(
            OUTPUT "${_spv_output}"
            COMMAND "${GLSLC_EXE}"
                -fshader-stage=compute
                --target-env=vulkan1.4
                --target-spv=spv1.4
                -O                                      # size optimization
                -DSUBGROUP_SIZE=${_w}
                ${_defines}
                -I "${RDNA4_SHADER_INCLUDE_DIR}"
                -o "${_spv_output}"
                "${_src}"
            DEPENDS "${_src}" "${RDNA4_SHADER_INCLUDE_DIR}/common.glsl"
            COMMENT "Compiling ${_name} (Wave${_w}) → SPIR-V"
            VERBATIM
        )
        list(APPEND RDNA4_SPIRV_FILES "${_spv_output}")

        # Step 2 (optional): spirv-opt pass
        if(EXISTS "${SPIRV_OPT_EXE}" AND CMAKE_BUILD_TYPE STREQUAL "Release")
            set(_spv_opt_output "${RDNA4_LLM_SPIRV_DIR}/${_name}_w${_w}_opt.spv")
            add_custom_command(
                OUTPUT "${_spv_opt_output}"
                COMMAND "${SPIRV_OPT_EXE}"
                    --eliminate-dead-code-aggressive
                    --merge-blocks
                    --simplify-instructions
                    --fold-spec-const-op-composite
                    -o "${_spv_opt_output}"
                    "${_spv_output}"
                DEPENDS "${_spv_output}"
                COMMENT "Optimizing ${_name}_w${_w}.spv"
                VERBATIM
            )
            set(_final_spv "${_spv_opt_output}")
        else()
            set(_final_spv "${_spv_output}")
        endif()

        # Step 3: SPIR-V → C header (Python embed script)
        add_custom_command(
            OUTPUT "${_h_output}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${RDNA4_LLM_GEN_DIR}"
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_spirv.py"
                --input  "${_final_spv}"
                --output "${_h_output}"
                --name   "shader_${_name}_w${_w}"
            DEPENDS "${_final_spv}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_spirv.py"
            COMMENT "Embedding ${_name}_w${_w}.spv → ${_name}_w${_w}.h"
            VERBATIM
        )
        list(APPEND RDNA4_HEADER_FILES "${_h_output}")
    endforeach()
endforeach()

# ─── Shader registry generator ───────────────────────────────────────
set(RDNA4_REGISTRY_H "${RDNA4_LLM_GEN_DIR}/shader_registry.h")
add_custom_command(
    OUTPUT "${RDNA4_REGISTRY_H}"
    COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_spirv.py"
        --generate-registry
        --output "${RDNA4_REGISTRY_H}"
        --shader-list "${RDNA4_HEADER_FILES}"
    DEPENDS ${RDNA4_HEADER_FILES} "${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_spirv.py"
    COMMENT "Generating shader_registry.h"
    VERBATIM
)
list(APPEND RDNA4_HEADER_FILES "${RDNA4_REGISTRY_H}")

# ─── Custom target for shader compilation (build system drive) ──────
add_custom_target(rdna4_shaders ALL
    DEPENDS ${RDNA4_HEADER_FILES}
    COMMENT "All RDNA4 LLM shaders compiled and embedded"
)

# ─── Source files ────────────────────────────────────────────────────
set(RDNA4_LLM_SOURCES
    # Vulkan device / buffer primitives
    main.cpp
    vk_device.cpp
    vk_buffer.cpp
    vk_pipeline.cpp
    vk_session.cpp
    vk_model.cpp
    vk_model_load.cpp
    vk_kv_cache.cpp
    vk_timeline.cpp
    vk_descriptor.cpp

    # GGUF parser
    gguf_parser.cpp

    # Sampling
    sample.cpp

    # VMA implementation (if .cpp provided, otherwise one source gets VMA_IMPLEMENTATION)
    ${VMA_SRC}
)

# ─── Executable ──────────────────────────────────────────────────────
add_executable(rdna4-llm ${RDNA4_LLM_SOURCES})

target_include_directories(rdna4-llm PRIVATE
    "${VULKAN_INCLUDE_DIR}"
    "${VMA_INCLUDE_DIR}"
    "${RDNA4_LLM_GEN_DIR}"           # generated shader headers
    "${CMAKE_CURRENT_SOURCE_DIR}"     # src/rdna4-llm/ for local includes
)

target_link_libraries(rdna4-llm PRIVATE
    "${VULKAN_LIB_DIR}/vulkan-1.lib"
)

# VMA compilation definition (exactly one TU must define VMA_IMPLEMENTATION)
if(NOT VMA_SRC)
    set_property(SOURCE vk_buffer.cpp APPEND PROPERTY COMPILE_DEFINITIONS "VMA_IMPLEMENTATION")
endif()
set_property(TARGET rdna4-llm PROPERTY COMPILE_DEFINITIONS
    "VMA_STATIC_VULKAN_FUNCTIONS=0"
    "VMA_DYNAMIC_VULKAN_FUNCTIONS=1"
)

# Generated headers dependency
add_dependencies(rdna4-llm rdna4_shaders)

# Windows subsystem: console (for stdout token output)
set_target_properties(rdna4-llm PROPERTIES
    WIN32_EXECUTABLE FALSE
    LINK_FLAGS_RELEASE "/OPT:REF /OPT:ICF /LTCG"
    LINK_FLAGS_RELWITHDEBINFO "/OPT:REF /OPT:ICF /DEBUG:FULL"
)

# ─── Install ─────────────────────────────────────────────────────────
install(TARGETS rdna4-llm
    RUNTIME DESTINATION bin
)

message(STATUS "[rdna4-llm] Build configured: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "[rdna4-llm] Vulkan SDK:  ${VULKAN_SDK_ROOT}")
message(STATUS "[rdna4-llm] Shader count: ${RDNA4_SHADERS} (×2 wave variants each)")
```

---

## 1.3 Python: SPIR-V Embedder Script

```python
# src/rdna4-llm/scripts/embed_spirv.py
"""Embeds SPIR-V binary as a C header (xxd -i style) and generates shader registry."""

import argparse
import os
import sys
import struct

def embed_spirv(input_path: str, output_path: str, var_name: str) -> None:
    """Read SPIR-V binary, write C header with static const uint32_t array."""
    with open(input_path, "rb") as f:
        data = f.read()

    if len(data) % 4 != 0:
        raise ValueError(f"SPIR-V file size {len(data)} is not a multiple of 4 bytes")

    word_count = len(data) // 4
    sanitized_name = var_name.replace(".", "_").replace("-", "_")

    lines = []
    lines.append(f"// Auto-generated from: {os.path.basename(input_path)}")
    lines.append(f"// Words: {word_count}  Bytes: {len(data)}")
    lines.append(f"#pragma once")
    lines.append(f"#include <stdint.h>")
    lines.append(f"static const uint32_t {sanitized_name}_data[] = {{")

    for i in range(0, len(data), 4):
        word = struct.unpack_from("<I", data, i)[0]
        lines.append(f"    0x{word:08X},")

    lines.append(f"}};")
    lines.append(f"static const size_t {sanitized_name}_size = {len(data)};")
    lines.append(f"static const uint32_t {sanitized_name}_word_count = {word_count};")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"  Embedded {os.path.basename(input_path)} → {os.path.basename(output_path)} "
          f"({word_count} words)")


def generate_registry(output_path: str, shader_list: str) -> None:
    """Generate shader_registry.h that maps (op, quant, wave) → embedded data ptr."""
    # Parse shader list: "path1;path2;path3..."
    shader_paths = shader_list.split(";") if shader_list else []

    lines = []
    lines.append("// Auto-generated shader registry — do not edit")
    lines.append("#pragma once")
    lines.append("#include <stdint.h>")
    lines.append("#include <stddef.h>")
    lines.append("")

    # Include all individual shader headers
    for path in shader_paths:
        if not path.strip():
            continue
        basename = os.path.basename(path)
        lines.append(f'#include "{basename}"')

    lines.append("")
    lines.append("typedef struct embedded_shader_t {")
    lines.append("    const char*    name;")
    lines.append("    const uint32_t* data;")
    lines.append("    size_t          byte_size;")
    lines.append("    uint32_t        word_count;")
    lines.append("    uint32_t        subgroup_size;    // 32 or 64")
    lines.append("    int             quant_type;       // GGUF_QUANT_* enum value")
    lines.append("    int             op_type;          // OP_RMS_NORM, OP_ATTN_QKV, etc.")
    lines.append("} embedded_shader_t;")
    lines.append("")

    # Generate the lookup table entries
    lines.append("static const embedded_shader_t EMBEDDED_SHADERS[] = {")

    for path in shader_paths:
        if not path.strip():
            continue
        stem = os.path.splitext(os.path.basename(path))[0]  # e.g. "attn_qkv_q4_k_w32"

        # Parse stem into parts: <op>_<quant>_w<wavesize>
        parts = stem.split("_")
        wave_size = 0
        for p in parts:
            if p.startswith("w") and p[1:].isdigit():
                wave_size = int(p[1:])
        if wave_size == 0:
            wave_size = 32  # default

        # Derive quant_type and op_type from the name prefix
        quant_type = "GGUF_QUANT_FP16"
        if "q4_k" in stem:
            quant_type = "GGUF_QUANT_Q4_K"
        elif "q6_k" in stem:
            quant_type = "GGUF_QUANT_Q6_K"
        elif "q8_0" in stem:
            quant_type = "GGUF_QUANT_Q8_0"
        elif "iq4_xs" in stem:
            quant_type = "GGUF_QUANT_IQ4_XS"

        op_type = "OP_UNKNOWN"
        if stem.startswith("rms_norm"):
            op_type = "OP_RMS_NORM"
        elif stem.startswith("attn_qkv"):
            op_type = "OP_ATTN_QKV"
        elif stem.startswith("attn_compute"):
            op_type = "OP_ATTN_COMPUTE"
        elif stem.startswith("attn_output"):
            op_type = "OP_ATTN_OUTPUT"
        elif stem.startswith("ffn_gate_up"):
            op_type = "OP_FFN_GATE_UP"
        elif stem.startswith("ffn_down"):
            op_type = "OP_FFN_DOWN"
        elif stem.startswith("lm_head"):
            op_type = "OP_LM_HEAD"
        elif stem.startswith("token_embed"):
            op_type = "OP_TOKEN_EMBED"

        var_name = f"shader_{stem}"
        lines.append(f"    {{ \"{stem}\", {var_name}_data, {var_name}_size, "
                     f"{var_name}_word_count, {wave_size}, {quant_type}, {op_type} }},")

    lines.append("    { NULL, NULL, 0, 0, 0, 0, 0 }  // sentinel")
    lines.append("};")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description="SPIR-V embed tool")
    parser.add_argument("--input", help="Input .spv file")
    parser.add_argument("--output", help="Output .h file")
    parser.add_argument("--name", help="C variable name prefix")
    parser.add_argument("--generate-registry", action="store_true",
                        help="Generate shader_registry.h")
    parser.add_argument("--shader-list", default="",
                        help="Semicolon-separated list of shader header paths")
    args = parser.parse_args()

    if args.generate_registry:
        generate_registry(args.output, args.shader_list)
    else:
        if not args.input or not args.output or not args.name:
            parser.error("--input, --output, and --name required for embed mode")
        embed_spirv(args.input, args.output, args.name)


if __name__ == "__main__":
    main()
```

---

## 1.4 PowerShell: Offline Shader Compilation Script

```powershell
# src/rdna4-llm/scripts/compile_shaders.ps1
# Usage: pwsh compile_shaders.ps1 [-VulkanSdk <path>] [-BuildDir <path>]
# Compiles and embeds all shaders outside of CMake (for CI / iteration)

param(
    [string]$VulkanSdk = "C:\VulkanSDK\1.4.357.0",
    [string]$BuildDir = "build\rdna4-llm"
)

$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path $MyInvocation.MyCommand.Path -Parent
$ShaderDir = Join-Path $ScriptDir "..\shaders"
$SpirvDir  = Join-Path $BuildDir "spirv"
$GenDir    = Join-Path $BuildDir "generated"

New-Item -ItemType Directory -Force -Path $SpirvDir, $GenDir | Out-Null

$Glslc  = Join-Path $VulkanSdk "Bin\glslc.exe"
$SpvOpt = Join-Path $VulkanSdk "Bin\spirv-opt.exe"
$EmbedPy = Join-Path $ScriptDir "embed_spirv.py"

if (-not (Test-Path $Glslc)) {
    Write-Error "glslc.exe not found at $Glslc"
    exit 1
}

$Shaders = @(
    @{Name="rms_norm";          Waves=@(32,64); Defines=""},
    @{Name="attn_qkv_fp16";     Waves=@(32,64); Defines="-DQUANT_FP16=1"},
    @{Name="attn_qkv_q4_k";     Waves=@(32,64); Defines="-DQUANT_Q4_K=1"},
    @{Name="attn_qkv_q6_k";     Waves=@(32,64); Defines="-DQUANT_Q6_K=1"},
    @{Name="attn_qkv_q8_0";     Waves=@(32,64); Defines="-DQUANT_Q8_0=1"},
    @{Name="attn_qkv_iq4_xs";   Waves=@(32,64); Defines="-DQUANT_IQ4_XS=1"},
    @{Name="attn_compute";      Waves=@(32,64); Defines=""},
    @{Name="attn_output_fp16";  Waves=@(32,64); Defines="-DQUANT_FP16=1"},
    @{Name="attn_output_q4_k";  Waves=@(32,64); Defines="-DQUANT_Q4_K=1"},
    @{Name="attn_output_q6_k";  Waves=@(32,64); Defines="-DQUANT_Q6_K=1"},
    @{Name="attn_output_q8_0";  Waves=@(32,64); Defines="-DQUANT_Q8_0=1"},
    @{Name="attn_output_iq4_xs";Waves=@(32,64); Defines="-DQUANT_IQ4_XS=1"},
    @{Name="ffn_gate_up_fp16";  Waves=@(32,64); Defines="-DQUANT_FP16=1"},
    @{Name="ffn_gate_up_q4_k";  Waves=@(32,64); Defines="-DQUANT_Q4_K=1"},
    @{Name="ffn_gate_up_q6_k";  Waves=@(32,64); Defines="-DQUANT_Q6_K=1"},
    @{Name="ffn_gate_up_q8_0";  Waves=@(32,64); Defines="-DQUANT_Q8_0=1"},
    @{Name="ffn_gate_up_iq4_xs";Waves=@(32,64); Defines="-DQUANT_IQ4_XS=1"},
    @{Name="ffn_down_fp16";     Waves=@(32,64); Defines="-DQUANT_FP16=1"},
    @{Name="ffn_down_q4_k";     Waves=@(32,64); Defines="-DQUANT_Q4_K=1"},
    @{Name="ffn_down_q6_k";     Waves=@(32,64); Defines="-DQUANT_Q6_K=1"},
    @{Name="ffn_down_q8_0";     Waves=@(32,64); Defines="-DQUANT_Q8_0=1"},
    @{Name="ffn_down_iq4_xs";   Waves=@(32,64); Defines="-DQUANT_IQ4_XS=1"},
    @{Name="lm_head_fp16";      Waves=@(32,64); Defines="-DQUANT_FP16=1"},
    @{Name="lm_head_q4_k";      Waves=@(32,64); Defines="-DQUANT_Q4_K=1"},
    @{Name="lm_head_q6_k";      Waves=@(32,64); Defines="-DQUANT_Q6_K=1"},
    @{Name="lm_head_q8_0";      Waves=@(32,64); Defines="-DQUANT_Q8_0=1"},
    @{Name="lm_head_iq4_xs";    Waves=@(32,64); Defines="-DQUANT_IQ4_XS=1"},
    @{Name="token_embed";       Waves=@(32,64); Defines=""}
)

$AllHeaders = @()

foreach ($s in $Shaders) {
    $Src = Join-Path $ShaderDir "$($s.Name).comp"
    if (-not (Test-Path $Src)) {
        Write-Warning "Shader source not found: $Src"
        continue
    }

    foreach ($w in $s.Waves) {
        $SpvOut = Join-Path $SpirvDir "$($s.Name)_w$w.spv"
        $HOut   = Join-Path $GenDir "$($s.Name)_w$w.h"

        # Compile
        $GlslcArgs = @(
            "-fshader-stage=compute",
            "--target-env=vulkan1.4",
            "--target-spv=spv1.4",
            "-O",
            "-DSUBGROUP_SIZE=$w"
        )
        if ($s.Defines) { $GlslcArgs += $s.Defines.Split(" ") }
        $GlslcArgs += @(
            "-I", $ShaderDir,
            "-o", $SpvOut,
            $Src
        )

        Write-Host "  glslc $($s.Name)_w$w" -ForegroundColor Cyan
        & $Glslc $GlslcArgs 2>&1 | ForEach-Object { Write-Host "    $_" }
        if ($LASTEXITCODE -ne 0) {
            Write-Error "glslc failed for $($s.Name)_w$w"
            exit 1
        }

        # Optional: spirv-opt
        if (Test-Path $SpvOpt) {
            & $SpvOpt --eliminate-dead-code-aggressive --merge-blocks `
                --simplify-instructions -o $SpvOut $SpvOut
        }

        # Embed
        & python $EmbedPy --input $SpvOut --output $HOut --name "shader_$($s.Name)_w$w"
        if ($LASTEXITCODE -ne 0) {
            Write-Error "embed_spirv.py failed for $($s.Name)_w$w"
            exit 1
        }

        $AllHeaders += $HOut
    }
}

# Generate registry
$HeaderList = $AllHeaders -join ";"
Write-Host "  Generating shader_registry.h" -ForegroundColor Cyan
& python $EmbedPy --generate-registry --output (Join-Path $GenDir "shader_registry.h") `
    --shader-list $HeaderList

Write-Host "`nDone: $($Shaders.Count) shaders × wave variants compiled." -ForegroundColor Green
Write-Host "SPIR-V: $SpirvDir" -ForegroundColor Green
Write-Host "Headers: $GenDir" -ForegroundColor Green
```

---

---

# PART 2: VMA Integration

---

## 2.1 VMA Allocator Creation

```c
// vk_buffer.cpp

#define VMA_IMPLEMENTATION  // exactly one translation unit
#include "vk_mem_alloc.h"

VmaAllocator create_vma_allocator(VkInstance instance,
                                   VkPhysicalDevice physical_device,
                                   VkDevice device) {
    // Dynamically load Vulkan function pointers
    VmaVulkanFunctions vma_funcs = {};
    vma_funcs.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vma_funcs.vkGetDeviceProcAddr   = vkGetDeviceProcAddr;
    vma_funcs.vkAllocateMemory             = (PFN_vkAllocateMemory)vkGetDeviceProcAddr(device, "vkAllocateMemory");
    vma_funcs.vkBindBufferMemory           = (PFN_vkBindBufferMemory)vkGetDeviceProcAddr(device, "vkBindBufferMemory");
    vma_funcs.vkCreateBuffer               = (PFN_vkCreateBuffer)vkGetDeviceProcAddr(device, "vkCreateBuffer");
    vma_funcs.vkDestroyBuffer              = (PFN_vkDestroyBuffer)vkGetDeviceProcAddr(device, "vkDestroyBuffer");
    vma_funcs.vkFreeMemory                 = (PFN_vkFreeMemory)vkGetDeviceProcAddr(device, "vkFreeMemory");
    vma_funcs.vkGetBufferMemoryRequirements = (PFN_vkGetBufferMemoryRequirements)vkGetDeviceProcAddr(device, "vkGetBufferMemoryRequirements");
    vma_funcs.vkGetPhysicalDeviceMemoryProperties = (PFN_vkGetPhysicalDeviceMemoryProperties)vkGetDeviceProcAddr(device, "vkGetPhysicalDeviceMemoryProperties");
    vma_funcs.vkGetPhysicalDeviceProperties = (PFN_vkGetPhysicalDeviceProperties)vkGetDeviceProcAddr(device, "vkGetPhysicalDeviceProperties");
    vma_funcs.vkMapMemory                  = (PFN_vkMapMemory)vkGetDeviceProcAddr(device, "vkMapMemory");
    vma_funcs.vkUnmapMemory                = (PFN_vkUnmapMemory)vkGetDeviceProcAddr(device, "vkUnmapMemory");
    vma_funcs.vkFlushMappedMemoryRanges    = (PFN_vkFlushMappedMemoryRanges)vkGetDeviceProcAddr(device, "vkFlushMappedMemoryRanges");
    vma_funcs.vkInvalidateMappedMemoryRanges = (PFN_vkInvalidateMappedMemoryRanges)vkGetDeviceProcAddr(device, "vkInvalidateMappedMemoryRanges");
    vma_funcs.vkCmdCopyBuffer              = (PFN_vkCmdCopyBuffer)vkGetDeviceProcAddr(device, "vkCmdCopyBuffer");

    // For buffer device address
    vma_funcs.vkGetBufferDeviceAddress = (PFN_vkGetBufferDeviceAddress)vkGetDeviceProcAddr(device, "vkGetBufferDeviceAddress");

    VmaAllocatorCreateInfo ci = {};
    ci.vulkanApiVersion = VK_API_VERSION_1_4;
    ci.physicalDevice   = physical_device;
    ci.device           = device;
    ci.instance         = instance;
    ci.pVulkanFunctions = &vma_funcs;

    // BDA support (required for descriptor indexing with device addresses)
    ci.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    // Memory budget extension — pollable for VRAM monitoring
    ci.flags |= VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;

    VmaAllocator allocator;
    VkResult result = vmaCreateAllocator(&ci, &allocator);
    if (result != VK_SUCCESS) {
        return VK_NULL_HANDLE;
    }

    return allocator;
}
```

## 2.2 Buffer Creation Wrappers

```c
// vk_buffer.h / vk_buffer.cpp

typedef enum vk_buffer_flags_t {
    VK_BUF_DEVICE_LOCAL  = 0,       // VRAM-only, no CPU access
    VK_BUF_HOST_VISIBLE  = 1 << 0,  // CPU-mappable
    VK_BUF_HOST_COHERENT = 1 << 1,  // No manual flush needed
    VK_BUF_UPLOAD        = 1 << 2,  // Staging: HOST_VISIBLE + mapped persistently
    VK_BUF_DOWNLOAD      = 1 << 3,  // Readback: HOST_VISIBLE + HOST_CACHED
} vk_buffer_flags_t;

typedef struct vk_buffer_t {
    VkBuffer        buffer;
    VmaAllocation   allocation;
    VmaAllocationInfo alloc_info;
    VkDeviceSize    size;
    VkDeviceAddress device_address;
    void*           mapped_ptr;
    bool            is_host_visible;
    bool            is_host_coherent;
} vk_buffer_t;

// Primary allocation function
bool vk_buffer_create(VmaAllocator allocator,
                      VkDeviceSize size,
                      VkBufferUsageFlags usage,
                      vk_buffer_flags_t flags,
                      vk_buffer_t* out_buf) {
    VkBufferCreateInfo bci = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size  = size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    // Always request BDA when supported — low cost, high utility
    if (usage & (VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                 VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT)) {
        bci.usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    }

    VmaAllocationCreateInfo aci = {};
    aci.usage = VMA_MEMORY_USAGE_AUTO;

    if (flags & VK_BUF_HOST_VISIBLE) {
        aci.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        if (flags & VK_BUF_HOST_COHERENT) {
            aci.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                               VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        } else {
            aci.requiredFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
        }
    } else {
        aci.requiredFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    }

    if (flags & VK_BUF_DOWNLOAD) {
        aci.flags |= VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
        aci.preferredFlags = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    }

    VkResult result = vmaCreateBuffer(allocator, &bci, &aci,
                                       &out_buf->buffer,
                                       &out_buf->allocation,
                                       &out_buf->alloc_info);

    if (result != VK_SUCCESS) return false;

    out_buf->size            = size;
    out_buf->is_host_visible = aci.requiredFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    out_buf->is_host_coherent= aci.requiredFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    out_buf->mapped_ptr      = out_buf->alloc_info.pMappedData;
    out_buf->device_address  = 0;

    // Fetch device address if BDA usage was requested
    if (bci.usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        // Need device handle accessible — pass via context or store globally
    }

    return true;
}

void vk_buffer_destroy(VmaAllocator allocator, vk_buffer_t* buf) {
    if (buf->buffer) {
        if (buf->mapped_ptr) {
            vmaUnmapMemory(allocator, buf->allocation);
            buf->mapped_ptr = NULL;
        }
        vmaDestroyBuffer(allocator, buf->buffer, buf->allocation);
        memset(buf, 0, sizeof(*buf));
    }
}

void vk_buffer_flush(VmaAllocator allocator, vk_buffer_t* buf,
                     VkDeviceSize offset, VkDeviceSize size) {
    if (!buf->is_host_coherent) {
        vmaFlushAllocation(allocator, buf->allocation, offset, size);
    }
}
```

## 2.3 Memory Budget Monitoring

```c
// In vk_device.cpp, called before LoadModel() to verify capacity

bool check_vram_budget(VmaAllocator allocator, VkDeviceSize required_bytes) {
    VmaTotalStatistics stats;
    vmaCalculateStatistics(allocator, &stats);

    // VmaBudget[] available if VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT set
    // For a quick check, use VMA heap statistics
    for (uint32_t i = 0; i < stats.total.memoryHeapCount; i++) {
        if (stats.total.memoryHeapStats[i].heapFlags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            VkDeviceSize budget = stats.total.memoryHeapStats[i].budget;
            VkDeviceSize usage  = stats.total.memoryHeapStats[i].usage;

            // Stay under 92% of budget
            VkDeviceSize available = (VkDeviceSize)(budget * 0.92) - usage;

            // Use VmaBudget for more detailed info
            VmaBudget budgets[VK_MAX_MEMORY_HEAPS];
            vmaGetHeapBudgets(allocator, budgets);

            VkDeviceSize heap_budget = budgets[i].budget;
            VkDeviceSize heap_usage  = budgets[i].usage;
            VkDeviceSize heap_avail  = (VkDeviceSize)(heap_budget * 0.92) - heap_usage;

            printf("VRAM heap[%u]: budget=%zu MB, usage=%zu MB, avail=%zu MB, required=%zu MB\n",
                   i, heap_budget / (1024*1024), heap_usage / (1024*1024),
                   heap_avail / (1024*1024), required_bytes / (1024*1024));

            return heap_avail >= required_bytes;
        }
    }
    return false;
}
```

## 2.4 Defragmentation

```c
// Inference workload: weights never moved after load, so defragmentation
// is NOT needed during inference. The only time it's relevant is if the
// model is unloaded and a different model loaded.

// For optional future use — defrag on model unload:
void defrag_if_needed(VmaAllocator allocator) {
    VmaDefragmentationInfo2 defrag_info = {};
    defrag_info.flags = 0;

    VmaDefragmentationStats defrag_stats;
    vmaDefragmentationBegin(allocator, &defrag_info, &defrag_stats, NULL);
    // vmaDefragmentationEnd() after completion
}
```

---

---

# PART 3: SPIR-V Compilation Pipeline

---

## 3.1 Compilation Flow

```
┌─────────────────────────────────────────────────────┐
│  Source: shaders/*.comp + common.glsl                │
│                                                      │
│  ┌─────────────────────────────────────────────┐     │
│  │ glslc.exe (Vulkan SDK)                       │     │
│  │   -fshader-stage=compute                     │     │
│  │   --target-env=vulkan1.4                     │     │
│  │   --target-spv=spv1.4                        │     │
│  │   -O (size optimization)                     │     │
│  │   -DSUBGROUP_SIZE=32 (or 64)                 │     │
│  │   -DQUANT_FP16=1 (or Q4_K, Q6_K, etc.)      │     │
│  │   -I shaders/  (include path for common.glsl)│     │
│  │   -o spirv/<name>_w<32|64>.spv               │     │
│  └────────────────────┬────────────────────────┘     │
│                       ↓                              │
│  ┌─────────────────────────────────────────────┐     │
│  │ spirv-opt.exe (optional, Release only)        │     │
│  │   --eliminate-dead-code-aggressive            │     │
│  │   --merge-blocks                              │     │
│  │   --simplify-instructions                     │     │
│  │   --fold-spec-const-op-composite              │     │
│  │   -o spirv/<name>_w32_opt.spv                 │     │
│  └────────────────────┬────────────────────────┘     │
│                       ↓                              │
│  ┌─────────────────────────────────────────────┐     │
│  │ embed_spirv.py                               │     │
│  │   Reads .spv binary                          │     │
│  │   Writes: static const uint32_t xxx_data[]  │     │
│  │   → generated/<name>_w32.h                   │     │
│  └────────────────────┬────────────────────────┘     │
│                       ↓                              │
│  ┌─────────────────────────────────────────────┐     │
│  │ embed_spirv.py --generate-registry           │     │
│  │   → generated/shader_registry.h              │     │
│  │   Index: (op_type, quant_type, wave_size)    │     │
│  │         → embedded_shader_t*                 │     │
│  └─────────────────────────────────────────────┘     │
│                                                      │
│  At runtime:                                         │
│  ┌─────────────────────────────────────────────┐     │
│  │ lookup_shader(op, quant, wave) → data[], size│     │
│  │ vkCreateShaderModule(data, size) → module    │     │
│  │ vkCreateComputePipelines(module, spec_const) │     │
│  └─────────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────┘
```

## 3.2 Wave32 vs Wave64 Strategy

**Dual-compilation**: Each shader is compiled to two SPIR-V variants — one with `#define SUBGROUP_SIZE 32` and one with `#define SUBGROUP_SIZE 64`. The SUBGROUP_SIZE define sets the `layout(local_size_x_id = 0)` specialization constant's default value AND controls `gl_SubgroupSize`.

At pipeline creation, `VK_EXT_subgroup_size_control` is used:

```c
// vk_pipeline.cpp

VkPipeline create_compute_pipeline(VkDevice device,
                                    const embedded_shader_t* shader,
                                    uint32_t required_subgroup_size,
                                    VkPipelineLayout layout,
                                    VkPipelineCache cache) {
    VkShaderModuleCreateInfo smci = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smci.codeSize = shader->byte_size;
    smci.pCode    = shader->data;

    VkShaderModule module;
    VK_CHECK(vkCreateShaderModule(device, &smci, NULL, &module));

    VkPipelineShaderStageCreateInfo ssci = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO
    };
    ssci.stage  = VK_SHADER_STAGE_COMPUTE_BIT;
    ssci.module = module;
    ssci.pName  = "main";

    // Specialization constants: local_size_x and SUBGROUP_SIZE
    uint32_t spec_data[2] = {
        shader->subgroup_size,  // constant_id=0: local_size_x (workgroup X size)
        required_subgroup_size  // constant_id=1: SUBGROUP_SIZE
    };

    VkSpecializationMapEntry spec_entries[2] = {
        {0, 0, sizeof(uint32_t)},  // constant_id=0 → offset 0
        {1, sizeof(uint32_t), sizeof(uint32_t)},  // constant_id=1 → offset 4
    };

    VkSpecializationInfo spec_info = {
        2, spec_entries, sizeof(spec_data), spec_data
    };
    ssci.pSpecializationInfo = &spec_info;

    // Subgroup size control
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup_ci = {
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO
    };
    // Use the SPIR-V's native subgroup size (from the compiled variant)
    // OR override to require a specific size
    // Strategy: match the compiled variant's size exactly
    subgroup_ci.requiredSubgroupSize = shader->subgroup_size;
    ssci.pNext = &subgroup_ci;

    VkComputePipelineCreateInfo cpci = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpci.stage  = ssci;
    cpci.layout = layout;

    VkPipeline pipeline;
    VK_CHECK(vkCreateComputePipelines(device, cache, 1, &cpci, NULL, &pipeline));

    vkDestroyShaderModule(device, module, NULL);

    return pipeline;
}
```

## 3.3 Specialization Constants Embedded in GLSL

```glsl
// common.glsl — shared across all shaders

#version 450
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_KHR_shader_subgroup_ballot : require
#extension GL_EXT_subgroup_size_control : require
#extension GL_KHR_shader_subgroup_clustered : require
#extension GL_EXT_nonuniform_qualifier : require
#extension GL_EXT_scalar_block_layout : require

// Specialization constants
layout(local_size_x_id = 0) in;          // Workgroup X size
layout(constant_id = 1) const uint SPEC_SUBGROUP_SIZE = SUBGROUP_SIZE;  // 32 or 64

// Model dimensions (set at pipeline creation via specialization constants)
layout(constant_id = 2) const uint SPEC_D       = 4096;
layout(constant_id = 3) const uint SPEC_FFN_DIM = 11008;
layout(constant_id = 4) const uint SPEC_HEAD_DIM = 128;
layout(constant_id = 5) const uint SPEC_N_HEADS  = 32;
layout(constant_id = 6) const uint SPEC_N_KV_HEADS = 8;
layout(constant_id = 7) const uint SPEC_VOCAB_SIZE = 128000;

// Quantization types (compile-time — set via #define, not spec constant)
// Allows dead-code elimination for unused quant paths
#ifdef QUANT_FP16
    #define QUANT_TYPE 0
#endif
#ifdef QUANT_Q4_K
    #define QUANT_TYPE 1
#endif
#ifdef QUANT_Q6_K
    #define QUANT_TYPE 2
#endif
#ifdef QUANT_Q8_0
    #define QUANT_TYPE 3
#endif
#ifdef QUANT_IQ4_XS
    #define QUANT_TYPE 4
#endif

// Q4_K block layout (from llama.cpp)
// Block size: 256 elements, 16 scales, 6-bit scale, min, quantized nibbles
struct q4_k_block {
    uint8_t  d;           // scale (fp16: first 2 bytes of d)
    uint8_t  dmin;        // min scale (fp16)
    uint8_t  scales[12];   // 6-bit scales × 16 = 12 bytes
    uint8_t  qs[128];      // 4-bit quantized values × 256 = 128 bytes
};
// Total: 1 + 1 + 12 + 128 = 142 bytes per block (for 256 elements)
// Effective bytes per element: 142 / 256 = 0.5547

// Q8_0 block layout
struct q8_0_block {
    uint8_t  d[2];        // fp16 scale
    uint8_t  qs[32];      // 8-bit quantized values × 32
};
// Total: 2 + 32 = 34 bytes per 32 elements
// Effective bytes per element: 34 / 32 = 1.0625

// IQ4_XS block layout (improved Q4, RDNA4-optimized)
struct iq4_xs_block {
    uint8_t  d[2];        // fp16 scale
    uint8_t  dshift;      // 4-bit shift value
    uint8_t  qs[16];      // 4-bit × 32 elements = 16 bytes
    uint8_t  extra[4];    // sign bits
};
// Total: 2 + 1 + 16 + 4 = 23 bytes per 32 elements
// Effective bytes per element: 23 / 32 = 0.71875

// Subgroup ops
#define SUBGROUP_SIZE gl_SubgroupSize
#define SG_INVOC_ID   gl_SubgroupInvocationID
#define SG_ID         gl_SubgroupID

// Descriptor sets (consistent across all shaders)
// Set 0: Static weights (UPDATE_AFTER_BIND, variable count)
// Set 1: IO buffers (push descriptors)
// Set 2: Tables (rope freqs, alibi slopes — static)

#define WEIGHT_SET  0
#define IO_SET      1
#define TABLE_SET   2

#define BINDING_IO_HIDDEN_IN   0
#define BINDING_IO_HIDDEN_OUT  1
#define BINDING_IO_K_CACHE     2
#define BINDING_IO_V_CACHE     3
#define BINDING_IO_SCRATCH     4
```

## 3.4 Shader Lookup at Runtime

```c
// vk_pipeline.cpp — runtime shader selection

const embedded_shader_t* lookup_shader(int op_type, int quant_type,
                                        uint32_t required_subgroup_size) {
    for (const embedded_shader_t* s = EMBEDDED_SHADERS; s->name != NULL; s++) {
        if (s->op_type == op_type &&
            s->quant_type == quant_type &&
            s->subgroup_size == required_subgroup_size) {
            return s;
        }
    }
    return NULL;
}

// Usage:
const embedded_shader_t* spv = lookup_shader(
    OP_ATTN_QKV, GGUF_QUANT_Q4_K, device->subgroup_size);
if (!spv) { /* fall back to FP16 */ }

VkPipeline pipeline = create_compute_pipeline(
    device->device, spv, device->subgroup_size, layout, cache);
```

---

---

# PART 4: GGUF Parser

---

## 4.1 GGUF File Format Reference

```
GGUF file layout:
┌─────────────────────────────────────────────────────┐
│ Offset 0:  Magic "GGUF" (4 bytes, 0x46475547)       │
│ Offset 4:  Version (uint32, currently 3)            │
│ Offset 8:  Tensor count (uint64)                    │
│ Offset 16: Metadata KV count (uint64)               │
│                                                      │
│ Offset 24: Metadata key-value pairs                  │
│   For each KV:                                       │
│     key:      string (uint64 len + len bytes)       │
│     val_type: uint32 (GGUF_TYPE_*)                   │
│     val:      varies by type                        │
│                                                      │
│ After metadata: Tensor info array                    │
│   For each tensor:                                   │
│     name:      string (uint64 len + len bytes)       │
│     n_dims:    uint32                                │
│     dims[]:    uint64[n_dims]                        │
│     type:      uint32 (GGUF_TYPE_*)                  │
│     offset:    uint64 (offset from start of file)    │
│                                                      │
│ Padding: 0x00 to next GGUF_ALIGNMENT (32) boundary   │
│                                                      │
│ Tensor data: binary at specified offsets             │
└─────────────────────────────────────────────────────┘
```

## 4.2 Core Types

```c
// gguf_types.h

#pragma once
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define GGUF_MAGIC          0x46554747  // "GGUF" little-endian
#define GGUF_VERSION        3
#define GGUF_ALIGNMENT      32
#define GGUF_DEFAULT_ALIGNMENT 32

// GGUF value types
typedef enum gguf_type_t {
    GGUF_TYPE_UINT8   = 0,
    GGUF_TYPE_INT8    = 1,
    GGUF_TYPE_UINT16  = 2,
    GGUF_TYPE_INT16   = 3,
    GGUF_TYPE_UINT32  = 4,
    GGUF_TYPE_INT32   = 5,
    GGUF_TYPE_FLOAT32 = 6,
    GGUF_TYPE_BOOL    = 7,
    GGUF_TYPE_STRING  = 8,
    GGUF_TYPE_ARRAY   = 9,
    GGUF_TYPE_UINT64  = 10,
    GGUF_TYPE_INT64   = 11,
    GGUF_TYPE_FLOAT64 = 12,
} gguf_type_t;

// GGUF tensor type (stored in tensor info, distinct from value types)
typedef enum gguf_tensor_type_t {
    GGUF_TENSOR_F32     = 0,
    GGUF_TENSOR_F16     = 1,
    GGUF_TENSOR_Q4_0    = 2,
    GGUF_TENSOR_Q4_1    = 3,
    GGUF_TENSOR_Q8_0    = 8,
    GGUF_TENSOR_Q8_1    = 9,
    GGUF_TENSOR_Q2_K    = 10,
    GGUF_TENSOR_Q3_K    = 11,
    GGUF_TENSOR_Q4_K    = 12,
    GGUF_TENSOR_Q5_K    = 13,
    GGUF_TENSOR_Q6_K    = 14,
    GGUF_TENSOR_Q8_K    = 15,
    GGUF_TENSOR_IQ4_XS  = 18,  // llama.cpp convention
    // ... other IQ types
} gguf_tensor_type_t;

// Our internal quant type enum (maps to pipeline variants)
typedef enum vk_quant_type_t {
    VK_QUANT_FP16    = 0,
    VK_QUANT_Q4_K    = 1,
    VK_QUANT_Q6_K    = 2,
    VK_QUANT_Q8_0    = 3,
    VK_QUANT_IQ4_XS  = 4,
    VK_QUANT_COUNT   = 5,
} vk_quant_type_t;

// Tensor info from GGUF header
typedef struct gguf_tensor_info_t {
    char*         name;       // null-terminated, owned
    uint32_t      n_dims;
    uint64_t*     dims;       // array of n_dims elements
    gguf_tensor_type_t type;
    uint64_t      offset;     // byte offset in file
    uint64_t      size;       // computed: element_count * type_size / quant_ratio
} gguf_tensor_info_t;

// Parsed GGUF file
typedef struct gguf_file_t {
    int            fd;        // OS file descriptor (mmap target)
    uint8_t*       data;      // mmap'd file contents (read-only)
    size_t         file_size;
    uint32_t       version;
    uint64_t       tensor_count;
    uint64_t       metadata_kv_count;

    // Metadata: simple string→string map (enough for model config)
    // Key pointers: all GGUF metadata values live in the mmap'd region
    // We store pointers into mmap'd data, no heap copies

    gguf_tensor_info_t* tensors;  // array of tensor_count

    // Cached model configuration (extracted from metadata)
    struct {
        char     architecture[64];  // "llama", "qwen2", "gemma", "phi3"
        uint32_t d;                 // hidden dimension
        uint32_t ffn_dim;           // FFN intermediate dimension
        uint32_t n_heads;           // number of attention heads
        uint32_t n_kv_heads;        // number of KV heads (GQA)
        uint32_t head_dim;          // head dimension (d / n_heads typically)
        uint32_t vocab_size;        // vocabulary size
        uint32_t n_layers;          // number of transformer layers
        float    rope_theta;        // RoPE base frequency
        float    norm_eps;          // RMS norm epsilon
        uint32_t context_length;    // max context length
    } config;
} gguf_file_t;

// Forward declarations
bool          gguf_open(const char* path, gguf_file_t* out);
void          gguf_close(gguf_file_t* f);
const uint8_t* gguf_get_tensor_data(const gguf_file_t* f,
                                     uint32_t tensor_idx,
                                     uint64_t* out_size);
int32_t       gguf_find_tensor(const gguf_file_t* f, const char* name);
vk_quant_type_t gguf_tensor_to_vk_quant(gguf_tensor_type_t tt);
```

## 4.3 GGUF Parser Implementation

```c
// gguf_parser.cpp

#ifdef _WIN32
    #define NOMINMAX
    #include <windows.h>
#else
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gguf_types.h"

// ─── Portable mmap wrapper ──────────────────────────────────────────
#ifdef _WIN32
typedef struct mmap_ctx_t {
    HANDLE hFile;
    HANDLE hMapping;
} mmap_ctx_t;

static void* mmap_file(const char* path, size_t* out_size, mmap_ctx_t* ctx) {
    ctx->hFile = CreateFileA(path, GENERIC_READ, FILE_SHARE_READ,
                              NULL, OPEN_EXISTING,
                              FILE_ATTRIBUTE_NORMAL, NULL);
    if (ctx->hFile == INVALID_HANDLE_VALUE) return NULL;

    LARGE_INTEGER li;
    GetFileSizeEx(ctx->hFile, &li);
    *out_size = (size_t)li.QuadPart;

    ctx->hMapping = CreateFileMappingA(ctx->hFile, NULL,
                                        PAGE_READONLY, 0, 0, NULL);
    if (!ctx->hMapping) {
        CloseHandle(ctx->hFile);
        return NULL;
    }

    void* ptr = MapViewOfFile(ctx->hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!ptr) {
        CloseHandle(ctx->hMapping);
        CloseHandle(ctx->hFile);
        return NULL;
    }
    return ptr;
}

static void munmap_file(void* ptr, size_t size, mmap_ctx_t* ctx) {
    UnmapViewOfFile(ptr);
    CloseHandle(ctx->hMapping);
    CloseHandle(ctx->hFile);
}
#else
typedef int mmap_ctx_t;

static void* mmap_file(const char* path, size_t* out_size, mmap_ctx_t* ctx) {
    *ctx = open(path, O_RDONLY);
    if (*ctx < 0) return NULL;

    struct stat st;
    fstat(*ctx, &st);
    *out_size = st.st_size;

    void* ptr = mmap(NULL, *out_size, PROT_READ, MAP_PRIVATE, *ctx, 0);
    if (ptr == MAP_FAILED) {
        close(*ctx);
        return NULL;
    }
    return ptr;
}

static void munmap_file(void* ptr, size_t size, mmap_ctx_t* ctx) {
    munmap(ptr, size);
    close(*ctx);
}
#endif

// ─── GGUF reader helpers ────────────────────────────────────────────

struct gguf_reader_t {
    const uint8_t* data;
    size_t         size;
    size_t         pos;
};

static uint32_t read_u32(struct gguf_reader_t* r) {
    uint32_t v;
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return v;
}

static uint64_t read_u64(struct gguf_reader_t* r) {
    uint64_t v;
    memcpy(&v, r->data + r->pos, 8);
    r->pos += 8;
    return v;
}

static float read_f32(struct gguf_reader_t* r) {
    float v;
    memcpy(&v, r->data + r->pos, 4);
    r->pos += 4;
    return v;
}

static bool read_bool(struct gguf_reader_t* r) {
    return r->data[r->pos++] != 0;
}

static uint64_t read_string(struct gguf_reader_t* r,
                              const char** out_str) {
    uint64_t len = read_u64(r);
    *out_str = (const char*)(r->data + r->pos);
    r->pos += len;
    return len;
}

static void skip_value(struct gguf_reader_t* r, uint32_t type) {
    switch (type) {
        case GGUF_TYPE_UINT8:
        case GGUF_TYPE_INT8:
        case GGUF_TYPE_BOOL:
            r->pos += 1; break;
        case GGUF_TYPE_UINT16:
        case GGUF_TYPE_INT16:
            r->pos += 2; break;
        case GGUF_TYPE_UINT32:
        case GGUF_TYPE_INT32:
        case GGUF_TYPE_FLOAT32:
            r->pos += 4; break;
        case GGUF_TYPE_UINT64:
        case GGUF_TYPE_INT64:
        case GGUF_TYPE_FLOAT64:
            r->pos += 8; break;
        case GGUF_TYPE_STRING: {
            uint64_t len = read_u64(r);
            r->pos += len;
            break;
        }
        case GGUF_TYPE_ARRAY: {
            uint32_t array_type = read_u32(r);
            uint64_t array_len  = read_u64(r);
            for (uint64_t i = 0; i < array_len; i++) {
                skip_value(r, array_type);
            }
            break;
        }
    }
}

static uint64_t align_offset(uint64_t offset, uint64_t alignment) {
    return (offset + alignment - 1) & ~(alignment - 1);
}

// ─── Main GGUF parser ───────────────────────────────────────────────

bool gguf_open(const char* path, gguf_file_t* out) {
    memset(out, 0, sizeof(*out));

    mmap_ctx_t ctx;
    out->data = (uint8_t*)mmap_file(path, &out->file_size, &ctx);
    if (!out->data) return false;

    struct gguf_reader_t r = {out->data, out->file_size, 0};

    // Header
    uint32_t magic = read_u32(&r);
    if (magic != GGUF_MAGIC) {
        munmap_file(out->data, out->file_size, &ctx);
        return false;
    }

    out->version = read_u32(&r);
    out->tensor_count     = read_u64(&r);
    out->metadata_kv_count = read_u64(&r);

    // Parse metadata
    for (uint64_t i = 0; i < out->metadata_kv_count; i++) {
        const char* key;
        uint64_t key_len = read_string(&r, &key);
        uint32_t val_type = read_u32(&r);

        // Extract model config from known keys
        // key is NOT null-terminated in the file buffer — compare via strncmp

        #define KEY_MATCHES(literal) \
            (key_len == (sizeof(literal)-1) && memcmp(key, literal, key_len) == 0)

        if (KEY_MATCHES("general.architecture")) {
            uint64_t arch_len = read_string(&r, (const char**)&key);
            size_t copy_len = arch_len < sizeof(out->config.architecture)-1
                            ? arch_len : sizeof(out->config.architecture)-1;
            memcpy(out->config.architecture, key, copy_len);
            out->config.architecture[copy_len] = '\0';
        }
        else if (KEY_MATCHES("llama.block_count")) {
            out->config.n_layers = read_u32(&r);
        }
        else if (KEY_MATCHES("llama.embedding_length")) {
            out->config.d = read_u32(&r);
        }
        else if (KEY_MATCHES("llama.feed_forward_length")) {
            out->config.ffn_dim = read_u32(&r);
        }
        else if (KEY_MATCHES("llama.attention.head_count")) {
            out->config.n_heads = read_u32(&r);
        }
        else if (KEY_MATCHES("llama.attention.head_count_kv")) {
            out->config.n_kv_heads = read_u32(&r);
        }
        else if (KEY_MATCHES("llama.context_length")) {
            out->config.context_length = read_u32(&r);
        }
        else if (KEY_MATCHES("llama.rope.theta")) {
            out->config.rope_theta = read_f32(&r);
        }
        else if (KEY_MATCHES("llama.attention.layer_norm_rms_epsilon")) {
            out->config.norm_eps = read_f32(&r);
        }
        else {
            skip_value(&r, val_type);
        }
        #undef KEY_MATCHES
    }

    // Compute head_dim if not explicitly in metadata
    if (out->config.n_heads > 0 && out->config.d > 0) {
        out->config.head_dim = out->config.d / out->config.n_heads;
    }

    // If vocab_size not in metadata, will be set after parsing tensors
    // (from embedding table dimensions)

    // Parse tensor info
    out->tensors = (gguf_tensor_info_t*)calloc(
        out->tensor_count, sizeof(gguf_tensor_info_t));

    for (uint64_t i = 0; i < out->tensor_count; i++) {
        gguf_tensor_info_t* t = &out->tensors[i];

        const char* name_ptr;
        uint64_t name_len = read_string(&r, &name_ptr);
        t->name = (char*)malloc(name_len + 1);
        memcpy(t->name, name_ptr, name_len);
        t->name[name_len] = '\0';

        t->n_dims = read_u32(&r);
        t->dims   = (uint64_t*)malloc(t->n_dims * sizeof(uint64_t));
        uint64_t total_elements = 1;
        for (uint32_t d = 0; d < t->n_dims; d++) {
            t->dims[d] = read_u64(&r);
            total_elements *= t->dims[d];
        }

        t->type   = (gguf_tensor_type_t)read_u32(&r);
        t->offset = read_u64(&r);

        // Compute byte size
        int32_t type_size = 0;
        float quant_ratio = 1.0f;
        switch (t->type) {
            case GGUF_TENSOR_F32:  type_size = 4; break;
            case GGUF_TENSOR_F16:  type_size = 2; break;
            case GGUF_TENSOR_Q4_0: type_size = 1; quant_ratio = 0.5f;  break;
            case GGUF_TENSOR_Q4_1: type_size = 1; quant_ratio = 0.5f;  break;
            case GGUF_TENSOR_Q8_0: type_size = 1; quant_ratio = 1.0f;  break;
            case GGUF_TENSOR_Q8_1: type_size = 1; quant_ratio = 1.0f;  break;
            case GGUF_TENSOR_Q4_K: type_size = 1; quant_ratio = 0.5625f; break;
            case GGUF_TENSOR_Q6_K: type_size = 1; quant_ratio = 0.84375f; break;
            case GGUF_TENSOR_Q8_K: type_size = 1; quant_ratio = 1.0f;  break;
            case GGUF_TENSOR_IQ4_XS: type_size = 1; quant_ratio = 0.71875f; break;
            // Q2_K, Q3_K, Q5_K omitted for brevity — add as needed
            default:
                fprintf(stderr, "Warning: unknown tensor type %d for %s\n",
                        t->type, t->name);
                type_size = 1;
                break;
        }
        t->size = (uint64_t)(total_elements * type_size * quant_ratio);
        if (t->size == 0) t->size = total_elements * sizeof(float); // fallback
    }

    // Detect vocab_size from embedding tensor if not in metadata
    if (out->config.vocab_size == 0) {
        int32_t emb_idx = gguf_find_tensor(out, "token_embd.weight");
        if (emb_idx >= 0 && out->tensors[emb_idx].n_dims >= 2) {
            out->config.vocab_size = (uint32_t)out->tensors[emb_idx].dims[0];
        }
    }

    return true;
}

void gguf_close(gguf_file_t* f) {
    if (!f->data) return;
    mmap_ctx_t ctx;  // dummy — we just need to pass something
#ifdef _WIN32
    UnmapViewOfFile(f->data);
#else
    munmap(f->data, f->file_size);
#endif
    if (f->tensors) {
        for (uint64_t i = 0; i < f->tensor_count; i++) {
            free(f->tensors[i].name);
            free(f->tensors[i].dims);
        }
        free(f->tensors);
    }
    memset(f, 0, sizeof(*f));
}

const uint8_t* gguf_get_tensor_data(const gguf_file_t* f,
                                     uint32_t tensor_idx,
                                     uint64_t* out_size) {
    if (tensor_idx >= f->tensor_count) return NULL;
    *out_size = f->tensors[tensor_idx].size;
    return f->data + f->tensors[tensor_idx].offset;
}

int32_t gguf_find_tensor(const gguf_file_t* f, const char* name) {
    for (uint64_t i = 0; i < f->tensor_count; i++) {
        if (strcmp(f->tensors[i].name, name) == 0) {
            return (int32_t)i;
        }
    }
    return -1;
}

vk_quant_type_t gguf_tensor_to_vk_quant(gguf_tensor_type_t tt) {
    switch (tt) {
        case GGUF_TENSOR_F16:   return VK_QUANT_FP16;
        case GGUF_TENSOR_F32:   return VK_QUANT_FP16;  // treat F32 as FP16
        case GGUF_TENSOR_Q4_K:  return VK_QUANT_Q4_K;
        case GGUF_TENSOR_Q6_K:  return VK_QUANT_Q6_K;
        case GGUF_TENSOR_Q8_0:  return VK_QUANT_Q8_0;
        case GGUF_TENSOR_IQ4_XS: return VK_QUANT_IQ4_XS;
        default:
            fprintf(stderr, "Unsupported quant type %d, falling back to FP16\n", tt);
            return VK_QUANT_FP16;
    }
}
```

---

---

# PART 5: Logits Sampling

---

```c
// sample.h / sample.cpp

#pragma once
#include <stdint.h>

// xoshiro256++ — fast, deterministic PRNG
typedef struct xoshiro256pp_t {
    uint64_t s[4];
} xoshiro256pp_t;

// Sampling parameters
typedef struct sampler_params_t {
    float temperature;      // 0.0 = greedy, 1.0 = neutral, >1.0 = creative
    int   top_k;           // 0 = disabled, >0 = keep top K
    float top_p;           // 0.0 = disabled, 0.0-1.0 = nucleus
    float min_p;           // 0.0 = disabled, >0.0 = probability floor
    float repetition_penalty; // 1.0 = disabled, >1.0 = penalize repeats
} sampler_params_t;

// API
void    xoshiro256pp_seed(xoshiro256pp_t* rng, uint64_t seed);
uint64_t xoshiro256pp_next(xoshiro256pp_t* rng);
float   xoshiro256pp_next_f32(xoshiro256pp_t* rng);  // uniform [0, 1)

int32_t sample_token(const float* logits, uint32_t vocab_size,
                     const sampler_params_t* params,
                     xoshiro256pp_t* rng);
```

```c
// sample.cpp

#include "sample.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// ─── xoshiro256++ ───────────────────────────────────────────────────

static inline uint64_t rotl(uint64_t x, int k) {
    return (x << k) | (x >> (64 - k));
}

void xoshiro256pp_seed(xoshiro256pp_t* rng, uint64_t seed) {
    // SplitMix64 seeding
    uint64_t z = seed + 0x9e3779b97f4a7c15ULL;
    for (int i = 0; i < 4; i++) {
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        rng->s[i] = z ^ (z >> 31);
    }
}

uint64_t xoshiro256pp_next(xoshiro256pp_t* rng) {
    uint64_t result = rotl(rng->s[0] + rng->s[3], 23) + rng->s[0];
    uint64_t t = rng->s[1] << 17;
    rng->s[2] ^= rng->s[0];
    rng->s[3] ^= rng->s[1];
    rng->s[1] ^= rng->s[2];
    rng->s[0] ^= rng->s[3];
    rng->s[2] ^= t;
    rng->s[3] = rotl(rng->s[3], 45);
    return result;
}

float xoshiro256pp_next_f32(xoshiro256pp_t* rng) {
    // Top 23 bits of random 64-bit int → uniform float [0, 1)
    return (float)(xoshiro256pp_next(rng) >> 40) / (float)(1 << 24);
}

// ─── Sampling ───────────────────────────────────────────────────────

// Helper: structure for sorted indices
typedef struct {
    int32_t index;
    float   prob;
} token_prob_t;

static int compare_token_prob_desc(const void* a, const void* b) {
    float diff = ((const token_prob_t*)b)->prob - ((const token_prob_t*)a)->prob;
    if (diff > 0.0f) return 1;
    if (diff < 0.0f) return -1;
    return 0;
}

int32_t sample_token(const float* logits, uint32_t vocab_size,
                     const sampler_params_t* params,
                     xoshiro256pp_t* rng) {

    // ─── Greedy: temperature == 0 ───────────────────────────────
    if (params->temperature <= 0.0f) {
        float max_val = logits[0];
        int32_t max_idx = 0;
        for (uint32_t i = 1; i < vocab_size; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = (int32_t)i;
            }
        }
        return max_idx;
    }

    // ─── Temperature scaling ────────────────────────────────────
    float* probs = (float*)malloc(vocab_size * sizeof(float));
    float inv_temp = 1.0f / params->temperature;

    float max_logit = logits[0];
    for (uint32_t i = 1; i < vocab_size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    float sum = 0.0f;
    for (uint32_t i = 0; i < vocab_size; i++) {
        probs[i] = expf((logits[i] - max_logit) * inv_temp);
        sum += probs[i];
    }

    // Normalize
    float inv_sum = 1.0f / sum;
    for (uint32_t i = 0; i < vocab_size; i++) {
        probs[i] *= inv_sum;
    }

    // ─── Top-K filtering ────────────────────────────────────────
    if (params->top_k > 0 && (uint32_t)params->top_k < vocab_size) {
        // Partial sort: create array of (index, prob), sort, keep top K
        token_prob_t* sorted = (token_prob_t*)malloc(vocab_size * sizeof(token_prob_t));
        uint32_t count = 0;
        for (uint32_t i = 0; i < vocab_size; i++) {
            if (probs[i] > 0.0f) {
                sorted[count].index = (int32_t)i;
                sorted[count].prob  = probs[i];
                count++;
            }
        }
        // Sort by probability descending
        qsort(sorted, count, sizeof(token_prob_t), compare_token_prob_desc);

        // Zero out everything below top-K
        uint32_t keep = (uint32_t)params->top_k;
        if (keep > count) keep = count;

        memset(probs, 0, vocab_size * sizeof(float));
        for (uint32_t i = 0; i < keep; i++) {
            probs[sorted[i].index] = sorted[i].prob;
        }
        free(sorted);
    }

    // ─── Top-P (nucleus) filtering ──────────────────────────────
    if (params->top_p > 0.0f && params->top_p < 1.0f) {
        token_prob_t* sorted = (token_prob_t*)malloc(vocab_size * sizeof(token_prob_t));
        uint32_t count = 0;
        for (uint32_t i = 0; i < vocab_size; i++) {
            if (probs[i] > 0.0f) {
                sorted[count].index = (int32_t)i;
                sorted[count].prob  = probs[i];
                count++;
            }
        }
        qsort(sorted, count, sizeof(token_prob_t), compare_token_prob_desc);

        float cumsum = 0.0f;
        uint32_t keep = 0;
        for (uint32_t i = 0; i < count; i++) {
            cumsum += sorted[i].prob;
            keep++;
            if (cumsum >= params->top_p) break;
        }

        memset(probs, 0, vocab_size * sizeof(float));
        for (uint32_t i = 0; i < keep; i++) {
            probs[sorted[i].index] = sorted[i].prob;
        }
        free(sorted);
    }

    // ─── Min-P filtering ────────────────────────────────────────
    if (params->min_p > 0.0f) {
        float max_prob = 0.0f;
        for (uint32_t i = 0; i < vocab_size; i++) {
            if (probs[i] > max_prob) max_prob = probs[i];
        }
        float threshold = max_prob * params->min_p;
        for (uint32_t i = 0; i < vocab_size; i++) {
            if (probs[i] < threshold) probs[i] = 0.0f;
        }
    }

    // ─── Renormalize ────────────────────────────────────────────
    sum = 0.0f;
    for (uint32_t i = 0; i < vocab_size; i++) sum += probs[i];

    if (sum <= 0.0f) {
        // All tokens filtered out — fall back to argmax
        free(probs);
        float max_val = logits[0];
        int32_t max_idx = 0;
        for (uint32_t i = 1; i < vocab_size; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = (int32_t)i;
            }
        }
        return max_idx;
    }

    inv_sum = 1.0f / sum;
    for (uint32_t i = 0; i < vocab_size; i++) probs[i] *= inv_sum;

    // ─── Multinomial sampling ───────────────────────────────────
    float r = xoshiro256pp_next_f32(rng);
    float cdf = 0.0f;
    int32_t chosen = 0;
    for (uint32_t i = 0; i < vocab_size; i++) {
        cdf += probs[i];
        if (r <= cdf) {
            chosen = (int32_t)i;
            break;
        }
    }

    free(probs);
    return chosen;
}
```

---

---

# PART 6: Main Entry Point

---

```c
// main.cpp

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "common.h"
#include "vk_device.h"
#include "vk_session.h"
#include "vk_model.h"
#include "vk_model_load.h"
#include "vk_timeline.h"
#include "gguf_types.h"
#include "gguf_parser.h"
#include "sample.h"

#ifdef _WIN32
    #define NOMINMAX
    #include <windows.h>
#else
    #include <sys/time.h>
#endif

// ─── Platform timer ──────────────────────────────────────────────────

static double get_time_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_sec * 1000.0 + (double)tv.tv_usec / 1000.0;
#endif
}

// ─── Tokenizer (minimal GGUF-based tokenizer) ────────────────────────
// Loads token list from GGUF metadata: tokenizer.ggml.tokens
// Assumes BPE/SPM tokens stored as metadata array
// For simplicity in this initial implementation: loads from GGUF metadata
// string array "tokenizer.ggml.tokens"

typedef struct simple_tokenizer_t {
    char** tokens;      // token_id → string
    uint32_t vocab_size;
    // BPE merge table + vocab mapping omitted for brevity
    // Full implementation requires SentencePiece/BPE decoding
} simple_tokenizer_t;

// CLI argument parsing
typedef struct cli_args_t {
    char    model_path[512];
    char    prompt[8192];
    int32_t max_tokens;
    float   temperature;
    int32_t top_k;
    float   top_p;
    float   min_p;
    int32_t seed;
    bool    interactive;  // -i flag for chat mode
    bool    verbose;      // -v for debug output
} cli_args_t;

static void print_usage(const char* prog) {
    printf("Usage: %s --model <path> [options]\n\n", prog);
    printf("Options:\n");
    printf("  --model <path>        GGUF model file (required)\n");
    printf("  --prompt <text>       Input prompt (default: \"Hello\")\n");
    printf("  --max-tokens <N>      Max tokens to generate (default: 256)\n");
    printf("  --temp <float>        Temperature (default: 0.8, 0=greedy)\n");
    printf("  --top-k <N>           Top-K sampling (default: 40, 0=off)\n");
    printf("  --top-p <float>       Top-P nucleus sampling (default: 0.95, 0=off)\n");
    printf("  --min-p <float>       Min-P floor (default: 0.0, off)\n");
    printf("  --seed <N>            RNG seed (default: time-based)\n");
    printf("  -i, --interactive     Interactive chat mode\n");
    printf("  -v, --verbose         Verbose debug output\n");
}

static bool parse_args(int argc, char* argv[], cli_args_t* args) {
    memset(args, 0, sizeof(*args));
    strcpy(args->prompt, "Hello, world!");
    args->max_tokens   = 256;
    args->temperature  = 0.8f;
    args->top_k        = 40;
    args->top_p        = 0.95f;
    args->min_p        = 0.0f;
    args->seed         = -1;  // time-based

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            strncpy(args->model_path, argv[++i], sizeof(args->model_path) - 1);
        } else if (strcmp(argv[i], "--prompt") == 0 && i + 1 < argc) {
            strncpy(args->prompt, argv[++i], sizeof(args->prompt) - 1);
        } else if (strcmp(argv[i], "--max-tokens") == 0 && i + 1 < argc) {
            args->max_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--temp") == 0 && i + 1 < argc) {
            args->temperature = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--top-k") == 0 && i + 1 < argc) {
            args->top_k = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--top-p") == 0 && i + 1 < argc) {
            args->top_p = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--min-p") == 0 && i + 1 < argc) {
            args->min_p = (float)atof(argv[++i]);
        } else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) {
            args->seed = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-i") == 0 || strcmp(argv[i], "--interactive") == 0) {
            args->interactive = true;
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            args->verbose = true;
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_usage(argv[0]);
            return false;
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            print_usage(argv[0]);
            return false;
        }
    }

    if (args->model_path[0] == '\0') {
        fprintf(stderr, "Error: --model is required\n");
        print_usage(argv[0]);
        return false;
    }

    return true;
}

// ─── Main ────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    printf("=== AMD RDNA4/RDNA2 Vulkan LLM Inference Engine ===\n\n");

    cli_args_t args;
    if (!parse_args(argc, argv, &args)) return 1;

    // ─── RNG setup ──────────────────────────────────────────────
    xoshiro256pp_t rng;
    if (args.seed < 0) {
        args.seed = (int32_t)time(NULL);
    }
    xoshiro256pp_seed(&rng, (uint64_t)args.seed);

    // ─── Sampler params ─────────────────────────────────────────
    sampler_params_t sampler = {
        .temperature        = args.temperature,
        .top_k              = args.top_k,
        .top_p              = args.top_p,
        .min_p              = args.min_p,
        .repetition_penalty = 1.0f,
    };

    // ─── GGUF: Open and parse ───────────────────────────────────
    printf("[1/6] Parsing GGUF model: %s\n", args.model_path);

    gguf_file_t gguf;
    if (!gguf_open(args.model_path, &gguf)) {
        fprintf(stderr, "ERROR: Failed to open GGUF file: %s\n", args.model_path);
        return 1;
    }

    printf("  Architecture: %s\n", gguf.config.architecture);
    printf("  Layers:       %u\n", gguf.config.n_layers);
    printf("  Dim:          %u\n", gguf.config.d);
    printf("  FFN Dim:      %u\n", gguf.config.ffn_dim);
    printf("  Heads:        %u (KV: %u)\n", gguf.config.n_heads, gguf.config.n_kv_heads);
    printf("  Head Dim:     %u\n", gguf.config.head_dim);
    printf("  Vocab Size:   %u\n", gguf.config.vocab_size);
    printf("  Tensors:      %llu\n", (unsigned long long)gguf.tensor_count);

    // ─── Vulkan Init: Device + Queues + VMA ─────────────────────
    printf("\n[2/6] Initializing Vulkan device...\n");

    vk_device_t device;
    if (!vk_device_init(&device, gguf.config.vocab_size * sizeof(float))) {
        fprintf(stderr, "ERROR: Vulkan device initialization failed\n");
        gguf_close(&gguf);
        return 1;
    }

    printf("  GPU:           %s\n", device.props.deviceName);
    printf("  Vulkan:        %d.%d.%d\n",
           VK_API_VERSION_MAJOR(device.props.apiVersion),
           VK_API_VERSION_MINOR(device.props.apiVersion),
           VK_API_VERSION_PATCH(device.props.apiVersion));
    printf("  Subgroup size: %u\n", device.subgroup_size);
    printf("  Max shared mem: %u KB\n", device.max_shared_memory / 1024);
    printf("  VRAM:          %zu MB\n",
           (size_t)(device.props.limits.maxMemoryAllocationCount > 0
                    ? 15920 : 0));  // use budget check instead

    // ─── Session: Allocate buffers, descriptor sets, timeline ───
    printf("\n[3/6] Creating inference session...\n");

    vk_session_t session;
    if (!vk_session_init(&session, &device, &gguf.config)) {
        fprintf(stderr, "ERROR: Session initialization failed\n");
        vk_device_destroy(&device);
        gguf_close(&gguf);
        return 1;
    }

    // ─── Model: Upload weights from GGUF to VRAM ────────────────
    printf("[4/6] Uploading model weights to VRAM...\n");
    double t_upload_start = get_time_ms();

    // Check VRAM budget
    uint64_t model_size_estimate = 0;
    for (uint64_t i = 0; i < gguf.tensor_count; i++) {
        model_size_estimate += gguf.tensors[i].size;
    }
    model_size_estimate += gguf.config.n_layers * gguf.config.pages_per_layer
                         * gguf.config.page_stride_bytes;  // KV cache

    if (!vk_check_vram(&device, model_size_estimate)) {
        fprintf(stderr, "ERROR: Insufficient VRAM. Need ~%llu MB\n",
                (unsigned long long)(model_size_estimate / (1024*1024)));
        vk_session_destroy(&session);
        vk_device_destroy(&device);
        gguf_close(&gguf);
        return 1;
    }

    if (!vk_model_load_weights(&session, &gguf)) {
        fprintf(stderr, "ERROR: Weight upload failed\n");
        vk_session_destroy(&session);
        vk_device_destroy(&device);
        gguf_close(&gguf);
        return 1;
    }

    double t_upload_end = get_time_ms();
    printf("  Upload complete: %.0f ms\n", t_upload_end - t_upload_start);

    // ─── Pipelines: Create compute pipelines ────────────────────
    printf("[5/6] Compiling compute pipelines...\n");
    double t_pipeline_start = get_time_ms();

    if (!vk_session_create_pipelines(&session)) {
        fprintf(stderr, "ERROR: Pipeline creation failed\n");
        vk_session_destroy(&session);
        vk_device_destroy(&device);
        gguf_close(&gguf);
        return 1;
    }

    double t_pipeline_end = get_time_ms();
    printf("  Pipelines ready: %.0f ms\n", t_pipeline_end - t_pipeline_start);

    // ─── Prefill: Process prompt ────────────────────────────────
    printf("\n[6/6] Prefilling prompt: \"%s\"\n", args.prompt);
    printf("──────────────────────────────────────────────────\n");

    // Tokenize prompt (simplified — real tokenizer would use GGUF metadata)
    // For now, treat each character as a placeholder token ID
    int32_t prompt_len = (int32_t)strlen(args.prompt);
    printf("[Prompt: %d tokens]\n", prompt_len);

    double t0 = get_time_ms();

    // Prefill phase
    vk_session_prefill(&session, (const uint8_t*)args.prompt, prompt_len);

    double t_prefill_end = get_time_ms();
    printf("\nPrefill: %.1f ms (%.1f tok/s)\n\n",
           t_prefill_end - t0,
           prompt_len / ((t_prefill_end - t0) / 1000.0));

    // ─── Decode Loop ────────────────────────────────────────────
    printf("Generating...\n");
    fflush(stdout);

    double total_decode_time = 0.0;
    int32_t tokens_generated = 0;
    int32_t next_token = 0;

    // Decode loop
    for (int32_t step = 0; step < args.max_tokens; step++) {
        double t_decode_start = get_time_ms();

        // Run one decode step (timeline-synchronized)
        float* logits = vk_session_decode_step(&session);

        double t_decode_end = get_time_ms();
        total_decode_time += (t_decode_end - t_decode_start);

        // Sample next token
        next_token = sample_token(logits, gguf.config.vocab_size, &sampler, &rng);

        // Print token (hex ID for now; real tokenizer would decode to string)
        if (args.verbose) {
            printf("%d ", next_token);
        } else {
            // Placeholder: just print the token ID in brackets
            // Real implementation would use the tokenizer to decode to UTF-8
            printf("[%d]", next_token);
        }
        fflush(stdout);

        tokens_generated++;

        // Check for EOS
        if (next_token == 2) {  // EOS token ID (model-dependent)
            break;
        }
    }

    double t_total_end = get_time_ms();

    // ─── Stats ──────────────────────────────────────────────────
    printf("\n\n──────────────────────────────────────────────────\n");
    printf("Generation complete.\n");
    printf("Tokens generated: %d\n", tokens_generated);
    printf("Decode time:      %.1f ms\n", total_decode_time);
    printf("Avg ms/token:     %.2f ms\n",
           tokens_generated > 0 ? total_decode_time / tokens_generated : 0.0);
    printf("Tokens/sec:       %.1f tok/s\n",
           tokens_generated > 0 ? tokens_generated / (total_decode_time / 1000.0) : 0.0);
    printf("Total runtime:    %.0f ms\n", t_total_end - t0);

    // ─── Cleanup ────────────────────────────────────────────────
    vk_session_destroy(&session);
    vk_device_destroy(&device);
    gguf_close(&gguf);

    printf("Shutdown complete.\n");
    return 0;
}
```

---

---

# PART 7: Key API Signatures & Struct Definitions

---

## 7.1 vk_device.h

```c
#pragma once
#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"

#define MAX_LAYERS 128

typedef struct vk_device_t {
    VkInstance                instance;
    VkPhysicalDevice          physical_device;
    VkDevice                  device;
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceMemoryProperties mem_props;

    uint32_t                  compute_qf_idx;
    uint32_t                  transfer_qf_idx;
    VkQueue                   compute_queue;
    VkQueue                   transfer_queue;

    bool has_descriptor_indexing;
    bool has_timeline_semaphore;
    bool has_push_descriptor;
    bool has_bda;
    bool has_subgroup_size_control;
    uint32_t subgroup_size;
    uint32_t max_push_descriptors;
    uint32_t max_shared_memory;

    VmaAllocator allocator;
    VkCommandPool compute_cmd_pool;
    VkCommandPool transfer_cmd_pool;
    VkPipelineCache pipeline_cache;
} vk_device_t;

bool vk_device_init(vk_device_t* dev, VkDeviceSize min_vram_for_logits);
void vk_device_destroy(vk_device_t* dev);
```

## 7.2 vk_buffer.h

```cpp
// (See Part 2 above — complete definitions provided there)
```

## 7.3 vk_pipeline.h

```c
#pragma once
#include <vulkan/vulkan.h>

typedef struct vk_pipeline_t {
    VkPipeline           pipeline;
    VkPipelineLayout     layout;
    VkShaderModule       module;
    VkDescriptorSetLayout set_layouts[3];  // [0]=weights, [1]=IO(push), [2]=tables
    uint32_t             push_constant_size;
    uint32_t             workgroup_x, workgroup_y, workgroup_z;
} vk_pipeline_t;

typedef enum vk_op_type_t {
    OP_RMS_NORM = 0,
    OP_ATTN_QKV,
    OP_ATTN_COMPUTE,
    OP_ATTN_OUTPUT,
    OP_FFN_GATE_UP,
    OP_FFN_DOWN,
    OP_LM_HEAD,
    OP_TOKEN_EMBED,
    OP_COUNT
} vk_op_type_t;

const embedded_shader_t* lookup_shader(int op_type, int quant_type,
                                        uint32_t subgroup_size);
VkPipeline vk_create_compute_pipeline(VkDevice device,
                                       const embedded_shader_t* shader,
                                       VkPipelineLayout layout,
                                       uint32_t required_subgroup_size,
                                       const VkSpecializationInfo* spec_info,
                                       VkPipelineCache cache);
void vk_destroy_pipeline(VkDevice device, vk_pipeline_t* pipeline);
```

## 7.4 vk_session.h

```c
#pragma once
#include "vk_device.h"
#include "vk_pipeline.h"
#include "vk_model.h"
#include "vk_timeline.h"
#include "vk_descriptor.h"
#include "gguf_types.h"

#define MAX_PIPELINE_VARIANTS (OP_COUNT * VK_QUANT_COUNT)

typedef struct vk_decode_state_t {
    VkCommandBuffer cb;
    vk_buffer_t     hidden_buf[2];
    uint32_t        hidden_toggle;
    vk_buffer_t     logits_buf;
    vk_buffer_t     norm_scratch;
    vk_buffer_t     qkv_scratch;
    vk_buffer_t     attn_scratch;
    vk_buffer_t     ffn_scratch;
    uint32_t        current_pos;    // current KV cache position
    uint32_t        total_seq_len;  // prompt + generated
} vk_decode_state_t;

typedef struct vk_session_t {
    vk_device_t*           device;
    vk_model_t             model;
    vk_timeline_t          timeline;
    vk_descriptor_arena_t  desc;
    VkFence                transfer_fence;

    vk_pipeline_t pipelines[MAX_PIPELINE_VARIANTS];
    int            pipeline_count;

    vk_decode_state_t  decode_state;
    vk_quant_type_t    weight_quant;  // detected from first weight tensor
} vk_session_t;

bool vk_session_init(vk_session_t* session, vk_device_t* device,
                     const struct /* gguf_config_t */ *config);
void vk_session_destroy(vk_session_t* session);
bool vk_session_create_pipelines(vk_session_t* session);
void vk_session_prefill(vk_session_t* session, const uint8_t* prompt, int32_t len);
float* vk_session_decode_step(vk_session_t* session);
```

## 7.5 vk_model.h

```c
#pragma once
#include "vk_buffer.h"
#include "gguf_types.h"

#define MAX_LAYERS 128
#define PAGE_SIZE_TOKENS 256

typedef struct vk_layer_weight_block_t {
    uint64_t q_offset;
    uint64_t k_offset;
    uint64_t v_offset;
    uint64_t o_offset;
    uint64_t attn_norm_offset;
    uint64_t ffn_norm_offset;
    uint64_t gate_offset;
    uint64_t up_offset;
    uint64_t down_offset;
} vk_layer_weight_block_t;

typedef struct vk_kv_cache_t {
    vk_buffer_t buffer;
    uint32_t    page_size_tokens;
    uint32_t    pages_per_layer;
    uint32_t    total_pages;
    uint64_t    page_stride_bytes;  // K+V for all heads per page
    vk_buffer_t page_table;
} vk_kv_cache_t;

typedef struct vk_model_t {
    struct {
        uint32_t d, ffn_dim, n_heads, n_kv_heads, head_dim;
        uint32_t vocab_size, n_layers;
        float    rope_theta, norm_eps;
    } config;

    vk_buffer_t              weight_buffers[MAX_LAYERS];
    vk_buffer_t              embedding_buffer;
    vk_buffer_t              lm_head_buffer;
    vk_kv_cache_t            kv_cache;
    vk_layer_weight_block_t  weight_layout[MAX_LAYERS];
    vk_quant_type_t          weight_quant;
} vk_model_t;

// vk_model_load.cpp
bool vk_model_load_weights(vk_session_t* session, const gguf_file_t* gguf);
bool vk_check_vram(vk_device_t* dev, uint64_t required_bytes);
```

## 7.6 vk_timeline.h

```c
#pragma once
#include <vulkan/vulkan.h>

typedef struct vk_timeline_t {
    VkSemaphore semaphore;
    uint64_t    current_value;
} vk_timeline_t;

bool     vk_timeline_init(VkDevice device, vk_timeline_t* t);
void     vk_timeline_destroy(VkDevice device, vk_timeline_t* t);
void     vk_timeline_wait(VkDevice device, vk_timeline_t* t, uint64_t value);
void     vk_timeline_signal(VkDevice device, vk_timeline_t* t, uint64_t value);
uint64_t vk_timeline_poll(VkDevice device, vk_timeline_t* t);
```

## 7.7 vk_descriptor.h

```c
#pragma once
#include <vulkan/vulkan.h>

typedef struct vk_descriptor_arena_t {
    VkDescriptorPool        pool;
    VkDescriptorSet         static_weight_set;
    VkDescriptorSet         table_set;
    VkDescriptorSetLayout   set0_layout;  // weights (UPDATE_AFTER_BIND, variable count)
    VkDescriptorSetLayout   set1_layout;  // IO (push descriptor)
    VkDescriptorSetLayout   set2_layout;  // tables (rope freqs, etc.)
} vk_descriptor_arena_t;

bool vk_descriptor_arena_init(vk_descriptor_arena_t* arena,
                               VkDevice device,
                               uint32_t num_layers);
void vk_descriptor_arena_destroy(VkDevice device, vk_descriptor_arena_t* arena);
void vk_descriptor_arena_update_weights(VkDevice device,
                                         vk_descriptor_arena_t* arena,
                                         const vk_buffer_t* weight_bufs,
                                         uint32_t num_layers);
void vk_descriptor_arena_update_tables(VkDevice device,
                                        vk_descriptor_arena_t* arena,
                                        vk_buffer_t rope_freqs,
                                        vk_buffer_t alibi_slopes);
```

## 7.8 common.h

```c
#pragma once

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

// ─── Debug / Release macros ────────────────────────────────────────

#ifdef _DEBUG
    #define VK_CHECK(call) do {                                         \
        VkResult _r = (call);                                           \
        if (_r != VK_SUCCESS) {                                         \
            fprintf(stderr, "[VK ERROR] %s:%d: %s = %d\n",             \
                    __FILE__, __LINE__, #call, _r);                     \
            __debugbreak();                                             \
        }                                                               \
    } while(0)
#else
    #define VK_CHECK(call) (call)
#endif

#define ARRAY_COUNT(arr) (sizeof(arr) / sizeof((arr)[0]))

// ─── GGUF → VK quant mapping helpers ────────────────────────────────

static inline const char* vk_quant_name(vk_quant_type_t q) {
    switch (q) {
        case VK_QUANT_FP16:   return "FP16";
        case VK_QUANT_Q4_K:   return "Q4_K";
        case VK_QUANT_Q6_K:   return "Q6_K";
        case VK_QUANT_Q8_0:   return "Q8_0";
        case VK_QUANT_IQ4_XS: return "IQ4_XS";
        default: return "UNKNOWN";
    }
}

static inline const char* vk_op_name(vk_op_type_t op) {
    switch (op) {
        case OP_RMS_NORM:      return "rms_norm";
        case OP_ATTN_QKV:      return "attn_qkv";
        case OP_ATTN_COMPUTE:  return "attn_compute";
        case OP_ATTN_OUTPUT:   return "attn_output";
        case OP_FFN_GATE_UP:   return "ffn_gate_up";
        case OP_FFN_DOWN:      return "ffn_down";
        case OP_LM_HEAD:       return "lm_head";
        case OP_TOKEN_EMBED:   return "token_embed";
        default: return "unknown";
    }
}
```

---

---

# PART 8: Push Constants Layout

---

```c
// Shared definition between C host code and GLSL shaders
// (Must match exactly between CPU and GPU sides)

typedef struct PushConstants {
    uint32_t layer_idx;            // offset 0:  4 bytes
    uint32_t kv_cache_pos;         // offset 4:  4 bytes
    uint32_t seq_len;              // offset 8:  4 bytes
    uint32_t n_heads;              // offset 12: 4 bytes
    uint32_t n_kv_heads;           // offset 16: 4 bytes
    uint32_t head_dim;             // offset 20: 4 bytes
    uint32_t page_size_tokens;     // offset 24: 4 bytes
    uint32_t pages_per_layer;      // offset 28: 4 bytes
    float    attn_scale;           // offset 32: 4 bytes
    float    rope_theta;           // offset 36: 4 bytes
    float    norm_eps;             // offset 40: 4 bytes
} PushConstants;
// Total: 44 bytes — well within the 256-byte push constant limit

// GLSL equivalent:
// layout(push_constant) uniform PushConstants {
//     uint  layer_idx;
//     uint  kv_cache_pos;
//     uint  seq_len;
//     uint  n_heads;
//     uint  n_kv_heads;
//     uint  head_dim;
//     uint  page_size_tokens;
//     uint  pages_per_layer;
//     float attn_scale;
//     float rope_theta;
//     float norm_eps;
// } pc;
```

---

---

# PART 9: Weight Upload Sequence

---

```c
// vk_model_load.cpp — transfer queue upload loop

bool vk_model_load_weights(vk_session_t* session, const gguf_file_t* gguf) {
    vk_device_t* dev = session->device;
    vk_model_t* model = &session->model;

    const uint32_t n_layers = gguf->config.n_layers;
    const uint32_t d        = gguf->config.d;
    const uint32_t ffn_dim  = gguf->config.ffn_dim;
    const uint32_t n_kv_heads = gguf->config.n_kv_heads;
    const uint32_t head_dim  = gguf->config.head_dim;
    const uint32_t vocab_size = gguf->config.vocab_size;

    // Detect weight quantization from first weight tensor
    model->weight_quant = VK_QUANT_FP16;  // default
    for (uint64_t i = 0; i < gguf->tensor_count; i++) {
        if (strstr(gguf->tensors[i].name, ".weight")) {
            model->weight_quant = gguf_tensor_to_vk_quant(gguf->tensors[i].type);
            break;
        }
    }
    printf("  Weight quant: %s\n", vk_quant_name(model->weight_quant));

    // Compute per-layer weight sizes
    uint64_t q_size   = d * d;                   // elements
    uint64_t k_size   = d * head_dim * n_kv_heads;
    uint64_t v_size   = k_size;
    uint64_t o_size   = q_size;
    uint64_t gate_size = d * ffn_dim;
    uint64_t up_size   = gate_size;
    uint64_t down_size = ffn_dim * d;
    uint64_t norm_size = d;                       // fp16 elements per norm

    // Compute byte sizes (weight quant dependent)
    float elem_bytes = 0.0f;
    switch (model->weight_quant) {
        case VK_QUANT_FP16:   elem_bytes = 2.0f; break;
        case VK_QUANT_Q4_K:   elem_bytes = 0.5625f; break;
        case VK_QUANT_Q6_K:   elem_bytes = 0.84375f; break;
        case VK_QUANT_Q8_0:   elem_bytes = 1.0f; break;
        case VK_QUANT_IQ4_XS: elem_bytes = 0.71875f; break;
        default: elem_bytes = 2.0f; break;
    }

    // Norms are always fp16
    uint64_t norm_bytes = norm_size * 2;

    uint64_t q_bytes   = (uint64_t)(q_size * elem_bytes);
    uint64_t k_bytes   = (uint64_t)(k_size * elem_bytes);
    uint64_t v_bytes   = (uint64_t)(v_size * elem_bytes);
    uint64_t o_bytes   = (uint64_t)(o_size * elem_bytes);
    uint64_t gate_bytes = (uint64_t)(gate_size * elem_bytes);
    uint64_t up_bytes   = (uint64_t)(up_size * elem_bytes);
    uint64_t down_bytes = (uint64_t)(down_size * elem_bytes);

    uint64_t layer_total = q_bytes + k_bytes + v_bytes + o_bytes
                         + gate_bytes + up_bytes + down_bytes
                         + norm_bytes * 2;  // attn_norm + ffn_norm

    // Allocate staging buffer (host-visible, enough for largest single tensor)
    uint64_t max_tensor_size = 0;
    max_tensor_size = max(max_tensor_size, q_bytes);
    max_tensor_size = max(max_tensor_size, gate_bytes);
    max_tensor_size = max(max_tensor_size, down_bytes);
    // Embedding + LM head
    uint64_t emb_bytes = (uint64_t)(vocab_size * d * 2);  // fp16 embeddings
    uint64_t lm_bytes  = (uint64_t)(vocab_size * d * elem_bytes);
    max_tensor_size = max(max_tensor_size, emb_bytes);
    max_tensor_size = max(max_tensor_size, lm_bytes);

    vk_buffer_t staging;
    if (!vk_buffer_create(dev->allocator, max_tensor_size,
                          VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                          VK_BUF_HOST_VISIBLE | VK_BUF_HOST_COHERENT,
                          &staging)) {
        fprintf(stderr, "Failed to allocate staging buffer\n");
        return false;
    }

    // Allocate transfer command buffer
    VkCommandBufferAllocateInfo cai = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = dev->transfer_cmd_pool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;

    VkCommandBuffer transfer_cb;
    vkAllocateCommandBuffers(dev->device, &cai, &transfer_cb);

    // Allocate weight buffers for layers + embedding + lm_head
    for (uint32_t l = 0; l < n_layers; l++) {
        if (!vk_buffer_create(dev->allocator, layer_total,
                              VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                              VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                              VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                              VK_BUF_DEVICE_LOCAL,
                              &model->weight_buffers[l])) {
            fprintf(stderr, "Failed to allocate weight buffer[%u]\n", l);
            return false;
        }

        // Build weight layout
        model->weight_layout[l] = (vk_layer_weight_block_t){
            .q_offset    = 0,
            .k_offset    = q_bytes,
            .v_offset    = q_bytes + k_bytes,
            .o_offset    = q_bytes + k_bytes + v_bytes,
            .attn_norm_offset = q_bytes + k_bytes + v_bytes + o_bytes,
            .ffn_norm_offset  = q_bytes + k_bytes + v_bytes + o_bytes + norm_bytes,
            .gate_offset = q_bytes + k_bytes + v_bytes + o_bytes + norm_bytes * 2,
            .up_offset   = q_bytes + k_bytes + v_bytes + o_bytes + norm_bytes * 2 + gate_bytes,
            .down_offset = q_bytes + k_bytes + v_bytes + o_bytes + norm_bytes * 2 + gate_bytes + up_bytes,
        };
    }

    // Embedding buffer (fp16)
    if (!vk_buffer_create(dev->allocator, emb_bytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_BUF_DEVICE_LOCAL,
                          &model->embedding_buffer)) {
        return false;
    }

    // LM head buffer
    if (!vk_buffer_create(dev->allocator, lm_bytes,
                          VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                          VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                          VK_BUF_DEVICE_LOCAL,
                          &model->lm_head_buffer)) {
        return false;
    }

    // Upload loop — copy each tensor from GGUF mmap to VRAM via staging
    for (uint64_t ti = 0; ti < gguf->tensor_count; ti++) {
        const gguf_tensor_info_t* t = &gguf->tensors[ti];
        uint64_t data_size;
        const uint8_t* data = gguf_get_tensor_data(gguf, (uint32_t)ti, &data_size);
        if (!data) continue;

        // Determine target buffer + offset
        VkBuffer target_buf = VK_NULL_HANDLE;
        uint64_t target_offset = 0;

        // Parse tensor name to determine destination
        // Format: "blk.N.attn_q.weight", "blk.N.attn_k.weight", etc.
        uint32_t layer_idx = 0;
        bool is_layer_tensor = false;

        if (sscanf(t->name, "blk.%u.", &layer_idx) == 1) {
            is_layer_tensor = true;
        }

        if (is_layer_tensor) {
            target_buf = model->weight_buffers[layer_idx].buffer;
            if (strstr(t->name, "attn_q")) {
                target_offset = model->weight_layout[layer_idx].q_offset;
            } else if (strstr(t->name, "attn_k")) {
                target_offset = model->weight_layout[layer_idx].k_offset;
            } else if (strstr(t->name, "attn_v")) {
                target_offset = model->weight_layout[layer_idx].v_offset;
            } else if (strstr(t->name, "attn_output")) {
                target_offset = model->weight_layout[layer_idx].o_offset;
            } else if (strstr(t->name, "attn_norm")) {
                target_offset = model->weight_layout[layer_idx].attn_norm_offset;
            } else if (strstr(t->name, "ffn_norm")) {
                target_offset = model->weight_layout[layer_idx].ffn_norm_offset;
            } else if (strstr(t->name, "ffn_gate")) {
                target_offset = model->weight_layout[layer_idx].gate_offset;
            } else if (strstr(t->name, "ffn_up")) {
                target_offset = model->weight_layout[layer_idx].up_offset;
            } else if (strstr(t->name, "ffn_down")) {
                target_offset = model->weight_layout[layer_idx].down_offset;
            } else {
                fprintf(stderr, "Unknown layer tensor: %s\n", t->name);
                continue;
            }
        } else if (strstr(t->name, "token_embd")) {
            target_buf = model->embedding_buffer.buffer;
        } else if (strstr(t->name, "output")) {
            target_buf = model->lm_head_buffer.buffer;
        } else {
            fprintf(stderr, "Unknown tensor: %s (skipped)\n", t->name);
            continue;
        }

        // Copy to staging, then transfer to device-local
        memcpy(staging.mapped_ptr, data, data_size);
        vk_buffer_flush(dev->allocator, &staging, 0, data_size);

        VkCommandBufferBeginInfo bi = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(transfer_cb, &bi);

        VkBufferCopy region = {0, target_offset, data_size};
        vkCmdCopyBuffer(transfer_cb, staging.buffer, target_buf, 1, &region);

        vkEndCommandBuffer(transfer_cb);

        VkSubmitInfo si = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &transfer_cb;

        vkQueueSubmit(dev->transfer_queue, 1, &si, session->transfer_fence);
        vkWaitForFences(dev->device, 1, &session->transfer_fence, VK_TRUE, UINT64_MAX);
        vkResetFences(dev->device, 1, &session->transfer_fence);
    }

    vkFreeCommandBuffers(dev->device, dev->transfer_cmd_pool, 1, &transfer_cb);
    vk_buffer_destroy(dev->allocator, &staging);

    return true;
}
```

---

---

# PART 10: Checklist — Build & Verification

---

| # | Step | Command |
|---|------|---------|
| 1 | CMake configure | `cmake -B build/rdna4-llm -S src/rdna4-llm -G "Visual Studio 17 2022" -A x64` |
| 2 | Build | `cmake --build build/rdna4-llm --config Release` |
| 3 | Run | `build/rdna4-llm/Release/rdna4-llm.exe --model models\Qwen2.5-0.5B-Instruct-Q4_K_M.gguf --prompt "Hello" --max-tokens 50` |
| 4 | Shader rebuild | `cmake --build build/rdna4-llm --target rdna4_shaders` (or touch any .comp file) |
| 5 | Offline compile | `pwsh src/rdna4-llm/scripts/compile_shaders.ps1 -BuildDir build/rdna4-llm` |
| 6 | Verify shader embedding | Check `build/rdna4-llm/generated/` for `.h` files with `static const uint32_t` arrays |
| 7 | Vulkan capabilities | Run `vulkaninfo.exe` to verify subgroup support and RDNA4 features before building |

---

## Minimal Requirements for First Running Build

To reach a **minimal working build** that loads a GGUF and generates tokens:

1. **Implement these files first** (in order):
   - `common.h` (macros)
   - `vk_device.h/.cpp` (init + destroy; hardcode query for gfx1201)
   - `vk_buffer.h/.cpp` (VMA wrappers)
   - `gguf_types.h + gguf_parser.h/.cpp` (load + parse GGUF)
   - `vk_model.h + vk_model_load.cpp` (weight upload)
   - `vk_pipeline.h/.cpp` (single pipeline: lm_head_fp16 for now)
   - `sample.h/.cpp` (sampling)
   - `main.cpp` (end-to-end)

2. **First milestone shader set** (just to get tokens flowing):
   - `lm_head_fp16.comp` (matrix multiply: d × vocab_size, output logits)
   - `common.glsl`

3. **Skip for milestone 1**:
   - KV cache (use full recomputation per step — slow but works)
   - Multiple quant types (hardcode FP16 path)
   - Prefill (use decode one-by-one for the prompt)
   - Interactive mode
   - Verification/validation layers

4. **Build target**: `rdna4-llm.exe` that accepts `--model` and prints sampled tokens.

---

**End of implementation design document.**
