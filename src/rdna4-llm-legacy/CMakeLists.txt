cmake_minimum_required(VERSION 3.20)
project(rdna4-llm LANGUAGES C CXX)

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

if(NOT EXISTS "${GLSLC_EXE}")
    message(FATAL_ERROR "[rdna4-llm] glslc.exe not found at ${GLSLC_EXE}")
endif()

set(VMA_SRC "")
set(VMA_INCLUDE_DIR "")

if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/vma/vk_mem_alloc.h")
    set(VMA_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/third_party/vma")
    if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/vma/vk_mem_alloc.cpp")
        set(VMA_SRC "${CMAKE_SOURCE_DIR}/third_party/vma/vk_mem_alloc.cpp")
    else()
        set(VMA_SRC "")
    endif()
    message(STATUS "[rdna4-llm] VMA found (bundled): ${VMA_INCLUDE_DIR}")
else()
    message(WARNING "[rdna4-llm] VMA not found at third_party/vma/. "
        "Download from https://github.com/GPUOpen-LibrariesAndSDKs/VulkanMemoryAllocator "
        "and place vk_mem_alloc.h under third_party/vma/")
    message(FATAL_ERROR "[rdna4-llm] VMA is required. Aborting.")
endif()

set(RDNA4_LLM_OUTPUT_DIR "${CMAKE_BINARY_DIR}/rdna4-llm")
set(RDNA4_LLM_SPIRV_DIR  "${RDNA4_LLM_OUTPUT_DIR}/spirv")
set(RDNA4_LLM_GEN_DIR    "${RDNA4_LLM_OUTPUT_DIR}/generated")
file(MAKE_DIRECTORY "${RDNA4_LLM_SPIRV_DIR}")
file(MAKE_DIRECTORY "${RDNA4_LLM_GEN_DIR}")

if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /arch:AVX2 /GL /GS- /fp:fast")
    set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} /O2 /arch:AVX2 /GL /Zi")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /Od /Zi /RTC1")
    add_compile_options("$<$<CONFIG:Release>:/MT>" "$<$<CONFIG:Debug>:/MTd>" "$<$<CONFIG:RelWithDebInfo>:/MT>")
endif()

set(RDNA4_SHADERS "")

macro(register_shader NAME WAVE_SIZES EXTRA_DEFINES)
    # WAVE_SIZES (e.g. "32;64") is joined with "," here, not "|", because the
    # unpacking loop below splits fields on "|" via string(REPLACE) -- if the
    # wave list's own ";" separators survived into that replace, CMake would
    # flatten NAME|32;64|DEFINES into a 4-element list (NAME, "32", "64",
    # DEFINES) and silently drop wave 64 from every shader's build (this
    # actually happened: no shader in this project has ever compiled a w64/
    # RDNA2 variant). Comma-joining here means "|" is the only delimiter
    # `_parts` can split on, so the unpack always yields exactly 3 fields.
    string(REPLACE ";" "," _wave_csv "${WAVE_SIZES}")
    set(_shader_entry "${NAME}|${_wave_csv}|${EXTRA_DEFINES}")
    list(APPEND RDNA4_SHADERS "${_shader_entry}")
endmacro()

register_shader("rms_norm"          "32;64" "")
register_shader("attn_qkv_fp16"     "32;64" "-DQUANT_FP16=1")
register_shader("attn_qkv_q4_k"     "32;64" "-DQUANT_Q4_K=1")
register_shader("attn_qkv_q6_k"     "32;64" "-DQUANT_Q6_K=1")
register_shader("attn_qkv_q8_0"     "32;64" "-DQUANT_Q8_0=1")
register_shader("attn_qkv_iq4_xs"   "32;64" "-DQUANT_IQ4_XS=1")
register_shader("attn_compute"      "32;64" "")
register_shader("rope"              "32;64" "")
register_shader("qk_norm"           "32;64" "")
register_shader("attn_output_fp16"  "32;64" "-DQUANT_FP16=1")
register_shader("attn_output_q4_k"  "32;64" "-DQUANT_Q4_K=1")
register_shader("attn_output_q6_k"  "32;64" "-DQUANT_Q6_K=1")
register_shader("attn_output_q8_0"  "32;64" "-DQUANT_Q8_0=1")
register_shader("attn_output_iq4_xs" "32;64" "-DQUANT_IQ4_XS=1")
register_shader("ffn_gate_up_fp16"  "32;64" "-DQUANT_FP16=1")
register_shader("ffn_gate_up_q4_k"  "32;64" "-DQUANT_Q4_K=1")
register_shader("ffn_gate_up_q6_k"  "32;64" "-DQUANT_Q6_K=1")
register_shader("ffn_gate_up_q8_0"  "32;64" "-DQUANT_Q8_0=1")
register_shader("ffn_gate_up_iq4_xs" "32;64" "-DQUANT_IQ4_XS=1")
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
register_shader("token_embed_q8_0"  "32;64" "-DQUANT_Q8_0=1")
register_shader("token_embed_q6_k"  "32;64" "-DQUANT_Q6_K=1")
register_shader("token_embed_q4_k"  "32;64" "-DQUANT_Q4_K=1")
register_shader("token_embed_iq4_xs" "32;64" "-DQUANT_IQ4_XS=1")

set(RDNA4_SPIRV_FILES "")
set(RDNA4_HEADER_FILES "")
set(RDNA4_SHADER_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")

find_package(Python3 COMPONENTS Interpreter)

foreach(_entry ${RDNA4_SHADERS})
    string(REPLACE "|" ";" _parts "${_entry}")
    list(LENGTH _parts _nparts)
    list(GET _parts 0 _name)
    list(GET _parts 1 _waves_csv)
    string(REPLACE "," ";" _waves "${_waves_csv}")
    if(_nparts GREATER 2)
        list(GET _parts 2 _defines)
    else()
        set(_defines "")
    endif()

    foreach(_w ${_waves})
        set(_spv_output "${RDNA4_LLM_SPIRV_DIR}/${_name}_w${_w}.spv")
        set(_h_output   "${RDNA4_LLM_GEN_DIR}/${_name}_w${_w}.h")
        set(_src         "${RDNA4_SHADER_INCLUDE_DIR}/${_name}.comp")

        if(NOT EXISTS "${_src}")
            message(WARNING "[rdna4-llm] Shader source not found: ${_src}")
            continue()
        endif()

        add_custom_command(
            OUTPUT "${_spv_output}"
            COMMAND "${GLSLC_EXE}"
                -fshader-stage=compute
                --target-env=vulkan1.4
                --target-spv=spv1.4
                -O
                -DSUBGROUP_SIZE=${_w}
                ${_defines}
                -I "${RDNA4_SHADER_INCLUDE_DIR}"
                -o "${_spv_output}"
                "${_src}"
            DEPENDS "${_src}" "${RDNA4_SHADER_INCLUDE_DIR}/common.glsl"
            COMMENT "Compiling ${_name} (Wave${_w}) -> SPIR-V"
            VERBATIM
        )
        list(APPEND RDNA4_SPIRV_FILES "${_spv_output}")

        if(EXISTS "${SPIRV_OPT_EXE}" AND CMAKE_BUILD_TYPE STREQUAL "Release")
            set(_spv_opt_output "${RDNA4_LLM_SPIRV_DIR}/${_name}_w${_w}_opt.spv")
            add_custom_command(
                OUTPUT "${_spv_opt_output}"
                COMMAND "${SPIRV_OPT_EXE}"
                    --eliminate-dead-code-aggressive
                    --merge-blocks
                    --simplify-instructions
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

        add_custom_command(
            OUTPUT "${_h_output}"
            COMMAND "${CMAKE_COMMAND}" -E make_directory "${RDNA4_LLM_GEN_DIR}"
            COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_spirv.py"
                --input  "${_final_spv}"
                --output "${_h_output}"
                --name   "shader_${_name}_w${_w}"
            DEPENDS "${_final_spv}" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/embed_spirv.py"
            COMMENT "Embedding ${_name}_w${_w}.spv -> ${_name}_w${_w}.h"
            VERBATIM
        )
        list(APPEND RDNA4_HEADER_FILES "${_h_output}")
    endforeach()
endforeach()

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

add_custom_target(rdna4_shaders ALL
    DEPENDS ${RDNA4_HEADER_FILES}
    COMMENT "All RDNA4 LLM shaders compiled and embedded"
)

set(RDNA4_LLM_SOURCES
    main.cpp
    vk_device.cpp
    vk_buffer.cpp
    vk_session.cpp
    vk_model.cpp
    vk_kv_cache.cpp
    vk_timeline.cpp
    vk_descriptor.cpp
    # vk_validate.cpp: temporarily excluded -- being rewritten (engine rewrite
    # plan, step 7) to call the real session dispatch path instead of a hand
    # duplicated command buffer. Re-add once that lands.
    gguf_parser.cpp
    sample.cpp
    ${VMA_SRC}
)

add_executable(rdna4-llm ${RDNA4_LLM_SOURCES})

target_include_directories(rdna4-llm PRIVATE
    "${VULKAN_INCLUDE_DIR}"
    "${VMA_INCLUDE_DIR}"
    "${RDNA4_LLM_GEN_DIR}"
    "${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(rdna4-llm PRIVATE
    "${VULKAN_LIB_DIR}/vulkan-1.lib"
)

if(NOT VMA_SRC)
    set_property(SOURCE vk_buffer.cpp APPEND PROPERTY COMPILE_DEFINITIONS "VMA_IMPLEMENTATION")
endif()
set_property(TARGET rdna4-llm PROPERTY COMPILE_DEFINITIONS
    "VMA_STATIC_VULKAN_FUNCTIONS=0"
    "VMA_DYNAMIC_VULKAN_FUNCTIONS=1"
)

add_dependencies(rdna4-llm rdna4_shaders)

set_target_properties(rdna4-llm PROPERTIES
    WIN32_EXECUTABLE FALSE
)

target_link_options(rdna4-llm PRIVATE
    "$<$<CONFIG:Release>:-O3>"
    "$<$<CONFIG:RelWithDebInfo>:-g>"
)

install(TARGETS rdna4-llm
    RUNTIME DESTINATION bin
)

message(STATUS "[rdna4-llm] Build configured: ${CMAKE_CXX_COMPILER_ID}")
message(STATUS "[rdna4-llm] Vulkan SDK:  ${VULKAN_SDK_ROOT}")
message(STATUS "[rdna4-llm] Shader count: ${RDNA4_SHADERS} (x2 wave variants each)")
