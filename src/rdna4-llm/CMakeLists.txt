cmake_minimum_required(VERSION 3.20)
project(rdna4-llm CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Pin the exact Vulkan SDK -- multiple versions (1.4.335.0, 1.4.350.0,
# 1.4.357.0) are installed on this machine under C:/VulkanSDK; without an
# explicit hint here, find_package(Vulkan) resolution depends on whatever
# VULKAN_SDK happens to be set to in the calling shell, which is silent and
# fragile across sessions/terminals.
if(NOT DEFINED VULKAN_SDK_ROOT)
    set(VULKAN_SDK_ROOT "C:/VulkanSDK/1.4.357.0")
endif()
set(ENV{VULKAN_SDK} "${VULKAN_SDK_ROOT}")
list(PREPEND CMAKE_PREFIX_PATH "${VULKAN_SDK_ROOT}")

find_package(Vulkan REQUIRED)
message(STATUS "Using Vulkan SDK: ${VULKAN_SDK_ROOT} (Vulkan_INCLUDE_DIRS=${Vulkan_INCLUDE_DIRS})")

if(NOT DEFINED Vulkan_GLSLC_EXECUTABLE)
    find_program(Vulkan_GLSLC_EXECUTABLE NAMES glslc HINTS "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
endif()
if(NOT Vulkan_GLSLC_EXECUTABLE)
    message(FATAL_ERROR "glslc not found -- set VULKAN_SDK or add glslc to PATH")
endif()

set(RDNA4_LLM_SHADER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/shaders")
set(RDNA4_LLM_SPV_OUT_DIR "${CMAKE_CURRENT_BINARY_DIR}/spv")
file(MAKE_DIRECTORY "${RDNA4_LLM_SPV_OUT_DIR}")

# Phase 0: single harness shader, compiled directly (no registry/codegen yet --
# that lands in Phase 2 alongside the push-constant schema).
set(RDNA4_LLM_HARNESS_SPV "${RDNA4_LLM_SPV_OUT_DIR}/harness_add.spv")
add_custom_command(
    OUTPUT "${RDNA4_LLM_HARNESS_SPV}"
    COMMAND "${Vulkan_GLSLC_EXECUTABLE}" --target-env=vulkan1.4
            "${RDNA4_LLM_SHADER_DIR}/harness_add.comp" -o "${RDNA4_LLM_HARNESS_SPV}"
    DEPENDS "${RDNA4_LLM_SHADER_DIR}/harness_add.comp"
    COMMENT "Compiling harness_add.comp -> SPIR-V"
    VERBATIM
)
add_custom_target(rdna4_llm_shaders ALL DEPENDS "${RDNA4_LLM_HARNESS_SPV}")

add_executable(rdna4-llm
    src/main.cpp
    src/vk/vk_device.cpp
    src/vk/vk_buffer.cpp
    src/vk/vma_impl.cpp
)

add_dependencies(rdna4-llm rdna4_llm_shaders)

target_include_directories(rdna4-llm PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/vma"
    "${Vulkan_INCLUDE_DIRS}"
)

target_link_libraries(rdna4-llm PRIVATE ${Vulkan_LIBRARIES})

target_compile_definitions(rdna4-llm PRIVATE
    RDNA4_LLM_SPV_DIR="${RDNA4_LLM_SPV_OUT_DIR}"
)

if(MSVC)
    target_compile_options(rdna4-llm PRIVATE /W3)
else()
    target_compile_options(rdna4-llm PRIVATE -Wall -Wextra -Wno-unused-parameter)
endif()
