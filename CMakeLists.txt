cmake_minimum_required(VERSION 3.20)
project(ai-compass VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

# Options
option(AI_COMPASS_BUILD_HIP_TRACER "Build HIP tracer DLL" ON)
option(AI_COMPASS_USE_ADLX "Use ADLX for GPU metrics" ON)
option(AI_COMPASS_USE_GPU_PERF_API "Use GPUPerfAPI for counters" ON)

# Core library
file(GLOB_RECURSE CORE_SRC
    src/core/src/*.cpp
)
add_library(ai_compass_core ${CORE_SRC})
target_include_directories(ai_compass_core
    PUBLIC
        src/core/include
        src/output
)
target_link_libraries(ai_compass_core PUBLIC)

# CLI tool
add_executable(aicompass
    src/cli/aicompass.cpp
)
target_link_libraries(aicompass PRIVATE ai_compass_core)
target_include_directories(aicompass PRIVATE src/core/include)

# HIP tracer plugin
if(AI_COMPASS_BUILD_HIP_TRACER)
    add_library(ai_hip_tracer SHARED
        src/hip-tracer/dll_main.cpp
        src/gpu-perf/adlx_poller.cpp
    )
    target_include_directories(ai_hip_tracer PRIVATE
        src/core/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    )
    target_link_libraries(ai_hip_tracer PRIVATE ai_compass_core)
    set_target_properties(ai_hip_tracer PROPERTIES
        PREFIX ""
        SUFFIX ".dll"
    )
endif()

# Output library
add_library(ai_output
    src/output/rocprofv2_output.cpp
)
target_include_directories(ai_output PUBLIC src/output)
target_link_libraries(ai_output PRIVATE ai_compass_core)

# Install
install(TARGETS aicompass ai_hip_tracer ai_compass_core ai_output
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
