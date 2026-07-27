cmake_minimum_required(VERSION 3.20)
project(ai-compass VERSION 0.1.0 LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(AI_COMPASS_BUILD_HIP_TRACER "Build HIP tracer DLL" ON)
option(AI_COMPASS_USE_ADLX "Use ADLX for GPU metrics" ON)
option(AI_COMPASS_USE_GPU_PERF_API "Use GPUPerfAPI for counters" ON)

# Core library
add_library(ai_compass_core
    src/core/src/logger.cpp
)
target_include_directories(ai_compass_core PUBLIC src/core/include)

# CLI tool
add_executable(aicompass
    src/cli/aicompass.cpp
)
target_link_libraries(aicompass PRIVATE ai_compass_core)

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

# Install
install(TARGETS aicompass ai_hip_tracer
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
)
