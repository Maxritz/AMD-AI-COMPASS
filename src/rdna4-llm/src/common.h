#pragma once

// Shared macros/constants for the whole engine. Kept deliberately minimal --
// this file grows only as later phases genuinely need something global
// (quant enums, arch config, etc. land in their own headers under src/model/,
// not here).

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <vulkan/vulkan.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include <vector>
#include <string>

#define VMA_STATIC_VULKAN_FUNCTIONS  0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

#if defined(_MSC_VER) && !defined(DEBUG_NO_BREAK)
#define RDNA4_DEBUG_BREAK() __debugbreak()
#else
#define RDNA4_DEBUG_BREAK() do { } while(0)
#endif

#ifndef NDEBUG
#define VK_CHECK(call) do {                                                    \
    VkResult _vk_result = (call);                                              \
    if (_vk_result != VK_SUCCESS) {                                            \
        fprintf(stderr, "[VK_ERROR] %s:%d: %s returned VkResult=%d\n",        \
                __FILE__, __LINE__, #call, (int)_vk_result);                   \
        RDNA4_DEBUG_BREAK();                                                   \
        exit(1);                                                               \
    }                                                                          \
} while(0)
#else
#define VK_CHECK(call) (call)
#endif

#define RDNA4_LOG(fmt, ...)    do { fprintf(stdout, "[RDNA4] " fmt "\n", ##__VA_ARGS__); } while(0)
#define RDNA4_ERROR(fmt, ...)  do { fprintf(stderr, "[RDNA4 ERROR] " fmt "\n", ##__VA_ARGS__); } while(0)

#define RDNA4_SUBGROUP_SIZE 32
#define RDNA2_SUBGROUP_SIZE 64
