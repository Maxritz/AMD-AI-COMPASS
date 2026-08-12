// Small test to dump hipFunction_t internals and discover the function pointer offset
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <cstdio>
#include <cstdint>
#include <string>
#include <unordered_map>

typedef void* hipFunction_t;
typedef void* hipModule_t;

typedef int (*hipModuleLoadData_t)(hipModule_t*, const void*);
typedef int (*hipModuleGetFunction_t)(hipFunction_t*, hipModule_t, const char*);
typedef const char* (*hipKernelNameRef_t)(hipFunction_t);

struct KernelInfo {
    std::string name;
    hipFunction_t handle;
    void* probable_func_ptr;
};

int main() {
    HMODULE hip = LoadLibraryW(L"amdhip64_7.dll");
    if (!hip) { printf("Failed to load amdhip64_7.dll\n"); return 1; }

    auto Real_hipModuleGetFunction = (hipModuleGetFunction_t)GetProcAddress(hip, "hipModuleGetFunction");
    auto Real_hipKernelNameRef = (hipKernelNameRef_t)GetProcAddress(hip, "hipKernelNameRef");

    if (!Real_hipModuleGetFunction) { printf("No hipModuleGetFunction\n"); return 1; }

    // Load the compiled code object from the app's loaded modules
    // For now, just demonstrate the struct layout by hooking at runtime
    // Instead, let's dump what we can from the hip kernels that are already loaded

    // Try to enumerate all modules and functions to find the offset
    // Since we can't easily enumerate, let's just document the known struct
    printf("hipKernelNameRef: %p\n", Real_hipKernelNameRef);
    printf("hipModuleGetFunction: %p\n", Real_hipModuleGetFunction);

    // The ihipModuleSymbol struct on ROCm Windows likely looks like:
    // [0] vtable pointer (8 bytes)
    // [8] function pointer (8 bytes) - this is what hipLaunchByPtr receives
    // [16] kernel name (std::string = 32 bytes on MSVC)
    // ...
    
    // Given the offset patterns, let's try to extract at compile time
    // by running a real test: hook hipModuleGetFunction and try each offset

    printf("\nTo find the offset, update hip_tracer.cpp Hook_hipModuleGetFunction:\n");
    printf("  for (int i = 0; i < 16; i++) {\n");
    printf("    void* val = ((void**)handle)[i];\n");
    printf("    printf(\"offset[%d] = %%p\\\\n\", val);\n");
    printf("  }\n");

    FreeLibrary(hip);
    return 0;
}
