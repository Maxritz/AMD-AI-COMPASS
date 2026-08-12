# Build-System → Shader-Registry Wiring

## 1. Overview

This document describes how `.comp` shader sources are compiled into SPIR-V
binaries, embedded as C headers, indexed in `shader_registry.h`, and bound to
Vulkan pipelines at runtime. It serves as the wiring map between CMake
build logic and the engine's runtime pipeline creation code.

## 2. Build Flow

```
src/rdna4-llm/shaders/*.comp
       │
       ▼  (glslc.exe, per shader × wave variant × quant define)
<spirv_dir>/
├── <name>_w32.spv           (SUBGROUP_SIZE=32, specialize: local_size_x_id=1)
├── <name>_w64.spv           (SUBGROUP_SIZE=64, specialize: local_size_x_id=1)
       │
       ▼  (spirv-to-header: embed .spv as uint32_t[] + size)
<gen_dir>/
├── <name>_w32.h             (C array: g_sv_<name>_w32[], g_sv_<name>_w32_len)
├── <name>_w64.h
├── shader_registry.h         (auto-generated lookup table)
       │
       ▼  (runtime pipeline creation)
pipeline_cache.bin            (VkPipelineCache, written on first run)
```

## 3. CMake Shader Registry (Implementation §1.2)

The shader database is declared in `src/rdna4-llm/CMakeLists.txt` as a list
of `register_shader` macro calls. Each entry encodes three fields:

```
<base_name> | <wave_sizes> | <extra_defines>
```

Example:

```cmake
register_shader("attn_qkv_fp16" "32;64" "-DQUANT_FP16=1")
```

This produces two compilation targets:

| Target              | glslc flags                                                  |
|---------------------|--------------------------------------------------------------|
| `attn_qkv_fp16_w32` | `--target-env=vulkan1.4 -DQUANT_FP16=1 -DSUBGROUP_SIZE=32`  |
| `attn_qkv_fp16_w64` | `--target-env=vulkan1.4 -DQUANT_FP16=1 -DSUBGROUP_SIZE=64`  |

> **Version note:** The active CMakeLists.txt (Phase 0, `src/rdna4-llm/`)
> currently pins `--target-env=vulkan1.4`. Older references in the legacy
> tree used `vulkan1.3`; all must be updated to `vulkan1.4` to match the
> Vulkan SDK 1.4.357.0 baseline (Shader Spec §0.3, GPU Capabilities §3).

### 3.1 Specialization Constants

- **SUBGROUP_SIZE** — passed as `-DSUBGROUP_SIZE=<wave>` to glslc. Used by
  shaders via `#if SUBGROUP_SIZE == 32` to select Wave32 vs Wave64 code paths.
- **(optional) local_size_x** — not used; shaders use `local_size_x_id = 1`
  with dynamic specialization at pipeline creation (Shader Spec §0.3).

### 3.2 Defines

| Define            | Purpose                              | Used By                            |
|-------------------|--------------------------------------|------------------------------------|
| `QUANT_FP16`      | Enable FP16 weight load path         | attn_qkv, attn_output, ffn_*, lm_head |
| `QUANT_Q4_K`      | Enable Q4_K dequant (dequant.glsl)   | attn_qkv, attn_output, ffn_*, lm_head |
| `QUANT_Q6_K`      | Enable Q6_K dequant                  | attn_qkv, attn_output, ffn_*, lm_head |
| `QUANT_Q8_0`      | Enable Q8_0 dequant                  | attn_qkv, attn_output, ffn_*, lm_head |
| `QUANT_IQ4_XS`    | Enable IQ4_XS dequant                | attn_qkv, attn_output, ffn_*, lm_head |
| (none for rms_norm, attn_compute, token_embed) | Non-quantized (fixed FP16 I/O) | N/A |

## 4. SPIR-V-to-Header Embedding

The CMakeLists.txt generates a `.h` file for each `.spv` binary. The format
is a C struct containing the raw SPIR-V bytecode as a `uint32_t` array:

```c
// generated/<name>_w32.h
#pragma once
#include <stdint.h>
static const uint32_t g_spv_<name>_w32[] = {
    0x07230203u, 0x000020025u, ...  // SPIR-V magic + version (1.3 = 0x10300)
};
static const size_t g_spv_<name>_w32_len = <bytecode_size_in_uint32s>;
```

The SPIR-V version embedded in each binary is **1.3** (0x10300), generated
with `--target-env=vulkan1.4`. This is correct — `vulkan1.4` target env
maps to SPIR-V 1.3 in glslc, which is the version compatible with Vulkan
1.4 runtime.

## 5. Shader Registry Header (`shader_registry.h`)

This auto-generated file provides a single lookup function that maps the
human-readable shader name + wave size to its SPIR-V header symbol:

```c
// generated/shader_registry.h
const spirv_entry_t* shader_registry_lookup(
    const char* name,    // e.g. "attn_qkv_fp16"
    uint32_t wave_size   // 32 or 64
);
```

The runtime calls this during pipeline initialization:

```
Init()
  └─ for each (shader_name, wave_size) in active set:
      spirv = shader_registry_lookup(name, wave_size)
      pipeline = vkCreateComputePipelines(device, ..., spirv)
      cache[pipeline_key_t{name, wave_size}] = pipeline
```

## 6. Runtime Pipeline Binding

### 6.1 Architecture → Wave Size Selection

| Architecture | Subgroup Size | SPIR-V Variant | CMake Target |
|-------------|---------------|----------------|-------------|
| RDNA4 (gfx1201) | 32 | `_w32` | `SUBGROUP_SIZE=32` |
| RDNA2 (gfx103x) | 64 | `_w64` | `SUBGROUP_SIZE=64` |

### 6.2 Per-Decode-Step Pipeline Sequence

During decode, the engine records 225 dispatches in a single command buffer
(Architecture §12.1). Each dispatch selects the pipeline from the registry
based on the model's quantization type at init time:

| Stage          | Quant Decision Point | Pipeline Key                    |
|----------------|----------------------|---------------------------------|
| RMS norm       | N/A (fixed FP16)     | `("rms_norm", WG_SIZE)`         |
| QKV projection | `model_weights.quant`| `("attn_qkv_" + quant, WG_SIZE)`|
| Attention      | N/A                  | `("attn_compute", WG_SIZE)`     |
| O-proj         | same as QKV          | `("attn_output_" + quant, WG_SIZE)`|
| FFN gate+up    | same                 | `("ffn_gate_up_" + quant, WG_SIZE)`|
| FFN down       | same                 | `("ffn_down_" + quant, WG_SIZE)` |
| LM head        | same                 | `("lm_head_" + quant, WG_SIZE)`  |

The quantization type is determined once at model load time and used to
select the entire pipeline set for the session.

## 7. Compile-Time Validation

Per Shader Spec §15, each generated `.spv` is validated:

```bash
spirv-val --target-env vulkan1.4 <shader>.spv
```

Checks:
1. Required capabilities declared
2. Entry point `main` present
3. Descriptor bindings match pipeline layout
4. No undefined references
5. Subgroup size constraints compatible

This validation runs in the CMake `add_custom_command` post-build step for
each shader variant. If `spirv-val` fails, the build aborts.

## 8. Debugging: Identifying Missing Shaders

If a shader is registered in CMake but the corresponding `.comp` source is
missing, CMake emits:

```
[rdna4-llm] Shader source not found: shaders/<name>.comp
```

and skips compilation for that entry. The runtime `shader_registry_lookup`
will return NULL for missing entries, causing a pipeline creation failure
with a clear error message at init time.
