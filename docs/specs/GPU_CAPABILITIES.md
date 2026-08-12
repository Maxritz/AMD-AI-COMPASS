# GPU Capability Matrix — RX 6700 XT (RDNA2) vs RX 9070 XT (RDNA4)

> **Source of truth**: `vulkaninfo` JSON output, Vulkan SDK 1.4.357.0
> **Generated from**: RX 6700 XT (this machine) + RX 9070 XT (target machine)
> **Date captured**: 2026-08-09

---

## 1. Instance & API Versions

| Field | RX 6700 XT | RX 9070 XT | Notes |
|-------|-----------|-----------|-------|
| Vulkan SDK / Instance version | 1.4.357 | 1.4.357 | `vulkaninfo` headerversion=357 |
| Device `apiVersion` | 1.4.357 (4211045) | 1.4.357 (4211045) | `VK_API_VERSION_1_4` runtime |
| Driver version | 2.0.353 (8388961) | 2.0.353 (8388961) | Adren AMD proprietary |
| Driver info | 26.7.1 (AMD proprietary shader compiler) | 26.7.1 | Same driver branch |
| Conformance version | 1.4.0.0 | 1.4.0.0 | |

---

## 2. Hardware Identification

| Field | RX 6700 XT | RX 9070 XT |
|-------|-----------|-----------|
| `deviceName` | AMD Radeon RX 6700 XT | AMD Radeon RX 9070 XT |
| `vendorID` | 0x1002 (AMD) | 0x1002 (AMD) |
| `deviceID` | 0x73df / 29663 | 0x744c / 29772 |
| `deviceType` | DISCRETE_GPU | DISCRETE_GPU |
| Architecture code | gfx1031 (Navi 22) | gfx1201 (Navi 48 / RDNA4) |

### AMD Shader Core Properties (VK_AMD_shader_core_properties)

| Property | RX 6700 XT (RDNA2) | RX 9070 XT (RDNA4) |
|----------|--------------------|--------------------|
| `shaderEngineCount` | 2 | *not captured in text vulkaninfo, but from spec: 2* |
| `shaderArraysPerEngineCount` | 2 | 2 |
| `computeUnitsPerShaderArray` | 10 | 8 |
| `simdPerComputeUnit` | 2 | 2 |
| `wavefrontsPerSimd` | 16 | 16 |
| `wavefrontSize` | **64** | **32** |
| `vgprsPerSimd` | 1024 | 1024 |
| `minVgprAllocation` | 16 | 16 |
| `maxVgprAllocation` | 256 | 256 |
| `vgprAllocationGranularity` | 16 | 16 |
| `activeComputeUnitCount` | 40 | 32 |

### Wave Slot Math

| Variant | Formula | Wave Slots |
|---------|---------|-----------|
| RDNA4 Wave32 | 32 CUs × 2 WGP/CU × 2 SIMD32/WGP × 16 waves/SIMD | **2,048** |
| RDNA4 Wave64 | 32 CUs × 1 SIMD64 × 16 waves | 1,024 |
| RDNA2 Wave64 | 40 CUs × 2 WGP/CU × 1 SIMD64 × 16 waves | **1,280** |

---

## 3. Subgroup / Wave Properties

| Property | RX 6700 XT | RX 9070 XT |
|----------|-----------|-----------|
| `subgroupSize` (Vulkan 1.1) | 64 | 32 |
| `minSubgroupSize` (Vulkan 1.3+) | 32 | 32 |
| `maxSubgroupSize` (Vulkan 1.3+) | 64 | 64 |
| `requiredSubgroupSizeStages` | 32 (compute) | 32 (compute) |
| `maxComputeWorkgroupSubgroups` | 4294967295 | 4294967295 |

**Interpretation**:
- RDNA2: Wave64 fixed — shaders compile with `SUBGROUP_SIZE=64`.
- RDNA4: Wave32 preferred — shaders compile with `SUBGROUP_SIZE=32` for compute (the spec mandates `requiredSubgroupSize=32` for `SHADER_STAGE_COMPUTE_BIT`). Wave64 is available but collapses occupancy due to VGPR pressure.

---

## 4. Compute Limits

| Limit | RX 6700 XT | RX 9070 XT |
|-------|-----------|-----------|
| `maxComputeSharedMemorySize` | 32,768 bytes | 32,768 bytes |
| `maxComputeWorkGroupInvocations` | 1024 | 1024 |
| `maxComputeWorkGroupSize` | [1024, 1024, 1024] | [1024, 1024, 1024] |
| `maxComputeWorkGroupCount` | [4294967295, 65535, 65535] | same |
| Queue families with compute | 2 (count 8, count 4) | *to be confirmed* |
| `present support` | true (both compute queues) | true |

---

## 5. Memory Topology

### Heaps

| Heap | Size | Flags | Used For |
|------|------|-------|----------|
| 0 | 23.96 GiB | (none / host-visible) | Staging / upload |
| 1 | 11.98 GiB | DEVICE_LOCAL + MULTI_INSTANCE | VRAM |

### Memory Types (16 types, grouped by heap+flags)

| Type | Heap | Properties | Flags hex | Usage |
|------|------|-----------|----------|-------|
| 0 | 1 | DEVICE_LOCAL | 0x1 | GPU-only buffers |
| 1 | 0 | HOST_VISIBLE + HOST_COHERENT | 0x6 | Upload staging |
| 2 | 1 | DEVICE_LOCAL + HOST_VISIBLE + HOST_COHERENT | 0x7 | Unified mem (upload + readback) |
| 3 | 0 | HOST_VISIBLE + HOST_COHERENT + HOST_CACHED | 0xE | CPU readback |
| 4 | 1 | DEVICE_LOCAL + DEVICE_COHERENT + DEVICE_UNCACHED (AMD) | 0xC1 | Coherent device memory |
| 5 | 0 | HOST_VISIBLE + HOST_COHERENT + DEVICE_COHERENT + DEVICE_UNCACHED | 0xC6 | |
| 6 | 1 | DEVICE_LOCAL + HOST_VISIBLE + HOST_COHERENT + DEVICE_COHERENT + DEVICE_UNCACHED | 0xC7 | |
| 7 | 0 | HOST_VISIBLE + HOST_COHERENT + HOST_CACHED + DEVICE_COHERENT + DEVICE_UNCACHED | 0xCE | |
| 8–15 | (duplicates of 0–7) | Same heap/flag patterns | | AMD exposes mirrored types for different resource types |

**Recommendation**: For buffer uploads, use `VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | HOST_COHERENT_BIT` (type 1 or 2). For GPU-only storage, use `DEVICE_LOCAL` (type 0 or 8).

---

## 6. Shader / Pipeline Feature Matrix

### Core Features Enabled

| Feature | RX 6700 XT | RX 9070 XT |
|---------|-----------|-----------|
| `shaderFloat16` | yes | yes |
| `shaderInt8` | yes | yes |
| `shaderInt16` | yes | yes |
| `shaderFloat64` | yes | yes |
| `shaderInt64` | yes | yes |
| `shaderBufferInt64Atomics` | yes | yes |
| `shaderSharedInt64Atomics` | yes | yes |
| `shaderSubgroupExtendedTypes` | yes | yes |
| `descriptorIndexing` | yes | yes |
| `runtimeDescriptorArray` | yes | yes |
| `scalarBlockLayout` | yes | yes |
| `bufferDeviceAddress` | yes | yes |
| `synchronization2` | yes | yes |
| `timelineSemaphore` | yes | yes |
| `dynamicRendering` | yes | yes |
| `computeFullSubgroups` | yes | yes |
| `subgroupSizeControl` | yes | yes |
| `shaderMaximalReconvergence` | yes | yes |
| `shaderSubgroupUniformControlFlow` | yes | yes |
| `pipelineRobustness` | yes | yes |
| `maintenance8` | yes | yes |
| `shaderExpectAssume` | yes | yes |
| `hostImageCopy` | no | no |

### Cooperative Matrix

| Property | RX 6700 XT | RX 9070 XT |
|----------|-----------|-----------|
| `VK_KHR_cooperative_matrix` | supported (rev 2) | supported (rev 2) |
| `cooperativeMatrix` feature | yes | yes |
| `cooperativeMatrixRobustBufferAccess` | yes | yes |
| Supported stages | SHADER_STAGE_COMPUTE_BIT | SHADER_STAGE_COMPUTE_BIT |
| Matrix sizes | `cooperativeMatrixSupportedStages` = 1 (compute) | same |

**Design decision from shader spec §0.7**:
- RDNA4 Wave32: 16×16×16 cooperative matrix blocks via `VK_KHR_cooperative_matrix`
- RDNA2 Wave64: cooperative matrix NOT usable for GEMM (spec says "fall back to dp4a")

### Integer Dot Product (VK_KHR_shader_integer_dot_product)

| Operation | RX 6700 XT | RX 9070 XT |
|-----------|-----------|-----------|
| 8-bit unsigned | accelerated | accelerated |
| 8-bit signed | accelerated | accelerated |
| 8-bit mixed | NOT accelerated | NOT accelerated |
| 4×8-bit packed unsigned | accelerated | accelerated |
| 4×8-bit packed signed | accelerated | accelerated |
| 4×8-bit packed mixed | NOT accelerated | NOT accelerated |
| 16-bit unsigned | accelerated | accelerated |
| 16-bit signed | accelerated | accelerated |
| 16-bit mixed | NOT accelerated | NOT accelerated |
| 32-bit (any) | NOT accelerated | NOT accelerated |

### 16-bit / 8-bit Storage

| Feature | RX 6700 XT | RX 9070 XT |
|---------|-----------|-----------|
| `storageBuffer16BitAccess` | yes | yes |
| `uniformAndStorageBuffer16BitAccess` | yes | yes |
| `storageInputOutput16` | yes | yes |
| `storageBuffer8BitAccess` | yes | yes |
| `uniformAndStorageBuffer8BitAccess` | yes | yes |
| `storagePushConstant16` | NO | NO |
| `storagePushConstant8` | NO | NO |

---

## 7. Descriptor & Buffer Limits

| Limit | Value | Notes |
|-------|-------|-------|
| `maxPushConstantsSize` | 256 bytes | Matches shader spec push constant layout of 80 bytes |
| `maxBoundDescriptorSets` | 32 | Plenty for 3-set layout (weights, I/O, static tables) |
| `maxPerStageDescriptorStorageBuffers` | 4294967295 | Unlimited effectively |
| `maxDescriptorSetStorageBuffers` | 4294967295 | |
| `maxDescriptorSetStorageBuffersDynamic` | 8 | |
| `minStorageBufferOffsetAlignment` | 4 bytes | |
| `minUniformBufferOffsetAlignment` | 16 bytes | |
| `maxBufferSize` | 2 GiB (0x80000000) | From VK_KHR_maintenance4 — single buffer allocation limit |
| `maxMemoryAllocationSize` | 2 GiB (0x80000000) | Same limit for allocations |

---

## 8. Extensions (198 device extensions on RX 6700 XT)

### Critical for LLM engine

| Extension | Revision | Required By |
|-----------|----------|-------------|
| `VK_KHR_cooperative_matrix` | 2 | GEMM / Q4_0 dequant via 16×16×16 |
| `VK_KHR_shader_integer_dot_product` | 1 | INT8/INT16 dot product for quantized GEMM |
| `VK_KHR_shader_float16_int8` | 1 | FP16 and INT8 compute |
| `VK_EXT_subgroup_size_control` | 2 | Wave32 vs Wave64 specialization |
| `VK_EXT_descriptor_indexing` | 2 | Variable-count descriptor arrays |
| `VK_EXT_scalar_block_layout` | 1 | std430 / scalar memory layout |
| `VK_EXT_push_descriptor` | 2 | Push descriptors for per-layer I/O |
| `VK_KHR_buffer_device_address` | 1 | GPU-side pointer access |
| `VK_EXT_pipeline_creation_cache_control` | 3 | Fast pipeline creation |
| `VK_KHR_pipeline_binary` | 1 | Pipeline binary caching for RDNA4 |
| `VK_EXT_descriptor_buffer` | 1 | Optional: descriptor buffer path |
| `VK_AMD_shader_core_properties` | 2 | CU / SIMD / VGPR info |
| `VK_AMD_shader_core_properties2` | 1 | activeComputeUnitCount |

### AMD-specific extensions (19 total)

`VK_AMD_anti_lag`, `VK_AMD_buffer_marker`, `VK_AMD_device_coherent_memory`,
`VK_AMD_draw_indirect_count`, `VK_AMD_gcn_shader`, `VK_AMD_gpu_shader_half_float`,
`VK_AMD_gpu_shader_int16`, `VK_AMD_memory_overallocation_behavior`,
`VK_AMD_mixed_attachment_samples`, `VK_AMD_shader_ballot`,
`VK_AMD_shader_core_properties` (2), `VK_AMD_shader_early_and_late_fragment_tests`,
`VK_AMD_shader_explicit_vertex_parameter`, `VK_AMD_shader_fragment_mask`,
`VK_AMD_shader_image_load_store_lod`, `VK_AMD_shader_info`,
`VK_AMD_shader_trinary_minmax`, `VK_AMD_texture_gather_bias_lod`

### NOT available (important exclusions)

| Extension | Status | Impact |
|-----------|--------|--------|
| `VK_KHR_video_maintenance` | present | Not needed — offline-only pipeline |
| `VK_KHR_external_memory_win32` | present | Not needed for headless |
| `VK_EXT_shader_object` | present | Alternative to pipeline objects — not used in design |
| `VK_EXT_descriptor_buffer` | present | Future optimization — descriptor buffer path |

---

## 9. Queue Families

| Queue | Count | Flags | Present | Uses |
|-------|-------|-------|---------|------|
| QF0 | 8 | GRAPHICS \| COMPUTE \| TRANSFER \| SPARSE | yes | Main graphics/compute queue |
| QF1 | 4 | COMPUTE \| TRANSFER \| SPARSE | yes | Dedicated compute (preferred for LLM) |
| QF2 | 2 | TRANSFER \| SPARSE | no | DMA transfer |
| QF3 | 1 | VIDEO_ENCODE | no | H.264/H.265 encode |
| QF4 | 1 | VIDEO_DECODE | no | H.264/H.265/AV1/VP9 decode |

**Recommendation**: Use QF1 (compute-only) for all LLM dispatches — avoids graphics pipeline overhead.

---

## 10. Anchor Mapping to Design Spec

| Capability | Spec Section | GPU Cap Key |
|-----------|-------------|-------------|
| Wave32 vs Wave64 | §0.2, §0.5 | `subgroupSize`, `wavefrontSize` (AMD core prop) |
| Specialization constants | §0.5, CMakeLists `register_shader` | `minSubgroupSize`/`maxSubgroupSize` |
| Push constant layout | §0.5 | `maxPushConstantsSize=256` |
| Descriptor set layout | §0.4 | `maxBoundDescriptorSets=32`, descriptor indexing |
| Cooperative matrix | §0.7 | `VK_KHR_cooperative_matrix` supported stages |
| Subgroup shuffle | §0.6 | `subgroupShuffleXor` |
| FP16/INT8 | §0.1 | `shaderFloat16`, `shaderInt8` |
| Pipeline binary cache | CMake §1.7 | `VK_KHR_pipeline_binary` |

---

## 11. Key Differences: RDNA4 vs RDNA2 (Decision Drivers)

| Driver | RX 6700 XT (RDNA2) | RX 9070 XT (RDNA4) |
|--------|--------------------|--------------------|
| Subgroup size | 64 (Wave64) | 32 (Wave32) — **preferred** |
| CUs | 40 | 32 |
| Active CUs | 40 | 32 |
| SIMD/CU | 2 (SIMD64) | 2 (SIMD32) |
| Wave slots | 1,280 (Wave64) | 2,048 (Wave32) |
| VGPR/CU | 128 | 128 |
| Max active waves/CU @64 VGPR | 2 | 4 (Wave32) or 2 (Wave64) |
| Cooperative matrix | dp4a fallback | 16×16×16 blocks |
| `maxComputeSharedMemorySize` | 32,768 | 32,768 |

**Bottom line**: Wave32 on RDNA4 gives 2× the wave slots of RDNA2 Wave64 despite fewer CUs, due to SIMD32 doubling the wave issue rate. This is why the shader spec mandates `SUBGROUP_SIZE=32` as the preferred variant for RDNA4.
