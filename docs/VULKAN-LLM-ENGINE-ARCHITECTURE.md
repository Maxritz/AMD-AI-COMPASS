# Pure Vulkan LLM Inference Engine Architecture
## AMD RDNA4 (gfx1201 / RX 9070 XT) + RDNA2 Target

---

## 0. Fatal Flaw Root-Cause Analysis

### Flaw 0.1: 640 vkQueueSubmit + vkWaitForFences Per Decode Step

**Mechanism**: For a 32-layer model with 8 operations per layer plus 64 attention-head dispatches:
```
Layer 1: norm → Q-proj → submit+wait → K-proj → submit+wait → ...
   × 32 layers × ~8 ops/op = 256 submit/wait pairs (layer ops)
   + 32 submit/wait pairs (32 head-wise attention dispatches)
   + ~350 submit/wait pairs (misc sync points)
```
Each `vkQueueSubmit` + `vkWaitForFences` pair costs **50-150µs** driver overhead on AMD Windows.

```
640 submits × 100µs avg = 64ms CPU-side overhead per decode
Max theoretical throughput: 1000ms / 64ms = 15.6 tok/s
```
This alone explains the 0.19-0.84 tok/s measured throughput.

### Flaw 0.2: Per-Head Attention Dispatches

32 separate dispatches for 32 heads. Each head dispatch is a tiny workgroup (128 threads) on a GPU with 32 CUs × 16 SIMD units = 512 available wavefronts. Launch overhead dominates; occupancy is <2%.

### Flaw 0.3: Validation Readbacks During Inference

The engine calls `vkCmdCopyBuffer` + `vkQueueWaitIdle` + `memcpy` to read back intermediate tensors for validation, creating full pipeline stalls mid-inference.

### Flaw 0.4: Descriptor Set Per Operation

Every `vkCmdBindDescriptorSets` allocates a NEW descriptor set from a pool. On RDNA4, `vkAllocateDescriptorSets` is not free — ~5-10µs per call. At 500+ calls per decode: 2.5-5ms wasted.

### Flaw 0.5: New Command Buffer Per Operation

Every operation calls `vkBeginCommandBuffer` + `vkEndCommandBuffer`. Entirely unnecessary — a single CB can record all dispatches for a decode step.

### Flaw 0.6: Linear Staging Buffer (Not Ring)

Staging buffers grow unboundedly. After N decode steps, the staging area is N × scratch_size. No reuse, no fence tracking. Fragments VRAM and causes unnecessary allocations.

### Flaw 0.7: Missing Pipeline Barriers (Batch Mode NaN)

Batch mode dispatches multiple kernels without `vkCmdPipelineBarrier` between them. GPU reordering causes RAW hazards — one kernel reads a buffer before the previous kernel finishes writing to it. NaN propagates through all subsequent layers.

### Flaw 0.8: 0.19 tok/s Baseline

All seven flaws compound multiplicatively. The theoretical ceiling from flaw 0.1 alone (15.6 tok/s) is reduced further by validation stalls, descriptor overhead, CB overhead, and poor occupancy from per-head launches.

---

## 1. Hardware Reference Data

### RX 9070 XT (gfx1201, RDNA4) — Primary Target

| Property | Value | Source |
|----------|-------|--------|
| Vulkan version | 1.4.357.0 | vkGetPhysicalDeviceProperties |
| Compute queue family 0 | 8 queues (graphics+compute+transfer) | queueFamiliesProperties[0] |
| Compute queue family 1 | 8 queues (compute+transfer) | queueFamiliesProperties[1] |
| Transfer queue family 2 | 1 queue (transfer only) | queueFamiliesProperties[2] |
| maxComputeWorkGroupInvocations | 1,024 | limits |
| maxComputeSharedMemorySize | 32,768 bytes (32KB) | limits (non-opt-in) |
| maxPushConstantsSize | 256 bytes | limits |
| maxBoundDescriptorSets | 32 | limits |
| maxPerStageDescriptorStorageBuffers | 4,294,967,295 | limits (descriptor indexing) |
| maxPerStageResources | 4,294,967,295 | limits (descriptor indexing) |
| maxStorageBufferRange | 4,294,967,295 (4GB) | limits |
| maxPushDescriptors (KHR) | 32 | VkPhysicalDevicePushDescriptorPropertiesKHR |
| minStorageBufferOffsetAlignment | 4 bytes | limits |
| minUniformBufferOffsetAlignment | 16 bytes | limits |
| nonCoherentAtomSize | 128 bytes | limits |
| optimalBufferCopyOffsetAlignment | 1 byte | limits |
| optimalBufferCopyRowPitchAlignment | 1 byte | limits |
| timestampPeriod | 10 ns | limits |
| subgroupSize (native max) | 64 | VkPhysicalDeviceSubgroupProperties |
| subgroupSize (min, supported) | 32 | VK_EXT_subgroup_size_control |
| cooperativeMatrix (KHR) | true | VkPhysicalDeviceCooperativeMatrixFeaturesKHR |
| cooperativeMatrixRobustBufferAccess | true | cooperative matrix ext |
| shaderBFloat16CooperativeMatrix | true | bf16 coopmat |
| shaderFloat8CooperativeMatrix | true | fp8 coopmat |
| timelineSemaphore | true | VkPhysicalDeviceVulkan12Features |
| bufferDeviceAddress | true | VkPhysicalDeviceVulkan12Features |
| pushDescriptor (KHR) | true | VkPhysicalDevicePushDescriptorPropertiesKHR |
| descriptorIndexing (EXT) | version 2 | extension version |
| bufferlessPushDescriptors | true | VkPhysicalDeviceDescriptorBufferPropertiesEXT |
| shaderInt8 | true | VkPhysicalDeviceVulkan12Features |
| shaderFloat16 | true | VkPhysicalDeviceVulkan12Features |
| shaderInt16 | true | VkPhysicalDeviceVulkan12Features |
| shaderBfloat16 | true | VK_KHR_shader_bfloat16 |
| shaderFloat8 | true | VK_EXT_shader_float8 |
| shaderAtomicFloat | true | VK_EXT_shader_atomic_float |
| shaderIntegerDotProduct (KHR) | true | VK_KHR_shader_integer_dot_product |
| 8-bit/16-bit storage access | true | VkPhysicalDevice16BitStorageFeatures / 8BitStorageFeatures |
| scalarBlockLayout | true | VkPhysicalDeviceScalarBlockLayoutFeatures |
| VRAM total | 15.92 GiB (~17,098 MB) | VkPhysicalDeviceMemoryBudgetPropertiesEXT |
| VRAM usable (92% ceiling) | ~14.65 GiB | budget × 0.92 |
| System RAM | 76.73 GiB | host query |
| CUs | 32 | RDNA4 architecture |
| L2 cache | 12 MB | RDNA4 architecture |
| LDS per WGP | 128 KB | RDNA4 ISA spec |
| Memory bandwidth | ~960 GB/s | GDDR6 256-bit @ 20Gbps |

### RX 6700 XT (gfx1031, RDNA2) — Secondary Target

| Property | Value |
|----------|-------|
| CUs | 40 |
| Wave size (native) | **64** (Wave64 only) |
| L2 cache | 4 MB |
| LDS per WGP | 128 KB |
| Memory bandwidth | ~384 GB/s |
| maxComputeSharedMemorySize | 32,768 bytes |
| maxComputeWorkGroupInvocations | 1,024 |
| Subgroup size | 64 (min=64, max=64) |

**Key architectural difference**: RDNA2 is Wave64-native. RDNA4 supports both but prefers Wave32. Shaders MUST use `VK_EXT_subgroup_size_control` with `VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT_EXT` and the correct `requiredSubgroupSize` (32 for RDNA4, 64 for RDNA2).

---

## 2. Memory Model

### 2.1 Weight VRAM Layout

**One large `VkBuffer` per layer**, sub-allocated into contiguous weight regions. All weights allocated at `LoadModel()` and never moved, freed, or resized.

```
Layer buffer layout (per layer, 0-indexed):
┌────────────────────────────────────────────────────┐
│ Offset 0:          Q_weight    (d × d × type_bytes)│
│ Offset Q_size:     K_weight    (d × head_dim × n_kv_heads × type_bytes)│
│ Offset K_size:     V_weight    (d × head_dim × n_kv_heads × type_bytes)│
│ Offset V_size:     O_weight    (d × d × type_bytes)│
│ Offset O_size:     attn_norm   (d × sizeof(fp16))  │
│ Offset attn_norm:  ffn_norm    (d × sizeof(fp16))  │
│ Offset ffn_norm:   gate_weight (d × ffn_dim × type_bytes)│
│ Offset gate_size:  up_weight   (d × ffn_dim × type_bytes)│
│ Offset up_size:    down_weight (ffn_dim × d × type_bytes)│
└────────────────────────────────────────────────────┘

Per-layer weight buffer size (Qwen3.5-9B, Q4_K, d=4096, ffn_dim=11008, n_kv_heads=8):
  Q:    4096 × 4096 × 0.5625 = 9,437,184 bytes
  K:    4096 × 128 × 8 × 0.5625 = 2,359,296 bytes
  V:    4096 × 128 × 8 × 0.5625 = 2,359,296 bytes
  O:    4096 × 4096 × 0.5625 = 9,437,184 bytes
  gate: 4096 × 11008 × 0.5625 = 25,362,432 bytes
  up:   4096 × 11008 × 0.5625 = 25,362,432 bytes
  down: 11008 × 4096 × 0.5625 = 25,362,432 bytes
  norms: 2 × 4096 × 2 = 16,384 bytes (fp16)
  Total per layer: ~99,696,640 bytes ≈ 95.08 MB
  32 layers: ~3,042 MB

LM head (output): vocab_size × d × type_bytes
  128K × 4096 × 0.5625 = ~295 MB

Embedding: vocab_size × d × type_bytes
  128K × 4096 × 0.5625 = ~295 MB

Total VRAM for model weights: ~3,632 MB (~3.55 GiB)
```

Weight buffers live in `VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT` memory (VRAM). No host mapping. Uploaded once via transfer queue staging.

### 2.2 KV Cache Layout (Paged)

**Paged attention with fixed-size pages.**

```
Page size: 256 tokens (chosen for 128-byte L2 cache line alignment:
   page_tokens × head_dim × sizeof(fp16) = 256 × 128 × 2 = 65,536 bytes = multiple of 128)

Per-page storage (Qwen3.5-9B, 8 KV heads, head_dim=128):
  K page: 256 × 8 × 128 × 2 = 524,288 bytes (512 KB)
  V page: 256 × 8 × 128 × 2 = 524,288 bytes (512 KB)
  Combined K+V per page: 1,048,576 bytes (1 MB)

Per-layer KV cache, 4096 context:
  pages = ceil(4096 / 256) = 16 pages
  per_layer_kv_size = 16 × 1,048,576 = 16,777,216 bytes (16 MB)

Full KV cache (32 layers):
  total_kv_size = 32 × 16,777,216 = 536,870,912 bytes (512 MB)

Page table (per layer, flat array):
  uint32_t page_table[num_pages];
  page_table[i] = physical_page_index or 0xFFFFFFFF (empty)
  32 layers × 16 entries × 4 bytes = 2,048 bytes (negligible, in device-local buffer)
```

**Physical KV buffer**: One contiguous `VkBuffer` of size `total_kv_size` in device-local memory. Pages are allocated linearly from this pool. The page table maps logical page indices to physical offsets within this buffer.

**Page allocation**: Simple bump allocator. Pages are assigned sequentially. For single-sequence inference, just pages 0..15 per layer with no holes.

**KV cache addressing in shaders**:
```glsl
// Push constant: kv_page_start (physical page index for layer 0 KV)
// Shader computes: k_addr = kv_page_start * PAGE_SIZE_BYTES + 
//                           layer_idx * PAGES_PER_LAYER * PAGE_SIZE_BYTES +
//                           page_idx * PAGE_SIZE_BYTES + 
//                           head_idx * PAGE_TOKENS * HEAD_DIM * sizeof(fp16) +
//                           token_idx * HEAD_DIM * sizeof(fp16)
```

### 2.3 Hidden State Double-Buffer

```
Hidden state size (fp16): d × 2 = 4096 × 2 = 8,192 bytes per buffer
Double-buffer: 2 × 8,192 = 16,384 bytes
Allocated as: VkBuffer in DEVICE_LOCAL, never host-mapped

Swap pattern:
  Layer 0 reads from buf[0], writes to buf[1]
  Layer 1 reads from buf[1], writes to buf[0]
  Layer 2 reads from buf[0], writes to buf[1]
  ...
  After final layer: hidden state resides in buf[layer_count % 2]
```

### 2.4 Staging Ring Buffer (Zero-Copy Uploads)

For model loading only (not hot path). During inference, no staging is needed.

```
Staging ring for weight upload:
  Size: 256 MB (host-visible, device-local preference)
  Write pointer: advanced by upload size, reset when all weights loaded
  Synchronization: transfer queue fence per upload batch
  
Weight upload sequence:
  1. Map staging ring at current write pointer
  2. memcpy weight data from GGUF file to staging ring
  3. Flush (vkFlushMappedMemoryRanges)
  4. vkCmdCopyBuffer(staging → device-local weight buffer)
  5. Submit to transfer queue (family 2)
  6. Wait on transfer fence
  7. Advance ring write pointer
```

### 2.5 Output Logits Buffer

```
Logits buffer (host-readable):
  Size: vocab_size × sizeof(fp32) = 128,000 × 4 = 512 KB
  Memory: VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
  Usage: GPU writes after LM head projection, CPU reads to sample next token
```

### 2.6 Memory Budget Summary (Gemma-4 12B Q4_K_M, 4096 context)

| Component | Size | Memory Type |
|-----------|------|-------------|
| Model weights (Q4_K_M, 6.86 GB GGUF) | ~6.5 GB | DEVICE_LOCAL |
| KV cache (32 layers × 16 pages) | 512 MB | DEVICE_LOCAL |
| Hidden state double-buffer | 16 KB | DEVICE_LOCAL |
| Scratch/intermediate tensors | 128 KB | DEVICE_LOCAL |
| Output logits | 512 KB | HOST_VISIBLE |
| Descriptor sets / pipeline objects | <10 MB | DEVICE_LOCAL |
| **Total** | **~7.02 GB** | |
| **VRAM headroom (15.92 GB total)** | **~8.9 GB** | |

---

## 3. Command Submission Model

### 3.1 Target: ONE vkQueueSubmit Per Decode Step

```
┌─────────────────────────────────────────────────────┐
│              Decode Step N Command Buffer             │
│                                                      │
│  vkBeginCommandBuffer(cb_decode, ...)                │
│                                                      │
│  // Layer 0                                          │
│  vkCmdPushConstants(layer=0, kv_pos=N, ...)         │
│  vkCmdPushDescriptorSetKHR(io_set, hidden_in=buf0,  │
│      hidden_out=buf1, k_cache=..., v_cache=...)      │
│  vkCmdBindPipeline(pipeline_attn_qkv)                │
│  vkCmdDispatch(qkv_wg_x, 1, 1)                      │
│  vkCmdPipelineBarrier(COMPUTE → COMPUTE,            │
│      buf0: WRITE→READ, buf1: NONE→WRITE)             │
│                                                      │
│  vkCmdPushDescriptorSetKHR(io_set, hidden_in=buf1,  │
│      hidden_out=buf1, k_cache=..., v_cache=...)      │
│  vkCmdBindPipeline(pipeline_attn_compute)            │
│  vkCmdDispatch(wg_nheads, wg_kv_len, 1)              │
│  vkCmdPipelineBarrier(...)                            │
│                                                      │
│  ... (FFN gate, up, down, norms) ...                 │
│                                                      │
│  // Layer 1..N-1 (same pattern, double-buffer swap)  │
│  ...                                                  │
│                                                      │
│  // LM Head                                          │
│  vkCmdPushDescriptorSetKHR(io_set, hidden_in=buf[X],│
│      output=logits_buf)                               │
│  vkCmdBindPipeline(pipeline_lm_head)                  │
│  vkCmdDispatch(wg_vocab, 1, 1)                       │
│                                                      │
│  // Host read barrier                                 │
│  vkCmdPipelineBarrier(COMPUTE → HOST,                │
│      logits_buf: WRITE→HOST_READ)                     │
│                                                      │
│  vkEndCommandBuffer(cb_decode)                        │
└─────────────────────────────────────────────────────┘

vkQueueSubmit(compute_queue, 1 submit_info):
  wait: timeline_semaphore = N-1     (previous decode complete)
  signal: timeline_semaphore = N     (this decode complete)
  fence: decode_fence               (optional, for CPU polling)
```

### 3.2 Command Buffer Strategy

**Pre-recorded CB for decode (re-recorded only when KV cache offsets change)**:

Each decode step differences ONLY in:
1. KV cache position (new token position advances by 1)
2. Hidden state buffer (double-buffer swap pattern — deterministic)

Strategy: **Re-record the decode CB each step**. This sounds expensive but:

```
vkBeginCommandBuffer:     ~1-2µs
256 vkCmdDispatch calls:  ~256 × 0.5µs = 128µs
255 vkCmdPipelineBarrier: ~255 × 0.3µs = 76µs
256 vkCmdPushConstants:   ~256 × 0.2µs = 51µs
256 vkCmdPushDescriptor:  ~256 × 0.5µs = 128µs  (push descriptors are very cheap)
vkEndCommandBuffer:       ~2-3µs
Total CB recording:       ~388µs
```
This is 0.4ms of CPU time per decode step — negligible compared to the 10ms GPU time budget for 100 tok/s.

**Alternative: Pre-recorded CB with indirect push constants via a uniform buffer**:
- Record the CB ONCE at Init(), write per-step parameters to a small uniform buffer
- Eliminates 0.4ms CPU overhead per step
- When KV cache position is the only variable, update it in the uniform buffer before submit
- Required for maximizing throughput on very fast models (>200 tok/s)

### 3.3 Prefill Command Buffer

**One CB per prefill operation** (not per layer). Prefill processes all prompt tokens at once.

```
prefill_cb per layer (submitted once for all layers):
  For each layer:
    QKV projection (batch=seq_len)
    Attention compute (batch=seq_len, full attention matrix)
    KV cache write (all seq_len tokens)
    FFN gate+up+down
  end with hidden state in VRAM
```

For 256-token prompt, ~30 layers, the prefill CB contains ~240 dispatches.
Single vkQueueSubmit for entire prefill. Timeline semaphore value advances by 1.

### 3.4 Queue Configuration

```
Compute queue (family 1, queue index 0): Primary inference queue
  - All decode and prefill dispatches
  - Timeline semaphore managed

Transfer queue (family 2, queue index 0): Model loading only
  - Weight upload at LoadModel()
  - Not used during inference

Queue priority: VK_QUEUE_GLOBAL_PRIORITY_HIGH_EXT (requires VK_EXT_global_priority)
```

---

## 4. Synchronization Model

### 4.1 Timeline Semaphore — Primary Sync Mechanism

```
Create:
  VkSemaphoreTypeCreateInfo typeInfo = {
    .semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE
  };
  VkSemaphore timeline_sem;
  initialValue = 0;

Decode step N:
  VkSemaphoreWaitInfo waitInfo = {
    .semaphoreCount = 1,
    .pSemaphores = &timeline_sem,
    .pValues = &(uint64_t){N-1}  // wait for previous step
  };

  VkSubmitInfo submitInfo = {
    .waitSemaphoreCount = 1,
    .pWaitSemaphores = &timeline_sem,
    .pWaitDstStageMask = COMPUTE_SHADER_BIT,
    .signalSemaphoreCount = 1,
    .pSignalSemaphores = &timeline_sem,
    .pSignalValues = &(uint64_t){N}  // signal completion
  };
  vkQueueSubmit(compute_queue, &submitInfo, fence_or_null);
```

### 4.2 Host Polling (Non-Blocking)

```
// Poll without blocking:
uint64_t completed_value;
vkGetSemaphoreCounterValue(device, timeline_sem, &completed_value);
if (completed_value >= N) {
    // Step N logits are ready, sample token
}
```

### 4.3 Intra-CB Synchronization (Pipeline Barriers)

Every dispatch-to-dispatch transition within one CB:

```c
VkMemoryBarrier memory_barrier = {
    .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
};

// Or more precisely using buffer memory barriers:
VkBufferMemoryBarrier buffer_barrier = {
    .sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER,
    .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT,
    .dstAccessMask = VK_ACCESS_SHADER_READ_BIT,
    .srcQueueFamilyIndex = compute_queue_family,
    .dstQueueFamilyIndex = compute_queue_family,
    .buffer = hidden_buf,
    .offset = 0,
    .size = VK_WHOLE_SIZE,
};

vkCmdPipelineBarrier(
    cb,
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,   // src stage
    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,   // dst stage
    0,
    1, &buffer_barrier,
    0, NULL, 0, NULL
);
```

### 4.4 Double-Buffering Semantics

```
Hidden state buffers: buf_a, buf_b
Swap toggle: (layer_idx & 1)

Layer 0: read buf_a, write buf_b  ← toggle = (0 & 1) = 0, swap → buf_a in, buf_b out
  [barrier: buf_b WRITE → READ]

Layer 1: read buf_b, write buf_a  ← toggle = (1 & 1) = 1, swap → buf_b in, buf_a out
  [barrier: buf_a WRITE → READ]

This guarantees:
  - Layer N+1 cannot read buf[X] until Layer N finishes writing buf[X]
  - Each layer writes to the opposite buffer from the previous layer
  - No RAW hazards: barriers ensure completion before next read
```

### 4.5 Fence Usage (Host-Visible Only)

Fences are used ONLY for non-hot-path operations:
- Model weight upload completion
- Initialization/Build steps
- Clean shutdown

During inference: NO fences waited. Only timeline semaphore polling.

---

## 5. Descriptor Strategy

### 5.1 Layout: 3 Descriptor Sets

```
Set 0: Static Weights (UPDATE_AFTER_BIND, descriptor indexing)
  ┌──────────────────────────────────────────────────────┐
  │ binding=0: VK_DESCRIPTOR_TYPE_STORAGE_BUFFER          │
  │   Count: num_layers (variable, descriptor indexing)   │
  │   Array of weight buffers, one per layer              │
  │   Indexed by layer_idx in shader                     │
  │                                                       │
  │ Each element contains:                                │
  │   - q_weight (VkDeviceAddress at layer-specific offset)│
  │   - k_weight, v_weight, o_weight                      │
  │   - gate_weight, up_weight, down_weight               │
  │   - attn_norm, ffn_norm                               │
  └──────────────────────────────────────────────────────┘

Set 1: Layer IO (Push Descriptors — updated every layer within a CB)
  ┌──────────────────────────────────────────────────────┐
  │ binding=0: hidden_in    (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) │
  │ binding=1: hidden_out   (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) │
  │ binding=2: k_cache       (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) │
  │ binding=3: v_cache       (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) │
  │ binding=4: scratch        (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) │
  │                                                       │
  │ Uses vkCmdPushDescriptorSetKHR (5 descriptors, < 32 max)│
  └──────────────────────────────────────────────────────┘

Set 2: Optional — Attention LUT / RoPE tables (static)
  ┌──────────────────────────────────────────────────────┐
  │ binding=0: rope_freqs  (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER) │
  │ binding=1: alibi_slopes (VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)│
  │                                                       │
  │ Allocated once, never updated during inference        │
  └──────────────────────────────────────────────────────┘
```

### 5.2 Push Constant Layout (256 bytes max)

```
struct PushConstants {
    uint32_t layer_idx;        // 0-31
    uint32_t kv_cache_pos;     // current token position
    uint32_t seq_len;          // total sequence length so far
    uint32_t head_dim;         // 128
    uint32_t num_heads;        // total heads (32)
    uint32_t num_kv_heads;     // KV heads (8 for GQA)
    uint32_t page_size_tokens; // 256
    uint32_t pages_per_layer;  // 16 (for 4096 context)
    float    attn_scale;       // 1/sqrt(head_dim)
    float    rope_theta;       // RoPE base frequency
    uint32_t kv_page_table_offset; // device address offset for page table
};
// Total: 44 bytes (well within 256 limit)
```

### 5.3 Descriptor Pool Allocation

```
One-time allocation at Init():

Descriptor pool sizes:
  Set 0 (static weights):  1 set with binding count = num_layers
    → poolSize[0] = { STORAGE_BUFFER, num_layers }
  
  Set 2 (attention tables): 1 set
    → poolSize[1] = { STORAGE_BUFFER, 2 }
  
  Push descriptors: allocated from pool
    → poolSize[2] = { STORAGE_BUFFER, 32 }  // max push descriptors

  VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT: not needed
    (sets are never freed)
```

### 5.4 Why Push Descriptors (Not Pre-Allocated + Bind)

Per-layer IO bindings change every layer. Without push descriptors, each layer would need:
- A pre-allocated descriptor set (32 layers × 1 set = 32 sets, plus 2× for double-buffer = 64 sets)
- `vkCmdBindDescriptorSets` per layer (cheap but still extra API call)

Push descriptors eliminate the allocation entirely:
- `vkCmdPushDescriptorSetKHR` writes descriptor directly into command buffer
- No pool pressure, no set allocation overhead
- Combined with `vkCmdPushConstants`, the per-layer parameter setup is 2 API calls

Cost: Push descriptors are slightly more expensive per-call than pre-bound sets (~0.5µs vs ~0.2µs), but the allocation savings dominate.

### 5.5 Descriptor Indexing for Weights

Set 0 uses `VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT_EXT` with `VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT_EXT`:

```c
VkDescriptorSetLayoutBinding binding = {
    .binding = 0,
    .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
    .descriptorCount = num_layers,  // variable count
    .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
};

VkDescriptorSetLayoutBindingFlagsCreateInfo flags = {
    .bindingCount = 1,
    .pBindingFlags = (VkDescriptorBindingFlags[]){
        VK_DESCRIPTOR_BINDING_VARIABLE_DESCRIPTOR_COUNT_BIT_EXT |
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT_EXT |
        VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT_EXT
    },
};
```

Shader access:
```glsl
#extension GL_EXT_nonuniform_qualifier : require

layout(set=0, binding=0) readonly buffer Weights {
    WeightBlock weights[];
} weight_array[];  // variable-size array

// Access layer L weights: weight_array[nonuniformEXT(L)].weights[...]
```

---

## 6. Transfer Queue Usage

### 6.1 Model Loading (One-Time)

```
Weight upload flow:
  1. Create staging buffer: 256 MB, HOST_VISIBLE | HOST_COHERENT
  2. Map staging buffer
  3. For each layer weight tensor:
     a. memcpy from GGUF file to staging buffer
     b. vkCmdCopyBuffer(staging, weight_buf[layer], COPY_REGION{offset, size})
     c. Submit batch to transfer queue (family 2)
     d. Fence-wait for transfer batch completion
  4. Unmap staging buffer
  5. Free staging buffer
```

Time: For a 6.86 GB model with PCIe 4.0 x16 (~25 GB/s effective), ~275ms for weight transfer.

### 6.2 Inference-Time Transfers

**None.** During inference, no CPU↔GPU transfers occur except:
- Final logits readback (512 KB, host-visible buffer, coherent — no explicit transfer needed)
- Token ID upload (4 bytes): written directly to a HOST_VISIBLE+COHERENT buffer

### 6.3 Async Weight Streaming (Future — MoE Models)

For Mixture-of-Experts models where experts don't fit in VRAM:
- Transfer queue streams expert weights from system RAM to VRAM on-demand
- Expert dispatch determines which experts are needed for the next token
- Prefetch: begin transfer for predicted experts while current expert runs
- Requires: separate expert weight buffer per expert slot, timeline semaphore for transfer-compute sync

---

## 7. Class Hierarchy

### 7.1 Design Principle: No Virtual Functions

All polymorphism is compile-time via templates or function tables (C-style vtables) where runtime dispatch is unavoidable (RDNA4 vs RDNA2 code path selection). No `virtual` anywhere.

### 7.2 Core Structures

```c
// ========== Device Layer ==========

typedef struct vk_device_t {
    VkInstance                instance;
    VkPhysicalDevice          physical_device;
    VkDevice                  device;
    VkPhysicalDeviceProperties props;
    VkPhysicalDeviceMemoryProperties mem_props;
    
    // Queue families
    uint32_t                  compute_qf_idx;   // family 1
    uint32_t                  transfer_qf_idx;  // family 2
    VkQueue                   compute_queue;
    VkQueue                   transfer_queue;
    
    // Extensions/features enabled
    bool                      has_descriptor_indexing;
    bool                      has_timeline_semaphore;
    bool                      has_push_descriptor;
    bool                      has_cooperative_matrix;
    bool                      has_subgroup_size_control;
    bool                      has_bda;
    bool                      has_float8;
    bool                      has_bf16_coopmat;
    uint32_t                  subgroup_size;      // 32 (RDNA4) or 64 (RDNA2)
    uint32_t                  max_push_descriptors;
    uint32_t                  max_shared_memory;
    uint32_t                  max_workgroup_invocations;
    
    // Allocator
    VmaAllocator              allocator;  // or custom suballocator
    
    // Command pools
    VkCommandPool             compute_cmd_pool;   // for inference CBs
    VkCommandPool             transfer_cmd_pool;  // for weight upload
    
    // Pipeline cache
    VkPipelineCache           pipeline_cache;
    
    // Debug
    VkDebugUtilsMessengerEXT  debug_messenger;    // NULL in release
} vk_device_t;

// ========== Buffer ==========

typedef struct vk_buffer_t {
    VkBuffer       buffer;
    VmaAllocation  allocation;
    VkDeviceSize   size;
    VkDeviceAddress device_address;  // BDA
    void*          mapped_ptr;       // NULL if not host-visible
    bool           is_host_visible;
    bool           is_host_coherent;
} vk_buffer_t;

// ========== Pipeline ==========

typedef struct vk_pipeline_t {
    VkPipeline               pipeline;
    VkPipelineLayout         layout;
    VkShaderModule           shader;
    VkDescriptorSetLayout    set_layouts[3];  // 0=weights, 1=IO(push), 2=tables
    uint32_t                 push_constant_size;
    uint32_t                 workgroup_x;
    uint32_t                 workgroup_y;
    uint32_t                 workgroup_z;
} vk_pipeline_t;

// ========== Timeline Sync ==========

typedef struct vk_timeline_t {
    VkSemaphore   semaphore;
    uint64_t      current_value;    // next signal value
    uint64_t      completed_value;  // last known completed value
} vk_timeline_t;

// ========== Descriptor ==========

typedef struct vk_descriptor_arena_t {
    VkDescriptorPool      pool;
    VkDescriptorSet       static_weight_set;  // set 0
    VkDescriptorSet       table_set;           // set 2
    VkDescriptorSetLayout set0_layout;
    VkDescriptorSetLayout set1_layout;  // push descriptor layout
    VkDescriptorSetLayout set2_layout;
} vk_descriptor_arena_t;

// ========== Model ==========

typedef struct vk_layer_config_t {
    uint32_t d;              // hidden dimension (4096)
    uint32_t ffn_dim;        // FFN intermediate (11008)
    uint32_t n_heads;        // query heads (32)
    uint32_t n_kv_heads;     // key/value heads (8, GQA)
    uint32_t head_dim;       // head dimension (128)
    uint32_t vocab_size;     // vocabulary (128000)
    uint32_t n_layers;       // number of layers (32 or 40)
    float    rope_theta;     // RoPE theta (500000.0 for Qwen)
    float    norm_eps;       // RMS norm epsilon (1e-6)
    bool     use_gqa;        // grouped query attention flag
} vk_layer_config_t;

typedef struct vk_layer_weight_block_t {
    uint64_t q_offset;       // device address offset from layer_base
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
    vk_buffer_t  buffer;           // single large KV buffer
    uint32_t     page_size_tokens; // 256
    uint32_t     pages_per_layer;  // 16 (for 4096 context)
    uint32_t     total_pages;      // n_layers * pages_per_layer
    uint32_t     page_stride_bytes;// bytes per page (K+V for all heads)
    vk_buffer_t  page_table;       // uint32_t[total_pages] on device
    
    // Per-page KV offset helper:
    // k_offset(layer, page) = layer * pages_per_layer * page_stride_bytes
    //                       + page * page_stride_bytes
    // v_offset = k_offset + K_bytes_per_page
} vk_kv_cache_t;

typedef struct vk_model_t {
    vk_layer_config_t  config;
    vk_buffer_t        layer_weight_buffers[MAX_LAYERS];  // one buffer per layer
    vk_buffer_t        embedding_buffer;                   // token embeddings
    vk_buffer_t        lm_head_buffer;                     // output projection
    vk_kv_cache_t      kv_cache;
    vk_layer_weight_block_t weight_layout[MAX_LAYERS];    // CPU-side layout descriptors
} vk_model_t;

// ========== Inference Session ==========

typedef struct vk_decode_state_t {
    VkCommandBuffer   cb;              // per-step command buffer
    vk_buffer_t       hidden_buf[2];   // double-buffer for hidden state
    uint32_t          hidden_toggle;   // 0 or 1
    vk_buffer_t       logits_buf;      // host-visible output
    
    // Pre-allocated push constant struct (CPU-side, copied each step)
    uint8_t           push_constants[256];
} vk_decode_state_t;

typedef struct vk_session_t {
    vk_device_t*          device;
    vk_model_t*           model;
    vk_timeline_t         timeline;
    vk_descriptor_arena_t descriptors;
    VkFence               transfer_fence;  // weight upload only
    
    // Pipeline library (indexed by operation type)
    vk_pipeline_t   pipeline_rms_norm;
    vk_pipeline_t   pipeline_attn_qkv;
    vk_pipeline_t   pipeline_attn_compute;
    vk_pipeline_t   pipeline_attn_output;
    vk_pipeline_t   pipeline_rope;
    vk_pipeline_t   pipeline_ffn_gate_up;   // fused gate+up
    vk_pipeline_t   pipeline_ffn_down;
    vk_pipeline_t   pipeline_lm_head;
    // ... per-quantization variants (Q4_K, IQ4_XS, NVFP4, Q6_K)
    
    vk_decode_state_t decode_state;
} vk_session_t;
```

### 7.3 Pipeline Variant Dispatch Table

```
Operation × Quantization matrix:
  attn_qkv:    Q4_K, IQ4_XS, NVFP4, Q6_K, Q8_0, FP16
  attn_compute: FP16 (QK product), FP16 (softmax+weighted sum)
  ffn_gate_up:  Q4_K, IQ4_XS, NVFP4, Q6_K, Q8_0, FP16
  ffn_down:     Q4_K, IQ4_XS, NVFP4, Q6_K, Q8_0, FP16

Pipeline selection: hash(operation, quant_type) → vk_pipeline_t*
Stored in a flat array indexed by enum.
```

### 7.4 Function Table (No Virtuals)

```c
// Architecture dispatch table (populated at Init based on gfx IP)
typedef struct vk_arch_ops_t {
    // Pipeline creation functions (different subgroup size per arch)
    vk_pipeline_t (*create_pipeline)(vk_device_t*, 
        const uint32_t* spirv, size_t spirv_size,
        uint32_t subgroup_size, const char* entry);
    
    // Workgroup calc functions (different tile sizes per arch)
    void (*calc_workgroup_dims)(const vk_layer_config_t* config,
        uint32_t* wx, uint32_t* wy, uint32_t* wz, int op_type);
    
    // Shader variant selection
    const uint32_t* (*select_shader_spirv)(int quant_type, int op_type);
} vk_arch_ops_t;

// Two instances: g_arch_rdna4, g_arch_rdna2
// Set at Init time based on detected gfx IP
```

---

## 8. Initialization Sequence

### 8.1 Init() — Device + Queue Creation

```
1.  vkCreateInstance(apiVersion=VK_API_VERSION_1_4)
    Required layers (debug): VK_LAYER_KHRONOS_validation
    Required extensions: VK_KHR_get_physical_device_properties2,
                         VK_EXT_debug_utils (debug only)

2.  Enumerate physical devices, select AMD (vendorID=0x1002)
    If gfx1201: set arch = RDNA4, subgroup_size = 32
    If gfx103x: set arch = RDNA2, subgroup_size = 64

3.  Query device features + properties:
    All VkPhysicalDeviceVulkan12Features
    All VkPhysicalDeviceVulkan13Features  
    All VkPhysicalDeviceVulkan14Features
    VkPhysicalDeviceDescriptorIndexingFeatures
    VkPhysicalDeviceTimelineSemaphoreFeatures
    VkPhysicalDeviceBufferDeviceAddressFeatures
    VkPhysicalDeviceScalarBlockLayoutFeatures
    VkPhysicalDeviceSubgroupSizeControlFeatures
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR
    VkPhysicalDeviceShaderFloat8FeaturesNV (or EXT)
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT
    VkPhysicalDevice16BitStorageFeatures
    VkPhysicalDevice8BitStorageFeatures
    VkPhysicalDeviceShaderIntegerDotProductFeatures
    VkPhysicalDevicePushDescriptorPropertiesKHR

4.  Create VkDevice:
    QueueCreateInfo[0]: family 1, count=1, priority=1.0
    QueueCreateInfo[1]: family 2, count=1, priority=1.0
    Enable all queried features
    Enable extensions: VK_KHR_swapchain (for graphics compat if needed),
                       VK_KHR_push_descriptor,
                       VK_KHR_timeline_semaphore,
                       VK_EXT_descriptor_indexing,
                       VK_KHR_buffer_device_address,
                       VK_EXT_subgroup_size_control,
                       VK_KHR_shader_bfloat16,
                       VK_EXT_shader_float8,
                       VK_KHR_shader_integer_dot_product,
                       VK_EXT_scalar_block_layout,
                       VK_EXT_shader_atomic_float,
                       VK_AMD_gpu_shader_half_float,
                       VK_AMD_gpu_shader_int16,
                       VK_AMD_shader_core_properties2

5.  Get queues: vkGetDeviceQueue(device, family_1, 0) → compute_queue
                 vkGetDeviceQueue(device, family_2, 0) → transfer_queue

6.  Create VMA allocator:
    VmaAllocatorCreateInfo with VK_API_VERSION_1_4
    flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT

7.  Create command pools:
    compute_cmd_pool:  family_1, RESET_COMMAND_BUFFER_BIT
    transfer_cmd_pool: family_2, TRANSIENT_BIT

8.  Create pipeline cache:
    VkPipelineCacheCreateInfo with initial data from disk cache

9.  Create descriptor pool + Set 0 layout + Set 2 layout
    (Set 1 is push descriptor, no pool allocation needed)

10. Create timeline semaphore: initial value = 0

11. Create transfer fence: VK_FENCE_CREATE_SIGNALED_BIT

12. Compile all SPIR-V shaders (from embedded .h files or load from disk)
    ├── rms_norm.comp      (fp32, subgroup 32/64 variant)
    ├── attn_qkv.comp      (per-quant type: 5 variants)
    ├── attn_compute.comp  (fp16, subgroup 32/64)
    ├── rope.comp          (fp16, subgroup 32/64)
    ├── ffn_gate_up.comp   (per-quant type: 5 variants)
    ├── ffn_down.comp      (per-quant type: 5 variants)
    └── lm_head.comp       (per-quant type: 5 variants)

13. Create all pipeline objects from SPIR-V modules
    (can be done after model load once quant type is known)
```

### 8.2 LoadModel() — Weight Upload + Descriptor Setup

```
1.  Open GGUF file, parse header
    Extract: d, ffn_dim, n_heads, n_kv_heads, head_dim, vocab_size, n_layers,
             rope_theta, norm_eps
    Detect quant type from tensor metadata (Q4_K = 12 blocks, Q6_K = 18 blocks, etc.)

2.  Allocate weight buffers:
    For each layer 0..n_layers-1:
      layer_weight_buffers[l] = allocate_device_buffer(
        layer_size = q_size + k_size + v_size + o_size + 
                    gate_size + up_size + down_size +
                    attn_norm_size + ffn_norm_size
      )
    embedding_buffer = allocate_device_buffer(vocab_size × d × sizeof(fp16))
    lm_head_buffer = allocate_device_buffer(vocab_size × d × type_bytes)

3.  Allocate KV cache:
    pages_per_layer = ceil(max_seq_len / 256)
    page_stride = 2 × n_kv_heads × head_dim × 256 × sizeof(fp16)
    kv_size = n_layers × pages_per_layer × page_stride
    kv_cache.buffer = allocate_device_buffer(kv_size)
    kv_cache.page_table = allocate_device_buffer(n_layers × pages_per_layer × sizeof(uint32_t))

4.  Allocate hidden state double-buffer:
    hidden_buf[0] = allocate_device_buffer(d × sizeof(fp16))
    hidden_buf[1] = allocate_device_buffer(d × sizeof(fp16))

5.  Allocate logits buffer:
    logits_buf = allocate_buffer(vocab_size × sizeof(fp32),
        HOST_VISIBLE | HOST_COHERENT | HOST_CACHED)

6.  Upload weights (via transfer queue):
    Allocate staging buffer (256 MB, HOST_VISIBLE)
    For each layer:
      For each weight tensor (Q, K, V, O, gate, up, down, norms):
        Read tensor data from GGUF
        Copy to staging buffer
        vkCmdCopyBuffer(staging → weight_buf + offset)
      Submit transfer batch, fence-wait
    Free staging buffer

7.  Populate descriptor Set 0 (static weights):
    For each layer:
      Write weight block descriptor entries:
        VkDescriptorBufferInfo for each weight sub-buffer
      Update Set 0 with num_layers descriptors

8.  Populate descriptor Set 2 (tables):
    Allocate rope frequency table buffer
    Fill with precomputed RoPE frequencies
    Update Set 2

9.  Create compute pipelines (based on detected quant type):
    Use pre-compiled SPIR-V, specialization constants for:
      - d, ffn_dim, n_heads, n_kv_heads, head_dim
      - subgroup_size
      - quant_block_size, quant_type
    vkCreateComputePipelines for each operation × quant variant
    (typically 8-20 pipeline objects total)
```

### 8.3 Prefill() — Prompt Processing

```
1.  Tokenize input prompt → token_ids[] with seq_len

2.  Upload token_ids to device buffer (HOST_VISIBLE, small — 1KB)

3.  Run embedding lookup:
    vkBeginCommandBuffer(cb_prefill_emb)
    vkCmdBindPipeline(embedding_lookup)
    vkCmdBindDescriptorSets(Set 0: embedding_table)
    vkCmdPushDescriptorSetKHR(Set 1: output=hidden_buf[0])
    vkCmdPushConstants(layer=EMBEDDING_LAYER)
    vkCmdDispatch(ceil(seq_len/256), ceil(d/64), 1)
    vkEndCommandBuffer(cb_prefill_emb)
    vkQueueSubmit(compute_queue, signal=timeline, value=1)

4.  For each layer 0..n_layers-1:
    Record prefill layer CB:
      // Same operations as decode but with batch dimensions
      // [seq_len, d] → instead of [1, d]
      // Attention: [seq_len, head_dim] × [seq_len, head_dim] matrix
      QKV projection: dispatch(seq_len, d/64, 1)
      KV cache write: dispatch(seq_len, n_kv_heads, 1)  // write all tokens
      Attention: dispatch(n_heads, seq_len, 1) * n_layers
      FFN: dispatch(seq_len, ffn_dim/64, 1) for gate+up, dispatch(seq_len, d/64, 1) for down
    
    Submit all layers in one CB:
      vkQueueSubmit(compute_queue, wait=timeline, value=prev, signal=timeline, value=next)
    
    Total dispatches per layer prefill: ~7 (QKV fused, attention, O-proj, RMS norm, FFN gate+up, FFN down, RMS norm)
    For 32 layers: ~224 dispatches in one CB

5.  After prefill: hidden state is in hidden_buf[prefill_layers % 2]
    KV cache contains seq_len tokens
    Timeline semaphore value = 1 + n_layers (or 1 if all layers in one batch)
```

### 8.4 DecodeStep() — Token Generation

```
DecodeStep():

1.  Advance position: current_pos++

2.  Set push constants for first layer:
    pc.layer_idx = 0
    pc.kv_cache_pos = current_pos
    pc.seq_len = current_pos + 1  // including new token

3.  Begin command buffer:
    vkResetCommandBuffer(decode_state.cb, 0)
    vkBeginCommandBuffer(decode_state.cb, ONE_TIME_SUBMIT_BIT)

4.  For each layer l = 0..n_layers-1:
    
    // --- RMS Norm (input) ---
    vkCmdPushDescriptorSetKHR(cb, set=1, {
        .hidden_in = hidden_buf[(l + hidden_toggle) & 1],
        .hidden_out = norm_scratch
    })
    vkCmdPushConstants(cb, layout, COMPUTE_BIT, offset=0, size=44, &pc)
    vkCmdBindPipeline(cb, COMPUTE_BIT, pipeline_rms_norm)
    vkCmdDispatch(cb, 1, 1, 1)  // small workgroup for norm
    vkCmdBufferBarrier(cb, norm_scratch: WRITE→READ)

    // --- QKV Projection ---
    vkCmdPushDescriptorSetKHR(cb, set=1, {
        .hidden_in = norm_scratch,
        .k_cache = kv_cache + l * layer_kv_stride + current_pos * k_head_stride,
        .v_cache = kv_cache + l * layer_kv_stride + current_pos * v_head_stride,
        .hidden_out = qkv_scratch
    })
    vkCmdPushConstants(cb, ...)  // pc unchanged
    vkCmdBindPipeline(cb, COMPUTE_BIT, pipeline_attn_qkv[quant_type])
    vkCmdDispatch(cb, d / (4 * subgroup_size), 1, 1)  // QKV in one dispatch
    vkCmdBufferBarrier(cb, qkv_scratch: WRITE→READ, kv_cache: WRITE→READ)

    // --- Attention Compute ---
    vkCmdPushDescriptorSetKHR(cb, set=1, {
        .hidden_in = qkv_scratch,
        .k_cache = kv_cache + l * layer_kv_stride,  // full cache
        .v_cache = kv_cache + l * layer_kv_stride,
        .hidden_out = attn_scratch
    })
    vkCmdBindPipeline(cb, COMPUTE_BIT, pipeline_attn_compute)
    vkCmdDispatch(cb, n_heads, 1, 1)  // all heads in one dispatch (workgroup per head)
    vkCmdBufferBarrier(cb, attn_scratch: WRITE→READ)

    // --- Attention Output Projection ---
    vkCmdPushDescriptorSetKHR(cb, set=1, {
        .hidden_in = attn_scratch,
        .hidden_out = hidden_buf[(l + 1 + hidden_toggle) & 1]  // residual dest
    })
    vkCmdBindPipeline(cb, COMPUTE_BIT, pipeline_attn_output[quant_type])
    vkCmdDispatch(cb, d / 64, 1, 1)
    vkCmdBufferBarrier(cb, hidden_out: WRITE→READ)

    // --- FFN RMS Norm ---
    // (similar to input norm but on the attention output)
    vkCmdPushDescriptorSetKHR(cb, set=1, {
        .hidden_in = hidden_buf[(l + 1 + hidden_toggle) & 1],
        .hidden_out = norm_scratch
    })
    vkCmdBindPipeline(cb, COMPUTE_BIT, pipeline_rms_norm)
    vkCmdDispatch(cb, 1, 1, 1)
    vkCmdBufferBarrier(cb, norm_scratch: WRITE→READ)

    // --- FFN Gate + Up (fused in one shader) ---
    vkCmdPushDescriptorSetKHR(cb, set=1, {
        .hidden_in = norm_scratch,
        .hidden_out = ffn_scratch  // gate activation result
    })
    vkCmdBindPipeline(cb, COMPUTE_BIT, pipeline_ffn_gate_up[quant_type])
    vkCmdDispatch(cb, ffn_dim / 64, 1, 1)
    vkCmdBufferBarrier(cb, ffn_scratch: WRITE→READ)

    // --- FFN Down ---
    vkCmdPushDescriptorSetKHR(cb, set=1, {
        .hidden_in = ffn_scratch,
        .hidden_out = hidden_buf[(l + hidden_toggle) & 1]  // residual add in shader
    })
    vkCmdBindPipeline(cb, COMPUTE_BIT, pipeline_ffn_down[quant_type])
    vkCmdDispatch(cb, d / 64, 1, 1)
    vkCmdBufferBarrier(cb, hidden_out: WRITE→READ)
    
    // Swap toggle for next layer
    hidden_toggle ^= 1

5.  // --- LM Head ---
    // After all layers, hidden state is in hidden_buf[X]
    vkCmdPushDescriptorSetKHR(cb, set=1, {
        .hidden_in = hidden_buf[(n_layers + hidden_toggle) & 1],
        .hidden_out = logits_buf
    })
    vkCmdPushConstants(cb, layout, COMPUTE_BIT, offset=0, size=44, &pc)
    vkCmdBindPipeline(cb, COMPUTE_BIT, pipeline_lm_head[quant_type])
    vkCmdDispatch(cb, vocab_size / 256, 1, 1)
    
    // Host read barrier
    vkCmdBufferBarrier(cb, logits_buf: SHADER_WRITE → HOST_READ)

6.  vkEndCommandBuffer(cb)

7.  // Submit
    VkSubmitInfo submit = {
        .waitSemaphoreCount = 1,
        .pWaitSemaphores = &timeline.semaphore,
        .pWaitDstStageMask = COMPUTE_SHADER_BIT,
        .waitSemaphoreValues = &(uint64_t){timeline.current_value - 1},  // wait for prev
        .commandBufferCount = 1,
        .pCommandBuffers = &decode_state.cb,
        .signalSemaphoreCount = 1,
        .pSignalSemaphores = &timeline.semaphore,
        .signalSemaphoreValues = &(uint64_t){timeline.current_value},
    };
    vkQueueSubmit(compute_queue, 1, &submit, VK_NULL_HANDLE);
    timeline.current_value++;

8.  // Poll for completion (non-blocking)
    while (1) {
        uint64_t val;
        vkGetSemaphoreCounterValue(device, timeline.semaphore, &val);
        if (val >= timeline.current_value - 1) break;
        // optional: Sleep(0) or process other work
    }

9.  // Sample next token from logits
    float* logits = (float*)logits_buf.mapped_ptr;
    next_token = sample(logits, vocab_size, temperature, top_p);

10. // Embed next token for next step
    // (token embedding is implicit in the next decode step 
    //  via the attention mechanism — KV cache handles it)
    // Actually: for Llama architecture, the input to layer 0 is the 
    // embedded token. We need to embed it:
    hidden_buf[0] = embedding_lookup(next_token)
    // But this can be done in the NEXT decode step's CB as the first op,
    // or pre-embedded before the loop.
```

### 8.5 Shutdown()

```
1.  vkDeviceWaitIdle(device)
2.  For each layer: vmaDestroyBuffer(weight_bufs[l])
3.  vmaDestroyBuffer(kv_cache.buffer)
4.  vmaDestroyBuffer(hidden_buf[0]), hidden_buf[1]
5.  vmaDestroyBuffer(logits_buf)
6.  vkDestroyPipeline for all pipelines
7.  vkDestroyPipelineCache
8.  vkDestroyDescriptorPool
9.  vkDestroySemaphore(timeline)
10. vkDestroyFence(transfer_fence)
11. vkDestroyCommandPool(compute_cmd_pool)
12. vkDestroyCommandPool(transfer_cmd_pool)
13. vmaDestroyAllocator
14. vkDestroyDevice
15. vkDestroyInstance
```

---

## 9. Error Handling

### 9.1 Debug Build

```
#define VK_CHECK(call) do {              \
    VkResult _r = (call);                \
    if (_r != VK_SUCCESS) {              \
        log_error("%s:%d: VkResult=%d",  \
            __FILE__, __LINE__, _r);     \
        __debugbreak();                  \
    }                                    \
} while(0)

// Validation layers enabled
// VK_EXT_debug_utils with callback that logs + breaks
// All bound buffers have debug names via VK_EXT_debug_utils
// Pipeline statistics queries enabled
```

### 9.2 Release Build

```
#define VK_CHECK(call) (call)  // stripped entirely

// No validation layers
// No debug utils messenger
// No debug names
// Minimal error checking:
//   - Only check vkQueueSubmit, vkDeviceWaitIdle, vkGetSemaphoreCounterValue
//   - All other Vulkan calls: assume success
```

### 9.3 Device Loss Handling

```
// After vkQueueSubmit or vkGetSemaphoreCounterValue:
if (result == VK_ERROR_DEVICE_LOST) {
    // Log DRED data if VK_EXT_device_fault available
    // Re-create device (or just abort with useful error)
}
```

### 9.4 OOM Handling

```
// vkAllocateMemory checks (delegated to VMA):
if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY) {
    log_error("VRAM exhausted. Model=%zu MB, KV=%zu MB, total=%zu MB / %zu MB",
        model_size, kv_size, total, vram_total);
    return cleanup_and_exit();
}
```

### 9.5 NaN Detection (Debug-Only Diagnostic)

```
// Optional debug pass: after each layer, a small validation shader checks:
//   if any(hidden == NaN) { validation_flag = 1; }
// This is a separate CB, only submitted in debug mode.
// In release: no NaN checking (expensive).
```

---

## 10. Multi-Queue Pipelining

### 10.1 Prefill/Decode Overlap (Not Possible in Autoregressive)

Autoregressive decoding is inherently sequential: prefill must complete before decode begins because decode uses the KV cache written by prefill. No overlap opportunity.

### 10.2 Multi-Sequence Continuous Batching

For serving multiple sequences concurrently:

```
Sequence A: decode on compute_queue[0]
Sequence B: decode on compute_queue[1]
Sequence C: decode on compute_queue[2]

Each sequence has:
  - Its own KV cache partition (page table segments)
  - Its own hidden state double-buffer
  - Its own command buffer
  - Timeline semaphore per-sequence OR shared timeline with different value ranges

Scheduling:
  Round-robin submits to 3 separate queues
  Each queue can run independently — no cross-sequence barriers needed
  Shared weight buffers: READ-ONLY, no synchronization needed between sequences
```

### 10.3 Transfer-Only Queue Usage

Family 2 (dedicated transfer, 1 queue) is used exclusively for:
- Weight upload at model load time
- (Future) KV cache offload to system RAM for long-context scenarios

During active inference: transfer queue is idle. No inference-time transfers.

### 10.4 Async Compute for Post-Processing (Future)

If token sampling is GPU-side:
```
Primary decode → compute_queue[0]
GPU sampling   → compute_queue[1]  (small dispatch, runs while next decode begins)
```
Not needed for initial release — CPU sampling is faster for single-token generation.

---

## 11. Wave32 vs Wave64 Decision Table

### 11.1 Selection Criteria

| Operation | RDNA4 (32 CUs) | RDNA2 (40 CUs) | Rationale |
|-----------|---------------|---------------|-----------|
| **RMS Norm** | **Wave32**, WG=32 | **Wave64**, WG=64 | Tiny work: 4096 elements. Wave32 gives double the wavefronts for better occupancy |
| **QKV Projection (decode)** | **Wave32**, WG=64 | **Wave64**, WG=128 | Decode batch=1. Small work per workgroup. More wavefronts = better latency hiding |
| **QKV Projection (prefill)** | **Wave64**, WG=256 | **Wave64**, WG=256 | Batch=seq_len. Large GEMM benefits from wider waves for memory coalescing |
| **Attention Compute** | **Wave32**, WG=128 | **Wave64**, WG=128 | Register-heavy (QK dot + softmax). Wave32: 2× occupancy vs Wave64 |
| **Attention Output** | **Wave32**, WG=64 | **Wave64**, WG=128 | Medium work. Wave32 favored for RDNA4 occupancy |
| **FFN Gate+Up (decode)** | **Wave32**, WG=64 | **Wave64**, WG=128 | Batch=1 memory-bound. Wave32 doubles occupancy for latency hiding |
| **FFN Gate+Up (prefill)** | **Wave64**, WG=256 | **Wave64**, WG=256 | Batch=seq_len. Large matrix multiply benefits from Wave64 |
| **FFN Down (decode)** | **Wave32**, WG=64 | **Wave64**, WG=128 | Same reasoning as gate+up |
| **FFN Down (prefill)** | **Wave64**, WG=256 | **Wave64**, WG=256 | Large matrix multiply |
| **LM Head** | **Wave32**, WG=64 | **Wave64**, WG=128 | Large output vocab, batch=1 |

### 11.2 Dual-Compilation Strategy

Each shader is compiled TWICE — once with `requiredSubgroupSize=32` and once with `requiredSubgroupSize=64`. Pipeline object selection at runtime based on detected GPU family.

### 11.3 Workgroup Dimension Constants

```
RDNA4 (subgroup=32):
  WG_QKV_DECODE    = 64   (2 subgroups per workgroup)
  WG_ATTN_COMPUTE  = 128  (4 subgroups)
  WG_FFN_GATE_UP   = 64
  WG_FFN_DOWN      = 64
  WG_LM_HEAD       = 64

RDNA2 (subgroup=64):
  WG_QKV_DECODE    = 128  (2 subgroups)
  WG_ATTN_COMPUTE  = 128  (2 subgroups)
  WG_FFN_GATE_UP   = 128
  WG_FFN_DOWN      = 128
  WG_LM_HEAD       = 128
```

Workgroup X dimension (number of workgroups) = ceil(elements / WG_SIZE).
Workgroup Y/Z = 1 for decode (batch=1). Y = seq_len for prefill.

---

## 12. Dispatch Count Summary

### 12.1 Decode Step (32 layers)

| Operation | Dispatches Per Layer | Total (32 layers) |
|-----------|---------------------|-------------------|
| RMS norm (input) | 1 | 32 |
| QKV projection | 1 | 32 |
| KV cache write | 0 (fused in QKV) | 0 |
| Attention compute | 1 | 32 |
| Attention O-proj | 1 | 32 |
| FFN RMS norm | 1 | 32 |
| FFN gate+up (fused) | 1 | 32 |
| FFN down | 1 | 32 |
| **Subtotal** | **7** | **224** |
| LM head (final) | 1 | 1 |
| **Total dispatches** | | **225** |

### 12.2 Pipeline Barriers (Decode, 32 layers)

| Between operations | Per Layer | Total |
|-------------------|-----------|-------|
| norm_scratch R/W | 2 (attn norm + FFN norm) | 64 |
| qkv_scratch R/W | 1 | 32 |
| attn_scratch R/W | 1 | 32 |
| hidden_buf R/W | 2 (attn output + FFN output) | 64 |
| ffn_scratch R/W | 1 | 32 |
| logits host barrier | 1 | 1 |
| **Total barriers** | | **225** |

### 12.3 API Calls Per Decode Step

| Call | Count | Time (µs) |
|------|-------|-----------|
| `vkCmdPushDescriptorSetKHR` | 225 | 112.5 |
| `vkCmdPushConstants` | 225 | 45.0 |
| `vkCmdBindPipeline` | 225 | 45.0 |
| `vkCmdDispatch` | 225 | 112.5 |
| `vkCmdPipelineBarrier` | 225 | 67.5 |
| `vkBeginCommandBuffer` | 1 | 2.0 |
| `vkEndCommandBuffer` | 1 | 3.0 |
| `vkQueueSubmit` | 1 | 50.0 |
| `vkGetSemaphoreCounterValue` | 1 | 5.0 |
| **Total CPU overhead** | | **~442.5 µs** |

At 100 tok/s target (10ms/step): CPU overhead is 4.4% of budget.

### 12.4 Memory Access Per Decode Step (9B Q4_K model)

| Component | Bytes Accessed | Bandwidth Time (960 GB/s) |
|-----------|---------------|--------------------------|
| Q weight | 9.44 MB | 9.8 µs |
| K weight | 2.36 MB | 2.5 µs |
| V weight | 2.36 MB | 2.5 µs |
| O weight | 9.44 MB | 9.8 µs |
| gate weight | 25.36 MB | 26.4 µs |
| up weight | 25.36 MB | 26.4 µs |
| down weight | 25.36 MB | 26.4 µs |
| KV cache read (4096 ctx) | 2.00 MB | 2.1 µs |
| KV cache write | 0.06 MB | 0.06 µs |
| **Per-layer subtotal** | ~101.74 MB | ~106 µs |
| **× 32 layers total** | ~3,256 MB | ~3.4 ms |
| **+ LM head (295 MB)** | ~295 MB | ~0.31 ms |
| **Total memory time** | | **~3.7 ms** (ideal) |

With 70% bandwidth efficiency: ~5.3ms memory time
With compute time (dequant + dotprod + softmax + activation): ~3-4ms
**Expected decode latency: ~8-9ms → 110-125 tok/s**

This matches the HIP backend's measured 103 tok/s on Qwen3.5-9B NVFP4.

---

## 13. RDNA2-Specific Considerations

### 13.1 Wave64 Is Mandatory

RDNA2 does not support Wave32. All shaders must use `requiredSubgroupSize=64`. This halves the number of available wavefronts but each wavefront does 2× the work per cycle.

### 13.2 Occupancy Calculation

```
RDNA2 (RX 6700 XT): 40 CUs × 4 SIMD × 16 wave slots = 2,560 wave slots
RDNA4 (RX 9070 XT): 32 CUs × 4 SIMD × 16 wave slots = 1,024 wave slots

RDNA2 Wave64 shader with WG=128 (2 wavefronts per workgroup):
  Occupancy = 2 × N_workgroups / 2,560
  For QKV decode (d=4096, WG=128): 32 workgroups → 64 wavefronts → 2.5% occupancy
  
  This is the same occupancy problem that exists in HIP MMVQ for RDNA2.
  Mitigation: use larger workgroups (WG=256) for better occupancy.
```

### 13.3 L2 Cache Pressure

RDNA2 has 4 MB L2 (vs RDNA4's 12 MB):
- For 9B model with 256 KB workgroup LDS: L2 hit rate drops
- Use smaller tile sizes (FFN: preferred K_PER_WG = 128 vs 256 for RDNA4)
- Prefer IQ4_XS over Q4_K_M (better cache utilization)

### 13.4 RDNA2 Expected Throughput

With 384 GB/s bandwidth (vs 960 GB/s on RDNA4), and considering the 4MB L2 limitation:
- Qwen3.5-9B decode: ~40-50 tok/s expected
- Gemma-4 12B decode: ~25-35 tok/s expected

---

## 14. Shader Operation Reference

### 14.1 Dequantization + Dot Product (QKV/FFN Projections)

```glsl
// Per-quantization-type shader specialization
// Example: Q4_K dequantization

#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_KHR_shader_subgroup_arithmetic : require
#extension GL_KHR_shader_subgroup_shuffle : require
#extension GL_EXT_subgroup_size_control : require

layout(local_size_x_id = 0, local_size_y = 1, local_size_z = 1) in;
layout(constant_id = 1) const uint SUBGROUP_SIZE = 32;

layout(set=0, binding=0) readonly buffer Weights { ... } weight_array[];
layout(set=1, binding=0) readonly buffer Input { float16_t x[]; } input_buf;
layout(set=1, binding=1) buffer Output { float16_t y[]; } output_buf;

void main() {
    uint row = gl_WorkGroupID.x * (gl_WorkGroupSize.x / SUBGROUP_SIZE) + gl_SubgroupID;
    // Each subgroup processes one output row
    // Each thread in subgroup processes a portion of the input column
    
    float sum = 0.0;
    for (uint k = gl_SubgroupInvocationID; k < K; k += SUBGROUP_SIZE) {
        // Load quantized block: scales, quants
        // Dequant → fp16
        // Multiply x[k] × weight[row][k]
        // Subgroup shuffle + accumulate
    }
    sum = subgroupAdd(sum);  // subgroup reduction
    y[row] = float16_t(sum);
}
```

### 14.2 Multi-Head Attention (Fused, Decode)

```glsl
// One workgroup per head × KV position
// Each thread processes one head_dim element

layout(local_size_x = 128, local_size_y_id = 1, local_size_z = 1) in;
// workgroup count = (num_heads, kv_len, 1)

void main() {
    uint head = gl_WorkGroupID.x;
    uint kv_pos = gl_WorkGroupID.y;
    
    // Q is broadcast: all threads read same Q values
    // K is indexed by kv_pos
    // Dot product Q·K across head_dim (subgroup reduction)
    float qk = dot_product_qk(head, kv_pos);  // subgroup reduction
    // Store to LDS for softmax
    // Softmax across kv_len (subgroup reduction for max+sum)
    // Weighted sum: qk_score × V[kv_pos]
    // Accumulate via subgroup shuffle
}
```

### 14.3 RMS Normalization

```glsl
layout(local_size_x = 32, local_size_y = 1, local_size_z = 1) in;
// 1 workgroup, 32 threads, each processes d/32 elements

void main() {
    float sum_sq = 0.0;
    for (uint i = gl_LocalInvocationID.x; i < D; i += WORKGROUP_SIZE) {
        float16_t val = input[i];
        sum_sq += float(val) * float(val);
    }
    sum_sq = subgroupAdd(sum_sq);
    // Single subgroup, barrier not needed for 32 threads
    
    float inv_rms = inversesqrt(sum_sq / float(D) + EPS);
    
    for (uint i = gl_LocalInvocationID.x; i < D; i += WORKGROUP_SIZE) {
        output[i] = float16_t(float(input[i]) * inv_rms * weight[i]);
    }
}
```

---

## 15. Build System Integration

### 15.1 SPIR-V Compilation

```
All shaders: .comp GLSL → glslc → .spv → xxd → .h (embedded C header)

CMake custom command:
  add_custom_command(
    OUTPUT ${CMAKE_BINARY_DIR}/shaders/rms_norm_rdna4.h
    COMMAND glslc -fshader-stage=compute 
        --target-spv=spv1.4
        -DSUBGROUP_SIZE=32
        -DENABLE_INT8 -DENABLE_FP16 -DENABLE_BF16
        -o ${CMAKE_BINARY_DIR}/shaders/rms_norm_rdna4.spv
        ${CMAKE_SOURCE_DIR}/shaders/rms_norm.comp
    COMMAND xxd -i ${CMAKE_BINARY_DIR}/shaders/rms_norm_rdna4.spv
        ${CMAKE_BINARY_DIR}/shaders/rms_norm_rdna4.h
  )
  
  // Repeat for rms_norm_rdna2 (SUBGROUP_SIZE=64)
  // Repeat for each shader × each quant variant × each subgroup size
```

### 15.2 Shader Specialization Constants

Used at `vkCreateComputePipeline` time to bake in model-specific dimensions:

```c
VkSpecializationMapEntry entries[] = {
    {0, offsetof(SpecConstants, SUBGROUP_SIZE), sizeof(uint32_t)},
    {2, offsetof(SpecConstants, D), sizeof(uint32_t)},
    {3, offsetof(SpecConstants, FFN_DIM), sizeof(uint32_t)},
    {4, offsetof(SpecConstants, HEAD_DIM), sizeof(uint32_t)},
    {5, offsetof(SpecConstants, N_HEADS), sizeof(uint32_t)},
    {6, offsetof(SpecConstants, N_KV_HEADS), sizeof(uint32_t)},
};
```

This allows one GLSL source to compile to SPIR-V that is specialized for different model architectures at pipeline creation time, reducing the number of GLSL files needed.

---

## 16. Performance Checklist

- [ ] **1 submit per decode**: Verify via Vulkan SDK layer counters or GPU profiler
- [ ] **0 vkWaitForFences in hot path**: Assert no fence waits during decode
- [ ] **0 vkAllocateDescriptorSets in hot path**: All sets pre-allocated
- [ ] **0 vkCreateCommandBuffer in hot path**: CBs pre-allocated, reset only
- [ ] **0 vkCmdCopyBuffer for validation**: No mid-inference readbacks
- [ ] **Pipeline barriers on EVERY dispatch transition**: Systematic, not ad-hoc
- [ ] **Timeline semaphore monotonic**: Verify value always increases
- [ ] **Host polling <100µs overhead**: Measure `vkGetSemaphoreCounterValue` cost
- [ ] **One GPU-side barrier per dispatch**: No missing barriers → no NaN
- [ ] **Descriptor Set 0 never updated**: Static weights, update-after-bind
- [ ] **Push descriptors <6 per layer**: IO buffers + scratch fits in 6 bindings
- [ ] **All shaders specialized for subgroup size**: No dynamic subgroup queries in hot path
