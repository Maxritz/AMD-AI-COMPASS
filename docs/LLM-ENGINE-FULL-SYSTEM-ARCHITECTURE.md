# Universal LLM Inference Engine — Full Systems Architecture

## Pure Vulkan, Windows, AMD RDNA4 primary / RDNA2 fallback, GGUF + safetensors + LiteRT-LM

**Status:** Design specification for handoff to an implementing agent (Kimi). Not implementation code.
**Companion documents in this repo:** `docs/VULKAN-LLM-ENGINE-ARCHITECTURE.md` and `docs/VULKAN-COMPUTE-SHADER-SPEC.md` contain an earlier performance post-mortem (`0.19–0.84 tok/s` root-cause analysis: per-op `vkQueueSubmit`/`vkWaitForFences` pairs, per-head dispatch launches, descriptor-set churn, missing barriers). This document does not repeat that analysis but assumes its conclusions — the Vulkan Compute Strategy section (§6) is written to avoid every flaw catalogued there by construction (single command buffer per decode step, no readback stalls, ring-buffered staging, explicit barriers).
**Companion plan:** `C:\Users\rr\.claude\plans\cheeky-orbiting-cray.md` — a 20-phase harness-first delivery plan for a narrower dense-model-first engine, written from direct debugging of three real bugs (RoPE convention, QKV bias, O-projection row-stride) found this session. This document supersedes/expands that plan's *scope* (adds safetensors, LiteRT-LM, MLA, hybrid VRAM/RAM/disk memory, speculative decoding); the phasing discipline and the specific bug classes it documents still apply and are referenced throughout.

---

## 1. Purpose and Scope

Build a single-binary, pure-Vulkan (no CUDA/ROCm/HIP dependency) LLM inference engine for Windows that:

1. Loads models from **GGUF**, **safetensors** (including AWQ/GPTQ/EXL2/HQQ-quantized variants), and **LiteRT-LM** (`.litertlm`) files.
2. Runs on **AMD RDNA4** (RX 9070 XT / gfx1201, primary target) with **RDNA2** as a supported fallback tier.
3. Supports the full range of quantization formats in current use (§4), not just one.
4. Supports dense, MoE, and MLA (latent-attention) architectures (§5) — MoE is not optional: by mid-2026 it is the dominant architecture for every frontier open-weight release (GLM-5.2, DeepSeek-V4-Pro, Kimi K2.6, MiniMax M3, Qwen3-Coder-480B, Llama 4 Maverick).
5. Runs models larger than VRAM by treating VRAM, system RAM, and NVMe disk as one managed memory hierarchy with expert-aware streaming (§7) — modeled on the real, working **Colibri** engine (`github.com/JustVugg/colibri`), which runs a 744B-parameter MoE model (GLM-5.2) in 25GB of RAM by streaming routed experts from a ~370GB NVMe checkpoint.
6. Is fast — not just correct. §6 specifies cooperative-matrix (hardware WMMA) as the primary GEMM path on RDNA4, not the subgroup-scalar-reduction approach used in prior sessions' work, because it is measurably faster (a comparable project, VulkanForge, measured Gemma-4 prefill go from 612→2629 tok/s moving to a coopmat flash-attention kernel).

### Non-goals

- Training or fine-tuning. Inference only.
- Multi-GPU / distributed inference (single-node, single-GPU is the target; note this as a future extension point in the architecture, don't design against it).
- Full feature parity with vLLM/SGLang's continuous batching for many concurrent users — this is a local/single-user engine first. Continuous batching is a valid later extension once single-stream inference is solid.

---

## 2. Lessons From Prior Sessions (why the architecture looks the way it does)

Three real, confirmed bugs were found this session by tracing a broken engine against a reference `llama.cpp` build, and all three share one root cause: **the same formula was reimplemented independently in more than one place and the copies drifted.**

1. **RoPE pairing convention is architecture-dependent** (NEOX split-half vs NORM interleaved), but an early version applied one convention to every model. Wrong for roughly half of all architectures; invisible only at sequence position 0.
2. **QKV projection bias was silently dropped** for architectures that need it (Qwen2-family), because the GLSL push-constant struct was a hand-maintained mirror of the C++ struct and had drifted out of sync — a field existed on one side and not the other.
3. **The attention-output (`O`) projection's weight row stride was computed from the residual-stream width `D` instead of the attention width `N_HEADS*HEAD_DIM`** in every quantization variant of that one shader. Invisible whenever those two happen to be numerically equal (true for Llama-3.2-1B and some Qwen2 sizes); silently corrupted every output row otherwise (true for MiniCPM, Qwen3-4B, and — critically for this document — true for essentially *every* MoE and MLA model, where attention width and residual width are routinely different).

**Design rule this document enforces everywhere:** any value that could be computed two different ways in two different files (a tensor's row stride, a push-constant's byte offset, a per-architecture flag) must have **exactly one function that computes it**, called from every site that needs it. Sections 6.2 (generic GEMM), 6.5 (push-constant codegen), and 8 (per-arch config table) are the concrete mechanisms for this rule. An implementing agent should treat "did I just write the same formula in a second place" as a stop-and-refactor signal, not a style nit.

---

## 3. Target Hardware

| Tier | GPU | Subgroup size | Matrix hardware | Notes |
|---|---|---|---|---|
| Primary | RDNA4 (RX 9070 XT, gfx1201) | 32 (wave32) | `VK_KHR_cooperative_matrix`; native FP8/BF8 WMMA with F32 accumulate (4x RDNA3's FP16 WMMA rate); native INT4/INT8 WMMA with I32 accumulate (4x RDNA3); 16×16 max tile; hardware 4:2 structured sparsity (2x on top of dense WMMA when used) | No confirmed native FP4 matrix path — treat FP4 formats (MXFP4/NVFP4) as storage/bandwidth formats that dequantize to FP8 before a WMMA op, not as a natively-multiplied format, until proven otherwise on real hardware |
| Fallback | RDNA2 | 64 (wave64) | No cooperative matrix; subgroup-scalar reduction path required | Existing project CMake already has (and had a real bug in) wave32/wave64 shader-variant registration — re-verify shader-variant counts at build time, don't trust it silently |

`maxPushConstantsSize` on the RX 9070 XT is 256 bytes (query at runtime, don't assume the Vulkan spec-minimum 128 — confirmed this session). `maxComputeSharedMemorySize` is 32768 bytes.

---

## 4. Model File Format Support

### 4.1 GGUF (primary format)

Self-contained: header + key-value metadata + tensor info table + tensor data, one file. Already the primary format in the companion plan; keep the existing (session-verified-correct) design:

- mmap the file, don't eagerly load it.
- Parse the tensor-info table into a single metadata table (`{name, quant_type, dims, byte_offset}` per tensor) built by a **dynamic cursor that only advances past tensors actually present** — this handles optional tensors (QKV bias, qk_norm, MoE routing tensors) correctly by construction, since a model that doesn't ship a tensor simply never appears in the table. This was correct in the prior session's engine and should be kept, not rewritten.
- Architecture string (`general.architecture` key) drives the per-arch config table (§8), but presence/absence of specific tensors (`attn_q.bias`, `attn_q_norm.weight`, `ffn_gate_inp.weight`) should drive *feature* detection — self-correcting, doesn't need a matrix entry per architecture.

### 4.2 safetensors (Hugging Face native)

Header (JSON, tensor name → dtype/shape/byte-range) + raw tensor bytes, no compression, mmap-friendly by design — this is *why* Colibri and similar engines read safetensors directly rather than converting to GGUF first. Two sub-cases:

- **Unquantized** (F32/F16/BF16): straightforward, same tensor-metadata-table approach as GGUF, different header parser.
- **Quantized via a paired quantization scheme**: AWQ, GPTQ, EXL2, HQQ, or bitsandbytes NF4 store quantization metadata (scales, zero-points, sometimes permutation indices for GPTQ's `desc_act`) as *additional* safetensors tensors alongside the packed weights, with a `quantize_config.json` or `quant_config.json` describing the scheme, group size, and bit-width. The loader needs one adapter per scheme (§4.2.1) that knows how to reconstruct `{quant_type, effective_bits, group_size, scale_tensor, zero_tensor}` from these files and feed them into the same generic GEMM kernel abstraction used for GGUF (§6.2) — the *quantization math* differs per scheme but the *row-stride/offset abstraction* should not fork.

#### 4.2.1 safetensors quantization scheme priority (2026 landscape, researched)

| Scheme | Bit-width | Quality (vs FP16, 4-bit class) | Where it's fastest | Priority |
|---|---|---|---|---|
| AWQ | 4-bit (activation-aware) | ~95% — best-in-class at 4-bit | Native vLLM/TGI GPU inference; default recommendation for GPU-served models in 2026 | High — implement first among safetensors schemes |
| GPTQ | 4-bit (calibration-based) | ~90% | Broad tooling support, marginally behind AWQ | High |
| EXL2 | Fractional, 2.0–8.0 bpw (mixed per-layer bit allocation) | Comparable to AWQ; finest-grained size/quality tradeoff | Squeezing large models into limited VRAM | Medium — valuable for the VRAM-constrained case this engine targets, but a more complex loader (per-layer bit-width table) |
| HQQ | Calibration-free; strong at 2-bit (8x compression, e.g. 70B→~20GB) | Good at 2-bit without calibration data | Fast to quantize, exports back to safetensors, usable cross-framework | Medium |
| bitsandbytes NF4 | 4-bit (normal-float) | Below AWQ/GPTQ | Simplicity, ubiquity in HF `transformers` | Low — implement only if a specific requested model ships only in this form |

### 4.3 LiteRT-LM (`.litertlm`, Google AI Edge)

Section-based container format (successor to the `.task` bundle format): sections for tokenizer, model graph(s), and metadata, packed together. Model graphs are stored as **TFLite flatbuffers**, not GGUF/safetensors tensors — a `TFLiteModelType` enum tags each embedded model as `PREFILL_DECODE` (the one this engine cares about) or vision/audio encoder-adapter pairs (multimodal, out of scope initially). This is architecturally the most different of the three formats: it is not "GGUF but different tensor layout," it is a different graph representation entirely (a compiled TFLite subgraph, not a flat tensor list this engine's own transformer-block code walks).

**Implementation approach:** do not attempt to execute the embedded TFLite graph directly (that would mean embedding a TFLite interpreter, defeating the point of a from-scratch Vulkan engine). Instead, write a **`.litertlm` → internal model-description extractor**: parse the flatbuffer schema for the `PREFILL_DECODE` subgraph, pull out the transformer hyperparameters (layer count, head count, hidden size, etc.) and raw weight tensors (TFLite flatbuffers store raw tensor data inline, addressable by buffer index), and feed them into the same internal tensor-metadata-table abstraction used for GGUF/safetensors. This is real, non-trivial parsing work — budget it as its own phase (§9, Phase L) after GGUF and safetensors are both solid, since LiteRT-LM models are overwhelmingly small edge-oriented models (Gemma-3n-scale, not 700B MoE) and are the lowest-priority of the three formats.

---

## 5. Model Architecture Compatibility Matrix

Extends the matrix already verified this session (RoPE type, QKV bias, qk_norm) with the attention-variant and MoE-routing dimensions needed for current frontier models.

| Architecture family | Attention | RoPE | QKV bias | qk_norm | MoE | Notes |
|---|---|---|---|---|---|---|
| `llama` | MHA/GQA | NORM | tensor-presence | tensor-presence | no | Dense-pipeline reference model (Llama-3.2-1B) |
| `qwen2`, `qwen2moe` | GQA | NEOX | **yes** | no | qwen2moe: yes | bias verification model |
| `qwen3`, `qwen3moe` | GQA | NEOX | no | **yes**, pre-RoPE | qwen3moe: yes | GQA; qk_norm bug still open as of this session |
| `deepseek2`, `deepseek3`-class (DeepSeek-V2/V3/R1, Kimi K2 use the same family) | **MLA** (multi-head latent attention) | NEOX (applied to a small "RoPE-carrying" slice, see §5.1) | tensor-presence | no | **yes**, fine-grained (many small experts + shared experts) | Requires a distinct attention kernel, not a GQA variant — see §5.1. This is not optional: DeepSeek-R1 (671B/37B-active) and derivatives are among the most-deployed open-weight MoE models |
| `gemma*` | GQA, some variants use sliding-window/global mix | NEOX | no | no | gemma3n/4 variants: yes | Embedding scaled by √d_model — a *different* mechanism from MiniCPM's embedding_scale, don't conflate |
| `minicpm` | MHA/GQA | NORM | no | no | no | embedding/residual/logit scale defaults (12.0 / 1.4/√n_layer / n_embd/256) |
| `minicpm3` | MHA/GQA | **NEOX** | no | no | no | Do not conflate with `minicpm` — different RoPE convention |
| `llama4` (Maverick/Scout) | GQA | NEOX-family | tensor-presence | tensor-presence | **yes**, few large experts (2 active × 8192 hidden) — coarser-grained than DeepSeek's routing | |
| `glm4`, `glm4_moe`, GLM-5.x family | GQA or MLA-like (verify per-checkpoint) | NEOX or MROPE depending on variant | tensor-presence | tensor-presence | **yes** (GLM-5.2 uses ~19,456 routed experts — the Colibri reference case, §7) | Highest-priority MoE target for the hybrid-memory feature given its scale |
| `laguna` (confirmed via reference `llama.cpp` fork's `src/models/laguna.cpp`) | mixed sliding-window/global | data-driven partial rotary + YaRN on global layers | tensor-presence | tensor-presence | **yes**, sigmoid gating, attention-output gating | Most demanding single architecture in scope; explicitly a late-phase target, only after MoE, MLA, YaRN, and partial rotary each work independently |
| *(unrecognized)* | assume GQA | default NORM | tensor-presence | tensor-presence | tensor-presence (`ffn_gate_inp.weight`) | Log a loud warning, never silently guess |

### 5.1 MLA (Multi-head Latent Attention) — required attention kernel variant

DeepSeek's MLA does not cache full per-head K/V. Instead it projects K/V into a shared low-rank **latent** space and caches only the compact latent vectors; per-head K/V are reconstructed from the latents via lightweight projection matrices at attention time. This dramatically shrinks the KV cache (the actual point of the design — it's a memory-bandwidth optimization, not a quality one) and is used by DeepSeek-V2/V3/R1 and derivatives including Kimi K2.

Two implementation modes exist in the reference literature: a "naive" mode that reconstructs full per-head K/V explicitly (simpler, more memory, easier to get numerically right first) and an "absorbed" mode that folds the up-projection into the query/output projections algebraically to avoid ever materializing full-size K/V (faster, more complex, do second). **Recommendation: implement naive-mode MLA first, verify it end-to-end against a reference, then implement absorbed mode as a Phase M2 optimization that must produce byte-identical output to naive mode** — this is the same "correctness gate before performance" discipline used throughout this document.

### 5.2 MoE routing variants to support

- **Coarse** (Llama 4 Maverick-style): few large experts, 2 active per token.
- **Fine-grained** (DeepSeek/GLM-style): many small experts (dozens to tens of thousands) plus one or more always-active "shared" experts. This is the harder case for the memory hierarchy (§7) because with tens of thousands of experts, no realistic amount of VRAM holds them all, and the router's choice is highly input-dependent — expert *streaming*, not just expert *offload*, is required.
- **Gating functions**: softmax (most common) and sigmoid (Laguna-style, per the reference source read this session) — both must be supported, selected per the arch table.

---

## 6. Vulkan Compute Strategy (the "fastest approach")

### 6.1 Cooperative matrix as the primary GEMM path on RDNA4

Prior work in this project (and the companion plan) designed around subgroup-cooperative *scalar* reduction (32/64 lanes jointly computing one dot product via `subgroupAdd`). That is correct and portable, but **not the fastest available path on RDNA4**, which has hardware WMMA (`VK_KHR_cooperative_matrix`) for FP8/BF8 (F32 accumulate) and INT4/INT8 (I32 accumulate) at up to 16×16 tile granularity. A directly comparable project (VulkanForge, `github.com/maeddesg/vulkanforge`) measured a coopmat flash-attention kernel taking Gemma-4 prefill from 612 to 2629 tok/s on this exact hardware class, and reports "near-parity decode vs llama.cpp Vulkan on RDNA4" (0.87–0.97×) using Q4_K_M GGUF + FP8 safetensors.

**Recommendation:**
- **RDNA4 path:** cooperative-matrix GEMM and cooperative-matrix flash-attention as the primary kernels, for any dtype combination that has a native WMMA path (FP8/BF8, INT4/INT8) or can be cheaply dequantized to one on load (most K-quants and IQ-quants dequantize to FP16/FP8 per-tile before the WMMA op — the dequant is a small per-tile cost, not per-element, since coopmat loads a whole tile at once).
- **RDNA2 fallback path (and any dtype without a practical coopmat mapping):** the subgroup-scalar-reduction design already validated this session — one subgroup per output row/element, strided accumulation, `subgroupAdd` reduction. Keep this as the always-correct reference path; coopmat is the accelerated path layered on top once correctness is established.
- Sequence within a phase: **get the scalar path numerically correct first** (it's simpler to reason about and debug — this is exactly how this session made progress), **then add the coopmat path as an alternative kernel that must produce output matching the scalar path within quantization/fp8 tolerance**, gated behind a capability check (`VK_KHR_cooperative_matrix` support + queried supported type combinations).

### 6.2 One generic, parameterized GEMM kernel (not one shader per operation)

This is the direct fix for bug class 3 (§2). Every linear projection (Q/K/V/O/gate/up/down/lm_head, MoE expert FFNs, MLA's latent projections) is the same operation: `Y[M,N] = X[M,K] @ W[N,K]^T (+bias) (+activation) (+residual*scale)`. One shader source (two variants: subgroup-scalar and coopmat, per §6.1), compiled per quantization format × wave width, replaces what would otherwise be dozens of hand-copied files. The row-stride/byte-offset math is computed **once, host-side**, directly from each tensor's real declared shape (never re-derived from a global config constant inside a shader — this exact mistake is what broke the O-projection last session).

```cpp
uint32_t gemm_row_stride_bytes(uint32_t K, quant_type_t q) {
    uint32_t block_elems = quant_block_elems(q);
    uint32_t block_bytes = quant_block_bytes(q);
    return ((K + block_elems - 1) / block_elems) * block_bytes;
}
```

MoE extends this kernel via an **indexed-GEMM mode**: `weight_byte_offset = base_offset + expert_id * per_expert_stride`, with `expert_id` supplied by a small router+top-k shader per token. Same row-stride formula, same kernel, one more input — not a forked kernel.

### 6.3 Dispatch discipline (avoiding the documented 0.19 tok/s failure mode)

Per the companion performance post-mortem: one command buffer per decode step (not per operation), explicit `vkCmdPipelineBarrier`s between dependent dispatches (not relying on submission order), descriptor sets bound via push descriptors or a small reused pool (not allocated fresh per operation), a ring-buffered staging area (not an unboundedly-growing linear one), and **zero mid-inference readback stalls** — validation/debugging tensor dumps must be a separate, explicitly-invoked code path, never inline in the hot loop.

### 6.4 Kernel fusion — after correctness, not instead of it

Native Vulkan kernel fusion has been measured (2026 literature) at 1.4–1.7× for something as simple as fusing RMSNorm into an adjacent op. Real, but a Phase-19-style late optimization: fuse only after the unfused version is verified correct and captured as a regression fixture, and require the fused version's output to remain byte-identical (or within documented fp8/quant tolerance) to the unfused baseline.

### 6.5 Push-constant layout: schema-generated, not hand-mirrored

Bug class 2 (§2) happened because the GLSL push-constant struct was a hand-maintained copy of the C++ one. Fix: one schema file (e.g. `tools/schema/push_constants.yaml`) is the single source of truth; a build-time script emits both the C++ struct and the GLSL block (with explicit per-field `layout(offset=N)`) from it. A shader can only "not see" a field if it's genuinely excluded from its schema family — it cannot silently miss a field that exists on the C++ side. Include a generated `static_assert(sizeof(...) <= <queried maxPushConstantsSize>)`.

---

## 7. Hybrid VRAM / System RAM / Disk Memory Hierarchy (the "Colibri" feature)

This is a real, working, open-source design — `github.com/JustVugg/colibri` — not a hypothetical. Confirmed architecture (researched this session):

- **Three-tier placement, one managed hierarchy**, not three independent caches:
  - **Dense/always-active layers and shared experts**: kept resident in RAM (or VRAM if it fits), typically quantized aggressively (int4) since they're touched every token and eating the memory-bandwidth cost every time is unavoidable. For GLM-5.2's ~17B dense parameters, this is ~9.9GB at int4.
  - **Routed (sparse) experts**: streamed from NVMe on demand. For GLM-5.2, 19,456 experts at ~19MB each (~370GB total) — never fully resident, by design.
  - **VRAM**: an optional fast cache tier above RAM for whichever experts are currently hottest, when a GPU is present (Colibri lists CUDA/Metal/Vulkan backends, Vulkan explicitly including AMD via RADV — directly applicable to this engine).
- **Per-layer LRU cache** for expert residency in RAM, plus a **learned "hot" pin-store**: the engine observes which experts actually get routed to over time and pins the frequently-used ones, persisting this usage pattern to disk (Colibri's `.coli_usage` file) so a *second* run of the same workload is faster than the first — "a JIT, but for weights."
- **One-layer-ahead predictive prefetch**: while computing layer *N*, the router's likely expert selection for layer *N+1* is predicted (Colibri reports 71.6% prediction accuracy) and those experts' disk reads are kicked off early, hiding NVMe latency behind compute.
- **Dual-SSD striping**: on machines with more than one NVMe device, reads are striped across them weighted by measured per-device bandwidth.
- Reads safetensors directly via mmap (no format conversion step) — reinforces §4.2's design (mmap-first for both GGUF and safetensors).

**Adaptation for this engine:**
1. Build the same three-tier placement decision at model-load time: query available VRAM (already queried this session, §3) and system RAM, compare against the model's dense-vs-routed parameter split (readable from the tensor metadata table — dense layer tensors vs `ffn_*_exps` MoE tensors are structurally distinguishable), and assign residency accordingly.
2. The **CPU compute backend** this requires (for RAM-resident or disk-streamed experts that can't be cheaply shuttled to the GPU every token) must share the *exact same* `gemm_row_stride_bytes`/tensor-metadata logic as the GPU path (§6.2, §2's rule) — a second, independently-written CPU formula is exactly the failure mode this whole document exists to avoid. It does not need to be fast; it needs to be a correct fallback and a correctness oracle (§9, Phase H's gate: a model that fits fully in VRAM must produce byte-identical output run fully-GPU vs forced-partial-CPU-offload).
3. Implement the learned hot-pin/prefetch mechanism as a genuine phase (not a footnote) — it's the difference between Colibri's usable performance and a naive "always cold-read every expert" implementation, which the search results note runs at only 0.05–0.1 tok/s.

---

## 8. Quantization Format Reference (comprehensive, researched 2026)

### 8.1 Weight quantization — GGUF-native block formats

| Format | Bits/weight | Block structure | Status | Priority |
|---|---|---|---|---|
| F32 / F16 / BF16 | 32/16/16 | Unquantized | Baseline, always supported | Required (dense-pipeline reference model uses this or Q8_0) |
| Q4_0, Q4_1, Q5_0, Q5_1 | 4/4/5/5 | Legacy, 32-element blocks, 1–2 float scales | Still in use, simple | Medium |
| Q8_0 | 8 | 32-element blocks, 1 FP16 scale | Common, high quality at 8-bit | High — first non-FP16 format to implement |
| Q2_K … Q6_K, Q8_K (K-quants) | 2–8 (variable per sub-block) | 256-element superblocks, per-sub-block scale+min, FP16 superblock scale | Current mainstream "good quality/size tradeoff" formats (`Q4_K_M` is the most commonly recommended default in 2026 sources) | High |
| IQ1_S, IQ1_M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL, IQ4_XS | 1–4 (importance-weighted) | Codebook-based, more complex dequant than K-quants | Used for aggressive compression of large models | Medium — implement after K-quants |
| TQ1_0 | 1.6875 | 5 trits packed per byte (3⁵=243<256) | **Ternary**, for BitNet b1.58 / TriLM models | Medium — CPU-only in upstream `llama.cpp` as of this research; this engine should target GPU support as a differentiator |
| TQ2_0 | 2.0625 | 2 bits/element | **Ternary**, same model family as TQ1_0, simpler packing | Medium, alongside TQ1_0 |
| MXFP4 | 4 (E2M1) | 32-element blocks, power-of-two scale (open OCP standard) | Cross-platform (confirmed on both NVIDIA Blackwell and AMD MI355X) — the more relevant FP4 format for AMD hardware of the two | Medium — no confirmed native RDNA4 consumer-GPU compute path (§3); implement as a storage format that dequantizes to FP8 before WMMA |
| NVFP4 | 4 (E2M1) | 16-element blocks, FP8 scale (NVIDIA) | Higher accuracy than MXFP4 at equal calibration, but NVIDIA-hardware-first with first-class toolchain only there | Low for this engine (AMD-first target) — support loading/dequant for portability, don't expect native-speed compute |
| DFloat11 (DF11) | ~11 (dynamic-length, lossless) | Entropy-style variable-length | **Lossless** compression, ~70% of original size at 100% accuracy, needs custom GPU kernels for the variable-length unpack | Low/exploratory — valuable if a "lossless but smaller" tier is wanted, but the dequant kernel is nontrivial |

### 8.2 Weight quantization — safetensors-paired schemes

Covered in §4.2.1 (AWQ, GPTQ, EXL2, HQQ, bitsandbytes NF4) — repeated here only as a pointer; don't duplicate the table.

### 8.3 KV-cache quantization (a *separate axis* from weight quantization)

**TurboQuant** (Google Research, ICLR 2026, `arXiv` "Online Vector Quantization with Near-optimal Distortion Rate") is **not a weight format** — it quantizes the KV cache specifically, online, as keys/values are written during inference: a random rotation followed by per-coordinate scalar quantization against a precomputed near-optimal codebook, achieving ~3-bit keys / 2-bit values with reported 6× memory reduction and up to 8× attention-logit speedup on H100, without calibration or fine-tuning. A community reference implementation exists (`github.com/0xSero/turboquant`, Triton kernels + vLLM integration) that would need to be reimplemented as Vulkan compute shaders for this engine — the algorithm (rotate, then quantize against a fixed codebook) is a reasonable compute-shader shape (a small matmul for the rotation + a per-element codebook lookup), but budget real design time for it; it is independent of and complementary to weight quantization (a model could run Q4_K weights *and* TurboQuant KV cache simultaneously) and should be scoped as its own phase, not bundled into the weight-quant phases.

### 8.4 Not quantization formats (flagged explicitly to avoid confusion)

**DSpark** and **DFlash** are **speculative-decoding drafter architectures**, not quantization schemes, despite superficially quant-adjacent naming:
- **DFlash**: a block-diffusion drafter that denoises a whole 16-token block in one parallel pass, reusing the target model's own embedding and LM-head weights.
- **DSpark**: a parallel 5-layer backbone that consumes the target model's hidden states and proposes a 7-token block at once, with a lightweight rank-256 "Markov head" correction (to avoid suffix-decay degeneration) and an optional confidence head for adaptive block length. Reported ~26–31% better accepted-length than Eagle3 and ~16–18% better than DFlash on Qwen-family targets.

Both are draft-then-verify speculative decoding: the small drafter proposes multiple tokens, the full model verifies them in one batched forward pass, and only genuinely-agreeing tokens are accepted — a decode-speed optimization, not a compression technique, and belongs in §9's speculative-decoding phase, not the quantization phases. (Support for MTP-style extra tensors in a GGUF, distinct from DSpark/DFlash, is a related but simpler requirement — see companion plan Phase 16 — the loader must not crash on unrecognized extra tensors regardless of which speculative scheme, if any, is implemented.)

---

## 9. Delivery Phasing (extends the companion plan; harness-first throughout)

The companion plan's Phases 0–19 remain the correct spine for the **dense-model pipeline** (device init → GGUF parsing → generic GEMM → RoPE/bias/qk_norm correctness gates → batched prefill → RDNA2 fallback → performance pass). This document adds the following, inserted at the indicated points — every phase still ends with a real generation test before the next starts, and Llama-3.2-1B remains the regression fixture that must stay green throughout:

| New/extended phase | Scope | Gate |
|---|---|---|
| Extend Phase 5 (quant formats) | Add MXFP4 (storage+dequant), TQ1_0/TQ2_0 ternary, remaining IQ-quants per §8.1's priority order | Bit-unpack unit tests against known vectors per format (mandatory before wiring into the shared GEMM path, per this session's demonstrated bug pattern around careless bit-packing) |
| New: safetensors loader | GGUF-parity tensor-metadata-table builder for safetensors headers; AWQ then GPTQ then EXL2 then HQQ adapters (§4.2.1 priority) | Same Llama-3.2-1B-equivalent-in-safetensors prompt test matches the GGUF path's output within quant tolerance |
| New: coopmat GEMM path (parallel track to the plan's Phase 19, can start once Phase 6's scalar GEMM is solid) | `VK_KHR_cooperative_matrix` variant of `gemm.comp` for FP8/BF8/INT4 dtype combos | Output matches the scalar-path baseline within fp8/quant tolerance; benchmark tok/s improvement (target: directionally match VulkanForge's reported gains) |
| New: MLA attention kernel | Naive-mode MLA first (§5.1), verified against a DeepSeek-family reference; absorbed-mode MLA second, byte-identical to naive mode | Coherent generation on a DeepSeek-V2/R1-class or Kimi-K2-class model |
| Extend companion Phase 13 (MoE) | Fine-grained routing (many small experts + shared experts, sigmoid *and* softmax gating) in addition to coarse routing | Coherent output + sane routing distribution on both a coarse-MoE (Llama 4-class) and fine-grained-MoE (GLM/DeepSeek-class) model |
| New: hybrid memory hierarchy (Colibri-style, §7) | Three-tier placement, per-layer LRU + learned hot-pin, one-layer-ahead prefetch, dual-SSD striping, CPU compute backend | A model too large for VRAM runs correctly; a model that fits both ways produces byte-identical output GPU-only vs forced-partial-offload; repeated runs of the same workload measurably speed up (hot-pin cache working) |
| New: KV-cache quantization (TurboQuant-style, §8.3) | Rotation + codebook-quantized KV cache as an independent toggle from weight quantization | Memory reduction matches expected ratio (~6×); generation quality within documented tolerance vs unquantized KV cache |
| New: speculative decoding (DSpark/DFlash-style, §8.4) | Small drafter model + batched verify pass | Output distribution matches non-speculative decoding (this is a speed optimization, must not change what gets generated in expectation); measure accepted-token-length and tok/s improvement |
| New: LiteRT-LM loader (§4.3) | `.litertlm` flatbuffer parser → internal tensor-metadata-table extractor for `PREFILL_DECODE` subgraphs | Coherent generation on a small edge-oriented LiteRT-LM model; lowest priority of the three format loaders, schedule last |
| Extend companion Phase 17 (Laguna) | Now explicitly requires: MLA or its sliding-window/global-mixed-attention sibling, YaRN, partial rotary, sigmoid MoE gating, attention-output gating, all independently verified first | Coherent generation on Laguna |

---

## 10. Open Questions and Risks (be honest with the implementing agent about these)

1. **RDNA4 native FP4 matrix support is unconfirmed.** Public sources describe FP8/BF8/INT4/INT8 WMMA explicitly; none describe native FP4 WMMA. Until proven otherwise on real hardware, MXFP4/NVFP4 should be architected as *storage* formats with a dequant-to-FP8-then-WMMA path, not assumed to run natively. If RDNA4 does gain native FP4 matrix support (driver/ISA update), this is a localized change to the coopmat kernel selection logic, not a redesign — confirms the value of keeping quant format and compute-kernel-selection as separate concerns (§6.2's abstraction already supports this).
2. **LiteRT-LM's actual on-disk tensor/quantization scheme within the TFLite flatbuffer needs deeper spec reading than this document performed** — the DeepWiki/Google sources describe the container structure (sections, `TFLiteModelType`) but not the exact quantization scheme(s) used for weights inside a `PREFILL_DECODE` subgraph. Treat §4.3 as a starting point, not a finished spec; the implementing agent should read `google-ai-edge/LiteRT-LM`'s actual schema files before writing the extractor.
3. **TurboQuant has no existing Vulkan implementation** to port from (the reference implementation is CUDA/Triton) — budget real design time, not just a porting pass.
4. **Colibri's exact codebase should be read directly** (`github.com/JustVugg/colibri`, pure C) before implementing §7 — this document summarizes its architecture from documentation/README-level research, not a full source read; the implementing agent has a real, working reference to study line-by-line rather than reinvent from this summary alone.
5. **NVFP4 priority is deliberately set low** given the AMD-first target — revisit if a specific high-value model is only available pre-quantized in NVFP4 with no MXFP4/GGUF equivalent.

---

## Sources (from this session's research; verify currency before implementation, especially anything version/date-specific)

- [GGUF quantization types — kaitchup.substack.com](https://kaitchup.substack.com/p/gguf-quantization-for-fast-and-memory)
- [Which Quantization Should I Use? — arXiv](https://arxiv.org/html/2601.14277v1)
- [llama.cpp quantize README — GitHub](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md)
- [NVFP4 vs MXFP4 decision guide — Spheron](https://www.spheron.network/blog/nvfp4-vs-mxfp4-gpu-cloud-4bit-quantization-guide/)
- [Day 4: Quantization on DGX Spark — Kubesimplify](https://blog.kubesimplify.com/day-4-quantization-demystified-bf16-fp8-nvfp4-mxfp4-int4-gguf-and-why-it-all-matters)
- [NVFP4 throughput — Medium/Data Science Collective](https://medium.com/data-science-collective/nvfp4-same-accuracy-with-2-3x-higher-throughput-for-4-bit-llms-03518ecba108)
- [LiteRT-LM model formats — DeepWiki](https://deepwiki.com/google-ai-edge/LiteRT-LM/6-model-formats-and-schema)
- [LiteRT-LM GitHub](https://github.com/google-ai-edge/LiteRT-LM)
- [TurboQuant — Google Research blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [TurboQuant community implementation — GitHub](https://github.com/0xSero/turboquant)
- [Colibri — Wavect blog](https://wavect.io/blog/colibri-glm-5-2-consumer-hardware/)
- [Colibri — GitHub (JustVugg)](https://github.com/JustVugg/colibri)
- [Colibri GLM-5.2 fork — GitHub (AvaBillions2040)](https://github.com/AvaBillions2040/colibri-LLM-on-SSD-14-07-2026)
- [DSpark/DFlash — mlx-dspark GitHub](https://github.com/ARahim3/mlx-dspark)
- [DSpark llama.cpp PR](https://github.com/ggml-org/llama.cpp/pull/25173)
- [DFlash/Spec V2 — LMSYS blog](https://www.lmsys.org/blog/2026-06-15-next-generation-speculative-decoding-dflash-v2/)
- [DFloat11 — arXiv](https://arxiv.org/pdf/2504.11651)
- [GGUF vs AWQ vs GPTQ vs EXL2 — GIGAGPU](https://gigagpu.com/awq-vs-gptq-vs-gguf-vs-exl2-2026/)
- [Quantization methods compared — ai.rs](https://ai.rs/ai-developer/quantization-methods-compared)
- [VulkanForge — GitHub](https://github.com/maeddesg/vulkanforge)
- [Vulkan cooperative matrix — Vulkanised 2025 slides](https://www.vulkan.org/user/pages/09.events/vulkanised-2025/T47-Jeff-Bolz-NVIDIA.pdf)
- [NVIDIA Vulkan ML success — Phoronix](https://www.phoronix.com/news/NVIDIA-Vulkan-AI-ML-Success)
- [Multi-head Latent Attention — DeepWiki](https://deepwiki.com/deepseek-ai/DeepSeek-V3/4.2-multi-head-latent-attention-(mla))
- [DeepSeek-V3 Technical Report — arXiv](https://arxiv.org/pdf/2412.19437)
- [Top LLMs as of August 2026 — Shakudo](https://www.shakudo.io/blog/top-9-large-language-models)
- [Top MoE models 2026 — Labellerr](https://www.labellerr.com/blog/top-open-source-moe-llms/)
- [RDNA4 FP8/INT4 WMMA — AMD GPUOpen](https://gpuopen.com/learn/accelerating_generative_ai_on_amd_radeon_gpus/)
- [llama.cpp ternary packing PR #8151](https://github.com/ggml-org/llama.cpp/pull/8151)
