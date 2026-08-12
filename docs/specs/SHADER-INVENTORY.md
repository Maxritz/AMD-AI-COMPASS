# Shader Inventory — Vulkan LLM Inference Engine

## 1. Overview

This document is the canonical cross-reference of every compute shader used by the
engine. Each shader entry maps the `.comp` source file, the SPIR-V binaries
produced, the CMake `register_shader` call, the pipeline stage it serves, and
the spec section that describes its internals.

**Total shaders registered:** 26 base names × 2 wave sizes (W32, W64)
= **52 SPIR-V binaries**.

Shader sources live in `src/rdna4-llm-legacy/shaders/` (full legacy set) and
`src/rdna4-llm/shaders/` (Phase 0 active harness). The full 26-shader set
materialsizes in Phase 2 when the CMakeLists.txt registry is expanded beyond
the current Phase-0 `harness_add.comp` stub.

## 2. Shader Registry (as defined in Implementation §1.2)

The canonical registration list (sorted by pipeline stage / execution order):

| # | Base Name            | Wave Sizes | Defines                  | Quantization | Pipeline Stage         |
|---|----------------------|------------|--------------------------|--------------|------------------------|
| 1 | `rms_norm`           | 32;64      | (none)                   | N/A          | Pre-attention norm     |
| 2 | `attn_qkv_fp16`      | 32;64      | `QUANT_FP16`             | FP16         | QKV projection + RoPE  |
| 3 | `attn_qkv_q4_k`      | 32;64      | `QUANT_Q4_K`             | Q4_K         | QKV projection + RoPE  |
| 4 | `attn_qkv_q6_k`      | 32;64      | `QUANT_Q6_K`             | Q6_K         | QKV projection + RoPE  |
| 5 | `attn_qkv_q8_0`      | 32;64      | `QUANT_Q8_0`             | Q8_0         | QKV projection + RoPE  |
| 6 | `attn_qkv_iq4_xs`    | 32;64      | `QUANT_IQ4_XS`           | IQ4_XS       | QKV projection + RoPE  |
| 7 | `attn_compute`       | 32;64      | (none)                   | N/A          | Attention softmax      |
| 8 | `attn_output_fp16`   | 32;64      | `QUANT_FP16`             | FP16         | Attention O-projection   |
| 9 | `attn_output_q4_k`   | 32;64      | `QUANT_Q4_K`             | Q4_K         | Attention O-projection   |
|10 | `attn_output_q6_k`   | 32;64      | `QUANT_Q6_K`             | Q6_K         | Attention O-projection   |
|11 | `attn_output_q8_0`   | 32;64      | `QUANT_Q8_0`             | Q8_0         | Attention O-projection   |
|12 | `attn_output_iq4_xs` | 32;64      | `QUANT_IQ4_XS`           | IQ4_XS       | Attention O-projection   |
|13 | `ffn_gate_up_fp16`   | 32;64      | `QUANT_FP16`             | FP16         | FFN gate+up (fused)     |
|14 | `ffn_gate_up_q4_k`   | 32;64      | `QUANT_Q4_K`             | Q4_K         | FFN gate+up (fused)     |
|15 | `ffn_gate_up_q6_k`   | 32;64      | `QUANT_Q6_K`             | Q6_K         | FFN gate+up (fused)     |
|16 | `ffn_gate_up_q8_0`   | 32;64      | `QUANT_Q8_0`             | Q8_0         | FFN gate+up (fused)     |
|17 | `ffn_gate_up_iq4_xs` | 32;64      | `QUANT_IQ4_XS`           | IQ4_XS       | FFN gate+up (fused)     |
|18 | `ffn_down_fp16`      | 32;64      | `QUANT_FP16`             | FP16         | FFN down-projection    |
|19 | `ffn_down_q4_k`      | 32;64      | `QUANT_Q4_K`             | Q4_K         | FFN down-projection    |
|20 | `ffn_down_q6_k`      | 32;64      | `QUANT_Q6_K`             | Q6_K         | FFN down-projection    |
|21 | `ffn_down_q8_0`      | 32;64      | `QUANT_Q8_0`             | Q8_0         | FFN down-projection    |
|22 | `ffn_down_iq4_xs`    | 32;64      | `QUANT_IQ4_XS`           | IQ4_XS       | FFN down-projection    |
|23 | `lm_head_fp16`       | 32;64      | `QUANT_FP16`             | FP16         | Final logits           |
|24 | `lm_head_q4_k`       | 32;64      | `QUANT_Q4_K`             | Q4_K         | Final logits           |
|25 | `lm_head_q6_k`       | 32;64      | `QUANT_Q6_K`             | Q6_K         | Final logits           |
|26 | `lm_head_q8_0`       | 32;64      | `QUANT_Q8_0`             | Q8_0         | Final logits           |
|27 | `lm_head_iq4_xs`     | 32;64      | `QUANT_IQ4_XS`           | IQ4_XS       | Final logits           |
|28 | `token_embed`        | 32;64      | (none)                   | N/A          | Token embedding lookup |

> **Note:** The numbering above (1–28) reflects the actual shader source files
> present in `src/rdna4-llm-legacy/shaders/`. The `register_shader` list in the
> Implementation doc (§1.2) lists 26 entries because `lm_head_iq4_xs` is
> registered as `lm_head_iq4_xs` (entry 27 in the file table) and
> `attn_output_iq4_xs` as entry 12 — the file on disk confirms all variants.

## 3. Quantization Variant Matrix

Each kernel family (except `rms_norm`, `attn_compute`, `token_embed`) is
parameterized by weight quantization type via preprocessor defines.

| Kernel Family     | FP16 | Q4_K | Q6_K | Q8_0 | IQ4_XS | Variants |
|-------------------|------|------|------|------|--------|----------|
| `attn_qkv`        |  Y   |  Y   |  Y   |  Y   |   Y    | 5        |
| `attn_output`     |  Y   |  Y   |  Y   |  Y   |   Y    | 5        |
| `ffn_gate_up`     |  Y   |  Y   |  Y   |  Y   |   Y    | 5        |
| `ffn_down`        |  Y   |  Y   |  Y   |  Y   |   Y    | 5        |
| `lm_head`         |  Y   |  Y   |  Y   |  Y   |   Y    | 5        |
| `rms_norm`        |  —   |  —   |  —   |  —   |   —    | 0 (N/A)  |
| `attn_compute`    |  —   |  —   |  —   |  —   |   —    | 0 (N/A)  |
| `token_embed`     |  —   |  —   |  —   |  —   |   —    | 0 (N/A)  |
| **Total**         |  5   |  5   |  5   |  5   |   5    | **25**   |

Plus 3 non-quantized shaders = **28 source files** total.

## 4. Output Naming Convention

Each shader produces two SPIR-V binaries — one per wave size:

```
spirv/<base_name>_w32.spv   ← compiled with --target-env=vulkan1.3 + subgroupSize=32
spirv/<base_name>_w64.spv   ← compiled with --target-env=vulkan1.3 + subgroupSize=64
```

Generated headers:

```
generated/<base_name>_w32.h
generated/<base_name>_w64.h
```

The auto-generated `shader_registry.h` provides an index mapping each
`(base_name, wave_size)` pair to its SPIR-V binary path and header symbol,
so the runtime can select the correct pipeline at initialization based on
the detected architecture (RDNA4 → W32, RDNA2 → W64).

## 5. Spec Reference Cross-Index

| Shader             | Spec Section | Description                          |
|--------------------|-------------|--------------------------------------|
| `rms_norm`          | §1          | RMS normalization + scale            |
| `attn_qkv_fp16`     | §2          | Fused QKV + RoPE + KV cache write    |
| `attn_qkv_q4_k`     | §2          | (same as above, Q4_K weights)        |
| `attn_qkv_q6_k`     | §2          | (same as above, Q6_K weights)        |
| `attn_qkv_q8_0`     | §2          | (same as above, Q8_0 weights)        |
| `attn_qkv_iq4_xs`   | §2          | (same as above, IQ4_XS weights)      |
| `attn_compute`      | §3          | Attention score × softmax × reduce   |
| `attn_output_fp16`  | §4          | O-projection (weighted sum of V)     |
| `attn_output_q4_k`  | §4          | (same as above, Q4_K weights)        |
| `attn_output_q6_k`  | §4          | (same as above, Q6_K weights)        |
| `attn_output_q8_0`  | §4          | (same as above, Q8_0 weights)        |
| `attn_output_iq4_xs`| §4          | (same as above, IQ4_XS weights)      |
| `ffn_gate_up_fp16`  | §5          | SiLU(gate) × up (fused)              |
| `ffn_gate_up_q4_k`  | §5          | (same as above, Q4_K weights)        |
| `ffn_gate_up_q6_k`  | §5          | (same as above, Q6_K weights)        |
| `ffn_gate_up_q8_0`  | §5          | (same as above, Q8_0 weights)        |
| `ffn_gate_up_iq4_xs`| §5          | (same as above, IQ4_XS weights)      |
| `ffn_down_fp16`     | §6          | Down-projection (SiLU(gate)×up → D)  |
| `ffn_down_q4_k`     | §6          | (same as above, Q4_K weights)        |
| `ffn_down_q6_k`     | §6          | (same as above, Q6_K weights)        |
| `ffn_down_q8_0`     | §6          | (same as above, Q8_0 weights)        |
| `ffn_down_iq4_xs`   | §6          | (same as above, IQ4_XS weights)      |
| `lm_head_fp16`      | §7          | Final linear → logits                |
| `lm_head_q4_k`      | §7          | (same as above, Q4_K weights)        |
| `lm_head_q6_k`      | §7          | (same as above, Q6_K weights)        |
| `lm_head_q8_0`      | §7          | (same as above, Q8_0 weights)        |
| `lm_head_iq4_xs`    | §7          | (same as above, IQ4_XS weights)      |
| `token_embed`       | §8          | Token ID → embedding vector lookup   |

## 6. File Inventory (on disk)

**Legacy shaders** (`src/rdna4-llm-legacy/shaders/`):

| File | Spec Section |
|------|-------------|
| `rms_norm.comp`        | §1  |
| `attn_qkv_fp16.comp`   | §2  |
| `attn_qkv_q4_k.comp`   | §2  |
| `attn_qkv_q6_k.comp`   | §2  |
| `attn_qkv_q8_0.comp`   | §2  |
| `attn_qkv_iq4_xs.comp` | §2  |
| `attn_compute.comp`    | §3  |
| `attn_output_fp16.comp`| §4  |
| `attn_output_q4_k.comp`| §4  |
| `attn_output_q6_k.comp`| §4  |
| `attn_output_q8_0.comp`| §4  |
| `attn_output_iq4_xs.comp` | §4 |
| `ffn_gate_up_fp16.comp`| §5  |
| `ffn_gate_up_q4_k.comp`| §5  |
| `ffn_gate_up_q6_k.comp`| §5  |
| `ffn_gate_up_q8_0.comp`| §5  |
| `ffn_gate_up_iq4_xs.comp` | §5 |
| `ffn_down_fp16.comp`  | §6  |
| `ffn_down_q4_k.comp`  | §6  |
| `ffn_down_q6_k.comp`  | §6  |
| `ffn_down_q8_0.comp`  | §6  |
| `ffn_down_iq4_xs.comp` | §6 |
| `lm_head_fp16.comp`   | §7  |
| `lm_head_q4_k.comp`   | §7  |
| `lm_head_q6_k.comp`   | §7  |
| `lm_head_q8_0.comp`   | §7  |
| `lm_head_iq4_xs.comp` | §7  |
| `token_embed.comp`    | §8  |
| `common.glsl`         | §0.6 (shared include) |

**Active shaders** (`src/rdna4-llm/shaders/`):

| File |
|------|
| `harness_add.comp` (Phase 0 test harness) |

## 7. Pipeline Stage Summary

Each decode step dispatches shaders in this order (Architecture §12.1):

1. **RMS norm** (input) — `rms_norm` → normalizes hidden state before attention
2. **QKV projection** — `attn_qkv_{quant}` → projects Q, K, V + RoPE + KV cache write
3. **Attention compute** — `attn_compute` → QK^T softmax × V reduction
4. **Attention output** — `attn_output_{quant}` → O-projection, adds residual
5. **RMS norm** (FFN input) — `rms_norm` → normalizes for FFN
6. **FFN gate+up** — `ffn_gate_up_{quant}` → SiLU(gate) × up (fused)
7. **FFN down** — `ffn_down_{quant}` → down-projection, adds residual
8. **LM head** (final) — `lm_head_{quant}` → logits projection

**Total per decode step:** 7 dispatches/layer × 32 layers + 1 LM head = 225
(Spec §12 dispatch ordering).
