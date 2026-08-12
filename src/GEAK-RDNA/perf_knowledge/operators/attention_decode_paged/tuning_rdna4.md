---
title: attention_decode_paged - tuning (RDNA4 / RX 9000)
kind: operator_overview
operator: attention_decode_paged
gens: [gfx1201, gfx1203, gfx1206, gfx1207]
dtypes: [fp16, bf16, fp8_e4m3]
regimes: [decode]
updated: 2026-08-12
sources:
  - ../hardware/rdna4_rx9000/arch.md
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
---

# attention_decode_paged - tuning (RDNA4 / RX 9000)

RDNA4 port of the CDNA tuning guide. Decode is **memory-bandwidth + launch-latency bound** on
RDNA4 too (one-token query, no GEMM to tile), but the machine differs from CDNA in ways that
change the levers:

| RDNA4 (gfx1201, RX 9070 XT) | vs CDNA3/4 (MI300X/MI355X) | Why it matters for decode |
|---|---|---|
| **32 lanes / Wave32-native** | 64-lane subgroups on CDNA | half the lanes per wavefront → more wavefronts needed to hide GDDR6 latency |
| **64 CUs active** | 304 / 256 CUs | tiny grid; a one-CU-per-(batch,head) decode starves even more → splitKV is more critical |
| **WMMA, no MFMA** | MFMA matrix cores | no `matrix_instr_nonkdim` lever; WMMA tile shapes differ, no MFMA-4/MFMA-16 split |
| **128 KiB LDS/CU, 32 banks** | 64 KiB (MI300X) / 64 KiB (MI355X) | 2x LDS room for KV staging per wave |
| **8 MB unified L2** | 4 MB (MI300X) / 4 MB (MI355X) | 2x L2 residency → KV-cache reuse matters more |
| **640 GB/s GDDR6** | 3.7 / 4.9 TB/s HBM3 | ~6-8x less bandwidth → **the** ceiling. KV reads dominate decode |
| **~2.4-3.0 GHz** | ~2.0 GHz | higher clock helps the (small) compute part, not the memory part |

## Measured RDNA4 baseline (llama.cpp ggml, hip_tracer, 33B MoE Q4_K)
- Attention **22.5%** of wall time, occupancy **5.2%** (RDNA4 32-wave profile).
- MMVQ 38.8%, Vector 29.2% — so on llama.cpp, decode attention is not even the top cost;
  the KV-cache read + mat-vec path (MMVQ) is.
- GPU busy 35.8% — half-idle GPU on a 33B MoE: launch latency + memory stalls dominate.

## Lever 1 - splitKV / flash-decoding (the main lever, more than on CDNA)
With 64 CUs and small batch, a naive one-CU-per-(batch,head) grid uses 2-8 CUs of 64. Split the KV
history across CUs, reduce partial `(O, m, l)` in a stage-2 kernel.
- llama.cpp ggml fattn: the tile-based `fattn-tile.cu` already splits; verify
  `GGML_CUDA_FATTN_BLOCK`/`GGML_CUDA_FATTN_DP` are picked for gfx1201, not falling back to the
  vec path (a 5.2% occupancy signature points at the vec path).
- vLLM/AITER on RDNA4: `ROCM_AITER_FA > ROCM_AITER_UNIFIED_ATTN > TRITON_ATTN > ROCM_ATTN`
  ranking holds; the TPS delta vs fallback is even larger at 640 GB/s because the fallback
  wastes bandwidth.
- **Rule**: more KV splits when batch×heads is small (fill 64 CUs); fewer when batch is already
  large. On RDNA4 the "small" region is wider than CDNA (fewer CUs to fill).

## Lever 2 - KV-cache layout & coalescing (the bandwidth lever)
- GDDR6 at 640 GB/s means **uncoalesced KV reads are fatal**. Goal: 128-bit coalesced loads
  (`global_load_dwordx4` / `buffer_load_dwordx4`) on every KV lane read.
- ggml: `GGML_CUDA_KV_SIZE`/KV layout — the fp16 KV path should read `dwordx4` per 8 fp16 values.
  Verify in ISA; a scalarized load is the #1 RDNA4 decode bandwidth killer.
- vLLM: `VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=1` applies on RDNA4 too.
- **8 MB L2**: keep the KV chunk for the current sequence L2-resident across decode steps when
  head_dim × seq is small enough; the 2x L2 vs CDNA makes this actually achievable.

## Lever 3 - Wave32 occupancy
- RDNA4 is Wave32-native; target **high wavefront count per CU** to hide GDDR6 latency — decode
  is latency-bound, so occupancy beats waves-per-SIMD register pressure.
- Favor small per-thread register footprint + more wavefronts over fewer fat waves. The 5.2%
  measured occupancy should climb to 50%+ with the vec→tile path + splitKV.
- No MFMA to trade against; WMMA fattn is FP16-friendly but the memory read is the wall.

## Lever 4 - launch overhead (small batch, RDNA4 is worse)
- 64 CU GPU → dispatch overhead is a bigger fraction than on 304-CU CDNA.
- Levers: HIP-graph capture for decode (ggml `GGML_CUDA_DISABLE_GRAPHS=0`), persistent decode
  kernels, unified attention (one kernel for prefill+decode). `HSA_NO_SCRATCH_RECLAIM=1` helps
  where supported (Linux).

## What to verify after a change (RDNA4)
1. Re-trace with hip_tracer GPU timing; confirm attention occupancy > 20% and
   MMVQ+Attention combined drops below ~50% of wall.
2. Check KV loads are `dwordx4` in ISA (RGA `-s bin` disassembly, gfx1201).
3. Guard: 640 GB/s floor. If TG on the 33B stays < ~45 t/s after tuning, you are still
   bandwidth-bound — attack KV coalescing, not FLOPs.
