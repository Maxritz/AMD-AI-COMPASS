# RDNA2 (gfx1031) support porting — AMD-AI-COMPASS

> Companion to `docs/specs/GPU_CAPABILITIES.md` (RX 6700 XT cap matrix) and
> `src/GEAK-RDNA/perf_knowledge/hardware/rdna2_rx6000/` (canonical RDNA2 facts).
> Target SKU: AMD Radeon RX 6700 XT, gfx1031, 40 CUs, 96 MiB Infinity Cache, 3 MiB L2.

## TL;DR — verdicts (per `ponytail-port` model)

```
detect_gpu         → adapt | feas: green | verify: r3 | risk: low | doc: ports/RDNA2-gfx1031-support.md
_autodetect_gpu_type → adapt | feas: green | verify: r3 | risk: low | doc: ports/RDNA2-gfx1031-support.md
ARCH_SPECS[gfx1031]  → direct | feas: green | verify: r3 | risk: low
```

## Status

| Area | Artifact | State |
|------|----------|-------|
| Detection (Arbor/Apex gpu-info MCP) | `src/Apex-RDNA4/tools/mcps/gpu_info/server.py` `GPUDetector.detect_gpu` | **BUG FIXED** (see §3.1) |
| Arch spec DB | `ARCH_SPECS["gfx1030"]/["gfx1031"]/["gfx1032"]` | **ADDED** (RDNA2 vector-only profile, no MFMA/WMMA/FP8) |
| Orchestrator type resolution | `src/hyperloom/inference_optimizer/gpu_types.py` `_GFX_TO_RUNNER` / `_AMD_GPU_DISPATCH_IDENTITIES` | Pre-existing: `gfx1031 -> rx6700xt (40 CU)` — no change needed |
| Roofline peaks | `src/hyperloom/orchestrator/kernel/roofline_ceiling.py` + `TraceLens/.../RX6700XT.json` | Pre-existing RDNA2 peaks (FP32 9–12.4, FP16 18–24.8, INT8 35–49.6) — no change needed |
| Kernel DB (Wave64) | `tools/analyze.py` `KERNEL_DB_RDNA2` | Pre-existing — no change needed |
| README supported GPUs | `README.md` §Supported GPUs | Pre-existing: lists `gfx1030, gfx1031, gfx1032` RDNA2 |

## 1. Why this port exists

RDNA2 has **no matrix hardware**: no MFMA (CDNA), no WMMA (RDNA3+/RDNA4), no FP8.
All GEMM/attention MATMUL must use scalar/vector **FMA** + packed dot-product
intrinsics (`v_fma_f32`, `v_dot2_f32_f16`, `sdot4`/`udot4`). Without an explicit
`gfx1030/1031` entry in `ARCH_SPECS`, the gpu-info MCP fell through to
`DEFAULT_ARCH="gfx950"` (MI355X, CDNA4) and fed agents **wrong hints**
(MFMA/WMMA/FP8 enabled, HBM bandwidth, Wave64-as-occupancy rules).

## 2. Added: ARCH_SPECS RDNA2 profile (`server.py`)

`gfx1030` (reference SKU = RX 6700 XT) with `gfx1031`/`gfx1032` aliases aliasing it.
Cross-checked against `rdna2_rx6000/peak_tables.md` RX 6700 XT row:

| Field | Value | Source |
|-------|-------|--------|
| compute_units | 40 | rocminfo `Compute Unit: 40` |
| wavefront_size | 64 | Wave64 fixed on RDNA2 |
| l2_cache_mb | 3 | peak_tables.md |
| infinity_cache_mb | 96 | 96 MiB MALL on 6700 XT |
| memory_bandwidth_gb_s | 384 | GDDR6 bus |
| effective_memory_bandwidth_gb_s | 1278 | IC-amortized (renderer key `effective_memory_bandwidth_gb_s`) |
| fp32_tflops | 12.4 / fp16 24.8 / int8 49.6 | peak_tables.md |
| mfma_support / wmma_support / fp8_support | **False / False / False** | no matrix unit |
| bf16_support | False | emulated via FP32 |
| optimal_tile_sizes | M/N {64,128,256}, K {16,32,64} | IC-aware tiling (renderer keys `gemm_m/n/k`) |
| optimal_block_sizes | [256,128,64] | Wave64 multiples |

Renderer (`get_gpu_info`) consumes `l2_cache_mb`, `infinity_cache_mb`,
`effective_memory_bandwidth_gb_s`, `memory_bandwidth_gb_s` — all present, so the
IC + effective-BW rows render for gfx1031.

## 3. Bugs found & fixed (detection)

### 3.1 `detect_gpu` — rocminfo parse-order (server.py)

rocminfo emits the GPU agent block as:
```
Agent 2
  Name:                    gfx1031          # <-- arch string FIRST
  Marketing Name:          AMD Radeon RX 6700 XT
  Vendor Name:             AMD              # <-- in_gpu_section flagged HERE
  ...
  Compute Unit:            40
```
The old loop only captured `arch` inside `elif in_gpu_section:`, but
`in_gpu_section` was only raised on `Vendor Name: AMD` — which arrives **after**
the `Name: gfx1031` line. So `arch` stayed `None` → `DEFAULT_ARCH="gfx950"`,
producing MI355X hints on a 6700 XT.

Fix: capture the gcn arch on the `Name: gfx…` line directly (it uniquely identifies
a GPU agent; the CPU agent's Name has no `gfx`), and raise `in_gpu_section` there.

### 3.2 `_autodetect_gpu_type` — spaced marketing name (gpu_types.py)

`rocm-smi --showproductname` returns `AMD Radeon RX 6700 XT`; the tag list tests
`"RX6700XT" in out.upper()` (`"AMDRADEONRX6700 XT"`), which is `False` because of
spaces. Note: on this host `rocm-smi` is a broken `~/.local/bin` wrapper
(`FileNotFoundError: bin/rocm-smi`) → the `except` clause falls through to the
`torch.cuda.get_device_properties(0).gcnArchName` path, which already resolves
`gfx1031`. The rocm-smi branch is still fixed (space-insensitive match) for
environments where rocm-smi works on `PATH`.

## 4. Verification (live RX 6700 XT, gfx1031, ROCm 10.2)

```
detect_gpu -> arch='gfx1031' marketing='AMD Radeon RX 6700 XT' CUs=40
RDNA2 profile: mfma=False wmma=False IC=96MB L2=3MB
gpu_types resolved -> ('gfx1031', 40)
```
- Before fix: `arch='gfx950'`, `marketing='AMD Radeon RX 6700 XT'` (mismatched —
  MI355X arch + 6700 XT CU count) → wrong hint tier.
- After fix: `arch='gfx1031'`, consistent RDNA2 profile.

Numbers match `rdna2_rx6000/peak_tables.md` (RX 6700 XT row: FP32 12.4, FP16 24.8,
INT8 49.6 TFLOPS/TOPS, 384 GB/s GDDR6, 3 MiB L2) and `docs/specs/GPU_CAPABILITIES.md`.

## 5. Porting guidance for new RDNA2 SKUs

To add another RDNA2 SKU (e.g. `gfx1030` for 6800 XT / `gfx1032` for 6600 XT):

1. `gpu_types.py`: `_GFX_TO_RUNNER` and `_AMD_GPU_DISPATCH_IDENTITIES` already
   register `gfx1030 -> rx6800xt` and `gfx1032 -> rx6600xt`. Verify CU count
   against the SKU and bump `compute_units` in `ARCH_SPECS` alias if it should
   differ from the 6700 XT (shared profile otherwise — no MFMA/WMMA/FP8, Wave64).
2. `server.py` `ARCH_SPECS`: only the per-SKU name/compute_units/TDP differ; the
   matrix-absence and FMA-based `optimization_priorities` are identical across
   the gfx103x family, so the alias pattern (`ARCH_SPECS["gfx1031"] =
   ARCH_SPECS["gfx1030"]`) is the intended extension point.
3. Roofline peaks: `roofline_ceiling.py` already models RDNA2; add the new SKU's
   FP32/FP16/INT8 peaks only if a new `gpu_type` runner is needed.

## 6. Alternative-kernel / extension matrix (RDNA2)

RDNA2 has **no matrix unit** — no WMMA/coopmat/MFMA/FP8/int4. The tool selects FMA /
packed-int8-dot *alternatives* (not emulations) wherever CDNA/RDNA4 routes matrix kernels.
Wiring lives in three places: `ARCH_SPECS` (hints), `KERNEL_DB_RDNA2` in `tools/analyze.py`
(Wave64 block shapes), and the Vulkan engine's `SUBGROUP_SIZE=64` int8 (`dot4add`/`dot2add`)
+ FP16 (`v_dot2_f32_f16`) FMA/dot fallback paths.

| Matrix path (CDNA/RDNA4) | RDNA2 alternative (no emulation) | Tool-side binding |
|---|---|---|
| MFMA FP8×FP8 (32×32×16) GEMM | FP16/FP32 vector FMA (`v_fma_f32`) + FP8→FP32 unpack | `ARCH_SPECS` `optimization_priorities`; `KERNEL_DB_RDNA2` MMVQ/MMQ shapes |
| MFMA BF16 (16×16×16) | BF16 load → FP32 FMA accumulate (~FP32 cost) | `chunk_gated_delta_rule` FMA fallback (`#else` non-MFMA branch) |
| WMMA 16×16×16 FP16 (RDNA3+) | `dot2add_u8packed` (dp2a, universal on RDNA2) + `v_dot2_f32_f16`; subgroup-scalar reduction | `docs/VULKAN-COMPUTE-SHADER-SPEC.md` §2.11 |
| Co-op matrix (16×16×16) GEMM | scalar/subgroup reduction + `dot2add_u8packed` (no native `v_dot4`) | `docs/VULKAN-COMPUTE-SHADER-SPEC.md` §0.7; §2.11 dp2a fallback |
| FP8 GEMM / KV quant | Not available — use INT8/FP16 FMA | `rdna2_rx6000/fp8_wmma.md`: FP8 applies RDNA4/CDNA4 only |
| Flash attention (WMMA/FA) | Flash-attn with `SUBGROUP_SIZE=64` subgroup reduction | `KERNEL_DB_RDNA2` Attention `flash_attn` block 128/256 |

### Discrepancy note: L2
`ARCH_PROFILES["rdna2"].l2 = 4096` (4 MB) is the **family max** (6900/6800 XT). The
 6700 XT SKU is **3 MB L2** — which `ARCH_SPECS` correctly reports (`l2_cache_mb: 3`,
  matching `rdna2_rx6000/peak_tables.md`). gpu-info reports the SKU-correct 3 MB for
  gfx1031; `analyze.py`'s generic profile is fine family-wide, but tiling budgets should
  use the SKU value (3 MB) for the 6700 XT.

## 7. DONE vs PENDING status

| # | Work item | Artifact | Scope | Status |
|---|-----------|----------|-------|--------|
| A | Document gfx1031 in supported-arch list | `src/Apex-RDNA4/AGENTS.md:15` | docs/arch | **DONE** — added gfx1031 (FMA/INT8-dot, no MFMA/WMMA/FP8/int4) |
| B | RDNA2/FMA fallback kernel mappings | `src/Magpie-RDNA4/Magpie/tools/amd_kernel_finder/searcher.py` `known_hip_mappings` | kernel search | **DONE** — added `MmaLayerDesc` + `DeviceGemm` rows mapping MFMA kernels to FMA/dp4a fallback |
| C | Wire gpu-info MCP + pin mcp API | `src/Apex-RDNA4/mcp_config.json`; `requirements-eval.txt`; 4×`pyproject.toml` | infra | **DONE** — created stdio `mcp_config.json` (gpu-info + 3 other servers); pinned `mcp>=1.0,<2` everywhere |
| D | SKU-aware L2 tiling budget | `tools/analyze.py` `ARCH_PROFILES["rdna2"]` | tuning | **DONE** — documented 3072 KB (6700 XT) vs 4096 KB (family max) in comment |
| E | gfx1031 gpu override block | `tools/data/gpu_overrides.json` | config | **DONE** — added gfx1031 block (AITER disabled, native: fp16/int8, emulated: bf16, unsupported: fp8/mxfp4/int4) |

### Already complete (no pending work)
- `detect_gpu` / `_autodetect_gpu_type` (§3): rocminfo fallback + space-insensitive rocm-smi tag match — **DONE**.
- `ARCH_SPECS[gfx1031]` in `server.py`: RDNA2 vector-only profile (mfma/wmme/fp8/int4=false, IC 96, L2 3, eff BW 1278, FMA/dot2 priorities) — **DONE**.
- Roofline peaks, kernel DB, README supported-GPUs, Vulkan §2.11/§14 fallback paths, `rdna2_rx6000/*` canonical facts — **DONE** (pre-existing).

### Additional work (items F–I), all DONE

| #  | Work item | Artifact | Scope | Status |
|----|-----------|----------|-------|--------|
| F  | Fix `RX6700XT.json` perf numbers | `src/TraceLens-RDNA4/.../arch/RX6700XT.json` | TraceLens roofline | **DONE** — FP32→12.4 (was 9), FP16→24.8 (was 18), INT8→49.6 (was 35); removed bogus `vector_int4`; added `vector_fp64` + `matrix_int8` to match schema |
| G  | Fix `detect.py` RDNA2 gfx string | `tools/detect.py:127-133` | hardware detect | **DONE** — 6700/6750→`gfx1031`, 6600→`gfx1032`, 6800/6900→`gfx1030` (was all→`gfx1030`) |
| H  | Auto-detect `gpu_arch_platform` | `TraceLens/Reporting/reporting_utils.py` | TraceLens CLI | **DONE** — `resolve_gpu_arch()` now auto-detects via rocminfo/rocm-smi when `--gpu_arch_platform` is unset |
| I  | Wire platform auto-detect in Magpie | `src/Magpie-RDNA4/Magpie/.../tracelens.py` | kernel tuning | **DONE** — `tracelens.py` appends `--gpu_arch_platform` auto-detected from `rocminfo`/`rocm-smi` when no explicit config |
| J  | rocprofv3 tracer for Linux RDNA2 | `tools/run_benchmark.py` | profiling | **DONE** — added `find_rocprofv3()` + `convert_rocprof_csv_to_aicompass()` + rocprofv3 fallback path (hip_tracer is Windows-only) |
| K  | `--platform` help text | `orchestrator_prepare.py:793` | docs | **DONE** — now lists RX6700XT/RX9070XT
