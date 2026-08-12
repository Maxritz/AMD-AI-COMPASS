#!/usr/bin/env python3
"""
AI-COMPASS Analyze  kernel profiling + bottleneck detection + perf report

Usage:
    python tools/analyze.py trace.csv [--output report_dir] [--compare baseline.csv] [--arch rdna4]
"""
import csv
import os
import sys
import json
import math
import argparse
from collections import defaultdict
from datetime import datetime

# Architecture profiles
# Each profile: (cu_count, wave_size, simd_per_cu, waves_per_simd, lds_per_cu, l2_cache_kb)
ARCH_PROFILES = {
    "rdna1":  {"cu": 36, "wave": 64, "simd": 4, "waves_per_simd": 16, "lds": 65536, "l2": 4096},
    "rdna2":  {"cu": 40, "wave": 64, "simd": 4, "waves_per_simd": 16, "lds": 65536, "l2": 4096},
    "rdna3":  {"cu": 32, "wave": 32, "simd": 4, "waves_per_simd": 16, "lds": 131072, "l2": 6144},
    "rdna3_5":{"cu": 36, "wave": 32, "simd": 4, "waves_per_simd": 16, "lds": 131072, "l2": 8192},
    "rdna4":  {"cu": 32, "wave": 32, "simd": 4, "waves_per_simd": 16, "lds": 131072, "l2": 12288},
}

# RDNA2-specific kernel patterns (different block sizing vs RDNA4)
# Keys: (block_x, block_y, block_z, min_gx, max_gx, min_smem, max_smem) -> (category, name)
KERNEL_DB_RDNA2 = [
    # MMVQ K-quant: block_y=4 not 8 (RDNA2 prefers fewer threads due to Wave64).
    # smem no longer gates this -- see the identical note on KERNEL_DB_RDNA4's
    # entry; static shared memory isn't observable via this tracer on any
    # AMD arch, not just RDNA4.
    ((64, 4, 1, 0, 9999, 0, 99999), "MMVQ", "mmvq_kq"),
    # MMVQ: block 128 or 256 typical for Wave64
    ((256, 1, 1, 1, 100, 0, 100), "MMVQ", "mul_mat_vec_q"),
    ((128, 1, 1, 1, 100, 0, 100), "MMVQ", "mul_mat_vec_q"),
    # MMQ with shared mem
    ((64, 1, 1, 100, 99999, 128, 99999), "MMQ", "mul_mat_q_K"),
    ((128, 1, 1, 100, 99999, 128, 99999), "MMQ", "mul_mat_q_K"),
    # MMQ without shared mem
    ((64, 1, 1, 100, 99999, 0, 100), "MMQ", "mul_mat_q"),
    # Attention
    ((1024, 1, 1, 1, 99999, 0, 99999), "Attention", "attn_head"),
    # Flash attention (block may be 256 or 128 on RDNA2)
    ((128, 1, 1, 16, 9999, 0, 100), "Attention", "flash_attn"),
    ((256, 1, 1, 16, 9999, 0, 100), "Attention", "flash_attn"),
    # Soft max
    ((256, 1, 1, 1, 50, 0, 100), "Attention", "soft_max"),
    # Soft max batched
    ((256, 1, 1, 51, 9999, 0, 100), "Attention", "soft_max_batch"),
    # RMS norm
    ((256, 1, 1, 100, 99999, 100, 99999), "Norm", "rms_norm"),
    # MoE sync fallback — tiny grid (8 or fewer), block 256, no smem
    ((256, 1, 1, 1, 4, 0, 100), "MoE", "sync_fallback"),
    # MoE dispatch (per-expert, medium grid, block 128)
    # MoE routing/top-k (block 128, 3D grid only — expert×token dims)
    ((128, 1, 1, 1, 99999, 1, 99999), "MoE", "moe_routing"),
    # MoE gather/scatter (block 32x2, large grid)
    ((32, 2, 1, 1000, 9999999, 0, 100), "MoE", "moe_gather"),
    ((32, 2, 1, 1, 99999, 1, 99999), "MoE", "moe_scatter"),
    # MoE expert compress (block 32x4, small smem)
    ((32, 4, 1, 1, 9999, 0, 30000), "MoE", "moe_compress"),
    # RoPE
    ((256, 1, 1, 1, 99, 0, 100), "RoPE", "rope"),
    # Elementwise
    ((256, 1, 1, 1, 99, 0, 100), "Vector", "elementwise"),
    # Get rows
    ((32, 8, 1, 100, 99999, 0, 100), "Vector", "get_rows"),
    # Copy
    ((32, 8, 1, 10000, 9999999, 0, 100), "Vector", "cpy"),
    # Scale
    ((1, 256, 1, 1, 99, 0, 100), "Vector", "scale"),
    # Dequantize
    ((1, 256, 1, 100, 99999, 0, 100), "Quantize", "dequantize"),
    # Reshape
    ((32, 2, 1, 1, 100, 0, 100), "Vector", "reshape"),
    # BitNet/Ternary Q1_0/Q2_0 compute (block 32x1 no smem = quant kernel; 24x1/24x5 = ternary merge)
    ((32, 1, 1, 5000, 999999, 0, 100), "Quantize", "bitnet_compute"),
    ((24, 1, 1, 1, 99999, 0, 100), "Quantize", "bitnet_compute"),
    ((24, 5, 1, 1, 9999, 0, 100), "Quantize", "bitnet_merge"),
    ((64, 1, 1, 50000, 9999999, 0, 100), "Quantize", "bitnet_gather"),
    # Cross entropy
    ((512, 1, 1, 1, 100, 0, 100), "Other", "cross_entropy"),
]

KERNEL_DB_RDNA4 = [
    # MMVQ decode path (mul_mat_vec_q, ncols_dst=1): block=(32,8,1) is the
    # RDNA4-tuned nwarps=8 config (ggml-cuda/mmvq.cu calc_launch_params/
    # mmvq_shapes), grid.x = weight matrix's output-row count. The smem range
    # below used to require >50000 bytes on the theory that MMVQ's K-tile
    # buffer would show up as launch-time shared memory; on this HIP tracer
    # (hip_tracer.cpp, AI-COMPASS) shared_mem is only ever what's passed to
    # hipLaunchKernel's sharedMemBytes parameter (dynamic shared memory) --
    # mmvq's shared memory is statically declared inside the kernel, which
    # never shows up there and reads back as 0 regardless of real usage.
    # Block shape alone is already unambiguous for this pattern (confirmed
    # against source, not guessed) so the smem range no longer gates it.
    ((32, 8, 1, 0, 9999, 0, 99999), "MMVQ", "mmvq_kq"),
    ((32, 1, 1, 1, 100, 0, 100), "MMVQ", "mul_mat_vec_q"),
    ((128, 1, 1, 1, 100, 0, 100), "MMVQ", "mul_mat_vec_q"),
    # MMQ K-tile
    ((64, 1, 1, 100, 99999, 128, 99999), "MMQ", "mul_mat_q_K"),
    ((64, 1, 1, 100, 99999, 0, 100), "MMQ", "mul_mat_q"),
    # Attention
    ((1024, 1, 1, 1, 99999, 0, 99999), "Attention", "attn_head"),
    ((128, 1, 1, 16, 9999, 0, 100), "Attention", "flash_attn"),
    ((256, 1, 1, 1, 50, 0, 100), "Attention", "soft_max"),
    ((256, 1, 1, 51, 9999, 0, 100), "Attention", "soft_max_batch"),
    # RMS norm
    ((256, 1, 1, 100, 99999, 100, 99999), "Norm", "rms_norm"),
    # MoE sync fallback — tiny grid (8 or fewer), block 256, no smem
    ((256, 1, 1, 1, 4, 0, 100), "MoE", "sync_fallback"),
    # MoE dispatch (per-expert, medium grid, block 128)
    # MoE routing/top-k (block 128, 3D grid only — expert×token dims)
    ((128, 1, 1, 1, 99999, 1, 99999), "MoE", "moe_routing"),
    # MoE gather/scatter (block 32x2, large grid)
    ((32, 2, 1, 1000, 9999999, 0, 100), "MoE", "moe_gather"),
    ((32, 2, 1, 1, 99999, 1, 99999), "MoE", "moe_scatter"),
    # MoE expert compress (block 32x4, small smem)
    ((32, 4, 1, 1, 9999, 0, 30000), "MoE", "moe_compress"),
    # RoPE
    ((256, 1, 1, 1, 99, 0, 100), "RoPE", "rope"),
    # Elementwise
    ((256, 1, 1, 1, 99, 0, 100), "Vector", "elementwise"),
    # Get rows
    ((32, 8, 1, 100, 99999, 0, 100), "Vector", "get_rows"),
    # Copy
    ((32, 8, 1, 10000, 9999999, 0, 100), "Vector", "cpy"),
    # Scale
    ((1, 256, 1, 1, 99, 0, 100), "Vector", "scale"),
    # Dequantize
    ((1, 256, 1, 100, 99999, 0, 100), "Quantize", "dequantize"),
    # Reshape
    ((32, 2, 1, 1, 100, 0, 100), "Vector", "reshape"),
    # BitNet/Ternary Q1_0/Q2_0 compute (block 32x1 no smem = quant kernel; 24x1/24x5 = ternary merge)
    ((32, 1, 1, 5000, 999999, 0, 100), "Quantize", "bitnet_compute"),
    ((24, 1, 1, 1, 99999, 0, 100), "Quantize", "bitnet_compute"),
    ((24, 5, 1, 1, 9999, 0, 100), "Quantize", "bitnet_merge"),
    ((64, 1, 1, 50000, 9999999, 0, 100), "Quantize", "bitnet_gather"),
    # Cross entropy
    ((512, 1, 1, 1, 100, 0, 100), "Other", "cross_entropy"),
]

# Supported architectures for CLI
ARCH_NAMES = {"rdna1", "rdna2", "rdna3", "rdna3_5", "rdna4"}


def detect_arch_from_gfx(gfx_str):
    """Detect architecture from gfx string like 'gfx1031' or 'gfx1201'."""
    if gfx_str.startswith("gfx120") or gfx_str.startswith("gfx121"):
        return "rdna4"
    if gfx_str.startswith("gfx115"):
        return "rdna3_5"
    if gfx_str.startswith("gfx110"):
        return "rdna3"
    if gfx_str.startswith("gfx103"):
        return "rdna2"
    if gfx_str.startswith("gfx101"):
        return "rdna1"
    # Fallback: try to identify by CU count
    return "rdna4"


def get_kernel_db(arch):
    """Get the appropriate kernel classification DB for the architecture."""
    if arch == "rdna2" or arch == "rdna1":
        return KERNEL_DB_RDNA2
    return KERNEL_DB_RDNA4


def classify_kernel(name, arch="rdna4", gx=0, gy=0, gz=0, bx=0, by=0, bz=0, smem=0):
    db = get_kernel_db(arch)
    # If name is NOT a kptr_, use name-based classification
    if name and not name.startswith("kptr_"):
        nl = name.lower()
        if any(k in nl for k in ["mul_mat_vec_q", "mmvq"]): return "MMVQ"
        if any(k in nl for k in ["mul_mat", "mmq"]): return "MMQ"
        if any(k in nl for k in ["flash_attn", "attn", "soft_max"]): return "Attention"
        if any(k in nl for k in ["rms_norm", "norm"]): return "Norm"
        if any(k in nl for k in ["rope"]): return "RoPE"
        if any(k in nl for k in ["silu", "gelu", "relu"]): return "Activation"
        if any(k in nl for k in ["moe", "expert", "top_k", "routing", "gate"]): return "MoE"
        if any(k in nl for k in ["quantize", "dequantize"]): return "Quantize"
        if any(k in nl for k in ["get_rows", "add", "mul", "cpy", "scale"]): return "Vector"
    # Else use pattern matching
    for pattern, cat, _ in db:
        pbx, pby, pbz, mn, mx, mns, mxs = pattern
        if (bx == pbx and by == pby and bz == pbz and
            mn <= gx <= mx and mns <= smem <= mxs):
            return cat
    # MoE-specific fallback: small grid (<16) with block 256 = sync fallback
    if bx == 256 and by == 1 and bz == 1 and gx < 16:
        return "MoE"
    # MoE gather/scatter: block 32x2, any grid
    if bx == 32 and by == 2:
        if gy > 1 or gz > 1:
            return "MoE"
    # MoE compress: block 32x4 with medium smem
    if bx == 32 and by == 4 and 1000 < smem < 50000:
        return "MoE"
    # Fallback by block dims (arch-aware)
    if arch in ("rdna1", "rdna2"):
        if bx == 256 and by == 1:
            if gx > 100: return "Norm"
            return "Vector"
        if bx == 64 and by == 4:
            return "MMVQ"
    else:
        if bx == 256 and by == 1:
            if gx > 100: return "Norm"
            return "Vector"
        if bx == 32 and by == 8:
            # Same tracer limitation as the KERNEL_DB_RDNA4 table entry above --
            # smem is never observable for this kernel's static shared memory,
            # so it no longer gates the classification (this branch exists for
            # grid_x outside the primary table's range, e.g. lm_head's
            # vocab-sized launch).
            return "MMVQ"
        if bx == 32 and by == 4:
            if smem > 1000:
                return "MoE"
            return "Vector"
    if bx == 128:
        return "Attention"
    if bx == 1 and by == 256:
        return "Quantize"
    return "Other"


def compute_occupancy(record, arch="rdna4"):
    """Estimate occupancy from grid/block dimensions, architecture-aware."""
    prof = ARCH_PROFILES.get(arch, ARCH_PROFILES["rdna4"])
    wave_size = prof["wave"]
    total_threads = (record["grid_x"] * record["grid_y"] * record["grid_z"] *
                     record["block_x"] * record["block_y"] * record["block_z"])
    total_waves = (total_threads + wave_size - 1) // wave_size
    max_waves = prof["cu"] * prof["simd"] * prof["waves_per_simd"]
    return min(100.0, total_waves / max_waves * 100.0) if max_waves else 0


def estimate_arithmetic_intensity(kernel_name, grid, shared_mem, arch="rdna4"):
    """Rough estimate: flops/byte ratio based on kernel type and arch."""
    cat = classify_kernel(kernel_name)
    # RDNA2 has lower L2 cache (4 MB vs 12 MB), so memory-bound kernels
    # may have lower effective intensity
    ratios = {
        "MMQ": 6.0, "MMVQ": 1.5, "Attention": 3.0, "Norm": 0.8,
        "RoPE": 0.5, "Activation": 0.3, "MoE": 4.0, "GEMM": 8.0,
        "Vector": 0.2, "Quantize": 1.2, "Other": 1.0
    }
    # RDNA4 has larger L2 cache so memory reuse is better
    if arch == "rdna4" or arch == "rdna3_5":
        ratios["MMQ"] = 8.0
        ratios["MMVQ"] = 2.0
        ratios["Attention"] = 4.0
        ratios["MoE"] = 6.0
    return ratios.get(cat, 1.0)


def generate_optimization_targets(category_stats, bottlenecks, arch="rdna4", records=None):
    targets = []
    prof = ARCH_PROFILES.get(arch, ARCH_PROFILES["rdna4"])
    is_wave64 = prof["wave"] == 64

    if is_wave64:
        if "MMVQ" in category_stats and category_stats["MMVQ"]["pct"] > 5:
            targets.append({
                "target": "MMVQ",
                "current_pct": category_stats["MMVQ"]["pct"],
                "suggestion": (
                    "RDNA2 Wave64: MMVQ thread mapping requires block_y >= 4 for "
                    "full occupancy. Check mmvq.cu nwarps tuning  prefer nwarps=4 "
                    "(256 threads) for Wave64."
                ),
            })
        if "MMQ" in category_stats and category_stats["MMQ"]["pct"] > 15:
            targets.append({
                "target": "MMQ",
                "current_pct": category_stats["MMQ"]["pct"],
                "suggestion": (
                    "RDNA2 has 4 MB L2 cache (vs 12 MB RDNA4). MMQ may be L2-bound. "
                    "Consider smaller K-tiles or increasing MMQ_ITER_K to reduce tile reloads."
                ),
            })
        if "Attention" in category_stats and category_stats["Attention"]["pct"] > 15:
            targets.append({
                "target": "Attention",
                "current_pct": category_stats["Attention"]["pct"],
                "suggestion": (
                    "RDNA2 Wave64: flash attention may underutilize SIMDs. "
                    "Consider group-size tuning for Wave64 alignment."
                ),
            })
    else:
        if "MMQ" in category_stats and category_stats["MMQ"]["pct"] > 15:
            targets.append({
                "target": "MMQ",
                "current_pct": category_stats["MMQ"]["pct"],
                "suggestion": "MMQ dominates. Consider K-tile doubling. Check if MMQ_ITER_K is optimal for RDNA4.",
            })
        if "MMVQ" in category_stats and category_stats["MMVQ"]["pct"] > 10:
            targets.append({
                "target": "MMVQ",
                "current_pct": category_stats["MMVQ"]["pct"],
                "suggestion": "MMVQ significant. Check Split-K heuristic and small_k path.",
            })
        if "Attention" in category_stats and category_stats["Attention"]["pct"] > 15:
            targets.append({
                "target": "Attention",
                "current_pct": category_stats["Attention"]["pct"],
                "suggestion": (
                    "Attention is a significant fraction. Consider flash attention tuning. "
                    "For production ROCm deployments, utilize ROCm AITER (https://github.com/ROCm/aiter) "
                    "for optimized MHA/MLA decoding attention kernels."
                ),
            })
    if "MoE" in category_stats and category_stats["MoE"]["pct"] > 5:
        s = category_stats["MoE"]
        targets.append({
            "target": "MoE",
            "current_pct": s["pct"],
            "suggestion": (
                f"MoE dispatch overhead at {s['pct']}% with {s['avg_occupancy_pct']}% occupancy. "
                "The sync fallback kernel (grid<16, block=256, occ~3%) is the host-side expert sort "
                "in ggml_cuda_mul_mat_id. Fix options:\n"
                "  1. Increase MMVQ_MAX_BATCH_SIZE (currently 8) and per-type mmvq_mmid_max_batch "
                "(currently 4 for Q4_K on RDNA4) to keep more MoE dispatches on the fast path\n"
                "  2. Use async memcpy + batched per-expert dispatch to avoid stream sync\n"
                "  3. Implement GPU-side expert sort (thrust::sort_by_key) to eliminate CPU bounce\n"
                "  4. Integrate ROCm AITER (https://github.com/ROCm/aiter) MoE kernels for fast fused MoE tiling\n"
                "  5. Profile with: --arch {arch} and check top slowest kernels for sync_fallback"
            ),
        })

    # Memory/KV Capacity warnings linking to rocm-aic
    if "ram" in category_stats or any("occupancy-bound" in b or "low occupancy" in b for b in bottlenecks):
        targets.append({
            "target": "Memory & KV Capacity",
            "current_pct": 0,
            "suggestion": (
                "Workload may be memory/KV cache capacity bound. Evaluate ROCm AIC (AMD Infinity Context: "
                "https://github.com/ROCm/rocm-aic) for disaggregated, low-latency KV cache storage tiering."
            )
        })

    # Memory Bandwidth & Host-to-Device Transfer Analysis
    if "Vector" in category_stats and category_stats["Vector"]["pct"] > 15:
        targets.append({
            "target": "Memory Bandwidth & Host-to-Device Transfers",
            "current_pct": category_stats["Vector"]["pct"],
            "suggestion": (
                "Host-to-device memory copies or Vector/getRow operations represent a high fraction of execution time. "
                "Consider using pinned (page-locked) host memory, pre-allocating GPU buffers, and using hipMemcpyAsync to "
                "overlap memory transfers with compute streams."
            )
        })

    # Cache hit rate & GEMM / MMQ optimization suggestion
    if arch in ("rdna1", "rdna2", "rdna3") and "MMQ" in category_stats and category_stats["MMQ"]["pct"] > 15:
        targets.append({
            "target": "L2 Cache Hit Rate",
            "current_pct": category_stats["MMQ"]["pct"],
            "suggestion": (
                f"MMQ (GEMM) represents a high fraction of execution time on {arch.upper()}. "
                "Since this GPU has a smaller L2 cache than RDNA4 (e.g. 4MB on RDNA2 vs 12MB on RDNA4), "
                "matrix multiplication may be L2 cache-thrashing. Adjust matrix tile sizing (reduce K-tiles or "
                "increase MMQ_ITER_K) to maximize L2 residency and avoid VRAM reload bandwidth bottlenecks."
            )
        })

    # Vulkan Pipeline Caching
    has_vk_compile = False
    if records:
        for r in records:
            name = r.get("kernel_name", "").lower()
            if "vkcreate" in name or "vkpipeline" in name or "vkshader" in name:
                has_vk_compile = True
                break
    if has_vk_compile:
        targets.append({
            "target": "Vulkan Pipeline Compilation",
            "current_pct": 0,
            "suggestion": (
                "Vulkan pipeline/shader creation detected during runtime. Use VK_KHR_pipeline_binary "
                "or pipeline caches to pre-compile and serialize shader binaries, eliminating runtime shader compilation stutters."
            )
        })

    # Command Submission Overhead (ExecuteIndirect / Indirect Dispatching)
    avg_us = 0
    if records:
        total_t = sum(r["duration_us"] for r in records)
        avg_us = total_t / len(records)
    if records and len(records) > 500 and avg_us < 15.0:
        targets.append({
            "target": "Command Submission Overhead",
            "current_pct": 0,
            "suggestion": (
                "High frequency of extremely small kernel dispatches detected. To reduce CPU-side command submission "
                "and driver overhead, batch dispatches using ExecuteIndirect (Direct3D 12) or indirect drawing/dispatching (Vulkan)."
            )
        })

    # Check for FP4/sub-byte quantization in the trace
    has_fp4 = False
    if records:
        for r in records:
            name = r.get("kernel_name", "").lower()
            if "fp4" in name or "mxfp4" in name or "nvfp4" in name or "rocmfp4" in name:
                has_fp4 = True
                break

    if has_fp4 or ("Quantize" in category_stats and category_stats["Quantize"]["pct"] > 5):
        targets.append({
            "target": "FP4 Quantization",
            "current_pct": category_stats.get("Quantize", {}).get("pct", 0),
            "suggestion": (
                "FP4 / Sub-byte Quantization detected. On RDNA3 and RDNA4 GPUs lacking native hardware FP4 instructions, "
                "the best and fastest way to implement FP4 is via Just-in-Time (JIT) Register-Level Dequantization. "
                "Do not store dequantized values in VRAM. Instead, implement a custom HIP/CUDA kernel that:\n"
                "  1. Loads packed FP4 weights into registers (2 elements per byte)\n"
                "  2. Dequantizes them to FP16 using register bit-shifts/masks (extract sign, shift exponent/mantissa, adjust exponent bias +14)\n"
                "  3. Amortizes memory overhead by performing block-scale multiplication (Marlin-style) on FP16 vectors\n"
                "  4. Executes the computation using native FP16 WMMA matrix instructions."
            )
        })

    return targets


def csv_has_measured_duration(csv_path):
    """True if the trace CSV carries at least some real per-dispatch GPU
    duration: either the external dispatch_id/grid_x/duration_us schema, or
    the AI-COMPASS hip_tracer.cpp native schema with HIP_TRACER_GPU_TIMING
    rows present ("hipKernelTiming"). Only a coarse yes/no for a console
    pre-check -- analyze_trace() computes the real per-record/coverage
    breakdown from the parsed records themselves, which is authoritative;
    this is just used to decide whether to print the upfront caveat."""
    try:
        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader, [])
            if "duration_us" in header and "grid_x" in header:
                return True
            if "api" in header:
                api_idx = header.index("api")
                return any(len(row) > api_idx and row[api_idx] == "hipKernelTiming" for row in reader)
        return False
    except OSError:
        return False


def analyze_trace(records, arch="rdna4", duration_is_measured=True):
    """Full analysis of a trace, returns structured dict."""
    if not records:
        return {"error": "No records"}

    prof = ARCH_PROFILES.get(arch, ARCH_PROFILES["rdna4"])
    total_kernels = len(records)
    total_time_us = sum(r["duration_us"] for r in records)
    wall_time_ms = total_time_us / 1000

    by_category = defaultdict(list)
    for r in records:
        cat = classify_kernel(r["kernel_name"],
            arch=arch,
            gx=int(r.get("grid_x", 0)), gy=int(r.get("grid_y", 0)),
            gz=int(r.get("grid_z", 0)), bx=int(r.get("block_x", 0)),
            by=int(r.get("block_y", 0)), bz=int(r.get("block_z", 0)),
            smem=int(r.get("shared_mem", 0)))
        by_category[cat].append(r)

    category_stats = {}
    for cat, recs in sorted(by_category.items(), key=lambda x: -sum(r["duration_us"] for r in x[1])):
        total_cat_us = sum(r["duration_us"] for r in recs)
        count = len(recs)
        avg_us = total_cat_us / count if count else 0
        max_us = max(r["duration_us"] for r in recs)
        occs = [compute_occupancy(r, arch) for r in recs]
        avg_occ = sum(occs) / len(occs) if occs else 0
        category_stats[cat] = {
            "count": count,
            "total_ms": round(total_cat_us / 1000, 2),
            "pct": round(total_cat_us / total_time_us * 100, 1),
            "avg_us": round(avg_us, 1),
            "max_us": round(max_us, 1),
            "avg_occupancy_pct": round(avg_occ, 1),
        }

    pp_cutoff = max(1, total_kernels // 10)
    pp_kernels = records[:pp_cutoff*2]

    def phase_summary(kernels, label):
        t = sum(k["duration_us"] for k in kernels)
        cnt = len(kernels)
        return {
            "phase": label,
            "kernel_count": cnt,
            "total_ms": round(t / 1000, 2),
            "pct_of_total": round(t / total_time_us * 100, 1) if total_time_us else 0,
            "avg_kernel_us": round(t / cnt, 1) if cnt else 0,
        }

    phases = []
    phases.append(phase_summary(pp_kernels, "Prompt Processing"))
    tg_kernels = records[pp_cutoff*2:]
    phases.append(phase_summary(tg_kernels, "Token Generation"))

    sorted_by_dur = sorted(records, key=lambda r: -r["duration_us"])[:10]
    top_slow = []
    for r in sorted_by_dur:
        top_slow.append({
            "kernel": r["kernel_name"],
            "duration_ms": round(r["duration_us"] / 1000, 3),
            "grid": f"{r['grid_x']}x{r['grid_y']}x{r['grid_z']}",
            "block": f"{r['block_x']}x{r['block_y']}x{r['block_z']}",
            "category": classify_kernel(r["kernel_name"], arch=arch),
            "occupancy_pct": round(compute_occupancy(r, arch), 1),
        })

    pp_time_ms = sum(k["duration_us"] for k in pp_kernels) / 1000
    tg_time_ms = sum(k["duration_us"] for k in tg_kernels) / 1000

    bottlenecks = []
    for cat, stats in category_stats.items():
        if stats["pct"] > 25:
            bottlenecks.append(f"{cat} dominates at {stats['pct']}% of total time")
        if stats["avg_occupancy_pct"] < 30 and stats["pct"] > 5:
            occ_msg = f"likely occupancy-bound on {arch.upper()} ({prof['wave']}-wave)"
            if cat == "MoE" and stats["avg_occupancy_pct"] < 10:
                occ_msg += " — this is the sync fallback (CPU sort, host<->device copies)"
            elif cat == "MoE":
                occ_msg += " — MoE dispatch overhead, check mmvq_mmid_max_batch limits"
            bottlenecks.append(f"{cat} has low occupancy ({stats['avg_occupancy_pct']}%)  {occ_msg}")

    ideal_time_us = sum(r["duration_us"] for r in records if r["block_x"] > 64)
    gpu_busy_pct = min(100, ideal_time_us / total_time_us * 100) if total_time_us else 0

    if any("is_measured" in r for r in records):
        measured_n = sum(1 for r in records if r.get("is_measured"))
        total_n = len(records)
        pct = round(100.0 * measured_n / total_n, 1) if total_n else 0.0
        if measured_n == total_n:
            duration_semantics = "measured_gpu_duration (real hipEvent timing, 100% coverage)"
        elif measured_n == 0:
            duration_semantics = (
                "estimated_dispatch_interval (host-side time to next launch; no launches "
                "in this trace got real hipEvent timing -- see console GPU timing coverage line)"
            )
        else:
            duration_semantics = (
                f"mixed: {pct}% of records have measured_gpu_duration (real hipEvent timing), "
                "the rest use estimated_dispatch_interval -- see per-record is_measured"
            )
    else:
        duration_semantics = "measured_gpu_duration" if duration_is_measured else (
            "estimated_dispatch_interval (host-side time to next launch; hip_tracer.cpp "
            "does not bracket async kernel launches with GPU timers, so this is NOT "
            "measured kernel execution time)"
        )

    return {
        "trace_file": "",
        "analysis_time": datetime.now().isoformat(),
        "arch": arch,
        "duration_semantics": duration_semantics,
        "architecture_profile": prof,
        "summary": {
            "total_kernels": total_kernels,
            "total_time_ms": round(wall_time_ms, 2),
            "avg_kernel_us": round(total_time_us / total_kernels, 1) if total_kernels else 0,
            "estimated_gpu_busy_pct": round(gpu_busy_pct, 1),
            "estimated_gpu_utilization_pct": round(gpu_busy_pct * 0.85, 1),
        },
        "phases": phases,
        "category_breakdown": category_stats,
        "top_slowest_kernels": top_slow,
        "bottlenecks": bottlenecks,
        "optimization_targets": generate_optimization_targets(category_stats, bottlenecks, arch, records),
    }


def generate_html_report(analysis, output_path):
    """Generate an HTML report from analysis data."""
    s = analysis["summary"]
    phases = analysis["phases"]
    cats = analysis["category_breakdown"]
    tops = analysis["top_slowest_kernels"]
    bots = analysis["bottlenecks"]
    targets = analysis["optimization_targets"]
    arch = analysis.get("arch", "rdna4")
    prof = analysis.get("architecture_profile", {})

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>AI-COMPASS Performance Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif; margin: 2em; background: #0d1117; color: #c9d1d9; }}
h1 {{ color: #58a6ff; }} h2 {{ color: #79c0ff; border-bottom: 1px solid #30363d; }}
table {{ border-collapse: collapse; width: 100%; margin: 1em 0; }}
th, td {{ padding: 8px 12px; text-align: left; border: 1px solid #30363d; }}
th {{ background: #161b22; }} tr:nth-child(even) {{ background: #161b22; }}
.card {{ background: #161b22; border: 1px solid #30363d; border-radius: 6px; padding: 16px; margin: 1em 0; }}
.bottleneck {{ color: #f85149; }} .ok {{ color: #3fb950; }} .warn {{ color: #d29922; }}
.meter {{ height: 20px; background: #21262d; border-radius: 10px; overflow: hidden; margin: 4px 0; }}
.meter-bar {{ height: 100%; background: #58a6ff; border-radius: 10px; }}
</style></head><body>
<h1>[AI] [AI-COMPASS] Performance Report</h1>
<p>Generated: {analysis["analysis_time"]}</p>
<p class="warn">Timing basis: {analysis.get("duration_semantics", "measured_gpu_duration")}</p>
<div class="card">
<h2>System</h2>
<table>
<tr><td>Architecture</td><td>{arch.upper()}</td></tr>
<tr><td>CUs</td><td>{prof.get("cu", "?")}</td></tr>
<tr><td>Wave Size</td><td>{prof.get("wave", "?")}</td></tr>
<tr><td>L2 Cache</td><td>{prof.get("l2", "?")} KB</td></tr>
</table>
</div>
<div class="card">
<h2>Summary</h2>
<table>
<tr><td>Total Kernels</td><td>{s["total_kernels"]}</td></tr>
<tr><td>Total GPU Time</td><td>{s["total_time_ms"]} ms</td></tr>
<tr><td>Avg Kernel</td><td>{s["avg_kernel_us"]} s</td></tr>
<tr><td>Est. GPU Busy</td><td>{s["estimated_gpu_busy_pct"]}%</td></tr>
<tr><td>Est. GPU Utilization</td><td>{s["estimated_gpu_utilization_pct"]}%</td></tr>
</table>
</div>
<div class="card">
<h2>Phases</h2>
<table><tr><th>Phase</th><th>Kernels</th><th>Time (ms)</th><th>% Total</th><th>Avg Kernel (s)</th></tr>
"""
    for p in phases:
        html += f"<tr><td>{p['phase']}</td><td>{p['kernel_count']}</td><td>{p['total_ms']}</td><td>{p['pct_of_total']}%</td><td>{p['avg_kernel_us']}</td></tr>"

    html += """</table></div><div class="card"><h2>Category Breakdown</h2><table><tr><th>Category</th><th>Count</th><th>Total (ms)</th><th>%</th><th>Avg (s)</th><th>Max (s)</th><th>Occupancy</th></tr>"""
    for cat, st in sorted(cats.items(), key=lambda x: -x[1]["pct"]):
        html += f"<tr><td>{cat}</td><td>{st['count']}</td><td>{st['total_ms']}</td><td>{st['pct']}%</td><td>{st['avg_us']}</td><td>{st['max_us']}</td><td>{st['avg_occupancy_pct']}%</td></tr>"

    html += """</table></div>"""

    if bots:
        html += """<div class="card"><h2>[BELL] Bottlenecks</h2><ul>"""
        for b in bots:
            html += f'<li class="bottleneck">{b}</li>'
        html += "</ul></div>"

    if targets:
        html += """<div class="card"><h2> Optimization Targets</h2>"""
        for t in targets:
            html += f"<p><b>{t['target']}</b> ({t['current_pct']}% of time): {t['suggestion']}</p>"
        html += "</div>"

    html += """<div class="card"><h2>Top-10 Slowest Kernels</h2><table><tr><th>Kernel</th><th>Category</th><th>Duration (ms)</th><th>Grid</th><th>Block</th><th>Occupancy</th></tr>"""
    for k in tops:
        html += f"<tr><td>{k['kernel']}</td><td>{k['category']}</td><td>{k['duration_ms']}</td><td>{k['grid']}</td><td>{k['block']}</td><td>{k['occupancy_pct']}%</td></tr>"
    html += """</table></div></body></html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


def compare_traces(baseline_path, target_path, arch="rdna4"):
    """Compare two traces (before/after optimization)."""
    base_recs = parse_trace(baseline_path)
    tgt_recs = parse_trace(target_path)
    base = analyze_trace(base_recs, arch, duration_is_measured=csv_has_measured_duration(baseline_path))
    tgt = analyze_trace(tgt_recs, arch, duration_is_measured=csv_has_measured_duration(target_path))

    diff = {
        "baseline": baseline_path,
        "target": target_path,
        "analysis_time": datetime.now().isoformat(),
        "improvements": {},
        "regressions": {},
    }

    bs = base["summary"]
    ts = tgt["summary"]
    for key in ["total_time_ms", "avg_kernel_us", "estimated_gpu_utilization_pct"]:
        bv = bs[key]
        tv = ts[key]
        if bv:
            chg = (tv - bv) / bv * 100
        else:
            chg = 0
        label = {"total_time_ms": "Total Time", "avg_kernel_us": "Avg Kernel Time",
                 "estimated_gpu_utilization_pct": "GPU Utilization"}[key]
        if chg < -5:
            diff["improvements"][label] = f"{chg:+.1f}%"
        elif chg > 5:
            diff["regressions"][label] = f"{chg:+.1f}%"

    return diff


def parse_trace(csv_path):
    """Parse a HIP trace CSV.

    Supports two schemas:
      - A schema with real per-dispatch duration_us + grid_x..block_z columns
        already present (e.g. a GPU-timestamped tracer) -- used as-is.
      - The AI-COMPASS hip_tracer.cpp native format: seq,us,api,kernel_name,
        grid,block,stream,result. Launch rows (hipModuleLaunchKernel/
        hipLaunchKernel) become dispatch records; "grid"/"block" ("x,y,z"
        strings) are split into grid_x/y/z, block_x/y/z.

        hip_tracer.cpp optionally (HIP_TRACER_GPU_TIMING=1) brackets each
        launch with real hipEvent timing and emits a "hipKernelTiming" row
        with the true elapsed GPU microseconds once it completes. When those
        rows are present, this function correlates the Nth hipKernelTiming
        row for a stream to the Nth launch row for that same stream
        (hip_tracer.cpp drains them strictly in submission order) and uses
        the real value. Known caveat: HIP graph-capturing targets (e.g.
        ggml/llama.cpp's HIP backend post-warmup) skip timing during capture
        and any launches after the fixed event pool is exhausted, so
        coverage can be partial -- those specific records fall back to the
        estimated dispatch interval (elapsed host-side microseconds until the
        next recorded launch on that stream) same as when GPU_TIMING was
        never enabled at all. is_measured is set per record so callers can
        tell which is which; parse_trace_coverage() reports the overall split.
    """
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return []

    if "duration_us" in rows[0] and "grid_x" in rows[0]:
        records = []
        for row in rows:
            records.append({
                "dispatch_id": int(row.get("dispatch_id", 0)),
                "kernel_name": row.get("kernel_name", "unknown"),
                "grid_x": int(row.get("grid_x", 0)),
                "grid_y": int(row.get("grid_y", 0)),
                "grid_z": int(row.get("grid_z", 0)),
                "block_x": int(row.get("block_x", 0)),
                "block_y": int(row.get("block_y", 0)),
                "block_z": int(row.get("block_z", 0)),
                "shared_mem": int(row.get("shared_mem", 0)),
                "duration_us": float(row.get("duration_us", 0)),
            })
        return records

    launch_apis = {"hipModuleLaunchKernel", "hipLaunchKernel"}
    launches = [r for r in rows if r.get("api") in launch_apis]

    # Real timing rows, grouped per stream in file order -- hip_tracer.cpp
    # drains its in-flight queue strictly FIFO, so the Nth timing row for a
    # stream corresponds to the Nth launch on that stream that actually got
    # an event pair (see acquire_timing_pair/begin_launch_timing).
    timing_by_stream = defaultdict(list)
    for r in rows:
        if r.get("api") == "hipKernelTiming":
            name = r.get("kernel_name") or ""
            us_val = 0
            if name.startswith("dt_us="):
                try:
                    us_val = int(name[len("dt_us="):])
                except ValueError:
                    us_val = 0
            timing_by_stream[r.get("stream")].append(us_val)
    timing_idx_by_stream = defaultdict(int)  # next unconsumed index per stream

    records = []
    measured_count = 0
    for i, row in enumerate(launches):
        gx, gy, gz = ((row.get("grid") or "0,0,0").split(",") + ["0", "0", "0"])[:3]
        bx, by, bz = ((row.get("block") or "0,0,0").split(",") + ["0", "0", "0"])[:3]
        us = int(row.get("us", 0) or 0)

        stream = row.get("stream")
        tq = timing_by_stream.get(stream)
        idx = timing_idx_by_stream[stream]
        if tq is not None and idx < len(tq):
            dt = tq[idx]
            timing_idx_by_stream[stream] += 1
            measured = True
            measured_count += 1
        else:
            if i + 1 < len(launches):
                next_us = int(launches[i + 1].get("us", us) or us)
                dt = max(0, next_us - us)
            else:
                dt = 0  # last dispatch: nothing follows it in the trace, don't guess
            measured = False

        records.append({
            "dispatch_id": int(row.get("seq", i) or i),
            "kernel_name": row.get("kernel_name", "unknown"),
            "grid_x": int(gx or 0), "grid_y": int(gy or 0), "grid_z": int(gz or 0),
            "block_x": int(bx or 0), "block_y": int(by or 0), "block_z": int(bz or 0),
            "shared_mem": int(row.get("shared_mem", 0) or 0),  # 0 for traces from before this column existed
            "duration_us": float(dt),
            "is_measured": measured,  # True = real hipEvent GPU time, False = estimated interval
        })

    if launches:
        pct = round(100.0 * measured_count / len(launches), 1)
        print(f"[AI] GPU timing coverage: {measured_count}/{len(launches)} launches "
              f"({pct}%) have real hipEvent-measured duration; the rest use the "
              f"estimated dispatch interval.")
    return records


def main():
    parser = argparse.ArgumentParser(description="AI-COMPASS Analyze  kernel profiling & bottleneck detection")
    parser.add_argument("trace", nargs="?", default=None, help="HIP trace CSV file")
    parser.add_argument("-o", "--output", default="analysis_output", help="Output directory")
    parser.add_argument("--compare", help="Baseline CSV for before/after comparison")
    parser.add_argument("--arch", default="rdna4", choices=sorted(ARCH_NAMES),
                        help="GPU architecture (default: rdna4)")
    parser.add_argument("--cu-count", type=int, default=0,
                        help="Override CU count (default: from architecture profile)")
    parser.add_argument("--html", action="store_true", default=True, help="Generate HTML report")
    parser.add_argument("--bench-pp", type=float, default=None,
                        help="Prompt processing throughput (t/s) from llama-bench")
    parser.add_argument("--bench-tg", type=float, default=None,
                        help="Token generation throughput (t/s) from llama-bench")
    parser.add_argument("--memory", action="store_true",
                        help="Publish report to TencentDB Agent Memory hub")
    args = parser.parse_args()

    prof = ARCH_PROFILES.get(args.arch, ARCH_PROFILES["rdna4"])
    cu_count = args.cu_count if args.cu_count > 0 else prof["cu"]
    override_arch = args.arch
    # If CU count was overridden, adjust arch inference
    if args.cu_count > 0 and args.cu_count == 40:
        override_arch = "rdna2"

    if args.trace and os.path.exists(args.trace):
        os.makedirs(args.output, exist_ok=True)

        records = parse_trace(args.trace)
        if not records:
            print(f"[FAIL] No records found in {args.trace}")
            return 1

        measured = csv_has_measured_duration(args.trace)
        print(f"[AI] Analyzing {len(records)} kernel records from {args.trace}...")
        print(f"[AI] Architecture: {override_arch.upper()} ({prof['wave']}-wave, {cu_count} CUs)")
        if not measured:
            print("[AI] NOTE: this trace has no GPU-measured kernel duration (AI-COMPASS "
                  "hip_tracer.cpp logs async launches, not timed ones). Timing figures below "
                  "are estimated dispatch intervals, not hardware kernel time.")

        analysis = analyze_trace(records, override_arch, duration_is_measured=measured)
        analysis["trace_file"] = args.trace

        s = analysis["summary"]
        print(f"\n[DATA] Summary:")
        print(f"   {s['total_kernels']} kernels | {s['total_time_ms']} ms total")
        print(f"   Avg kernel: {s['avg_kernel_us']} s | Est. GPU busy: {s['estimated_gpu_busy_pct']}%")

        print(f"\n Phases:")
        for p in analysis["phases"]:
            print(f"   {p['phase']}: {p['kernel_count']} kernels, {p['total_ms']} ms ({p['pct_of_total']}%)")

        print(f"\n  Category Breakdown:")
        for cat, st in sorted(analysis["category_breakdown"].items(), key=lambda x: -x[1]["pct"]):
            print(f"   {cat:15s} {st['count']:5d} kernels  {st['total_ms']:8.1f} ms  {st['pct']:5.1f}%  occ:{st['avg_occupancy_pct']:5.1f}%")

        if analysis["bottlenecks"]:
            print(f"\n[BELL] Bottlenecks:")
            for b in analysis["bottlenecks"]:
                print(f"   [WARN] {b}")

        if analysis["optimization_targets"]:
            print(f"\n Optimization Targets:")
            for t in analysis["optimization_targets"]:
                print(f"   [{t['target']}] ({t['current_pct']}%): {t['suggestion']}")

        json_path = os.path.join(args.output, "analysis.json")
        with open(json_path, "w") as f:
            json.dump(analysis, f, indent=2)
        print(f"\n JSON report: {json_path}")

        if args.memory:
            try:
                from memory_hub import publish_report_json
                session_id = f"report:{args.output}"
                result = publish_report_json(json_path, session_id=session_id)
                print(f" Published to memory hub (accepted_ids={result.get('accepted_ids', [])})")
            except (ImportError, OSError) as exc:
                print(f" Memory publish skipped (gateway unavailable): {exc}")

        if args.html:
            html_path = os.path.join(args.output, "report.html")
            generate_html_report(analysis, html_path)
            print(f"[FILE] HTML report: {html_path}")

        if args.compare:
            print(f"\n[SYNC] Comparing against baseline: {args.compare}...")
            diff = compare_traces(args.compare, args.trace, override_arch)
            diff_path = os.path.join(args.output, "comparison.json")
            with open(diff_path, "w") as f:
                json.dump(diff, f, indent=2)
            if diff["improvements"]:
                print(f"\n[OK] Improvements:")
                for k, v in diff["improvements"].items():
                    print(f"   {k}: {v}")
            if diff["regressions"]:
                print(f"\n[FAIL] Regressions:")
                for k, v in diff["regressions"].items():
                    print(f"   {k}: {v}")
            print(f" Comparison: {diff_path}")

    else:
        # No trace file: show architecture profile only
        print(f"\n[AI] AI-COMPASS Architecture Profile: {override_arch.upper()}")
        print(f"   Architecture: {override_arch}")
        for k, v in prof.items():
            print(f"     {k}: {v}")
        print()

    # Show benchmark results if provided
    if args.bench_pp is not None or args.bench_tg is not None:
        print(f"\n[DATA] llama-bench Results:")
        if args.bench_pp:
            print(f"   Prompt Processing: {args.bench_pp:.1f} t/s")
        if args.bench_tg:
            print(f"   Token Generation:  {args.bench_tg:.1f} t/s")

        # Estimate memory bandwidth utilization for RDNA2 (6700 XT: ~384 GB/s)
        if args.bench_tg and override_arch == "rdna2":
            # RDNA2 memory bandwidth estimate for model size
            bw_est = args.bench_tg * 17.4  # rough: t/s * model_size_gb
            print(f"   Est. Mem BW utilized: {bw_est:.0f} GB/s (of ~384 GB/s GDDR6)")
            pct = bw_est / 384.0 * 100.0
            print(f"   Mem BW utilization: {pct:.0f}%  model likely bandwidth-bound on RDNA2")
        elif args.bench_tg and override_arch == "rdna4":
            bw_est = args.bench_tg * 17.4
            print(f"   Est. Mem BW utilized: {bw_est:.0f} GB/s (of ~960 GB/s GDDR6)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
