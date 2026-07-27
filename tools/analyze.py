#!/usr/bin/env python3
"""
AI-COMPASS Analyze — kernel profiling + bottleneck detection + perf report

Usage:
    python tools/analyze.py trace.csv [--output report_dir] [--compare baseline.csv]
"""
import csv
import os
import sys
import json
import math
import argparse
from collections import defaultdict
from datetime import datetime

# ── Kernel classification ──────────────────────────────────────────────
MMQ_KERNELS = {"mul_mat_vec", "quantize_mul_mat", "dequantize_mul_mat_vec"}
MMVQ_KERNELS = {"mul_mat_vec_q"}
ATTN_KERNELS = {"flash_attn", "attn", "soft_max", "attn_vec"}
NORM_KERNELS = {"rms_norm", "norm", "layer_norm"}
ROPE_KERNELS = {"rope", "rope_neox"}
ACT_KERNELS = {"silu", "gelu", "relu", "sigmoid", "hard_swish"}
CONV_KERNELS = {"im2col", "conv"}
MOE_KERNELS = {"moe", "expert", "ffn_gate", "top_k", "soft_max_expert"}
QUANT_KERNELS = {"quantize", "dequantize", "quant"}
SGEMV_KERNELS = {"sgemm", "gemm", "mat_mul"}
VEC_KERNELS = {"vec", "add", "mul", "cpy", "dup", "get_rows", "scale"}

def classify_kernel(name):
    nl = name.lower()
    if any(k in nl for k in MMQ_KERNELS): return "MMQ"
    if any(k in nl for k in MMVQ_KERNELS): return "MMVQ"
    if any(k in nl for k in ATTN_KERNELS): return "Attention"
    if any(k in nl for k in NORM_KERNELS): return "Norm"
    if any(k in nl for k in ROPE_KERNELS): return "RoPE"
    if any(k in nl for k in ACT_KERNELS): return "Activation"
    if any(k in nl for k in CONV_KERNELS): return "Conv"
    if any(k in nl for k in MOE_KERNELS): return "MoE"
    if any(k in nl for k in QUANT_KERNELS): return "Quantize"
    if any(k in nl for k in SGEMV_KERNELS): return "GEMM"
    if any(k in nl for k in VEC_KERNELS): return "Vector"
    return "Other"


def detect_phase(wall_time_ms, total_tokens):
    """Heuristic: 1st ~10% of tokens = prompt processing, rest = TG."""
    tt = total_tokens if total_tokens else 256
    pp_tokens = max(1, int(tt * 0.10))
    return pp_tokens, tt - pp_tokens


def parse_trace(csv_path):
    records = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
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


def compute_occupancy(record, cu_count=32, simd_per_cu=4, waves_per_simd=16):
    """Estimate occupancy from grid/block dimensions."""
    total_waves = (record["grid_x"] * record["grid_y"] * record["grid_z"] *
                   record["block_x"] * record["block_y"] * record["block_z"]) // 64
    max_waves = cu_count * simd_per_cu * waves_per_simd
    return min(100.0, total_waves / max_waves * 100.0) if max_waves else 0


def estimate_arithmetic_intensity(kernel_name, grid, shared_mem):
    """Rough estimate: flops/byte ratio based on kernel type."""
    cat = classify_kernel(kernel_name)
    # These are heuristic estimates for RDNA4
    ratios = {
        "MMQ": 8.0, "MMVQ": 2.0, "Attention": 4.0, "Norm": 1.0,
        "RoPE": 0.5, "Activation": 0.3, "MoE": 6.0, "GEMM": 10.0,
        "Vector": 0.2, "Quantize": 1.5, "Other": 1.0
    }
    return ratios.get(cat, 1.0)


def analyze_trace(records, cu_count=32):
    """Full analysis of a trace, returns structured dict."""
    if not records:
        return {"error": "No records"}

    total_kernels = len(records)
    total_time_us = sum(r["duration_us"] for r in records)
    wall_time_ms = total_time_us / 1000

    # Classify kernels
    by_category = defaultdict(list)
    for r in records:
        cat = classify_kernel(r["kernel_name"])
        by_category[cat].append(r)

    # Per-category stats
    category_stats = {}
    for cat, recs in sorted(by_category.items(), key=lambda x: -sum(r["duration_us"] for r in x[1])):
        total_cat_us = sum(r["duration_us"] for r in recs)
        count = len(recs)
        avg_us = total_cat_us / count if count else 0
        max_us = max(r["duration_us"] for r in recs)
        occs = [compute_occupancy(r, cu_count) for r in recs]
        avg_occ = sum(occs) / len(occs) if occs else 0
        category_stats[cat] = {
            "count": count,
            "total_ms": round(total_cat_us / 1000, 2),
            "pct": round(total_cat_us / total_time_us * 100, 1),
            "avg_us": round(avg_us, 1),
            "max_us": round(max_us, 1),
            "avg_occupancy_pct": round(avg_occ, 1),
        }

    # Phase detection (heuristic: split into PP and TG)
    # PP usually has higher occupancy and larger grids
    pp_cutoff = max(1, total_kernels // 10)  # first ~10% of dispatches
    pp_kernels = records[:pp_cutoff*2]  # approximate

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

    # Top-10 slowest kernels
    sorted_by_dur = sorted(records, key=lambda r: -r["duration_us"])[:10]
    top_slow = []
    for r in sorted_by_dur:
        top_slow.append({
            "kernel": r["kernel_name"],
            "duration_ms": round(r["duration_us"] / 1000, 3),
            "grid": f"{r['grid_x']}x{r['grid_y']}x{r['grid_z']}",
            "block": f"{r['block_x']}x{r['block_y']}x{r['block_z']}",
            "category": classify_kernel(r["kernel_name"]),
            "occupancy_pct": round(compute_occupancy(r, cu_count), 1),
        })

    # Throughput estimates
    # PP: prompt processing throughput (tokens/s) — rough estimate from kernel count
    pp_time_ms = sum(k["duration_us"] for k in pp_kernels) / 1000
    tg_time_ms = sum(k["duration_us"] for k in tg_kernels) / 1000

    # Bottleneck detection
    bottlenecks = []
    for cat, stats in category_stats.items():
        if stats["pct"] > 25:
            bottlenecks.append(f"{cat} dominates at {stats['pct']}% of total time")
        if stats["avg_occupancy_pct"] < 30 and stats["pct"] > 5:
            bottlenecks.append(f"{cat} has low occupancy ({stats['avg_occupancy_pct']}%) — likely occupancy-bound")

    # Check GPU utilization (from ADLX metrics if available, else estimate)
    ideal_time_us = sum(r["duration_us"] for r in records if r["block_x"] > 64)
    gpu_busy_pct = min(100, ideal_time_us / total_time_us * 100) if total_time_us else 0

    return {
        "trace_file": "",
        "analysis_time": datetime.now().isoformat(),
        "summary": {
            "total_kernels": total_kernels,
            "total_time_ms": round(wall_time_ms, 2),
            "avg_kernel_us": round(total_time_us / total_kernels, 1) if total_kernels else 0,
            "estimated_gpu_busy_pct": round(gpu_busy_pct, 1),
            "estimated_gpu_utilization_pct": round(gpu_busy_pct * 0.85, 1),  # rough adjustment
        },
        "phases": phases,
        "category_breakdown": category_stats,
        "top_slowest_kernels": top_slow,
        "bottlenecks": bottlenecks,
        "optimization_targets": generate_optimization_targets(category_stats, bottlenecks),
    }


def generate_optimization_targets(category_stats, bottlenecks):
    targets = []
    if "MMQ" in category_stats and category_stats["MMQ"]["pct"] > 15:
        targets.append({
            "target": "MMQ",
            "current_pct": category_stats["MMQ"]["pct"],
            "suggestion": "MMQ dominates. Consider K-tile doubling (already applied). Check if MMQ_ITER_K is optimal for RDNA4.",
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
            "suggestion": "Attention is a significant fraction. Consider flash attention tuning.",
        })
    if "MoE" in category_stats and category_stats["MoE"]["pct"] > 10:
        targets.append({
            "target": "MoE",
            "current_pct": category_stats["MoE"]["pct"],
            "suggestion": "MoE dispatch overhead. Check ExpertPool async prefetch.",
        })
    return targets


def generate_html_report(analysis, output_path):
    """Generate an HTML report from analysis data."""
    s = analysis["summary"]
    phases = analysis["phases"]
    cats = analysis["category_breakdown"]
    tops = analysis["top_slowest_kernels"]
    bots = analysis["bottlenecks"]
    targets = analysis["optimization_targets"]

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
<h1>🧭 AI-COMPASS Performance Report</h1>
<p>Generated: {analysis["analysis_time"]}</p>
<div class="card">
<h2>Summary</h2>
<table>
<tr><td>Total Kernels</td><td>{s["total_kernels"]}</td></tr>
<tr><td>Total GPU Time</td><td>{s["total_time_ms"]} ms</td></tr>
<tr><td>Avg Kernel</td><td>{s["avg_kernel_us"]} µs</td></tr>
<tr><td>Est. GPU Busy</td><td>{s["estimated_gpu_busy_pct"]}%</td></tr>
<tr><td>Est. GPU Utilization</td><td>{s["estimated_gpu_utilization_pct"]}%</td></tr>
</table>
</div>
<div class="card">
<h2>Phases</h2>
<table><tr><th>Phase</th><th>Kernels</th><th>Time (ms)</th><th>% Total</th><th>Avg Kernel (µs)</th></tr>
"""
    for p in phases:
        html += f"<tr><td>{p['phase']}</td><td>{p['kernel_count']}</td><td>{p['total_ms']}</td><td>{p['pct_of_total']}%</td><td>{p['avg_kernel_us']}</td></tr>"

    html += """</table></div><div class="card"><h2>Category Breakdown</h2><table><tr><th>Category</th><th>Count</th><th>Total (ms)</th><th>%</th><th>Avg (µs)</th><th>Max (µs)</th><th>Occupancy</th></tr>"""
    for cat, st in sorted(cats.items(), key=lambda x: -x[1]["pct"]):
        html += f"<tr><td>{cat}</td><td>{st['count']}</td><td>{st['total_ms']}</td><td>{st['pct']}%</td><td>{st['avg_us']}</td><td>{st['max_us']}</td><td>{st['avg_occupancy_pct']}%</td></tr>"

    html += """</table></div>"""

    if bots:
        html += """<div class="card"><h2>🚨 Bottlenecks</h2><ul>"""
        for b in bots:
            html += f'<li class="bottleneck">{b}</li>'
        html += "</ul></div>"

    if targets:
        html += """<div class="card"><h2>🔧 Optimization Targets</h2>"""
        for t in targets:
            html += f"<p><b>{t['target']}</b> ({t['current_pct']}% of time): {t['suggestion']}</p>"
        html += "</div>"

    html += """<div class="card"><h2>Top-10 Slowest Kernels</h2><table><tr><th>Kernel</th><th>Category</th><th>Duration (ms)</th><th>Grid</th><th>Block</th><th>Occupancy</th></tr>"""
    for k in tops:
        html += f"<tr><td>{k['kernel']}</td><td>{k['category']}</td><td>{k['duration_ms']}</td><td>{k['grid']}</td><td>{k['block']}</td><td>{k['occupancy_pct']}%</td></tr>"
    html += """</table></div></body></html>"""

    with open(output_path, "w") as f:
        f.write(html)
    return output_path


def compare_traces(baseline_path, target_path, cu_count=32):
    """Compare two traces (before/after optimization)."""
    base_recs = parse_trace(baseline_path)
    tgt_recs = parse_trace(target_path)
    base = analyze_trace(base_recs, cu_count)
    tgt = analyze_trace(tgt_recs, cu_count)

    diff = {
        "baseline": baseline_path,
        "target": target_path,
        "analysis_time": datetime.now().isoformat(),
        "improvements": {},
        "regressions": {},
    }

    # Compare summary
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


def main():
    parser = argparse.ArgumentParser(description="AI-COMPASS Analyze — kernel profiling & bottleneck detection")
    parser.add_argument("trace", help="HIP trace CSV file")
    parser.add_argument("-o", "--output", default="analysis_output", help="Output directory")
    parser.add_argument("--compare", help="Baseline CSV for before/after comparison")
    parser.add_argument("--cu-count", type=int, default=32, help="GPU CU/WGP count (default: 32 for RX 9070 XT)")
    parser.add_argument("--html", action="store_true", default=True, help="Generate HTML report")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    records = parse_trace(args.trace)
    if not records:
        print(f"❌ No records found in {args.trace}")
        return 1

    print(f"🧭 Analyzing {len(records)} kernel records from {args.trace}...")

    analysis = analyze_trace(records, args.cu_count)
    analysis["trace_file"] = args.trace

    # Print summary
    s = analysis["summary"]
    print(f"\n📊 Summary:")
    print(f"   {s['total_kernels']} kernels | {s['total_time_ms']} ms total")
    print(f"   Avg kernel: {s['avg_kernel_us']} µs | Est. GPU busy: {s['estimated_gpu_busy_pct']}%")

    # Print phases
    print(f"\n📈 Phases:")
    for p in analysis["phases"]:
        print(f"   {p['phase']}: {p['kernel_count']} kernels, {p['total_ms']} ms ({p['pct_of_total']}%)")

    # Print category breakdown
    print(f"\n🏷️  Category Breakdown:")
    for cat, st in sorted(analysis["category_breakdown"].items(), key=lambda x: -x[1]["pct"]):
        print(f"   {cat:15s} {st['count']:5d} kernels  {st['total_ms']:8.1f} ms  {st['pct']:5.1f}%  occ:{st['avg_occupancy_pct']:5.1f}%")

    # Print bottlenecks
    if analysis["bottlenecks"]:
        print(f"\n🚨 Bottlenecks:")
        for b in analysis["bottlenecks"]:
            print(f"   ⚠ {b}")

    # Print optimization targets
    if analysis["optimization_targets"]:
        print(f"\n🔧 Optimization Targets:")
        for t in analysis["optimization_targets"]:
            print(f"   [{t['target']}] ({t['current_pct']}%): {t['suggestion']}")

    # Save JSON
    json_path = os.path.join(args.output, "analysis.json")
    with open(json_path, "w") as f:
        json.dump(analysis, f, indent=2)
    print(f"\n💾 JSON report: {json_path}")

    # HTML report
    if args.html:
        html_path = os.path.join(args.output, "report.html")
        generate_html_report(analysis, html_path)
        print(f"📄 HTML report: {html_path}")

    # Compare mode
    if args.compare:
        print(f"\n🔄 Comparing against baseline: {args.compare}...")
        diff = compare_traces(args.compare, args.trace, args.cu_count)
        diff_path = os.path.join(args.output, "comparison.json")
        with open(diff_path, "w") as f:
            json.dump(diff, f, indent=2)
        if diff["improvements"]:
            print(f"\n✅ Improvements:")
            for k, v in diff["improvements"].items():
                print(f"   {k}: {v}")
        if diff["regressions"]:
            print(f"\n❌ Regressions:")
            for k, v in diff["regressions"].items():
                print(f"   {k}: {v}")
        print(f"💾 Comparison: {diff_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
