"""Compare two benchmark runs with arch-aware analysis."""
import csv
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze import analyze_trace, parse_trace, ARCH_PROFILES

def main():
    if len(sys.argv) < 3:
        print("Usage: python compare_bench.py <baseline.csv> <optimized.csv> [--arch rdna4]")
        return 1

    base_path = sys.argv[1]
    opt_path = sys.argv[2]
    arch = "rdna4"
    if len(sys.argv) > 3 and sys.argv[3] == "--arch":
        arch = sys.argv[4] if len(sys.argv) > 4 else "rdna4"

    base_recs = parse_trace(base_path)
    opt_recs = parse_trace(opt_path)

    base_analysis = analyze_trace(base_recs, arch)
    opt_analysis = analyze_trace(opt_recs, arch)

    prof = ARCH_PROFILES.get(arch, ARCH_PROFILES["rdna4"])
    print(f"Architecture: {arch.upper()} ({prof['wave']}-wave, {prof['cu']} CUs)")
    print(f"Baseline: {len(base_recs)} kernels, {base_analysis['summary']['total_time_ms']:.2f} ms")
    print(f"Optimized: {len(opt_recs)} kernels, {opt_analysis['summary']['total_time_ms']:.2f} ms")

    chg = (opt_analysis['summary']['total_time_ms'] - base_analysis['summary']['total_time_ms'])
    chg /= base_analysis['summary']['total_time_ms'] * 100
    print(f"Change: {chg:+.2f}%")

    # Per-category comparison
    base_cats = base_analysis["category_breakdown"]
    opt_cats = opt_analysis["category_breakdown"]
    all_cats = set(list(base_cats.keys()) + list(opt_cats.keys()))
    if all_cats:
        print(f"\n{'Category':15s} {'Base (ms)':12s} {'Opt (ms)':12s} {'Change':10s}")
        print("-" * 50)
        for cat in sorted(all_cats):
            b = base_cats.get(cat, {}).get("total_ms", 0)
            o = opt_cats.get(cat, {}).get("total_ms", 0)
            c = ((o - b) / b * 100) if b else 0
            print(f"{cat:15s} {b:8.2f}ms   {o:8.2f}ms   {c:+7.2f}%")

    return 0

if __name__ == "__main__":
    sys.exit(main())
