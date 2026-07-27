"""
Tag HIP trace kernels with ggml names based on grid/block/shared_mem patterns.

Each ggml kernel has a unique dispatch signature. We match by:
  (block_x, block_y, block_z, grid_x_range, shared_mem_range)

Usage: python tools/fix_kernel_names.py trace.csv --output fixed.csv
"""
import csv, sys, argparse
from collections import defaultdict

# Pattern DB: (block_x, block_y, block_z, min_gx, max_gx, min_smem, max_smem, category, name)
PATTERNS = [
    # MMVQ - K-quant matmul vec with large shared mem (K tile)
    (32, 8, 1, 0, 9999, 50000, 99999, "MMVQ", "mmvq_kq"),
    # MMVQ - small grid, block 128
    (128, 1, 1, 1, 100, 0, 100, "MMVQ", "mul_mat_vec_q"),
    # MMVQ - small grid, block 32
    (32, 1, 1, 1, 100, 0, 100, "MMVQ", "mul_mat_vec_q"),

    # MMQ K-tile - large grid, block 64, shared mem
    (64, 1, 1, 100, 99999, 128, 99999, "MMQ", "mul_mat_q_K"),
    # MMQ - large grid, block 64, no shared mem
    (64, 1, 1, 100, 99999, 0, 100, "MMQ", "mul_mat_q"),

    # GEMM (dequantize + gemm) - block 1x256
    (1, 256, 1, 100, 99999, 0, 100, "Quantize", "dequantize"),

    # Flash attention - block 128, grid in [16, 9999]
    (128, 1, 1, 16, 9999, 0, 100, "Attention", "flash_attn"),

    # Soft max - block 256, small grid
    (256, 1, 1, 1, 50, 0, 100, "Attention", "soft_max"),
    (256, 1, 1, 100, 9999, 0, 100, "Attention", "soft_max_batch"),

    # RMS norm - block 256, medium grid (hidden dim)
    (256, 1, 1, 100, 9999, 100, 9999, "Norm", "rms_norm"),

    # RoPE - block 256
    (256, 1, 1, 1, 9999, 0, 100, "RoPE", "rope"),

    # Element-wise (add, mul, silu, gelu) - block 256, small grid
    (256, 1, 1, 1, 99, 0, 100, "Vector", "elementwise"),

    # Get rows - block 32x8, grid up to hidden dim
    (32, 8, 1, 100, 99999, 0, 100, "Vector", "get_rows"),

    # Copy/dup - block 32x8, large grid
    (32, 8, 1, 1000, 999999, 0, 100, "Vector", "cpy"),

    # Scale - block 1x256, any grid
    (1, 256, 1, 1, 99, 0, 100, "Vector", "scale"),

    # Reshape views - block 32x2
    (32, 2, 1, 1, 100, 0, 100, "Vector", "reshape"),

    # Cross-entropy / loss - block 512
    (512, 1, 1, 1, 100, 0, 100, "Other", "cross_entropy"),
]

def classify(gx, gy, gz, bx, by, bz, smem):
    if bx == 0: return ("Other", "unknown")
    for pbx, pby, pbz, mn, mx, mns, mxs, cat, name in PATTERNS:
        if (bx == pbx and by == pby and bz == pbz and
            mn <= gx <= mx and mns <= smem <= mxs):
            return (cat, name)
    # Fallback: try dims-based matching
    if bx == 32 and by == 8 and bz == 1:
        return ("Vector", "vec32x8")
    if bx == 128 and by == 1 and bz == 1:
        if gx < 100: return ("MMVQ", "mul_mat_vec_q")
        return ("Attention", "flash_attn")
    if bx == 256 and by == 1:
        if gx > 100: return ("Norm", "rms_norm")
        return ("Vector", "elementwise")
    if bx == 1 and by == 256:
        return ("Quantize", "dequantize")
    if bx == 512:
        return ("Other", "special")
    return ("Other", f"block_{bx}x{by}x{bz}")


def main():
    parser = argparse.ArgumentParser(description="Tag HIP trace kernels by dispatch pattern")
    parser.add_argument("trace", help="HIP trace CSV")
    parser.add_argument("-o", "--output", help="Output fixed CSV")
    parser.add_argument("--show", action="store_true", help="Show pattern groups")
    args = parser.parse_args()

    with open(args.trace) as f:
        records = list(csv.DictReader(f))

    # Group by unique kernel dispatch signature
    groups = defaultdict(list)
    for r in records:
        key = (int(r["grid_x"]), int(r["grid_y"]), int(r["grid_z"]),
               int(r["block_x"]), int(r["block_y"]), int(r["block_z"]),
               int(r["shared_mem"]))
        groups[key].append(r)

    if args.show:
        print(f"Total: {len(records)} dispatches, {len(groups)} unique patterns")
        print(f"\n{'Pattern (grid x block x smem)':<55s} {'Count':>5s} {'Category':>12s} {'Name':>20s}")
        print("-"*95)
        for key, recs in sorted(groups.items(), key=lambda x: -len(x[1])):
            gx, gy, gz, bx, by, bz, smem = key
            cat, name = classify(gx, gy, gz, bx, by, bz, smem)
            pat = f"{gx}x{gy}x{gz} / {bx}x{by}x{bz} / {smem}B"
            print(f"  {pat:<55s} {len(recs):>5d} {cat:>12s} {name:>20s}")

    if args.output:
        with open(args.output, "w", newline="") as f:
            fieldnames = list(records[0].keys()) + ["fixed_name", "category"]
            w = csv.writer(f)
            w.writerow(fieldnames)
            for r in records:
                gx, gy, gz = int(r["grid_x"]), int(r["grid_y"]), int(r["grid_z"])
                bx, by, bz = int(r["block_x"]), int(r["block_y"]), int(r["block_z"])
                smem = int(r["shared_mem"])
                cat, name = classify(gx, gy, gz, bx, by, bz, smem)
                w.writerow(list(r.values()) + [name, cat])
        print(f"\nFixed CSV: {args.output}")

    # Category summary
    cats = defaultdict(int)
    names = defaultdict(int)
    for r in records:
        gx, gy, gz = int(r["grid_x"]), int(r["grid_y"]), int(r["grid_z"])
        bx, by, bz = int(r["block_x"]), int(r["block_y"]), int(r["block_z"])
        smem = int(r["shared_mem"])
        cat, name = classify(gx, gy, gz, bx, by, bz, smem)
        cats[cat] += 1
        names[name] += 1

    print(f"\nCategory distribution:")
    for cat, cnt in sorted(cats.items(), key=lambda x: -x[1]):
        pct = cnt / len(records) * 100
        print(f"  {cat:15s} {cnt:5d} ({pct:5.1f}%)")
    print(f"\nKernel distribution:")
    for name, cnt in sorted(names.items(), key=lambda x: -x[1]):
        pct = cnt / len(records) * 100
        print(f"  {name:20s} {cnt:5d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
