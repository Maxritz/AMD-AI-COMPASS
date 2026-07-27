"""Compare two benchmark runs."""
import csv, sys

base = sys.argv[1]
opt = sys.argv[2]

with open(base) as f: base_rows = list(csv.DictReader(f))
with open(opt) as f: opt_rows = list(csv.DictReader(f))

base_ms = sum(float(r["duration_us"]) for r in base_rows) / 1000
opt_ms = sum(float(r["duration_us"]) for r in opt_rows) / 1000

print(f"Baseline: {len(base_rows)} kernels, {base_ms:.2f} ms")
print(f"Optimized: {len(opt_rows)} kernels, {opt_ms:.2f} ms")
chg = (opt_ms - base_ms) / base_ms * 100
print(f"Change: {chg:+.2f}%")
