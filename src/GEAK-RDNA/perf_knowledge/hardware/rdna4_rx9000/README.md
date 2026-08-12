| Feature | RDNA4 RX 9000 | CDNA4 MI350X |
|---|---|---|
| Target | Gaming/Consumer | Datacenter/Accelerator |
| Architecture | WMMA (wavefront ALUs) | Matrix cores (dedicated units) |
| CU Count | 40-64 | 256 |
| Memory | GDDR6 16GB | HBM3E 288GB |
| Bandwidth | 640 GB/s | 8 TB/s |
| FP8 Support | WMMA native | Matrix cores |
| FP6/FP4 | N/A | Matrix cores |
| TDP | 300W | 1000-1400W |
| TDP/Peak FP16 | 1.54 W/TFLOPS | 0.33-0.4 W/TFLOPS |
| L2 | 8 MB unified | 256 MB + Infinity Cache |
| LDS | 128 KiB, 32 banks | 160 KiB, 64 banks |

## RDNA4 Optimization Levers

1. **FP8 WMMA** — enable via vLLM patches (`-DVLLM_DISABLE_AITER=1`)
2. **Tile sizing** — 16×16×128 for FP8 WMMA, 64×64×32 for FP16
3. **Sparsity** — INT4 path achieves 1.5k AI TOPS with 2:1 weight sparsity
4. **Memory access patterns** — 32 LDS banks, avoid bank conflicts
5. **GDDR6 bandwidth** — use Q4_0/Q4_1 quantization, PagedAttention

## RDNA4 Limitations

- **No dedicated matrix cores** — WMMA shares ALU bandwidth
- **GDDR6 bottleneck** — 640 GB/s vs CDNA4's 8 TB/s
- **Smaller memory** — 16GB vs 288GB HBM3E
- **Power efficiency** — 300W TDP caps sustained throughput

## RDNA4 vs RDNA3 Comparison

| Feature | RDNA3 (RX 7900) | RDNA4 (RX 9070) |
|---|---|---|
| CU Count | 96 | 64 |
| FP16 Peak | 346 TFLOPS | 195 TFLOPS |
| AI Accelerators | 2nd-gen | 2nd-gen |
| FP8 | No native | WMMA native |
| LDS | 128 KiB, 64 banks | 128 KiB, 32 banks |
| Infinity Cache | 80 MiB | 80 MiB |