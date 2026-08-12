---
title: RDNA2 / RX 6000 series (gfx1030) knowledge base
kind: hardware
gens: [gfx1030, gfx1031, gfx1032]
updated: 2026-08-12
---

# RDNA2 / RX 6000 (gfx1030)

Hardware knowledge base for AMD Radeon RX 6000 series (RDNA2, gfx1030/gfx1031/gfx1032).

Docs:
- [arch.md](arch.md) — architecture overview + cheat sheet
- [peak_tables.md](peak_tables.md) — SKU peak tables + roofline
- [lds.md](lds.md) — LDS specifics
- [wmma.md](wmma.md) — WMMA (absent on RDNA2)
- [fp8_wmma.md](fp8_wmma.md) — FP8 (absent on RDNA2)
- [windows_deployment.md](windows_deployment.md) — deployment (Linux ROCm / Windows Vulkan)

Key facts:
- Wave64, vector-only (no WMMA/FP8/AI-accelerators), FP16 native, BF16 emulated, INT8 native.
- LLM decode is GDDR6-bandwidth-bound (~512 GB/s top SKU).
