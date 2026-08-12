---
title: RDNA2 / RX 6000 FP8 status (gfx1030)
kind: hardware
gens: [gfx1030, gfx1031, gfx1032]
updated: 2026-08-12
---

# FP8 on RDNA2

**Not supported.** RDNA2 has no FP8 hardware, no FP8 WMMA, no AI accelerators.

- Do not propose FP8/E4M3/E5M2 kernels for gfx1030.
- Use **INT8** (native packed int8) or **FP16** instead.
- FP4/MXFP: no hardware support; emulated FP4 is a loss in every dimension on RDNA2.
