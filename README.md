# AI-COMPASS

AI Compute Performance Analysis & Statistics Suite

A unified RDNA1–4 AI compute performance toolset for Windows, integrating
GPUPerfAPI, HIP tracer, ADLX, GEAK, Hyperloom, rocprofv3, RCV, and more.

## Quick Start

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release

# Trace an inference run
aicompass trace -- llm-cli -m model.gguf -n 256 -p "Once upon a time"

# Full profile with GPU counters
aicompass profile --counters -- llm-cli -m model.gguf -n 256
```

## Components

| Component     | Purpose                    | RDNA4 | RDNA3 | RDNA2 | RDNA1 |
|---------------|----------------------------|-------|-------|-------|-------|
| HIP Tracer    | Kernel dispatch hooking    | ✅    | ✅    | ✅    | ✅    |
| GPUPerfAPI    | HW performance counters    | ✅    | ✅    | ✅    | ✅    |
| ADLX          | GPU system metrics         | ✅    | ✅    | ✅    | ✅    |
| GEAK          | Agent optimization         | ✅    | ⚠     | ⚠     | ❌    |
| Hyperloom     | Performance tuning         | ✅    | ⚠     | ⚠     | ❌    |
| RCV           | Trace visualization        | ✅    | ✅    | ✅    | ⚠     |
| rocprofv3     | Compute profiling          | ✅    | ✅    | ✅    | ✅    |
| AQLProfile    | Low-level SQ counters      | ✅    | ✅    | ✅    | ⚠     |

✅ Full support  ⚠ Partial  ❌ Not supported

## GPU Support

Architecture detection is automatic via `hipGetDeviceProperties` / `ADLX`:

```cpp
enum class GpuArch {
    RDNA1,   // gfx1010-1012
    RDNA2,   // gfx1030-1035
    RDNA3,   // gfx1100-1103
    RDNA3_5, // gfx1150-1151
    RDNA4    // gfx1200-1201
};
```

## CLI Reference

```
aicompass trace     — Capture HIP kernel + GPU counter trace
aicompass profile   — Full profile with all plugins (trace+counters+metrics)
aicompass analyze   — Post-process trace data, find bottlenecks
aicompass visualize — Launch RCV viewer on trace data
aicompass list      — List installed plugins and their status
```

## License

MIT
