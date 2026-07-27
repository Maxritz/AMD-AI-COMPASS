# AI-COMPASS Architecture
## AI Compute Performance Analysis & Statistics Suite

### Design Philosophy
A unified plugin-based toolset integrating all AMD RDNA4 AI compute tools
(HIP tracer, GPUPerfAPI, GEAK, Hyperloom, Magpie, intellikit, Apex,
TraceLens, rocprofv3, ADLX, ROCm SMI, AQLProfile) into a single coherent
system with a unified CLI, data pipeline, and plugin registry.

### Plugin System
Each tool becomes a plugin implementing:
- `init(config)` — one-time setup
- `run(context)` — execute with trace context
- `shutdown()` — cleanup

### Data Pipeline
```
KernelRecords → CounterData → MetricsSamples → FormattedOutput
     ↑              ↑              ↑
  HIP Tracer    GPUPerfAPI     ADLX Poller
```

### Commands
```
aicompass trace     — capture HIP kernel + GPU counter trace
aicompass profile   — run full profile with all plugins
aicompass analyze   — post-process trace data
aicompass visualize — launch RCV viewer
aicompass tune      — apply optimizations via GEAK/Hyperloom
```

### Components Integrated
| Component     | Role                     | Status     |
|---------------|--------------------------|------------|
| HIP Tracer    | Kernel dispatch hook     | ✅ Built   |
| GPUPerfAPI    | SQ/SX/TA/TD counters     | ✅ Built   |
| ADLX          | GPU system metrics       | SDK cloned |
| GEAK-RDNA     | Agent-based optimization | Ported     |
| Hyperloom-RDNA| Performance tuning       | Ported     |
| Magpie        | Analysis                 | Ported     |
| intellikit    | AI toolkit               | Ported     |
| Apex          | Optimization             | Ported     |
| TraceLens     | Trace visualization      | Ported     |
| rocprofv3     | Compute profiler CLI     | Cloned     |
| RCV           | Trace GUI viewer         | Cloned     |
| ROCm SMI      | GPU monitoring           | Cloned     |
| AQLProfile    | Low-level counters       | Cloned     |
