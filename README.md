# ROCm Hyperloom

[![Tests](https://github.com/AMD-AGI/Hyperloom/actions/workflows/tests-coverage.yml/badge.svg)](https://github.com/AMD-AGI/Hyperloom/actions/workflows/tests-coverage.yml)
[![Lint](https://github.com/AMD-AGI/Hyperloom/actions/workflows/lint.yml/badge.svg)](https://github.com/AMD-AGI/Hyperloom/actions/workflows/lint.yml)
[![Version](https://img.shields.io/badge/version-1.0.0a3-blue)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)

**Hyperloom** is an autonomous agentic system that optimizes end-to-end inference
workloads on AMD GPUs. It profiles your workload, pinpoints bottlenecks in both
host code and GPU kernels, implements targeted fixes, and validates the results
without manual intervention.

Traced through **TraceLens**, accelerated by **GEAK** kernel agents, and powered
by **Arbor** tree-search, Hyperloom combines:

- **Trace analysis** — identifies bottleneck kernels and bridge planning via the
  [TraceLens](https://github.com/AMD-AGI/TraceLens) Agent (Magpie + IntelliKit backends).
- **Kernel optimization** — GPU compute hotspots auto-tuned via the
  [GEAK](https://github.com/AMD-AGI/GEAK) backend.
- **Agentic search** — [Arbor](https://arxiv.org/abs/2606.12563), a tree-based
  cognition layer with dynamic specialist agents, long-horizon campaigns, and
  a curated hardware knowledge base of pitfalls and prior campaign artifacts.

<p align="center"><img width="600" alt="Hyperloom architecture" src="docs/images/Hyperloom_architecture.png" /></p>

## Supported GPUs

| GPU Family | Models | Architecture | GFX Target | ROCm Support |
|------------|--------|--------------|------------|--------------|
| Instinct MI300 | MI300X, MI308X, MI325X | CDNA3 | gfx940, gfx942 | ROCm native |
| Instinct MI350 / MI200 | MI355X, MI250X, MI200 | CDNA4 / CDNA2 | gfx950, gfx90a | ROCm native |
| Radeon RX 6000 series | RX 6900 XT, RX 6800 XT, RX 6800, RX 6700 XT, RX 6750 XT, RX 6650 XT, RX 6600 XT | RDNA2 | gfx1030, gfx1031, gfx1032 | Linux ROCm |
| Radeon RX 7000 series | RX 7900 XT, RX 7900 XTX, RX 7800 XT, RX 7700 XT | RDNA3 | gfx1100, gfx1151 | Linux ROCm |
| Radeon RX 9000 series | RX 9070 XT, RX 9070, RX 9060 XT, RX 9000 XT | RDNA4 | gfx1201, gfx1206, gfx1207 | Windows ROCm 7.3+ (WMMA FP8) |

Representative peak FP16 (dense) TFLOPS for entry-level consumer cards:

| GPU | CUs | Boost Clock | FP16 Dense | FP16 Achievable |
|-----|-----|-------------|------------|-----------------|
| RX 6600 XT | 32 | 2.586 GHz | 8.3 | 6.5 |
| RX 6700 XT | 40 | 2.42 GHz | 24.8 | 18.0 |
| RX 6800 XT | 72 | 2.25 GHz | 41.5 | 28.0 |
| RX 9070 XT | 64 | 2.66 GHz | 194.8 (matrix) | 170.0 |
| RX 9060 XT | 32 | 2.53 GHz | 102.7 | 71.2 |

## Get Started

| Goal | Guide |
|------|-------|
| Set up Hyperloom and run a demo | [Quickstart](examples/README.md) |
| Launch and monitor an optimization | [Run an optimization](docs/how-to/optimize.md) |
| Understand the algorithm | [Optimization loop](docs/conceptual/optimization-loop.md) |
| Vulkan LLM engine (rdna4-llm) | [Architecture](docs/VULKAN-LLM-ENGINE-ARCHITECTURE.md) |

## Toolkits

Hyperloom vendors or wraps several toolkits — run `python aicompass.py status`
for a live summary of what is available in your environment.

| Toolkit | Location | Role |
|---------|----------|------|
| **Hyperloom** | `src/hyperloom/` | Core orchestrator, CLI, agent loop |
| **GEAK** | `vendor/GEAK/` | Multi-agent GPU kernel optimizer |
| **Magpie** | `vendor/Magpie/` | Trace collection & benchmark runners |
| **TraceLens** | `src/TraceLens-RDNA4/` | Profiling analysis & roofline targets |
| **IntelliKit** | `src/intellikit-RDNA4/` | LLM engine, metrix profiling, agent-reach |
| **Arbor** | `src/arboreal/` | Tree-search cognitive layer |
| **rdna4-llm** | `src/rdna4-llm/` | Vulkan compute-shader LLM inference engine |

## Developer Entry Points

| Purpose | Path |
|---------|------|
| CLI entry point | `python -m hyperloom.inference_optimizer.cli optimize` |
| Main agent instructions | [`src/hyperloom/inference_optimizer/SKILL.md`](src/hyperloom/inference_optimizer/SKILL.md) |
| Operator tools | `python -m hyperloom.inference_optimizer.tools.*` |
| Unified tool CLI | `python aicompass.py status` |
| GPU arch resolution | `src/hyperloom/inference_optimizer/gpu_types.py` |
| Roofline specs | `src/hyperloom/orchestrator/kernel/roofline_ceiling.py` |
| Per-platform arch JSONs | `src/TraceLens-RDNA4/TraceLens/Agent/Analysis/utils/arch/` |

## Documentation

| Topic | Link |
|-------|------|
| ROCm Docs | [Hyperloom](https://rocm.docs.amd.com/projects/hyperloom/en/latest/index.html) |
| Authentication and credentials | [Authentication & credentials](docs/reference/authentication.md) |
| Environment variables | [Environment variables](docs/reference/environment-variables.md) |
| Compatibility matrix | [Compatibility matrix](docs/compatibility.rst) |
| Troubleshooting | [Troubleshooting](docs/reference/troubleshooting.md) |

## Third-party / Vendored components

| Path | What | Usage |
|------|------|-------|
| `third_party/vma/vk_mem_alloc.h` | Vulkan Memory Allocator (header-only) | used by rdna4-llm buffer allocation |
| `src/rdna4-llm/third_party/vma/` | VMA copy for current engine | same |
| `vendor/agentreach/` | Agent Reach v1.5.0 (internet capability layer) | `python aicompass.py agentreach` or `agent-reach` |

All vendored code is unmodified upstream, MIT-licensed. The AMD Radeon Developer
Tool Suite is a manual download — gitignored, never synced to git.

---

## Acknowledgments

This repo vendors **[Agent Reach](https://github.com/Panniantong/Agent-Reach)**
(v1.5.0, MIT) to give AI agents internet access. Agent Reach is built and
maintained by **Panniantong** ([@Panniantong](https://github.com/Panniantong)).

---

## Licensing

Hyperloom is released under the **MIT License**. See [`LICENSE`](LICENSE).
Third-party tools and agents are governed by their own separate license terms.
