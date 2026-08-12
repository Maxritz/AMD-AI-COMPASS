#!/usr/bin/env python3
"""
AI-COMPASS Toolkit Bridge
=========================
Discovers and imports AMD AGI toolkits.
Priority:
  1. Bundled inside AI-COMPASS (`AI-COMPASS/vendor/`) — 100% self-contained!
  2. Installed Python packages in site-packages
  3. External repository check (sibling directories)

Toolkits bundled & supported:
  - GEAK               : GPU kernel optimization agent
  - Hyperloom          : Autonomous inference optimizer
  - Magpie             : GPU kernel evaluator & LLM benchmarker
  - TraceLens          : Profiler trace analysis library
  - intellikit         : Kerncap, Metrix, Linex, Nexus, Accordo
  - Apex               : RL-based GPU kernel optimization pipeline

Usage:
    from toolkit_bridge import toolkit_status, try_import

    status = toolkit_status()
    magpie = try_import("Magpie")       # Never fails if vendor/ is present
    tracelens = try_import("TraceLens") # Never fails if vendor/ is present
"""

import os
import sys
import importlib
from pathlib import Path
from typing import Optional, Dict, Any

_THIS_DIR = Path(__file__).resolve().parent
_AI_COMPASS_ROOT = _THIS_DIR.parent
_VENDOR_DIR = _AI_COMPASS_ROOT / "vendor"
_AMD_AI_ROOT = _AI_COMPASS_ROOT.parent

# Automatically insert vendor/ into sys.path at module load time
if _VENDOR_DIR.exists() and str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

_PORTS_DIR = _AI_COMPASS_ROOT / "external" / "rdna4-ports"

_TOOLKITS = {
    "GEAK": {
        "vendor_package": "geak",
        "ext_path": _PORTS_DIR / "GEAK-RDNA",
        "description": "Autonomous GPU kernel optimization agent (RDNA4 + Instinct)",
        "entry": "geak.bootstrap:main",
    },
    "Hyperloom": {
        "vendor_package": "hyperloom",
        "ext_path": _AI_COMPASS_ROOT / "src",
        "description": "Autonomous end-to-end inference optimizer (vLLM/SGLang)",
        "entry": "hyperloom.inference_optimizer.cli.__init__:main",
    },
    "Magpie": {
        "vendor_package": "Magpie",
        "ext_path": _PORTS_DIR / "Magpie-RDNA4",
        "description": "GPU kernel evaluator, LLM benchmarker (RDNA4 port)",
        "entry": "Magpie.main:main",
    },
    "TraceLens": {
        "vendor_package": "TraceLens",
        "ext_path": _PORTS_DIR / "TraceLens-RDNA4",
        "description": "Automated profiler trace analysis library + reporting scripts (RDNA4 port)",
    },
    "intellikit": {
        "vendor_package": "kerncap",
        "ext_path": _PORTS_DIR / "intellikit-RDNA4",
        "description": "Agent-first AMD tools: Kerncap, Metrix, Linex, Nexus, Accordo",
        "nested_packages": ["kerncap/kerncap", "accordo/accordo",
                            "linex/src", "metrix/src", "nexus/nexus"],
        "components": ["kerncap", "accordo", "linex", "metrix", "nexus"],
    },
    "Apex": {
        "vendor_package": "pipeline",
        "ext_path": _PORTS_DIR / "Apex-RDNA4",
        "description": "RL-based GPU kernel optimization pipeline (RDNA4 port)",
    },
    "AgentReach": {
        "vendor_package": "agent_reach",
        "ext_path": _VENDOR_DIR / "agentreach",
        "description": "Internet capability layer for AI agents (v1.5.0, MIT)",
        "entry": "agent_reach.cli:main",
    },
}


def try_import(toolkit_name: str):
    """Import a toolkit by name. Returns module or None — never raises."""
    info = _TOOLKITS.get(toolkit_name)
    if not info:
        return None

    pkg = info["vendor_package"]

    def _try(pkg_name):
        try:
            return importlib.import_module(pkg_name)
        except (ImportError, ModuleNotFoundError):
            return None

    # 1. Prefer the RDNA4 port dir (external/rdna4-ports) if present
    ext_path = info["ext_path"]
    if ext_path.exists() and str(ext_path) not in sys.path:
        sys.path.insert(0, str(ext_path))
    mod = _try(pkg)
    if mod:
        return mod

    # 1b. For intellikit, try adding nested package paths
    for nested in info.get("nested_packages") or []:
        nested_path = info["ext_path"] / nested
        if nested_path.exists() and str(nested_path) not in sys.path:
            sys.path.insert(0, str(nested_path))
    if info.get("nested_packages"):
        mod = _try(pkg) or _try("kerncap")
        if mod:
            return mod

    # 2. Fallback to vendor/ or site-packages
    return _try(pkg)


def toolkit_status() -> Dict[str, Any]:
    """Return dict: toolkit name -> {available, source, description}."""
    status = {}
    for name, info in _TOOLKITS.items():
        pkg = info["vendor_package"]
        available = False
        source = "missing"

        def _try(pkg_name):
            try:
                importlib.import_module(pkg_name)
                return True
            except (ImportError, ModuleNotFoundError):
                return False

        # Add nested packages to path for intellikit
        for nested in info.get("nested_packages") or []:
            nested_path = info["ext_path"] / nested
            if nested_path.exists() and str(nested_path) not in sys.path:
                sys.path.insert(0, str(nested_path))

        # Check RDNA4 port first (external/rdna4-ports)
        is_port = _PORTS_DIR in info["ext_path"].parents
        if info["ext_path"].exists():
            if str(info["ext_path"]) not in sys.path:
                sys.path.insert(0, str(info["ext_path"]))
            try:
                importlib.import_module(pkg)
                available = True
                source = "RDNA4 port (external/)" if is_port else "bundled (vendor/)"
            except (ImportError, ModuleNotFoundError):
                pass

        # Check vendor / site-packages fallback
        if not available and ((_VENDOR_DIR / pkg).exists() or (_VENDOR_DIR / name).exists() or info.get("nested_packages")):
            try:
                importlib.import_module(pkg)
                available = True
                source = "bundled (vendor/)"
            except (ImportError, ModuleNotFoundError):
                pass

        status[name] = {
            "available": available,
            "source": source,
            "description": info["description"],
        }
    return status


def print_status():
    """Print human-readable toolkit and built-in tool availability."""
    status = toolkit_status()
    print("=" * 70)
    print("AI-COMPASS Standalone Toolkit Status")
    print("=" * 70)
    ok = sum(1 for v in status.values() if v["available"])
    print(f"  Toolkits available: {ok}/{len(status)} (RDNA4 ports in external/rdna4-ports, bundled in vendor/ or src/)")
    print()
    for name, info in status.items():
        if info["available"]:
            icon = "[OK]  "
        else:
            icon = "[---] "
        print(f"  {icon} {name:<14} [{info['source']:<17}] {info['description']}")
    print()

    print("  Built-in tools (always available in this repo):")
    builtin = [
        ("detect.py",           "CPU + GPU hardware detection (Windows & Linux)"),
        ("detect_instinct.py",  "AMD Instinct GPU detection via amd-smi (Linux)"),
        ("validate.py",         "Pre-flight environment validation"),
        ("cpu_tune.py",         "CPU topology + affinity tuning for vLLM"),
        ("check_model.py",      "vLLM model architecture support checker"),
        ("estimate_memory.py",  "CPU-serving RAM estimator (GGUF / HF models)"),
        ("estimate_vram.py",    "GPU VRAM estimator (quantized, MLA, TP-aware)"),
        ("sync_recipes.py",     "Refresh vLLM recipes cache from GitHub"),
        ("analyze.py",          "HIP kernel trace analysis + optimization report"),
        ("run_benchmark.py",    "End-to-end benchmark runner with HIP tracer"),
    ]
    for fname, desc in builtin:
        icon = "[OK]" if (_THIS_DIR / fname).exists() else "[--]"
        print(f"    {icon}  {fname:<26} {desc}")
    print()

    extra = [
        ("compare_bench.py",     "Benchmark run comparator with arch-aware analysis"),
        ("fix_kernel_names.py",  "HIP kernel name tagger (ggml pattern matching)"),
        ("memory_hub.py",        "TencentDB Agent Memory bridge (capture/recall/search)"),
        ("bootstrap_memory.py",  "One-shot gateway clone + install + start"),
    ]
    print("  Additional tools (in tools/):")
    for fname, desc in extra:
        icon = "[OK]" if (_THIS_DIR / fname).exists() else "[--]"
        print(f"    {icon}  {fname:<26} {desc}")

    # AMD Radeon Developer Tool Suite (never synced to git; manual download)
    rdts_dirs = sorted(_AI_COMPASS_ROOT.glob("RadeonDeveloperToolSuite-*/"))
    if rdts_dirs:
        print(f"  AMD Radeon Developer Tool Suite: {rdts_dirs[-1].name} (present)")
    else:
        print("  AMD Radeon Developer Tool Suite: NOT INSTALLED -")
        print("        download from AMD (GPUOpen) and extract to this folder; gitignored, never synced")
    print()

    # metrix profiler engine stub warning
    metrix_engine = _VENDOR_DIR / "metrix" / "src" / "metrix" / "profiler" / "engine.py"
    if metrix_engine.exists():
        try:
            src = metrix_engine.read_text(encoding="utf-8")
            if "NotImplementedError" in src and "not yet implemented" in src:
                print()
                print("  Note: metrix profiling engine is a stub (NotImplementedError).")
                print("        Use metrix for metrics listing / info; profiler needs backend.")
        except OSError:
            pass

    print()

    print("  Bundled AMD Skills (in skills/):")
    skills_dir = _AI_COMPASS_ROOT / "skills"
    skills = [
        ("serving-llms-on-instinct",        "Serve LLMs on AMD Instinct MI300X/MI350 (Linux)"),
        ("serving-llms-on-epyc",            "Serve LLMs on AMD EPYC/Ryzen + RDNA GPU (Win/Linux)"),
        ("magpie-kernel-evaluator",         "Kernel correctness + perf evaluation (Magpie)"),
        ("tracelens-analysis-orchestrator", "Agentic TraceLens analysis workflow"),
        ("local-ai-use",                    "Local Lemonade Server (Ryzen AI / NPU)"),
        ("local-ai-app-integration",        "Local AI integration for Windows apps"),
    ]
    for sname, sdesc in skills:
        icon = "[OK]" if (skills_dir / sname).exists() else "[--]"
        print(f"    {icon}  {sname:<38} {sdesc}")
    print("=" * 70)


def bootstrap_memory_context(label: str = "") -> str:
    """Pull relevant optimization context from TencentDB Agent Memory hub."""
    try:
        from memory_hub import recall_context
        ctx = recall_context(label or "AMD GPU kernel optimization")
        if ctx:
            print(f"\n[MEMORY] Boostrap context ({len(ctx)} chars):\n{ctx[:2000]}")
            return ctx
    except (ImportError, OSError):
        pass
    return ""


if __name__ == "__main__":
    print_status()
