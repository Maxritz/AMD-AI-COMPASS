#!/usr/bin/env python3
"""
AI-COMPASS — Unified AI Compute Suite Entry Point
=================================================
Single CLI front-end unifying all sub-tools, profilers, evaluators, and agentic workflows.
"""

import sys
import os
import argparse
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR / "tools"))
sys.path.insert(0, str(_THIS_DIR / "vendor"))

try:
    import toolkit_bridge
except ImportError as exc:
    print(f"[FATAL] toolkit_bridge module missing: {exc}")
    sys.exit(1)


def print_banner():
    print("""
    +------------------------------------------------------------------+
    |                      AI-COMPASS v0.1.0                           |
    |   Unified AI Compute Performance & Agentic Optimization Suite    |
    |            AMD RDNA (1-4) & Instinct (CDNA) — Multi-Backend      |
    +------------------------------------------------------------------+
""")


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help", "help"):
        print_banner()
        print("Usage: python aicompass.py <command> [options]\n")
        print("Primary Commands:")
        print("  detect            Detect CPU & GPU hardware (Windows & Linux)")
        print("  validate          Run system environment pre-flight checks")
        print("  cpu-tune          Calculate optimal CPU affinity & NUMA bindings")
        print("  estimate-ram      Estimate host RAM fit for CPU serving")
        print("  estimate-vram     Estimate GPU VRAM fit (supports TP, FP8, MLA)")
        print("  check-model       Check if model is supported by vLLM")
        print("  sync-recipes      Sync vLLM recipe database from GitHub")
        print("  benchmark         Run model benchmark with HIP kernel tracer")
        print("  analyze           Analyze trace CSV and generate HTML report")
        print("  compare-bench     Compare two benchmark runs with arch analysis")
        print("  fix-kernel-names  Tag HIP trace kernels with ggml names")
        print("  status            Display status of all bundled toolkits & skills")
        print("\nAgentic & Vendor Toolkits:")
        print("  magpie            Run Magpie kernel evaluator / benchmark")
        print("  tracelens         Show TraceLens reporting scripts and analysis tools")
        print("  geak              Bootstrap GEAK autonomous kernel optimizer")
        print("  hyperloom         Run Hyperloom autonomous inference optimizer")
        print("  intellikit        Show bundled AMD agent tools (Kerncap, Metrix, Accordo...)")
        print("  apex              Show Apex RL-based GPU kernel optimization pipeline")
        print("  agentreach        Run Agent-Reach internet capability CLI")
        print("\nMemory Hub:")
        print("  memory            Manage TencentDB Agent Memory (capture/search/status)")
        print("  bootstrap-memory  First-time: start the memory gateway daemon")
        print("\nTry: python aicompass.py status")
        sys.exit(0)

    cmd = sys.argv[1].lower()
    sys.argv.pop(1)

    if cmd == "status":
        print_banner()
        toolkit_bridge.print_status()
    elif cmd in ("detect", "detect-hardware"):
        try:
            import detect
            detect.main()
        except ImportError as exc:
            print(f"Error: detect tool not available: {exc}")
            sys.exit(1)
    elif cmd in ("detect-instinct", "detect-gpu"):
        try:
            import detect_instinct
            detect_instinct.main()
        except ImportError as exc:
            print(f"Error: detect_instinct tool not available: {exc}")
            sys.exit(1)
    elif cmd == "validate":
        try:
            import validate
            validate.main()
        except ImportError as exc:
            print(f"Error: validate tool not available: {exc}")
            sys.exit(1)
    elif cmd in ("cpu-tune", "cputune"):
        try:
            import cpu_tune
            cpu_tune.main()
        except ImportError as exc:
            print(f"Error: cpu_tune tool not available: {exc}")
            sys.exit(1)
    elif cmd in ("estimate-ram", "estimate-memory"):
        try:
            import estimate_memory
            estimate_memory.main()
        except ImportError as exc:
            print(f"Error: estimate_memory tool not available: {exc}")
            sys.exit(1)
    elif cmd in ("estimate-vram", "estimatevram"):
        try:
            import estimate_vram
            estimate_vram.main()
        except ImportError as exc:
            print(f"Error: estimate_vram tool not available: {exc}")
            sys.exit(1)
    elif cmd in ("check-model", "checkmodel"):
        try:
            import check_model
            check_model.main()
        except ImportError as exc:
            print(f"Error: check_model tool not available: {exc}")
            sys.exit(1)
    elif cmd in ("sync-recipes", "syncrecipes"):
        try:
            import sync_recipes
            sync_recipes.main()
        except ImportError as exc:
            print(f"Error: sync_recipes tool not available: {exc}")
            sys.exit(1)
    elif cmd == "analyze":
        try:
            import analyze
            analyze.main()
        except ImportError as exc:
            print(f"Error: analyze tool not available: {exc}")
            sys.exit(1)
    elif cmd == "benchmark":
        try:
            import run_benchmark
            run_benchmark.main()
        except ImportError as exc:
            print(f"Error: benchmark tool not available: {exc}")
            sys.exit(1)
    elif cmd == "magpie":
        magpie = toolkit_bridge.try_import("Magpie")
        if magpie:
            try:
                from Magpie.main import main as magpie_main
                magpie_main()
            except ImportError as exc:
                print(f"Error: Magpie.main not found: {exc}")
                sys.exit(1)
        else:
            print("Error: Magpie toolkit not available.")
            sys.exit(1)
    elif cmd == "tracelens":
        tracelens = toolkit_bridge.try_import("TraceLens")
        if tracelens:
            print("TraceLens loaded. Available reporting/analysis scripts in vendor/TraceLens/:")
            import glob as _glob
            scripts = sorted(_glob.glob(str(_THIS_DIR / "vendor" / "TraceLens" / "Reporting" / "*.py")) +
                            [str(_THIS_DIR / "vendor" / "TraceLens" / "PerfModel" / "run_perf_model.py")])
            for s in scripts:
                name = Path(s).name
                print(f"  Reporting/{name}" if "Reporting" in str(s) else f"  PerfModel/{name}")
            print("\nRun directly: python vendor/TraceLens/Reporting/<script>.py --input trace.json")
        else:
            print("Error: TraceLens toolkit not available.")
            sys.exit(1)
    elif cmd == "geak":
        geak = toolkit_bridge.try_import("GEAK")
        if geak:
            toolkit_bridge.bootstrap_memory_context("GEAK AMD GPU kernel optimization")
            print(f"GEAK v{getattr(geak, '__version__', '4.0.0')} loaded.")
            print("Run bootstrap to set up: python -m geak.bootstrap")
            print("Then invoke: claude --agent geak-internal <task>")
        else:
            print("Error: GEAK toolkit not available.")
            sys.exit(1)
    elif cmd == "hyperloom":
        hyperloom = toolkit_bridge.try_import("Hyperloom")
        if hyperloom:
            toolkit_bridge.bootstrap_memory_context("Hyperloom AMD inference optimization")
            print("Hyperloom loaded. Entry points:")
            print("  CLI:     python -m hyperloom.inference_optimizer.cli optimize --config cfg.yaml")
            print("  Setup:   python -m hyperloom.inference_optimizer.setup --check-only")
            print("  Quant:   python -m hyperloom.agents.quantization.cli --help")
            print("  Kernel:  python -m hyperloom.agents.kernel.tools.kernel_optimization")
        else:
            print("Error: Hyperloom toolkit not available.")
            sys.exit(1)
    elif cmd == "intellikit":
        print("Bundled AMD Agent Toolkit Components (vendor/):")
        components = [
            ("kerncap",  "Kerncap: GPU kernel capture, extract, validate, replay"),
            ("metrix",   "Metrix: GPU metrics profiler (engine stub — metrics listing/info functional)"),
            ("accordo",  "Accordo: Kernel binary validation (HIP kernel comparison)"),
            ("linex",    "Linex: Source-to-instruction mapping (AMD GPU ISA decoder)"),
            ("nexus",    "Nexus: GPU trace capture & analysis API"),
        ]
        for comp, desc in components:
            p = _THIS_DIR / "vendor" / comp
            icon = "[OK]" if p.exists() else "[--]"
            print(f"  {icon} {comp:<12} {desc}")
        print("\nInstalled via pip from vendor/ subdirectories:")
        print("  pip install -e vendor/kerncap    # kerncap")
        print("  pip install -e vendor/metrix     # metrix")
        print("  pip install -e vendor/accordo    # accordo")
        print("  pip install -e vendor/linex      # linex")
        print("  pip install -e vendor/nexus      # nexus")
    elif cmd == "apex":
        apex = toolkit_bridge.try_import("Apex")
        print("Apex RL-based GPU Kernel Optimization Pipeline")
        if apex:
            print("  Module loaded. Scripts in vendor/pipeline/:")
            import glob as _glob
            for s in sorted(_glob.glob(str(_THIS_DIR / "vendor" / "pipeline" / "*.py"))):
                if not s.endswith("__init__.py"):
                    print(f"    {Path(s).name}")
            print("\nRun: python vendor/pipeline/export_rl_dataset.py")
        else:
            print("  pipeline module not directly importable — use scripts directly:")
            print("    python vendor/pipeline/export_rl_dataset.py")
    elif cmd == "agentreach":
        ar = toolkit_bridge.try_import("AgentReach")
        if ar:
            try:
                from agent_reach.cli import main as ar_main
                ar_main()
            except ImportError as exc:
                print(f"Error: agent_reach.cli not found: {exc}")
                sys.exit(1)
        else:
            print("Error: AgentReach toolkit not available (vendor/agentreach).")
            sys.exit(1)
    elif cmd == "compare-bench":
        try:
            import compare_bench
            sys.exit(compare_bench.main())
        except ImportError as exc:
            print(f"Error: compare_bench tool not available: {exc}")
            sys.exit(1)
    elif cmd == "fix-kernel-names":
        try:
            import fix_kernel_names
            fix_kernel_names.main()
            sys.exit(0)
        except ImportError as exc:
            print(f"Error: fix_kernel_names tool not available: {exc}")
            sys.exit(1)
    elif cmd == "memory":
        try:
            import memory_hub
        except ImportError as exc:
            print(f"Error: memory_hub tool not available: {exc}")
            sys.exit(1)
        hub = memory_hub.MemoryHub(timeout=2)
        if not hub.is_available():
            print("[!] Memory gateway not running. Auto-starting...")
            try:
                import subprocess as _sp
                _sp.run([sys.executable, str(_THIS_DIR / "tools" / "ensure_gateway.py")],
                        timeout=15, capture_output=True)
            except (OSError, _sp.TimeoutExpired, _sp.SubprocessError):
                pass
            if hub.is_available():
                print("[OK] Gateway now live.\n")
            else:
                print("    Run `python tools/bootstrap_memory.py` or")
                print("    `python tools/memory_gateway.py --daemon`\n")
        memory_hub.main()
    elif cmd == "bootstrap-memory":
        try:
            import bootstrap_memory
            sys.exit(bootstrap_memory.main())
        except ImportError as exc:
            print(f"Error: bootstrap_memory tool not available: {exc}")
            sys.exit(1)
    else:
        print(f"Unknown command: '{cmd}'. Run 'python aicompass.py help' for usage.")
        sys.exit(1)


if __name__ == "__main__":
    main()
