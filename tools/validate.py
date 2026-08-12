#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""
Validate the environment before serving or developing LLM engines on AMD hardware.
Supports Windows 11 and Linux.

Checks:
- Container runtime (docker, podman)
- Python environment (vllm, zentorch, or compile/build tools on Windows)
- Development/porting tools (git, cmake, MSVC compiler for DirectX/DirectML porting)
- DirectML/DirectX 12 availability on Windows
- HF_TOKEN presence
- RAM size

JSON output to stdout. Exits 0 on success, 1 on failure.
"""

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys


def _sh(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, timeout=timeout)
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return 1, "", f"timed out after {timeout}s"


def _detect_runtime():
    if shutil.which("docker"):
        rc, _, err = _sh("docker ps -q")
        if rc == 0:
            return "docker", "docker reachable"
        last = (err or "docker ps failed").splitlines()[0][:120]
    else:
        last = "docker not installed"
    if shutil.which("podman"):
        rc, _, err = _sh("podman info --format '{{.Host.Arch}}'")
        if rc == 0:
            return "podman", "podman available (rootless)"
        last = (err or last).splitlines()[0][:120] if err else last
    return None, last


def _get_ram_gb():
    os_name = platform.system().lower()
    if os_name == "windows":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return int(stat.ullTotalPhys // (1024 ** 3))
        except Exception:
            pass
        return 0
    else:
        rc, out, _ = _sh("grep MemTotal /proc/meminfo | awk '{print int($2/1024/1024)}'")
        try:
            return int(out)
        except ValueError:
            return 0


def run_validation(image=""):
    issues = []
    os_name = platform.system().lower()

    # 1. Container runtime (prerequisite): docker > podman, else conda/host fallback.
    runtime, detail = _detect_runtime()
    conda_ok = _sh('python -c "import vllm, zentorch"')[0] == 0

    if runtime is None:
        if conda_ok:
            issues.append({"check": "container_runtime", "severity": "warning",
                           "message": f"No accessible container runtime ({detail}); using the conda/host path.",
                           "fix": "For the container path, make docker accessible or install rootless podman."})
        else:
            severity = "warning" if os_name == "windows" else "error"
            issues.append({"check": "container_runtime", "severity": severity,
                           "message": f"No accessible container runtime ({detail}) and no host vllm+zentorch.",
                           "fix": "On Windows: Install Docker Desktop or build from source. On Linux: add your user to the docker group."})

    # 2. Image present check (applicable for container runtimes)
    if runtime and image:
        repo = image.rsplit(":", 1)[0]
        rc, out, _ = _sh(f"{runtime} images {repo} --format '{{{{.Repository}}}}:{{{{.Tag}}}}'")
        if image not in (out or ""):
            issues.append({"check": "image", "severity": "advisory",
                           "message": f"Image {image} not pulled yet; first launch will download it.",
                           "fix": f"{runtime} pull {image}"})
        else:
            rc, ver, err = _sh(f'{runtime} run --rm {image} '
                               f'python -c "import vllm,zentorch;print(vllm.__version__,zentorch.__version__)"', timeout=90)
            if rc == 0 and ver:
                issues.append({"check": "image_stack", "severity": "advisory",
                               "message": f"Image has vllm+zentorch ({ver})."})
            else:
                issues.append({"check": "image_stack", "severity": "warning",
                               "message": f"Image {image} is present but `import vllm, zentorch` failed inside it: {(err or 'unknown')[:120]}",
                               "fix": "Use an image tag that bundles the zentorch plugin."})

    # 3. Host vllm+zentorch (for the conda path)
    if conda_ok:
        _, ver, _ = _sh('python -c "import vllm,zentorch;print(vllm.__version__,zentorch.__version__)"')
        issues.append({"check": "host_stack", "severity": "advisory",
                       "message": f"Host vllm+zentorch importable ({ver}); conda path available."})
    elif runtime:
        issues.append({"check": "host_stack", "severity": "advisory",
                       "message": "Host `import vllm, zentorch` not available; use the container path."})

    # 4. Windows 11 Build & Development Tools validation
    if os_name == "windows":
        # Check compiler tools
        git_val = shutil.which("git")
        cmake_val = shutil.which("cmake")
        cl_val = shutil.which("cl") or shutil.which("msbuild")

        if not git_val:
            issues.append({"check": "git", "severity": "warning",
                           "message": "git is not in PATH. Required for cloning repositories during porting.",
                           "fix": "Install Git for Windows and add it to your PATH."})
        else:
            issues.append({"check": "git", "severity": "advisory",
                           "message": "Git is available."})

        if not cmake_val:
            issues.append({"check": "cmake", "severity": "warning",
                           "message": "cmake is not in PATH. Required for building llama.cpp (DirectML) or other inference engines.",
                           "fix": "Install CMake and add it to your PATH."})
        else:
            issues.append({"check": "cmake", "severity": "advisory",
                           "message": "CMake is available."})

        if not cl_val:
            issues.append({"check": "msvc_compiler", "severity": "warning",
                           "message": "MSVC C++ compiler (cl.exe) or MSBuild is not detected in PATH.",
                           "fix": "Install Visual Studio 2022 (with 'Desktop development with C++' workload) and run from the Developer Command Prompt."})
        else:
            issues.append({"check": "msvc_compiler", "severity": "advisory",
                           "message": "MSVC Build Tools are available."})

        # Check DirectML.dll
        sysroot = os.environ.get("SystemRoot") or r"C:\Windows"
        dml_path = os.path.join(sysroot, "System32", "DirectML.dll")
        if not os.path.exists(dml_path):
            issues.append({"check": "directml", "severity": "warning",
                           "message": "DirectML.dll was not found in System32. DirectX-based AI acceleration may not be functional.",
                           "fix": "Ensure you are on Windows 11 with the latest graphics drivers installed."})
        else:
            issues.append({"check": "directml", "severity": "advisory",
                           "message": f"DirectML.dll is present at {dml_path} (DirectX 12 backend supported)."})

        # Check HAGS and TDR via Windows Registry
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\GraphicsDrivers") as key:
                try:
                    hags, _ = winreg.QueryValueEx(key, "HwSchMode")
                    if hags == 2:
                        issues.append({"check": "hags", "severity": "advisory",
                                       "message": "Hardware-Accelerated GPU Scheduling (HAGS) is enabled."})
                    else:
                        issues.append({"check": "hags", "severity": "warning",
                                       "message": f"Hardware-Accelerated GPU Scheduling (HAGS) is disabled (value: {hags}). Performance may suffer.",
                                       "fix": "Enable HAGS in Windows Graphics Settings."})
                except FileNotFoundError:
                    issues.append({"check": "hags", "severity": "warning",
                                   "message": "Hardware-Accelerated GPU Scheduling (HAGS) registry entry not found (likely disabled).",
                                   "fix": "Enable HAGS in Windows Graphics Settings."})

                try:
                    tdr_level, _ = winreg.QueryValueEx(key, "TdrLevel")
                    if tdr_level == 0:
                        issues.append({"check": "tdr", "severity": "advisory",
                                       "message": "TDR (Timeout Detection and Recovery) is disabled (TdrLevel=0). Safe from timeout crashes."})
                    else:
                        issues.append({"check": "tdr", "severity": "warning",
                                       "message": f"TDR (Timeout Detection and Recovery) is enabled (TdrLevel={tdr_level}). Long model dispatches may cause Windows to reset the graphics driver.",
                                       "fix": "Set TdrLevel registry value to 0 to disable TDR or increase TdrDelay."})
                except FileNotFoundError:
                    issues.append({"check": "tdr", "severity": "advisory",
                                   "message": "TDR (Timeout Detection and Recovery) is enabled (default Windows behavior)."})
        except Exception as reg_err:
            issues.append({"check": "registry_checks", "severity": "warning",
                           "message": f"Failed to query graphics registry keys: {str(reg_err)}",
                           "fix": "Ensure you are running with sufficient permissions to query HKLM registry."})


    # 5. HF_TOKEN
    if not os.environ.get("HF_TOKEN"):
        issues.append({"check": "hf_token", "severity": "advisory",
                       "message": "HF_TOKEN not set. Required for gated models (Llama, Gemma).",
                       "fix": "On Windows (PowerShell): $env:HF_TOKEN='hf_...'; On Linux: export HF_TOKEN=hf_..."})

    # 6. RAM
    ram_gb = _get_ram_gb()
    if 0 < ram_gb < 32:
        issues.append({"check": "ram", "severity": "warning",
                       "message": f"Only {ram_gb} GB RAM. Model weights + KV cache are stored in RAM; large models may not fit.",
                       "fix": "Use a smaller model or increase host RAM."})

    # 7. Perf libraries for Linux host/conda path
    if os_name == "linux" and conda_ok:
        ld = os.environ.get("LD_PRELOAD", "")
        missing = [lib for lib in ("libtcmalloc", "libiomp") if lib not in ld]
        if missing:
            issues.append({"check": "perf_libs", "severity": "advisory",
                           "message": f"LD_PRELOAD is missing {', '.join(missing)}; vLLM CPU performance may suffer.",
                           "fix": "export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libtcmalloc_minimal.so.4:$CONDA_PREFIX/lib/libiomp5.so:$LD_PRELOAD"})

    # 8. CUDA_VISIBLE_DEVICES footgun — empty string hides all GPUs from ROCm
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        if cvd == "":
            issues.append({"check": "cuda_visible_devices", "severity": "error",
                           "message": "CUDA_VISIBLE_DEVICES is set to '' (empty string). This hides ALL GPUs from the ROCm/HIP runtime.",
                           "fix": "On Windows (PowerShell): Remove-Item Env:CUDA_VISIBLE_DEVICES; On Linux: unset CUDA_VISIBLE_DEVICES"})
        else:
            issues.append({"check": "cuda_visible_devices", "severity": "advisory",
                           "message": f"CUDA_VISIBLE_DEVICES={cvd!r}. ROCm maps this to HIP_VISIBLE_DEVICES. Only the listed GPUs are visible.",
                           "fix": "unset CUDA_VISIBLE_DEVICES to use all GPUs"})

    errors = [i for i in issues if i["severity"] == "error"]
    result = {
        "ready": len(errors) == 0,
        "runtime": runtime,
        "runtime_detail": detail,
        "conda_path_available": conda_ok,
        "ram_gb": ram_gb,
        "errors": errors,
        "warnings": [i for i in issues if i["severity"] == "warning"],
        "advisories": [i for i in issues if i["severity"] == "advisory"],
    }
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", default="", help="container image to check for (advisory)")
    args = p.parse_args()

    result = run_validation(args.image)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["ready"] else 1)



if __name__ == "__main__":
    main()


