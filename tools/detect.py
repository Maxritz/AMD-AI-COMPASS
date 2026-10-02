#!/usr/bin/env python3
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
"""
Detect AMD CPU and GPU hardware for LLM serving and development.
Supports Windows 11 and Linux.

Usage:
    python3 scripts/detect.py
    python3 scripts/detect.py --host user@hostname

Output: JSON with cpu_model, vendor, is_amd_epyc, is_amd_zen, zen_generation,
logical_cores, physical_cores, sockets, threads_per_core, numa_nodes, memory_gb,
os_family, gpus, gpu_count.
Exits 0 on success, 1 on failure.
"""

import argparse
import json
import os
import platform
import re
import subprocess
import sys


def _is_local(host):
    return not host or host in ("local", "localhost", "127.0.0.1")


def _run(cmd, host, user, port, timeout=20):
    if _is_local(host):
        # cmd can be a list or string
        r = subprocess.run(cmd, shell=isinstance(cmd, str), stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, timeout=timeout)
    else:
        # Remote execution over SSH (Linux only path)
        ssh_target = f"{user}@{host}" if user else host
        cmd_str = cmd if isinstance(cmd, str) else " ".join(cmd)
        ssh = ["ssh", "-o", "StrictHostKeyChecking=accept-new",
               "-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
               "-o", "LogLevel=ERROR", "-p", str(port), ssh_target, cmd_str]
        r = subprocess.run(ssh, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           text=True, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


def _lscpu_field(lscpu_out, label):
    m = re.search(rf"^{re.escape(label)}:\s*(.+)$", lscpu_out, re.MULTILINE)
    return m.group(1).strip() if m else ""


def _classify_amd_zen_arch(model):
    """Map AMD CPU model name to (generation, zen_arch)."""
    model_upper = model.upper()
    if "EPYC" in model_upper:
        m = re.search(r"EPYC\s+(\d{4})", model_upper)
        if m:
            num = m.group(1)
            last = num[3]
            if num[0] == "7" and last == "3":
                return "Milan", "Zen3"
            elif num[0] == "7" and last == "2":
                return "Rome", "Zen2"
            elif num[0] == "7" and last == "1":
                return "Naples", "Zen1"
            elif num[0] == "8" and last == "4":
                return "Siena", "Zen4c"
            elif num[0] == "9":
                if num.startswith("97") and last == "4":
                    return "Bergamo", "Zen4c"
                if last == "4":
                    return "Genoa", "Zen4"
                if last == "5":
                    return "Turin", "Zen5"
    elif "RYZEN" in model_upper or "THREADRIPPER" in model_upper:
        m = re.search(r"\b(\d{4})\b", model_upper)
        if m:
            num = m.group(1)
            if num.startswith("5"):
                return "Vermeer/Cezanne", "Zen3"
            elif num.startswith("7") or num.startswith("8"):
                return "Raphael/Phoenix", "Zen4"
            elif num.startswith("9"):
                return "Granite Ridge", "Zen5"
            elif num.startswith("3") or num.startswith("4"):
                return "Matisse/Renoir", "Zen2"
            elif num.startswith("1") or num.startswith("2"):
                return "Summit Ridge/Pinnacle Ridge", "Zen1"
    return "unknown", "unknown"


def _gpu_info_windows():
    rc, out, _ = _run([
        "powershell", "-NoProfile", "-Command",
        "Get-CimInstance Win32_VideoController | "
        "Select-Object -Property Name,PNPDeviceID,DriverVersion | "
        "ConvertTo-Json -Compress"
    ], "localhost", "", 0)
    if rc != 0 or not out.strip():
        return []
    try:
        data = json.loads(out)
        return [data] if isinstance(data, dict) else data
    except Exception:
        return []


def _classify_gpu_architecture(name, pnp):
    n = name.lower()
    p = pnp.lower()
    is_amd = "ven_1002" in p or "amd" in n or "radeon" in n
    if not is_amd:
        return "unknown", "unknown", False

    # Check RDNA 4
    if "rx 9" in n or "navi 4" in n:
        return "gfx1201", "RDNA4", True
    # Check RDNA 3.5
    if any(k in n for k in ("ryzen ai max", "strix halo", "8050s", "8060s", "8045s", "880m", "890m")):
        return "gfx1151", "RDNA3.5", True
    # Check RDNA 3
    if "rx 7" in n or any(k in n for k in ("780m", "760m", "740m", "phoenix", "hawk point", "strix point")):
        return "gfx1103", "RDNA3", True
     # Check RDNA 2
    if "rx 6" in n:
        if "6700" in n or "6750" in n:
            return "gfx1031", "RDNA2", True
        if "6600" in n:
            return "gfx1032", "RDNA2", True
        if "6800" in n or "6900" in n:
            return "gfx1030", "RDNA2", True
        return "gfx1030", "RDNA2", True
    if any(k in n for k in ("680m", "660m", "rembrandt")):
        return "gfx1032", "RDNA2", True
    
    return "unknown", "unknown", True


def detect_hardware(host="", user="", port=0):
    if "@" in host:
        user, host = host.split("@", 1)
    host = host or os.environ.get("ZEN_SSH_HOST", "")
    user = user or os.environ.get("ZEN_SSH_USER", "")
    port = port or int(os.environ.get("ZEN_SSH_PORT", "22"))

    os_family = platform.system().lower()

    if os_family == "windows" and _is_local(host):
        # Windows Native Path
        rc, out, err = _run([
            "powershell", "-NoProfile", "-Command",
            "Get-CimInstance Win32_Processor | "
            "Select-Object -Property Name,Manufacturer,NumberOfCores,NumberOfLogicalProcessors | "
            "ConvertTo-Json -Compress"
        ], host, user, port)
        if rc != 0 or not out.strip():
            raise RuntimeError(f"Get-CimInstance Win32_Processor failed: {err.strip()}")

        try:
            cpu_data = json.loads(out)
            if isinstance(cpu_data, list):
                cpu_data = cpu_data[0]
        except Exception as e:
            raise RuntimeError(f"Failed to parse CPU info: {str(e)}")

        model = cpu_data.get("Name") or "unknown"
        mfg = cpu_data.get("Manufacturer") or ""
        vendor = "AuthenticAMD" if "AMD" in mfg or "AMD" in model.upper() else "unknown"

        logical = cpu_data.get("NumberOfLogicalProcessors", 1)
        physical = cpu_data.get("NumberOfCores", 1)
        sockets = 1  # Standard consumer Windows 11 machines
        threads_per_core = logical // physical if physical else 1

        # Query total RAM
        rc_mem, out_mem, _ = _run([
            "powershell", "-NoProfile", "-Command",
            "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"
        ], host, user, port)
        try:
            memory_gb = int(out_mem.strip()) // (1024 ** 3) if rc_mem == 0 else 0
        except Exception:
            memory_gb = 0

        numa_nodes = 1
        avx512 = False

        # GPU detection on Windows
        win_gpus = _gpu_info_windows()
        gpus = []
        for g in win_gpus:
            gname = g.get("Name") or ""
            gpnp = g.get("PNPDeviceID") or ""
            gversion = g.get("DriverVersion") or "unknown"
            gfx, gen, is_amd = _classify_gpu_architecture(gname, gpnp)
            if is_amd:
                gpus.append({
                    "name": gname,
                    "pnp_device_id": gpnp,
                    "driver_version": gversion,
                    "gfx_target": gfx,
                    "architecture": gen
                })

    else:
        # Linux Path (Local or Remote)
        rc, lscpu_out, err = _run("lscpu", host, user, port)
        if rc != 0 or not lscpu_out:
            raise RuntimeError(f"lscpu failed: {err.strip() or f'exit {rc}'}")

        model = _lscpu_field(lscpu_out, "Model name") or "unknown"
        vendor = _lscpu_field(lscpu_out, "Vendor ID")

        def _int(label, default=0):
            v = _lscpu_field(lscpu_out, label)
            try:
                return int(v)
            except ValueError:
                return default

        sockets = _int("Socket(s)", 1)
        cores_per_socket = _int("Core(s) per socket", 0)
        threads_per_core = _int("Thread(s) per core", 1) or 1
        numa_nodes = _int("NUMA node(s)", 1)

        rc, nproc_out, _ = _run("nproc --all", host, user, port)
        try:
            logical = int(nproc_out.strip())
        except (ValueError, AttributeError):
            logical = sockets * cores_per_socket * threads_per_core

        physical = sockets * cores_per_socket if cores_per_socket else logical // threads_per_core

        rc, mem_out, _ = _run("grep MemTotal /proc/meminfo", host, user, port)
        mem_kb = 0
        m = re.search(r"(\d+)", mem_out or "")
        if m:
            mem_kb = int(m.group(1))
        memory_gb = mem_kb // (1024 * 1024)
        avx512 = "avx512f" in _lscpu_field(lscpu_out, "Flags").split()

        # GPU detection on Linux
        gpus = []
        rc_gpu, lspci_out, _ = _run("lspci -nn", host, user, port)
        if rc_gpu == 0:
            for line in lspci_out.splitlines():
                if "VGA" in line or "3D" in line or "Display" in line:
                    is_amd = "1002" in line or "Advanced Micro Devices" in line or "AMD" in line
                    if is_amd:
                        m = re.search(r"\[1002:([\da-fA-F]{4})\]", line)
                        pnp = f"PCI\\VEN_1002&DEV_{m.group(1)}" if m else "PCI\\VEN_1002"
                        gfx, gen, _ = _classify_gpu_architecture(line, pnp)
                        gpus.append({
                            "name": line.strip(),
                            "pnp_device_id": pnp,
                            "gfx_target": gfx,
                            "architecture": gen
                        })

    generation, zen_arch = _classify_amd_zen_arch(model)
    is_epyc = vendor == "AuthenticAMD" and "EPYC" in model.upper()
    is_zen = zen_arch != "unknown"

    return {
        "cpu_model": model,
        "vendor": vendor,
        "is_amd_epyc": is_epyc,
        "is_amd_zen": is_zen,
        "epyc_generation": generation,
        "zen_arch": zen_arch,
        "avx512": avx512,
        "logical_cores": logical,
        "physical_cores": physical,
        "sockets": sockets,
        "threads_per_core": threads_per_core,
        "numa_nodes": numa_nodes,
        "memory_gb": memory_gb,
        "os_family": os_family,
        "gpus": gpus,
        "gpu_count": len(gpus),
        "target": "local" if _is_local(host) else host,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--host", default="", help="[user@]host (default: local or ZEN_SSH_HOST)")
    p.add_argument("--user", default="")
    p.add_argument("--port", type=int, default=0)
    args = p.parse_args()

    try:
        res = detect_hardware(args.host, args.user, args.port)
        print(json.dumps(res, indent=2))
        sys.exit(0)
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)


if __name__ == "__main__":
    main()

