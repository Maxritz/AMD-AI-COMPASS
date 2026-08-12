###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
GPU monitoring for benchmark mode.

Provides background monitoring of GPU temperature, frequency, and power
during benchmark execution.
"""

import logging
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class GPUSample:
    """Single GPU sample data point."""
    timestamp: float
    temperature_c: float = 0.0
    gpu_clock_mhz: int = 0
    mem_clock_mhz: int = 0
    power_watts: float = 0.0


@dataclass
class GPUMonitorStats:
    """GPU monitoring statistics computed from samples."""
    sample_count: int = 0
    duration_sec: float = 0.0
    
    # Temperature statistics (Celsius)
    temp_min: float = 0.0
    temp_max: float = 0.0
    temp_avg: float = 0.0
    
    # GPU clock statistics (MHz)
    gpu_clock_min: int = 0
    gpu_clock_max: int = 0
    gpu_clock_avg: float = 0.0
    
    # Memory clock statistics (MHz)
    mem_clock_min: int = 0
    mem_clock_max: int = 0
    mem_clock_avg: float = 0.0
    
    # Power statistics (Watts)
    power_min: float = 0.0
    power_max: float = 0.0
    power_avg: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "sample_count": self.sample_count,
            "duration_sec": round(self.duration_sec, 2),
            "temperature_c": {
                "min": round(self.temp_min, 1),
                "max": round(self.temp_max, 1),
                "avg": round(self.temp_avg, 1),
            },
            "gpu_clock_mhz": {
                "min": self.gpu_clock_min,
                "max": self.gpu_clock_max,
                "avg": round(self.gpu_clock_avg, 1),
            },
            "mem_clock_mhz": {
                "min": self.mem_clock_min,
                "max": self.mem_clock_max,
                "avg": round(self.mem_clock_avg, 1),
            },
            "power_watts": {
                "min": round(self.power_min, 1),
                "max": round(self.power_max, 1),
                "avg": round(self.power_avg, 1),
            },
        }


class GPUMonitor:
    """
    Background GPU monitor for collecting hardware metrics during benchmark.
    
    Collects temperature, GPU/memory clock frequencies, and power consumption
    at regular intervals using rocm-smi (AMD) or nvidia-smi (NVIDIA).
    
    Example:
        monitor = GPUMonitor(device_id=0, interval_sec=2.0)
        monitor.start()
        
        # ... run benchmark ...
        
        stats = monitor.stop()
        print(f"Avg temp: {stats.temp_avg}°C, Max power: {stats.power_max}W")
    """
    
    def __init__(
        self,
        device_id: int = 0,
        interval_sec: float = 2.0,
        vendor: str = "auto",
    ):
        """
        Initialize GPU monitor.
        
        Args:
            device_id: GPU device index to monitor
            interval_sec: Sampling interval in seconds (default 2.0)
            vendor: GPU vendor ("amd", "nvidia", or "auto" for detection)
        """
        self.device_id = device_id
        self.interval = max(0.5, interval_sec)  # Minimum 0.5s
        self.vendor = self._detect_vendor() if vendor == "auto" else vendor.lower()
        
        self._samples: List[GPUSample] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: float = 0.0
        self._lock = threading.Lock()
    
    def _detect_vendor(self) -> str:
        """Auto-detect GPU vendor."""
        try:
            result = subprocess.run(
                ["rocm-smi", "--showid"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and "GPU" in result.stdout:
                return "amd"
        except Exception:
            pass
        
        try:
            result = subprocess.run(
                ["nvidia-smi", "-L"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0 and "GPU" in result.stdout:
                return "nvidia"
        except Exception:
            pass
        
        return "unknown"
    
    def start(self) -> bool:
        """
        Start background monitoring.
        
        Returns:
            True if monitoring started successfully, False if skipped (unsupported vendor)
        """
        if self._running:
            logger.warning("GPU monitor already running")
            return True
        
        # Only AMD GPUs are supported currently
        if self.vendor != "amd":
            logger.warning(
                f"GPU monitor: vendor '{self.vendor}' not supported, skipping. "
                "Currently only AMD GPUs (rocm-smi) are supported."
            )
            return False
        
        self._samples.clear()
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        logger.debug(f"GPU monitor started (device={self.device_id}, interval={self.interval}s)")
        return True
    
    def stop(self) -> GPUMonitorStats:
        """
        Stop monitoring and return statistics.
        
        Returns:
            GPUMonitorStats with computed min/max/avg values
        """
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=self.interval + 1.0)
        
        stats = self._compute_stats()
        logger.debug(f"GPU monitor stopped: {stats.sample_count} samples collected")
        return stats
    
    def _monitor_loop(self) -> None:
        """Background monitoring loop."""
        while self._running:
            sample = self._collect_sample()
            if sample:
                with self._lock:
                    self._samples.append(sample)
            time.sleep(self.interval)
    
    def _collect_sample(self) -> Optional[GPUSample]:
        """Collect a single GPU sample."""
        try:
            if self.vendor == "amd":
                return self._collect_amd_sample()
            elif self.vendor == "nvidia":
                return self._collect_nvidia_sample()
            else:
                return None
        except Exception as e:
            logger.debug(f"Failed to collect GPU sample: {e}")
            return None
    
    def _collect_amd_sample(self) -> Optional[GPUSample]:
        """Collect sample from AMD GPU using rocm-smi."""
        sample = GPUSample(timestamp=time.time())
        
        try:
            # Collect all data in one command for efficiency
            result = subprocess.run(
                ["rocm-smi", "-d", str(self.device_id),
                 "--showtemp", "--showclocks", "--showpower"],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode != 0:
                return None
            
            output = result.stdout
            
            # Parse temperature (junction)
            temp_match = re.search(
                r"Temperature.*junction.*?(\d+\.?\d*)",
                output, re.IGNORECASE
            )
            if temp_match:
                sample.temperature_c = float(temp_match.group(1))
            
            # Parse GPU clock (sclk)
            sclk_match = re.search(
                r"sclk.*?(\d+)\s*Mhz",
                output, re.IGNORECASE
            )
            if sclk_match:
                sample.gpu_clock_mhz = int(sclk_match.group(1))
            
            # Parse memory clock (mclk)
            mclk_match = re.search(
                r"mclk.*?(\d+)\s*Mhz",
                output, re.IGNORECASE
            )
            if mclk_match:
                sample.mem_clock_mhz = int(mclk_match.group(1))
            
            # Parse power (various formats)
            # "Current Socket Graphics Package Power (W): 145.0"
            # "Average Graphics Package Power (W): 145.0"
            power_match = re.search(
                r"(?:Current|Average).*?Power.*?:\s*(\d+\.?\d*)",
                output, re.IGNORECASE
            )
            if not power_match:
                # Fallback: any number followed by W
                power_match = re.search(r"(\d+\.?\d*)\s*W", output)
            if power_match:
                sample.power_watts = float(power_match.group(1))
            
            return sample
            
        except subprocess.TimeoutExpired:
            logger.debug("rocm-smi timed out")
            return None
        except Exception as e:
            logger.debug(f"Error collecting AMD sample: {e}")
            return None
    
    def _collect_nvidia_sample(self) -> Optional[GPUSample]:
        """Collect sample from NVIDIA GPU using nvidia-smi."""
        sample = GPUSample(timestamp=time.time())
        
        try:
            result = subprocess.run(
                ["nvidia-smi", "-i", str(self.device_id),
                 "--query-gpu=temperature.gpu,clocks.gr,clocks.mem,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            
            if result.returncode != 0:
                return None
            
            # Parse CSV output: temp, gpu_clock, mem_clock, power
            values = result.stdout.strip().split(",")
            if len(values) >= 4:
                sample.temperature_c = float(values[0].strip())
                sample.gpu_clock_mhz = int(values[1].strip())
                sample.mem_clock_mhz = int(values[2].strip())
                sample.power_watts = float(values[3].strip())
            
            return sample
            
        except subprocess.TimeoutExpired:
            logger.debug("nvidia-smi timed out")
            return None
        except Exception as e:
            logger.debug(f"Error collecting NVIDIA sample: {e}")
            return None
    
    def _compute_stats(self) -> GPUMonitorStats:
        """Compute statistics from collected samples."""
        with self._lock:
            samples = list(self._samples)
        
        if not samples:
            return GPUMonitorStats()
        
        temps = [s.temperature_c for s in samples if s.temperature_c > 0]
        gpu_clocks = [s.gpu_clock_mhz for s in samples if s.gpu_clock_mhz > 0]
        mem_clocks = [s.mem_clock_mhz for s in samples if s.mem_clock_mhz > 0]
        powers = [s.power_watts for s in samples if s.power_watts > 0]
        
        duration = samples[-1].timestamp - self._start_time if samples else 0.0
        
        stats = GPUMonitorStats(
            sample_count=len(samples),
            duration_sec=duration,
        )
        
        if temps:
            stats.temp_min = min(temps)
            stats.temp_max = max(temps)
            stats.temp_avg = sum(temps) / len(temps)
        
        if gpu_clocks:
            stats.gpu_clock_min = min(gpu_clocks)
            stats.gpu_clock_max = max(gpu_clocks)
            stats.gpu_clock_avg = sum(gpu_clocks) / len(gpu_clocks)
        
        if mem_clocks:
            stats.mem_clock_min = min(mem_clocks)
            stats.mem_clock_max = max(mem_clocks)
            stats.mem_clock_avg = sum(mem_clocks) / len(mem_clocks)
        
        if powers:
            stats.power_min = min(powers)
            stats.power_max = max(powers)
            stats.power_avg = sum(powers) / len(powers)
        
        return stats
    
    @property
    def is_running(self) -> bool:
        """Check if monitor is currently running."""
        return self._running
    
    @property
    def sample_count(self) -> int:
        """Get current number of collected samples."""
        with self._lock:
            return len(self._samples)
