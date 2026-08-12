###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Performance evaluation configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import shlex
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .pipeline import KernelType


class PerfBackend(Enum):
    """Performance profiling backend."""

    ROCPROF_COMPUTE = auto()  # rocprof-compute for HIP/AMD GPUs
    NCU = auto()  # NVIDIA Nsight Compute for CUDA
    METRIX = auto()  # IntelliKit Metrix for HIP/AMD GPUs (uses rocprofv3)
    NONE = auto()  # No profiling


# Default metric blocks for fast profiling on gfx942 (MI300 series)
# Block meanings:
#   1: System Info
#   2: System Speed-of-Light (FLOPS, utilization, bandwidth, etc.)
#   5: Command Processor (CPC/CPF)
#  10: Compute Units - Instruction Mix (VALU, VMEM, LDS, MFMA, etc.)
#  11: Compute Units - Compute Pipeline
#  12: Local Data Share (LDS)
#  14: Scalar L1 Data Cache
#  16: Vector L1 Data Cache
#  17: L2 Cache
DEFAULT_ROCPROF_METRIC_BLOCKS = ["1", "2", "5", "10", "11", "12", "14", "16", "17"]


# Key metrics to extract from rocprof-compute analyze output
# Format: {csv_metric_name: display_name}
ROCPROF_KEY_METRICS = {
    # Speed of Light metrics
    "VALU FLOPs": "VALU_FLOPs",
    "VALU IOPs": "VALU_IOPs",
    "MFMA FLOPs (F16)": "MFMA_FLOPs_F16",
    "MFMA FLOPs (BF16)": "MFMA_FLOPs_BF16",
    "MFMA FLOPs (F32)": "MFMA_FLOPs_F32",
    "MFMA FLOPs (F64)": "MFMA_FLOPs_F64",
    "MFMA FLOPs (F8)": "MFMA_FLOPs_F8",
    "MFMA IOPs (Int8)": "MFMA_IOPs_Int8",
    "SALU Utilization": "SALU_Util",
    "VALU Utilization": "VALU_Util",
    "MFMA Utilization": "MFMA_Util",
    "VMEM Utilization": "VMEM_Util",
    "Branch Utilization": "Branch_Util",
    "IPC": "IPC",
    "Wavefront Occupancy": "Wavefront_Occ",
    "CU Utilization": "CU_Util",
    # Memory metrics
    "Theoretical LDS Bandwidth": "LDS_BW",
    "LDS Bank Conflicts/Access": "LDS_Conflicts",
    "vL1D Cache Hit Rate": "vL1D_HitRate",
    "vL1D Cache BW": "vL1D_BW",
    "L2 Cache Hit Rate": "L2_HitRate",
    "L2 Cache BW": "L2_BW",
    "L2-Fabric Read BW": "HBM_Read_BW",
    "L2-Fabric Write BW": "HBM_Write_BW",
    # Instruction mix
    "VALU": "Inst_VALU",
    "VMEM": "Inst_VMEM",
    "LDS": "Inst_LDS",
    "MFMA": "Inst_MFMA",
    "SALU": "Inst_SALU",
    "SMEM": "Inst_SMEM",
}


METRIX_KEY_METRICS = {
    "memory.hbm_read_bandwidth": "HBM_Read_BW",
    "memory.hbm_write_bandwidth": "HBM_Write_BW",
    "memory.hbm_bandwidth_utilization": "HBM_BW_Util",
    "memory.bytes_transferred_hbm": "HBM_Bytes",
    "memory.bytes_transferred_l2": "L2_Bytes",
    "memory.bytes_transferred_l1": "L1_Bytes",
    "memory.l1_hit_rate": "L1_HitRate",
    "memory.l2_hit_rate": "L2_HitRate",
    "memory.l2_bandwidth": "L2_BW",
    "memory.coalescing_efficiency": "Coalescing_Eff",
    "memory.global_load_efficiency": "Global_Load_Eff",
    "memory.global_store_efficiency": "Global_Store_Eff",
    "memory.lds_bank_conflicts": "LDS_Conflicts",
    "memory.atomic_latency": "Atomic_Latency",
    "compute.total_flops": "Total_FLOPs",
    "compute.hbm_gflops": "HBM_GFLOPS",
    "compute.hbm_arithmetic_intensity": "HBM_Arith_Intensity",
    "compute.l2_arithmetic_intensity": "L2_Arith_Intensity",
    "compute.l1_arithmetic_intensity": "L1_Arith_Intensity",
}


@dataclass
class MetrixConfig:
    """
    Configuration for IntelliKit Metrix profiler.

    Metrix wraps rocprofv3 and provides human-readable GPU metrics.
    CLI: ``metrix profile [options] -- <command>``

    Attributes:
        profile: Preset profile name ("quick", "memory", "compute") or None
        metrics: Specific metric names to collect, e.g. ["memory.l2_hit_rate"]
        kernel_filter: Regex to filter kernels by name
        num_replays: Number of profiling runs for min/max/avg statistics
        timeout_seconds: Timeout per profiling run in seconds
        extra_args: Additional CLI arguments passed to ``metrix profile``
    """

    output_dir: Optional[str] = None
    profile: Optional[str] = None
    metrics: List[str] = field(default_factory=list)
    kernel_filter: Optional[str] = None
    num_replays: int = 1
    timeout_seconds: int = 60
    extra_args: List[str] = field(default_factory=list)

    def get_profile_args(self, output_path: str) -> List[str]:
        """Build ``metrix profile`` command arguments.

        Args:
            output_path: Path for the JSON output file.

        Returns:
            List of CLI arguments (without the leading ``metrix profile``).
        """
        args: List[str] = []

        args.extend(["--output", output_path])
        args.extend(["--num-replays", str(self.num_replays)])
        args.extend(["--timeout", str(self.timeout_seconds)])
        args.append("--aggregate")

        if self.profile:
            args.extend(["--profile", self.profile])

        if self.metrics:
            args.extend(["--metrics", ",".join(self.metrics)])

        if self.kernel_filter:
            args.extend(["--kernel", self.kernel_filter])

        args.extend(self.extra_args)

        return args


@dataclass
class RocprofComputeConfig:
    """
    Configuration for rocprof-compute profiler.

    rocprof-compute is a two-stage profiler:
    1. Profile stage: Collect hardware counters
       `rocprof-compute profile -b <blocks> -n <name> -o <workload_dir> -- <command>`
    2. Analyze stage: Process data and generate metrics
       `rocprof-compute analyze -b <blocks> <workload_dir>`

    Attributes:
        workload_dir: Base directory to save profiling workloads
        profile_args: Additional arguments for rocprof-compute profile
        analyze_args: Additional arguments for rocprof-compute analyze
        metric_blocks: Metric block IDs to collect
                      Default: ["1", "2", "5", "10", "11", "12", "14", "16", "17"]
        kernel_filter: Kernel name filter (optional, -k flag)
        dispatch_filter: Dispatch ID filter (optional, -d flag)
        output_format: Output format for analyze (csv, json, etc.)
        target_gpu: Target GPU architecture (e.g., "gfx942", auto-detected if None)
    """

    output_dir: Optional[str] = None
    workload_dir: str = "./workloads"
    profile_args: List[str] = field(default_factory=list)
    analyze_args: List[str] = field(default_factory=list)
    metric_blocks: List[str] = field(
        default_factory=lambda: DEFAULT_ROCPROF_METRIC_BLOCKS.copy()
    )
    kernel_filter: Optional[str] = None
    dispatch_filter: Optional[List[int]] = None
    output_format: str = "csv"
    target_gpu: Optional[str] = None

    def get_profile_args(
        self, workload_name: str, output_dir: Optional[str] = None
    ) -> List[str]:
        """
        Build profile command arguments.

        Args:
            workload_name: Name for this profiling workload
            output_dir: Override output directory (optional)

        Returns:
            List of command line arguments
        """
        args = []

        # Add workload name (required)
        args.extend(["-n", workload_name])

        # Add output path (rocprof-compute uses -p for path)
        out_dir = output_dir or self.workload_dir
        args.extend(["-p", out_dir])

        # Add metric block filters
        if self.metric_blocks:
            args.extend(["-b"] + self.metric_blocks)

        # Add kernel filter
        if self.kernel_filter:
            args.extend(["-k", self.kernel_filter])

        # Add dispatch filter
        if self.dispatch_filter:
            args.extend(["-d"] + [str(d) for d in self.dispatch_filter])

        # Add custom args (split to support single-string entries)
        args.extend(self._expand_args(self.profile_args))

        return args

    def get_analyze_args(
        self, workload_path: str, output_dir: Optional[str] = None
    ) -> List[str]:
        """
        Build analyze command arguments.

        Args:
            workload_path: Path to the workload directory to analyze
            output_dir: Directory to save analysis dataframe CSV files (optional)

        Returns:
            List of command line arguments
        """
        args = []

        # Add workload path (rocprof-compute uses -p for path)
        args.extend(["-p", workload_path])

        # Add metric block filters for analysis
        if self.metric_blocks:
            args.extend(["-b"] + self.metric_blocks)

        if output_dir and self.output_format == "csv":
            args.extend(["--output-format", "csv", "--output-name", output_dir])
        elif self.output_format == "csv":
            args.extend(["--output-format", "csv"])

        # Add custom analyze args (split to support single-string entries)
        args.extend(self._expand_args(self.analyze_args))
        return args

    def _expand_args(self, raw_args: List[str]) -> List[str]:
        """Expand args entries that may contain whitespace into tokens."""
        expanded: List[str] = []
        for arg in raw_args:
            if isinstance(arg, str):
                expanded.extend(shlex.split(arg))
        return expanded

    def get_workload_path(
        self, workload_name: str, base_dir: Optional[str] = None
    ) -> Path:
        """Get the full path for a workload directory."""
        base = Path(base_dir or self.workload_dir)
        return base / workload_name


@dataclass
class NcuConfig:
    """
    Configuration for NVIDIA Nsight Compute (ncu) profiler.

    Attributes:
        args: Additional arguments for ncu
        metrics: Specific metrics to collect
        kernel_filter: Kernel name filter
    """

    args: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    kernel_filter: Optional[str] = None


@dataclass
class PerformanceConfig:
    """
    Configuration for performance evaluation.

    Attributes:
        enabled: Whether to enable performance evaluation
        backend: Profiling backend (auto-selected based on kernel_type/gpu_arch if not specified)
        kernel_type: Kernel type (used for auto-selecting backend)
        gpu_arch: GPU architecture string (e.g. "gfx942", "sm_90") for
                  cross-platform kernels like Triton where the profiler
                  depends on the target GPU, not the source language.
        timeout_seconds: Maximum execution time per profiling run
        profiler_args: Additional arguments for the profiler (legacy, use rocprof_config or ncu_config)
        rocprof_config: Configuration for rocprof-compute
        ncu_config: Configuration for ncu
    """

    enabled: bool = True
    backend: Optional[PerfBackend] = None
    kernel_type: Optional["KernelType"] = None
    gpu_arch: Optional[str] = None
    timeout_seconds: float = 120.0
    profiler_args: List[str] = field(default_factory=list)  # Legacy
    rocprof_config: Optional[RocprofComputeConfig] = None
    ncu_config: Optional[NcuConfig] = None
    metrix_config: Optional[MetrixConfig] = None

    def __post_init__(self):
        # Auto-select backend based on kernel type if not specified
        if self.backend is None and self.kernel_type is not None:
            from .pipeline import KernelType

            if self.kernel_type == KernelType.HIP:
                self.backend = PerfBackend.ROCPROF_COMPUTE
            elif self.kernel_type == KernelType.CUDA:
                self.backend = PerfBackend.NCU
            elif self.kernel_type == KernelType.TRITON:
                self.backend = self._backend_for_gpu_arch()
            else:
                self.backend = PerfBackend.NONE

        # Initialize default configs
        if self.rocprof_config is None:
            self.rocprof_config = RocprofComputeConfig()
        if self.ncu_config is None:
            self.ncu_config = NcuConfig()
        if self.metrix_config is None:
            self.metrix_config = MetrixConfig()

    def _backend_for_gpu_arch(self) -> PerfBackend:
        """Select profiling backend based on detected GPU architecture.

        Triton JIT-compiles to HIP on AMD and CUDA on NVIDIA, so we use the
        same system profiler as native kernels on each platform.
        """
        if self.gpu_arch:
            if self.gpu_arch.startswith("gfx"):
                return PerfBackend.ROCPROF_COMPUTE
            if self.gpu_arch.startswith("sm_"):
                return PerfBackend.NCU
        return PerfBackend.NONE

    def get_backend(self) -> PerfBackend:
        """Get the performance backend."""
        return self.backend or PerfBackend.NONE
