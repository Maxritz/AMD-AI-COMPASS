###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Configuration module for Magpie.

This module defines all configuration classes and enums used throughout
the evaluation framework.
"""

from .pipeline import (
    KernelType,
    EvalMode,
    CompilingConfig,
    PipelineConfig,
)
from .kernel import (
    KernelEvalConfig,
)
from .correctness import (
    CorrectnessMode,
    CorrectnessBackend,
    CorrectnessConfig,
    AccordoConfig,
    AlgorithmThresholds,
)
from .performance import (
    PerfBackend,
    PerformanceConfig,
    RocprofComputeConfig,
    NcuConfig,
    MetrixConfig,
    ROCPROF_KEY_METRICS,
    METRIX_KEY_METRICS,
    DEFAULT_ROCPROF_METRIC_BLOCKS,
)

__all__ = [
    # Pipeline configuration
    "KernelType",
    "EvalMode",
    "CompilingConfig",
    "PipelineConfig",
    # Kernel evaluation configuration
    "KernelEvalConfig",
    # Correctness configuration
    "CorrectnessMode",
    "CorrectnessBackend",
    "CorrectnessConfig",
    "AccordoConfig",
    "AlgorithmThresholds",
    # Performance configuration
    "PerfBackend",
    "PerformanceConfig",
    "RocprofComputeConfig",
    "NcuConfig",
    "MetrixConfig",
    "ROCPROF_KEY_METRICS",
    "METRIX_KEY_METRICS",
    "DEFAULT_ROCPROF_METRIC_BLOCKS",
]
