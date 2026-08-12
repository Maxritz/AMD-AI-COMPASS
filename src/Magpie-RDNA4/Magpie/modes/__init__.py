###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Evaluation modes for Magpie.

This module contains different evaluation modes:
- analyze_eval: Analyze individual kernels for correctness, performance, etc.
- compare_eval: Compare two or more kernel implementations
- benchmark: Framework-level benchmark with vLLM/SGLang profiling
"""

from .analyze_eval import AnalyzeMode
from .compare_eval import CompareMode
from .benchmark import BenchmarkMode, BenchmarkConfig, BenchmarkResult

__all__ = [
    "AnalyzeMode",
    "CompareMode",
    "BenchmarkMode",
    "BenchmarkConfig",
    "BenchmarkResult",
]
