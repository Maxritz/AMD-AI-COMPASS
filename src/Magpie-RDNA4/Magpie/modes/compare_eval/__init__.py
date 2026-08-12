###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Compare evaluation mode.

This module provides functionality for comparing two or more GPU kernel
implementations, enabling side-by-side performance and correctness analysis.
"""

from .comparator import CompareMode

__all__ = ["CompareMode"]
