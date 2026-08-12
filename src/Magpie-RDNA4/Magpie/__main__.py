###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Entry point for running Magpie as a module.

Usage:
    python -m Magpie analyze kernel.hip -t "./test.sh"
    python -m Magpie compare kernel1.hip kernel2.hip
"""

import sys
from .main import main

if __name__ == "__main__":
    sys.exit(main())
