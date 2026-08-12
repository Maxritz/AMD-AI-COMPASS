###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Correctness evaluation configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple


class CorrectnessMode(Enum):
    """Correctness evaluation mode."""

    TESTCASE = auto()  # Run provided testcase command
    RESULT_COMPARISON = auto()  # Compare outputs between kernels (for compare mode)


class CorrectnessBackend(Enum):
    """Correctness evaluation backend.

    TESTCASE: Default -- run testcase commands and check exit codes / output.
    ACCORDO:  Use IntelliKit Accordo for HSA-level kernel output validation.
    """

    TESTCASE = auto()
    ACCORDO = auto()


@dataclass
class AlgorithmThresholds:
    """
    Thresholds for correctness comparison algorithms.
    """

    atol: float = 1e-5  # Absolute tolerance
    rtol: float = 1e-4  # Relative tolerance


@dataclass
class AccordoConfig:
    """
    Configuration for IntelliKit Accordo correctness validation.

    Accordo captures GPU kernel output buffers via HSA interception and
    compares reference vs. optimized snapshots using ``np.allclose``.

    Attributes:
        kernel_name: GPU kernel function name to intercept.
        reference_binary: Command (or path) to the reference binary.
        optimized_binary: Command (or path) to the optimized binary.
        atol: Absolute tolerance for ``np.allclose`` (default ``1e-6``).
        rtol: Relative tolerance (default ``1e-5``, same as ``torch.allclose``).
        equal_nan: If true, pass ``--equal-nan`` to ``accordo validate``; if false, omit it (CLI default).
        timeout_seconds: Timeout per snapshot capture in seconds.
                         *None* means inherit from the task-level timeout.
        kernel_args: Optional manual kernel args as ``[(name, type), ...]``.
                     Auto-extracted from the binary when *None*.
        working_directory: Working directory for binary execution.
        workspace_path: Result workspace directory (set by the pipeline).
                        The ``accordo`` CLI subprocess runs with this as cwd.
    """

    kernel_name: Optional[str] = None
    reference_binary: Optional[str] = None
    optimized_binary: Optional[str] = None
    atol: float = 1e-6
    rtol: float = 1e-5
    equal_nan: bool = False
    timeout_seconds: Optional[int] = None
    kernel_args: Optional[List[Tuple[str, str]]] = None
    working_directory: Optional[str] = None
    workspace_path: Optional[str] = None


@dataclass
class CorrectnessConfig:
    """
    Configuration for correctness evaluation.

    ``mode`` and ``backend`` serve different roles:

    * **mode** — *what* to evaluate:
      - ``TESTCASE``: run a provided testcase command (analyze mode).
      - ``RESULT_COMPARISON``: compare outputs between multiple kernels
        (compare mode).
    * **backend** — *how* to evaluate:
      - ``TESTCASE``: check testcase exit codes / output (default).
      - ``ACCORDO``: use IntelliKit Accordo HSA-level GPU buffer comparison.

    When ``backend == ACCORDO`` the Accordo path runs *before* the mode-specific
    logic, so ``mode`` is effectively bypassed.  ``mode`` still matters when
    ``backend == TESTCASE``.

    Attributes:
        mode: Correctness evaluation mode (TESTCASE or RESULT_COMPARISON).
        backend: Which correctness backend to use (default: TESTCASE).
        testcase_command: Command to run for testcase verification (for analyze mode).
        iteration_count: Number of iterations for correctness testing.
        thresholds: Tolerance thresholds for result comparison.
        check_nan: Whether to check for NaN values.
        check_inf: Whether to check for Inf values.
        accordo_config: Configuration for Accordo backend (used when backend == ACCORDO).
    """

    mode: CorrectnessMode = CorrectnessMode.TESTCASE
    backend: CorrectnessBackend = CorrectnessBackend.TESTCASE
    testcase_command: Optional[List[str]] = None
    iteration_count: int = 1
    thresholds: AlgorithmThresholds = field(default_factory=AlgorithmThresholds)
    check_nan: bool = True
    check_inf: bool = True
    accordo_config: Optional[AccordoConfig] = None

    def has_testcase(self) -> bool:
        """Check if testcase command is provided."""
        return self.testcase_command is not None and len(self.testcase_command) > 0

    @classmethod
    def from_dict(
        cls, cfg: Dict[str, Any], mode: CorrectnessMode = CorrectnessMode.TESTCASE
    ) -> "CorrectnessConfig":
        """Build a ``CorrectnessConfig`` from a plain dict (e.g. from YAML).

        Args:
            cfg: Dict with optional ``backend`` and ``accordo`` keys.
            mode: The correctness *mode* to use (TESTCASE for analyze,
                  RESULT_COMPARISON for compare).  Independent of *backend*.
        """
        if not cfg:
            return cls(mode=mode)

        backend_str = cfg.get("backend", "testcase")
        backend = (
            CorrectnessBackend.ACCORDO
            if backend_str == "accordo"
            else CorrectnessBackend.TESTCASE
        )

        accordo_cfg = None
        if backend == CorrectnessBackend.ACCORDO or "accordo" in cfg:
            acc = cfg.get("accordo", {})
            accordo_cfg = AccordoConfig(
                kernel_name=acc.get("kernel_name"),
                reference_binary=acc.get("reference_binary"),
                optimized_binary=acc.get("optimized_binary"),
                atol=acc.get("atol", 1e-6),
                rtol=acc.get("rtol", 1e-5),
                equal_nan=acc.get("equal_nan", False),
                timeout_seconds=acc.get("timeout_seconds"),
                kernel_args=acc.get("kernel_args"),
                working_directory=acc.get("working_directory"),
                workspace_path=cfg.get("workspace_path"),
            )

        return cls(
            mode=mode,
            backend=backend,
            accordo_config=accordo_cfg,
        )
