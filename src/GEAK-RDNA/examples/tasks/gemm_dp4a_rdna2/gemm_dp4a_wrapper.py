# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# RDNA2 (gfx1031) GEMM demos: INT8 via native sdot4, FP16 via native v_dot2.
# A[M,K] x B[K,N] (B pre-transposed to [N,K] by the wrapper).
import torch

from kernel_loader import gemm_dp4a_ext


def gemm_dp4a_int8(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """INT8 sdot4 GEMM. A[M,K], B[K,N] int8 -> C[M,N] int32."""
    assert A.dtype == torch.int8 and B.dtype == torch.int8
    assert A.is_contiguous() and B.is_contiguous()
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb and K % 4 == 0, "K must be a multiple of 4"
    Bt = B.t().contiguous()  # [N,K]
    C = torch.empty((M, N), dtype=torch.int32, device=A.device)
    gemm_dp4a_ext.launch_int8(A, Bt, C)
    return C


def gemm_f16x2(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """FP16 v_dot2_f32_f16 GEMM. A[M,K], B[K,N] f16 -> C[M,N] f32."""
    assert A.dtype == torch.float16 and B.dtype == torch.float16
    assert A.is_contiguous() and B.is_contiguous()
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb and K % 2 == 0, "K must be a multiple of 2"
    Bt = B.t().contiguous()  # [N,K]
    C = torch.empty((M, N), dtype=torch.float32, device=A.device)
    gemm_dp4a_ext.launch_f16x2(A, Bt, C)
    return C
