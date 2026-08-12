# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# Validates the RDNA2 (gfx1031) sdot4 INT8 GEMM against torch reference.
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch

from gemm_dp4a_wrapper import gemm_dp4a_int8, gemm_f16x2


def test_gemm_dp4a(device):
    torch.manual_seed(0)
    # Note: shapes must be >= 128x128 to spawn more than one block; small shapes
    # run on 1 CU (1 block = 8 waves = 8 SIMDs). Use tall/wide shapes to engage
    # all 40 CUs of the RX 6700 XT.
    for M, N, K in [(128, 128, 16), (256, 256, 64), (4096, 4096, 256)]:
        A = torch.randint(-128, 128, (M, K), dtype=torch.int8).to(device)
        B = torch.randint(-128, 128, (K, N), dtype=torch.int8).to(device)
        C = gemm_dp4a_int8(A, B)
        ref = A.to(torch.int32) @ B.to(torch.int32)
        assert torch.equal(C, ref), f"M={M} N={N} K={K}: mismatch"
        print(f"OK int8 M={M} N={N} K={K}")


def test_gemm_f16x2(device):
    torch.manual_seed(0)
    for M, N, K in [(128, 128, 16), (256, 256, 64), (4096, 4096, 256)]:
        A = torch.randn(M, K, dtype=torch.float16).to(device)
        B = torch.randn(K, N, dtype=torch.float16).to(device)
        C = gemm_f16x2(A, B)
        ref = A.to(torch.float32) @ B.to(torch.float32)
        assert torch.allclose(C, ref, atol=2.0, rtol=1e-2), \
            f"M={M} N={N} K={K}: mismatch"
        print(f"OK f16x2 M={M} N={N} K={K}")


if __name__ == "__main__":
    test_gemm_dp4a(torch.device("cuda"))
    test_gemm_f16x2(torch.device("cuda"))
    print("all passed")
