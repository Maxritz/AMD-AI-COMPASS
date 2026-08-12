# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
from torch.utils.cpp_extension import load

gemm_dp4a_ext = load(name="gemm_dp4a_rdna2",
                     sources=["src/gemm_dp4a.hip", "src/gemm_f16x2.hip",
                              "src/gemm_dp4a_bind.cpp"],
                     verbose=False)
