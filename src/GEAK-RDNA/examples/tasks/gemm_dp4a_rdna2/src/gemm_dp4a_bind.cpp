// RDNA2 INT8 sdot4 GEMM torch binding. Row-major A[M,K] B[K,N] C[M,N].
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>

#define CHECK_HIP(x) TORCH_CHECK(x.is_cuda(), #x, " must be a HIP/CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous")

void gemm_dp4a_launch(int M, int N, int K,
                      const int8_t *A, const int8_t *B, int32_t *C,
                      hipStream_t stream);

void gemm_f16x2_launch(int M, int N, int K,
                       const half *A, const half *B, float *C,
                       hipStream_t stream);

void gemm_dp4a_wrapper(at::Tensor A, at::Tensor B, at::Tensor C) {
    CHECK_HIP(A); CHECK_HIP(B); CHECK_HIP(C);
    CHECK_CONTIGUOUS(A); CHECK_CONTIGUOUS(B); CHECK_CONTIGUOUS(C);

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    gemm_dp4a_launch(M, N, K,
                     (const int8_t *) A.data_ptr<int8_t>(),
                     (const int8_t *) B.data_ptr<int8_t>(),
                     (int32_t *) C.data_ptr<int32_t>(),
                     stream);
}

void gemm_f16x2_wrapper(at::Tensor A, at::Tensor B, at::Tensor C) {
    CHECK_HIP(A); CHECK_HIP(B); CHECK_HIP(C);
    CHECK_CONTIGUOUS(A); CHECK_CONTIGUOUS(B); CHECK_CONTIGUOUS(C);

    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);

    hipStream_t stream = at::hip::getCurrentHIPStreamMasqueradingAsCUDA();
    gemm_f16x2_launch(M, N, K,
                      (const half *) A.data_ptr<at::Half>(),
                      (const half *) B.data_ptr<at::Half>(),
                      (float *) C.data_ptr<float>(),
                      stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("launch_int8", &gemm_dp4a_wrapper, "RDNA2 INT8 sdot4 GEMM");
    m.def("launch_f16x2", &gemm_f16x2_wrapper, "RDNA2 FP16 v_dot2 GEMM");
}
