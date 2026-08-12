// Standalone correctness runner for the RDNA2 GEMM kernels. No torch.
// Compile: hipcc -O3 main.cpp gemm_dp4a.hip gemm_f16x2.hip
// (windows: gemm_dp4a.hip gemm_f16x2.hip have host stubs; main provides main())
#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

extern "C" void gemm_dp4a_launch(int M, int N, int K,
                                 const int8_t *A, const int8_t *Bt, int32_t *C,
                                 hipStream_t stream);
extern "C" void gemm_f16x2_launch(int M, int N, int K,
                                  const __half *A, const __half *Bt, float *C,
                                  hipStream_t stream);

static void ref_int8(int M, int N, int K,
                     const int8_t *A, const int8_t *Bt, int32_t *C) {
    // A[M,K], Bt[N,K] -> C[M,N]
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            int64_t s = 0;
            for (int k = 0; k < K; ++k) s += (int)A[i*K+k] * (int)Bt[j*K+k];
            C[i*N+j] = (int32_t) s;
        }
}

static void ref_f16(int M, int N, int K,
                    const __half *A, const __half *Bt, float *C) {
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float s = 0;
            for (int k = 0; k < K; ++k) s += (float)A[i*K+k] * (float)Bt[j*K+k];
            C[i*N+j] = s;
        }
}

static int test_int8(int M, int N, int K) {
    std::vector<int8_t> A(M*K), B(K*N), Bt(N*K);
    std::vector<int32_t> ref(M*N), out(M*N);
    for (auto &v : A) v = rand() % 256 - 128;
    for (auto &v : B) v = rand() % 256 - 128;
    for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) Bt[n*K+k] = B[k*N+n];
    ref_int8(M, N, K, A.data(), Bt.data(), ref.data());

    int8_t *dA, *dBt; int32_t *dC;
    hipMalloc(&dA, A.size()); hipMalloc(&dBt, Bt.size()); hipMalloc(&dC, out.size()*4);
    hipMemcpy(dA, A.data(), A.size(), hipMemcpyHostToDevice);
    hipMemcpy(dBt, Bt.data(), Bt.size(), hipMemcpyHostToDevice);
    gemm_dp4a_launch(M, N, K, dA, dBt, dC, 0);
    hipMemcpy(out.data(), dC, out.size()*4, hipMemcpyDeviceToHost);
    hipFree(dA); hipFree(dBt); hipFree(dC);

    for (size_t i = 0; i < out.size(); ++i)
        if (out[i] != ref[i]) { printf("int8 FAIL M=%d N=%d K=%d at %zu: %d != %d\n", M,N,K,i,out[i],ref[i]); return 1; }
    printf("int8  OK M=%d N=%d K=%d\n", M, N, K);
    return 0;
}

static int test_f16(int M, int N, int K) {
    std::vector<__half> A(M*K), B(K*N), Bt(N*K);
    std::vector<float> ref(M*N), out(M*N);
    for (auto &v : A) v = __float2half(rand() % 10 / 2.0f - 2.5f);
    for (auto &v : B) v = __float2half(rand() % 10 / 2.0f - 2.5f);
    for (int k = 0; k < K; ++k) for (int n = 0; n < N; ++n) Bt[n*K+k] = B[k*N+n];
    ref_f16(M, N, K, A.data(), Bt.data(), ref.data());

    __half *dA, *dBt; float *dC;
    hipMalloc(&dA, A.size()*2); hipMalloc(&dBt, Bt.size()*2); hipMalloc(&dC, out.size()*4);
    hipMemcpy(dA, A.data(), A.size()*2, hipMemcpyHostToDevice);
    hipMemcpy(dBt, Bt.data(), Bt.size()*2, hipMemcpyHostToDevice);
    gemm_f16x2_launch(M, N, K, dA, dBt, dC, 0);
    hipMemcpy(out.data(), dC, out.size()*4, hipMemcpyDeviceToHost);
    hipFree(dA); hipFree(dBt); hipFree(dC);

    for (size_t i = 0; i < out.size(); ++i)
        if (fabs(out[i] - ref[i]) > 1.0f) { printf("f16 FAIL M=%d N=%d K=%d at %zu: %f != %f\n", M,N,K,i,out[i],ref[i]); return 1; }
    printf("f16x2 OK M=%d N=%d K=%d\n", M, N, K);
    return 0;
}

int main() {
    hipDeviceProp_t prop;
    hipGetDeviceProperties(&prop, 0);
    printf("device: %s wave=%d\n", prop.name, prop.warpSize);
    srand(42);
    int rc = 0;
    rc |= test_int8(128, 128, 16);
    rc |= test_int8(256, 256, 64);
    rc |= test_int8(4096, 4096, 256);
    rc |= test_f16(128, 128, 16);
    rc |= test_f16(256, 256, 64);
    rc |= test_f16(4096, 4096, 256);
    printf(rc ? "FAILED\n" : "ALL PASSED\n");
    return rc;
}
