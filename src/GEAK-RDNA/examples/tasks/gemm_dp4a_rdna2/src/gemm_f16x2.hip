#include "hip/hip_runtime.h"
#include "hip/hip_fp16.h"
#include <cstdint>
// RDNA2 (gfx1031) FP16 GEMM via native v_dot2_f32_f16 (packed fp16 x2 -> f32 acc).
// The fp16 companion to gemm_dp4a.hip: same register-tile + Wave64 structure,
// but the dot product is v_dot2_f32_f16 (2 fp16 muls + add per instr, native).
// FP16 is native on RDNA2 at 2x FP32 rate (24.8 TF on RX 6700 XT).
// Layout: A[M,K] f16, Bt[N,K] f16 (pre-transposed), C[M,N] f32. K even.
// acc must be fp32 (f16 acc would round after every dot).

#define WAVE_SIZE 64
#define BLOCK_SIZE 256
#define TC_M 8
#define TC_N 8
#define TILES 16

__device__ __forceinline__ float ggml_v_dot2_f16(const half2 v, const half2 u, float acc) {
    asm volatile("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(v), "v"(u));
    return acc;
}

__global__ __launch_bounds__(BLOCK_SIZE)
void gemm_f16x2_rdna2(
        int M, int N, int K, int Nstride,
        const half *__restrict__ A, const half *__restrict__ Bt,
        float *__restrict__ C) {
    const int tid = threadIdx.x;

    // 16x16 grid of 8x8 register tiles per block. Same layout as gemm_dp4a.
    const int tile_r = tid / 16;
    const int tile_c = tid % 16;
    const int n0 = blockIdx.y * TILES * TC_N + tile_c * TC_N;
    const int m0 = blockIdx.x * TILES * TC_M + tile_r * TC_M;

    float acc[TC_M][TC_N] = {0.0f};

    for (int k = 0; k < K; k += 2) {
#pragma unroll
        for (int rr = 0; rr < TC_M; ++rr) {
            const int i = m0 + rr;
            if (i >= M) continue;
            const half2 av = *reinterpret_cast<const half2*>(&A[i * K + k]);
#pragma unroll
            for (int cc = 0; cc < TC_N; ++cc) {
                const int j = n0 + cc;
                if (j >= N) continue;
                // Bt[j][k..k+1] contiguous.
                const half2 bv = *reinterpret_cast<const half2*>(&Bt[j * K + k]);
                acc[rr][cc] = ggml_v_dot2_f16(av, bv, acc[rr][cc]);
            }
        }
    }

#pragma unroll
    for (int rr = 0; rr < TC_M; ++rr) {
        const int i = m0 + rr;
        if (i >= M) continue;
#pragma unroll
        for (int cc = 0; cc < TC_N; ++cc) {
            const int j = n0 + cc;
            if (j >= N) continue;
            C[i * Nstride + j] = acc[rr][cc];
        }
    }
}

extern "C" __host__ void gemm_f16x2_launch(
        int M, int N, int K, const __half *A, const __half *Bt, float *C,
        hipStream_t stream) {
    const int rows_per_block = TILES * TC_M; // 128
    const int cols_per_block = TILES * TC_N; // 128

    const int N_half = (N + 1) / 2;

    hipStream_t s0 = stream;
    hipStream_t s1 = nullptr;
    if (N > 64) hipStreamCreate(&s1);

    const int n_split[2] = { 0, N_half };
    for (int h = 0; h < 2; ++h) {
        const int Nn = (h == 0) ? N_half : (N - N_half);
        if (Nn <= 0) continue;
        const int base = n_split[h];
        dim3 grid((M + rows_per_block - 1) / rows_per_block,
                  (Nn + cols_per_block - 1) / cols_per_block);
        gemm_f16x2_rdna2<<<grid, BLOCK_SIZE, 0, (h == 0) ? s0 : s1>>>(
            M, Nn, K, N, A, Bt + base * K, C + base);
    }

    if (s1) {
        hipEvent_t ev;
        hipEventCreate(&ev);
        hipEventRecord(ev, s1);
        hipStreamWaitEvent(s0, ev, 0);
        hipEventDestroy(ev);
        hipStreamDestroy(s1);
    }
}
