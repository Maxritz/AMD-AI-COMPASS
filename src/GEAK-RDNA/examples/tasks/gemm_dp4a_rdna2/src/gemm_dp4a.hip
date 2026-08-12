#include "hip/hip_runtime.h"
// RDNA2 (gfx1031) INT8 GEMM, dp4a register-tiled path.
// Levers worth pulling on gfx1031 (RX 6700 XT, 40 CU, 384 GB/s):
//   - sdot4 native (v_dot4_i32_i8) = 4 int8 muls+add per instr, 2x FP16 rate
//   - Wave64 (no wave32 on RDNA2)
//   - 512 VGPR/SIMD (2x CDNA1) -> 16 int32 accumulators/lane is register-cheap
//   - B pre-transposed to [N][K] on host -> all dot reads contiguous, L2 handles reuse
// Layout: A[M,K] row-major; Bt[N,K] row-major (transposed); C[M,N] int32. K % 4 == 0.

#define WAVE_SIZE 64
#define BLOCK_SIZE 256
// Register tile 8x8 = 64 int32 accumulators/lane. 512 VGPR/SIMD on RDNA2
// (2x CDNA1's 256) makes this free; bytes/output = K/TC_N + K/TC_M halves
// vs 4x4 on a bandwidth-starved (384 GB/s) card.
#define TC_M 8
#define TC_N 8
#define TILES 16 // threads per tile-grid row = 256/16

__global__ __launch_bounds__(BLOCK_SIZE)
void gemm_dp4a_int8_rdna2(
        int M, int N, int K, int Nstride,
        const int8_t *__restrict__ A, const int8_t *__restrict__ Bt,
        int32_t *__restrict__ C) {
    const int tid = threadIdx.x;

    // 16x16 grid of 8x8 register tiles per block (256 threads).
    // blockIdx.x strips M, blockIdx.y strips N -> parallel across all CUs.
    const int tile_r = tid / 16; // 0..15
    const int tile_c = tid % 16; // 0..15
    const int n0 = blockIdx.y * TILES * TC_N + tile_c * TC_N;
    const int m0 = blockIdx.x * TILES * TC_M + tile_r * TC_M;

    int32_t acc[TC_M][TC_N] = {0};

    for (int k = 0; k < K; k += 4) {
#pragma unroll
        for (int rr = 0; rr < TC_M; ++rr) {
            const int i = m0 + rr;
            if (i >= M) continue;
            const int a4 = *reinterpret_cast<const int*>(&A[i * K + k]);
#pragma unroll
            for (int cc = 0; cc < TC_N; ++cc) {
                const int j = n0 + cc;
                if (j >= N) continue;
                // Bt[j][k..k+3] contiguous.
                const int b4 = *reinterpret_cast<const int*>(&Bt[j * K + k]);
                acc[rr][cc] = __builtin_amdgcn_sdot4(a4, b4, acc[rr][cc], false);
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

extern "C" __host__ void gemm_dp4a_launch(
        int M, int N, int K, const int8_t *A, const int8_t *Bt, int32_t *C,
        hipStream_t stream) {
    const int rows_per_block  = TILES * TC_M; // 128
    const int cols_per_block  = TILES * TC_N; // 128

    // Split N across 2 HIP streams so both RDNA2 async-compute queues (the
    // "Compute 0" / "Compute 1" bars in Task Manager) get work. The default
    // stream alone only ever uses one ACE.
    const int N_half = (N + 1) / 2;

    hipStream_t s0 = stream;
    hipStream_t s1 = nullptr;
    if (N > 64) hipStreamCreate(&s1); // 2nd queue only if enough N to matter

    const int n_split[2] = { 0, N_half };
    for (int h = 0; h < 2; ++h) {
        const int Nn = (h == 0) ? N_half : (N - N_half);
        if (Nn <= 0) continue;
        const int base = n_split[h];
        dim3 grid((M + rows_per_block - 1) / rows_per_block,
                  (Nn + cols_per_block - 1) / cols_per_block);
        gemm_dp4a_int8_rdna2<<<grid, BLOCK_SIZE, 0, (h == 0) ? s0 : s1>>>(
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
