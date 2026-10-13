#include "common/cuda_utils.cuh"

namespace {

/*
 * Part 6.2: register-blocked matrix multiplication.
 *
 * The tiled kernel of Part 4 does one multiply-add per two shared-memory
 * reads, so shared memory, not arithmetic, limits it. Here a block computes a
 * BM x BN tile of C and each thread computes a TM x TN sub-tile that it keeps
 * in registers. For every k the thread reads TM values of A and TN values of
 * B from shared memory and does TM * TN multiply-adds (an outer product):
 * with 8 x 8 that is 64 multiply-adds per 16 reads, eight times fewer reads
 * per operation than Part 4.
 *
 * Two more changes on top of that:
 *  - The A tile is stored transposed (tileA[k][m]), so the TM values a thread
 *    needs are adjacent in shared memory, like the TN values of B.
 *  - When N is a multiple of the tile size, global loads and stores move four
 *    floats per instruction (float4).
 *
 * Guard = true is the general version for any N: loads outside the matrix
 * read as zero, which adds nothing to the sums, and stores outside it are
 * skipped. Guard = false requires N to be a multiple of BM and BN.
 */
template <int BM, int BN, int BK, int TM, int TN, bool Guard>
__global__ void __launch_bounds__((BM * BN) / (TM * TN))
    matrixMultiplyRegBlock(const float *A, const float *B, float *C, int N) {
    static_assert(BM % TM == 0 && BN % TN == 0, "sub-tiles must fill the tile");
    static_assert(BK % 4 == 0 && BN % 4 == 0 && TN % 4 == 0,
                  "float4 loads and stores need multiples of four");
    constexpr int kThreads = (BM * BN) / (TM * TN);

    __shared__ float tileA[BK * BM]; /* transposed: tileA[k * BM + m] */
    __shared__ float tileB[BK * BN]; /* tileB[k * BN + n] */

    const int tid = threadIdx.x;
    const int rowBase = blockIdx.y * BM; /* first row of this block in C */
    const int colBase = blockIdx.x * BN;
    const int subRow =
        (tid / (BN / TN)) * TM; /* this thread, inside the tile */
    const int subCol = (tid % (BN / TN)) * TN;

    float acc[TM * TN] = {0.0f};
    float a[TM];
    float b[TN];

    for (int k0 = 0; k0 < N; k0 += BK) {
        /* Load BM x BK of A and BK x BN of B; every thread takes an equal
         * share, striding by the number of threads. */
        if (Guard) {
#pragma unroll
            for (int e = tid; e < BM * BK; e += kThreads) {
                const int m = e / BK;
                const int k = e % BK;
                const int r = rowBase + m;
                const int c = k0 + k;
                tileA[k * BM + m] = (r < N && c < N) ? A[r * N + c] : 0.0f;
            }
#pragma unroll
            for (int e = tid; e < BK * BN; e += kThreads) {
                const int k = e / BN;
                const int n = e % BN;
                const int r = k0 + k;
                const int c = colBase + n;
                tileB[k * BN + n] = (r < N && c < N) ? B[r * N + c] : 0.0f;
            }
        } else {
#pragma unroll
            for (int e = tid; e < BM * BK / 4; e += kThreads) {
                const int m = e / (BK / 4);
                const int k = (e % (BK / 4)) * 4;
                const float4 v = *reinterpret_cast<const float4 *>(
                    &A[(rowBase + m) * N + k0 + k]);
                tileA[(k + 0) * BM + m] = v.x;
                tileA[(k + 1) * BM + m] = v.y;
                tileA[(k + 2) * BM + m] = v.z;
                tileA[(k + 3) * BM + m] = v.w;
            }
#pragma unroll
            for (int e = tid; e < BK * BN / 4; e += kThreads) {
                const int k = e / (BN / 4);
                const int n = (e % (BN / 4)) * 4;
                *reinterpret_cast<float4 *>(&tileB[k * BN + n]) =
                    *reinterpret_cast<const float4 *>(
                        &B[(k0 + k) * N + colBase + n]);
            }
        }
        __syncthreads(); /* tiles complete before anyone reads them */

        /* Fully unrolled so acc, a and b stay in registers. */
#pragma unroll
        for (int k = 0; k < BK; k++) {
#pragma unroll
            for (int i = 0; i < TM; i++)
                a[i] = tileA[k * BM + subRow + i];
#pragma unroll
            for (int i = 0; i < TN; i++)
                b[i] = tileB[k * BN + subCol + i];
#pragma unroll
            for (int m = 0; m < TM; m++)
#pragma unroll
                for (int n = 0; n < TN; n++)
                    acc[m * TN + n] += a[m] * b[n];
        }
        __syncthreads(); /* everyone done before the tiles are overwritten */
    }

    if (Guard) {
#pragma unroll
        for (int m = 0; m < TM; m++)
#pragma unroll
            for (int n = 0; n < TN; n++) {
                const int r = rowBase + subRow + m;
                const int c = colBase + subCol + n;
                if (r < N && c < N)
                    C[r * N + c] = acc[m * TN + n];
            }
    } else {
#pragma unroll
        for (int m = 0; m < TM; m++)
#pragma unroll
            for (int n = 0; n < TN; n += 4)
                *reinterpret_cast<float4 *>(
                    &C[(rowBase + subRow + m) * N + colBase + subCol + n]) =
                    make_float4(acc[m * TN + n], acc[m * TN + n + 1],
                                acc[m * TN + n + 2], acc[m * TN + n + 3]);
    }
}

template <int BM, int BN, int BK, int TM, int TN>
int launchRegBlock(const float *dA, const float *dB, float *dC, int n) {
    const int threads = (BM * BN) / (TM * TN);
    const dim3 grid((n + BN - 1) / BN, (n + BM - 1) / BM);
    if (n % BM == 0 && n % BN == 0)
        matrixMultiplyRegBlock<BM, BN, BK, TM, TN, false>
            <<<grid, threads>>>(dA, dB, dC, n);
    else
        matrixMultiplyRegBlock<BM, BN, BK, TM, TN, true>
            <<<grid, threads>>>(dA, dB, dC, n);
    return LAB6_CHECK_LAUNCH();
}

/* Smallest N that uses 128 x 128 tiles: it still gives 16 x 16 blocks. */
constexpr int kLargeTileMinN = 2048;

} // namespace

int lab6_matmul_optimized(const float *A, const float *B, float *C, int N,
                          lab6_timings *timings) {
    return lab6::run_matmul(
        A, B, C, N, timings,
        [](const float *dA, const float *dB, float *dC, int n) {
            /* Large tiles reuse each loaded value more, but a small matrix
             * then has too few blocks to occupy every SM. Tile sizes were
             * chosen by measurement; see report/B.md. */
            if (n >= kLargeTileMinN)
                return launchRegBlock<128, 128, 16, 8, 8>(dA, dB, dC, n);
            return launchRegBlock<64, 64, 16, 8, 8>(dA, dB, dC, n);
        });
}
