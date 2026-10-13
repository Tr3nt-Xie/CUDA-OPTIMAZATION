#include "common/cuda_utils.cuh"

namespace {

constexpr int kTile = 16;

/*
 * Part 4: shared-memory tiling. One block of kTile x kTile threads computes
 * one kTile x kTile tile of C. The dot product is split into steps of kTile:
 * in each step the block copies one tile of A and one tile of B into shared
 * memory, then every thread does kTile multiply-adds from shared memory.
 * Each value read from global memory is therefore used kTile times.
 */
__global__ void matrixMultiplyTiled(const float *A, const float *B, float *C,
                                    int N) {
    __shared__ float tileA[kTile][kTile];
    __shared__ float tileB[kTile][kTile];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * kTile + ty; /* element of C this thread owns */
    const int col = blockIdx.x * kTile + tx;

    float sum = 0.0f;
    for (int m = 0; m < (N + kTile - 1) / kTile; m++) {
        /* Row of A is fixed, the tile moves right; column of B is fixed,
         * the tile moves down. Outside the matrix the tile is padded with
         * zeros, which add nothing to the sum, so any N works. */
        const int aCol = m * kTile + tx;
        const int bRow = m * kTile + ty;
        tileA[ty][tx] = (row < N && aCol < N) ? A[row * N + aCol] : 0.0f;
        tileB[ty][tx] = (bRow < N && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads(); /* both tiles complete before anyone reads them */

        for (int k = 0; k < kTile; k++)
            sum += tileA[ty][k] * tileB[k][tx];
        __syncthreads(); /* everyone done before the tiles are overwritten */
    }
    if (row < N && col < N)
        C[row * N + col] = sum;
}

} // namespace

int lab6_matmul_tiled(const float *A, const float *B, float *C, int N,
                      lab6_timings *timings) {
    return lab6::run_matmul(
        A, B, C, N, timings,
        [](const float *dA, const float *dB, float *dC, int n) {
            /* The block must be exactly one tile: threadIdx indexes the
             * shared arrays directly. */
            const dim3 block(kTile, kTile);
            const dim3 grid((n + kTile - 1) / kTile, (n + kTile - 1) / kTile);
            matrixMultiplyTiled<<<grid, block>>>(dA, dB, dC, n);
            return LAB6_CHECK_LAUNCH();
        });
}
