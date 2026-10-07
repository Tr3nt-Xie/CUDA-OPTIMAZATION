#include "common/cuda_utils.cuh"

namespace {

constexpr int kBlock = 16;

__global__ void matrixMultiplyGPU(const float *A, const float *B, float *C,
                                  int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++)
            sum += A[row * N + k] * B[k * N + col];
        C[row * N + col] = sum;
    }
}

} // namespace

int lab6_matmul_naive(const float *A, const float *B, float *C, int N,
                      lab6_timings *timings) {
    return lab6::run_matmul(
        A, B, C, N, timings,
        [](const float *dA, const float *dB, float *dC, int n) {
            const dim3 block(kBlock, kBlock);
            const dim3 grid((n + kBlock - 1) / kBlock,
                            (n + kBlock - 1) / kBlock);
            matrixMultiplyGPU<<<grid, block>>>(dA, dB, dC, n);
            return LAB6_CHECK_LAUNCH();
        });
}
