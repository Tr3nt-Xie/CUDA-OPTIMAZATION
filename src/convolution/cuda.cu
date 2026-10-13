#include "common/cuda_utils.cuh"

namespace {

constexpr int kTile = 16;
/* Shared memory every block may request without opting in to more. */
constexpr size_t kMaxSharedBytes = 48 * 1024;

/*
 * Part 8.2: CUDA convolution, same definition as the CPU version:
 *   out[y][x] = sum over i, j of filter[i][j] * image[y + r - i][x + r - j]
 * with r = K / 2, the filter flipped (true convolution), stride 1, and image
 * values outside the image taken as zero.
 *
 * A block computes a kTile x kTile tile of the output. Those outputs read a
 * (kTile + K - 1)^2 window of the image: the tile plus r pixels on each
 * side. The block first copies that window into shared memory, converting to
 * float and writing zero for positions outside the image, so the inner loop
 * needs no bounds checks. It also copies the filter, already flipped, so the
 * inner loop walks both arrays forwards.
 */
__global__ void convolveTiled(const uint32_t *image, const float *filter,
                              float *output, int M, int K) {
    extern __shared__ float shared[]; /* size is set at launch */
    const int r = K / 2;
    const int W = kTile + K - 1;
    float *window = shared;       /* W x W */
    float *taps = shared + W * W; /* K x K, flipped */

    const int tid = threadIdx.y * kTile + threadIdx.x;
    const int x0 = blockIdx.x * kTile - r; /* image position of window[0] */
    const int y0 = blockIdx.y * kTile - r;
    for (int e = tid; e < W * W; e += kTile * kTile) {
        const int iy = y0 + e / W;
        const int ix = x0 + e % W;
        window[e] = (iy >= 0 && iy < M && ix >= 0 && ix < M)
                        ? static_cast<float>(image[iy * M + ix])
                        : 0.0f;
    }
    /* Flipping both axes of a K x K array reverses its flattened order. */
    for (int e = tid; e < K * K; e += kTile * kTile)
        taps[e] = filter[K * K - 1 - e];
    __syncthreads(); /* window and taps complete before anyone reads them */

    const int x = blockIdx.x * kTile + threadIdx.x;
    const int y = blockIdx.y * kTile + threadIdx.y;
    if (x >= M || y >= M) /* after the barrier: every thread helped load */
        return;
    float sum = 0.0f;
    for (int a = 0; a < K; a++)
        for (int b = 0; b < K; b++)
            sum += taps[a * K + b] *
                   window[(threadIdx.y + a) * W + threadIdx.x + b];
    output[y * M + x] = sum;
}

/*
 * One thread per output pixel, reading the image and filter from global
 * memory. Used only when the window for convolveTiled does not fit in shared
 * memory (very large K).
 */
__global__ void convolveDirect(const uint32_t *image, const float *filter,
                               float *output, int M, int K) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= M || y >= M)
        return;
    const int r = K / 2;
    float sum = 0.0f;
    for (int i = 0; i < K; i++) {
        const int iy = y + r - i;
        if (iy < 0 || iy >= M)
            continue;
        for (int j = 0; j < K; j++) {
            const int ix = x + r - j;
            if (ix < 0 || ix >= M)
                continue;
            sum += filter[i * K + j] * static_cast<float>(image[iy * M + ix]);
        }
    }
    output[y * M + x] = sum;
}

} // namespace

int lab6_convolve_cuda(const uint32_t *image, const float *filter,
                       float *output, int M, int K, lab6_timings *timings) {
    return lab6::run_convolution(
        image, filter, output, M, K, timings,
        [](const uint32_t *dImage, const float *dFilter, float *dOutput, int m,
           int k) {
            const dim3 block(kTile, kTile);
            const dim3 grid((m + kTile - 1) / kTile, (m + kTile - 1) / kTile);
            const size_t w = static_cast<size_t>(kTile) + k - 1;
            const size_t sharedBytes =
                (w * w + static_cast<size_t>(k) * k) * sizeof(float);
            if (sharedBytes <= kMaxSharedBytes)
                convolveTiled<<<grid, block, sharedBytes>>>(dImage, dFilter,
                                                            dOutput, m, k);
            else
                convolveDirect<<<grid, block>>>(dImage, dFilter, dOutput, m, k);
            return LAB6_CHECK_LAUNCH();
        });
}
