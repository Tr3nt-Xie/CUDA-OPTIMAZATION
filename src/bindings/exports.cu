#include "lab6.h"

/* Parts 7 and 8.3 (C): current C ABI exports for Python. */
extern "C" int gpu_matrix_multiply(const float *A, const float *B, float *C,
                                   int N) {
    return lab6_matmul_tiled(A, B, C, N, nullptr);
}

extern "C" int gpu_convolve(const uint32_t *image, const float *filter,
                            float *output, int M, int K) {
    return lab6_convolve_cuda(image, filter, output, M, K, nullptr);
}
