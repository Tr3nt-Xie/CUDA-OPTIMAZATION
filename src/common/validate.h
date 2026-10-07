#ifndef LAB6_VALIDATE_H
#define LAB6_VALIDATE_H

#include <stddef.h>
#include <stdint.h>

#include "lab6.h"

/* Optional argument-checking helpers for C and CUDA implementations. */

static inline int lab6_square_elems(int n, size_t elem_size, size_t *elems) {
    if (n <= 0)
        return 0;
    size_t count = (size_t)n * (size_t)n;
    if (count / (size_t)n != (size_t)n || count > SIZE_MAX / elem_size)
        return 0;
    *elems = count;
    return 1;
}

static inline int lab6_check_matmul_args(const float *A, const float *B,
                                         const float *C, int N, size_t *elems) {
    if (!A || !B || !C || !lab6_square_elems(N, sizeof(float), elems))
        return LAB6_INVALID_ARGUMENT;
    return LAB6_OK;
}

static inline int lab6_check_convolve_args(const uint32_t *image,
                                           const float *filter,
                                           const float *output, int M, int K,
                                           size_t *image_elems,
                                           size_t *filter_elems) {
    if (!image || !filter || !output || K <= 0 || K % 2 == 0)
        return LAB6_INVALID_ARGUMENT;
    if (!lab6_square_elems(M, sizeof(uint32_t), image_elems) ||
        !lab6_square_elems(K, sizeof(float), filter_elems))
        return LAB6_INVALID_ARGUMENT;
    return LAB6_OK;
}

#endif
