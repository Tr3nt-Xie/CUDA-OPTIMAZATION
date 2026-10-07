#ifndef LAB6_BENCH_HARNESS_H
#define LAB6_BENCH_HARNESS_H

#include "lab6.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef int (*bench_matmul_fn)(const float *A, const float *B, float *C, int N,
                               lab6_timings *timings);
typedef int (*bench_convolve_fn)(const uint32_t *image, const float *filter,
                                 float *output, int M, int K,
                                 lab6_timings *timings);

/* Either function may be NULL when the implementation does not provide it. */
typedef struct {
    const char *name;
    bench_matmul_fn matmul;
    bench_convolve_fn convolve;
} bench_impl;

int bench_main(int argc, char **argv, const char *program,
               const bench_impl *impls, int impl_count);

#ifdef __cplusplus
}
#endif

#endif
