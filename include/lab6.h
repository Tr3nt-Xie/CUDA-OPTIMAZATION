#ifndef LAB6_H
#define LAB6_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    LAB6_OK = 0,
    LAB6_INVALID_ARGUMENT = 1,
    LAB6_RUNTIME_ERROR = 2,
    LAB6_NOT_IMPLEMENTED = 3
} lab6_status;

/* Valid only when the function returns LAB6_OK. All times are milliseconds. */
typedef struct {
    double compute_ms;
    double end_to_end_ms;
} lab6_timings;

/*
 * All pointers refer to host memory, contiguous and row-major.
 * Caller allocates output, which must not overlap inputs.
 * Matrix: C = A * B, N > 0, float32.
 * Convolution: uint32 image in [0,255], float32 filter/output;
 * M > 0, odd K > 0, true convolution, stride 1, zero padding, same size.
 * timings may be NULL. All GPU operations finish before return.
 * On failure, output and timings are unspecified and must not be reported.
 */
int lab6_matmul_cpu(const float *A, const float *B, float *C, int N,
                    lab6_timings *timings);
int lab6_matmul_naive(const float *A, const float *B, float *C, int N,
                      lab6_timings *timings);
int lab6_matmul_tiled(const float *A, const float *B, float *C, int N,
                      lab6_timings *timings);
int lab6_matmul_optimized(const float *A, const float *B, float *C, int N,
                          lab6_timings *timings);
int lab6_matmul_cublas(const float *A, const float *B, float *C, int N,
                       lab6_timings *timings);

int lab6_convolve_cpu(const uint32_t *image, const float *filter, float *output,
                      int M, int K, lab6_timings *timings);
int lab6_convolve_cuda(const uint32_t *image, const float *filter,
                       float *output, int M, int K, lab6_timings *timings);

/* Public Python ABI. Matrix entry uses the Part 4 tiled implementation. */
int gpu_matrix_multiply(const float *A, const float *B, float *C, int N);
int gpu_convolve(const uint32_t *image, const float *filter, float *output,
                 int M, int K);

#ifdef __cplusplus
}
#endif

#endif
