#include "common/timer.h"
#include "common/validate.h"
#include "lab6.h"

int lab6_matmul_cpu(const float *A, const float *B, float *C, int N,
                    lab6_timings *timings) {
    size_t elems = 0;
    int status = lab6_check_matmul_args(A, B, C, N, &elems);
    if (status != LAB6_OK)
        return status;

    const size_t n = (size_t)N;
    const double begin_ms = lab6_now_ms();
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t k = 0; k < n; k++)
                sum += A[i * n + k] * B[k * n + j];
            C[i * n + j] = sum;
        }
    }
    if (timings) {
        timings->compute_ms = lab6_now_ms() - begin_ms;
        timings->end_to_end_ms = timings->compute_ms;
    }
    return LAB6_OK;
}
