#include <cublas_v2.h>

#include "common/cuda_utils.cuh"
#include "lab6.h"

namespace {
bool cublas_ok(cublasStatus_t status, const char *operation) {
    if (status == CUBLAS_STATUS_SUCCESS)
        return true;
    std::fprintf(stderr, "%s failed (cuBLAS status %d)\n", operation,
                 static_cast<int>(status));
    return false;
}

class Handle {
  public:
    cublasHandle_t value = nullptr;
    Handle() = default;
    Handle(const Handle &) = delete;
    Handle &operator=(const Handle &) = delete;
    ~Handle() {
        if (value)
            cublasDestroy(value);
    }
    bool close() {
        cublasHandle_t old = value;
        value = nullptr;
        return cublas_ok(cublasDestroy(old), "cublasDestroy");
    }
};
} // namespace

/* Part 6.1: FP32 SGEMM, host inputs/output, row-major C = A * B. */
int lab6_matmul_cublas(const float *A, const float *B, float *C, int N,
                       lab6_timings *timings) {
    size_t elems = 0;
    int status = lab6_check_matmul_args(A, B, C, N, &elems);
    if (status != LAB6_OK)
        return status;

    const double begin_ms = lab6_now_ms();
    lab6_timings measured = {};
    {
        Handle handle;
        if (!cublas_ok(cublasCreate(&handle.value), "cublasCreate") ||
            !cublas_ok(
                cublasSetPointerMode(handle.value, CUBLAS_POINTER_MODE_HOST),
                "cublasSetPointerMode") ||
            !cublas_ok(cublasSetMathMode(handle.value, CUBLAS_PEDANTIC_MATH),
                       "cublasSetMathMode"))
            return LAB6_RUNTIME_ERROR;

        status = lab6::run_matmul(
            A, B, C, N, timings ? &measured : nullptr,
            [&handle](const float *dA, const float *dB, float *dC, int n) {
                const float alpha = 1.0f, beta = 0.0f;
                // Column-major view: C^T = B^T * A^T. Swap operands,
                // not transpose flags; the host output remains row-major.
                return cublas_ok(cublasSgemm(handle.value, CUBLAS_OP_N,
                                             CUBLAS_OP_N, n, n, n, &alpha, dB,
                                             n, dA, n, &beta, dC, n),
                                 "cublasSgemm")
                           ? LAB6_OK
                           : LAB6_RUNTIME_ERROR;
            });
        if (status != LAB6_OK)
            return status;
        if (!handle.close())
            return LAB6_RUNTIME_ERROR;
    }
    if (timings) {
        timings->compute_ms = measured.compute_ms;
        // Includes this call's handle creation/destruction, allocation,
        // transfers, synchronization and cleanup. Events cover only SGEMM.
        timings->end_to_end_ms = lab6_now_ms() - begin_ms;
    }
    return LAB6_OK;
}
