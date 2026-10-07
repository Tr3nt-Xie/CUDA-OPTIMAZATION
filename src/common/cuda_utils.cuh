#ifndef LAB6_CUDA_UTILS_CUH
#define LAB6_CUDA_UTILS_CUH

#include <cstdio>

#include <cuda_runtime.h>

#include "common/timer.h"
#include "common/validate.h"
#include "lab6.h"

/*
 * Optional host-side helpers, currently used by naive.cu. run_matmul and
 * run_convolution provide validation, allocation, transfers, synchronization,
 * cleanup and the compute / end_to_end timing described in docs/CONTRACTS.md.
 * Callers using these helpers supply a launch callback that enqueues work on
 * the default stream and returns a lab6_status.
 *
 * Each .cu is compiled separately without -rdc, so everything defined here
 * must be static, inline, or a template.
 */

#define LAB6_CUDA_OK(call) ::lab6::cuda_ok((call), #call, __FILE__, __LINE__)
#define LAB6_CHECK_LAUNCH()                                                    \
    (LAB6_CUDA_OK(cudaGetLastError()) ? LAB6_OK : LAB6_RUNTIME_ERROR)

namespace lab6 {

inline bool cuda_ok(cudaError_t err, const char *expr, const char *file,
                    int line) {
    if (err == cudaSuccess)
        return true;
    std::fprintf(stderr, "%s:%d: %s: %s\n", file, line, expr,
                 cudaGetErrorString(err));
    return false;
}

template <typename T> class DeviceBuffer {
  public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    ~DeviceBuffer() {
        if (ptr_)
            cudaFree(ptr_);
    }

    bool allocate(size_t count) {
        bytes_ = count * sizeof(T);
        void *raw = nullptr;
        if (!LAB6_CUDA_OK(cudaMalloc(&raw, bytes_)))
            return false;
        ptr_ = static_cast<T *>(raw);
        return true;
    }
    bool upload(const T *host) {
        return LAB6_CUDA_OK(
            cudaMemcpy(ptr_, host, bytes_, cudaMemcpyHostToDevice));
    }
    bool download(T *host) const {
        return LAB6_CUDA_OK(
            cudaMemcpy(host, ptr_, bytes_, cudaMemcpyDeviceToHost));
    }
    T *get() const { return ptr_; }

  private:
    T *ptr_ = nullptr;
    size_t bytes_ = 0;
};

/* A disabled timer creates no events and reports 0 ms. */
class KernelTimer {
  public:
    explicit KernelTimer(bool enabled) : enabled_(enabled) {}
    KernelTimer(const KernelTimer &) = delete;
    KernelTimer &operator=(const KernelTimer &) = delete;
    ~KernelTimer() {
        if (start_)
            cudaEventDestroy(start_);
        if (stop_)
            cudaEventDestroy(stop_);
    }

    bool start() {
        if (!enabled_)
            return true;
        return LAB6_CUDA_OK(cudaEventCreate(&start_)) &&
               LAB6_CUDA_OK(cudaEventCreate(&stop_)) &&
               LAB6_CUDA_OK(cudaEventRecord(start_));
    }
    bool stop() { return !enabled_ || LAB6_CUDA_OK(cudaEventRecord(stop_)); }
    bool elapsed_ms(double *ms) {
        *ms = 0.0;
        if (!enabled_)
            return true;
        float value = 0.0f;
        if (!LAB6_CUDA_OK(cudaEventSynchronize(stop_)) ||
            !LAB6_CUDA_OK(cudaEventElapsedTime(&value, start_, stop_)))
            return false;
        *ms = value;
        return true;
    }

  private:
    bool enabled_;
    cudaEvent_t start_ = nullptr;
    cudaEvent_t stop_ = nullptr;
};

/* launch(dA, dB, dC, N): device pointers, row-major N x N. */
template <typename Launch>
int run_matmul(const float *A, const float *B, float *C, int N,
               lab6_timings *timings, Launch launch) {
    size_t elems = 0;
    int status = lab6_check_matmul_args(A, B, C, N, &elems);
    if (status != LAB6_OK)
        return status;

    const double begin_ms = lab6_now_ms();
    double compute_ms = 0.0;
    {
        DeviceBuffer<float> dA, dB, dC;
        KernelTimer timer(timings != nullptr);
        if (!dA.allocate(elems) || !dB.allocate(elems) || !dC.allocate(elems) ||
            !dA.upload(A) || !dB.upload(B) || !timer.start())
            return LAB6_RUNTIME_ERROR;
        status = launch(dA.get(), dB.get(), dC.get(), N);
        if (status != LAB6_OK)
            return status;
        if (!timer.stop() || !LAB6_CUDA_OK(cudaDeviceSynchronize()) ||
            !timer.elapsed_ms(&compute_ms) || !dC.download(C))
            return LAB6_RUNTIME_ERROR;
    }
    if (timings) {
        timings->compute_ms = compute_ms;
        timings->end_to_end_ms = lab6_now_ms() - begin_ms;
    }
    return LAB6_OK;
}

/* launch(dImage, dFilter, dOutput, M, K): device pointers, row-major. */
template <typename Launch>
int run_convolution(const uint32_t *image, const float *filter, float *output,
                    int M, int K, lab6_timings *timings, Launch launch) {
    size_t image_elems = 0;
    size_t filter_elems = 0;
    int status = lab6_check_convolve_args(image, filter, output, M, K,
                                          &image_elems, &filter_elems);
    if (status != LAB6_OK)
        return status;

    const double begin_ms = lab6_now_ms();
    double compute_ms = 0.0;
    {
        DeviceBuffer<uint32_t> dImage;
        DeviceBuffer<float> dFilter, dOutput;
        KernelTimer timer(timings != nullptr);
        if (!dImage.allocate(image_elems) || !dFilter.allocate(filter_elems) ||
            !dOutput.allocate(image_elems) || !dImage.upload(image) ||
            !dFilter.upload(filter) || !timer.start())
            return LAB6_RUNTIME_ERROR;
        status = launch(dImage.get(), dFilter.get(), dOutput.get(), M, K);
        if (status != LAB6_OK)
            return status;
        if (!timer.stop() || !LAB6_CUDA_OK(cudaDeviceSynchronize()) ||
            !timer.elapsed_ms(&compute_ms) || !dOutput.download(output))
            return LAB6_RUNTIME_ERROR;
    }
    if (timings) {
        timings->compute_ms = compute_ms;
        timings->end_to_end_ms = lab6_now_ms() - begin_ms;
    }
    return LAB6_OK;
}

} // namespace lab6

#endif
