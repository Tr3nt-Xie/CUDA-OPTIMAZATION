#include "common/timer.h"
#include "common/validate.h"
#include "lab6.h"

int lab6_convolve_cpu(const uint32_t *image, const float *filter, float *output,
                      int M, int K, lab6_timings *timings) {
    size_t image_elems = 0;
    size_t filter_elems = 0;
    int status = lab6_check_convolve_args(image, filter, output, M, K,
                                          &image_elems, &filter_elems);
    if (status != LAB6_OK)
        return status;

    const int r = K / 2;
    const double begin_ms = lab6_now_ms();
    for (int y = 0; y < M; y++) {
        for (int x = 0; x < M; x++) {
            float sum = 0.0f;
            for (int i = 0; i < K; i++) {
                const int iy = y + r - i;
                if (iy < 0 || iy >= M)
                    continue;
                for (int j = 0; j < K; j++) {
                    const int ix = x + r - j;
                    if (ix < 0 || ix >= M)
                        continue;
                    sum += filter[(size_t)i * K + j] *
                           (float)image[(size_t)iy * M + ix];
                }
            }
            output[(size_t)y * M + x] = sum;
        }
    }
    if (timings) {
        timings->compute_ms = lab6_now_ms() - begin_ms;
        timings->end_to_end_ms = timings->compute_ms;
    }
    return LAB6_OK;
}
