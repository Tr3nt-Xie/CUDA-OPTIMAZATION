#include "harness.h"

static const bench_impl kImpls[] = {
    {"naive", lab6_matmul_naive, nullptr},
    {"tiled", lab6_matmul_tiled, nullptr},
    {"optimized", lab6_matmul_optimized, nullptr},
    {"cublas", lab6_matmul_cublas, nullptr},
    {"cuda", nullptr, lab6_convolve_cuda},
};

int main(int argc, char **argv) {
    return bench_main(argc, argv, "bench_cuda", kImpls,
                      static_cast<int>(sizeof kImpls / sizeof kImpls[0]));
}
