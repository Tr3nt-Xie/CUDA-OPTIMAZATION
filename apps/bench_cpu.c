#include "harness.h"

static const bench_impl kImpls[] = {
    {"cpu", lab6_matmul_cpu, lab6_convolve_cpu},
};

int main(int argc, char **argv) {
    return bench_main(argc, argv, "bench_cpu", kImpls,
                      (int)(sizeof kImpls / sizeof kImpls[0]));
}
