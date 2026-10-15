# Member A — Parts 1–3 and 8.1

**Measurement date:** October 6, 2026, PDT. **Platform:** AWS us-west-2, g4dn.2xlarge, NVIDIA Tesla T4.

## Parts 1–2: CPU and naive CUDA matrix multiplication

The CPU baseline uses a single-threaded i–j–k loop. The naive CUDA kernel assigns one output element to each thread, using 16×16 thread blocks and bounds checks. Both compute C=A×B with row-major float32 matrices generated from seed 542. The host is an 8-vCPU Intel Xeon Platinum 8259CL instance with 32 GiB RAM and a T4 reporting 15,360 MiB VRAM. Measurements use Ubuntu 22.04.5, GCC 11.4.0 with `-O2`, and CUDA Toolkit 12.8.93 with `-O2 -arch=sm_75`; the driver is 595.91.07.

Each configuration has one warm-up and three measured calls. Table 1 reports arithmetic mean ± sample standard deviation, retaining every sample. CPU time covers the multiplication loop; CUDA compute time uses events around the kernel. CUDA end-to-end time covers device allocation, host-to-device copies, kernel execution, synchronization, device-to-host copy and cleanup. Input generation, the double-precision reference, correctness checks and file I/O are excluded. These are warmed calls, not process startup or first CUDA-context initialization.

| N | CPU loop (ms) | CUDA compute (ms) | CUDA end-to-end (ms) | CPU / CUDA end-to-end |
|---:|---:|---:|---:|---:|
| 256 | 19.931 ± 0.016 | 0.065 ± 0.000 | 0.527 ± 0.005 | 37.80× |
| 512 | 190.189 ± 0.468 | 0.444 ± 0.002 | 1.932 ± 0.045 | 98.44× |
| 1024 | 3,244.281 ± 1.925 | 3.466 ± 0.003 | 7.025 ± 0.052 | 461.82× |
| 2048 | 60,545.615 ± 533.899 | 31.391 ± 0.901 | 43.610 ± 0.889 | 1388.36× |

**Table 1.** Same-host CPU and naive CUDA matrix results. Speedup is the ratio of mean CPU time to mean CUDA end-to-end time. N=512/1024/2048 follow the assignment; N=256 is the team's additional baseline.

![CPU and naive CUDA matrix runtimes with sample standard deviation](../results/figures/A-20261006-232110-matrix-aws/matrix-runtime.png)

**Figure 1.** Runtime increases with matrix size; the vertical axis is logarithmic. The arithmetic work grows as N³, but measured CPU time does not follow an exact eightfold increase when N doubles. The baseline reads B with a stride in its inner loop; memory-system effects are a plausible contributor, but were not isolated with a profiler. GPU compute time grows from 0.065 ms at N=256 to 31.391 ms at N=2048. The end-to-end speedup increases from 37.80× to 1388.36× against this specific single-thread baseline.

All 24 formal measured calls and 24 supplementary small-size calls passed `abs(error) <= 1e-3 + 1e-4 * abs(reference)`, with nonfinite outputs rejected. The largest formal absolute error was 0.00160635; correctness uses the combined absolute and relative tolerance. CPU and naive CUDA self-tests also passed, including non-block-aligned matrices. Supplementary N=16/32/64/128 measurements support the Q2 discussion below.

### Analysis questions Q1 and Q2

**Q1. How does performance change as matrix size increases?** All three timings grow with N, but at different rates. Each doubling of N multiplies the arithmetic work by eight. The CPU loop grew by 9.5×, 17.1× and 18.7× across the three formal doublings, faster than the arithmetic alone. This is consistent with the strided reads of B leaving the caches at larger N, although no profiler isolated the cause. The naive kernel grew by 6.8×, 7.8× and 9.1×, approaching the cubic rate once the GPU is fully occupied. GPU end-to-end time grew more slowly, by 3.7×, 3.6× and 6.2×. At small N it is dominated by work outside the kernel, whose share of end-to-end time falls from 87.6% at N=256 to 28.0% at N=2048 (Figure 3). As a result, the end-to-end speedup over this single-thread CPU baseline rises from 37.80× to 1388.36×.

**Q2. At what point does the GPU significantly outperform the CPU?** The answer depends on the timing scope (Figure 2). Counting only the kernel, the GPU is slower at N=16 (0.35×) and already faster at N=32 (3.28×). Including allocation, copies, synchronization and cleanup, the GPU needs roughly 0.17–0.19 ms even for N=16–64, so the CPU stays faster at N=32 (0.18×). The measured end-to-end crossover lies between N=32 and N=64, where the GPU is only 1.25× faster. A clear advantage appears by N=128 (8.73×) and grows to 37.80× at N=256. This bracket applies to this T4 host, this naive kernel and warmed calls; three samples per size do not establish an exact threshold.

![Supplementary small-size sweep and CPU/GPU end-to-end ratio](../results/figures/A-20261006-232110-matrix-aws/matrix-crossover.png)

**Figure 2.** Left: CPU loop and GPU end-to-end time for N=16–256. Right: ratio of mean CPU time to mean GPU end-to-end time; the dashed line marks equal time. N=16/32/64/128 are supplementary sizes added for Q2; N=256 reuses the formal batch.

![Naive CUDA end-to-end time split into kernel and remaining wrapper time](../results/figures/A-20261006-232110-matrix-aws/gpu-overhead.png)

**Figure 3.** Mean end-to-end time split into kernel compute and the remainder (allocation, copies, synchronization and cleanup). The remainder is end-to-end minus kernel time, not a separately measured transfer time.

## Part 3: Cloud GPU execution

The CUDA environment was also verified inside the official `nvidia/cuda:12.8.1-devel-ubuntu22.04` GPU container. A fresh in-container build, naive CUDA self-tests, and nine measured calls across N=256/512/1024 all passed. The container had no network access during this verification and was removed after exit. Table 1 contains the separate host measurements, so container checks are not mixed into the comparison.

## Part 8.1: CPU convolution

### Images and filters

The three original procedural grayscale images contain geometric edges, fine textures and a soft synthetic scene. Each 2048×2048 original was converted to M=512/1024/2048 unsigned 32-bit pixel arrays in [0,255]. The CPU implementation performs true convolution with stride 1, zero padding and same-size float32 output. Filter weights and accumulation are float32.

| Filter | K×K weights | Purpose |
|---|---|---|
| Mean | Every entry is 1/K² | Smooth local intensity variation |
| Edge | Center K²−1; all others −1 | Highlight intensity transitions |
| Sharpen | 2 × identity kernel − mean kernel | Increase local contrast |

![CPU convolution of three images with mean, edge, and sharpening filters](../results/figures/A-20261006-232110-part81-cpu-aws/filter-comparison.png)

**Figure 4.** CPU results at M=512, K=3. Mean filtering attenuates fine texture, edge detection responds to boundaries, and sharpening increases local contrast. PNG display rounds and clips to [0,255]; the edge column first takes the absolute value. Validation uses the signed, unclipped float32 output. Border responses are expected with zero padding.

### Correctness and CPU measurements

Each image was tested with all nine combinations of M=512/1024/2048 and K=3/5/7 using mean filters, plus edge and sharpen at M=512, K=3. All 33 configurations and 99 measured calls passed the double-precision C reference, with maximum absolute error 1.73343×10⁻⁴. Nine saved display-case outputs also passed an independent NumPy float64 convolution, with maximum absolute error 7.56532×10⁻⁵. The self-test includes an asymmetric filter to distinguish convolution from cross-correlation.

The following CPU times were collected on the same AWS host as Table 1, using GCC 11.4.0 and `-O2`. Values are mean ± sample standard deviation over three measured calls after one warm-up. Loading images, building the reference, validation and PNG export are excluded. CPU compute and end-to-end fields both cover the convolution loop and therefore have identical values.

| Image | M | K=3 (ms) | K=5 (ms) | K=7 (ms) |
|---|---:|---:|---:|---:|
| geometry | 512 | 5.317 ± 0.014 | 12.709 ± 0.014 | 23.396 ± 0.020 |
| geometry | 1024 | 21.262 ± 0.026 | 51.038 ± 0.090 | 93.921 ± 0.121 |
| geometry | 2048 | 85.031 ± 0.034 | 204.197 ± 0.113 | 375.848 ± 0.071 |
| frequency | 512 | 5.298 ± 0.005 | 12.729 ± 0.024 | 23.447 ± 0.014 |
| frequency | 1024 | 21.280 ± 0.011 | 51.007 ± 0.019 | 93.816 ± 0.022 |
| frequency | 2048 | 85.081 ± 0.081 | 204.141 ± 0.078 | 375.455 ± 0.074 |
| soft_scene | 512 | 5.307 ± 0.016 | 12.702 ± 0.004 | 23.390 ± 0.005 |
| soft_scene | 1024 | 21.187 ± 0.011 | 50.899 ± 0.018 | 93.807 ± 0.016 |
| soft_scene | 2048 | 85.546 ± 0.979 | 204.112 ± 0.133 | 375.790 ± 0.446 |

**Table 2.** Mean-filter CPU convolution times. For fixed K, doubling M increases time by approximately four times, consistent with four times as many output pixels. Increasing K increases the work per pixel. These results provide the CPU baseline for the later CUDA convolution comparison.
