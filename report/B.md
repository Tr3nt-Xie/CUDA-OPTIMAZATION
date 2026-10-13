# Member B — Parts 4, 6.2 and 8.2

**Status:** the three implementations are complete and pass every correctness check. Formal timing on the shared AWS T4 host is **pending**, so this file contains no formal measurements, no speedups and no answer to Q3 yet. Development and verification ran on a separate machine (NVIDIA RTX 5070 Ti, driver 610.88, CUDA Toolkit 13.3, Ubuntu 24.04 under WSL2, `NVCCFLAGS='-O2 -std=c++14 -arch=native'`). Numbers from that machine appear only where they explain a design choice; they must not be compared with the T4 data in `report/A.md`.

**状态：** 三个实现均已完成并通过全部正确性检查。共用 AWS T4 主机上的正式计时**尚未进行**，因此本文件暂无正式测量值、加速比和 Q3 答案。开发与验证在另一台机器上完成（NVIDIA RTX 5070 Ti，驱动 610.88，CUDA Toolkit 13.3，WSL2 下的 Ubuntu 24.04，`NVCCFLAGS='-O2 -std=c++14 -arch=native'`）。该机器的数字只用于说明设计取舍，不能与 `report/A.md` 中的 T4 数据比较。

## Part 4: shared-memory tiled multiplication

`src/matrix/tiled.cu` follows the kernel in the assignment. A 16×16 block computes a 16×16 tile of C. The dot product is processed in steps of 16: in each step every thread copies one element of A and one of B into shared memory, the block synchronizes, and each thread performs 16 multiply-adds from shared memory. A second synchronization keeps the tiles from being overwritten while other threads still read them. Positions outside the matrix are loaded as zero, so N need not be a multiple of 16.

`src/matrix/tiled.cu` 按实验文档的内核实现。一个 16×16 线程块计算 C 的一个 16×16 分块。点积按 16 为一段处理：每段中每个线程把 A、B 各一个元素拷入共享内存，线程块同步后，每个线程从共享内存做 16 次乘加；第二次同步防止其他线程仍在读取时分块被覆盖。矩阵范围以外的位置按零载入，因此 N 不必是 16 的倍数。

Each value read from global memory is used 16 times instead of once. The remaining cost is in shared memory: the inner loop performs one multiply-add for every two shared-memory reads. Part 6.2 addresses that.

每个从全局内存读入的数值被使用 16 次而不是 1 次。剩余开销在共享内存：内层循环每做一次乘加要读两次共享内存，这正是 Part 6.2 要处理的问题。

## Part 6.2: further optimization

`src/matrix/optimized.cu` applies register blocking. A block computes a BM×BN tile of C, and each thread computes an 8×8 sub-tile whose 64 partial sums stay in registers. For every k the thread reads 8 values of A and 8 of B from shared memory and forms their outer product: 64 multiply-adds for 16 reads, against 1 for 2 in Part 4.

`src/matrix/optimized.cu` 采用寄存器分块。一个线程块计算 C 的 BM×BN 分块，每个线程计算其中 8×8 的子块，64 个部分和保存在寄存器中。对每个 k，线程从共享内存读取 A 的 8 个值和 B 的 8 个值并做外积：16 次读取完成 64 次乘加，Part 4 为 2 次读取完成 1 次。

Two further changes:

- The A tile is stored transposed in shared memory, so the 8 values a thread needs are adjacent, as they already are for B.
- When N is a multiple of the tile size, global loads and stores move four floats per instruction (`float4`).

另有两项改动：

- A 的分块在共享内存中转置存放，线程需要的 8 个值因此相邻，与 B 的情况一致。
- N 为分块大小的整数倍时，全局内存的读写每条指令传输 4 个 float（`float4`）。

**Any N.** The fast path assumes N is a multiple of the tile size. A second instantiation of the same kernel checks bounds: loads outside the matrix read as zero and stores outside it are skipped. The launch code picks the fast path when it applies. This is what lets the kernel pass the N=2, 3 and 31 self-tests; the formal sizes 256–2048 all use the fast path.

**任意 N。** 快速路径要求 N 是分块大小的整数倍。同一内核的另一个实例带边界检查：矩阵外的读取视为零，矩阵外的写入被跳过。启动代码在条件满足时选择快速路径。N=2、3、31 的自检因此可以通过；正式尺寸 256–2048 均走快速路径。

**Tile size.** Larger tiles reuse each loaded value more, but a small matrix then has too few blocks to occupy every SM. The implementation uses 64×64 tiles (64 threads per block) below N=2048 and 128×128 tiles (256 threads) from N=2048, both with BK=16. The choice came from the sweep below on the development machine. It has not been re-tuned on the T4, which has fewer SMs, and should be re-checked there before the formal run.

**分块大小。** 分块越大，每个载入值的复用次数越多，但小矩阵的线程块数量会不足以占满全部 SM。实现在 N<2048 时使用 64×64 分块（每块 64 线程），N≥2048 时使用 128×128 分块（256 线程），BK 均为 16。该选择来自开发机上的下列扫描；尚未在 SM 数量更少的 T4 上重新调整，正式测量前应在 T4 上复核。

| Tile, BK, per-thread sub-tile | N=256 | N=512 | N=1024 | N=2048 |
|---|---:|---:|---:|---:|
| 128×128, 8, 8×8 | 0.062 | 0.094 | 0.154 | 0.707 |
| 128×128, 16, 8×8 | 0.057 | 0.091 | 0.146 | **0.696** |
| 64×64, 8, 8×8 | 0.044 | 0.068 | 0.143 | 0.761 |
| 64×64, 16, 8×8 | **0.041** | 0.062 | **0.135** | 0.740 |
| 64×64, 8, 4×4 | 0.042 | 0.060 | 0.155 | 0.874 |
| 32×32, 8, 4×4 | 0.041 | 0.058 | 0.168 | 0.994 |

**Table B1.** Development-machine tuning sweep, kernel time in ms, mean of 15 calls after one warm-up. Not formal data.

**表 B1。** 开发机调参扫描，内核耗时（毫秒），一次预热后 15 次调用的均值，非正式数据。

**Sources.** The register-blocking structure, the transposed A tile and the vectorized memory access follow Simon Boehm, "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog" (siboehm.com, 2022). The reasoning that fewer resident threads can be faster when each thread has more independent work is from Vasily Volkov, "Better Performance at Lower Occupancy" (GTC 2010).

**方法来源。** 寄存器分块结构、A 分块转置和向量化访存参考 Simon Boehm 的 "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog"（siboehm.com，2022）。"每个线程有更多独立运算时，驻留线程更少也可能更快"的依据来自 Vasily Volkov 的 "Better Performance at Lower Occupancy"（GTC 2010）。

**Not done.** Double buffering, warp-level tiling and Tensor Cores were not attempted. Results stay in FP32, as the team contract requires.

**未做的部分。** 没有尝试双缓冲、warp 级分块和 Tensor Core。按组内约定，运算保持 FP32。

## Part 8.2: CUDA convolution

`src/convolution/cuda.cu` implements the team definition: true convolution with the filter flipped, stride 1, zero padding and same-size float32 output. A 16×16 block computes a 16×16 tile of the output. Those outputs read a (16+K−1)² window of the image, the tile plus K/2 pixels on each side. The block first copies that window into shared memory, converting to float and writing zero for positions outside the image, so the inner loop has no bounds checks. The filter is copied into shared memory already flipped, so the inner loop walks both arrays forwards.

`src/convolution/cuda.cu` 实现组内定义：翻转滤波器的数学卷积、步长 1、零填充、同尺寸 float32 输出。一个 16×16 线程块计算输出的一个 16×16 分块，这些输出读取图像中 (16+K−1)² 的窗口，即分块加上四周各 K/2 个像素。线程块先把该窗口拷入共享内存，同时转换为 float 并在图像外的位置写零，内层循环因此无需边界检查。滤波器以翻转后的顺序拷入共享内存，内层循环对两个数组都顺序访问。

The window size depends on K, so the shared memory is sized at launch. If the window and filter exceed 48 KB (K of 71 and above), a second kernel reads directly from global memory. The team test program limits K to 49, so that path cannot be reached through it; it was verified by temporarily forcing it and running the self-tests.

窗口大小取决于 K，因此共享内存大小在启动时指定。窗口与滤波器超过 48 KB（K≥71）时，改用直接读取全局内存的第二个内核。组内测试程序限制 K≤49，无法经它触发该路径；该路径通过临时强制启用并运行自检完成验证。

On the development machine the shared-memory kernel and the direct kernel were within noise for M=512 and about 10–20% apart at M=2048 (K=7: 0.165 ms against 0.212 ms), so the shared-memory kernel is the default. This comparison is not formal data.

在开发机上，共享内存内核与直接读取内核在 M=512 时差异在噪声范围内，M=2048 时相差约 10–20%（K=7：0.165 ms 对 0.212 ms），因此默认使用共享内存内核。该比较不是正式数据。

## Correctness

All checks use the team reference and tolerance, `abs(got - ref) <= 1e-3 + 1e-4 * abs(ref)`, on the development machine:

- `build/bench_cuda selftest --impl naive,tiled,optimized,cuda`: all passed, including the hand-computed cases, N=2/3/31, the asymmetric filter that separates convolution from correlation, and the `timings=NULL` path.
- Matrix, N=256/512/1024/2048: tiled and optimized passed with the same maximum absolute error as naive (0.00161 at N=2048).
- Additional sizes N=1, 17, 100, 129, 1000, 2049 (bounds-checked path of the optimized kernel): passed.
- Convolution, the nine M×K mean-filter configurations on random input: passed.
- Convolution on the three team images at M=1024 with mean, edge and sharpen filters at K=3/5/7: 27 configurations passed.

全部检查使用组内参考与容差 `abs(got - ref) <= 1e-3 + 1e-4 * abs(ref)`，在开发机上完成：

- `build/bench_cuda selftest --impl naive,tiled,optimized,cuda`：全部通过，包括手算样例、N=2/3/31、区分卷积与互相关的非对称滤波器，以及 `timings=NULL` 路径。
- 矩阵 N=256/512/1024/2048：tiled 与 optimized 均通过，最大绝对误差与 naive 相同（N=2048 时为 0.00161）。
- 补充尺寸 N=1、17、100、129、1000、2049（optimized 内核的边界检查路径）：通过。
- 卷积，九组 M×K 均值滤波配置（随机输入）：通过。
- 三张组内图片在 M=1024 下配合均值、边缘、锐化滤波器，K=3/5/7：27 个配置全部通过。

## Reproduce

```sh
export NVCCFLAGS='-O2 -std=c++14 -arch=sm_75'   # T4; use the arch of the host
make -j4 all
build/bench_cuda selftest --impl tiled,optimized,cuda
build/bench_cuda matrix --impl naive,tiled,optimized --member B --csv "$run_dir/matrix.csv"
build/bench_cuda conv --impl cuda --member B --csv "$run_dir/conv.csv"
```

## Pending

- Formal measurements on the shared AWS T4 host: matrix (tiled, optimized against A's CPU and naive data) and convolution (against A's CPU data on the same inputs), with raw CSV and `environment.txt` under `results/raw/`.
- Re-check the tile-size choice on the T4.
- Q3 (tiled against naive) and the before/after numbers for Part 6.2, from the T4 data.
- Clean-checkout build and reproduction check of the whole repository.

## 待完成

- 共用 AWS T4 主机上的正式测量：矩阵（tiled、optimized，对照 A 的 CPU 与 naive 数据）与卷积（同一输入下对照 A 的 CPU 数据），原始 CSV 与 `environment.txt` 存入 `results/raw/`。
- 在 T4 上复核分块大小的选择。
- 依据 T4 数据回答 Q3（tiled 对比 naive），并给出 Part 6.2 优化前后的数据。
- 对整个仓库做干净检出后的构建与复现检查。
