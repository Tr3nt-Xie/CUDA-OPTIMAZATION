# Member B — Parts 4, 6.2 and 8.2

**Measurement date:** October 13, 2026, PDT. **Platform:** the shared AWS us-west-2 g4dn.2xlarge host with an NVIDIA Tesla T4, the same host as `report/A.md`. **Run ID:** `B-20261013-232059` (UTC), source commit `93e9f2d`. Build: CUDA Toolkit 12.8.93, `NVCCFLAGS='-O2 -std=c++14 -arch=sm_75'`.

**测量日期：** 2026 年 10 月 13 日（PDT）。**平台：** 与 `report/A.md` 相同的共用 AWS us-west-2 g4dn.2xlarge / Tesla T4 主机。**批次：** `B-20261013-232059`（UTC），源码 commit `93e9f2d`。构建：CUDA Toolkit 12.8.93，`NVCCFLAGS='-O2 -std=c++14 -arch=sm_75'`。

All values are mean ± sample standard deviation over three measured calls after one warm-up, in milliseconds, under the timing scopes of `docs/CONTRACTS.md`. Raw data: [matrix](../results/raw/B-20261013-232059-matrix-aws/), [convolution](../results/raw/B-20261013-232059-conv-aws/), [tuning sweeps](../results/raw/B-20261013-232059-tuning-aws/). Each `environment.txt` reports the worktree as dirty; the only uncommitted items it lists are the result directories themselves.

所有数值为一次预热后三次计时的均值 ± 样本标准差，单位毫秒，计时范围按 `docs/CONTRACTS.md`。原始数据见上述三个目录。各 `environment.txt` 显示工作区为 dirty，其中列出的未提交项只有结果目录本身。

## Part 4: shared-memory tiled multiplication

`src/matrix/tiled.cu` follows the kernel in the assignment. A 16×16 block computes a 16×16 tile of C. The dot product is processed in steps of 16: in each step every thread copies one element of A and one of B into shared memory, the block synchronizes, and each thread performs 16 multiply-adds from shared memory. A second synchronization keeps the tiles from being overwritten while other threads still read them. Positions outside the matrix are loaded as zero, so N need not be a multiple of 16.

`src/matrix/tiled.cu` 按实验文档的内核实现。一个 16×16 线程块计算 C 的一个 16×16 分块。点积按 16 为一段处理：每段中每个线程把 A、B 各一个元素拷入共享内存，线程块同步后，每个线程从共享内存做 16 次乘加；第二次同步防止其他线程仍在读取时分块被覆盖。矩阵范围以外的位置按零载入，因此 N 不必是 16 的倍数。

| N | naive compute | tiled compute | optimized compute | naive end-to-end | tiled end-to-end | optimized end-to-end |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.066 ± 0.001 | 0.047 ± 0.001 | 0.031 ± 0.003 | 0.550 ± 0.029 | 0.524 ± 0.015 | 0.503 ± 0.017 |
| 512 | 0.442 ± 0.004 | 0.287 ± 0.004 | 0.126 ± 0.004 | 1.857 ± 0.007 | 1.695 ± 0.012 | 1.549 ± 0.007 |
| 1024 | 3.515 ± 0.001 | 2.175 ± 0.003 | 0.541 ± 0.002 | 6.951 ± 0.050 | 5.542 ± 0.020 | 3.912 ± 0.014 |
| 2048 | 31.545 ± 0.978 | 18.667 ± 0.445 | 3.823 ± 0.021 | 44.035 ± 1.007 | 31.246 ± 0.574 | 16.373 ± 0.208 |

**Table B1.** Matrix multiplication on the T4, ms. The naive kernel is A's implementation, re-measured in this batch so that all three share the same inputs and session. All 36 measured calls passed the correctness check.

**表 B1。** T4 上的矩阵乘法耗时（毫秒）。naive 为 A 的实现，在本批次中重新测量，使三者使用相同输入并处于同一次会话。36 次计时调用全部通过正确性检查。

| N | naive ÷ tiled, compute | tiled ÷ optimized, compute | naive ÷ optimized, compute | naive ÷ tiled, end-to-end | naive ÷ optimized, end-to-end | CPU ÷ tiled, end-to-end | CPU ÷ optimized, end-to-end |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 256 | 1.41× | 1.50× | 2.11× | 1.05× | 1.09× | 38.0× | 39.6× |
| 512 | 1.54× | 2.27× | 3.50× | 1.10× | 1.20× | 112.2× | 122.8× |
| 1024 | 1.62× | 4.02× | 6.50× | 1.25× | 1.78× | 585.4× | 829.4× |
| 2048 | 1.69× | 4.88× | 8.25× | 1.41× | 2.69× | 1937.7× | 3697.9× |

**Table B2.** Ratios of mean times, computed from unrounded means. CPU is A's single-thread baseline from `report/A.md` Table 1 (same host, same inputs, measured October 6).

**表 B2。** 平均耗时之比，由未取整的均值计算。CPU 为 `report/A.md` 表 1 中 A 的单线程基线（同一主机、相同输入，10 月 6 日测量）。

### Analysis question Q3

**Q3. How much speedup is gained by tiling over naive CUDA?** Counting only the kernel, tiling is 1.41×, 1.54×, 1.62× and 1.69× faster at N=256, 512, 1024 and 2048. Including allocation, copies, synchronization and cleanup, the gain is 1.05×, 1.10×, 1.25× and 1.41×, because at small N most of the end-to-end time lies outside the kernel (91% for the tiled kernel at N=256, 40% at N=2048).

**Q3. 分块相对 naive CUDA 的加速有多少？** 只看内核，N=256、512、1024、2048 时分别快 1.41×、1.54×、1.62×、1.69×。计入分配、拷贝、同步和释放后为 1.05×、1.10×、1.25×、1.41×：小尺寸时整体耗时大部分在内核之外（tiled 在 N=256 时为 91%，N=2048 时为 40%）。

The gain is far below 16×, although tiling cuts global-memory reads per multiply-add by that factor. Two effects are consistent with this, though neither was isolated with a profiler. First, the naive kernel's repeated reads need not all reach device memory, since caches can serve part of them. Second, the tiled inner loop still performs two shared-memory reads for every multiply-add, and synchronizes the block twice per step. The Part 6.2 result supports the second point: reducing shared-memory reads per multiply-add by a factor of eight made the kernel 4.88× faster at N=2048.

加速比远小于 16×，尽管分块使每次乘加对应的全局内存读取减少到 1/16。有两个因素与此相符，但均未用性能分析器单独验证：其一，naive 内核的重复读取不一定都到达显存，缓存可以承担一部分；其二，tiled 的内层循环每次乘加仍要读两次共享内存，且每段要同步两次。Part 6.2 的结果支持第二点：把每次乘加对应的共享内存读取减少到 1/8 后，N=2048 时内核快了 4.88×。

## Part 6.2: further optimization

`src/matrix/optimized.cu` applies register blocking. A block computes a BM×BN tile of C, and each thread computes a small sub-tile (8×8 for N≥512) whose partial sums stay in registers. For every k the thread reads 8 values of A and 8 of B from shared memory and forms their outer product: 64 multiply-adds for 16 reads, against 1 for 2 in Part 4.

`src/matrix/optimized.cu` 采用寄存器分块。一个线程块计算 C 的 BM×BN 分块，每个线程计算其中一个小子块（N≥512 时为 8×8），部分和保存在寄存器中。对每个 k，线程从共享内存读取 A 的 8 个值和 B 的 8 个值并做外积：16 次读取完成 64 次乘加，Part 4 为 2 次读取完成 1 次。

Two further changes:

- The A tile is stored transposed in shared memory, so the values a thread needs are adjacent, as they already are for B.
- When N is a multiple of the tile size, global loads and stores move four floats per instruction (`float4`).

另有两项改动：

- A 的分块在共享内存中转置存放，线程需要的值因此相邻，与 B 的情况一致。
- N 为分块大小的整数倍时，全局内存的读写每条指令传输 4 个 float（`float4`）。

**Any N.** The fast path assumes N is a multiple of the tile size. A second instantiation of the same kernel checks bounds: loads outside the matrix read as zero and stores outside it are skipped. The launch code picks the fast path when it applies. The formal sizes all use the fast path; the bounds-checked path is exercised by the N=2, 3 and 31 self-tests and by additional runs at N=1, 17, 100, 129, 300, 511, 513, 1000, 1023, 1025 and 2049, all of which passed.

**任意 N。** 快速路径要求 N 是分块大小的整数倍。同一内核的另一个实例带边界检查：矩阵外的读取视为零，矩阵外的写入被跳过。启动代码在条件满足时选择快速路径。正式尺寸均走快速路径；边界检查路径由 N=2、3、31 的自检以及 N=1、17、100、129、300、511、513、1000、1023、1025、2049 的补充运行覆盖，全部通过。

**Tile size.** Larger tiles reuse each loaded value more, but a small matrix then has too few blocks to occupy every SM. The sweep below, run on the T4, selected three settings: 128×128 tiles with 8×8 outputs per thread from N=1024, 64×64 tiles with 8×8 from N=512, and 64×64 tiles with 4×4 below that.

**分块大小。** 分块越大，每个载入值的复用次数越多，但小矩阵的线程块数量会不足以占满全部 SM。下表是在 T4 上的扫描，据此选定三档：N≥1024 用 128×128 分块、每线程 8×8；N≥512 用 64×64 分块、每线程 8×8；更小的 N 用 64×64 分块、每线程 4×4。

| Tile, BK, outputs per thread | N=256 | N=512 | N=1024 | N=2048 |
|---|---:|---:|---:|---:|
| 128×128, 8, 8×8 | 0.074 | 0.141 | 0.547 | **3.714** |
| 128×128, 16, 8×8 | 0.071 | 0.133 | **0.540** | 3.749 |
| 64×64, 8, 8×8 | 0.058 | 0.119 | 0.711 | 4.087 |
| 64×64, 16, 8×8 | 0.042 | **0.101** | 0.607 | 3.895 |
| 64×64, 8, 4×4 | 0.038 | 0.109 | 0.637 | 4.646 |
| 64×64, 16, 4×4 | **0.030** | 0.105 | 0.626 | 4.548 |
| 32×32, 8, 4×4 | 0.031 | 0.114 | 0.723 | 5.653 |
| 128×128, 16, 4×4 | 0.070 | 0.134 | 0.665 | 4.609 |

**Table B3.** Tile sweep on the T4, kernel time in ms, mean of 15 calls after one warm-up. This is a tuning sweep, separate from the formal batch; the build adds an environment-variable switch that is not in the committed source (patch and log in the tuning directory). At N=2048 the two 128×128, 8×8 rows differ by 1%; BK=16 was kept for all sizes.

**表 B3。** T4 上的分块扫描，内核耗时（毫秒），一次预热后 15 次调用的均值。这是调参扫描，与正式批次分开；该构建加入了未提交到源码的环境变量开关（补丁和日志在 tuning 目录）。N=2048 时两个 128×128、8×8 配置相差 1%，各尺寸统一使用 BK=16。

**Result.** Against the Part 4 kernel, the optimized kernel is 1.50×, 2.27×, 4.02× and 4.88× faster at N=256, 512, 1024 and 2048 (Table B2); against naive, 2.11× to 8.25×. At N=2048 it performs 2N³ operations in 3.823 ms, about 4,490 GFLOP/s, compared with 920 for tiled and 545 for naive. End to end the gain over naive is smaller, 2.69× at N=2048, because the kernel is then only 23% of the call; the remaining 77% is allocation, copies, synchronization and cleanup, which this optimization does not touch. The comparison with cuBLAS belongs to Part 6.1 and Q4.

**结果。** 相对 Part 4 的内核，optimized 在 N=256、512、1024、2048 时分别快 1.50×、2.27×、4.02×、4.88×（表 B2）；相对 naive 快 2.11× 至 8.25×。N=2048 时它在 3.823 ms 内完成 2N³ 次运算，约 4,490 GFLOP/s，tiled 为 920，naive 为 545。整体耗时上相对 naive 的提升较小，N=2048 时为 2.69×，因为此时内核只占调用的 23%，其余 77% 是分配、拷贝、同步和释放，本优化不涉及这部分。与 cuBLAS 的比较属于 Part 6.1 和 Q4。

One difference between the batches is unexplained: at N=512 the formal batch measured 0.126 ms, while the 15-call sweep measured 0.101 ms for the same configuration. The formal value is the one reported.

两批数据之间有一处差异尚无解释：N=512 时正式批次为 0.126 ms，而同一配置在 15 次调用的扫描中为 0.101 ms。报告采用正式批次的数值。

**Sources.** The register-blocking structure, the transposed A tile and the vectorized memory access follow Simon Boehm, "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog" (siboehm.com, 2022). The reasoning that fewer resident threads can be faster when each thread has more independent work is from Vasily Volkov, "Better Performance at Lower Occupancy" (GTC 2010).

**方法来源。** 寄存器分块结构、A 分块转置和向量化访存参考 Simon Boehm 的 "How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog"（siboehm.com，2022）。"每个线程有更多独立运算时，驻留线程更少也可能更快"的依据来自 Vasily Volkov 的 "Better Performance at Lower Occupancy"（GTC 2010）。

**Not done.** Double buffering, warp-level tiling and Tensor Cores were not attempted. Results stay in FP32, as the team contract requires. The tile settings are specific to the T4: on a development machine with a different GPU the best settings differed, so they would need re-measuring on other hardware.

**未做的部分。** 没有尝试双缓冲、warp 级分块和 Tensor Core。按组内约定，运算保持 FP32。分块设置针对 T4：在使用另一款 GPU 的开发机上最优设置不同，换硬件需重新测量。

## Part 8.2: CUDA convolution

`src/convolution/cuda.cu` implements the team definition: true convolution with the filter flipped, stride 1, zero padding and same-size float32 output. It contains two kernels.

`src/convolution/cuda.cu` 实现组内定义：翻转滤波器的数学卷积、步长 1、零填充、同尺寸 float32 输出。文件包含两个内核。

- **Direct.** One thread per output pixel, reading the image and filter from global memory with the same bounds checks as the CPU loop.
- **Shared-memory window.** A 16×16 block computes a 16×16 tile of the output. Those outputs read a (16+K−1)² window of the image. The block first copies the window into shared memory, converting to float and writing zero outside the image, so the inner loop has no bounds checks; the filter is copied already flipped.

- **直接读取。** 每个线程计算一个输出像素，从全局内存读取图像和滤波器，边界检查与 CPU 循环相同。
- **共享内存窗口。** 一个 16×16 线程块计算输出的一个 16×16 分块，这些输出读取图像中 (16+K−1)² 的窗口。线程块先把窗口拷入共享内存，同时转换为 float 并在图像外写零，内层循环因此无需边界检查；滤波器以翻转后的顺序拷入。

| M=2048 | K=3 | K=5 | K=7 | K=9 | K=11 | K=15 | K=21 | K=31 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| shared-memory window | 0.262 | 0.389 | 0.682 | 1.082 | 1.589 | 2.899 | 5.609 | 12.091 |
| direct | 0.242 | 0.369 | 0.680 | 1.107 | 1.626 | 2.963 | 5.764 | 12.552 |

**Table B4.** The two kernels on the T4, kernel time in ms, mean of 15 calls after one warm-up, random input, mean filter (tuning comparison, log in the tuning directory, which also has M=512 and 1024).

**表 B4。** 两个内核在 T4 上的内核耗时（毫秒），一次预热后 15 次调用的均值，随机输入、均值滤波（调参比较，日志在 tuning 目录，其中还有 M=512 和 1024）。

The shared-memory kernel did not help at the formal filter sizes: the direct kernel was 5–8% faster at K=3 and 5 for M=2048, and the two were equal at K=7. From K=9 the shared-memory kernel was 2–4% faster. The implementation therefore uses the direct kernel below K=9 and the shared-memory kernel from K=9, falling back to the direct kernel if the window would exceed 48 KB of shared memory (K≥71). A likely reason, not verified with a profiler, is that the overlapping reads of neighbouring threads are already served from cache, so copying them into shared memory adds work without removing device-memory traffic.

共享内存内核在正式滤波器尺寸下没有带来提升：M=2048 时直接读取内核在 K=3、5 下快 5–8%，K=7 时两者相同；K≥9 时共享内存内核快 2–4%。因此实现在 K<9 时使用直接读取内核，K≥9 时使用共享内存内核；窗口超过 48 KB 共享内存（K≥71）时回退到直接读取内核。可能的原因（未用性能分析器验证）是相邻线程的重叠读取本已由缓存承担，再拷入共享内存只增加了工作量而没有减少显存访问。

| M | K | CPU (A) | CUDA compute | CUDA end-to-end | CPU ÷ CUDA compute | CPU ÷ CUDA end-to-end |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 3 | 5.307 | 0.022 | 1.110 | 242× | 4.8× |
| 512 | 5 | 12.713 | 0.034 | 1.113 | 375× | 11.4× |
| 512 | 7 | 23.411 | 0.054 | 1.123 | 430× | 20.8× |
| 1024 | 3 | 21.243 | 0.068 | 2.569 | 313× | 8.3× |
| 1024 | 5 | 50.981 | 0.102 | 2.602 | 499× | 19.6× |
| 1024 | 7 | 93.848 | 0.181 | 2.681 | 520× | 35.0× |
| 2048 | 3 | 85.219 | 0.242 | 8.495 | 353× | 10.0× |
| 2048 | 5 | 204.150 | 0.369 | 8.536 | 553× | 23.9× |
| 2048 | 7 | 375.698 | 0.677 | 8.889 | 555× | 42.3× |

**Table B5.** Mean-filter convolution on the T4, ms. Each entry is the mean over the three team images of the per-image means; per-image values differ by less than 5% and are in the raw CSV files. CPU values are A's single-thread results from `report/A.md` Table 2 on the same inputs. Ratios use unrounded means.

**表 B5。** T4 上的均值滤波卷积耗时（毫秒）。每项为三张组内图片各自均值的平均；各图片之间相差不到 5%，逐图数值见原始 CSV。CPU 为 `report/A.md` 表 2 中 A 在相同输入上的单线程结果。比值由未取整的均值计算。

The kernel is 242× to 555× faster than the CPU loop, but end to end the gain is 4.8× to 42.3×. Between 92% and 98% of each call lies outside the kernel: the call uploads M² four-byte pixels and downloads M² four-byte results, and allocates and frees the buffers. End-to-end time is therefore nearly independent of K (8.5–8.9 ms at M=2048) while CPU time grows with K², so the end-to-end advantage grows with the filter size.

内核比 CPU 循环快 242× 至 555×，但整体耗时只快 4.8× 至 42.3×。每次调用有 92% 至 98% 的时间在内核之外：需要上传 M² 个 4 字节像素、下载 M² 个 4 字节结果，并分配和释放缓冲区。因此整体耗时几乎与 K 无关（M=2048 时为 8.5–8.9 ms），而 CPU 耗时随 K² 增长，整体优势随滤波器增大而增大。

## Correctness

All checks use the team tolerance, `abs(got - ref) <= 1e-3 + 1e-4 * abs(ref)`, on the T4:

- `build/bench_cuda selftest --impl naive,tiled,optimized,cuda`: all passed, including the hand-computed cases, N=2/3/31, the asymmetric filter that separates convolution from correlation, and the `timings=NULL` path (log in the matrix directory).
- Matrix: 36 measured calls passed. Tiled and optimized have the same maximum absolute error as naive (0.00161 at N=2048).
- Convolution: 33 configurations and 99 measured calls passed, on the three team images, M=512/1024/2048 with K=3/5/7 mean filters, plus edge and sharpen at M=512, K=3. Maximum absolute error 1.73×10⁻⁴.
- The nine CUDA outputs at M=512, K=3 were compared element by element with A's saved CPU outputs in `/opt/lab6-a-reference/20261006-232110/outputs/`; all matched (maximum difference 6.1×10⁻⁵, zero for the edge filter). See `cpu-output-comparison.txt`.
- Both convolution kernels were exercised at M=1, 17, 31, 1000 with K=1, 3, 7, 9, 11, 49 and all three filters: 72 configurations passed.

全部检查使用组内容差 `abs(got - ref) <= 1e-3 + 1e-4 * abs(ref)`，在 T4 上完成：

- `build/bench_cuda selftest --impl naive,tiled,optimized,cuda`：全部通过，包括手算样例、N=2/3/31、区分卷积与互相关的非对称滤波器，以及 `timings=NULL` 路径（日志在 matrix 目录）。
- 矩阵：36 次计时调用全部通过；tiled 与 optimized 的最大绝对误差与 naive 相同（N=2048 时为 0.00161）。
- 卷积：33 个配置、99 次计时调用全部通过，覆盖三张组内图片、M=512/1024/2048 与 K=3/5/7 的均值滤波，以及 M=512、K=3 的边缘和锐化滤波。最大绝对误差 1.73×10⁻⁴。
- M=512、K=3 的九份 CUDA 输出与 A 保存在 `/opt/lab6-a-reference/20261006-232110/outputs/` 的 CPU 输出逐元素比较，全部吻合（最大差值 6.1×10⁻⁵，边缘滤波为零），见 `cpu-output-comparison.txt`。
- 两个卷积内核在 M=1、17、31、1000 与 K=1、3、7、9、11、49 及三种滤波器下均被覆盖：72 个配置全部通过。

**Host consistency.** The host was stopped and restarted between A's batch and this one. Naive at N=2048 measured 31.545 ms here against 31.391 ms in A's batch. CPU spot checks were 1.6% to 3.7% slower than A's values (matrix N=256 and 512; convolution geometry, M=1024, K=5). The CPU ratios above use A's formal values, so they are slightly lower than same-session ratios would be.

**主机一致性。** A 的批次与本批次之间主机曾停机重启。本批次 naive 在 N=2048 为 31.545 ms，A 的批次为 31.391 ms。CPU 抽查比 A 的数值慢 1.6% 至 3.7%（矩阵 N=256、512；卷积 geometry、M=1024、K=5）。上文的 CPU 比值采用 A 的正式数值，因此略低于同一会话内的比值。

## Reproduce

```sh
export NVCCFLAGS='-O2 -std=c++14 -arch=sm_75'
make clean && make -j4 all
build/bench_cuda selftest --impl naive,tiled,optimized,cuda
build/bench_cuda matrix --impl naive,tiled,optimized --member B --csv "$run_dir/measurements.csv"
build/bench_cuda conv --impl cuda --image "$inputs/geometry_1024.u32" --m 1024 --k 5 --filter mean \
    --member B --csv "$run_dir/geometry-M1024-K5-mean.csv"
```

The exact commands of each batch are in its `commands.txt`.

各批次的完整命令见其 `commands.txt`。

## Pending

- Comparison of the optimized kernel with cuBLAS (Part 6.1, Q4), once C's implementation is available.
- Video segment for Parts 4, 6.2 and 8.2.
- Clean-checkout build and reproduction check of the whole repository after all parts are merged.

## 待完成

- C 的实现就绪后，对比 optimized 内核与 cuBLAS（Part 6.1、Q4）。
- Parts 4、6.2、8.2 的视频片段。
- 全部内容合并后，对整个仓库做干净检出的构建与复现检查。
