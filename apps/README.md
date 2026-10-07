# C/CUDA 测试程序（A 负责框架，占位待实现）

算法放 src；这里负责参数解析、生成/读取输入、调用 `lab6_*`、和 CPU 参考比对、预热 1 次 + 计时 3 次、输出 CSV。

| 文件 | 内容 | 构建 |
|---|---|---|
| bench_cpu.c | CPU 矩阵和卷积 | `make cpu` → build/bench_cpu |
| bench_cuda.cu | naive/tiled/optimized/cuBLAS/CUDA 卷积，B/C 只需加自己的实现名分支 | `make cuda` → build/bench_cuda |

建议参数：`--operation matrix|convolution --impl NAME --n N`（卷积用 `--m M --k K --filter NAME`）。
先检查返回码，再校验输出，最后记录耗时；返回 LAB6_NOT_IMPLEMENTED 时程序非零退出。
GPU compute 时间取自库返回的 `lab6_timings`，不要在异步 launch 前后量 host 时间。
