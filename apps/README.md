# C/CUDA 实验入口（占位，待实现）

算法放 src，本目录只负责参数解析、读取共用输入、调用算法、正确性检查及输出实验记录。

| 文件 | 负责人 | 职责 | 构建 |
|---|---|---|---|
| bench_cpu.c | A | 矩阵和卷积 CPU 基线，链接 build/liblab6_cpu.a | `make cpu` → build/bench_cpu |
| bench_cuda.cu | A 建入口，B/C 增加分支 | naive/tiled/optimized/cuBLAS/卷积，链接 build/liblab6_cuda.so | `make cuda` → build/bench_cuda |

建议参数统一为 `--operation matrix|convolution --implementation NAME --input PATH`。
输入 manifest 提供维度/类型，实验配置由 scripts 读取后展开；C 程序不必重复实现 JSON 配置解析。

两个文件目前只打印未实现并返回 LAB6_NOT_IMPLEMENTED；Makefile 目标已就绪，实现后把运行命令加入根 README。
需要先检查返回码，再校验输出，最后记录耗时；LAB6_NOT_IMPLEMENTED 必须让程序失败退出。
GPU compute 计时来自库返回的 lab6_timings；不能在异步 launch 前后只量 host 时间。
