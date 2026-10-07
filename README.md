# EE542 Lab6 — CUDA Optimization

本仓库包含矩阵乘法、卷积、CUDA 优化和 Python 共享库调用的代码及实验记录。

**当前状态：项目骨架。算法函数均为待实现占位，返回 `LAB6_NOT_IMPLEMENTED`；编译成功不代表实验完成。**
任务依据为课程 `lab6-f26.pdf`（原文件保留在仓库外的实验目录）。

## 目录与分工

```text
.
├── include/lab6.h           # 公共 C 接口、错误码、计时结果
├── src/
│   ├── common/cuda_utils.cuh # A · 所有 GPU 实现共用的错误检查、显存和计时辅助
│   ├── matrix/
│   │   ├── cpu.c           # A · Part 1
│   │   ├── naive.cu        # A · Part 2
│   │   ├── tiled.cu        # B · Part 4
│   │   ├── optimized.cu    # B · Part 6.2
│   │   └── cublas.cu       # C · Part 6.1
│   ├── convolution/
│   │   ├── cpu.c           # A · Part 8.1
│   │   └── cuda.cu         # B · Part 8.2
│   └── bindings/exports.cu # C · Part 7 / 8.3，Python 导出入口
├── apps/                   # A/B/C · bench_cpu.c / bench_cuda.cu 命令行实验入口
├── python/                 # C · ctypes 封装和 Python 实验入口
├── scripts/                # A · 数据生成；C · 实验调度、汇总和绘图
├── configs/benchmark.json  # 全组统一测试参数
├── data/                   # 共用输入和数据清单
├── results/
│   ├── raw/                # 原始 CSV、日志和运行环境
│   ├── summary/            # 从原始数据生成的汇总表
│   └── figures/            # 报告、视频使用的图
├── docs/
│   ├── CONTRACTS.md        # 数据、接口、计时和 Git 协作约定
│   ├── ENVIRONMENT.md      # A · GPU 环境记录模板
│   └── submission-checklist.md # Part、负责人和交付映射
├── report/                 # 各人交接材料及最终报告、视频稿
├── .clang-format           # 统一 C/CUDA 代码格式
└── Makefile                # CPU 静态库 / CUDA 共享库 / 实验程序构建
```

A/B/C 是分工代号，不代表已完成的贡献。具体交接见 [提交清单](docs/submission-checklist.md)。

## 开始开发

1. 先读 [统一约定](docs/CONTRACTS.md)，公共接口以 [lab6.h](include/lab6.h) 为准。
2. 各自创建功能分支，主要编辑本人负责的文件；接口和配置变更先同步给全组。
3. 实现对应占位函数，并补充命令行入口、正确性检查、运行说明及本人材料。
4. 小输入通过后，在同一台 NVIDIA GPU 主机上轮流跑正式测试，提交原始记录。

合入 main 的代码必须能通过 `make cpu`（有 GPU 环境时还要通过 `make cuda`）；未完成的函数保持返回 `LAB6_NOT_IMPLEMENTED`，不要提交编译不过的半成品。
GPU 实现的公共 host 端逻辑放 `src/common/cuda_utils.cuh`，不在各自 `.cu` 里重复写一份。
提交前用仓库根目录的 `.clang-format` 格式化（编辑器开启 format on save 即可），避免纯格式 diff。

矩阵测试尺寸 **256、512、1024、2048 全部运行，不是任选一个**。其中后三项来自 PDF，256 是组内补充。卷积采用三个图像尺寸 × 三个滤波器尺寸，共九组；配置详见 JSON。

## 构建

在仓库根目录执行，需 Make 和 C 编译器；GPU 构建另需 NVIDIA CUDA Toolkit（含 nvcc、cuBLAS）。

```sh
make help
make cpu       # build/liblab6_cpu.a + build/bench_cpu，Mac 可做本地构建检查
make cuda      # build/liblab6_cuda.so + build/bench_cuda，在 Linux CUDA 开发环境执行
make all       # 上述全部
make clean     # 删除 build/
```

CUDA 源文件逐个编译后再链接成共享库：某个 `.cu` 出错时报错会指明文件，其余文件增量重编。

可使用 `make cpu CC=gcc` 或 `make cuda NVCC=/path/to/nvcc` 指定编译器。
CPU 默认 `-O2`。CUDA 架构由目标主机确定，必要时设置 `NVCCFLAGS`，例如加入该主机支持的 `-arch`；不在仓库锁定某一型号。

[C/CUDA 实验程序](apps/README.md) 已有可编译的占位入口，运行时报未实现并非零退出；[Python 程序](python/README.md) 和 [数据脚本](scripts/README.md) 的职责及预留文件名已经确定，尚未实现。最终运行命令应在各入口完成后补入 README。

Mac 上的 CPU 构建仅用于开发；正式性能对比必须在统一实验机上重新运行。需要 GPU 时再安排共享机器或云资源。

## 提交内容

保留源码、构建/运行说明、配置、小型验证输入、原始结果与环境记录、汇总图表和报告材料。
不提交编译产物、虚拟环境、密钥及可再生的大型输入。最终可从源码构建共享库，无需将本机生成的二进制加入 Git。
