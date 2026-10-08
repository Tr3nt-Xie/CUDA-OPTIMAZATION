# EE542 Lab6 — CUDA Optimization

本仓库包含矩阵乘法、卷积、CUDA 优化和 Python 共享库调用的代码及实验记录。

**当前状态：框架（共用 CUDA 流程、测试程序、脚本）和 A 的 CPU / naive 实现已完成；B、C 的函数仍是返回 `LAB6_NOT_IMPLEMENTED` 的占位。**
A 的 Parts 1–3、8.1 已完成 AWS 同机实验：四种正式矩阵尺寸及补充小尺寸共 48 次计时调用；CPU 卷积 33 个配置、99 次计时调用和 9 份独立图像检查全部通过。GPU 容器内也完成三种矩阵尺寸的运行，见 [A 的报告材料](report/A.md)、[原始数据和图表](results/README.md) 与 [实际环境记录](docs/ENVIRONMENT.md)。A 的 Q1/Q2 和录制脚本已准备，视频脚本位于仓库外；实际录制和全组材料合并尚待完成。
任务依据为课程 `lab6-f26.pdf`（原文件保留在仓库外的实验目录）。

## 目录与分工

| 成员 | 负责内容 |
|---|---|
| A | Part 1–3：CPU 矩阵乘法、naive CUDA、GPU 运行环境；Part 8.1：CPU 卷积、测试图片和滤波器；公共框架（CUDA 公共流程、测试程序）；合并最终报告 |
| B | Part 4、6.2：分块矩阵乘法与进一步优化；Part 8.2：CUDA 卷积；检查构建能否复现 |
| C | Part 5、6.1：cuBLAS、性能汇总与画图；Part 7、8.3：共享库与 Python 调用；合并视频 |

各文件的负责人标在下面的目录树中。

```text
.
├── include/lab6.h           # 公共 C 接口、错误码、计时结果
├── src/
│   ├── common/             # A · 共用代码
│   │   ├── cuda_utils.cuh  #     GPU 公共流程 run_matmul / run_convolution（分配、拷贝、计时、释放）
│   │   ├── validate.h      #     参数检查
│   │   └── timer.h         #     单调时钟
│   ├── matrix/
│   │   ├── cpu.c           # A · Part 1  CPU 三重循环（已实现）
│   │   ├── naive.cu        # A · Part 2  每线程一个输出元素，16×16 block（已通过 T4 自测）
│   │   ├── tiled.cu        # B · Part 4
│   │   ├── optimized.cu    # B · Part 6.2
│   │   └── cublas.cu       # C · Part 6.1
│   ├── convolution/
│   │   ├── cpu.c           # A · Part 8.1  CPU 数学卷积，零填充（已实现）
│   │   └── cuda.cu         # B · Part 8.2
│   └── bindings/exports.cu # C · Part 7 / 8.3，Python 导出入口
├── apps/                   # A · 统一测试程序
│   ├── harness.h / .c      #     selftest / matrix / conv、双精度参考、滤波器、预热计时、CSV
│   ├── bench_cpu.c         #     登记 cpu
│   └── bench_cuda.cu       #     登记 naive / tiled / optimized / cublas / cuda
├── python/                 # C · ctypes 封装和 Python 实验入口
├── scripts/
│   ├── prepare_images.py   # A · 图片 ↔ .u32 原始图像 / 滤波输出 → PNG
│   ├── generate_test_images.py # A · 原创灰度测试图
│   ├── run_part81_cpu.py    # A · 8.1 CPU 实验与效果图
│   ├── run_part_a_matrix.py # A · CPU/naive 同机正式及补充矩阵实验
│   ├── plot_part_a.py      # A · 基线曲线和加速比
│   ├── collect_env.sh      # A · 生成 environment.txt
│   └── requirements.txt    # A · NumPy、Pillow、Matplotlib
├── configs/benchmark.json  # 全组统一测试参数
├── data/                   # A
│   ├── images/             #     卷积展示用原图（提交，注明来源）
│   └── generated/          #     prepare_images.py 输出的 .u32（Git 忽略）
├── results/
│   ├── raw/                # 原始 CSV、日志和运行环境
│   ├── summary/            # 从原始数据生成的汇总表
│   └── figures/            # 报告、视频使用的图
├── docs/
│   ├── CONTRACTS.md        # 统一接口、数据、计时和协作规范
│   ├── ENVIRONMENT.md      # A · 实际 GPU 环境与运行证据
│   └── submission-checklist.md # Part、负责人和交付映射
├── report/                 # 各人报告材料及最终报告
├── .clang-format           # 统一 C/CUDA 代码格式
└── Makefile                # CPU 静态库 / CUDA 共享库 / 实验程序构建 / make test
```

A/B/C 是分工代号，不代表已完成的贡献。具体交接见 [提交清单](docs/submission-checklist.md)。

## 开始开发

实验任务、算法要求和交付内容以 `lab6-f26.pdf` 为准，按 [分工表](docs/submission-checklist.md) 完成本人负责的 Part。上面的目录树描述当前代码位置；内部实现、是否复用辅助封装、脚本拆分和文件命名由各成员依据实验文档决定。

组内统一的接口、数据、测试和提交规范继续适用，详见 [统一开发和测试约定](docs/CONTRACTS.md)，公共声明见 [lab6.h](include/lab6.h)。`src/common/cuda_utils.cuh` 是 A 的 naive 实现使用的辅助代码，可按需复用。

现有测试入口支持 `bench_cuda selftest --impl <名字>`，可只检查本人实现。其他成员尚未完成的函数保留占位状态；出现 `LAB6_NOT_IMPLEMENTED` 不代表需要代做其任务。

合入 main 前通过 `make cpu`，有 CUDA 环境时同时通过 `make cuda`；提交的 C/CUDA 代码使用仓库 `.clang-format`。每人使用自己的 Git 身份，公共接口、配置或构建方式调整时说明影响并同步给全组。

矩阵测试尺寸 **256、512、1024、2048 全部运行，不是任选一个**。其中后三项来自 PDF，256 是组内补充。卷积采用三个图像尺寸 × 三个滤波器尺寸，共九组；配置详见 JSON。

## 构建

在仓库根目录执行，需 Make 和 C 编译器；GPU 构建另需 NVIDIA CUDA Toolkit（含 nvcc、cuBLAS）。

```sh
make help
make cpu       # build/liblab6_cpu.a + build/bench_cpu，Mac 可做本地构建检查
make cuda      # build/liblab6_cuda.so + build/bench_cuda，在 Linux CUDA 开发环境执行
make all       # 上述全部
make test      # build/bench_cpu selftest
make clean     # 删除 build/
```

CUDA 源文件逐个编译后再链接成共享库：某个 `.cu` 出错时报错会指明文件，其余文件增量重编。

可使用 `make cpu CC=gcc` 或 `make cuda NVCC=/path/to/nvcc` 指定编译器。
CPU 默认 `-O2`。CUDA 架构由目标主机确定，必要时设置 `NVCCFLAGS`，例如加入该主机支持的 `-arch`；不在仓库锁定某一型号。

## 运行

以下是 A 的 naive 矩阵测试示例，在 GPU 环境就绪且对应实现通过正确性检查后执行。默认测试全部四种 N；B/C 可用 `--impl` 选择本人已完成的实现。每次运行使用新的结果目录。

```sh
build/bench_cuda selftest --impl naive
lab6_run_dir="results/raw/A-$(date +%Y%m%d-%H%M%S)-naive"
mkdir "$lab6_run_dir"
scripts/collect_env.sh > "$lab6_run_dir/environment.txt"
build/bench_cuda matrix --impl naive --member A --csv "$lab6_run_dir/measurements.csv"
```

完整参数见 [apps/README.md](apps/README.md)，图片转换见 [scripts/README.md](scripts/README.md)。[Python 接口现状](python/README.md) 和全组性能对比结果尚待 C 完成。

Mac 上可完成 CPU 正确性验证、图片效果展示及本地计时；正式 CPU/GPU 性能对比使用统一实验机的数据。A 的 AWS CPU/naive 基线已采集，B/C 的后续实现可按同一口径对照。共用环境的连接信息由 A 单独提供；固定输入和 CPU 参考输出位于实验机只读目录 `/opt/lab6-a-reference/20261006-232110/`，详见结果入口。

## 提交内容

保留源码、构建/运行说明、配置、小型验证输入、原始结果与环境记录、汇总图表和报告材料。
不提交编译产物、虚拟环境、密钥及可再生的大型输入。最终可从源码构建共享库，无需将本机生成的二进制加入 Git。
