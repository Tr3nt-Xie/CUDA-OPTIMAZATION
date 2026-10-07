# 统一开发和测试约定

本文保留组内已确认的接口、数据、正确性、计时和协作规范，用于各实现接入和结果比较。实验任务与算法要求以 `lab6-f26.pdf` 为准；各成员在这些共同约定下决定内部实现。

统一测试参数以 `configs/benchmark.json` 为准。变更时同步受影响的测试入口、文档和全组，避免各自维护不同参数。现有 C/CUDA 测试程序的默认值与配置一致，但不会自动读取 JSON；配置变更后需同步默认值或显式传入命令行参数。读取配置和组织脚本的方式由实现者决定。

## 数据和正确性

- 矩阵：`float32`、C 行主序连续存储、`C=A×B`。正式测试四种 N 全部运行；PDF 指定 512、1024、2048，组内补充 256。小规模 2、3、31 用于正确性和非整块边界检查。
- 当前矩阵测试程序用固定 seed 和 splitmix64 在程序内生成输入，取值范围 [0,1)。参考结果由测试程序里独立的双精度实现计算，CPU 实现也对照它。跨实现比较使用相同输入；不同随机生成器的 seed 相同不代表数据相同。
- 卷积测试图片原图放 `data/images/` 并注明来源；由脚本转换成各尺寸的 `uint32` 原始二进制放 `data/generated/`，C 与 Python 读同一份。
- 卷积图像为 `uint32` 灰度值 0–255，权重和原始输出为 `float32`。保留负数输出；展示图片才另行裁剪/映射，不拿展示图片做数值校验。
- 组内统一 stride=1、zero padding、same 输出、翻转滤波器的数学卷积。K 为正奇数，r=K/2：
  `out[y,x] = Σ filter[i,j] × image[y+r-i, x+r-j]`，越界图像值视为 0。
- 卷积正式测试 M=512/1024/2048 与 K=3/5/7 的九组笛卡尔积，统一使用归一化均值滤波器。另用 edge/sharpen 展示图像效果。现有测试程序 `apps/harness.c` 中三种核对任意奇数 K 的定义如下：
  - mean：每个元素 1/K²
  - edge：每个元素 -1，中心 K²-1（元素和为 0）
  - sharpen：2×单位核 − mean，即每个元素 -1/K²，中心再加 2（元素和为 1）
- 三种图像尺寸、三种滤波器尺寸来自 PDF 要求，具体数值、九组配对方式和边界规则为组内约定。
- 正确性逐元素要求 `abs(got-ref) <= 1e-3 + 1e-4*abs(ref)`，拒绝 NaN/Inf，同时记录最大绝对误差。手算小例子写在测试程序 `selftest` 里，其中卷积用只有左上角非零的核来区分卷积与互相关。
- CPU、CUDA、Python 输出均对照同一参考。cuBLAS 的结果也应符合行主序的 `C=A×B` 定义，不能用全相同输入掩盖转置问题。正式比较保持 FP32 运算，不主动开启 TF32 或 fast-math，保持相同的正确性标准。

## 代码与接口

公共声明在 `include/lab6.h`，保持可由 C 和 `ctypes` 调用的 ABI，不向公共 C 头文件暴露 CUDA 类型。各实现接收 **host 指针**，输出由调用方分配，不能与输入重叠。实现必须验证尺寸和指针，计算字节数时防止整数溢出；CUDA 错误需返回非零并释放已分配资源。

`lab6_matmul_*` / `lab6_convolve_*` 是现有 C/CUDA 实验入口的调用接口，返回时 host 输出已就绪。设备内部的内存管理、kernel 组织和调用流程由实现决定。

`src/common/cuda_utils.cuh` 提供参数检查、分配、传输、计时、同步和释放的辅助封装；当前 `src/matrix/naive.cu` 使用这些封装。它们可按需复用。

最后一个参数 `lab6_timings *` 可为 NULL：
- 非 NULL：填写本次调用的 compute 和 end_to_end 毫秒值。
- NULL：调用方不读取耗时，返回时结果仍须就绪。
- 返回非零：输出及计时无效，不得当作成功记录或加入性能平均值。

按 PDF Part 7 / 8.3，Python 分别调用 Part 4 的 tiled 矩阵实现和 Part 8.2 的 CUDA 卷积实现；当前 `src/bindings/exports.cu` 已连接这些入口。Python 调用前检查 dtype、形状和 C 连续布局，`ctypes` 参数类型与 `include/lab6.h` 一致，并设置 `restype=c_int`；非零返回值作为调用失败处理，不继续把输出当作有效结果。封装和文件组织由 C 决定。

共享库对外名称沿用 `gpu_matrix_multiply` 和 `gpu_convolve`。PDF 示例的 void 返回改为 int 是组内错误处理约定，Python 必须与头文件匹配。

## 计时与结果

每个配置先预热 1 次，正式测量 3 次，保留全部记录并取算术平均值。预热次数、三次重复和均值为组内规定。

| scope | 计时边界 |
|---|---|
| CPU compute | 仅 CPU 运算，不含输入生成/文件读写 |
| GPU compute | CUDA event 测 kernel/cuBLAS 运算，正确同步，不含显存分配和传输 |
| C/CUDA end_to_end | 已有 host 输入到结果返回，包含本次分配、H2D、运算、D2H、释放；若本次创建库句柄也计入 |
| Python end_to_end | 已有合规 NumPy 输入到 ctypes 调用完成，包含跨语言调用和上述 GPU 流程 |

CPU 无传输时可在同一次调用中记录两种 scope，清楚注明边界。Python compute 可通过公开的 `lab6_*(..., timings)` 接口取得设备计时；Python 总耗时用外层单调时钟测量。不要将外层总耗时标为 kernel time。

正式 CPU/GPU 数据在**同一台实验机**采集；共享 GPU 按人轮流，不能同时跑性能实验。每批正式测试保留统一格式的原始 CSV 和一份 `environment.txt`（硬件、驱动/Toolkit、编译命令、git commit），放法见 `results/README.md`。CSV 可按配置拆分；通过文件名或运行清单保留输入标识，汇总时仍能区分不同图片或矩阵输入。云平台与 GPU 型号不写死在代码中。

CSV 按配置中的列顺序输出。一行对应一次运行的一种 timing_scope：
- `operation`：matrix / convolution；`N` 为矩阵边长，`M,K` 为卷积尺寸，`filter` 为 mean / edge / sharpen。不适用的字段留空。
- `implementation`：cpu / naive / tiled / optimized / cublas / cuda / python_tiled / python_cuda。
- `run`：1–3；预热不混入正式 CSV。`correct`：true / false。
- 有有效耗时但正确性失败的行保留 false，由汇总脚本排除并显式报告。
- CPU/GPU 加速比必须基于相同输入、相同 scope 的均值，不混合 compute 与 end_to_end。
- 进一步优化不保证超过 cuBLAS；保留真实改善或退化的结果及解释。

## 分工与接入

负责范围见 `docs/submission-checklist.md`。公共接口或配置发生变化时，同步受影响的调用方和测试说明。尚未交付的其他成员实现保留占位状态。

## 共同开发规范

- 各自在功能分支开发；沿用建议名称 `feat/a-baselines`、`feat/b-cuda`、`feat/c-integration`。
- 公共头文件、配置或 Makefile 有改动时，在 PR 中说明影响并同步给全组。
- 合入 main 前通过 `make cpu`；有 CUDA 环境时同时通过 `make cuda`。尚未进行的 GPU 编译和运行验证明确标为待验证。
- 提交的 C/CUDA 代码按仓库 `.clang-format` 格式化，避免无关的全仓库格式改动。
- 每人使用自己的 Git 身份提交，不改写队友历史。
- 本人实现先通过小规模正确性检查，再按统一参数测量；未完成的其他成员函数保持 `LAB6_NOT_IMPLEMENTED`。

## 交接顺序

1. A 提供公共接口、CPU 参考实现、测试输入和测试工具；B/C 可按各自 Part 并行开发。
2. A/B 的对应实现可运行后，C 联调共享库和 Python；各人核对本人实现的正确性。
3. 在同一实验机轮流测量，交接原始 CSV、environment.txt、运行命令及本人报告和视频材料。
4. C 汇总数据与图表，A 合并报告，B 检查源码和构建复现，C 合并视频材料。视频脚本仍放在代码仓库外的 lab6 实验目录。
