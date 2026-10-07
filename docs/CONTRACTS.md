# 全组开发约定

以下分为 PDF 要求和组内实现约定。具体参数只在 `configs/benchmark.json` 维护；需要变更时同步本文件和全组。

## 数据和正确性

- 矩阵：`float32`、C 行主序连续存储、`C=A×B`。正式测试四种 N 全部运行；PDF 指定 512、1024、2048，组内补充 256。小规模 2、3、31 用于正确性和非整块边界检查。
- A 生成全组共用输入文件和 manifest，记录形状、类型、种子与 SHA-256。C/Python 读取同一份数据，不能只约定同一个 seed 后各自生成。随机值范围及生成器版本写入 manifest。
- 卷积图像为 `uint32` 灰度值 0–255，权重和原始输出为 `float32`。保留负数输出；展示图片才另行裁剪/映射，不拿展示图片做数值校验。
- 组内统一 stride=1、zero padding、same 输出、翻转滤波器的数学卷积。K 为正奇数，r=K/2：
  `out[y,x] = Σ filter[i,j] × image[y+r-i, x+r-j]`，越界图像值视为 0。
- 卷积正式测试 M=512/1024/2048 与 K=3/5/7 的九组笛卡尔积，统一使用归一化均值滤波器。另用 edge/sharpen 展示图像效果；具体核由 A 保存共享。三种图像尺寸、三种滤波器尺寸来自 PDF 要求，具体数值、九组配对方式和边界规则为组内约定。
- 正确性逐元素要求 `abs(got-ref) <= 1e-3 + 1e-4*abs(ref)`，拒绝 NaN/Inf，同时记录最大绝对误差。CPU 基准也需用手算小例子或独立实现核对；卷积增加非对称滤波器的小例子以识别卷积与互相关混淆。
- CPU、CUDA、Python 输出均对照同一参考。cuBLAS 要处理行/列主序差异；禁止靠“输入全相同”掩盖转置问题。正式比较保持 FP32 运算，不主动开启 TF32 或 fast-math。

## 代码与接口

公共声明在 `include/lab6.h`。各实现接收 **host 指针**，输出由调用方分配，不能与输入重叠。实现必须验证尺寸和指针，计算字节数时防止整数溢出；CUDA 错误需返回非零并释放已分配资源。

`lab6_matmul_*` / `lab6_convolve_*` 是 C/CUDA 实验入口的调用接口。GPU 实现负责分配设备内存、传输、计算、同步、回传及释放；可在各自 .cu 文件中放私有 kernel/helper，避免暴露 CUDA 类型到公共 C 头文件。多个实现共用的 host 端逻辑（CUDA 错误检查与清理、显存分配/释放、compute/end_to_end 计时）统一放 `src/common/cuda_utils.cuh`，由 A 维护，保证各实现计时边界一致。

最后一个参数 `lab6_timings *` 可为 NULL：
- 非 NULL：填写本次调用的 compute 和 end_to_end 毫秒值。
- NULL：只计算，不创建额外计时对象；返回时结果仍须就绪。
- 返回非零：输出及计时无效，不得当作成功记录或加入性能平均值。

`src/bindings/exports.cu` 复用同一 tiled 矩阵和 CUDA 卷积实现。C 不复制 B 的 kernel；B 修改内部实现不改变公开接口。Python 通过 `ctypes` 设置参数类型和 `restype=c_int`，检查 dtype、形状和 C contiguous，非零返回转成异常。

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

正式 CPU/GPU 数据在**同一台实验机**采集；共享 GPU 按人轮流，不能同时跑性能实验。保存 CPU/GPU、驱动/Toolkit、编译命令、git commit、工作区是否有改动和原始日志。云平台与 GPU 型号不写死在代码中。

CSV 按配置中的列顺序输出。一行对应一次运行的一种 timing_scope：
- `operation`：matrix / convolution；`N` 为矩阵边长，`M,K` 为卷积尺寸。不适用的字段留空。
- `implementation`：cpu / naive / tiled / optimized / cublas / cuda / python_tiled / python_cuda。
- `run`：1–3；预热不混入正式 CSV。`correct`：true / false；`commit` 填真实 SHA。
- `input_id` 关联 data manifest，`run_id` 关联环境、配置快照与日志。
- 失败保留在日志；有有效耗时但正确性失败的行保留 false，由汇总脚本排除并显式报告。
- CPU/GPU 加速比必须基于相同输入、相同 scope 的均值，不混合 compute 与 end_to_end。
- 进一步优化不保证超过 cuBLAS；保留真实改善或退化的结果及解释。

## Git 协作与交接顺序

建议分支 `feat/a-baselines`、`feat/b-cuda`、`feat/c-integration`。这些是命名约定，当前未创建远端分支。

1. A 先提供小输入、CPU 可信输出和环境记录；B 同时实现 tiled/卷积；C 同时做 cuBLAS、ctypes 和结果工具。
2. A/B 提供可运行基线后，C 联调共享库与 Python；各人核对自己的实现。
3. 小规模正确性通过后，预约同一 GPU 轮流测量；每人提供 CSV、日志、运行命令和本人报告/视频材料。
4. C 汇总数据和图表；A 合并报告；B 检查源码与构建复现；C 合并视频材料。

A 主要维护数据生成和 CPU/naive 入口；B 维护 tiled/optimized/CUDA 卷积；C 维护绑定、配置消费和分析脚本。公共头文件、配置、Makefile 改动在 PR 中说明影响，避免多人各自另造接口。合入 main 前须能编译通过，未完成的函数保持返回 `LAB6_NOT_IMPLEMENTED`；代码按根目录 `.clang-format` 格式化。每人使用自己的 Git 身份提交，不改写队友历史。
