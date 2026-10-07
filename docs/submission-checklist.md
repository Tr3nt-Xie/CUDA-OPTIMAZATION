# Lab6 分工与交付检查表

依据：`lab6-f26.pdf`，共 8 页。下表为任务分配和交付位置，所有实验目前均**待实现/待运行**。A/B/C 姓名待填。

| Part / PDF 页码 | 负责人 | 工作与源码 | 验收 / 材料 |
|---|---|---|---|
| 1 · p1–2 | A | src/matrix/cpu.c，CPU 三重循环 | 正确性、三种必测尺寸；report 中的时间记录与曲线 |
| 2 · p2–3 | A | src/matrix/naive.cu，朴素 CUDA 与 host 流程 | 可运行程序、CPU 对照与原始计时 |
| 3 · p3 | A | docs/ENVIRONMENT.md，共用 GPU 环境 | 环境版本与实际成功运行证据 |
| 4 · p3–4 | B | src/matrix/tiled.cu，共享内存分块 | 对照 CPU/naive，正确性、计时、边界处理 |
| 5 · p4 | C 汇总，A/B 供数据 | scripts，results/summary、figures | CPU/naive/tiled 对比、加速比与传输开销 |
| 6.1 · p4–5 | C | src/matrix/cublas.cu | 同尺寸 cuBLAS 对照、正确性与计时 |
| 6.2 · p5 | B | src/matrix/optimized.cu | 进一步优化尝试、方法来源、前后数据与视频解释 |
| 7 · p5–7 | C，依赖 B | src/bindings/exports.cu，python | Part 4 共享库、ctypes 调用和正确输出 |
| 8.1 · p7 | A | src/convolution/cpu.c，数据/滤波器生成 | ≥3 图像尺寸、≥3 滤波器尺寸；report 中展示图像和滤波结果 |
| 8.2 · p8 | B，依赖 A 的输入 | src/convolution/cuda.cu | 同输入 CPU/CUDA 数值与性能对照 |
| 8.3 · p8 | C，依赖 B | 卷积共享库与 Python 入口 | CPU C / 独立 CUDA / Python CUDA 三方比较 |
| Deliverables · p8 | 全员 | GitHub 源码及可复现说明 | CPU、CUDA、优化、cuBLAS、共享库、Python 全部可定位；视频含曲线、规模变化、开销、优化取舍、共享库说明 |

## Part 6 五个问题

| 题号 | 内容 | 主答 / 数据来源 |
|---|---|---|
| Q1 | 性能如何随矩阵规模变化 | A；使用 C 汇总图表 |
| Q2 | 从什么规模起 GPU 明显更快 | A；分清 compute / end_to_end |
| Q3 | tiled 相比 naive 的效果 | B |
| Q4 | 手工优化与 cuBLAS 比较 | C；B 提供优化说明 |
| Q5 | cuBLAS 为什么更快 | C；结合实测与资料解释 |

以上问题保留完整答案，纳入报告工作稿并用于视频讲解；这是组内整理方式，不额外声称 PDF 指定了所有问题的唯一提交载体。

## 每人的交接包

- 自己的源码、依赖及真实编译/运行命令。
- 正确性证据、原始 CSV/日志、环境与输入标识。
- 本人所负责 Part 的报告材料、图像说明和视频片段/口播。
- 当前限制、失败项或未完成项；不得以占位数据充当结果。

建议在 report 下按 A/B/C 建立交接稿，之后合并成唯一 `report/report.md` 和 `report/video-script.md`。当前不生成空的最终报告，也不填写虚构贡献、结果或视频链接。

## 最后收尾

- [ ] A：合并报告，确保 Part 1 曲线、Part 8.1 图像及相关实验材料齐全。
- [ ] B：干净检出后核对 CPU/GPU 构建、所有源文件、README 和复现流程。
- [ ] C：生成对比图，合并视频材料，核对共享库/Python 演示。
- [ ] 全员：核对全部 Part、五个问题、真实贡献和最终链接；按课程提交页面核对提交入口及附件。

本地骨架创建不等于 GitHub 已 push、课程已提交或实验已完成。
