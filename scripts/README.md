# 脚本

下表只列现有工具。C 的性能对比与分析按实验文档完成，脚本拆分及文件命名由 C 决定。

| 脚本 | 负责人 | 作用 | 状态 |
|---|---|---|---|
| prepare_images.py | A | 图片 ↔ 测试程序用的原始二进制 | 已完成 |
| generate_test_images.py | A | 生成三张原创、可复现的灰度测试图 | 已完成 |
| run_part81_cpu.py | A | 8.1 CPU 图片实验、独立正确性核对、CSV 汇总和效果图 | Mac 验证及 AWS 同机正式测量通过 |
| run_part_a_matrix.py | A | 同机 CPU/naive 正式矩阵基线与 Q2 补充小尺寸测量 | AWS T4 测量通过 |
| plot_part_a.py | A | 从已通过的矩阵批次生成曲线、加速比与来源记录 | 正式数据绘图完成 |
| collect_env.sh | A | 生成 environment.txt | 已完成 |

依赖：Python 3.9+，`python3 -m pip install -r scripts/requirements.txt`（NumPy、Pillow；Matplotlib 用于矩阵绘图）。

## A 的同机矩阵实验

在空闲的 CUDA 实验机上运行；脚本只采集 A 的 CPU/naive，不会启动云资源或调用 B/C 的占位实现。

```sh
export NVCCFLAGS='-O2 -std=c++14 -arch=sm_75'  # 本次 T4 使用的选项
python3 scripts/run_part_a_matrix.py --run-id A-YOUR-UNIQUE-RUN-ID --context 'Describe the actual host'
python3 scripts/plot_part_a.py A-YOUR-UNIQUE-RUN-ID
```

正式尺寸取自配置文件（256/512/1024/2048）。脚本另测 16/32/64/128，用于 A 的 Q2 起效规模分析；这些是补充实验，不改变全组必测尺寸。可用 `--small-sizes ''` 关闭补充测量。CPU/naive 串行执行，各预热 1 次、测量 3 次，全部样本均保留。

原始 CSV、完整命令/退出码、双精度参考检查、自测、环境、源码快照及文件/二进制哈希在 `results/raw/<run-id>/`。`manifest.json` 只有所有尺寸正确且源文件未变时才标记 `passed`。`results/summary/<run-id>.csv` 为均值、样本标准差与范围；`-comparison.csv` 中的加速比为两个均值之比。

绘图脚本输出运行时间、补充小尺寸曲线和 GPU 开销图（PNG/SVG）。`provenance.json` 记录输入汇总、绘图脚本和图片的哈希。GPU 的 `end_to_end - compute` 是分配、拷贝、同步和释放等合计的剩余耗时，不能直接称为纯 PCIe 传输时间。

## Part 8.1 CPU 图片实验

在仓库根目录执行：

```sh
python3 scripts/run_part81_cpu.py
```

脚本使用并校验 `data/images/manifest.json` 记录的原图；没有原图清单时才生成测试图。随后转换图片、独立构建 CPU 程序、运行 selftest，并串行执行每张图的 M=512/1024/2048 × K=3/5/7 均值滤波。另在 M=512、K=3 展示边缘检测和锐化，共 33 个配置，每组预热 1 次、测量 3 次。

每个配置用 C 双精度参考核对；九份展示输出另用 NumPy float64、补零的滑动窗口和翻转后的滤波器逐像素核对。CPU 计算的原始 float32 输出与 PNG 的显示映射分开保存。

- `results/raw/<run-id>/`：每个配置的 CSV/日志、完整命令和退出码、环境、输入哈希、源码快照与总清单。
- `results/summary/<run-id>.csv`：逐图、逐配置的均值、样本标准差、最小/最大值，两种计时范围分别记录。
- `results/figures/<run-id>/`：原图、三种滤波效果图和 `filter-comparison.png`。
- `data/generated/<run-id>/`：九份展示用 CPU float32 参考输出，Git 忽略。

`--run-id NAME` 可指定实验编号；已有编号会被拒绝，避免覆盖记录。`--context '实际实验机与测量目的'` 用于明确本地验证或共用 AWS 主机上的 CPU 基线。Mac 测量不能与另一台主机的 CUDA 耗时直接计算加速比。

## 单独转换图片

```sh
# data/images 下所有图片 → data/generated/<名字>_<M>.u32（灰度、居中裁成正方形、缩放到配置里的各尺寸）
python3 scripts/prepare_images.py to-raw

# 跑一个展示滤波并转回 PNG；edge 输出有正有负，用 --mode abs 显示
build/bench_cpu conv --image data/generated/geometry_512.u32 --k 3 --filter edge --save edge.f32
python3 scripts/prepare_images.py to-png edge.f32 edge.png --mode abs
```

环境记录命令见 [仓库运行示例](../README.md#运行)；先创建本次结果目录，再保存 `collect_env.sh` 的输出。

`to-png` 的 `--mode`：`clip`（默认，裁到 0–255）、`abs`（取绝对值再裁剪）、`normalize`（线性拉伸到 0–255）。只影响展示，数值校验永远用原始 float32 输出。
