# 脚本

下表只列现有工具。C 的性能对比与分析按实验文档完成，脚本拆分及文件命名由 C 决定。

| 脚本 | 负责人 | 作用 | 状态 |
|---|---|---|---|
| prepare_images.py | A | 图片 ↔ 测试程序用的原始二进制 | 已完成 |
| generate_test_images.py | A | 生成三张原创、可复现的灰度测试图 | 已完成 |
| run_part81_cpu.py | A | 8.1 CPU 图片实验、独立正确性核对、CSV 汇总和效果图 | 已在 Mac 验证 |
| collect_env.sh | A | 生成 environment.txt | 已完成 |

依赖：Python 3.9+，`python3 -m pip install -r scripts/requirements.txt`（NumPy、Pillow）。

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

`--run-id NAME` 可指定实验编号；已有编号会被拒绝，避免覆盖记录。本脚本是 A 的 8.1 本地验证入口。本次 Mac 测量不能与另一台主机的 CUDA 耗时直接计算加速比。

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
