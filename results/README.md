# 实验结果

```text
raw/<member>-<YYYYMMDD-HHMM>/
  *.csv             # 可用 measurements.csv 或按配置拆分；统一列见 configs/benchmark.json
  environment.txt   # scripts/collect_env.sh 生成
summary/            # 汇总脚本生成的均值和加速比
figures/            # 由真实数据生成的图
```

每次正式测试新建一个目录，不覆盖已有结果；失败或缺测在报告里说明，不挑最快的一次。CSV 的拆分和文件命名可按实际运行安排，但需保留输入标识及其与结果的对应关系，汇总时不混合不同输入。

## A 的正式 AWS 结果（当前报告使用）

2026-10-06 23:21–23:28 PDT，在同一台 us-west-2 g4dn.2xlarge / Xeon 8259CL / T4 上顺序完成。基础代码 commit 为 `a2f7c1c3bb871ff5bcdf38097ac740e9e8927f3b`；正式测量使用未提交的采集脚本，实际内容和工作区状态均保存在批次的源码快照与 manifest，算法源码没有改动。

| 批次 | 范围 | 验证结果 |
|---|---|---|
| [matrix-aws](raw/A-20261006-232110-matrix-aws/manifest.json) | CPU/naive，正式 N=256/512/1024/2048；Q2 补充 N=16/32/64/128 | 1 次预热 + 3 次计时；48 次计时调用全部正确，96 行原始 CSV |
| [part81-cpu-aws](raw/A-20261006-232110-part81-cpu-aws/manifest.json) | 3 张图，M=512/1024/2048 × K=3/5/7 均值滤波；另有边缘/锐化 | 33 个配置，99 次计时调用全部正确，198 行原始 CSV，9 份 NumPy 独立校验 |
| [container-part3](raw/A-20261006-232110-container-part3/manifest.json) | 官方 CUDA 容器内重新编译、自检，N=256/512/1024 | 9 次 naive 计时调用全部正确，18 行 CSV；独立保存，不混入宿主机对比 |

每次调用同时输出 compute/end_to_end 两行，因此 CSV 行数是计时调用数的两倍。输入生成、参考计算、正确性验证和文件读写不计时。均值保留全部三次样本；标准差为样本标准差。

- [矩阵计时汇总](summary/A-20261006-232110-matrix-aws.csv)、[加速比与开销汇总](summary/A-20261006-232110-matrix-aws-comparison.csv)。CPU 是单线程三重循环基线，加速比为两个均值之比。
- [正式矩阵曲线](figures/A-20261006-232110-matrix-aws/matrix-runtime.png)、[Q2 补充曲线](figures/A-20261006-232110-matrix-aws/matrix-crossover.png)、[GPU 开销图](figures/A-20261006-232110-matrix-aws/gpu-overhead.png)；同目录保留 SVG 和绘图来源哈希。
- [CPU 卷积汇总](summary/A-20261006-232110-part81-cpu-aws.csv)、[滤波效果图](figures/A-20261006-232110-part81-cpu-aws/filter-comparison.png)。

本次数据：N=256/512/1024/2048 的整体加速比分别约 37.80/98.44/461.82/1388.36×。小尺寸中，N=32 的整体 GPU 耗时仍较长，N=64 仅约快 1.25×，到 N=128 约快 8.73×。这是本次硬件/实现/预热计时口径的观察，三次测量未用于显著性检验。`GPU end_to_end - compute` 包含分配、拷贝、同步及释放，不能视作单独测得的传输时间。

批次的 `commands.jsonl` 保存每条命令、起止时间与退出码，`source-snapshot.tar.gz` 保存当时源码。最初 Mac 传输附带的两个 `._` 元数据文件也被原样记录在快照中，它们不参与编译，之后从远端工作区清理；历史记录不改写。后续增加的绘图脚本/依赖不属于当时的计时代码，以图表目录中的 `provenance.json` 单独追溯。

在空闲实验机重新采集时，使用新的 run ID，不覆盖已有数据：

```sh
source ~/.venv/bin/activate
export NVCCFLAGS='-O2 -std=c++14 -arch=sm_75'
python scripts/run_part_a_matrix.py --run-id A-NEW-matrix --context 'Shared AWS T4 host'
python scripts/run_part81_cpu.py --run-id A-NEW-conv --context 'CPU baseline on shared AWS T4 host'
python scripts/plot_part_a.py A-NEW-matrix
```

### B/C 可使用的固定输入和 CPU 输出

实验机目录 `/opt/lab6-a-reference/20261006-232110/` 为只读参考副本：

- `inputs/`：三张图的三种尺寸，共 9 个 `.u32`，与 CPU 批次输入哈希完全相同。
- `outputs/`：M=512、K=3 下均值/边缘/锐化的 9 个未裁剪 `.f32`。
- `evidence/`：三批原始记录与 CPU/naive 汇总；卷积 manifest 列出每个输入/输出的 SHA-256。
- `report/` 与 `results/figures/`：A 的报告及所用图表，方便队友在 Git 更新前查阅。

B/C 普通账号分别读取并核对了全部 18 个文件的哈希。可直接读取或复制到自己的工作目录；无需进入 A 的 home，也未改动队友仓库。其余尺寸可通过同一 CPU 程序重新生成参考输出。目录提供输入与基线，不新增 B/C 的算法、脚本组织或实现要求。

## A 的 AWS 环境验证

[`A-20261007-055952-aws-smoke`](raw/A-20261007-055952-aws-smoke/) 保存 us-west-2 T4 上的初次编译、自测、GPU 容器运行日志和实际环境。目录时间为 UTC，对应 2026-10-06 晚上 PDT。CPU 和 naive 各做了 N=256、1 次预热、1 次计时，均正确；这些 CSV 只用于部署验证，没有参与上面的正式数据汇总。

## Part 8.1 卷积 CSV 的组织方式

`part81-cpu-aws` 中每个原始 CSV 的文件名包含图片、M、K 和滤波器，内部仍保留全组统一列。一个配置有 3 次运行 × 2 个 timing_scope，共 6 行。图像读取、参考计算、验证、PNG 生成不计入 CPU compute；当前 CPU 实现的 end_to_end 与 compute 相等。独立核验针对未裁剪的 float32 输出。

早期在 Mac 上做过的本地卷积验证批次已从仓库删除（可在 commit `a2f7c1c` 中查到）：它们与 AWS 不是同一台机器，不能与 GPU 数据计算加速比，现由上面的 AWS 同机批次取代。
