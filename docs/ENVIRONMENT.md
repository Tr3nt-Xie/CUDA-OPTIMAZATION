# 实验环境记录（A 负责）

状态：AWS 部署暂缓，后续由 A 安排共用 GPU 环境，再填写实际验证结果。此文件是环境记录模板；B/C 可继续本人实现，当前不要求其开通云资源。

| 字段 | 实际值 |
|---|---|
| 实验机代号 / 平台 | 待填 |
| OS / CPU / 内存 | 待填 |
| GPU 型号 / 显存 | 待填 |
| NVIDIA driver / CUDA Toolkit | 待填 |
| C 编译器 / nvcc / Python / NumPy | 待填 |
| 实际构建命令与优化选项 | 待填 |
| 代码 commit / 未提交变更 | 待填 |
| 环境验证命令、输出位置与日期 | 待填 |

每次正式实验先新建结果目录，再将 `scripts/collect_env.sh` 的输出保存为该目录中的 `environment.txt`；命令示例见仓库 README。
不要记录账户密码、访问密钥或私钥。

## 后续主机准备

- 平台、GPU、镜像和软件版本在准备实际环境时确认，并记录在上表；不把此前的平台或安装建议作为 B/C 的开发前提。
- 构建选项与目标主机匹配，记录实际编译器和优化参数；正式比较遵守 `docs/CONTRACTS.md` 的统一数据和计时要求。
- 验证顺序：`nvidia-smi` → `make all` → `build/bench_cpu selftest` → `build/bench_cuda selftest --impl naive` → `scripts/collect_env.sh`。

PDF Part 3 的 GPU 示例包括 T4/V100，平台示例为 Google Cloud；实际采用的环境写清楚。此前提到的课堂口头显卡要求需以老师确认内容为准，不把仓库构建成功当作课程环境验收。
