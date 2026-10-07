# 数据与实验工具（待实现）

| 预留脚本 | 负责人 | 输入 → 输出 |
|---|---|---|
| generate_inputs.py | A | configs/benchmark.json → data/generated 下的共用输入、SHA-256 manifest、小型参考输出 |
| run_benchmarks.py | C，A/B 配合入口 | 配置 + 已构建程序 → results/raw/<run_id>/ 下的原始记录 |
| summarize.py | C | 正确性通过的 raw CSV → results/summary |
| plot_results.py | C | summary → results/figures |

配置参数只从 configs/benchmark.json 读取。首次生成先定 manifest 格式，让 C 与 Python 读同一份数据。
建议大数组用带 manifest 的小端原始二进制，小型手算样例单独保存；文件顺序为 row-major。
不要在本阶段加入虚构数据或以随机耗时代替运行。脚本完成后补依赖和真实命令。
