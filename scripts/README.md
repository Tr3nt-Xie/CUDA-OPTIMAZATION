# 脚本（待实现）

| 预留脚本 | 负责人 | 作用 |
|---|---|---|
| prepare_images.py | A | data/images 原图 → data/generated 下各尺寸的 uint32 原始二进制 |
| run_benchmarks.py | C | 按 configs/benchmark.json 调用测试程序 → results/raw/<member>-<时间>/ |
| summarize.py | C | raw CSV（只取 correct=true）→ results/summary |
| plot_results.py | C | summary → results/figures |

参数只从 configs/benchmark.json 读取。脚本完成后在这里补依赖和运行命令。
