# 实验结果

```text
raw/<member>-<YYYYMMDD-HHMM>/
  measurements.csv  # 列见 configs/benchmark.json
  environment.txt   # 硬件、驱动/Toolkit、编译命令、git commit
summary/            # 汇总脚本生成的均值和加速比
figures/            # 由真实数据生成的图
```

每次正式测试新建一个目录，不覆盖已有结果；失败或缺测在报告里说明，不挑最快的一次。
