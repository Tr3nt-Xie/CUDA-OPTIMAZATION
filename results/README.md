# 实验记录

正式实验后采用：
```text
raw/<run_id>/
  measurements.csv  # 字段见 configs/benchmark.json
  environment.txt   # 实际硬件、软件与构建信息
  config.json       # 本次配置快照
  inputs.json       # 本次输入 manifest / 哈希
  command.txt       # 真实命令、开始时间与时区、commit/dirty 状态
  run.log           # stdout/stderr、失败记录、退出码
summary/            # 脚本生成的平均时间和加速比
figures/            # 由真实数据生成的图
```

run_id 建议含时间、成员、实现，例如 `<timestamp>_<member>_<implementation>`，不要覆盖已有运行目录。
CPU/GPU 计算时间与端到端时间分别汇总；Python 与 standalone 比较需标清调用方式。
保留三次有效运行的全部原始行；同时报告失败或缺测，不悄悄挑选最快一次。
当前只有目录占位，没有正式实验结果。
