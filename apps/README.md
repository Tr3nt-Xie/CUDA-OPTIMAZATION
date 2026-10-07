# 统一测试程序（A 维护）

`harness.c` 负责参数解析、按 seed 生成输入或读取图片、调用 `lab6_*`、与双精度参考比对、预热 + 计时、输出 CSV。
`bench_cpu.c`（`make cpu`）和 `bench_cuda.cu`（`make cuda`）登记了当前实现入口；接入现有接口后，可用 `--impl` 选择要测试的实现。

| 程序 | 实现名 |
|---|---|
| build/bench_cpu | `cpu`（矩阵 + 卷积） |
| build/bench_cuda | `naive` `tiled` `optimized` `cublas`（矩阵），`cuda`（卷积） |

其他实现尚未完成时，显式使用 `--impl` 选择本人已完成的实现；默认 `all` 会调用相应操作下的全部入口，包括占位函数。统一参数以 `configs/benchmark.json` 为准；当前默认值与其一致，程序不直接读取 JSON。

```text
bench_* <selftest|matrix|conv> [options]
  --impl LIST     逗号分隔的实现名或 all（默认 all）
  --n LIST        矩阵尺寸（默认 256,512,1024,2048）
  --m LIST        图像尺寸（默认 512,1024,2048）
  --k LIST        滤波器尺寸（默认 3,5,7）
  --filter NAME   mean|edge|sharpen（默认 mean）
  --image PATH    data/generated 下的 .u32 图片；不给则按 seed 随机生成
  --save PATH     保存 float32 输出（仅限一个实现 / 一个 M / 一个 K）
  --csv PATH      追加写 CSV，文件为空时写表头；父目录需事先存在
  --member NAME   CSV 的 member 列（默认取环境变量 LAB6_MEMBER）
  --warmup N --runs N --seed N --atol X --rtol X   默认 1 / 3 / 542 / 1e-3 / 1e-4
```

- `selftest`：手算样例（非对称矩阵、只有左上角非零的卷积核，可区分卷积与互相关）、尺寸 2/3/31 随机用例、`timings=NULL` 路径、非法参数拒绝。
- `matrix` / `conv`：每个配置先预热，再测 `--runs` 次；每次运行写 compute 和 end_to_end 两行 CSV，终端打印均值。
- 库函数返回非零状态（包括未实现）时，该次调用不写计时行，程序最终退出码为 1。测量阶段库函数返回成功、但数值校验失败的行保留 `correct=false`，程序同样非零退出；参数错误退出码为 2。
- 双精度参考算 N=2048 需要几秒，同一次调用里多个实现共用一份；需要比较多个实现时用逗号一起传。
