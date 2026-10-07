# 实验结果

```text
raw/<member>-<YYYYMMDD-HHMM>/
  *.csv             # 可用 measurements.csv 或按配置拆分；统一列见 configs/benchmark.json
  environment.txt   # scripts/collect_env.sh 生成
summary/            # 汇总脚本生成的均值和加速比
figures/            # 由真实数据生成的图
```

每次正式测试新建一个目录，不覆盖已有结果；失败或缺测在报告里说明，不挑最快的一次。CSV 的拆分和文件命名可按实际运行安排，但需保留输入标识及其与结果的对应关系，汇总时不混合不同输入。

## A 的 Part 8.1 本地结果

当前报告采用 `A-20261006-195441-part81-cpu-local`，2026-10-06 19:54 PDT，在 macOS arm64 上完成。33 个配置、99 次计时运行全部正确，另有 9 份展示输出通过独立 NumPy 检查。

- [实验清单及输入哈希](raw/A-20261006-195441-part81-cpu-local/manifest.json)
- [汇总 CSV](summary/A-20261006-195441-part81-cpu-local.csv)
- [滤波效果总览](figures/A-20261006-195441-part81-cpu-local/filter-comparison.png)

每个原始 CSV 的文件名包含图片、M、K 和滤波器，内部仍保留全组统一列。一个配置有 3 次运行 × 2 个 timing_scope，共 6 行。图像读取、参考计算、验证、PNG 生成不计入 CPU compute；当前 CPU 实现的 end_to_end 与 compute 相等。独立核验针对未裁剪的 float32 输出。

保留了之前的运行记录：`195231` 在可选硬件查询被本机权限限制后停止，尚未计时；`195302` 完整通过。最终修改使脚本优先使用仓库原图并校验哈希，避免不同图片库版本重新生成输入，因此按最终脚本完整重跑得到 `195441`。报告使用这个完整批次，没有从多个批次挑选最快数据。

实际 CPU 型号与内存查询不可用，`environment.txt` 保留了原始错误；已记录 macOS/arm64、编译器、编译选项和依赖版本。这些结果用于本地验证。正式 CPU/CUDA 加速比仍需在同一台实验机重新采集。
