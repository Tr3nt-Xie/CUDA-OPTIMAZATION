# 实验环境记录（A 负责）

状态：2026-10-06 PDT 已完成 AWS us-west-2 共用 T4 环境、A 的 CPU/naive 全尺寸正式测量、补充小尺寸测量、CPU 卷积基线，以及 GPU 容器内的编译、自测和三种矩阵尺寸运行。B/C 的实现与性能比较仍按各自 Part 完成。

| 字段 | 实际值 |
|---|---|
| 实验机代号 / 平台 | lab6-cuda；AWS us-west-2a；按需 g4dn.2xlarge |
| OS / CPU / 内存 | Ubuntu 22.04.5 LTS，Linux 6.8.0-1066-aws；Xeon Platinum 8259CL，8 vCPU；实例规格 32 GiB，系统显示约 30 GiB |
| GPU 型号 / 显存 | NVIDIA Tesla T4；nvidia-smi 显示 15360 MiB |
| NVIDIA driver / CUDA Toolkit | 595.91.07 / 12.8（nvcc 12.8.93）；nvidia-smi 的 CUDA 13.2 为驱动支持版本，非本次编译工具版本 |
| C 编译器 / Python / NumPy / Pillow | GCC 11.4.0 / Python 3.10.12 / NumPy 2.2.6 / Pillow 12.3.0 |
| 实际构建命令与优化选项 | `export NVCCFLAGS='-O2 -std=c++14 -arch=sm_75'; make -j4 all`；C 为 Makefile 默认 `-O2 -std=c11 -Wall -Wextra -Wpedantic` |
| 代码 commit / 未提交变更 | 基础 commit `a2f7c1c3bb871ff5bcdf38097ac740e9e8927f3b`；最初环境验证时 clean，正式测量时新增矩阵采集脚本、修改卷积记录脚本，工作区 dirty；算法源码未变，实际脚本与源码哈希/快照随批次保留 |
| 环境验证命令、输出位置与日期 | 2026-10-07 05:59–06:02 UTC，即 10-06 22:59–23:02 PDT；[原始记录](../results/raw/A-20261007-055952-aws-smoke/) |

每次正式实验先新建结果目录，再将 `scripts/collect_env.sh` 的输出保存为该目录中的 `environment.txt`；命令示例见仓库 README。
不要记录账户密码、访问密钥或私钥。

## 已完成的验证

- `make -j4 all` 编译成功；`build/bench_cpu selftest` 与 `build/bench_cuda selftest --impl naive` 全部通过。
- CPU、naive 各运行 N=256，预热 1 次、计时 1 次，数值校验通过。CSV 仅用于环境运行检查；没有据此生成正式性能结论、Q1/Q2 答案或加速比曲线。
- Docker 29.8.2、NVIDIA Container Toolkit 1.20.1 已可用。官方 `nvidia/cuda:12.8.1-devel-ubuntu22.04` 容器中实际执行了 `nvidia-smi`、`nvcc --version`、`make all` 和 naive CUDA 自测，均通过。
- 容器镜像 digest 为 `sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97`。使用 `--gpus all`、非 root 用户、只读源码挂载和临时构建目录；运行结束自动移除容器，镜像保留用于复现。完整命令见原始记录的 README。
- ubuntu、lab6-b、lab6-c 的 SSH 与 GPU 访问已验证。B/C 各有独立 home、仓库和 `.venv`，没有 sudo/Docker 管理权限；GitHub 写入认证由各人使用自己的身份配置。

## A 的正式实验

2026-10-06 23:21–23:28 PDT，测量与容器验证顺序执行：

- [矩阵批次](../results/raw/A-20261006-232110-matrix-aws/manifest.json)：CPU/naive 各覆盖正式 N=256/512/1024/2048 及补充 N=16/32/64/128。各预热 1 次、计时 3 次，48 次计时调用全部正确。CPU 为单线程三重循环，不使用多线程 BLAS。
- [CPU 卷积批次](../results/raw/A-20261006-232110-part81-cpu-aws/manifest.json)：3 张原图，33 个配置，99 次计时调用全部正确，另有 9 份 NumPy 独立检查。此批次是 CPU/CUDA 卷积对比的同机基线。
- [容器矩阵批次](../results/raw/A-20261006-232110-container-part3/manifest.json)：固定同一官方镜像 digest，在容器内重新编译、自检，N=256/512/1024 各预热 1 次、计时 3 次，全部通过。该批次用于 Part 3 执行证明，不与宿主机测量混合。

正式构建分别在 `build/<run-id>/`，避免复用旧二进制；完整编译命令、输入种子/哈希、源码快照、二进制哈希和退出码见各批次记录。矩阵和卷积数值算法保持基础 commit 的实现。部署阶段的一次计时 smoke CSV 不参与正式均值。

输入及 CPU 参考输出已复制到只读共享目录 `/opt/lab6-a-reference/20261006-232110/`。B/C 普通账号分别验证了 9 个输入和 9 个输出的 SHA-256，未修改他们的仓库。目录用法见 [结果入口](../results/README.md)。

## 使用环境

```bash
cd ~/CUDA-OPTIMAZATION
source ~/.venv/bin/activate
export NVCCFLAGS='-O2 -std=c++14 -arch=sm_75'
make -j4 all
build/bench_cuda selftest --impl naive
```

三人的登录环境统一选择 `/usr/local/cuda-12.8`。构建选项与 T4 匹配；换 GPU 时重新记录实际选项，不将 `sm_75` 作为所有平台的构建要求。正式比较继续遵守 `docs/CONTRACTS.md` 的数据和计时约定。

连接地址、私钥及管理员开关机说明保存在仓库外，由管理员单独交接。实例停止后再次启动可能更换公网 IP。

PDF Part 3 的 GPU 示例包括 T4/V100，并要求创建 GPU 容器；本次采用 AWS T4，已保存真实的容器编译运行证据。此前提到的课堂口头显卡要求仍以老师确认内容为准；这些技术验证不代表课程已经提交或验收。
