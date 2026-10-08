# AWS T4 环境验证记录

日期：2026-10-07 05:59–06:02 UTC（10-06 22:59–23:02 PDT）。
平台：AWS us-west-2a，g4dn.2xlarge，Ubuntu 22.04.5，Tesla T4。
源码：`a2f7c1c3bb871ff5bcdf38097ac740e9e8927f3b`，远端初始工作区 clean。

宿主机命令：

```bash
cd ~/CUDA-OPTIMAZATION
source /etc/profile.d/lab6-cuda.sh
source ~/.venv/bin/activate
export NVCCFLAGS='-O2 -std=c++14 -arch=sm_75'
scripts/collect_env.sh > environment.txt
make -j4 all
build/bench_cpu selftest
build/bench_cuda selftest --impl naive
build/bench_cpu matrix --impl cpu --n 256 --runs 1 --warmup 1 --member A --csv cpu-smoke.csv
build/bench_cuda matrix --impl naive --n 256 --runs 1 --warmup 1 --member A --csv naive-smoke.csv
```

实际日志位于本目录。两个 CSV 各 2 行（compute/end_to_end），全部 `correct=true`。只用于证明环境可运行，不能替代全部尺寸、三次重复的正式实验。

GPU 容器在管理员账号运行，使用如下命令；不需要给队友 Docker 管理权限：

```bash
sudo docker pull nvidia/cuda:12.8.1-devel-ubuntu22.04
sudo docker run --rm --gpus all --network none --cap-drop ALL --security-opt no-new-privileges \
  --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$HOME/CUDA-OPTIMAZATION:/src:ro" -w /tmp \
  nvidia/cuda:12.8.1-devel-ubuntu22.04 bash -lc '
    set -e
    mkdir /tmp/lab6
    cp -r /src/include /src/src /src/apps /src/Makefile /tmp/lab6/
    cd /tmp/lab6
    nvidia-smi
    nvcc --version
    make -j4 all NVCCFLAGS="-O2 -std=c++14 -arch=sm_75"
    build/bench_cuda selftest --impl naive
  '
```

实际拉取的镜像 digest 保存在 `container-digest.txt`；完整成功输出见 `container-gpu-test.log`。容器结束后自动删除，原镜像保留。B/C 占位文件仅参与编译，没有将其记录为功能或性能验证通过。
