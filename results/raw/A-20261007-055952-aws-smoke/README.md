# AWS T4 Environment Check Record

Date: 2026-10-07 05:59–06:02 UTC (10-06 22:59–23:02 PDT).
Platform: AWS us-west-2a, g4dn.2xlarge, Ubuntu 22.04.5, Tesla T4.
Source: `a2f7c1c3bb871ff5bcdf38097ac740e9e8927f3b`; the remote worktree was initially clean.

Host commands:

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

The actual logs are in this directory. Each of the two CSVs has 2 rows (compute/end_to_end), all with `correct=true`. They only prove that the environment runs; they do not replace the formal experiments with all sizes and three repetitions.

The GPU container was run under the administrator account with the command below; teammates do not need Docker admin rights:

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

The digest of the image actually pulled is saved in `container-digest.txt`; the full successful output is in `container-gpu-test.log`. The container is deleted automatically when it exits, and the image is kept. B's and C's placeholder files were only compiled; they were not recorded as passing any functional or performance check.
