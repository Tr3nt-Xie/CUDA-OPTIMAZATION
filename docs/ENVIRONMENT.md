# Lab Environment Record (owner: A)

Status: as of 2026-10-06 PDT, the shared AWS us-west-2 T4 environment is set up; A's full-size formal CPU/naive measurements, the supplementary small-size measurements, and the CPU convolution baseline are complete, as are the build, self-test, and three matrix-size runs inside a GPU container. B's and C's implementations and performance comparisons are done under their own Parts.

| Field | Actual value |
|---|---|
| Machine name / platform | lab6-cuda; AWS us-west-2a; on-demand g4dn.2xlarge |
| OS / CPU / memory | Ubuntu 22.04.5 LTS, Linux 6.8.0-1066-aws; Xeon Platinum 8259CL, 8 vCPU; 32 GiB instance spec, about 30 GiB reported by the system |
| GPU model / memory | NVIDIA Tesla T4; nvidia-smi reports 15360 MiB |
| NVIDIA driver / CUDA Toolkit | 595.91.07 / 12.8 (nvcc 12.8.93); the CUDA 13.2 shown by nvidia-smi is the version the driver supports, not the compiler used here |
| C compiler / Python / NumPy / Pillow | GCC 11.4.0 / Python 3.10.12 / NumPy 2.2.6 / Pillow 12.3.0 |
| Actual build command and options | `export NVCCFLAGS='-O2 -std=c++14 -arch=sm_75'; make -j4 all`; C uses the Makefile default `-O2 -std=c11 -Wall -Wextra -Wpedantic` |
| Code commit / uncommitted changes | Base commit `a2f7c1c3bb871ff5bcdf38097ac740e9e8927f3b`; clean during the initial environment check. During the formal measurements a matrix collection script was added and the convolution recording script modified, so the worktree was dirty; algorithm sources were unchanged, and the actual scripts and source hashes/snapshots are kept with each batch |
| Environment check commands, output location, and date | 2026-10-07 05:59–06:02 UTC, i.e. 10-06 22:59–23:02 PDT; [raw record](../results/raw/A-20261007-055952-aws-smoke/) |

For every formal experiment, first create a new result directory, then save the output of `scripts/collect_env.sh` as `environment.txt` in it, for example `scripts/collect_env.sh > results/raw/<run-id>/environment.txt`.
Never record account passwords, access keys, or private keys.

## Completed checks

- `make -j4 all` builds successfully; `build/bench_cpu selftest` and `build/bench_cuda selftest --impl naive` all pass.
- CPU and naive were each run at N=256 with 1 warm-up and 1 measured call, and passed numerical validation. These CSVs were only used to check the environment; no formal performance conclusions, Q1/Q2 answers, or speedup curves were derived from them.
- Docker 29.8.2 and NVIDIA Container Toolkit 1.20.1 are available. Inside the official `nvidia/cuda:12.8.1-devel-ubuntu22.04` container, `nvidia-smi`, `nvcc --version`, `make all`, and the naive CUDA self-test were actually run, and all passed.
- The container image digest is `sha256:a99a1860ba8e2916e5c3e73b72ec4c4301653a84586e05bfc9a2aa2d58027e97`. The run uses `--gpus all`, a non-root user, a read-only source mount, and a temporary build directory; the container is removed automatically when it exits, and the image is kept for reproduction. The full commands are in the raw record's README.
- SSH and GPU access were verified for the ubuntu, lab6-b, and lab6-c accounts. B and C each have their own home directory, repository, and `.venv`, without sudo or Docker admin rights; each person configures GitHub write access with their own identity.

## A's formal experiments

2026-10-06 23:21–23:28 PDT; measurements and container checks were run sequentially:

- [Matrix batch](../results/raw/A-20261006-232110-matrix-aws/manifest.json): CPU and naive each cover formal N=256/512/1024/2048 and supplementary N=16/32/64/128, with 1 warm-up and 3 measured calls each; all 48 measured calls are correct. The CPU version is a single-threaded triple loop and does not use multithreaded BLAS.
- [CPU convolution batch](../results/raw/A-20261006-232110-part81-cpu-aws/manifest.json): 3 source images, 33 configurations, all 99 measured calls correct, plus 9 independent NumPy checks. This batch is the same-host baseline for the CPU/CUDA convolution comparison.
- [Container matrix batch](../results/raw/A-20261006-232110-container-part3/manifest.json): pinned to the same official image digest, rebuilt and self-tested inside the container, with 1 warm-up and 3 measured calls at each of N=256/512/1024, all passing. This batch proves Part 3 execution and is not mixed with host measurements.

Formal builds go into separate `build/<run-id>/` directories to avoid reusing old binaries; the full build commands, input seeds/hashes, source snapshots, binary hashes, and exit codes are in each batch's records. The matrix and convolution algorithms are the base commit's implementations. The single-call smoke CSV from deployment is not part of any formal mean.

The inputs and CPU reference outputs were copied to the read-only shared directory `/opt/lab6-a-reference/20261006-232110/`. B's and C's regular accounts each verified the SHA-256 of the 9 inputs and 9 outputs, without modifying their repositories. See [the results index](../results/README.md) for how to use the directory.

## Using the environment

```bash
cd ~/CUDA-OPTIMAZATION
source ~/.venv/bin/activate
export NVCCFLAGS='-O2 -std=c++14 -arch=sm_75'
make -j4 all
build/bench_cuda selftest --impl naive
```

All three members' login environments use `/usr/local/cuda-12.8`. The build options match the T4; on a different GPU, record the options actually used, and do not treat `sm_75` as a build requirement for every platform. Formal comparisons continue to follow the data and timing conventions in `docs/CONTRACTS.md`.

Connection addresses, private keys, and the administrator's start/stop instructions are kept outside the repository and handed over separately by the administrator. The public IP may change after the instance is stopped and restarted.

The GPU examples in handout Part 3 include T4/V100, and the handout asks for a GPU container; this lab uses an AWS T4, and real container build and run evidence is saved. Any GPU requirement mentioned verbally in class is subject to the instructor's confirmation; these technical checks do not mean the course submission has been made or accepted.
