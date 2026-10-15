# EE542 Lab6 — CUDA Optimization

Matrix multiplication and image convolution on the CPU and an NVIDIA GPU: a naive CUDA kernel, shared-memory tiling, a register-blocked optimized kernel, cuBLAS, and a CUDA shared library called from Python. All measurements were taken on an AWS g4dn.2xlarge with a Tesla T4 and CUDA 12.8.

## Team

| Member | Parts |
|---|---|
| A — Wenxu Zhang | 1–3: CPU, naive CUDA, cloud GPU environment; 8.1: CPU convolution; shared framework |
| B — Fengmao Xie | 4: tiled CUDA; 6.2: optimized CUDA; 8.2: CUDA convolution |
| C — Xiaopeng Wu | 5: performance comparison; 6.1: cuBLAS; 7 and 8.3: shared library and Python |

## Layout

```text
.
├── include/lab6.h           # Public C interface, error codes, timing results
├── src/
│   ├── common/             # A · shared code
│   │   ├── cuda_utils.cuh  #     Common GPU flow run_matmul / run_convolution (allocate, copy, time, free)
│   │   ├── validate.h      #     Argument checks
│   │   └── timer.h         #     Monotonic clock
│   ├── matrix/
│   │   ├── cpu.c           # A · Part 1  CPU triple loop
│   │   ├── naive.cu        # A · Part 2  One output element per thread, 16×16 blocks
│   │   ├── tiled.cu        # B · Part 4
│   │   ├── optimized.cu    # B · Part 6.2
│   │   └── cublas.cu       # C · Part 6.1
│   ├── convolution/
│   │   ├── cpu.c           # A · Part 8.1  CPU mathematical convolution, zero padding
│   │   └── cuda.cu         # B · Part 8.2
│   └── bindings/exports.cu # C · Parts 7 / 8.3, exports for Python
├── apps/                   # A · shared benchmark program
│   ├── harness.h / .c      #     selftest / matrix / conv, double-precision reference, filters, warm-up timing, CSV
│   ├── bench_cpu.c         #     Registers cpu
│   └── bench_cuda.cu       #     Registers naive / tiled / optimized / cublas / cuda
├── python/                 # C · ctypes wrapper and Python experiment entry points
├── scripts/
│   ├── prepare_images.py   # A · image ↔ raw .u32 image / filter output → PNG
│   ├── generate_test_images.py # A · original grayscale test images
│   ├── run_part81_cpu.py    # A · Part 8.1 CPU experiments and result images
│   ├── run_part_a_matrix.py # A · CPU/naive formal and supplementary matrix runs on one host
│   ├── plot_part_a.py      # A · baseline curves and speedups
│   ├── collect_env.sh      # A · generates environment.txt
│   └── requirements.txt    # A · NumPy, Pillow, Matplotlib
├── configs/benchmark.json  # Team-wide benchmark parameters
├── data/                   # A
│   ├── images/             #     Source images for the convolution demo (committed, with sources)
│   └── generated/          #     .u32 files written by prepare_images.py (ignored by Git)
├── results/
│   ├── raw/                # Raw CSVs, logs, and run environments
│   ├── summary/            # Summary tables generated from raw data
│   └── figures/            # Figures used in the reports
├── docs/
│   ├── CONTRACTS.md        # Shared interface, data, timing, and collaboration conventions
│   ├── ENVIRONMENT.md      # A · actual GPU environment and run evidence
│   └── submission-checklist.md # Mapping of Parts, owners, and deliverables
├── report/                 # Per-member report materials and the final report
├── .clang-format           # Shared C/CUDA code style
└── Makefile                # CPU static library / CUDA shared library / benchmark builds / make test
```

## Build

Requires Make and a C compiler; the GPU build also needs the NVIDIA CUDA Toolkit (nvcc and cuBLAS) on Linux.

```sh
make cpu       # build/liblab6_cpu.a, build/bench_cpu
make cuda      # build/liblab6_cuda.so, build/bench_cuda
make all       # both
make test      # CPU self-test
```

On the T4, set `NVCCFLAGS='-O2 -std=c++14 -arch=sm_75'`; change `-arch` for other GPUs.

## Run

```sh
build/bench_cuda selftest --impl naive,tiled,optimized,cublas,cuda
build/bench_cuda matrix --impl naive,tiled,optimized,cublas --n 512,1024,2048
build/bench_cpu matrix --impl cpu --n 512,1024
```

Python (3.10+, from the repository root):

```python
import numpy as np, sys
sys.path.insert(0, "python")
from lab6_cuda import CudaLibrary

lib = CudaLibrary()  # loads build/liblab6_cuda.so
a = np.random.rand(1024, 1024).astype(np.float32)
b = np.random.rand(1024, 1024).astype(np.float32)
c = lib.matrix_multiply(a, b).output  # Part 4 tiled kernel via gpu_matrix_multiply
```

More details: [apps/README.md](apps/README.md) (benchmark options), [python/README.md](python/README.md) (Python interface and Part C runs), [scripts/README.md](scripts/README.md) (data and plots), [results/README.md](results/README.md) (measurement batches), [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md) (GPU environment).
