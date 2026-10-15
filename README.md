# EE542 Lab6 — CUDA Optimization

This repository contains the code and experiment records for matrix multiplication, convolution, CUDA optimization, and calling a CUDA shared library from Python.

**Current status: the framework (shared CUDA host flow, benchmark program, scripts) and A's CPU / naive implementations are complete. B's tiled, optimized, and CUDA convolution implementations are done, with formal measurements on the same T4 (see [B's report materials](report/B.md)). C's cuBLAS, Python interface, measurement collection, and summary/plotting code are implemented; see [the Python and Part C guide](python/README.md).**
A's Parts 1–3 and 8.1 experiments were completed on the same AWS host: 48 timed calls across the four formal matrix sizes and the supplementary small sizes, plus 33 CPU convolution configurations, 99 timed calls, and 9 independent image checks, all passing. Three matrix sizes were also run inside a GPU container. See [A's report materials](report/A.md), [raw data and figures](results/README.md), and [the actual environment record](docs/ENVIRONMENT.md). A's Q1/Q2 answers are in the report materials; merging the whole team's materials is still pending.
The assignment is the course handout `lab6-f26.pdf` (the original file is kept in the lab directory outside this repository).

## Layout and responsibilities

| Member | Responsibilities |
|---|---|
| A | Parts 1–3: CPU matrix multiplication, naive CUDA, GPU environment; Part 8.1: CPU convolution, test images and filters; shared framework (common CUDA flow, benchmark program); final report merge |
| B | Parts 4 and 6.2: tiled matrix multiplication and further optimization; Part 8.2: CUDA convolution; build reproducibility check |
| C | Parts 5 and 6.1: cuBLAS, performance summary and plots; Parts 7 and 8.3: shared library and Python calls |

The owner of each file is marked in the tree below.

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

A/B/C are role labels for the division of work, not a record of completed contributions. See [the submission checklist](docs/submission-checklist.md) for handoffs.

## Getting started

The handout `lab6-f26.pdf` defines the tasks, algorithm requirements, and deliverables; each member completes their Parts according to [the division of work](docs/submission-checklist.md). The tree above shows where the code lives; internal implementation, reuse of helper wrappers, script organization, and file naming are up to each member, following the handout.

The team's shared interface, data, testing, and submission conventions still apply; see [shared development and testing conventions](docs/CONTRACTS.md), with public declarations in [lab6.h](include/lab6.h). `src/common/cuda_utils.cuh` holds the helper code used by A's naive implementation and can be reused as needed.

The test entry point supports `bench_cuda selftest --impl <name>` to check a single implementation. The repository contains the CPU, naive, tiled, optimized, cuBLAS, and CUDA convolution implementations; select which ones to check with `--impl`.

Before merging into main, `make cpu` must pass, and `make cuda` as well where a CUDA environment is available. Committed C/CUDA code uses the repository's `.clang-format`. Everyone commits under their own Git identity; any change to the public interface, configuration, or build is explained and shared with the team.

Matrix sizes **256, 512, 1024, and 2048 are all run, not just one of them**. The last three come from the handout; 256 is a team addition. Convolution uses three image sizes × three filter sizes, nine combinations in total; see the JSON configuration.

## Build

Run from the repository root. Requires Make and a C compiler; GPU builds also need the NVIDIA CUDA Toolkit (nvcc and cuBLAS).

```sh
make help
make cpu       # build/liblab6_cpu.a + build/bench_cpu; works as a local build check on a Mac
make cuda      # build/liblab6_cuda.so + build/bench_cuda; run in a Linux CUDA environment
make all       # all of the above
make test      # build/bench_cpu selftest
make clean     # remove build/
```

CUDA sources are compiled one by one and then linked into the shared library: if a `.cu` file fails, the error names that file, and the other files rebuild incrementally.

Use `make cpu CC=gcc` or `make cuda NVCC=/path/to/nvcc` to choose a compiler.
The CPU build defaults to `-O2`. The CUDA architecture depends on the target host; set `NVCCFLAGS` when needed, for example adding the host's `-arch`. The repository does not pin a specific GPU model.

## Run

The following is A's naive matrix benchmark example, to be run once the GPU environment is ready and the implementation passes its correctness checks. All four values of N are tested by default; B/C can select their own implementations with `--impl`. Use a new result directory for every run.

```sh
build/bench_cuda selftest --impl naive
lab6_run_dir="results/raw/A-$(date +%Y%m%d-%H%M%S)-naive"
mkdir "$lab6_run_dir"
scripts/collect_env.sh > "$lab6_run_dir/environment.txt"
build/bench_cuda matrix --impl naive --member A --csv "$lab6_run_dir/measurements.csv"
```

See [apps/README.md](apps/README.md) for all options and [scripts/README.md](scripts/README.md) for image conversion. [The Python interface, Part C run instructions, and plotting commands](python/README.md) are also provided; new results go into their own run ID directory.

A Mac can be used for CPU correctness checks, image demos, and local timing; formal CPU/GPU comparisons use data from the shared lab machine. A's AWS CPU/naive baseline has been collected, and B/C's implementations can be compared under the same conventions. Connection details for the shared environment are provided separately by A. Fixed inputs and CPU reference outputs are in the read-only directory `/opt/lab6-a-reference/20261006-232110/` on the lab machine; see the results index.

## What is committed

Source code, build/run instructions, configuration, small validation inputs, raw results and environment records, summary tables and figures, and report materials are kept.
Build outputs, virtual environments, keys, and large regenerable inputs are not committed. The shared library can be built from source, so locally built binaries are not added to Git.
