# Shared Development and Testing Conventions

This document records the interface, data, correctness, timing, and collaboration conventions agreed by the team, used for integrating each implementation and comparing results. The handout `lab6-f26.pdf` defines the tasks and algorithm requirements; each member decides their internal implementation within these shared conventions.

`configs/benchmark.json` is the source of truth for benchmark parameters. When it changes, update the affected test entry points and documentation and tell the team, so nobody maintains a different set of parameters. The existing C/CUDA benchmark programs use the same defaults but do not read the JSON automatically; after a configuration change, update the defaults or pass the values explicitly on the command line. How scripts read the configuration is up to their authors.

## Data and correctness

- Matrices: `float32`, C row-major contiguous storage, `C=A×B`. All four values of N are run in the formal tests: the handout specifies 512, 1024, and 2048, and the team added 256. Small sizes 2, 3, and 31 are used for correctness and non-block-aligned boundary checks.
- The matrix benchmark generates inputs in-program with a fixed seed and splitmix64, with values in [0,1). Reference results are computed by an independent double-precision implementation in the benchmark program, and the CPU implementation is checked against it as well. Cross-implementation comparisons use identical inputs; the same seed in different random generators does not mean the same data.
- Source images for the convolution tests live in `data/images/` with their sources noted. A script converts them to raw `uint32` binaries at each size in `data/generated/`; C and Python read the same files.
- Convolution images are `uint32` grayscale values 0–255; weights and raw outputs are `float32`. Negative outputs are kept; clipping/mapping is applied only to display images, and display images are never used for numerical validation.
- The team uses mathematical convolution with stride=1, zero padding, same-size output, and a flipped filter. K is a positive odd number and r=K/2:
  `out[y,x] = Σ filter[i,j] × image[y+r-i, x+r-j]`, with out-of-bounds image values treated as 0.
- Formal convolution tests use the 9-way Cartesian product of M=512/1024/2048 and K=3/5/7, all with the normalized mean filter. Edge and sharpen filters are additionally used to show image effects. The three filters in `apps/harness.c` are defined for any odd K as:
  - mean: every element 1/K²
  - edge: every element -1, center K²-1 (elements sum to 0)
  - sharpen: 2×identity − mean, i.e. every element -1/K², plus 2 at the center (elements sum to 1)
- Three image sizes and three filter sizes come from the handout; the specific values, the nine pairings, and the boundary rule are team conventions.
- Correctness requires `abs(got-ref) <= 1e-3 + 1e-4*abs(ref)` element-wise, rejects NaN/Inf, and records the maximum absolute error. Hand-computed small examples are in the benchmark program's `selftest`; the convolution case uses a filter that is nonzero only in its top-left corner, to distinguish convolution from correlation.
- CPU, CUDA, and Python outputs are all checked against the same reference. cuBLAS results must also match the row-major `C=A×B` definition; identical inputs must not be used in a way that hides transpose mistakes. Formal comparisons stay in FP32 arithmetic without enabling TF32 or fast-math, under the same correctness criteria.

## Code and interfaces

Public declarations are in `include/lab6.h`, which keeps an ABI callable from C and `ctypes` and does not expose CUDA types in the public C header. Every implementation takes **host pointers**; outputs are allocated by the caller and must not overlap the inputs. Implementations must validate sizes and pointers and guard against integer overflow when computing byte counts; CUDA errors must return nonzero and free any allocated resources.

`lab6_matmul_*` / `lab6_convolve_*` are the calling interfaces of the existing C/CUDA benchmark entry points; the host output is ready when they return. Device memory management, kernel organization, and call flow inside each implementation are up to the implementer.

`src/common/cuda_utils.cuh` provides helper wrappers for argument checks, allocation, transfers, timing, synchronization, and cleanup; `src/matrix/naive.cu` uses them. They can be reused as needed.

The last argument, `lab6_timings *`, may be NULL:
- Non-NULL: fill in the compute and end_to_end milliseconds for this call.
- NULL: the caller does not read the timings; the result must still be ready on return.
- Nonzero return: the output and timings are invalid and must not be recorded as a success or included in performance averages.

Following handout Parts 7 / 8.3, Python calls the Part 4 tiled matrix implementation and the Part 8.2 CUDA convolution implementation; `src/bindings/exports.cu` connects these entry points. Before calling, Python checks dtype, shape, and C-contiguous layout; `ctypes` argument types match `include/lab6.h`, and `restype=c_int` is set. A nonzero return value is treated as a failed call, and the output is not used as a valid result. The wrapper and file organization are C's choice.

The shared library's exported names stay `gpu_matrix_multiply` and `gpu_convolve`. Changing the handout example's void return to int is the team's error-handling convention, and Python must match the header.

## Timing and results

Each configuration gets 1 warm-up call followed by 3 measured calls; all records are kept and the arithmetic mean is reported. The warm-up count, the three repetitions, and the mean are team rules.

| scope | Timing boundary |
|---|---|
| CPU compute | CPU arithmetic only, excluding input generation and file I/O |
| GPU compute | CUDA events around the kernel/cuBLAS computation, properly synchronized, excluding device allocation and transfers |
| C/CUDA end_to_end | From existing host inputs to returned results, including this call's allocation, H2D, computation, D2H, and cleanup; library handle creation is included if done in this call |
| Python end_to_end | From existing valid NumPy inputs to completion of the ctypes call, including the cross-language call and the GPU flow above |

When the CPU has no transfers, both scopes may be recorded from the same call, with the boundary clearly noted. Python compute timings can be obtained through the public `lab6_*(..., timings)` interfaces; Python total time is measured with an outer monotonic clock. Do not label outer total time as kernel time.

Formal CPU/GPU data is collected on **the same lab machine**; the shared GPU is used by one person at a time, and performance experiments never run concurrently. Every formal batch keeps raw CSVs in the shared format and one `environment.txt` (hardware, driver/Toolkit, build command, git commit); see `results/README.md` for layout. CSVs may be split by configuration, as long as file names or a run manifest keep the input identity so summaries can still tell different images or matrix inputs apart. The cloud platform and GPU model are not hard-coded.

CSV columns follow the order in the configuration. Each row is one timing_scope of one run:
- `operation`: matrix / convolution; `N` is the matrix side length, `M,K` are the convolution sizes, and `filter` is mean / edge / sharpen. Fields that do not apply are left empty.
- `implementation`: cpu / naive / tiled / optimized / cublas / cuda / python_tiled / python_cuda.
- `run`: 1–3; warm-up calls are not written to the formal CSV. `correct`: true / false.
- Rows with a valid time but a failed correctness check keep false; the summary scripts exclude them and report them explicitly.
- CPU/GPU speedups must use means over identical inputs and the same scope, never mixing compute and end_to_end.
- Further optimization is not guaranteed to beat cuBLAS; real improvements or regressions are kept along with their explanations.

## Ownership and integration

See `docs/submission-checklist.md` for responsibilities. When the public interface or configuration changes, update the affected callers and test instructions. Implementations of other members that have not yet been delivered stay as placeholders.

## Shared development rules

- Develop on feature branches; suggested names are `feat/a-baselines`, `feat/b-cuda`, and `feat/c-integration`.
- Changes to the public header, configuration, or Makefile are explained in the PR and shared with the team.
- `make cpu` must pass before merging into main; `make cuda` must also pass where a CUDA environment is available. GPU builds and runs that have not been done yet are explicitly marked as pending verification.
- Committed C/CUDA code is formatted with the repository's `.clang-format`, avoiding unrelated repository-wide formatting changes.
- Everyone commits under their own Git identity and does not rewrite teammates' history.
- Each implementation first passes small-scale correctness checks and is then measured with the shared parameters; unfinished functions of other members keep returning `LAB6_NOT_IMPLEMENTED`.

## Handoff order

1. A provides the public interface, CPU reference implementations, test inputs, and test tools; B and C can develop their Parts in parallel.
2. Once A's and B's implementations run, C integrates the shared library and Python; each member checks the correctness of their own implementation.
3. Members take turns measuring on the same lab machine and hand over raw CSVs, environment.txt, run commands, and their report materials.
4. C summarizes data and figures, A merges the report, and B checks the source and build reproducibility.
