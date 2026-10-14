# Member C: cuBLAS and Python CUDA Integration

This contribution covers Parts 5, 6.1, 7, and 8.3. It implements FP32 cuBLAS matrix multiplication, NumPy/ctypes bindings, correctness checks, benchmark collection, and result plots. The existing C ABI, Makefile, and team benchmark configuration are unchanged.

## Files

- `src/matrix/cublas.cu`: row-major FP32 matrix multiplication using `cublasSgemm`, error handling, and compute/end-to-end timing.
- `python/lab6_cuda.py`: validated NumPy/ctypes calls to the shared library.
- `python/benchmark.py`: independent references, self-tests, and measurements.
- `python/demo.py`: mean, edge-detection, and sharpening image filters.
- `scripts/run_part_c.py`: sequential Linux CUDA collection with environment, source hashes, and execution logs.
- `scripts/summarize_results.py`: correctness/coverage checks, means, sample standard deviations, speedups, and plots.
- `scripts/plot_c_final.py`: additional presentation figures from a completed full run.
- `tests/test_part_c.py`: CPU-only tests of input contracts, routing, references, and aggregation.

## Build and Check

Run from the repository root on Linux with a CUDA-capable NVIDIA GPU. Python requires version 3.10 or later. Install the existing Python dependencies:

```sh
python -m pip install -r scripts/requirements.txt
python -m unittest discover -s tests -v
```

The AWS T4 setup uses CUDA 12.8 and architecture sm_75. Adjust the CUDA path and architecture for another GPU:

```sh
export PATH="/usr/local/cuda-12.8/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}"
export NVCCFLAGS='-O2 -std=c++14 -arch=sm_75'
export OPENBLAS_NUM_THREADS=1
make -j4 all
make test
build/bench_cuda selftest --impl naive,tiled,optimized,cublas,cuda
python python/benchmark.py selftest --operation matrix --matrix-impl cublas
python python/benchmark.py selftest
```

The unit tests use explicitly marked Python stand-ins for native functions and do not replace GPU execution checks.

## Call the Library from Python

Start Python from the repository root with `PYTHONPATH=python python`, or add the `python/` directory to the import path:

```python
import numpy as np
from lab6_cuda import CudaLibrary

lib = CudaLibrary()
a = np.array([[1, 2], [3, 4]], dtype=np.float32)
b = np.array([[5, 6], [7, 8]], dtype=np.float32)
print(lib.matrix_multiply(a, b).output)
# [[19., 22.], [43., 50.]]

image = np.arange(9, dtype=np.uint32).reshape(3, 3)
kernel = np.ones((3, 3), dtype=np.float32) / np.float32(9)
print(lib.convolve(image, kernel).output)
```

The default library is `build/liblab6_cuda.so`; pass a path to `CudaLibrary(path)` or the CLI `--library` option for another build. Default matrix calls use `gpu_matrix_multiply`, backed by B's tiled kernel; convolution calls use `gpu_convolve`. Both are already exported by `src/bindings/exports.cu`. Timed calls use the public `lab6_*` interfaces to retrieve CUDA event timings. Invalid shapes, dtypes, memory layouts, output overlap, and nonzero native return codes raise errors.

## Image Application

```sh
python python/demo.py data/images/geometry.png --output results/python-demo-new
```

Use a fresh output directory. The application performs true convolution with a flipped filter, stride 1, zero padding, and an output of the same size. Input pixels are uint32 in [0, 255]; filters and outputs are float32. Signed outputs are preserved in `.npy` arrays. PNG rounding/clipping is for display only; edge PNGs show absolute magnitude.

## Collect and Plot Results

Reserve an idle measurement window on the shared machine. Use a new run ID every time:

```sh
python scripts/run_part_c.py --stage full --run-id C-NEW-full --context 'Shared AWS T4 host; agreed idle measurement window'
python scripts/plot_c_final.py --run-id C-NEW-full
```

The full collector tests six matrix implementations at N=256/512/1024/2048, including Python tiled. Convolution uses three images, M=512/1024/2048 and K=3/5/7, comparing CPU C, standalone CUDA, and Python CUDA. It also runs the image application and generates the standard summary plots automatically.

By default, shared `.u32` inputs are read from `/opt/lab6-a-reference/20261006-232110/inputs/`; use `--inputs /actual/path` if needed. Results are written beneath `results/raw/<run-id>/` and `results/summary/<run-id>/`; presentation figures go to `results/figures/<run-id>/`. Plotting requires results from the specified run. Historical raw results are not included in this C code upload.

Each configuration has one warm-up call and three measured calls. Correctness uses `atol=1e-3` and `rtol=1e-4` against independent float64 references. Means use all three measurements, and standard deviations are sample standard deviations. Speedup pairs a CPU baseline only with the same host, input, and timing scope. Failed, missing, duplicate, or nonfinite measurements are reported rather than filled in.

For a custom set of CSVs, provide an explicit comparison plan to the generic summarizer:

```sh
python scripts/summarize_results.py path/to/comparison-plan.json --output results/summary/new-comparison --require-complete
```

## Timing and Provenance

`compute` is CUDA-event device timing. Native `end_to_end` includes allocation, transfers, computation, synchronization, and cleanup. cuBLAS complete-call timing also includes per-call handle creation and destruction. Python end-to-end timing surrounds the ctypes call; input validation, NumPy output allocation, reference computation, and file writing are outside that interval.

The C implementation was validated on AWS T4 / CUDA 12.8 on 2026-10-13 using B's kernels at commit `961a8847d56831ba9d66a6b2fbb555ddd2e27d56`. The current main branch contains subsequent B tuning. Historical timing figures describe their recorded source snapshot; they are not new performance measurements of the current main branch. New full runs capture the actual source and input hashes for their own comparisons.
