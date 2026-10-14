#!/usr/bin/env python3
"""Part 7/8.3 correctness checks and Python CUDA measurements."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from lab6_cuda import CudaLibrary, Lab6Error

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/benchmark.json").read_text(encoding="utf-8"))


def matrix_inputs(n, seed=542):
    """Exactly match apps/harness.c: reset seed per N, fill A then B."""
    with np.errstate(over="ignore"):
        z = np.arange(1, 2 * n * n + 1, dtype=np.uint64)
        z = z * np.uint64(0x9E3779B97F4A7C15) + np.uint64(seed)
        z = (z ^ (z >> 30)) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> 27)) * np.uint64(0x94D049BB133111EB)
        z ^= z >> 31
    values = (z >> 40).astype(np.float32) * np.float32(1.0 / 16777216.0)
    return values[:n*n].reshape(n, n), values[n*n:].reshape(n, n)


def filter_weights(kind, k):
    if k <= 0 or k % 2 != 1:
        raise ValueError("filter size must be positive and odd")
    inv = np.float32(1) / np.float32(k * k)
    if kind == "mean":
        return np.full((k, k), inv, dtype=np.float32)
    if kind == "edge":
        result = np.full((k, k), -1, dtype=np.float32)
        result[k//2, k//2] = k*k - 1
    elif kind == "sharpen":
        result = np.full((k, k), -inv, dtype=np.float32)
        result[k//2, k//2] += np.float32(2)
    else:
        raise ValueError(f"unknown filter {kind}")
    return result


def convolution_reference(image, weights):
    # Float64 independent reference; each slice reverses filter offsets.
    m, k = image.shape[0], weights.shape[0]
    padded = np.pad(image.astype(np.float64), k // 2)
    result = np.zeros((m, m), dtype=np.float64)
    for i in range(k):
        for j in range(k):
            result += float(weights[i, j]) * padded[k-1-i:k-1-i+m, k-1-j:k-1-j+m]
    return result


def check(got, ref):
    if not np.isfinite(got).all() or not np.isfinite(ref).all():
        return False, float("inf")
    error = np.abs(got.astype(np.float64) - ref)
    limits = CONFIG["correctness"]
    return bool(np.all(error <= limits["atol"] + limits["rtol"] * np.abs(ref))), float(error.max())


def selftest(lib, operation, matrix_impl="tiled"):
    if operation in {"matrix", "all"}:
        for n in CONFIG["matrix"]["correctness_sizes"]:
            a, b = matrix_inputs(n)
            ref = a.astype(np.float64) @ b.astype(np.float64)
            for timed in (False, True):
                result = lib.matrix_multiply(a, b, timed=timed, implementation=matrix_impl)
                if not check(result.output, ref)[0]:
                    raise RuntimeError(f"matrix selftest failed: N={n}, timed={timed}")
        print(f"PASS Python matrix selftest ({matrix_impl})")
    if operation in {"convolution", "all"}:
        rng = np.random.default_rng(542)
        for m in (2, 3, 31):
            for k in (1, 3, 5):
                image = rng.integers(0, 256, (m, m), dtype=np.uint32)
                kernel = rng.uniform(-1, 1, (k, k)).astype(np.float32)
                ref = convolution_reference(image, kernel)
                for timed in (False, True):
                    result = lib.convolve(image, kernel, timed=timed)
                    if not check(result.output, ref)[0]:
                        raise RuntimeError(f"convolution selftest failed: M={m}, K={k}")
        print("PASS Python convolution selftest (asymmetric filters, signed output)")


def measure(call, ref, writer, metadata):
    output = np.full(ref.shape, np.nan, dtype=np.float32)
    # Correctness gate and warm-up precede the three recorded calls.
    for _ in range(CONFIG["measurement"]["warmup_runs"]):
        result = call(output)
        if not check(result.output, ref)[0]:
            raise RuntimeError(f"warm-up correctness failed: {metadata}")
    passed = True
    for run in range(1, CONFIG["measurement"]["measured_runs"] + 1):
        output.fill(np.nan)
        result = call(output)
        correct, error = check(result.output, ref)
        passed &= correct
        for scope, ms in (("compute", result.compute_ms),
                          ("end_to_end", result.python_end_to_end_ms)):
            if ms is None or not np.isfinite(ms) or ms < 0:
                raise RuntimeError("missing/invalid timing from shared library")
            writer.writerow(dict(metadata, run=run, timing_scope=scope, time_ms=f"{ms:.6f}",
                                 correct=str(correct).lower(), max_abs_error=f"{error:.9g}"))
    if not passed:
        raise RuntimeError("incorrect measured output; false rows retained for diagnosis")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("selftest", "matrix", "conv"))
    p.add_argument("--library", type=Path)
    p.add_argument("--operation", choices=("matrix", "convolution", "all"), default="all")
    p.add_argument("--matrix-impl", choices=("tiled", "naive", "cublas", "optimized"), default="tiled",
                   help="Selftest only; Part 7 measurements always use tiled")
    p.add_argument("--n", type=int, nargs="+", default=CONFIG["matrix"]["sizes"])
    p.add_argument("--k", type=int, nargs="+", default=CONFIG["convolution"]["filter_sizes"])
    p.add_argument("--filter", choices=("mean", "edge", "sharpen"), default="mean")
    p.add_argument("--image", type=Path, help="An existing square little-endian uint32 .u32 input")
    p.add_argument("--csv", type=Path)
    args = p.parse_args()
    if any(n <= 0 for n in args.n) or any(k <= 0 or k % 2 == 0 for k in args.k):
        p.error("sizes must be positive; filter sizes must be odd")
    lib = CudaLibrary(args.library)
    if args.command == "selftest":
        selftest(lib, args.operation, args.matrix_impl)
        return
    if args.csv is None:
        p.error("--csv is required for measurements")
    if args.command == "conv" and args.image is None:
        p.error("--image is required to preserve CPU/CUDA/Python input identity")
    selftest(lib, "matrix" if args.command == "matrix" else "convolution")
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("x", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=CONFIG["csv_columns"])
        writer.writeheader()
        base = dict(member="C", operation="matrix", implementation="python_tiled",
                    N="", M="", K="", filter="")
        if args.command == "matrix":
            for n in args.n:
                a, b = matrix_inputs(n, CONFIG["input_generation"]["seed"])
                ref = a.astype(np.float64) @ b.astype(np.float64)
                measure(lambda out: lib.matrix_multiply(a, b, out=out, timed=True),
                        ref, writer, dict(base, N=n))
                print(f"PASS Python tiled N={n}", flush=True)
        else:
            image = np.fromfile(args.image, dtype="<u4")
            m = math.isqrt(image.size)
            if not m or m*m != image.size or args.image.stat().st_size != image.size*4:
                raise ValueError("image file must contain a square uint32 array")
            image = image.reshape(m, m)
            for k in args.k:
                kernel = filter_weights(args.filter, k)
                ref = convolution_reference(image, kernel)
                measure(lambda out: lib.convolve(image, kernel, out=out, timed=True), ref,
                        writer, dict(base, operation="convolution", implementation="python_cuda",
                                     M=m, K=k, filter=args.filter))
                print(f"PASS Python CUDA M={m}, K={k}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (Lab6Error, RuntimeError, ValueError, OSError) as error:
        raise SystemExit(str(error)) from error
