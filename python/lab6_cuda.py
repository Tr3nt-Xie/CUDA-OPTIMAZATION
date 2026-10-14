"""Validated NumPy/ctypes interface for Parts 7 and 8.3 (Linux + CUDA).

Untimed tiled/convolution calls use gpu_matrix_multiply/gpu_convolve.
Timed calls use the existing public lab6_* ABI to obtain CUDA event timings.
No conversions, output allocation or validation are included in call timing.
"""

import ctypes as ct
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter_ns

import numpy as np


class Lab6Error(RuntimeError):
    def __init__(self, operation, status):
        self.status = int(status)
        descriptions = {1: "invalid argument", 2: "CUDA/cuBLAS runtime error",
                        3: "not implemented (required team implementation is missing)"}
        super().__init__(f"{operation}: {descriptions.get(self.status, 'unknown error')} "
                         f"[status={self.status}]")


class Timings(ct.Structure):
    _fields_ = [("compute_ms", ct.c_double), ("end_to_end_ms", ct.c_double)]


@dataclass
class Result:
    output: np.ndarray
    python_end_to_end_ms: float
    compute_ms: float | None = None
    native_end_to_end_ms: float | None = None


def square_array(value, dtype, name, *, writable=False):
    if not isinstance(value, np.ndarray) or value.dtype != np.dtype(dtype):
        raise TypeError(f"{name} must be a NumPy {np.dtype(dtype).name} array")
    if value.ndim != 2 or value.shape[0] != value.shape[1] or not value.shape[0]:
        raise ValueError(f"{name} must be a nonempty square 2D array")
    if value.shape[0] > np.iinfo(np.int32).max:
        raise ValueError(f"{name} dimension exceeds the C int ABI")
    if not value.flags.c_contiguous or not value.flags.aligned:
        raise ValueError(f"{name} must be aligned and C-contiguous")
    if writable and not value.flags.writeable:
        raise ValueError(f"{name} must be writable")
    return value


class CudaLibrary:
    def __init__(self, path=None):
        self.path = Path(path or Path(__file__).resolve().parents[1] /
                         "build/liblab6_cuda.so").resolve()
        self.lib = ct.CDLL(str(self.path))
        f32 = np.ctypeslib.ndpointer(dtype=np.float32, ndim=2,
                                    flags=("C_CONTIGUOUS", "ALIGNED"))
        u32 = np.ctypeslib.ndpointer(dtype=np.uint32, ndim=2,
                                    flags=("C_CONTIGUOUS", "ALIGNED"))
        timing = ct.POINTER(Timings)
        signatures = {
            "gpu_matrix_multiply": [f32, f32, f32, ct.c_int],
            "gpu_convolve": [u32, f32, f32, ct.c_int, ct.c_int],
        }
        for impl in ("naive", "tiled", "optimized", "cublas"):
            signatures[f"lab6_matmul_{impl}"] = [f32, f32, f32, ct.c_int, timing]
        signatures["lab6_convolve_cuda"] = [u32, f32, f32, ct.c_int, ct.c_int, timing]
        for name, args in signatures.items():
            fn = getattr(self.lib, name)
            fn.argtypes, fn.restype = args, ct.c_int

    @staticmethod
    def _output(shape, out, inputs):
        out = np.empty(shape, dtype=np.float32) if out is None else out
        square_array(out, np.float32, "out", writable=True)
        if out.shape != shape:
            raise ValueError("out has the wrong shape")
        if any(np.shares_memory(out, value) for value in inputs):
            raise ValueError("out must not overlap any input")
        return out

    def _call(self, name, args, out, timed):
        measured = Timings()
        if name.startswith("lab6_"):
            args = [*args, ct.byref(measured) if timed else None]
        fn = getattr(self.lib, name)
        start = perf_counter_ns()
        status = fn(*args)
        elapsed = (perf_counter_ns() - start) / 1e6
        if status:
            raise Lab6Error(name, status)
        return Result(out, elapsed, measured.compute_ms if timed else None,
                      measured.end_to_end_ms if timed else None)

    def matrix_multiply(self, a, b, *, out=None, timed=False, implementation="tiled"):
        square_array(a, np.float32, "a")
        square_array(b, np.float32, "b")
        if a.shape != b.shape:
            raise ValueError("a and b must have the same shape")
        if implementation not in {"naive", "tiled", "optimized", "cublas"}:
            raise ValueError("unknown matrix implementation")
        out = self._output(a.shape, out, (a, b))
        name = ("gpu_matrix_multiply" if implementation == "tiled" and not timed
                else f"lab6_matmul_{implementation}")
        return self._call(name, [a, b, out, a.shape[0]], out, timed)

    def convolve(self, image, kernel, *, out=None, timed=False):
        square_array(image, np.uint32, "image")
        square_array(kernel, np.float32, "kernel")
        if kernel.shape[0] % 2 != 1:
            raise ValueError("kernel size must be odd")
        if image.max() > 255:
            raise ValueError("image pixels must be in [0,255]")
        out = self._output(image.shape, out, (image, kernel))
        name = "lab6_convolve_cuda" if timed else "gpu_convolve"
        return self._call(name, [image, kernel, out, image.shape[0], kernel.shape[0]],
                          out, timed)
