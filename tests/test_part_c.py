"""CPU-only checks of C's Python contracts and data processing, not GPU tests."""

import ctypes as ct
import csv
import json
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "python"), str(ROOT / "scripts")]
from lab6_cuda import CudaLibrary, Lab6Error, Timings
from benchmark import matrix_inputs, convolution_reference, check, filter_weights, measure
from lab6_cuda import Result
from summarize_results import aggregate, CONFIG


class FakeFunction:
    """A Python stand-in to exercise routing/validation; never a CUDA result."""
    def __init__(self, name, owner):
        self.name, self.owner = name, owner

    def __call__(self, *args):
        self.owner.calls.append(self.name)
        if self.owner.status:
            return self.owner.status
        if "convolve" in self.name:
            image, weights, out = args[:3]
            out[:] = convolution_reference(image, weights)
        else:
            a, b, out = args[:3]
            out[:] = a @ b
        if self.name.startswith("lab6_") and args[-1] is not None:
            args[-1]._obj.compute_ms = 1.0
            args[-1]._obj.end_to_end_ms = 2.0
        return 0


class FakeLibrary:
    def __init__(self):
        self.calls, self.status = [], 0

    def __getattr__(self, name):
        fn = FakeFunction(name, self)
        setattr(self, name, fn)
        return fn


class WrapperTests(unittest.TestCase):
    def setUp(self):
        self.native = FakeLibrary()
        with patch("lab6_cuda.ct.CDLL", return_value=self.native):
            self.lib = CudaLibrary("test-only-library")
        self.a = np.array([[1, 2], [3, 4]], dtype=np.float32)
        self.b = np.array([[5, 6], [7, 8]], dtype=np.float32)

    def test_required_export_and_return_type(self):
        got = self.lib.matrix_multiply(self.a, self.b)
        np.testing.assert_array_equal(got.output, [[19, 22], [43, 50]])
        self.assertEqual(self.native.calls, ["gpu_matrix_multiply"])
        self.assertIs(self.native.gpu_matrix_multiply.restype, ct.c_int)
        self.assertEqual(ct.sizeof(Timings), 16)

    def test_timed_route(self):
        got = self.lib.matrix_multiply(self.a, self.b, timed=True)
        self.assertEqual(self.native.calls, ["lab6_matmul_tiled"])
        self.assertEqual((got.compute_ms, got.native_end_to_end_ms), (1, 2))

    def test_no_fallback_for_missing_b(self):
        self.native.status = 3
        with self.assertRaises(Lab6Error) as caught:
            self.lib.matrix_multiply(self.a, self.b)
        self.assertEqual(caught.exception.status, 3)
        self.assertEqual(self.native.calls, ["gpu_matrix_multiply"])

    def test_reject_dtype_shape_and_strides(self):
        for value in (self.a.astype(np.float64), self.a.T, self.a.ravel(), self.a[:1, :]):
            with self.subTest(shape=value.shape, dtype=value.dtype):
                with self.assertRaises((TypeError, ValueError)):
                    self.lib.matrix_multiply(value, self.b)
        self.assertFalse(self.native.calls)

    def test_reject_output_overlap_and_readonly(self):
        with self.assertRaises(ValueError):
            self.lib.matrix_multiply(self.a, self.b, out=self.a)
        out = np.empty_like(self.a)
        out.flags.writeable = False
        with self.assertRaises(ValueError):
            self.lib.matrix_multiply(self.a, self.b, out=out)

    def test_convolution_export_and_negative_output(self):
        image = np.arange(1, 10, dtype=np.uint32).reshape(3, 3)
        weights = np.zeros((3, 3), dtype=np.float32)
        weights[0, 0] = -1
        result = self.lib.convolve(image, weights)
        np.testing.assert_array_equal(result.output, [[-5, -6, 0], [-8, -9, 0], [0, 0, 0]])
        self.assertEqual(self.native.calls, ["gpu_convolve"])

    def test_convolution_reject_even_kernel_and_out_of_range(self):
        image = np.ones((3, 3), dtype=np.uint32)
        with self.assertRaises(ValueError):
            self.lib.convolve(image, np.ones((2, 2), dtype=np.float32))
        image[0, 0] = 256
        with self.assertRaises(ValueError):
            self.lib.convolve(image, np.ones((3, 3), dtype=np.float32))


class InputTests(unittest.TestCase):
    def test_splitmix_matches_independent_scalar_c_algorithm(self):
        mask = (1 << 64) - 1
        for n in (2, 3, 31):
            state, values = 542, []
            for _ in range(2*n*n):
                state = (state + 0x9E3779B97F4A7C15) & mask
                z = ((state ^ (state >> 30)) * 0xBF58476D1CE4E5B9) & mask
                z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
                z ^= z >> 31
                values.append((z >> 40) / 16777216.0)
            a, b = matrix_inputs(n)
            np.testing.assert_array_equal(np.concatenate((a.ravel(), b.ravel())), np.array(values, dtype=np.float32))

    def test_math_convolution_not_correlation(self):
        image = np.arange(1, 10, dtype=np.uint32).reshape(3, 3)
        weights = np.zeros((3, 3), dtype=np.float32)
        weights[0, 0] = 1
        np.testing.assert_array_equal(convolution_reference(image, weights), [[5, 6, 0], [8, 9, 0], [0, 0, 0]])

    def test_tolerance_and_nonfinite(self):
        self.assertTrue(check(np.array([1.001], dtype=np.float32), np.array([1.]))[0])
        self.assertFalse(check(np.array([1.1], dtype=np.float32), np.array([1.]))[0])
        self.assertFalse(check(np.array([np.nan]), np.array([1.]))[0])

    def test_filter_definitions(self):
        for k in (3, 5, 7):
            self.assertAlmostEqual(float(filter_weights("mean", k).sum()), 1, places=6)
            self.assertEqual(float(filter_weights("edge", k).sum()), 0)
            self.assertAlmostEqual(float(filter_weights("sharpen", k).sum()), 1, places=6)


class MeasurementTests(unittest.TestCase):
    def test_one_warmup_three_calls_two_scopes(self):
        calls = []
        ref = np.array([[7.]], dtype=np.float64)
        def call(out):
            calls.append(1)
            out[:] = ref
            return Result(out, 2.5, 1.0, 2.0)
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=CONFIG["csv_columns"])
        writer.writeheader()
        measure(call, ref, writer, dict(member="C", operation="matrix", implementation="python_tiled", N=1))
        rows = list(csv.DictReader(io.StringIO(stream.getvalue())))
        self.assertEqual(len(calls), 4)
        self.assertEqual(len(rows), 6)
        self.assertEqual([r["run"] for r in rows], ["1", "1", "2", "2", "3", "3"])
        self.assertEqual(float(rows[1]["time_ms"]), 2.5)

    def test_failed_warmup_writes_no_measurements(self):
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=CONFIG["csv_columns"])
        def wrong(out):
            out[:] = 0
            return Result(out, 2.5, 1.0, 2.0)
        with self.assertRaises(RuntimeError):
            measure(wrong, np.ones((1, 1)), writer, {})
        self.assertEqual(stream.getvalue(), "")


class SummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)

    def data(self, name, impl="cpu", scope="compute", times=(9, 10, 11), correct="true", input_id="shared", context="host"):
        path = self.base / name
        with path.open("w", newline="") as fp:
            writer = csv.DictWriter(fp, fieldnames=CONFIG["csv_columns"])
            writer.writeheader()
            for run, time in enumerate(times, 1):
                writer.writerow(dict(member="C", operation="matrix", implementation=impl,
                                     N=256, M="", K="", filter="", timing_scope=scope,
                                     run=run, time_ms=time, correct=correct, max_abs_error=0))
        dataset = dict(csv=name, input_id=input_id, context=context)
        expected = dict(dataset, operation="matrix", implementation=impl, N=256, timing_scope=scope)
        return dataset, expected

    def summarize(self, pairs):
        return aggregate(dict(datasets=[p[0] for p in pairs], expected=[p[1] for p in pairs]), self.base)

    def test_mean_std_and_same_scope_speedup(self):
        rows, coverage = self.summarize([self.data("cpu.csv", times=(90, 100, 110)), self.data("gpu.csv", impl="cublas")])
        gpu = next(r for r in rows if r["implementation"] == "cublas")
        self.assertEqual((gpu["mean_ms"], gpu["std_ms"], gpu["cpu_speedup"]), (10, 1, 10))
        self.assertEqual(coverage["status"], "complete")

    def test_never_pair_different_scope_input_or_host(self):
        for overrides in ({"scope": "end_to_end"}, {"input_id": "different"}, {"context": "other-host"}):
            rows, _ = self.summarize([self.data("cpu.csv"), self.data("gpu.csv", impl="cublas", **overrides)])
            self.assertEqual(next(r for r in rows if r["implementation"] == "cublas")["cpu_speedup"], "")

    def test_exclude_failed_nonfinite_or_incomplete_group(self):
        for kwargs in ({"correct": "false"}, {"times": (1, 2)}, {"times": (1, float("nan"), 3)}):
            rows, coverage = self.summarize([self.data("bad.csv", **kwargs)])
            self.assertFalse(rows)
            self.assertEqual((len(coverage["rejected"]), len(coverage["missing"])), (1, 1))

    def test_duplicate_dataset_does_not_double_sample_size(self):
        pair = self.data("cpu.csv")
        rows, coverage = self.summarize([pair, pair])
        self.assertFalse(rows)
        self.assertIn("duplicate", coverage["rejected"][0]["reason"])

    def test_missing_implementation_is_explicit(self):
        dataset, expected = self.data("cpu.csv")
        plan = dict(datasets=[dataset], expected=[expected, dict(expected, implementation="cublas")])
        rows, coverage = aggregate(plan, self.base)
        self.assertEqual(len(rows), 1)
        self.assertEqual(coverage["status"], "partial")
        self.assertEqual(coverage["missing"][0]["implementation"], "cublas")


if __name__ == "__main__":
    unittest.main()
