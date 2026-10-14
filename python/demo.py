#!/usr/bin/env python3
"""Apply Part 8.3 CUDA convolution to a square grayscale image."""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from benchmark import check, convolution_reference, filter_weights
from lab6_cuda import CudaLibrary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("image", type=Path)
    p.add_argument("--output", type=Path, required=True, help="New directory for float32 arrays and PNGs")
    p.add_argument("--library", type=Path)
    args = p.parse_args()
    with Image.open(args.image) as source:
        image = np.array(source.convert("L"), dtype=np.uint32)
    if image.shape[0] != image.shape[1]:
        p.error("use a square input image")
    lib = CudaLibrary(args.library)
    args.output.mkdir(parents=True, exist_ok=False)
    for kind in ("mean", "edge", "sharpen"):
        kernel = filter_weights(kind, 3)
        result = lib.convolve(image, kernel)  # Uses the required gpu_convolve export.
        passed, error = check(result.output, convolution_reference(image, kernel))
        if not passed:
            raise RuntimeError(f"{kind} failed verification: max_abs_error={error}")
        np.save(args.output / f"{kind}.npy", result.output)  # Preserve signed values.
        display = np.abs(result.output) if kind == "edge" else result.output
        Image.fromarray(np.clip(np.rint(display), 0, 255).astype(np.uint8)).save(
            args.output / f"{kind}.png")
        print(f"{kind}: PASS, max_abs_error={error:.6g}; display-only clipping")


if __name__ == "__main__":
    main()
