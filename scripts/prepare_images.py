#!/usr/bin/env python3
"""Convert convolution test images between PNG and the raw files bench_* uses.

  to-raw  every image in data/images -> data/generated/<stem>_<M>.u32
          (grayscale, center-cropped to a square, resized to each M)
  to-png  a .u32 image or the float32 output of `bench_* conv --save` -> PNG
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "configs" / "benchmark.json"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def to_raw(args):
    config = json.loads(CONFIG.read_text())
    src = ROOT / config["input_generation"]["images_dir"]
    dst = ROOT / config["input_generation"]["generated_dir"]
    sizes = args.sizes or config["convolution"]["image_sizes"]
    images = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        sys.exit(f"no images found in {src}")

    dst.mkdir(parents=True, exist_ok=True)
    for path in images:
        with Image.open(path) as img:
            gray = img.convert("L")
        side = min(gray.size)
        left = (gray.width - side) // 2
        top = (gray.height - side) // 2
        square = gray.crop((left, top, left + side, top + side))
        for m in sizes:
            pixels = np.asarray(square.resize((m, m), Image.LANCZOS), dtype=np.uint8)
            out = dst / f"{path.stem}_{m}.u32"
            pixels.astype("<u4").tofile(out)
            print(out.relative_to(ROOT))


def to_png(args):
    dtype = "<u4" if args.input.suffix == ".u32" else "<f4"
    data = np.fromfile(args.input, dtype=dtype).astype(np.float64)
    m = math.isqrt(data.size)
    if data.size == 0 or m * m != data.size:
        sys.exit(f"{args.input}: not a square image")
    img = data.reshape(m, m)
    if args.mode == "abs":
        img = np.abs(img)
    elif args.mode == "normalize":
        lo, hi = img.min(), img.max()
        img = (img - lo) * (255.0 / (hi - lo)) if hi > lo else np.zeros_like(img)
    pixels = np.clip(np.rint(img), 0, 255).astype(np.uint8)
    Image.fromarray(pixels).save(args.output)
    print(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    raw = sub.add_parser("to-raw", help="data/images -> data/generated/*.u32")
    raw.add_argument("--sizes", type=int, nargs="+",
                     help="image sizes (default: convolution.image_sizes in config)")
    raw.set_defaults(func=to_raw)

    png = sub.add_parser("to-png", help="raw image or filter output -> PNG")
    png.add_argument("input", type=Path)
    png.add_argument("output", type=Path)
    png.add_argument("--mode", choices=["clip", "abs", "normalize"], default="clip",
                     help="display mapping; edge outputs usually look best with abs")
    png.set_defaults(func=to_png)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
