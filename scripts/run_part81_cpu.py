#!/usr/bin/env python3
"""Run A's Part 8.1 CPU image checks and preserve local measurement evidence.

This runner does not deploy resources or benchmark a GPU. It deliberately keeps
each image's measurements in a separate standard-format CSV.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import re
import shlex
import statistics
import subprocess
import sys
import tarfile
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
TZ = ZoneInfo("America/Los_Angeles")


def now():
    return datetime.now(TZ).isoformat(timespec="seconds")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2) + "\n")


def command(args, raw, label, required=True):
    args = [str(a) for a in args]
    start = now()
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    result = subprocess.run(args, cwd=ROOT, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    log = raw / f"{label}.log"
    log.write_text(f"$ {shlex.join(args)}\n{result.stdout}\nexit_code={result.returncode}\n")
    record = {"started_at": start, "ended_at": now(), "argv": args,
              "cwd": str(ROOT), "exit_code": result.returncode, "log": relative(log)}
    with (raw / "commands.jsonl").open("a") as fp:
        fp.write(json.dumps(record) + "\n")
    if result.returncode and required:
        raise RuntimeError(f"{label} failed; see {relative(log)}")
    return result.stdout + (f"\n[exit_code={result.returncode}; optional probe unavailable]\n"
                            if result.returncode else "")


def source_hashes():
    paths = [ROOT / "Makefile", ROOT / "configs/benchmark.json",
             ROOT / "src/convolution/cpu.c", ROOT / "src/matrix/cpu.c"]
    for folder, suffixes in (("include", {".h"}), ("apps", {".c", ".h"}),
                             ("src/common", {".h"}), ("scripts", {".py", ".sh", ".txt"})):
        paths += [p for p in (ROOT / folder).iterdir() if p.suffix in suffixes]
    return {relative(p): digest(p) for p in sorted(paths)}


def weights(kind, k):
    inv = np.float32(1.0) / np.float32(k * k)
    if kind == "mean":
        return np.full((k, k), inv, dtype=np.float32)
    if kind == "edge":
        result = np.full((k, k), -1, dtype=np.float32)
        result[k // 2, k // 2] = k * k - 1
        return result
    if kind == "sharpen":
        result = np.full((k, k), -inv, dtype=np.float32)
        result[k // 2, k // 2] += np.float32(2)
        return result
    raise ValueError(kind)


def check_output(image_path, output_path, kind, m, k, config):
    image = np.fromfile(image_path, dtype="<u4").reshape(m, m)
    got = np.fromfile(output_path, dtype="<f4").reshape(m, m)
    window = np.lib.stride_tricks.sliding_window_view(
        np.pad(image.astype(np.float64), k // 2), (k, k))
    ref = np.einsum("ijkl,kl->ij", window,
                    weights(kind, k)[::-1, ::-1].astype(np.float64))
    error = np.abs(got.astype(np.float64) - ref)
    limits = config["correctness"]
    passed = bool(np.isfinite(got).all() and
                  np.all(error <= limits["atol"] + limits["rtol"] * np.abs(ref)))
    return {"output": relative(output_path), "filter": kind, "M": m, "K": k,
            "reference": "NumPy float64 padded sliding windows with reversed filter",
            "correct": passed, "max_abs_error": float(error.max()),
            "sha256": digest(output_path), "min": float(got.min()), "max": float(got.max())}


def summarize(cases, target, runs):
    rows = []
    for case in cases:
        records = list(csv.DictReader((ROOT / case["csv"]).open()))
        if len(records) != 2 * runs or any(r["correct"] != "true" for r in records):
            raise RuntimeError(f"Incomplete or incorrect measurements: {case['csv']}")
        for scope in ("compute", "end_to_end"):
            sample = [r for r in records if r["timing_scope"] == scope]
            if sorted(int(r["run"]) for r in sample) != list(range(1, runs + 1)):
                raise RuntimeError(f"Bad run IDs: {case['csv']}")
            values = [float(r["time_ms"]) for r in sample]
            if any(not math.isfinite(v) or v < 0 for v in values):
                raise RuntimeError(f"Invalid timing: {case['csv']}")
            rows.append({"image": case["image"], "M": case["M"], "K": case["K"],
                         "filter": case["filter"], "timing_scope": scope,
                         "runs": runs, "mean_ms": statistics.mean(values),
                         "std_ms": statistics.stdev(values) if runs > 1 else 0,
                         "min_ms": min(values), "max_ms": max(values),
                         "max_abs_error": max(float(r["max_abs_error"]) for r in sample),
                         "input_sha256": case["input_sha256"], "source_csv": case["csv"]})
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def make_figures(images, outputs, folder, m, k):
    folder.mkdir(parents=True, exist_ok=True)
    title_font = ImageFont.load_default(size=25)
    font = ImageFont.load_default(size=19)
    tile, left, gap, top, row_gap = 384, 40, 18, 120, 52
    width = left * 2 + tile * 4 + gap * 3
    height = top + len(images) * (tile + row_gap) + 52
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((left, 20), f"Part 8.1 | CPU convolution | M = {m}, K = {k}",
              fill="#152333", font=title_font)
    labels = ("Original", "Mean blur", "Edge detection | absolute", "Sharpen")
    for col, label in enumerate(labels):
        draw.text((left + col * (tile + gap), 66), label, fill="#152333", font=font)
    for row, image in enumerate(images):
        name = image["name"]
        y = top + row * (tile + row_gap)
        draw.text((left, y - 25), image["title"], fill="#334155", font=font)
        original = np.fromfile(ROOT / f"data/generated/{name}_{m}.u32", dtype="<u4").reshape(m, m)
        arrays = [original]
        for kind in ("mean", "edge", "sharpen"):
            data = np.fromfile(outputs[(name, kind)], dtype="<f4").reshape(m, m)
            arrays.append(np.abs(data) if kind == "edge" else data)
        for col, data in enumerate(arrays):
            pixels = np.clip(np.rint(data), 0, 255).astype(np.uint8)
            picture = Image.fromarray(pixels)
            picture.save(folder / f"{name}_{('original', 'mean', 'edge', 'sharpen')[col]}.png")
            sheet.paste(picture.resize((tile, tile), Image.Resampling.LANCZOS),
                        (left + col * (tile + gap), y))
    draw.text((left, height - 48), "Original procedural inputs. Zero padding, stride 1, true convolution.",
              fill="#475569", font=font)
    draw.text((left, height - 25), "Display: round and clip to [0,255]; edge uses absolute value. Validation uses unclipped float32 outputs.",
              fill="#475569", font=font)
    sheet.save(folder / "filter-comparison.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="A-" + datetime.now(TZ).strftime("%Y%m%d-%H%M%S") + "-part81-cpu-local")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("run-id must contain only letters, digits, underscores and hyphens")
    raw = ROOT / "results/raw" / args.run_id
    raw.mkdir(parents=True, exist_ok=False)
    output_dir = ROOT / "data/generated" / args.run_id
    output_dir.mkdir(parents=True, exist_ok=False)
    figures = ROOT / "results/figures" / args.run_id
    config = json.loads((ROOT / "configs/benchmark.json").read_text())
    manifest = {"run_id": args.run_id, "started_at": now(), "status": "running",
                "measurement_context": "Local CPU development measurements; rerun CPU and CUDA on the same GPU host for speedups.",
                "platform": platform.platform(), "python": sys.version,
                "python_executable": sys.executable, "numpy": np.__version__,
                "pillow": PIL.__version__, "config": config,
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
                "source_sha256": source_hashes(), "cases": [], "independent_checks": []}
    write_json(raw / "manifest.json", manifest)
    print(f"RUN_ID={args.run_id}", flush=True)
    try:
        with tarfile.open(raw / "source-snapshot.tar.gz", "w:gz") as archive:
            for path in manifest["source_sha256"]:
                archive.add(ROOT / path, arcname=path)
        image_manifest_path = ROOT / "data/images/manifest.json"
        if not image_manifest_path.exists():
            command([sys.executable, "scripts/generate_test_images.py"], raw, "generate-images")
        image_manifest = json.loads(image_manifest_path.read_text())
        for image in image_manifest["images"]:
            if digest(ROOT / image["path"]) != image["sha256"]:
                raise RuntimeError(f"Input differs from the image manifest: {image['path']}")
        command([sys.executable, "scripts/prepare_images.py", "to-raw"], raw, "prepare-images")
        manifest["images"] = image_manifest
        build = Path("build") / args.run_id
        command(["make", f"BUILD={build}", "cpu"], raw, "build")
        exe = ROOT / build / "bench_cpu"
        command([exe, "selftest"], raw, "selftest")
        environment = command(["bash", "scripts/collect_env.sh"], raw, "collect-environment")
        if sys.platform == "darwin":
            environment += "\n" + command(["sysctl", "machdep.cpu.brand_string", "hw.memsize", "hw.ncpu"], raw, "hardware", required=False)
        (raw / "environment.txt").write_text(environment)
        dims = config["convolution"]["image_sizes"]
        kernels = config["convolution"]["filter_sizes"]
        dm, dk = min(dims), min(kernels)
        outputs = {}
        benchmark_filter = config["convolution"]["benchmark_filter"]
        showcase = config["convolution"]["showcase_filters"]
        for image in image_manifest["images"]:
            name = image["name"]
            combinations = [(m, k, benchmark_filter) for m in dims for k in kernels]
            combinations += [(dm, dk, kind) for kind in showcase]
            for m, k, kind in combinations:
                label = f"{name}-M{m}-K{k}-{kind}"
                image_path = ROOT / f"data/generated/{name}_{m}.u32"
                csv_path = raw / f"{label}.csv"
                cmd = [exe, "conv", "--image", image_path, "--k", str(k),
                       "--filter", kind, "--member", "A", "--csv", csv_path,
                       "--warmup", str(config["measurement"]["warmup_runs"]),
                       "--runs", str(config["measurement"]["measured_runs"]),
                       "--atol", str(config["correctness"]["atol"]),
                       "--rtol", str(config["correctness"]["rtol"])]
                save = m == dm and k == dk
                output_path = output_dir / f"{label}.f32"
                if save:
                    cmd += ["--save", output_path]
                print(f"Checking {label}", flush=True)
                command(cmd, raw, label)
                manifest["cases"].append({"image": name, "M": m, "K": k,
                                          "filter": kind, "csv": relative(csv_path),
                                          "input": relative(image_path),
                                          "input_sha256": digest(image_path)})
                if save:
                    check = check_output(image_path, output_path, kind, m, k, config)
                    manifest["independent_checks"].append(check)
                    if not check["correct"]:
                        raise RuntimeError(f"Independent NumPy check failed: {label}")
                    outputs[(name, kind)] = output_path
                write_json(raw / "manifest.json", manifest)
        summary_path = ROOT / "results/summary" / f"{args.run_id}.csv"
        summarize(manifest["cases"], summary_path, config["measurement"]["measured_runs"])
        make_figures(image_manifest["images"], outputs, figures, dm, dk)
        if source_hashes() != manifest["source_sha256"]:
            raise RuntimeError("Source changed during the run; do not treat these measurements as one version")
        manifest.update(status="passed", ended_at=now(),
                        case_count=len(manifest["cases"]),
                        measured_calls=len(manifest["cases"]) * config["measurement"]["measured_runs"],
                        summary=relative(summary_path), figures=relative(figures))
        print(f"PASS: {manifest['case_count']} configurations, {manifest['measured_calls']} measured calls, "
              f"{len(manifest['independent_checks'])} independent image checks", flush=True)
        print(f"Figures: {relative(figures)}", flush=True)
    except Exception as error:
        manifest.update(status="failed", ended_at=now(), error=str(error))
        raise
    finally:
        write_json(raw / "manifest.json", manifest)


if __name__ == "__main__":
    main()
