#!/usr/bin/env python3
"""Build and measure C's work sequentially on the shared Linux CUDA host.

cublas stage: independently runnable CPU/naive/cuBLAS matrix comparison.
full stage: all matrix implementations, Python tiled, and CPU/CUDA/Python
convolution on all three shared images and all nine M/K combinations.
Only run after reserving an idle measurement window with the team.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/benchmark.json").read_text(encoding="utf-8"))


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=("cublas", "full"), default="cublas")
    p.add_argument("--run-id", required=True, help="A new ID, e.g. C-20261013-cublas")
    p.add_argument("--context", required=True, help="Actual host identity and agreed measurement window")
    p.add_argument("--inputs", type=Path, default=Path("/opt/lab6-a-reference/20261006-232110/inputs"))
    args = p.parse_args()
    if platform.system() != "Linux":
        p.error("run this collector on the Linux CUDA host")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        p.error("run ID must contain only letters, digits, underscores and hyphens")
    raw = ROOT / "results/raw" / args.run_id
    raw.mkdir(parents=True, exist_ok=False)
    build = ROOT / "build" / args.run_id
    manifest = {"run_id": args.run_id, "stage": args.stage, "started_at": now(),
                "status": "running", "context": args.context, "config": CONFIG,
                "datasets": [], "expected": [], "sources": {}}

    def save():
        (raw / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        plan = {name: manifest[name] for name in ("datasets", "expected")}
        (raw / "comparison-plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    def run(command, label):
        command = [str(x) for x in command]
        print(f"[{now()}] {label}", flush=True)
        result = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (raw / f"{label}.log").write_text("$ " + shlex.join(command) + "\n" + result.stdout +
                                        f"\nexit_code={result.returncode}\n", encoding="utf-8")
        if result.returncode:
            raise RuntimeError(f"{label} failed; inspect {raw / (label + '.log')}")
        return result.stdout

    def expect(csv_name, operation, impl, input_id, dimensions):
        manifest["datasets"].append(dict(csv=csv_name, context=args.context, input_id=input_id))
        for dimension in dimensions:
            for scope in CONFIG["measurement"]["scopes"]:
                item = dict(context=args.context, input_id=input_id, operation=operation,
                            implementation=impl, N="", M="", K="", filter="", timing_scope=scope)
                item.update(dimension)
                manifest["expected"].append(item)
        save()

    try:
        paths = [ROOT / "Makefile", ROOT / "configs/benchmark.json"]
        for folder in ("include", "src", "apps", "python", "scripts"):
            paths.extend(p for p in (ROOT / folder).rglob("*") if p.suffix in {".c", ".cu", ".h", ".cuh", ".py", ".sh", ".txt"})
        manifest["sources"] = {p.relative_to(ROOT).as_posix(): sha256(p) for p in paths}
        save()
        with tarfile.open(raw / "source-snapshot.tar.gz", "w:gz") as archive:
            for name in manifest["sources"]:
                archive.add(ROOT / name, arcname=name)
        probe = run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader"], "gpu-idle-check")
        if probe.strip():
            raise RuntimeError("GPU has active compute processes; coordinate another measurement window")
        environment = run(["bash", "scripts/collect_env.sh"], "environment")
        (raw / "environment.txt").write_text(environment, encoding="utf-8")
        flags = os.environ.get("NVCCFLAGS", "-O2 -std=c++14 -arch=sm_75")
        manifest["nvcc_flags"] = flags
        run(["make", "-j4", f"BUILD={build}", f"NVCCFLAGS={flags}", "all"], "build")
        manifest["binary_sha256"] = {name: sha256(build / name) for name in ("bench_cpu", "bench_cuda", "liblab6_cuda.so")}
        run([build / "bench_cpu", "selftest"], "cpu-selftest")
        implementations = ["naive", "cublas"] if args.stage == "cublas" else ["naive", "tiled", "optimized", "cublas", "cuda"]
        run([build / "bench_cuda", "selftest", "--impl", ",".join(implementations)], "cuda-selftest")
        python_base = [sys.executable, ROOT / "python/benchmark.py"]
        library = ["--library", build / "liblab6_cuda.so"]
        run([*python_base, "selftest", *library, "--operation", "matrix", "--matrix-impl", "cublas"], "python-cublas-selftest")
        if args.stage == "full":
            run([*python_base, "selftest", *library], "python-selftest")
        measurement = CONFIG["measurement"]
        common = ["--member", "C", "--warmup", str(measurement["warmup_runs"]), "--runs", str(measurement["measured_runs"]),
                  "--seed", str(CONFIG["input_generation"]["seed"]), "--atol", str(CONFIG["correctness"]["atol"]), "--rtol", str(CONFIG["correctness"]["rtol"])]
        sizes = CONFIG["matrix"]["sizes"]
        for impl in ["cpu", *[i for i in implementations if i != "cuda"], *(["python_tiled"] if args.stage == "full" else [])]:
            csv_name = f"matrix-{impl}.csv"
            expect(csv_name, "matrix", impl, f"harness-splitmix64-seed-{CONFIG['input_generation']['seed']}", [{"N": n} for n in sizes])
            if impl == "python_tiled":
                command = [*python_base, "matrix", *library, "--n", *map(str, sizes), "--csv", raw / csv_name]
            else:
                exe = build / ("bench_cpu" if impl == "cpu" else "bench_cuda")
                command = [exe, "matrix", "--impl", impl, "--n", ",".join(map(str, sizes)), *common, "--csv", raw / csv_name]
            run(command, f"matrix-{impl}")
        if args.stage == "full":
            for name in ("geometry", "frequency", "soft_scene"):
                for m in CONFIG["convolution"]["image_sizes"]:
                    image = args.inputs / f"{name}_{m}.u32"
                    if not image.is_file():
                        raise RuntimeError(f"missing shared input: {image}; pass --inputs with the correct directory")
                    kernels = CONFIG["convolution"]["filter_sizes"]
                    input_hash = sha256(image)
                    identity = f"{name}:sha256:{input_hash}"
                    for impl in ("cpu", "cuda", "python_cuda"):
                        label = f"{name}-M{m}-{impl}"
                        expect(label + ".csv", "convolution", impl, identity,
                               [dict(M=m, K=k, filter="mean") for k in kernels])
                        if impl == "python_cuda":
                            command = [*python_base, "conv", *library, "--image", image, "--k", *map(str, kernels), "--csv", raw / (label + ".csv")]
                        else:
                            exe = build / ("bench_cpu" if impl == "cpu" else "bench_cuda")
                            command = [exe, "conv", "--impl", impl, "--image", image, "--k", ",".join(map(str, kernels)), *common, "--csv", raw / (label + ".csv")]
                        run(command, label)
                        if sha256(image) != input_hash:
                            raise RuntimeError(f"input changed during measurement: {image}")
            run([sys.executable, ROOT / "python/demo.py", ROOT / "data/images/geometry.png",
                 *library, "--output", raw / "python-filter-demo"], "python-filter-demo")
        if any(sha256(ROOT / name) != value for name, value in manifest["sources"].items()):
            raise RuntimeError("source files changed during measurement")
        run([sys.executable, ROOT / "scripts/summarize_results.py", raw / "comparison-plan.json",
             "--output", ROOT / "results/summary" / args.run_id, "--require-complete"], "summary")
        manifest.update(status="passed", ended_at=now())
        print(f"PASS stage={args.stage}; full Part 7/8.3 requires stage=full", flush=True)
    except Exception as error:
        manifest.update(status="failed", ended_at=now(), error=str(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
