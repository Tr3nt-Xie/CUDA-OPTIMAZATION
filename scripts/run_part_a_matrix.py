#!/usr/bin/env python3
"""Collect A's CPU/naive matrix baselines on one CUDA host, sequentially.

The optional small-size sweep supports Q2 without changing the team's formal
sizes. Run when the shared host is idle; no cloud resources are created here.
"""

import argparse
import csv
import json
import math
import os
import re
import statistics
import subprocess
import tarfile
from collections import defaultdict
from datetime import datetime

from run_part81_cpu import ROOT, TZ, command, digest, now, relative, write_json


def sources():
    paths = [ROOT / "Makefile", ROOT / "configs/benchmark.json"]
    for folder in ("include", "apps", "src", "scripts"):
        paths += [p for p in (ROOT / folder).rglob("*")
                  if p.suffix in {".c", ".cu", ".h", ".cuh", ".py", ".sh", ".txt"}]
    return {relative(p): digest(p) for p in sorted(paths)}


def summarize(cases, target, runs):
    summary = []
    for case in cases:
        with (ROOT / case["csv"]).open() as fp:
            records = list(csv.DictReader(fp))
        expected = {(n, scope) for n in case["sizes"] for scope in ("compute", "end_to_end")}
        groups = defaultdict(list)
        for row in records:
            if (row["correct"] != "true" or row["implementation"] != case["implementation"]
                    or row["operation"] != "matrix" or row["member"] != "A"):
                raise RuntimeError(f"Wrong or incorrect result: {case['csv']}")
            groups[(int(row["N"]), row["timing_scope"])].append(row)
        if set(groups) != expected:
            raise RuntimeError(f"Incomplete dimensions/scopes: {case['csv']}")
        for (n, scope), rows in sorted(groups.items()):
            if sorted(int(r["run"]) for r in rows) != list(range(1, runs + 1)):
                raise RuntimeError(f"Missing/duplicate runs: {case['csv']}")
            values = [float(r["time_ms"]) for r in rows]
            if any(not math.isfinite(v) or v <= 0 for v in values):
                raise RuntimeError(f"Invalid timing: {case['csv']}")
            summary.append({"series": case["series"], "implementation": case["implementation"],
                            "N": n, "timing_scope": scope, "runs": runs,
                            "mean_ms": statistics.mean(values), "std_ms": statistics.stdev(values),
                            "min_ms": min(values), "max_ms": max(values),
                            "max_abs_error": max(float(r["max_abs_error"]) for r in rows),
                            "source_csv": case["csv"]})
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="A-" + datetime.now(TZ).strftime("%Y%m%d-%H%M%S") + "-matrix")
    parser.add_argument("--context", required=True, help="Actual measurement host and purpose")
    parser.add_argument("--small-sizes", default="16,32,64,128", help="Supplementary Q2 sweep; empty disables")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_id):
        parser.error("Invalid run ID")
    try:
        small = [int(v) for v in args.small_sizes.split(",") if v]
    except ValueError:
        parser.error("small-sizes must be comma-separated positive integers")
    if any(n <= 0 for n in small) or len(set(small)) != len(small):
        parser.error("small-sizes must be unique positive integers")
    config = json.loads((ROOT / "configs/benchmark.json").read_text())
    formal = config["matrix"]["sizes"]
    if set(formal) & set(small):
        parser.error("Supplementary sizes must not duplicate formal sizes")
    raw = ROOT / "results/raw" / args.run_id
    raw.mkdir(parents=True, exist_ok=False)
    manifest = {"run_id": args.run_id, "started_at": now(), "status": "running",
                "measurement_context": args.context, "config": config,
                "supplementary_sizes": small, "formal_sizes": formal,
                "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
                "source_sha256": sources(), "cases": [],
                "input_identity": "apps/harness.c SplitMix64; reset seed for each N; fill A then B; float32 in [0,1).",
                "notes": "CPU compute=end_to_end (loop). GPU event compute and host wrapper end_to_end. Validation excluded. No outlier removal."}
    write_json(raw / "manifest.json", manifest)
    print(f"RUN_ID={args.run_id}", flush=True)
    try:
        with tarfile.open(raw / "source-snapshot.tar.gz", "w:gz") as archive:
            for path in manifest["source_sha256"]:
                archive.add(ROOT / path, arcname=path)
        (raw / "tracked-changes.patch").write_text(command(["git", "diff", "HEAD", "--", "Makefile", "configs", "include", "apps", "src", "scripts"], raw, "git-diff"))
        build = "build/" + args.run_id
        flags = os.environ.get("NVCCFLAGS", "-O2 -std=c++14")
        command(["make", "-j4", f"BUILD={build}", f"NVCCFLAGS={flags}", "all"], raw, "build")
        manifest["binary_sha256"] = {f"{build}/{name}": digest(ROOT / build / name)
                                     for name in ("bench_cpu", "bench_cuda", "liblab6_cuda.so")}
        command([ROOT / build / "bench_cpu", "selftest"], raw, "cpu-selftest")
        command([ROOT / build / "bench_cuda", "selftest", "--impl", "naive"], raw, "naive-selftest")
        (raw / "environment.txt").write_text(command(["bash", "scripts/collect_env.sh"], raw, "collect-environment"))
        command(["ps", "-eo", "user,pid,pcpu,pmem,args"], raw, "processes-before")
        command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"], raw, "gpu-processes-before")
        series = [("formal", formal)] + ([("supplementary", small)] if small else [])
        for series_name, sizes in series:
            for impl in ("cpu", "naive"):
                label = f"{series_name}-{impl}"
                csv_path = raw / f"{label}.csv"
                exe = ROOT / build / ("bench_cpu" if impl == "cpu" else "bench_cuda")
                print(f"Measuring {label}, N={sizes}", flush=True)
                command([exe, "matrix", "--impl", impl, "--n", ",".join(map(str, sizes)),
                         "--member", "A", "--csv", csv_path,
                         "--warmup", str(config["measurement"]["warmup_runs"]),
                         "--runs", str(config["measurement"]["measured_runs"]),
                         "--seed", str(config["input_generation"]["seed"]),
                         "--atol", str(config["correctness"]["atol"]),
                         "--rtol", str(config["correctness"]["rtol"])], raw, label)
                manifest["cases"].append({"series": series_name, "implementation": impl,
                                          "sizes": sizes, "csv": relative(csv_path)})
                write_json(raw / "manifest.json", manifest)
        target = ROOT / "results/summary" / f"{args.run_id}.csv"
        rows = summarize(manifest["cases"], target, config["measurement"]["measured_runs"])
        command(["nvidia-smi"], raw, "gpu-after")
        if sources() != manifest["source_sha256"]:
            raise RuntimeError("Source changed during measurement")
        manifest.update(status="passed", ended_at=now(), summary=relative(target),
                        measured_calls=len(rows) // 2 * config["measurement"]["measured_runs"])
        print(f"PASS: {manifest['measured_calls']} measured matrix calls", flush=True)
    except Exception as error:
        manifest.update(status="failed", ended_at=now(), error=str(error))
        raise
    finally:
        write_json(raw / "manifest.json", manifest)


if __name__ == "__main__":
    main()
