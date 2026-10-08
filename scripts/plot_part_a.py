#!/usr/bin/env python3
"""Create A's CPU/naive figures from one completed matrix measurement batch."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, ScalarFormatter

ROOT = Path(__file__).resolve().parent.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_id")
    args = parser.parse_args()
    if Path(args.run_id).name != args.run_id or args.run_id in {".", ".."}:
        parser.error("Use a run ID, not a path")
    manifest_path = ROOT / "results/raw" / args.run_id / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["status"] != "passed":
        raise ValueError("Only a complete, passed batch may be plotted")
    summary = ROOT / manifest["summary"]
    with summary.open() as fp:
        records = list(csv.DictReader(fp))
    lookup = {(r["series"], int(r["N"]), r["implementation"], r["timing_scope"]): r for r in records}
    if len(lookup) != len(records):
        raise ValueError("Duplicate summary records")
    compared = []
    for series, sizes in (("formal", manifest["formal_sizes"]), ("supplementary", manifest["supplementary_sizes"])):
        for n in sizes:
            cpu = lookup[(series, n, "cpu", "compute")]
            compute = lookup[(series, n, "naive", "compute")]
            total = lookup[(series, n, "naive", "end_to_end")]
            c, g, e = [float(r["mean_ms"]) for r in (cpu, compute, total)]
            if not (c > 0 and e >= g > 0):
                raise ValueError("Invalid mean timings")
            compared.append({"series": series, "N": n, "cpu_ms": c,
                             "cpu_std_ms": float(cpu["std_ms"]),
                             "gpu_compute_ms": g, "gpu_compute_std_ms": float(compute["std_ms"]),
                             "gpu_end_to_end_ms": e, "gpu_end_to_end_std_ms": float(total["std_ms"]),
                             "compute_speedup": c / g, "end_to_end_speedup": c / e,
                             "gpu_other_ms": e - g, "gpu_other_percent": (e - g) / e * 100})
    target = ROOT / "results/summary" / f"{args.run_id}-comparison.csv"
    with target.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(compared[0]))
        writer.writeheader()
        writer.writerows(compared)
    out = ROOT / "results/figures" / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    formal = [r for r in compared if r["series"] == "formal"]
    ns = [r["N"] for r in formal]
    calls = manifest["config"]["measurement"]["measured_runs"]
    subtitle = f"One shared host | 1 warm-up + {calls} measured calls | means with sample SD"

    def finish(fig, name):
        fig.savefig(out / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
        fig.savefig(out / f"{name}.svg", bbox_inches="tight", facecolor="white")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.4, 5.8), layout="constrained")
    for key, err, label, color in (
        ("cpu_ms", "cpu_std_ms", "Single-thread CPU loop", "#2457A7"),
        ("gpu_compute_ms", "gpu_compute_std_ms", "Naive CUDA: kernel compute", "#D35B16"),
        ("gpu_end_to_end_ms", "gpu_end_to_end_std_ms", "Naive CUDA: end-to-end wrapper", "#19805C"),
    ):
        ax.errorbar(ns, [r[key] for r in formal], yerr=[r[err] for r in formal],
                    marker="o", capsize=4, lw=2, label=label, color=color)
    ax.set(xscale="log", yscale="log", xticks=ns, xlabel="Matrix side N (N × N)",
           ylabel="Time (ms, logarithmic scale)", title="A: CPU and naive CUDA matrix multiplication\n" + subtitle)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(alpha=.2, which="major")
    ax.legend(loc="upper left")
    finish(fig, "matrix-runtime")

    small = sorted([r for r in compared if r["series"] == "supplementary"] + formal[:1], key=lambda r: r["N"])
    if small:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), layout="constrained")
        xs = [r["N"] for r in small]
        for key, err, label, color in (("cpu_ms", "cpu_std_ms", "CPU loop", "#2457A7"),
                                      ("gpu_end_to_end_ms", "gpu_end_to_end_std_ms", "GPU end-to-end", "#19805C")):
            axes[0].errorbar(xs, [r[key] for r in small], yerr=[r[err] for r in small],
                            marker="o", capsize=3, color=color, label=label)
        axes[0].set(xscale="log", yscale="log", xticks=xs, xlabel="Matrix side N",
                    ylabel="Time (ms, log scale)", title="Supplementary small-size sweep")
        axes[0].xaxis.set_major_formatter(ScalarFormatter())
        axes[0].xaxis.set_minor_locator(NullLocator())
        axes[0].legend()
        axes[1].plot(xs, [r["end_to_end_speedup"] for r in small], "o-", color="#6B45A1")
        axes[1].axhline(1, color="#555555", ls="--", label="Equal time (1×)")
        axes[1].set(xscale="log", yscale="log", xticks=xs, xlabel="Matrix side N",
                    ylabel="CPU mean / GPU end-to-end mean (×)", title="Q2: measured crossover bracket")
        axes[1].xaxis.set_major_formatter(ScalarFormatter())
        axes[1].xaxis.set_minor_locator(NullLocator())
        axes[1].legend()
        for ax in axes:
            ax.grid(alpha=.2)
        fig.suptitle(f"{calls} measured calls per size; N={ns[0]} is reused from the formal batch", fontsize=11)
        finish(fig, "matrix-crossover")

    fig, ax = plt.subplots(figsize=(9.4, 5.3), layout="constrained")
    xs = list(range(len(formal)))
    kernels = [r["gpu_compute_ms"] for r in formal]
    ax.bar(xs, kernels, label="Kernel compute", color="#D35B16")
    ax.bar(xs, [r["gpu_other_ms"] for r in formal], bottom=kernels,
           label="Other wrapper time (allocation, copies, sync, cleanup)", color="#89BFA9")
    ax.set(xticks=xs, xticklabels=ns, xlabel="Matrix side N", ylabel="Mean time (ms)",
           title="Naive CUDA end-to-end timing breakdown\nResidual = mean end-to-end − mean kernel; not an isolated transfer measurement")
    ax.grid(axis="y", alpha=.2)
    ax.legend()
    finish(fig, "gpu-overhead")
    metadata = {"run_id": args.run_id, "manifest_sha256": sha(manifest_path),
                "input_summary": manifest["summary"], "input_summary_sha256": sha(summary),
                "comparison_csv": str(target.relative_to(ROOT)),
                "script_sha256": sha(Path(__file__)), "matplotlib": matplotlib.__version__,
                "speedup_definition": "ratio of arithmetic means, not mean of per-run ratios",
                "figures": {p.name: sha(p) for p in sorted(out.glob("*")) if p.suffix in {".png", ".svg"}}}
    (out / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(target.relative_to(ROOT))
    print(out.relative_to(ROOT))


if __name__ == "__main__":
    main()
