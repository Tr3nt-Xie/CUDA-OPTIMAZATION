#!/usr/bin/env python3
"""Summarize explicit CSV datasets; never infer that different inputs/hosts match.

A JSON plan supplies datasets [{csv, context, input_id}] and expected group keys.
Relative CSV paths are resolved from the plan file. Failed/incomplete groups are
reported and excluded. Partial coverage is clearly marked; no values are filled.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

KEYS = ("context", "input_id", "operation", "implementation", "N", "M", "K", "filter", "timing_scope")
ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/benchmark.json").read_text(encoding="utf-8"))


def key_of(row):
    return tuple(str(row.get(field, "")) for field in KEYS)


def aggregate(plan, base):
    groups = defaultdict(list)
    sources = {}
    for dataset in plan["datasets"]:
        if not dataset.get("context") or not dataset.get("input_id"):
            raise ValueError("every dataset needs explicit context and input_id")
        path = (base / dataset["csv"]).resolve()
        sources[dataset["csv"]] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(encoding="utf-8-sig", newline="") as fp:
            reader = csv.DictReader(fp)
            if reader.fieldnames != CONFIG["csv_columns"]:
                raise ValueError(f"unexpected CSV columns: {path}")
            for row in reader:
                row.update(context=dataset["context"], input_id=dataset["input_id"], source=dataset["csv"])
                groups[key_of(row)].append(row)
    summary, rejected = [], []
    expected_runs = list(range(1, CONFIG["measurement"]["measured_runs"] + 1))
    for key, rows in sorted(groups.items()):
        identity = dict(zip(KEYS, key))
        try:
            if identity["operation"] not in {"matrix", "convolution"}:
                raise ValueError("unknown operation")
            if identity["timing_scope"] not in CONFIG["measurement"]["scopes"]:
                raise ValueError("unknown timing scope")
            if any(r["correct"] != "true" for r in rows):
                raise ValueError("correctness failure; entire configuration excluded")
            if sorted(int(r["run"]) for r in rows) != expected_runs:
                raise ValueError("missing or duplicate runs; expected exactly 1,2,3")
            times = [float(r["time_ms"]) for r in rows]
            errors = [float(r["max_abs_error"]) for r in rows]
            if any(not math.isfinite(v) or v < 0 for v in times + errors):
                raise ValueError("invalid/nonfinite timing or error")
            summary.append(dict(identity, runs=len(rows), mean_ms=statistics.mean(times),
                                std_ms=statistics.stdev(times), max_abs_error=max(errors),
                                cpu_speedup="", source_csv=" | ".join(sorted({r["source"] for r in rows}))))
        except (ValueError, TypeError) as error:
            rejected.append(dict(identity, reason=str(error)))
    means = {key_of(r): r["mean_ms"] for r in summary}
    for row in summary:
        cpu = dict(row, implementation="cpu")
        baseline = means.get(key_of(cpu))
        if baseline is not None and row["mean_ms"] > 0:
            row["cpu_speedup"] = baseline / row["mean_ms"]
    expected = {key_of(row) for row in plan.get("expected", [])}
    if not expected:
        raise ValueError("plan.expected must declare required configurations")
    missing = [dict(zip(KEYS, key)) for key in sorted(expected - means.keys())]
    unexpected = [dict(zip(KEYS, key)) for key in sorted(means.keys() - expected)]
    return summary, {"status": "complete" if not missing and not rejected and not unexpected else "partial",
                     "missing": missing, "rejected": rejected, "unexpected": unexpected,
                     "accepted_groups": len(summary), "source_sha256": sources}


def plot_results(rows, out, status):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sets = defaultdict(list)
    for row in rows:
        sets[(row["context"], row["input_id"], row["operation"], row["M"],
              row["filter"], row["timing_scope"])].append(row)
    generated = []
    for index, (identity, records) in enumerate(sorted(sets.items()), 1):
        context, input_id, operation, image_size, kind, scope = identity
        xfield = "N" if operation == "matrix" else "K"
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
        implementations = sorted({r["implementation"] for r in records})
        for impl in implementations:
            selected = sorted((r for r in records if r["implementation"] == impl), key=lambda r: int(r[xfield]))
            xs = [int(r[xfield]) for r in selected]
            axes[0].errorbar(xs, [r["mean_ms"] for r in selected],
                             yerr=[r["std_ms"] for r in selected], marker="o", capsize=3, label=impl)
            paired = [r for r in selected if r["cpu_speedup"] != ""]
            if paired:
                axes[1].plot([int(r[xfield]) for r in paired], [r["cpu_speedup"] for r in paired], marker="o", label=impl)
        axes[0].set(ylabel="Runtime (ms), mean +/- sample SD", yscale="log")
        axes[1].set(ylabel="CPU / implementation (same scope)", yscale="log")
        axes[1].axhline(1, color="gray", linestyle="--", linewidth=1)
        for ax in axes:
            ax.set_xlabel("Matrix size N" if operation == "matrix" else "Filter size K")
            ax.set_xticks(sorted({int(r[xfield]) for r in records}))
            ax.grid(alpha=0.2)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
        dimensions = "" if operation == "matrix" else f" | M={image_size}, {kind}"
        title = f"{operation.title()} | {scope}{dimensions} | {status.upper()}"
        fig.suptitle(title + "\n" + context[:80], fontsize=12)
        stem = f"{index:02d}-{operation}-{scope}"
        for extension in ("png", "svg"):
            fig.savefig(out / f"{stem}.{extension}", dpi=160)
        plt.close(fig)
        generated.append({"file": stem, "context": context, "input_id": input_id,
                          "operation": operation, "M": image_size,
                          "timing_scope": scope, "filter": kind})
    (out / "figures.json").write_text(json.dumps(generated, indent=2) + "\n", encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("plan", type=Path)
    p.add_argument("--output", type=Path, required=True, help="New output directory")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--require-complete", action="store_true")
    args = p.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    rows, coverage = aggregate(plan, args.plan.resolve().parent)
    args.output.mkdir(parents=True, exist_ok=False)
    fields = [*KEYS, "runs", "mean_ms", "std_ms", "max_abs_error", "cpu_speedup", "source_csv"]
    with (args.output / "summary.csv").open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "coverage.json").write_text(json.dumps(coverage, indent=2) + "\n", encoding="utf-8")
    if not args.no_plots and rows:
        plot_results(rows, args.output, coverage["status"])
    print(f"{coverage['status'].upper()}: {len(rows)} groups; "
          f"{len(coverage['missing'])} missing; {len(coverage['rejected'])} rejected; "
          f"{len(coverage['unexpected'])} unexpected")
    if coverage["rejected"] or coverage["unexpected"] or (args.require_complete and coverage["missing"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
