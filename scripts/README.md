# Scripts

The table lists only the existing tools. C's performance comparison and analysis follow the handout; how C splits and names those scripts is up to C.

| Script | Owner | Purpose | Status |
|---|---|---|---|
| prepare_images.py | A | Images ↔ raw binaries used by the benchmark program | Done |
| generate_test_images.py | A | Generates three original, reproducible grayscale test images | Done |
| run_part81_cpu.py | A | Part 8.1 CPU image experiments, independent correctness checks, CSV summary, and result images | Validated on a Mac and in formal same-host AWS measurements |
| run_part_a_matrix.py | A | Same-host CPU/naive formal matrix baseline and supplementary small sizes for Q2 | Measured on the AWS T4 |
| plot_part_a.py | A | Generates curves, speedups, and provenance records from a passed matrix batch | Plots of the formal data done |
| collect_env.sh | A | Generates environment.txt | Done |

Dependencies: Python 3.9+, `python3 -m pip install -r scripts/requirements.txt` (NumPy, Pillow; Matplotlib for the matrix plots).

## A's same-host matrix experiments

Run on an idle CUDA lab machine; the script only collects A's CPU/naive results, does not start any cloud resources, and does not call B's or C's placeholder implementations.

```sh
export NVCCFLAGS='-O2 -std=c++14 -arch=sm_75'  # options used on the T4
python3 scripts/run_part_a_matrix.py --run-id A-YOUR-UNIQUE-RUN-ID --context 'Describe the actual host'
python3 scripts/plot_part_a.py A-YOUR-UNIQUE-RUN-ID
```

The formal sizes come from the configuration file (256/512/1024/2048). The script additionally measures 16/32/64/128 for A's Q2 analysis of where the GPU starts to pay off; these are supplementary experiments and do not change the team's required sizes. Use `--small-sizes ''` to turn them off. CPU and naive run sequentially, each with 1 warm-up and 3 measured calls, and all samples are kept.

Raw CSVs, full commands/exit codes, double-precision reference checks, self-tests, environment, source snapshot, and file/binary hashes are written to `results/raw/<run-id>/`. `manifest.json` is marked `passed` only if every size is correct and the source files did not change. `results/summary/<run-id>.csv` holds means, sample standard deviations, and ranges; the speedups in `-comparison.csv` are ratios of two means.

The plotting script outputs the runtime curves, the supplementary small-size curves, and the GPU overhead plot (PNG/SVG). `provenance.json` records the hashes of the summary inputs, the plotting script, and the images. The GPU `end_to_end - compute` is the remaining time for allocation, copies, synchronization, cleanup, and so on combined, and must not be called pure PCIe transfer time.

## Part 8.1 CPU image experiments

Run from the repository root:

```sh
python3 scripts/run_part81_cpu.py
```

The script uses and verifies the originals recorded in `data/images/manifest.json`, generating test images only if there is no manifest. It then converts the images, builds the CPU program separately, runs the selftest, and sequentially runs the mean filter for each image at M=512/1024/2048 × K=3/5/7. Edge detection and sharpening are also shown at M=512, K=3, for 33 configurations in total, each with 1 warm-up and 3 measured calls.

Every configuration is checked against the C double-precision reference; the nine demo outputs are additionally checked pixel by pixel with NumPy float64 using a zero-padded sliding window and the flipped filter. The raw float32 CPU outputs are saved separately from the PNG display mapping.

- `results/raw/<run-id>/`: per-configuration CSVs/logs, full commands and exit codes, environment, input hashes, source snapshot, and the overall manifest.
- `results/summary/<run-id>.csv`: per-image, per-configuration means, sample standard deviations, and min/max, recorded separately for both timing scopes.
- `results/figures/<run-id>/`: the originals, the three filter result images, and `filter-comparison.png`.
- `data/generated/<run-id>/`: the nine CPU float32 reference outputs used for the demos, ignored by Git.

`--run-id NAME` sets the experiment ID; an existing ID is rejected to avoid overwriting records. `--context 'actual machine and purpose of the measurement'` states whether the run is a local check or the CPU baseline on the shared AWS host. Speedups must not be computed between Mac measurements and CUDA times from another host.

## Converting images on their own

```sh
# All images in data/images → data/generated/<name>_<M>.u32 (grayscale, center-cropped to a square, resized to each configured size)
python3 scripts/prepare_images.py to-raw

# Run one demo filter and convert back to PNG; edge output has both signs, so display it with --mode abs
build/bench_cpu conv --image data/generated/geometry_512.u32 --k 3 --filter edge --save edge.f32
python3 scripts/prepare_images.py to-png edge.f32 edge.png --mode abs
```

See [the repository run example](../README.md#run) for the environment-record commands; create the result directory for the run first, then save the output of `collect_env.sh`.

`--mode` for `to-png`: `clip` (default, clipped to 0–255), `abs` (absolute value, then clipped), `normalize` (linearly stretched to 0–255). It only affects display; numerical validation always uses the raw float32 output.
