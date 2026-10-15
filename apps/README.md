# Shared Benchmark Program (maintained by A)

`harness.c` handles argument parsing, generating inputs from a seed or reading images, calling `lab6_*`, comparing against the double-precision reference, warm-up + timing, and CSV output.
`bench_cpu.c` (`make cpu`) and `bench_cuda.cu` (`make cuda`) register the current implementation entry points; once an implementation is connected to the existing interface, select it with `--impl`.

| Program | Implementation names |
|---|---|
| build/bench_cpu | `cpu` (matrix + convolution) |
| build/bench_cuda | `naive` `tiled` `optimized` `cublas` (matrix), `cuda` (convolution) |

While other implementations are unfinished, use `--impl` explicitly to select the ones that are done; the default `all` calls every entry point for the operation, including placeholders. `configs/benchmark.json` is the source of truth for shared parameters; the current defaults match it, but the program does not read the JSON directly.

```text
bench_* <selftest|matrix|conv> [options]
  --impl LIST     comma-separated implementation names, or all (default all)
  --n LIST        matrix sizes (default 256,512,1024,2048)
  --m LIST        image sizes (default 512,1024,2048)
  --k LIST        filter sizes (default 3,5,7)
  --filter NAME   mean|edge|sharpen (default mean)
  --image PATH    .u32 image under data/generated; random from the seed if omitted
  --save PATH     save the float32 output (only one implementation / one M / one K)
  --csv PATH      append to a CSV, writing the header if the file is empty; the parent directory must exist
  --member NAME   member column in the CSV (default: the LAB6_MEMBER environment variable)
  --warmup N --runs N --seed N --atol X --rtol X   defaults 1 / 3 / 542 / 1e-3 / 1e-4
```

- `selftest`: hand-computed cases (a nonsymmetric matrix, and a convolution filter that is nonzero only in its top-left corner, which distinguishes convolution from correlation), random cases at sizes 2/3/31, the `timings=NULL` path, and rejection of invalid arguments.
- `matrix` / `conv`: each configuration is warmed up and then measured `--runs` times; every run writes two CSV rows, compute and end_to_end, and the terminal prints the means.
- When a library function returns a nonzero status (including not implemented), that call writes no timing rows and the program finally exits with 1. During measurement, rows where the library call succeeded but numerical validation failed are kept with `correct=false`, and the program also exits nonzero; argument errors exit with 2.
- The double-precision reference takes a few seconds at N=2048 and is shared by all implementations in the same invocation; to compare several implementations, pass them together separated by commas.
