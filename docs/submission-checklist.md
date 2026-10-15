# Lab6 Division of Work and Deliverables Checklist

Based on `lab6-f26.pdf` (8 pages). A's Parts 1–3 and 8.1 experiments and report materials are complete: same-host AWS CPU/naive formal and supplementary matrix runs, CPU convolution, and the GPU container run all passed. Q1/Q2 are answered in report/A.md. B's Parts 4, 6.2, and 8.2 materials are in the repository; C's code has been merged, and C's report materials and figures are in the repository; the final team merge is pending. Names for A/B/C are still to be filled in.

File paths in the table are the locations in the current skeleton. Each member completes their Parts according to the handout and decides their own internal implementation and file organization, keeping the interface and result conventions consistent when integrating.

| Part / handout pages | Owner | Work and source | Acceptance / materials |
|---|---|---|---|
| 1 · p1–2 | A | src/matrix/cpu.c, CPU triple loop | Correctness, the three required sizes; timing records and curves in the report |
| 2 · p2–3 | A | src/matrix/naive.cu, naive CUDA and host flow | Runnable program, CPU comparison, and raw timings |
| 3 · p3 | A | docs/ENVIRONMENT.md, shared GPU environment | Environment versions and evidence of actual successful runs |
| 4 · p3–4 | B | src/matrix/tiled.cu, shared-memory tiling | Comparison with CPU/naive, correctness, timing, boundary handling |
| 5 · p4 | C summarizes, A/B provide data | scripts, results/summary, figures | CPU/naive/tiled comparison, speedups, and transfer overhead |
| 6.1 · p4–5 | C | src/matrix/cublas.cu | cuBLAS comparison at the same sizes, correctness and timing |
| 6.2 · p5 | B | src/matrix/optimized.cu | Further optimization attempts, sources of the methods, before/after data and explanation |
| 7 · p5–7 | C, depends on B | src/bindings/exports.cu, python | Part 4 shared library, ctypes calls, and correct output |
| 8.1 · p7 | A | src/convolution/cpu.c, data/filter generation | Done on the same AWS host: 3 images, 3×3 size combinations, edge/sharpen demos, 99 timed calls; report/A.md; inputs/outputs placed in the read-only shared directory |
| 8.2 · p8 | B, depends on A's inputs | src/convolution/cuda.cu | CPU/CUDA numerical and performance comparison on the same inputs |
| 8.3 · p8 | C, depends on B | Convolution in the shared library and Python entry point | Three-way comparison: CPU C / standalone CUDA / Python CUDA |
| Deliverables · p8 | Everyone | Source on GitHub with reproducible instructions | CPU, CUDA, optimized, cuBLAS, shared library, and Python can all be located |

## The five Part 6 questions

| No. | Question | Main answer / data source |
|---|---|---|
| Q1 | How does performance change as matrix size increases? | A; CPU/naive curves done, merged into the team figures by C |
| Q2 | At what point does the GPU significantly outperform the CPU? | A; distinguish compute and end_to_end |
| Q3 | How much speedup does tiling give over naive CUDA? | B |
| Q4 | How close is the hand-optimized kernel to cuBLAS? | C; B provides the optimization description |
| Q5 | Why might cuBLAS still be faster? | C; explained with measurements and references |

The questions are as written on handout p.5; the table records the team's division of work, and where the answers appear follows the lab requirements.

## Each member's handoff package

- Their own source code, dependencies, and actual build/run commands.
- Correctness evidence, raw CSVs/logs, environment, and input identifiers.
- Report materials and image descriptions for the Parts they own.
- Current limitations, failures, or unfinished items; placeholder data must never be presented as results.

Report materials from each member are kept in `report/` and finally merged by A into a single `report/report.md`. File names and section organization of the member materials follow their actual content. No empty final report is created now, and no fictional contributions or results are filled in.

## Final wrap-up

- [x] A's shared AWS T4 environment: SSH accounts for all three members, host CPU/naive self-tests, GPU container build/run, and an N=256 small-scale check; evidence in `docs/ENVIRONMENT.md`.
- [x] A's formal matrix data: four formal sizes and four supplementary small sizes, all 48 timed calls passed; Part 1/2 curves and Q1/Q2 answers prepared.
- [x] A's 8.1 comparison data: three source images with sources (`data/README.md`), 33 configurations on the same AWS host, 99 timed calls, 9 independent checks, report images; fixed inputs/CPU outputs placed in the read-only shared directory and verified readable by B/C.
- [x] A's Part 3 container execution: in-container build, self-test, and nine timed calls at N=256/512/1024, all passing.
- [x] A's English report: report/A.md, including Q1/Q2.
- [ ] A: merge the report, making sure the Part 1 curves, Part 8.1 images, and related experiment materials are complete.
- [ ] B: from a clean checkout, check the CPU/GPU builds, all source files, the README, and the reproduction steps.
- [x] C's report materials and figures: report/C.md (Parts 5, 6.1, 7, 8.3; Q4/Q5) and results/figures/C-20261013-full-02/; C's raw run directories stay on the shared server.
- [ ] Everyone: check every Part, the five questions, actual contributions, and final links; confirm the submission entry and attachments on the course submission page.

Creating the local skeleton does not mean anything has been pushed to GitHub, submitted for the course, or that the lab is complete.
