# Convolution Test Images (owner: A)

- `images/`: the original test images, committed to Git, with their sources documented below.
- `generated/`: raw `uint32` binaries (row-major) at each size, converted by `scripts/prepare_images.py`; ignored by Git and can be regenerated at any time.

Matrix inputs are not stored as files; the benchmark program generates them from a fixed seed. Convolution uses a random image when `--image` is not given; the Part 8.1 timing and demos explicitly use the images below. See `scripts/README.md` for the conversion commands.

## Image sources

All three images were generated from scratch by this repository's `scripts/generate_test_images.py`; they are not photographs and no external images are used. The originals are 2048×2048 8-bit grayscale PNGs with pixel values 0–255, converted to little-endian `uint32` before being passed to the C program.

| File | Content | What it checks |
|---|---|---|
| `images/geometry.png` | Boxes, rings, triangles, diagonal lines of different widths, and gradients | Contours, orientation, and the zero-padding boundary |
| `images/frequency.png` | Stripes of varying frequency, concentric textures, and a checkerboard | How the mean filter smooths fine textures, and the contrast change after sharpening |
| `images/soft_scene.png` | A procedurally drawn and softened house, hills, trees, and fence | Edge response on soft contours and the sharpening effect |

The SHA-256 of each original is in `images/manifest.json`. The fixed generation parameters are size 2048 and seed 542; this run used NumPy 2.3.5 and Pillow 12.3.0. `generate_test_images.py` never overwrites an existing original whose content differs.

Teammates should use these three PNGs from the repository. `run_part81_cpu.py` verifies the original hashes and then converts them to `.u32` at M=512/1024/2048 via `prepare_images.py`. Every experiment record also stores the SHA-256 of the corresponding `.u32`, which must match before comparing across machines; the same file name or size does not prove the inputs are identical.

The nine CPU float32 outputs at 512×512, K=3 are saved in `data/generated/<run-id>/` and can be regenerated with the same script. They keep negative values and values above 255 for numerical comparison; the PNG result images are for display only.
