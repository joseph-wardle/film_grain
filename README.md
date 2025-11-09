# Film Grain

Film Grain is a Rust renderer for physically based film grain built on Monte Carlo-filtered Boolean models. It implements both the pixel-wise and grain-wise algorithms from [Newson et al. (2017)](https://www.ipol.im/pub/art/2017/192/). The core library powers a command-line tool (`film_grain`) for batch work and an egui/wgpu viewer (`viewer`) for interactive tweaking.

## Highlights
- Pixel-wise and grain-wise Monte Carlo integrators with automatic selection and both CPU/GPU backends.
- Comprehensive parameterization: luma/RGB modes, ROI cropping, zoom or explicit sizing, deterministic seeding, dry-run/explain flags, and flexible output formats.
- CPU/GPU execution with deterministic seeding, ROI cropping, zoom or fixed sizing, RGB/luma modes, and dry-run/explain flags for inspection.
- `tools/film_grain_bench.py` sweeps parameter grids to benchmark pixel vs. grain renderers across CPU/GPU variants.

![Interactive Viewer Screenshot](.github/assets/viewer_screenshot.png)

## Try it
- CLI: `cargo run --release -- <input.png> --output out.png --radius 0.12 --iters 64 --device gpu --color-mode luma --explain`
- Viewer: `cargo run --release --bin viewer`.

## Hacking
- `cargo test` to validate the workspace.
- `tools/film_grain_bench.py` builds release binaries and sweeps parameter grids when you need performance data.
