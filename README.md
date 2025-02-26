# Realistic Film Grain Rendering

A Rust-based project that implements a film grain synthesis algorithm inspired by Newson et al. (2017). The project offers two distinct rendering approaches—a grain-wise method and a pixel-wise method—with an automatic mode that selects the most efficient algorithm based on your parameters.

## Features

- **Physically–Motivated Model:** Simulates film grain using a stochastic Boolean model.
- **Dual Rendering Approaches:** Choose between grain-wise and pixel-wise implementations.

## Usage

Compile the project with Cargo:

```bash
cargo build --release
```

Run the renderer using the CLI. For example, to render an image with automatic algorithm selection:

```bash
cargo run --release -- input.jpg output.jpg --radius 0.1 --sigmaR 0.0 --sigmaFilter 0.8 --zoom 1.0 --algorithm automatic
```

## License

This project is provided under the Unlicense. See the LICENSE file for details.

---

Explore the code, experiment with different parameters, and enjoy adding a vintage film look to your digital images!
